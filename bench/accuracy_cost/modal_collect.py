"""The local half of modal_fits.py (no modal import, so the test suite can load it):
the data upload list, the Volume -> /storage download, and the status.json path
rewrite from the container's run root to the local one.  `vol` is anything with
modal.Volume's listdir(path, recursive=True) and read_file_into_fileobj(path, f)."""
import json
import os
import pathlib
import subprocess

from . import common

VOLUME = "ace-jax-accuracy-cost"
MOUNT = "/vol"                                          # the Volume in the containers; ACEJAX_AC_ROOT there
REMOTE_DATA = "data/si_accuracy_cost"                   # = common.DATA relative to ACEJAX_AC_ROOT
DATA_FILES = ("train.xyz", "test.xyz", "split.json", "weights_mace.json",
              "gp_iter6_sparse9k.xml")                  # the .xml: weights.command_line reads GAP-18's sigmas
MACE_SIZES = ("c8L0", "c16L0", "c32L0", "c16L1", "c32L1", "c64L1")   # sweep.yaml's mace (modal's python has no yaml)
MACE_PY = "/storage/eng/essswb/venvs/ac-mace/bin/python"           # CPU symmetrix: the .json export runs here


def local_runs(runs):
    """The local tree for a Volume run root ("runs/accuracy_cost" -> common.RUNS's default)."""
    return common.ROOT / runs


def remote_runs(runs):
    return f"{MOUNT}/{runs}"


def read_json(p, log=print):
    """A status.json, or None with a warning when it is missing or half-written."""
    try:
        return json.loads(pathlib.Path(p).read_text())
    except (OSError, ValueError) as e:
        log(f"warning: skipping {p}: {type(e).__name__}: {e}")
        return None


def complete(family, st):
    """A fit collect may use: status ok, and for MACE also finish() done (n_params recorded:
    fit_once writes ok before the stage-two swap that replaces model.model)."""
    return bool(st) and st.get("status") == "ok" and (family != "mace" or bool(st.get("n_params")))


def relocate(status_path, old, new, log=print):
    """Rewrite the container paths (model, ladder) in one status.json; True if it changed."""
    p = pathlib.Path(status_path)
    st = read_json(p, log)
    if st is None:
        return False
    out = {k: (str(new) + v[len(str(old)):] if isinstance(v, str) and v.startswith(str(old) + "/") else v)
           for k, v in st.items()}
    if out == st:
        return False
    tmp = p.with_name(p.name + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
    return True


def collect(vol, runs, families=("pace", "mace"), dest=None, log=print):
    """Download <runs>/<family>/** from the Volume into dest (default local_runs(runs)), skipping
    files already there with the same size and mtime, then relocate every status.json.  Re-runnable."""
    dest = pathlib.Path(dest or local_runs(runs))
    n = 0
    for fam in families:
        try:
            entries = vol.listdir(f"{runs}/{fam}", recursive=True)
        except Exception as e:                          # nothing fitted yet for this family
            log(f"{runs}/{fam}: {e}"); continue
        for f in entries:
            if getattr(f.type, "name", str(f.type)) != "FILE":
                continue
            rel = pathlib.PurePosixPath(f.path.lstrip("/")).relative_to(runs)
            out = dest / rel
            if out.exists() and out.stat().st_size == f.size and int(out.stat().st_mtime) == f.mtime \
                    and out.name != "status.json":      # status.json: relocated locally, so always re-read
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            try:
                with open(tmp, "wb") as fh:
                    vol.read_file_into_fileobj(f.path.lstrip("/"), fh)
            except FileNotFoundError:                   # removed since listdir (MACE rotates ckpt/ files)
                tmp.unlink(missing_ok=True); log(f"warning: {f.path} vanished during collect"); continue
            os.utime(tmp, (f.mtime, f.mtime)); os.replace(tmp, out); n += 1
        for st in (dest / fam).glob("*/status.json"):
            relocate(st, remote_runs(runs), dest, log)
    log(f"{n} files -> {dest}")
    return dest


def statuses(dest, families=("pace", "mace"), log=print):
    """{family/size: status} for every readable status.json (a half-written one is skipped)."""
    out = {}
    for fam in families:
        for p in sorted((pathlib.Path(dest) / fam).glob("*/status.json")):
            st = read_json(p, log)
            if st is not None:
                out[f"{fam}/{p.parent.name}"] = st
    return out


def symmetrix_due(dest, sizes, log=print):
    """MACE sizes whose model.json is missing or older than model.model, among complete fits only."""
    out = []
    for s in sizes:
        d = pathlib.Path(dest) / "mace" / s
        if not (d / "status.json").exists() or not complete("mace", read_json(d / "status.json", log)):
            continue
        m, j = d / "model.model", d / "model.json"
        if m.exists() and (not j.exists() or m.stat().st_mtime > j.stat().st_mtime):
            out.append(s)
    return out


def symmetrix_local(dest, sizes, python=MACE_PY, log=print):
    """The Symmetrix .json for each finished MACE fit, on the local CPU (symmetrix is not in the image)."""
    ok = symmetrix_due(dest, sizes, log)
    if not ok:
        return []
    env = {**os.environ, "ACEJAX_AC_RUNS": str(dest), "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    subprocess.run([python, "-m", "accuracy_cost.fit_mace", *ok, "--symmetrix-only"],
                   cwd=common.HERE.parent, env=env, check=True)
    return ok
