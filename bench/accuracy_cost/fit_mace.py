"""MACE trained from scratch: mace_run_train's defaults, with the shared cutoff, E0,
virials (--loss=virials, MACE's own global virials weight) and GAP-18 per-type weights
(as config_type_weights: MACE weights a config's whole loss, so the force sigma sets the
ratio).  2 layers (sweep `interactions`), max_ell 3, correlation 3; channels and max_L are swept.
Run under the MACE venv.  $ACEJAX_AC_DEVICE overrides the training device (default cuda).
`--restart_latest` resumes from ckpt/ after a preempted (Modal) container, and is a no-op on a
fresh run.  The Symmetrix .json is a separate step (`--symmetrix-only`), so a fit can run
where symmetrix is not built (Modal) and be exported locally."""
import json
import os
import pathlib
import subprocess
import sys

from . import common
from .common import DATA, E0_SI, RCUT


def command(e, d):
    w = json.loads((DATA / "weights_mace.json").read_text()) if (DATA / "weights_mace.json").exists() else {"Default": 1.0}
    return ["mace_run_train", "--name=model", f"--train_file={e.get('train_file') or DATA / 'train.xyz'}", "--valid_fraction=0.05",
            f"--test_file={DATA / 'test.xyz'}", "--energy_key=energy", "--forces_key=forces",
            "--virials_key=virial", "--loss=virials",
            f"--E0s={{14: {E0_SI}}}", f"--config_type_weights={json.dumps(w)}", "--model=MACE",
            f"--num_interactions={e.get('interactions', 2)}", f"--num_channels={e['channels']}", f"--max_L={e['max_L']}",
            "--max_ell=3", "--correlation=3", f"--r_max={RCUT}", "--batch_size=5",
            f"--max_num_epochs={e.get('epochs', 300)}", "--swa", "--ema", "--default_dtype=float64",
            f"--device={os.environ.get('ACEJAX_AC_DEVICE', 'cuda')}",
            "--seed=0", f"--results_dir={d}", f"--model_dir={d}", f"--checkpoints_dir={d / 'ckpt'}", "--restart_latest"]


def subset(d, n):
    """DATA/train.xyz strided to n frames (the file is sorted by config_type), for a smoke fit."""
    from .split import _write, read_atoms
    ats = read_atoms(DATA / "train.xyz")
    p = pathlib.Path(d) / "train_subset.xyz"; p.parent.mkdir(parents=True, exist_ok=True)
    _write(p, ats[::max(1, len(ats) // n)][:n])
    return str(p)


def _path():
    os.environ["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]   # mace_run_train, symmetrix_extract_mace


def fit(size, overrides=None, symmetrix=True):
    """overrides: sweep-entry keys (epochs, ...) plus n_train, a strided train subset (smoke)."""
    from .split import ensure_split
    from .weights import write_all
    _path()
    h = ensure_split(); write_all(DATA)
    e, d = {**common.entry("mace", size), **(overrides or {})}, common.run_dir("mace", size)
    if e.get("n_train"):
        e["train_file"] = subset(d, e["n_train"])
    st = common.fit_once("mace", size, "model.model", command(e, d), split_hash=h)
    if st["status"] == "ok" and not st.get("n_params"):
        st = finish(d, st)
    if st["status"] == "ok" and symmetrix:
        export_symmetrix(size)
    return st


def load_cpu(path):
    """torch.load onto the CPU.  e3nn's CodeGenMixin re-loads each TorchScript submodule with a plain
    torch.jit.load, which ignores torch.load's map_location, so a model saved from a CUDA fit keeps those
    constants on cuda (or fails without a GPU): force map_location="cpu" for the duration."""
    import torch
    orig = torch.jit.load
    torch.jit.load = lambda f, map_location=None, *a, **k: orig(f, "cpu", *a, **k)
    try:
        m = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        torch.jit.load = orig
    return m.to("cpu")


def non_cpu_tensors(obj, path="model", _seen=None):
    """[(attribute path, device)] of every tensor under a module/object that is not on the CPU: parameters,
    buffers and plain tensor attributes (which .to() does not move), through submodules, dicts, lists."""
    import torch
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return []
    seen.add(id(obj)); out = []
    if isinstance(obj, torch.Tensor):
        return [(path, str(obj.device))] if obj.device.type != "cpu" else []
    if isinstance(obj, torch.nn.Module):
        items = [(f"{path}.{k}", v) for k, v in vars(obj).items() if k not in ("_parameters", "_buffers", "_modules")]
        for d in ("_parameters", "_buffers", "_modules"):
            items += [(f"{path}.{k}", v) for k, v in vars(obj).get(d, {}).items() if v is not None]
    elif isinstance(obj, dict):
        items = [(f"{path}[{k!r}]", v) for k, v in obj.items()]
    elif isinstance(obj, (list, tuple)):
        items = [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
    else:
        return []
    for p, v in items:
        out += non_cpu_tensors(v, p, seen)
    return out


def cpu_copy(d):
    """d/model-cpu.model: a CPU copy of d/model.model (left unchanged), rebuilt when model.model is newer."""
    import torch
    src, dst = pathlib.Path(d) / "model.model", pathlib.Path(d) / "model-cpu.model"
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        m = load_cpu(src)
        bad = non_cpu_tensors(m)
        if bad:
            raise RuntimeError(f"{src}: tensors still off the CPU: {bad[:5]}")
        torch.save(m, dst)
    return dst


def export_symmetrix(size):
    """model.model -> model.json (Symmetrix) via a CPU copy (model-cpu.model: Modal-trained models keep CUDA
    tensors), again whenever model.model is newer (the stage-two swap replaces it); needs
    symmetrix_extract_mace (the local MACE venv)."""
    _path()
    d = common.run_dir("mace", size)
    if symmetrix_stale(d):
        sys.path.insert(0, str(common.HERE.parent))
        from scaling.models import symmetrix_cmd
        subprocess.run(symmetrix_cmd(str(cpu_copy(d)), [14], None, str(d / "model.json")), check=True)


def symmetrix_stale(d):
    m, j = pathlib.Path(d) / "model.model", pathlib.Path(d) / "model.json"
    return m.exists() and (not j.exists() or m.stat().st_mtime > j.stat().st_mtime)


def finish(d, st):
    """Deploy stage two as model.model, write the float32 copy and record n_params."""
    import shutil
    two = d / "model_stagetwo.model"          # --swa: mace_run_train keeps stage one as model.model and the stage-two
    if two.exists() and not (d / "model_stage1.model").exists():   # (energy-weighted) final in its own file; deploy that
        shutil.move(d / "model.model", d / "model_stage1.model"); shutil.copy(two, d / "model.model")
    import torch
    m = load_cpu(d / "model.model")
    n = int(sum(p.numel() for p in m.parameters()))
    m.float(); torch.save(m, d / "model-f32.model")                       # the float32 timing variant
    common.write_status(d, **{**{k: v for k, v in st.items() if k != "host"}, "n_params": n,
                              **({"host": st["host"]} if st.get("host") else {})})   # the fit's host, not this step's
    return common.read_status(d)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("size", nargs="+", help="sizes, or `all`")
    ap.add_argument("--overrides", default="", help="JSON sweep-entry overrides (smoke)")
    ap.add_argument("--no-symmetrix", action="store_true", help="skip the .json export (Modal)")
    ap.add_argument("--symmetrix-only", action="store_true", help="only export finished fits' .json")
    a = ap.parse_args()
    sizes = [e["size"] for e in common.load_sweep()["mace"]] if a.size == ["all"] else a.size
    for s in sizes:
        if a.symmetrix_only:
            export_symmetrix(s); print(s, "symmetrix ok", flush=True)
        else:
            print(s, fit(s, json.loads(a.overrides) if a.overrides else None, not a.no_symmetrix)["status"], flush=True)
