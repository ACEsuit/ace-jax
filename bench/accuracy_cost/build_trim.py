"""ACEpotentials.jl's LAMMPS export of the linear ACE fits: one `juliac --trim=safe`
library per `ace` model (bench/scaling/julia/build_trim_acejax.jl), for timing.py's
`acepotentials-trim` rows (pair_style ace, CPU only).

    python -m accuracy_cost.build_trim --env ENV.json [SIZE ...]      # default: every ok ace size

The Julia, its depot and the pinned PR 309 project come from the env json (`julia`,
`julia_depot`, `julia_project`), as for bench/scaling's lines.  juliac targets the build
host's CPU (JULIA_CPU_TARGET overrides), so build on the node type that times it.  Each
library goes to <run dir>/trim/libace.so with libace.build.json (the driver's gates and
identity checks, its inputs' hashes, the build CPU); a library is rebuilt only when the
model, the Julia env, the build CPU or the CPU target changed, or with --force.  The
driver's first gate holds the rebuilt ACEpotentials model to ace-jax's E / F on a rattled
64-atom cell, written here."""
import hashlib
import json
import os
import pathlib
import sys

from . import common

sys.path.insert(0, str(common.HERE.parent))
FAMILY = "ace"


def trim_lib(run_dir):
    return pathlib.Path(run_dir) / "trim" / "libace.so"


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _cpu_model():
    from scaling.models import _cpu_model
    return _cpu_model()


def write_gate(npz, out):
    """{"xyz", "E", "F"}: ace-jax's analytic model on the scaling gate cell (64 atoms,
    rattled, round-tripped through extxyz so Julia reads the same positions)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.calc.point import ACECalculator
    from scaling import acepot
    from scaling.structures import supercell
    out = pathlib.Path(out)
    at = acepot.roundtrip(supercell("Si", 64, seed=0, rattle=0.1), out.with_suffix(".extxyz"))
    at.calc = ACECalculator(str(npz), skin=0)
    out.write_text(json.dumps({"xyz": str(out.with_suffix(".extxyz")), "E": float(at.get_potential_energy()),
                               "F": at.get_forces().tolist()}))
    return out


def build(size, cfg, force=False):
    """Build (or keep) one size's library; returns its build record."""
    from scaling import acepot
    d = common.run_dir(FAMILY, size)
    st = common.read_status(d)
    if not st or st.get("status") != "ok":
        return {"size": size, "status": "skipped", "why": "no ok fit"}
    npz, so = d / "model.npz", trim_lib(d)
    rec_path = so.with_suffix(".build.json")
    manifest = pathlib.Path(cfg["project"]) / "Manifest.toml"
    base = {"builder": "bench/scaling/julia/build_trim_acejax.jl", "from": str(npz), "from_sha256": _sha(npz),
            "env_sha256": _sha(manifest) if manifest.exists() else None, "build_cpu": _cpu_model(),
            "cpu_target": os.environ.get("JULIA_CPU_TARGET", "native")}
    old = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    if (not force and so.exists() and old.get("sha256") == _sha(so)
            and all(old.get(k) == v for k, v in base.items() if k != "from")):
        return {"size": size, **old, "status": "fresh"}
    so.parent.mkdir(parents=True, exist_ok=True)
    gate = write_gate(npz, so.parent / "gate.json")
    try:
        out = acepot.run_julia(cfg, "build_trim_acejax.jl", [str(npz), str(gate), str(so)],
                               threads=1, timeout=4 * 3600)
    except RuntimeError as ex:
        so.unlink(missing_ok=True)
        rec = {**base, "status": "unsupported", "unsupported": str(ex)[-1500:]}
        rec_path.write_text(json.dumps(rec, indent=1))
        return {"size": size, **rec}
    rec = {**base, "status": "ok", "sha256": _sha(so),
           **{k: out[k] for k in ("build_id", "gates", "identity", "versions", "export_s", "juliac_s",
                                  "so_bytes") if k in out}}
    rec_path.write_text(json.dumps(rec, indent=1))
    return {"size": size, **rec}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sizes", nargs="*", help="ace sizes (default: every ok ace fit)")
    ap.add_argument("--env", required=True, help="host env json with julia / julia_depot / julia_project")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    from scaling import acepot
    cfg = acepot.julia_config(json.loads(pathlib.Path(a.env).read_text()))
    for size in a.sizes or [e["size"] for e in common.load_sweep()[FAMILY]]:
        r = build(size, cfg, force=a.force)
        g = r.get("gates", {}).get("ace_model_vs_acejax", {})
        print(size, r["status"], *([f"juliac {r['juliac_s']:.0f} s", f"max dF {g.get('max_dF'):.1e}"]
                                   if r["status"] in ("ok", "fresh") else [r.get("why") or r.get("unsupported", "")[-200:]]),
              flush=True)


if __name__ == "__main__":
    main()
