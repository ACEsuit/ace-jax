"""Benchmark models: what is planned, how each is built, and the manifest.

    python bench/scaling/models.py pace   # pyace venv: random-coefficient .yace x3 x2
    python bench/scaling/models.py ace    # Julia: linear ACE .npz x3 x2
    python bench/scaling/models.py mace   # MACE-MP-0b2 s/m/l + MH-1 + Symmetrix .json per system
Model files live in bench/scaling/models/ (git-ignored); manifest.json records
provenance (builder, parameters, n_params, sha256).
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIR = ROOT / "bench" / "scaling" / "models"
SIZES = ("small", "medium", "large")
ELEMENTS = {"SiGe": ["Si", "Ge"], "Cantor": ["Cr", "Mn", "Fe", "Co", "Ni"]}
PACE_FUNCS = {"small": 100, "medium": 500, "large": 2000}          # per element
# (ACE_ORDER, ACE_TOTALDEGREE) targeting ~100 / 700 / 2800 basis functions per
# element; `models.py ace` records the actual count and warns if off by > 2x
ACE_DEG = {"small": (3, 8), "medium": (3, 12), "large": (4, 12)}
# five species multiply the basis: Cantor gets its own (order, degree) ladder so
# its per-element size tracks the same targets (measured: (3,8) -> 765/element)
ACE_DEG_BY_SYSTEM = {"SiGe": ACE_DEG,
                     "Cantor": {"small": (3, 6), "medium": (3, 8), "large": (3, 11)}}
# explicit bond length where ACEpotentials has no default (fcc nearest neighbour
# a / sqrt(2) at a = 3.59 A); SiGe uses the library defaults
ACE_R0 = {"Cantor": 2.54}
# MACE-MP-0b2, not the original MP-0: Symmetrix only exports a plain/density
# first interaction block, and MP-0 (like MH-1) has a residual one
MACE = {"small": "small-0b2", "medium": "medium-0b2", "large": "large-0b2",
        "mh1": "mh-1"}                                              # MACE-MH-1 (multi-head)
MACE_SIZES = tuple(MACE)
# multi-head models: the head both the standalone calculator and the Symmetrix
# export evaluate (materials PBE -- the one that fits SiGe and Cantor)
MACE_HEAD = {"mh1": "omat_pbe"}


def planned_models():
    rows = []
    for system, els in ELEMENTS.items():
        for size in SIZES:
            yace = DIR / f"pace_{system}_{size}.yace"
            rows += [dict(name=f"acejax-pace/{system}/{size}", code="acejax-pace", system=system,
                          size=size, path=str(yace), elements=els),
                     dict(name=f"mlpace/{system}/{size}", code="mlpace", system=system,
                          size=size, path=str(yace), elements=els),
                     dict(name=f"acejax-ace/{system}/{size}", code="acejax-ace", system=system,
                          size=size, path=str(DIR / f"ace_{system}_{size}.npz"), elements=els),
                     ]
        for size in MACE_SIZES:
            rows.append(dict(name=f"mace/{system}/{size}", code="mace", system=system, size=size,
                             path=str(DIR / f"mace_{size}.model"), elements=els,
                             head=MACE_HEAD.get(size),
                             symmetrix=str(DIR / f"mace_{size}_{system}.json")))
    return rows


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def _update_manifest(entries):
    DIR.mkdir(parents=True, exist_ok=True)
    mf = DIR / "manifest.json"
    cur = json.loads(mf.read_text()) if mf.exists() else {}
    cur.update(entries)
    mf.write_text(json.dumps(cur, indent=1, sort_keys=True))


def load_manifest():
    return json.loads((DIR / "manifest.json").read_text())


def build_pace():
    """Run inside pace_ref/.venv (pyace)."""
    import numpy as np
    from pyace import ACEBBasisSet, create_multispecies_basis_config
    out = {}
    for system, els in ELEMENTS.items():
        for size, nf in PACE_FUNCS.items():
            cfg = {"deltaSplineBins": 0.001, "elements": els,
                   "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled",
                                          "fs_parameters": [1, 1, 1, 0.5], "ndensity": 2}},
                   "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.25], "rcut": 5.0,
                                     "dcut": 0.01, "inner_cutoff_type": "distance",
                                     "r_in": 1.0, "delta_in": 0.5,
                                     "core-repulsion": [100.0, 5.0]}},
                   "functions": {"number_of_functions_per_element": nf,
                                 "ALL": {"nradmax_by_orders": [15, 6, 4, 3, 2],
                                         "lmax_by_orders": [0, 4, 3, 2, 1]}}}
            bb = ACEBBasisSet(create_multispecies_basis_config(cfg))
            rng = np.random.default_rng(nf + len(els))
            c = np.asarray(bb.all_coeffs)
            bb.all_coeffs = rng.normal(scale=0.05, size=c.shape)
            p = DIR / f"pace_{system}_{size}.yace"
            DIR.mkdir(parents=True, exist_ok=True)
            bb.to_ACECTildeBasisSet().save_yaml(str(p))
            out[str(p)] = {"builder": "pyace", "functions_per_element": nf,
                           "n_params": int(c.size), "sha256": _sha(p)}
    _update_manifest(out)


def build_ace():
    for system, els in ELEMENTS.items():
        for size, (order, deg) in ACE_DEG_BY_SYSTEM[system].items():
            p = DIR / f"ace_{system}_{size}.npz"
            if p.exists() and not os.environ.get("FORCE"):
                continue
            # ACE_NOFIT=1: random weights, timing only -- the script's built-in fit is
            # on a silicon-only set, meaningless for SiGe/Cantor and singular (Cholesky
            # fails) once the basis outgrows it
            env = {**os.environ, "ACE_ELEMENTS": ",".join(els), "ACE_ORDER": str(order),
                   "ACE_TOTALDEGREE": str(deg), "ACE_RCUT": "5.0", "ACE_NOFIT": "1",
                   **({"ACE_R0": str(ACE_R0[system])} if system in ACE_R0 else {})}
            # ACE_JULIA selects the Julia (e.g. "julia +1.11": the export env does
            # not resolve on 1.13, and CI's julia-parity job pins 1.11)
            julia = os.environ.get("ACE_JULIA", "julia").split()
            subprocess.run([*julia, "--project=julia", "julia/export_model.jl", str(p), "ace1"],
                           cwd=ROOT, env=env, check=True)
            import numpy as np
            n_b = int(json.loads(bytes(np.load(p)["meta_json"]).decode())["n_B"])
            _update_manifest({str(p): {"builder": "julia/export_model.jl", "order": order,
                                        "totaldegree": deg, "n_params": n_b, "weights": "random",
                                        "sha256": _sha(p)}})   # per model: a later failure keeps it


def symmetrix_cmd(model, zs, head, out):
    cmd = [shutil.which("symmetrix_extract_mace") or "symmetrix_extract_mace", "--model", str(model),
           "--atomic-numbers", *map(str, zs), "--output", str(out)]
    return cmd + (["--head", head] if head else [])


def build_mace():
    """Run inside the MACE venv (mace-torch + symmetrix): download MACE-MP-0b2 / MH-1 and
    extract one Symmetrix .json per (size, system) -- each is element-specific."""
    from ase.data import atomic_numbers
    from mace.calculators.foundations_models import download_mace_mp_checkpoint
    out = {}
    DIR.mkdir(parents=True, exist_ok=True)
    for size, tag in MACE.items():
        src = download_mace_mp_checkpoint(tag)
        p = DIR / f"mace_{size}.model"
        p.write_bytes(pathlib.Path(src).read_bytes())
        out[str(p)] = {"builder": f"mace-mp:{tag}", "tag": tag,
                       "sha256": _sha(p)}
        for system, els in ELEMENTS.items():
            zs = sorted(atomic_numbers[e] for e in els)
            dst = DIR / f"mace_{size}_{system}.json"
            dst.unlink(missing_ok=True)
            r = subprocess.run(symmetrix_cmd(p, zs, MACE_HEAD.get(size), dst), cwd=DIR,
                               capture_output=True, text=True)
            if r.returncode == 2:                        # argparse usage error: our bug
                raise RuntimeError(r.stderr[-500:])
            if r.returncode != 0 or not dst.exists():
                # a model Symmetrix cannot export: its LAMMPS rows become
                # "unsupported"; standalone PyTorch rows still run
                out[str(dst)] = {"builder": "symmetrix_extract_mace", "from": str(p),
                                 "unsupported": (r.stderr or r.stdout)[-300:]}
                continue
            out[str(dst)] = {"builder": "symmetrix_extract_mace", "from": str(p),
                             "atomic_numbers": zs, "head": MACE_HEAD.get(size), "sha256": _sha(dst)}
    _update_manifest(out)


if __name__ == "__main__":
    {"pace": build_pace, "ace": build_ace, "mace": build_mace}[sys.argv[1]]()
