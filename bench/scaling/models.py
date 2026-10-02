"""Benchmark models: what is planned, how each is built, and the manifest.

    python bench/scaling/models.py pace   # pyace venv: random-coefficient .yace x3 x2
    python bench/scaling/models.py ace    # Julia: linear ACE .npz x3 x2
    python bench/scaling/models.py mace   # MACE-MP-0b2 s/m/l + MH-1 + Symmetrix .json per system
    python bench/scaling/models.py ace-learned  # learned-radial proxies of ace_*_medium.npz (after `ace`)
    python bench/scaling/models.py sizes  # basis functions per central element -> model_sizes.json
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
SIZES_JSON = ROOT / "bench" / "scaling" / "model_sizes.json"      # committed; models are not
SIZES = ("small", "medium", "large")
ELEMENTS = {"SiGe": ["Si", "Ge"], "Cantor": ["Cr", "Mn", "Fe", "Co", "Ni"]}
PACE_FUNCS = {"small": 100, "medium": 500, "large": 2000}          # per element
# The comparable count for a linear ACE model is n_B, "basis functions per
# central element": every B function carries its own weight per central
# element (WB is n_B x NZ), not n_B / NZ (see ace_functions_per_element).
# (ACE_ORDER, ACE_TOTALDEGREE) below are chosen so n_B lands within ~20% of
# the actual PACE n_B for that system/size (bench/scaling/model_sizes.json:
# SiGe 100/499/1684, Cantor 96/496/1998) -- not the round PACE_FUNCS targets,
# which pyace only approximates. `models.py ace` records the actual n_B and
# warns if it differs from a matching pace_*.yace by more than 25%.
ACE_DEG = {"small": (4, 5), "medium": (3, 9), "large": (4, 10)}   # small: (3,6) gives 123, (4,5) 101
# five species multiply the basis combinatorially: order 3 jumps too far
# between consecutive totaldegree values to land near Cantor's (lower) PACE
# targets (e.g. totaldegree 4 -> 370, 5 -> 734 against a target of 496), so
# Cantor uses order 2 at small/medium -- its own (order, degree) ladder, so
# its n_B still tracks the same per-system PACE targets as SiGe.
# totaldegree <= 3 is not just coarse but unusable here: ACEpotentials derives
# lmax from totaldegree, and at this rcut/r0 that hits lmax=0, which crashes
# solid-harmonics evaluation in SpheriCart (BoundsError, independent of
# order) -- so 4 is a hard floor, not a tuning choice.
ACE_DEG_BY_SYSTEM = {"SiGe": ACE_DEG,
                     "Cantor": {"small": (2, 4), "medium": (2, 7), "large": (4, 5)}}
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
# Learned-radial proxy lines (docs/dev/learned-radial-splining.md): the medium
# linear ACE model on the analytic radial branch with perturbed weights
# (`build_ace_learned`), evaluated as deployed -- spline_tol="auto", which
# splines a learned radial at 1e-10 -- and kept analytic (None, exact).  Medium
# only: that is the size the other figures compare at.
LEARNED_CODES = {"acejax-ace-learned": "auto", "acejax-ace-analytic": None}   # code -> spline_tol
LEARNED_SIZES = ("medium",)
LEARNED_RECIPE = {"n_q": 12, "scale": 0.1, "seed": 0}


def learned_path(system, size="medium"):
    return DIR / f"ace_{system}_{size}_learned.npz"


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
            if size in LEARNED_SIZES:
                rows += [dict(name=f"{code}/{system}/{size}", code=code, system=system, size=size,
                              path=str(learned_path(system, size)), elements=els, spline_tol=tol)
                         for code, tol in LEARNED_CODES.items()]
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
            yace = DIR / f"pace_{system}_{size}.yace"
            if yace.exists():
                n_pace = pace_functions_per_element(yace)
                if abs(n_b - n_pace) > 0.25 * n_pace:
                    print(f"WARNING: {p.name}: n_B={n_b} vs pace {yace.name} n_B={n_pace} "
                          f"(off by {100 * (n_b - n_pace) / n_pace:+.0f}%)", file=sys.stderr)
            _update_manifest({str(p): {"builder": "julia/export_model.jl", "order": order,
                                        "totaldegree": deg, "n_params": n_b, "weights": "random",
                                        "sha256": _sha(p)}})   # per model: a later failure keeps it


def learned_proxy(src, dst, n_q=12, scale=0.1, seed=0):
    """Write a learned-radial proxy of the linear ACE model at `src` to `dst`.

    Recipe: `to_analytic(model, n_q)` (projects the spline R_nl onto n_q
    normalized Legendre polynomials; the zero rows of ACE1's per-z_j pattern
    project to exact zeros), then every active row (`row_active`) of rnl_Wnlq
    gets W += scale * rms(row) * N(0, 1), with numpy's default_rng(seed), and
    the zero rows stay exactly zero -- as learning keeps them
    (`radial_model.normalise`), so `lean`'s species compaction still applies.
    `with_radial(..., learned=True)` marks it learned and `patch_radial_npz`
    stores meta_json "radial_learned": true; WB, Wpair, the coupling and the
    pair radial are copied verbatim (random-weight timing models).  Returns
    (n_active, n_rows) of the radial table.  Deterministic: same src and
    arguments, same arrays."""
    import jax
    jax.config.update("jax_enable_x64", True)            # to_analytic's projection is f64
    import numpy as np
    from ace_jax.basis.export import patch_radial_npz
    from ace_jax.eval import load
    from ace_jax.fit.radial_model import row_active, to_analytic, with_radial
    model, _, _ = load(str(src))
    m, _ = to_analytic(model, n_q)
    W = np.asarray(m.rnl_Wnlq, np.float64)
    act = np.asarray(row_active(W))
    rms = np.sqrt(np.mean(W[act] ** 2, axis=-1, keepdims=True))
    rng = np.random.default_rng(seed)
    W = W.copy()
    W[act] += scale * rms * rng.standard_normal(W[act].shape)
    patch_radial_npz(src, dst, with_radial(m, W, learned=True))
    return int(act.sum()), int(act.size)


def build_ace_learned():
    """The learned-radial proxies, from the medium linear ACE models (run
    `models.py ace` first).  Rebuilt every time: seconds per model."""
    import numpy as np
    for system in ELEMENTS:
        for size in LEARNED_SIZES:
            src, dst = DIR / f"ace_{system}_{size}.npz", learned_path(system, size)
            n_act, n_rows = learned_proxy(src, dst, **LEARNED_RECIPE)
            _update_manifest({str(dst): {
                "builder": "models.py ace-learned (learned_proxy)", "from": str(src),
                "from_sha256": _sha(src), **LEARNED_RECIPE,
                "recipe": "to_analytic(n_q); active rnl_Wnlq rows += scale * row rms * N(0,1), "
                          "default_rng(seed); with_radial(learned=True); patch_radial_npz",
                "active_rows": n_act, "rows": n_rows, "radial_learned": True,
                "n_params": int(np.load(dst)["WB"].shape[0]), "weights": "random",
                "sha256": _sha(dst)}})


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


def pace_functions_per_element(yace):
    """Basis functions per central element of a .yace (the `functions:` block
    lists them per element, one `    - {...}` line each; the counts are equal)."""
    counts, on = {}, False
    for line in pathlib.Path(yace).read_text().splitlines():
        if line.startswith("functions:"):
            on, el = True, None
        elif on and line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            el = line.strip()[:-1]
        elif on and line.startswith("    - "):
            counts[el] = counts.get(el, 0) + 1
        elif on and line and not line.startswith(" "):
            on = False
    return max(counts.values())


def ace_functions_per_element(npz):
    """Basis functions per central element of a linear ACE model: every B
    function carries its own weight per central element (WB is n_B x NZ), so
    it is n_B -- not n_B / NZ."""
    import numpy as np
    return int(np.load(npz, allow_pickle=True)["WB"].shape[0])


def basis_sizes():
    """{"<system>/<size>": {code: functions per central element}} for the
    ace-jax and ML-PACE models present in DIR (MACE has no comparable count)."""
    out = {}
    for r in planned_models():
        p = pathlib.Path(r["path"])
        if r["code"] == "mace" or not p.exists():
            continue
        n = (ace_functions_per_element(p) if r["code"].startswith("acejax-ace")
             else pace_functions_per_element(p))
        out.setdefault(f"{r['system']}/{r['size']}", {})[r["code"]] = n
    return out


def write_sizes():
    SIZES_JSON.write_text(json.dumps(basis_sizes(), indent=1, sort_keys=True) + "\n")
    print(SIZES_JSON)


if __name__ == "__main__":
    {"pace": build_pace, "ace": build_ace, "ace-learned": build_ace_learned, "mace": build_mace,
     "sizes": write_sizes}[sys.argv[1]]()
