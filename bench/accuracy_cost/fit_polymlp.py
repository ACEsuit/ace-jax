"""Polynomial MLPs (Seko; pypolymlp, BSD-3) fitted with pypolymlp on the train split, on a CPU.

    <ac-polymlp venv>/bin/python -m accuracy_cost.fit_polymlp SIZE [SIZE ...]      (from bench/)
    <ac-polymlp venv>/bin/python -m accuracy_cost.fit_polymlp --count              (coefficients per sweep entry)

- Venv: conda-forge pypolymlp (envs/polymlp/environment.yml); no ace_jax there, so the split is read by
  split._read_plain (as the pacemaker venv does).
- Model: the sweep entry sets model_type, max_p, feature_type, gtinv_order, gtinv_maxl and n_gaussians (Seko's
  grid, pypolymlp utils/grid_search); cutoff RCUT; gtinv_version 1 (the only one the LAMMPS package reads).
- Energies: pypolymlp has no intercept.  `atomic_energy` E0_SI is subtracted from the totals, and `calculator`
  adds it back (the saved polymlp.yaml and LAMMPS give energies relative to the isolated atom).
- Regulariser: pypolymlp solves ridge for a grid of alphas (10^linspace(*REG_ALPHA)) and keeps the one with the lowest weighted RMSE on the datasets it is given as "test".  Here those
  are a validation split of *train* (`val_split`: VAL_FRAC of each config type, seeded; the indices are saved in
  the run directory).  The benchmark test set never enters pypolymlp.  The coefficients are the train-only
  solution at that alpha (pypolymlp does not refit on train + validation).
- Weights: pypolymlp has one scalar per dataset, which multiplies that dataset's energy, force and stress
  residuals.  One dataset per (config type, has a virial), weight sigma_F,default / sigma_F: the square root of
  DATA/weights_mace.json, as fit_nep.py.  pypolymlp's own row weights stay on (energy rows by energy tier,
  force components above 1 eV/A as 1/|F|, stress rows 0.1 with large values damped: mlp_dev/core/utils_weights.py).
- Virials: fitted for the configurations that have one (include_stress per dataset).  pypolymlp's "stress" is
  sum r (x) f in eV per cell, the libAtoms virial, so info["virial"] goes in unchanged.
- Effective size: `design_rank` is the numerical rank (singular values above RANK_TOL x the largest) of the
  weighted training design matrix with its columns scaled to unit norm, next to the coefficient count.
"""
import json
import resource
import time

import numpy as np

from . import common
from .common import DATA, E0_SI, RCUT

VAL_FRAC, VAL_SEED = 0.1, 0
# log10 grid of ridge alphas.  Seko's grid search uses (-4, 1, 6) (utils/grid_search/api_grid_search.py); on this
# training set validation picked its smallest value (smoke fit of m3g6-l44), so the grid continues down to 1e-8
REG_ALPHA = (-8.0, 1.0, 10)
RANK_TOL = 1e-10
RANK_CHUNK_ATOMS = 2000                       # atoms per design-matrix block (6000 force rows)


def params_kw(e):
    """Pypolymlp.set_params arguments for a sweep entry."""
    kw = dict(elements=["Si"], cutoff=float(e.get("cutoff", RCUT)), model_type=int(e["model_type"]),
              max_p=int(e["max_p"]), feature_type=e.get("feature_type", "gtinv"), n_gaussians=int(e["n_gaussians"]),
              reg_alpha_params=tuple(e.get("reg_alpha", REG_ALPHA)), atomic_energy=[E0_SI], gtinv_version=1,
              include_force=True, include_stress=True)
    if kw["feature_type"] == "gtinv":
        kw |= dict(gtinv_order=int(e["gtinv_order"]), gtinv_maxl=tuple(e["gtinv_maxl"]))
    return kw


def model(e):
    from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
    p = Pypolymlp()
    p.set_params(**params_kw(e))
    return p


def val_split(types, frac=VAL_FRAC, seed=VAL_SEED):
    """(train indices, validation indices) of the train split: round(frac x n) frames of each config type
    (so none from a type with fewer than 1 / (2 frac) frames, which stays whole in train), seeded."""
    rng = np.random.default_rng(seed)
    types = np.asarray(types)
    val = []
    for t in sorted(set(types)):
        idx = np.flatnonzero(types == t)
        k = int(round(frac * len(idx)))
        val += sorted(rng.choice(idx, k, replace=False).tolist()) if k else []
    val = sorted(val)
    keep = np.ones(len(types), bool); keep[val] = False
    return np.flatnonzero(keep).tolist(), val


def structure(at):
    from pypolymlp.core.data_format import PolymlpStructure
    n = len(at)
    return PolymlpStructure(axis=np.asarray(at.cell).T.copy(), positions=at.get_scaled_positions().T.copy(),
                            n_atoms=[n], elements=list(at.get_chemical_symbols()), types=[0] * n,
                            volume=float(at.get_volume()))


def groups(atoms):
    """{(config type, has virial): [atoms]} in a fixed order: pypolymlp's unit for weights and stress rows."""
    out = {}
    for at in atoms:
        out.setdefault((at.info["config_type"], "virial" in at.info), []).append(at)
    return dict(sorted(out.items()))


def dataset(name, ats, weight, stress):
    from pypolymlp.core.dataset import Dataset
    from pypolymlp.core.dataset_utils import DatasetDFT
    dft = DatasetDFT([structure(a) for a in ats], np.array([a.info["energy"] for a in ats], float),
                     forces=[np.asarray(a.arrays["forces"], float).T.copy() for a in ats],
                     stresses=np.array([np.reshape(a.info["virial"], (3, 3)) for a in ats]) if stress else None,
                     elements=["Si"])
    return Dataset(dataset_type="vasp", files=name, name=name, include_force=True, include_stress=stress,
                   weight=float(weight), dft=dft)


def dataset_list(atoms, weights, tag, chunk_atoms=None):
    """DatasetList of `atoms`: one Dataset per (config type, has virial), weight from `weights`; with
    `chunk_atoms`, each split into blocks of at most that many atoms (design_rank's memory bound).  Energies are
    relative to E0_SI (pypolymlp subtracts params.atomic_energy)."""
    from pypolymlp.core.dataset import DatasetList
    ds = []
    for (t, v), ats in groups(atoms).items():
        blocks, cur, n = [], [], 0
        for a in ats:
            if chunk_atoms and cur and n + len(a) > chunk_atoms:
                blocks.append(cur); cur, n = [], 0
            cur.append(a); n += len(a)
        blocks.append(cur)
        for k, b in enumerate(blocks):
            ds.append(dataset(f"{tag}:{t}:{'v' if v else 'nov'}:{k}", b, weights.get(t, weights["Default"]), v))
    return DatasetList(ds).subtract_atomic_energy([E0_SI])


def fit_weights():
    return {k: float(np.sqrt(v)) for k, v in json.loads((DATA / "weights_mace.json").read_text()).items()}


def design_rank(prm, atoms, weights, tol=RANK_TOL, chunk_atoms=RANK_CHUNK_ATOMS):
    """Numerical rank of the weighted training design matrix (energy, stress and force rows as pypolymlp weights
    them), columns scaled to unit norm: the triangular factor of an updating QR over blocks, then its singular
    values; rank = #(s > tol x s_max).  Returns {"rank", "n_rows", "s_max", "s_min", "tol"}."""
    import scipy.linalg as sla
    from pypolymlp.core.dataset import DatasetList
    from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore
    core, R, rows = PolymlpDevCore(prm), None, 0
    full = dataset_list(atoms, weights, "rank", chunk_atoms=chunk_atoms)
    for ds in full:
        xy = core.calc_xy(DatasetList([ds]), scales=np.ones(core.n_features), min_energy=-1.0)
        X = np.asarray(xy.x)
        X = X[np.any(X != 0, axis=1)]                       # zero-weight rows (a stress-less dataset's stress rows)
        rows += len(X)
        R = sla.qr(X if R is None else np.vstack([R, X]), mode="r", overwrite_a=True, check_finite=False)[0]
        R = R[:min(R.shape)]
    norm = np.linalg.norm(R, axis=0)
    s = sla.svdvals(R / np.where(norm > 0, norm, 1.0), check_finite=False)
    return {"rank": int(np.sum(s > tol * s[0])), "n_rows": int(rows), "s_max": float(s[0]), "s_min": float(s[-1]),
            "tol": tol}


def versions():
    import pypolymlp
    return {"pypolymlp": pypolymlp.__version__, "numpy": np.__version__}


def fit(size, rank=True):
    from pypolymlp.mlp_dev.fit.api_fit import fit_polymlp
    from .split import ensure_split, read_atoms
    h = ensure_split()
    e, d = common.entry("polymlp", size), common.run_dir("polymlp", size)
    st = common.read_status(d)
    if st and st.get("status") == "ok" and (d / "polymlp.yaml").exists():
        return st
    d.mkdir(parents=True, exist_ok=True)
    atoms, w = read_atoms(DATA / "train.xyz"), fit_weights()
    tr, va = val_split([a.info["config_type"] for a in atoms])
    (d / "val_split.json").write_text(json.dumps({"seed": VAL_SEED, "frac": VAL_FRAC, "train": tr, "val": va}))
    p = model(e)
    prm = p.parameters
    t0 = time.perf_counter()
    try:
        res = fit_polymlp(prm, dataset_list([atoms[i] for i in tr], w, "train"),
                          dataset_list([atoms[i] for i in va], w, "val"), batch_size=e.get("batch_size"),
                          verbose=True)
        best, alphas = res.best_model, [float(m.alpha) for m in res.all_models]
        best.save_mlp(filename=str(d / "polymlp.yaml"))
        extra = {"n_params": int(p.n_features), "alpha": float(best.alpha), "alphas": alphas,
                 "val_rmse_weighted": [float(m.rmse_test) for m in res.all_models],
                 "alpha_at_edge": bool(float(best.alpha) in (min(alphas), max(alphas))),
                 "fit_wall_s": time.perf_counter() - t0}
        if rank and e.get("rank", True):                   # (`rank: false` on a sweep entry: too large for the SVD)
            t1 = time.perf_counter()
            extra |= {"design_rank": design_rank(prm, [atoms[i] for i in tr], w), "rank_wall_s": None}
            extra["rank_wall_s"] = time.perf_counter() - t1
        status = "ok"
    except Exception as ex:                                  # a failure is recorded, not retried (common.fit_once)
        status, extra = "failed", {"error": repr(ex)[-2000:], "fit_wall_s": time.perf_counter() - t0}
    common.write_status(d, family="polymlp", size=size, status=status, model=str(d / "polymlp.yaml"),
                        split_hash=h, e0_per_atom=E0_SI, n_train=len(tr), n_val=len(va), entry=e,
                        fit_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                        versions=versions(), **extra)
    return common.read_status(d)


def calculator(path):
    """ASE calculator on a polymlp.yaml: pypolymlp's evaluator (float64) with E0_SI per atom added back and the
    stress in ASE's convention.  pypolymlp returns sum r (x) f = -V sigma in eV per cell, ordered xx yy zz xy yz
    zx (its own PolymlpASECalculator passes that through unconverted)."""
    from ase.calculators.calculator import Calculator, all_changes
    from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

    class Polymlp(Calculator):
        implemented_properties = ["energy", "free_energy", "forces", "stress"]

        def __init__(self, pot, **kw):
            super().__init__(**kw)
            self.mlp = PypolymlpCalc(pot=str(pot), verbose=False)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            E, F, S = self.mlp.eval([structure(self.atoms)])
            E = float(np.ravel(E)[0]) + len(self.atoms) * E0_SI
            s = np.ravel(S)[:6]
            self.results = {"energy": E, "free_energy": E, "forces": np.asarray(F[0]).T.copy(),
                            "stress": -s[[0, 1, 2, 4, 5, 3]] / self.atoms.get_volume()}
    return Polymlp(path)


def raw_ef(path, at):
    """E (relative to the isolated atom: what LAMMPS evaluates) and F of the saved model: the timing gate's
    reference."""
    from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
    E, F, _ = PypolymlpCalc(pot=str(path), verbose=False).eval([structure(at)])
    return float(np.ravel(E)[0]), np.asarray(F[0]).T.copy()


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--count"]:
        for e in common.load_sweep()["polymlp"]:
            print(e["size"], model(e).n_features, e.get("n_params"))
    else:
        for s in sys.argv[1:]:
            r = fit(s)
            print(json.dumps({k: r.get(k) for k in ("size", "status", "n_params", "alpha", "alpha_at_edge",
                                                    "design_rank", "fit_wall_s", "error")}))
