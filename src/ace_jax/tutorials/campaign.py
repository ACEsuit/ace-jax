"""Helpers for the curation tutorials (tutorial support, not stable API): per-atom
descriptors on a fixed reference basis, seeded MD pools, and descriptor-space novelty.

The MLIP school's workbench served these (get_descriptors, ace_md); here they are
ace-jax and ASE. The reference basis -- order 3, total degree 10, rcut 5.5, 120
functions for Si -- matches the school's fixed descriptor basis."""
import functools

import numpy as np


@functools.lru_cache(maxsize=8)
def reference_basis(elements=("Si",)):
    """The fixed descriptor basis (order 3, max degree 10, rcut 5.5); cached per element set."""
    from ..basis.model import BasisSpec, build_basis
    return build_basis(BasisSpec(order=3, max_degree=10, rcut=5.5, elements=tuple(elements)))


def _model_meta(basis_or_model):
    if hasattr(basis_or_model, "eval_pair"):             # a Basis
        return basis_or_model.eval_pair()
    from ..eval import load
    m, meta, _ = load(str(basis_or_model))               # a model file
    return m, meta


def atom_descriptors(structures, basis_or_model):
    """Per-structure (n_atoms, D) site descriptors, from a Basis or a model.npz path."""
    from ..eval.api import site_descriptors
    model, meta = _model_meta(basis_or_model)
    return [np.asarray(site_descriptors(model, a.positions, a.numbers, a.cell.array, a.pbc, meta=meta))
            for a in structures]


def md_pool(model_file, starts, *, temperature=400.0, dt_fs=1.0, n_steps=60, every=4, seed=0,
            friction=0.01):
    """Langevin MD (NVT) from each start with ACECalculator(model_file); the frames every
    `every` steps, the starting frame included (n_steps // every + 1 per start), concatenated
    in start order. Seeded: the same seed gives the same frames on one machine."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from ..calc.point import ACECalculator
    calc = ACECalculator(str(model_file))
    rng = np.random.default_rng(seed)
    frames = []
    for k, start in enumerate(starts):
        a = start.copy(); a.calc = calc
        MaxwellBoltzmannDistribution(a, temperature_K=temperature, rng=rng)
        dyn = Langevin(a, dt_fs * units.fs, temperature_K=temperature, friction=friction / units.fs,
                       rng=np.random.default_rng(seed + 1 + k))
        dyn.attach(lambda: frames.append(_frame(a, start)), interval=every)   # noqa: B023  (ASE also calls it at step 0)
        dyn.run(n_steps)
    return frames


def _frame(a, start):
    f = a.copy(); f.calc = None
    f.info = {k: v for k, v in start.info.items() if k in ("config_type", "miller")}
    f.wrap()
    return f


def physical(frame, min_sep=1.7):
    """No two atoms closer than min_sep (A, minimum image): MD at a bad potential can collapse."""
    if len(frame) < 2:
        return True
    d = frame.get_all_distances(mic=bool(np.any(frame.pbc)))
    return bool(d[np.triu_indices(len(frame), 1)].min() >= min_sep)


def _pairwise(a, b):
    """Euclidean distances (n, m) by the Gram identity, clamped at 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))


def nn_ratio(train_rows, query_rows, q=0.5):
    """The q-quantile (default the median; 1.0 the most exposed atom) of the query atoms'
    distances to their nearest training atom, over the median nearest-neighbour spacing within
    the training atoms (clamped at 1e-12: symmetric atoms of an unrattled crystal sit at zero
    spacing). Rows are per atom, (n, D)."""
    T = np.asarray(train_rows, float)
    dt = _pairwise(T, T); np.fill_diagonal(dt, np.inf)
    spacing = max(float(np.median(dt.min(1))), 1e-12)
    return float(np.quantile(_pairwise(np.asarray(query_rows, float), T).min(1), q) / spacing)


def score_novelty(pool_rows, train_rows):
    """Per pool structure: the largest, over its atoms, distance to the nearest training atom
    (the school's novelty score: one unfamiliar atom makes a structure novel)."""
    T = np.concatenate([np.asarray(r, float) for r in train_rows])
    return np.array([float(_pairwise(r, T).min(1).max()) for r in pool_rows])
