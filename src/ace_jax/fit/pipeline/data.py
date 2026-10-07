import dataclasses
import io
import os
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ...eval import load
from ...basis.model import basis_r0
from ..data import build_dataset, load_configs


class FitData(NamedTuple):
    train: list; test: list; ood: list
    train_o: list; test_o: list; ood_o: list
    base_train: list; base_test: list; base_ood: list
    E0: np.ndarray
    ds_train: object; ds_test: object; ds_ood: object
    model: object; meta: dict; z: object
    perm: object
    r0: float | None = None             # mean radial length of the basis, when it records one
    source: str = ""                    # log label: the model path, or "basis Si order 3 ..."
    basis: object = None                # the Basis when built/passed in memory, else None; its
                                        # .model is `model` before E0 (one copy of the arrays)


def _zero_base(c):
    return (0.0, np.zeros((len(c.numbers), 3)), np.zeros((3, 3)))


def infer_elements(*splits):
    """Sorted atomic numbers present in any of the config lists."""
    return sorted({int(z) for s in splits for c in s for z in np.unique(c.numbers)})


def _symbols(elements):
    from ase.data import chemical_symbols
    from ...basis.model import _zs
    return [chemical_symbols[z] for z in _zs(elements)]


def _model_source(cfg, train_o, test_o, ood_o, log):
    """(load target, label, Basis or None) for cfg.model: a path stays a path; a
    BasisSpec is built (species from the data unless given) and, like a Basis,
    handed on as an in-memory npz -- approach A of the fit-builds-basis spec,
    so everything downstream stays on the file-shaped path."""
    from ...basis.export import save_npz
    from ...basis.model import Basis, BasisSpec, _zs, build_basis
    m = cfg.model
    if isinstance(m, BasisSpec):
        seen = infer_elements(train_o, test_o, ood_o)
        if not seen:
            raise ValueError("no configurations to infer the basis elements from")
        inferred = m.elements is None
        if inferred:
            m = dataclasses.replace(m, elements=tuple(seen))
        missing = sorted(set(seen) - set(_zs(m.elements)))
        if missing:
            raise ValueError(f"species {', '.join(_symbols(missing))} in the data are not in the "
                             f"basis elements {list(m.elements)}")
        label = f"basis {','.join(_symbols(m.elements))} order {m.order} max-degree {m.max_degree}"
        b = build_basis(m, seed=cfg.seed)
        log(f"{label} -> {b.meta['n_B']} B functions" + (" (elements from the data)" if inferred else ""))
        m = b
    else:
        label = None
    if isinstance(m, Basis):
        buf = io.BytesIO()
        save_npz(buf, m)
        buf.seek(0)
        return buf, label or f"basis {','.join(_symbols(m.meta['elements']))}", m
    return m, str(m), None


def _isolated(c, rcut):
    """A single atom with no neighbour (periodic image) within rcut: its energy is E0 alone."""
    if len(c.numbers) != 1 or c.energy is None:
        return False
    if not np.any(c.pbc):
        return True
    from ...eval import sparse_graph
    return len(sparse_graph(c.positions, c.cell, c.pbc, rcut).senders) == 0


def lsq_e0(configs, elements, rcut, log=print):
    """E0 per element (in `elements` order) for e0='lsq'.  An isolated-atom config
    is predicted as E0 alone, so a least-squares compromise between it and the bulk
    shifts every energy: a species with isolated atoms takes E0 = their mean energy,
    and the other species are fitted by least squares to the remaining configs'
    energies with the fixed E0s subtracted.  With no isolated atoms this is plain
    least squares on the composition counts, as before."""
    iso = [_isolated(c, rcut) for c in configs]
    fixed = {}
    for e in elements:
        Es = [c.energy for c, i in zip(configs, iso) if i and c.numbers[0] == e]
        if Es:
            fixed[e] = float(np.mean(Es))
    if not fixed:
        counts = np.array([[np.sum(c.numbers == e) for e in elements] for c in configs], float)
        E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in configs]), rcond=None)
        return E0
    log(f"E0: isolated-atom energies for {', '.join(f'Z={e} {v:.6f}' for e, v in fixed.items())}")
    free = [e for e in elements if e not in fixed]
    E0 = np.array([fixed.get(e, 0.0) for e in elements])
    if free:
        lab = [c for c, i in zip(configs, iso) if not i and c.energy is not None]
        counts = np.array([[np.sum(c.numbers == e) for e in free] for c in lab], float)
        r = np.array([c.energy - sum(np.sum(c.numbers == e) * v for e, v in fixed.items()) for c in lab])
        sol, *_ = np.linalg.lstsq(counts, r, rcond=None)
        for e, v in zip(free, sol):
            E0[elements.index(e)] = v
    return E0


def split_configs(configs, ntrain, ntest, test_start=None, seed=0):
    """run.py's split: a seeded permutation; train = perm[:ntrain], test =
    perm[test_start : test_start + ntest] (test_start defaults to ntrain)."""
    n = len(configs)
    ts = ntrain if test_start is None else test_start
    if ntrain > n:
        raise ValueError(f"ntrain = {ntrain} but the data file has only {n} configs")
    if ts + ntest > n:
        raise ValueError(f"the test slice [{ts}, {ts + ntest}) runs past the {n} configs in the data file "
                         f"(set ntest / test_start)")
    perm = np.random.default_rng(seed).permutation(n)
    return [configs[i] for i in perm[:ntrain]], [configs[i] for i in perm[ts:ts + ntest]], perm


def _config_type_weights(path):
    """sigma_type: a weight-neutral named-weights dict over the file's (or Atoms list's)
    config_type labels, so load_configs sets each config's type index (run.py behaviour)."""
    from ..xyz import read_extxyz
    cts = []
    for at in (read_extxyz(path) if isinstance(path, (str, os.PathLike)) else path):
        ct = str(at.info.get("config_type", ""))
        if ct and ct not in cts:
            cts.append(ct)
    return {"default": {"E": 1.0, "F": 1.0, "V": 1.0}, **{ct: {"E": 1.0, "F": 1.0, "V": 1.0} for ct in cts}}


def load_fit_data(cfg, *, data=None, train=None, test=None, ood=None, log=print):
    """Configs, baselines, E0 and datasets.  Either `data` (one file, split with
    split_configs) or `train` (+ optional `test`; defaults to train) files.  Each may be
    a path or a list of ase.Atoms (see fit.data.load_configs).  With no test file the
    test Dataset is the training one (ds_test is ds_train), not a second build."""
    if (data is None) == (train is None):
        raise ValueError("pass exactly one of data= (split) or train= (+ test=)")
    keys = dict(energy_key=cfg.energy_key, force_key=cfg.force_key, virial_key=cfg.virial_key,
                stress_key=cfg.stress_key)
    if cfg.factors:
        keys["factors"] = cfg.factors
    if cfg.sigma_type:
        keys["weights"] = _config_type_weights(data or train)
    elif cfg.weights is not None:
        keys["weights"] = cfg.weights
    perm = None
    if data is not None:
        train_o, test_o, perm = split_configs(load_configs(data, **keys), cfg.ntrain, cfg.ntest,
                                              cfg.test_start, cfg.seed)
    else:
        train_o = load_configs(train, **keys)
        test_o = load_configs(test, **keys) if test else train_o
    ood_o = load_configs(ood, **keys) if ood else []
    # the model after the configs: a basis built here takes its species from them
    src, label, basis = _model_source(cfg, train_o, test_o, ood_o, log)
    model, meta, z = load(src)
    if basis is not None:               # drop the built model's own arrays: the loaded one replaces it
        basis = basis._replace(model=model)
    from ...eval.pace_model import PACEModel
    if isinstance(model, PACEModel):
        raise ValueError(f"{label}: a PACE .yace model is evaluate-only and cannot be fitted; "
                         "fit a linear ACE model (.npz, e.g. from `aj basis`) instead")

    if cfg.baseline:
        from ..baseline import load_mean, subtract_baseline
        mean = load_mean(cfg.baseline)
        tr, base_train = subtract_baseline(train_o, mean)
        te, base_test = subtract_baseline(test_o, mean)
        od, base_ood = subtract_baseline(ood_o, mean) if ood_o else ([], [])
    elif cfg.base_npz:
        if perm is None:
            raise ValueError("base_npz offsets are indexed by the data file: use data=")
        zb = np.load(cfg.base_npz)
        off = np.concatenate([[0], np.cumsum(zb["natoms"])])
        ts = cfg.ntrain if cfg.test_start is None else cfg.test_start
        base = lambda idx: [(float(zb["E"][i]), zb["F"][off[i]:off[i + 1]], zb["V"][i]) for i in idx]
        sub = lambda cs, bs: [c._replace(energy=None if c.energy is None else c.energy - b[0],
                                         forces=None if c.forces is None else c.forces - b[1],
                                         virial=None if c.virial is None else c.virial - b[2])
                              for c, b in zip(cs, bs)]
        base_train, base_test = base(perm[:cfg.ntrain]), base(perm[ts:ts + cfg.ntest])
        tr, te, od = sub(train_o, base_train), sub(test_o, base_test), ood_o
        base_ood = [_zero_base(c) for c in ood_o]
    else:
        tr, te, od = train_o, test_o, ood_o
        base_train = [_zero_base(c) for c in train_o]
        base_test = [_zero_base(c) for c in test_o]
        base_ood = [_zero_base(c) for c in ood_o]

    els = [int(e) for e in meta["elements"]]
    if cfg.e0 in ("lsq", "prefit"):    # lsq refines it jointly in the fit (cfg.joint_e0)
        E0 = lsq_e0(tr, els, float(meta["rcut"]), log=log)
        model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
    elif cfg.e0 == "model":
        E0 = np.asarray(z["E0"])
    else:
        raise ValueError(f"e0 must be 'lsq', 'prefit' or 'model', got {cfg.e0!r}")
    pk = dict(pack=cfg.pack_mode, log=log)
    ds_train = build_dataset(tr, meta, E0, cfg.batch, **pk)
    if data is None and not test:     # no test file: the "test" split is the training set, as built
        log("no test set: test metrics are on the training set")
        ds_test = ds_train
    else:
        ds_test = build_dataset(te, meta, E0, cfg.batch, **pk)
    ds_ood = build_dataset(od, meta, E0, cfg.batch, **pk) if od else None
    return FitData(tr, te, od, train_o, test_o, ood_o, base_train, base_test, base_ood,
                   np.asarray(E0), ds_train, ds_test, ds_ood, model, meta, z, perm,
                   r0=basis_r0(meta), source=label, basis=basis)
