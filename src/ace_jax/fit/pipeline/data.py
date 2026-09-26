from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ...eval import load
from ..data import build_dataset, load_configs


class FitData(NamedTuple):
    train: list; test: list; ood: list
    train_o: list; test_o: list; ood_o: list
    base_train: list; base_test: list; base_ood: list
    E0: np.ndarray
    ds_train: object; ds_test: object; ds_ood: object
    model: object; meta: dict; z: object
    perm: object


def _zero_base(c):
    return (0.0, np.zeros((len(c.numbers), 3)), np.zeros((3, 3)))


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
    """sigma_type: a weight-neutral named-weights dict over the file's config_type
    labels, so load_configs sets each config's type index (run.py behaviour)."""
    from ase.io import read
    cts = []
    for at in read(path, index=":"):
        ct = str(at.info.get("config_type", ""))
        if ct and ct not in cts:
            cts.append(ct)
    return {"default": {"E": 1.0, "F": 1.0, "V": 1.0}, **{ct: {"E": 1.0, "F": 1.0, "V": 1.0} for ct in cts}}


def load_fit_data(cfg, *, data=None, train=None, test=None, ood=None):
    """Configs, baselines, E0 and datasets.  Either `data` (one file, split with
    split_configs) or `train` (+ optional `test`; defaults to train) files."""
    if (data is None) == (train is None):
        raise ValueError("pass exactly one of data= (split) or train= (+ test=)")
    model, meta, z = load(cfg.model)
    keys = dict(energy_key=cfg.energy_key, force_key=cfg.force_key, virial_key=cfg.virial_key)
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
    if cfg.e0 == "lsq":
        counts = np.array([[np.sum(c.numbers == e) for e in els] for c in tr], float)
        E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in tr]), rcond=None)
        model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
    elif cfg.e0 == "model":
        E0 = np.asarray(z["E0"])
    else:
        raise ValueError(f"e0 must be 'lsq' or 'model', got {cfg.e0!r}")
    ds_train = build_dataset(tr, meta, E0, cfg.batch)
    ds_test = build_dataset(te, meta, E0, cfg.batch)
    ds_ood = build_dataset(od, meta, E0, cfg.batch) if od else None
    return FitData(tr, te, od, train_o, test_o, ood_o, base_train, base_test, base_ood,
                   np.asarray(E0), ds_train, ds_test, ds_ood, model, meta, z, perm)
