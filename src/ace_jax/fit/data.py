"""extxyz -> padded, stacked batches on device.

ACEfit conventions (src/atoms_data.jl): per configuration the observations are
[E, F (atom-major, xyz), V (6, Voigt)], the energy target is E - sum_i E0[z_i],
and the structural weights are 1/sqrt(n_atoms) on E and V rows, 1 on F rows,
times the per-config-type (E, F, V) weight dict.  Padded or absent observations
carry weight ZERO -- that is how masking reaches the sufficient statistics."""
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from ..eval import dense_graph, species_indices, sparse_graph
from .weights import ConfigType, Structural, compose

VOIGT = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))


class Config(NamedTuple):
    positions: np.ndarray
    numbers: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    energy: float | None
    forces: np.ndarray | None
    virial: np.ndarray | None
    w_E: float
    w_F: float
    w_V: float
    type_idx: int = 0          # config-type index (default type -> 0), for per-type sigma


def _get(d, key):
    """ACEfit's fuzzy lookup: exact key, else case-insensitive, else None."""
    if key is None:
        return None
    if key in d:
        return d[key]
    for k in d:
        if k.lower() == key.lower():
            return d[k]
    return None


def load_configs(path, energy_key="energy", force_key="forces", virial_key="virial",
                 weights=None, weight_key="config_type", factors=None):
    """`factors`: an optional list of `weights.WeightFactor`s (see
    `ace_jax.fit.weights`), composed via `compose()` into a single
    fn(meta, quantity) -> float that produces w_E/w_F/w_V.  `factors=None`
    (the default) reconstructs the CLASSIC weighting -- structural 1/sqrt(n)
    on E,V (1 on F), times the resolved per-config-type {E,F,V} dict from
    `weights`/`weight_key` -- as `[Structural(), ConfigType(...)]`, so every
    existing caller (none of which pass `factors`) gets bit-identical
    w_E/w_F/w_V to before this was added.  `weights`/`weight_key` ALSO still
    drive `type_idx` (Task 6's per-type sigma routing) independently of
    `factors` -- that wiring is untouched."""
    from ase.io import read
    weights = weights or {"default": {"E": 1.0, "F": 1.0, "V": 1.0}}
    if "default" not in weights:
        raise ValueError("weights dict needs a 'default' entry")
    # Type index: the default type is 0; each NAMED weights entry (in insertion
    # order) is 1, 2, ...  A config whose config_type matches a named key takes
    # that index, else the default 0.  A single default type -> every config 0.
    type_index = {name: i + 1 for i, name in enumerate(k for k in weights if k != "default")}
    if factors is None:
        named = {k: v for k, v in weights.items() if k != "default"}
        factors = [Structural(), ConfigType(named, key=weight_key, default=weights["default"])]
    weigh = compose(factors)
    out = []
    for at in read(str(path), index=":"):
        n = len(at)
        ct = str(at.info.get(weight_key, ""))
        ti = next((idx for name, idx in type_index.items() if name.lower() == ct.lower()), 0)
        E = _get(at.info, energy_key)
        F = _get(at.arrays, force_key)
        V = _get(at.info, virial_key)
        V = None if V is None else np.asarray(V, float).reshape(3, 3)
        meta = {"n_atoms": n, "config_type": at.info.get(weight_key), **at.info}
        out.append(Config(
            positions=np.asarray(at.positions, float), numbers=np.asarray(at.numbers),
            cell=np.asarray(at.cell.array, float), pbc=np.asarray(at.pbc, bool),
            energy=None if E is None else float(E),
            forces=None if F is None else np.asarray(F, float),
            virial=V,
            w_E=weigh(meta, "E"), w_F=weigh(meta, "F"), w_V=weigh(meta, "V"),
            type_idx=ti))
    return out


class Dataset(NamedTuple):
    rij: jnp.ndarray;       nbr: jnp.ndarray;       nbr_mask: jnp.ndarray
    node_z: jnp.ndarray;    node_cfg: jnp.ndarray;  node_mask: jnp.ndarray
    y_E: jnp.ndarray;       w_E: jnp.ndarray
    y_F: jnp.ndarray;       w_F: jnp.ndarray
    y_V: jnp.ndarray;       w_V: jnp.ndarray
    n_atoms: jnp.ndarray;   cfg_mask: jnp.ndarray
    cfg_type: jnp.ndarray;  node_type: jnp.ndarray   # per-config / per-node config-type index

    @property
    def n_batches(self):
        return self.y_E.shape[0]


def flat_edges(rij, nbr, nbr_mask):
    """Dense (Ncap, K) -> flat edge list, the layout `ACEModel.edge_jacobian`
    takes.  senders is the node axis repeated K times."""
    n, K = nbr.shape
    senders = jnp.repeat(jnp.arange(n, dtype=jnp.int32), K)
    return rij.reshape(n * K, 3), senders, nbr.reshape(n * K), nbr_mask.reshape(n * K)


def _batch(configs, meta, E0, rcut, C, n_cap, k_cap):
    """One padded batch from up to C configs."""
    rij, nbr, nmask, node_z, node_cfg, node_ty = [], [], [], [], [], []
    yE, wE, yV, wV, nat, cm = np.zeros(C), np.zeros(C), np.zeros((C, 6)), np.zeros(C), np.zeros(C), np.zeros(C, bool)
    ctype = np.zeros(C, np.int32)
    yF, wF = [], []
    off = 0
    for c, cfg in enumerate(configs):
        n = len(cfg.numbers)
        g = dense_graph(cfg.positions, cfg.cell, cfg.pbc, rcut, k_cap)   # padded slots at the cutoff
        m = g.mask
        rij.append(g.rij); nbr.append(np.where(m, g.idx + off, 0)); nmask.append(m)
        zi = species_indices(meta, cfg.numbers)
        node_z.append(zi); node_cfg.append(np.full(n, c)); node_ty.append(np.full(n, cfg.type_idx))
        nat[c] = n; cm[c] = True; ctype[c] = cfg.type_idx
        if cfg.energy is not None:
            yE[c] = cfg.energy - E0[zi].sum(); wE[c] = cfg.w_E
        if cfg.forces is not None:
            yF.append(cfg.forces); wF.append(np.full(n, cfg.w_F))
        else:
            yF.append(np.zeros((n, 3))); wF.append(np.zeros(n))
        if cfg.virial is not None:
            yV[c] = [cfg.virial[i, j] for i, j in VOIGT]; wV[c] = cfg.w_V
        off += n
    cat = lambda xs, dt=float: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    rij = cat(rij).reshape(-1, k_cap, 3) if rij else np.zeros((0, k_cap, 3))
    nbr, nmask = cat(nbr, np.int32).reshape(-1, k_cap), cat(nmask, bool).reshape(-1, k_cap)
    node_z, node_cfg = cat(node_z, np.int32), cat(node_cfg, np.int32)
    node_ty = cat(node_ty, np.int32)
    yF, wF = cat(yF).reshape(-1, 3), cat(wF)
    n_nodes = off
    assert n_nodes <= n_cap, (n_nodes, n_cap)
    pn = n_cap - n_nodes
    pad = lambda a, k, v: np.concatenate([a, np.full((k,) + a.shape[1:], v, a.dtype)])
    # padded nodes get K padded slots at the cutoff, where the envelope vanishes
    # and the gradient stays defined (tests/test_padding.py records why not zero)
    rij_pad = np.tile(np.array([rcut, 0.0, 0.0]), (pn, k_cap, 1))
    return dict(
        rij=np.concatenate([rij, rij_pad]), nbr=pad(nbr, pn, 0), nbr_mask=pad(nmask, pn, False),
        node_z=pad(node_z, pn, 0), node_cfg=pad(node_cfg, pn, C),
        node_mask=np.concatenate([np.ones(n_nodes, bool), np.zeros(pn, bool)]),
        y_E=yE, w_E=wE, y_F=pad(yF, pn, 0.0), w_F=pad(wF, pn, 0.0),
        y_V=yV, w_V=wV, n_atoms=nat, cfg_mask=cm,
        cfg_type=ctype, node_type=pad(node_ty, pn, 0))


def build_dataset(configs, meta, E0, configs_per_batch, rcut=None, n_cap=None, k_cap=None,
                  node_chunk=32):
    rcut = float(meta["rcut"] if rcut is None else rcut)
    E0 = np.asarray(E0, float)
    C = int(configs_per_batch)
    groups = [configs[i:i + C] for i in range(0, len(configs), C)]
    if n_cap is None:
        n_cap = max(sum(len(c.numbers) for c in g) for g in groups)
    n_cap = -(-n_cap // node_chunk) * node_chunk          # Task 7 scans node chunks
    if k_cap is None:
        k_cap = max(int(np.bincount(sparse_graph(c.positions, c.cell, c.pbc, rcut).senders,
                                    minlength=len(c.numbers)).max())
                    for c in configs)
    batches = [_batch(g, meta, E0, rcut, C, n_cap, k_cap) for g in groups]
    stack = lambda k: jnp.asarray(np.stack([b[k] for b in batches]))
    return Dataset(**{k: stack(k) for k in Dataset._fields})
