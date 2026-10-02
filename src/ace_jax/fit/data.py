"""extxyz -> padded, stacked batches on device.

ACEfit conventions (src/atoms_data.jl): per configuration the observations are
[E, F (atom-major, xyz), V (6, Voigt)], the energy target is E - sum_i E0[z_i],
and the structural weights are 1/sqrt(n_atoms) on E and V rows, 1 on F rows,
times the per-config-type (E, F, V) weight dict.  Padded or absent observations
carry weight ZERO -- that is how masking reaches the sufficient statistics."""
import os
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
    config_type: str = None    # the config_type label as written (reports, eval output)


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


def _label(value, key, shape, where):
    """A label as float64 of `shape`, or a ValueError naming it -- never None for
    a value that is present but unreadable (that would silently drop the label)."""
    try:
        a = np.asarray(value, float)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: label {key!r} is not numeric: {value!r:.80}") from None
    if a.size != int(np.prod(shape)):
        raise ValueError(f"{where}: label {key!r} has {a.size} values, expected shape {shape}")
    return a.reshape(shape)


def _atoms_frames(source):
    """ase.Atoms -> fit.xyz.Frame records, keys as named; SinglePointCalculator results
    (what MACE and ASE leave behind) become info energy/stress and arrays forces."""
    from .xyz import Frame
    out = []
    for a in source:
        info = dict(a.info)
        arrays = {k: np.asarray(v) for k, v in a.arrays.items() if k not in ("numbers", "positions")}
        res = getattr(a.calc, "results", None) or {}
        if "energy" in res or "free_energy" in res:
            info.setdefault("energy", float(res.get("energy", res.get("free_energy"))))
        if "stress" in res:
            info.setdefault("stress", np.asarray(res["stress"]))
        if "forces" in res:
            arrays.setdefault("forces", np.asarray(res["forces"]))
        cell = np.asarray(a.cell.array, float)
        out.append(Frame(np.asarray(a.numbers), np.asarray(a.positions, float), cell,
                         np.asarray(a.pbc, bool) if cell.any() else np.zeros(3, bool), info, arrays))
    return out


def _stress_to_virial(s, cell):
    """virial = -stress * volume; stress as 3x3, Voigt-6 (xx yy zz yz xz xy) or flat 9."""
    from ase.stress import voigt_6_to_full_3x3_stress
    s = np.asarray(s, float)
    S = voigt_6_to_full_3x3_stress(s) if s.shape == (6,) else s.reshape(3, 3)
    return -S * abs(np.linalg.det(np.asarray(cell, float)))


def load_configs(source, energy_key="energy", force_key="forces", virial_key="virial", stress_key=None,
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
    `factors` -- that wiring is untouched.

    Read with libAtoms extxyz (`fit.xyz.read_extxyz`): every label comes back
    under the name it was written with, including `energy` / `forces`, which
    ase.io hides in a calculator.  `source` may instead be a list of ase.Atoms
    (labels in info / arrays, or a SinglePointCalculator's results).  `stress_key`:
    a periodic config with no virial label takes virial = -stress * volume from it."""
    from .xyz import read_extxyz
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
    out, found, seen, periodic = [], set(), {"info": set(), "arrays": set()}, False
    is_path = isinstance(source, (str, os.PathLike))
    for i, at in enumerate(read_extxyz(source) if is_path else _atoms_frames(source)):
        n, where = len(at.numbers), (f"{source} config {i}" if is_path else f"structure {i}")
        ct = str(at.info.get(weight_key, ""))
        ti = next((idx for name, idx in type_index.items() if name.lower() == ct.lower()), 0)
        E = _get(at.info, energy_key)
        F = _get(at.arrays, force_key)
        V = _get(at.info, virial_key)
        if V is None and stress_key is not None and np.any(at.pbc):
            S = _get(at.info, stress_key)
            if S is not None:
                V = _stress_to_virial(S, at.cell); found.add(virial_key)
        found |= {k for k, v in ((energy_key, E), (force_key, F), (virial_key, V)) if v is not None}
        seen["info"] |= set(at.info); seen["arrays"] |= set(at.arrays)
        periodic = periodic or bool(np.any(at.pbc))
        meta = {"n_atoms": n, "config_type": at.info.get(weight_key), **at.info}
        out.append(Config(
            positions=at.positions, numbers=at.numbers, cell=at.cell, pbc=at.pbc,
            energy=None if E is None else float(_label(E, energy_key, (), where)),
            forces=None if F is None else _label(F, force_key, (n, 3), where),
            virial=None if V is None else _label(V, virial_key, (3, 3), where),
            w_E=weigh(meta, "E"), w_F=weigh(meta, "F"), w_V=weigh(meta, "V"),
            type_idx=ti, config_type=None if at.info.get(weight_key) is None else str(at.info.get(weight_key))))
    # a key the caller named (not the default) that no config has is a typo, not "no label"
    for key, default, where, expected in ((energy_key, "energy", "info", True),
                                          (force_key, "forces", "arrays", True),
                                          (virial_key, "virial", "info", periodic)):   # no virial without a cell
        if expected and key is not None and key != default and key not in found and out:
            raise ValueError(f"{source if is_path else 'structures'}: no config has the label {key!r}; its per-config {where} keys are "
                             f"{sorted(seen[where] - {'config_type'})}")
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
    # symmetric cutoff list: <= k_cap edges into any node (predict._dtc_deriv_residual relies on it)
    assert not nbr.size or np.bincount(nbr[nmask], minlength=1).max() <= k_cap
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
    # at least one (masked) neighbour slot: a batch of edgeless structures (an
    # isolated atom) still needs an (n, K, 3) edge array, and the masked slot is
    # parked at the cutoff, so it adds nothing -- E0 + the empty-environment term
    k_cap = max(int(k_cap), 1)
    batches = [_batch(g, meta, E0, rcut, C, n_cap, k_cap) for g in groups]
    stack = lambda k: jnp.asarray(np.stack([b[k] for b in batches]))
    return Dataset(**{k: stack(k) for k in Dataset._fields})
