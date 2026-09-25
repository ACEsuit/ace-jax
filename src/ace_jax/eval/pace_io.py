"""PACE `.yace` (ACECTildeBasisSet YAML) reading and writing.

Keys and defaults follow ML-PACE `ace_c_basis.cpp::load_yaml` and
`ACERadialFunctions::init` (lammps-user-pace @ 99aa6e6).  Bond keys are flow
sequences (`[0, 1]:`), which PyYAML cannot hash, so they load as tuples.
"""
import copy
import warnings
from dataclasses import dataclass, field

import numpy as np
import yaml
from ase.data import atomic_numbers

try:
    _BaseLoader, _BaseDumper = yaml.CSafeLoader, yaml.CSafeDumper
except AttributeError:                       # PyYAML without libyaml
    _BaseLoader, _BaseDumper = yaml.SafeLoader, yaml.SafeDumper


class _Loader(_BaseLoader):
    pass


def _mapping(loader, node):
    loader.flatten_mapping(node)
    out = {}
    for kn, vn in node.value:
        k = loader.construct_object(kn, deep=True)
        out[tuple(k) if isinstance(k, list) else k] = loader.construct_object(vn, deep=True)
    return out


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


class _Dumper(_BaseDumper):
    pass


_Dumper.add_representer(tuple, lambda d, t: d.represent_sequence(
    "tag:yaml.org,2002:seq", list(t), flow_style=True))


def load_tree(path):
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)


def dump_tree(tree, path):
    with open(path, "w") as f:
        yaml.dump(tree, f, Dumper=_Dumper, default_flow_style=None,
                  sort_keys=False, width=1 << 30)


@dataclass
class PACESpec:
    """The parsed file, verbatim.  Kept beside the model (not on it: a dict is
    not a hashable Equinox static field) so export never regenerates layout."""
    tree: dict
    element_names: list
    functions: list = field(default_factory=list)   # (el, idx, term_start, num_ms, ndens)


# ACEBondSpecification::from_YAML (ace_abstract_basis.h): these are the values
# used when a key is absent; prehc / lambdahc are required there, and here.
_BOND_DEFAULTS = {"rcut_in": 0.0, "dcut_in": 0.0, "inner_cutoff_type": "density"}
_RADBASES = ("ChebExpCos", "ChebPow", "ChebLinear", "SBessel")
_INNER = ("density", "distance", "zbl")


def parse_yace(path):
    tree = load_tree(path)
    names = list(tree["elements"])
    NZ = len(names)
    try:
        Z = np.array([atomic_numbers[e] for e in names], np.int64)
    except KeyError as e:
        raise ValueError(f"unknown element {e.args[0]!r} in {path}") from None

    bonds = {k: {**_BOND_DEFAULTS, **v} for k, v in tree["bonds"].items()}
    for k, b in bonds.items():
        for key in ("prehc", "lambdahc"):
            if key not in b:
                raise ValueError(f"{path}: bond {list(k)} lacks required key {key!r}")
    kinds = {b["radbasename"] for b in bonds.values()}
    if any(k.startswith("ACE.jl") for k in kinds):
        raise NotImplementedError(f"{path}: ACE.jl tabulated radials are not supported")
    if len(kinds) != 1:
        raise NotImplementedError(f"{path}: per-bond differing radbasename {sorted(kinds)}")
    if not kinds <= set(_RADBASES):
        raise NotImplementedError(f"{path}: radbasename {sorted(kinds)} not supported "
                                  f"(supported: {', '.join(_RADBASES)})")
    inner = {b["inner_cutoff_type"] for b in bonds.values()}
    if len(inner) != 1:
        raise ValueError(f"{path}: bonds disagree on inner_cutoff_type {sorted(inner)}; "
                         "ML-PACE keeps one global value (last bond wins), so the file is ill-defined")
    if not inner <= set(_INNER):
        raise NotImplementedError(f"{path}: inner_cutoff_type {sorted(inner)} not supported "
                                  f"(supported: {', '.join(_INNER)})")
    for k in [(i, j) for i in range(NZ) for j in range(NZ)]:
        if k not in bonds:
            raise ValueError(f"{path}: missing bond {list(k)}")

    nradmax = max(int(b["nradmax"]) for b in bonds.values())
    lmax = max(int(b["lmax"]) for b in bonds.values())
    K = max(int(b["nradbasemax"]) for b in bonds.values())
    crad = np.zeros((NZ, NZ, nradmax, lmax + 1, K))
    radparams = np.zeros((NZ, NZ, 5))
    core = np.zeros((NZ, NZ, 2))
    for (i, j), b in bonds.items():
        c = np.asarray(b["radcoefficients"], float)          # [n][l][k]
        crad[i, j, :c.shape[0], :c.shape[1], :c.shape[2]] = c
        radparams[i, j] = [b["radparameters"][0], b["rcut"], b["dcut"], b["rcut_in"], b["dcut_in"]]
        core[i, j] = [b["prehc"], b["lambdahc"]]

    embs = tree["embeddings"]
    P = max(int(embs[e]["ndensity"]) for e in range(NZ))
    fs = np.zeros((NZ, 2 * P))
    fs[:, 1::2] = 1.0                                       # padded densities: w=0, m=1
    rho_core_cut = np.zeros((NZ, 2))
    npoti = []
    for e in range(NZ):
        em = embs[e]
        p = np.asarray(em["FS_parameters"], float)
        fs[e, :p.size] = p
        rho_core_cut[e] = [em["rho_core_cutoff"], em["drho_core_cutoff"]]
        npoti.append(em["npoti"])
        if em["npoti"] not in ("FinnisSinclair", "FinnisSinclairShiftedScaled"):
            raise NotImplementedError(f"{path}: embedding {em['npoti']!r}")

    funcs, spec_funcs, ctil = [], [], []
    term = 0
    for e in range(NZ):
        for idx, f in enumerate(tree["functions"].get(e, [])):
            rank, nms, nd = int(f["rank"]), int(f["num_ms_combs"]), int(f["ndensity"])
            ms = np.asarray(f["ms_combs"], np.int64).reshape(nms, rank)
            c = np.zeros((nms, P))
            c[:, :nd] = np.asarray(f["ctildes"], float).reshape(nms, nd)
            for n, l in zip(f["ns"], f["ls"]):
                if rank == 1 and not 1 <= n <= K:
                    raise ValueError(f"{path}: rank-1 function n={n} exceeds nradbasemax {K}")
                if rank > 1 and not (1 <= n <= nradmax and 0 <= l <= lmax):
                    raise ValueError(f"{path}: function (n={n}, l={l}) exceeds bond basis "
                                     f"(nradmax {nradmax}, lmax {lmax})")
            funcs.append({"el": e, "rank": rank, "mus": list(f["mus"]), "ns": list(f["ns"]),
                          "ls": list(f["ls"]), "ms": ms})
            spec_funcs.append((e, idx, term, nms, nd))
            ctil.append(c)
            term += nms
    arrays = {
        "E0": np.asarray(tree["E0"], float), "Z": Z, "crad": crad, "radparams": radparams,
        "core": core, "fs_params": fs, "rho_core_cut": rho_core_cut,
        "ctilde_complex": np.concatenate(ctil) if ctil else np.zeros((0, P)),
        "radbasename": kinds.pop(), "inner_cutoff_type": inner.pop(),
        "npoti": tuple(npoti), "ndensity": P, "nradmax": nradmax, "lmax": lmax,
        "nradbase": K, "funcs": funcs,
    }
    return PACESpec(tree=tree, element_names=names, functions=spec_funcs), arrays


def write_yace(model, spec, path):
    """Serialise a PACEModel: numeric fields from its leaves, everything else
    verbatim from `spec.tree` (function layout is never regenerated)."""
    from .pace_model import PACEModel
    if not isinstance(model, PACEModel):
        raise TypeError("write_yace only writes PACEModel (PACE radials); see docs/pace-yace-spec.md")
    if any(getattr(x, "dtype", np.float64) != np.float64 for x in
           (model.crad, model.ctilde_complex, model.fs_params, model.E0)):
        warnings.warn("write_yace: model leaves are float32, so coefficients are written at "
                      "float32 precision; load with dtype=float64 to export exactly",
                      UserWarning, stacklevel=2)
    f64 = lambda x: np.asarray(x, np.float64)
    t = copy.deepcopy(spec.tree)
    t["E0"] = f64(model.E0).tolist()
    fs, rcc = f64(model.fs_params), f64(model.rho_core_cut)
    for e in range(len(spec.element_names)):
        em = t["embeddings"][e]
        em["FS_parameters"] = fs[e, :2 * int(em["ndensity"])].tolist()
        em["rho_core_cutoff"], em["drho_core_cutoff"] = rcc[e].tolist()
    crad, rp, core = f64(model.crad), f64(model.radparams), f64(model.core)
    for (i, j), b in t["bonds"].items():
        n, l, k = int(b["nradmax"]), int(b["lmax"]), int(b["nradbasemax"])
        # padded entries (beyond this bond's own sizes, up to the global ones) are
        # live -- g_k runs to the global nradbase on every bond -- so if they were
        # edited, widen the bond rather than drop them.  ML-PACE sizes to the max.
        nzi = np.nonzero(crad[i, j])
        if nzi[0].size:
            n, l, k = (max(n, int(nzi[0].max()) + 1), max(l, int(nzi[1].max())),
                       max(k, int(nzi[2].max()) + 1))
            if (n, l, k) != (b["nradmax"], b["lmax"], b["nradbasemax"]):
                b["nradmax"], b["lmax"], b["nradbasemax"] = n, l, k
        b["radcoefficients"] = crad[i, j, :n, :l + 1, :k].tolist()
        b["radparameters"] = [float(rp[i, j, 0])] + list(b["radparameters"][1:])
        b["rcut"], b["dcut"] = float(rp[i, j, 1]), float(rp[i, j, 2])
        b["prehc"], b["lambdahc"] = float(core[i, j, 0]), float(core[i, j, 1])
        for key, v in (("rcut_in", rp[i, j, 3]), ("dcut_in", rp[i, j, 4])):
            if key in b or float(v) != _BOND_DEFAULTS[key]:
                b[key] = float(v)
    ct = f64(model.ctilde_complex)
    for el, idx, start, nms, nd in spec.functions:
        t["functions"][el][idx]["ctildes"] = ct[start:start + nms, :nd].reshape(-1).tolist()
    dump_tree(t, path)
