"""PACE `.yace` (ACECTildeBasisSet YAML) reading and writing.

Keys and defaults follow ML-PACE `ace_c_basis.cpp::load_yaml` and
`ACERadialFunctions::init` (lammps-user-pace @ 99aa6e6).  Bond keys are flow
sequences (`[0, 1]:`), which PyYAML cannot hash, so they load as tuples.
"""
import copy
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


_BOND_DEFAULTS = {"rcut_in": 0.0, "dcut_in": 1e-5, "prehc": 0.0, "lambdahc": 1.0,
                  "inner_cutoff_type": "density"}


def parse_yace(path):
    tree = load_tree(path)
    names = list(tree["elements"])
    NZ = len(names)
    try:
        Z = np.array([atomic_numbers[e] for e in names], np.int64)
    except KeyError as e:
        raise ValueError(f"unknown element {e.args[0]!r} in {path}") from None

    bonds = {k: {**_BOND_DEFAULTS, **v} for k, v in tree["bonds"].items()}
    kinds = {b["radbasename"] for b in bonds.values()}
    if any(k.startswith("ACE.jl") for k in kinds):
        raise NotImplementedError(f"{path}: ACE.jl tabulated radials are not supported")
    if len(kinds) != 1:
        raise NotImplementedError(f"{path}: per-bond differing radbasename {sorted(kinds)}")
    inner = {b["inner_cutoff_type"] for b in bonds.values()}
    if len(inner) != 1:
        raise ValueError(f"{path}: bonds disagree on inner_cutoff_type {sorted(inner)}; "
                         "ML-PACE keeps one global value (last bond wins), so the file is ill-defined")
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
