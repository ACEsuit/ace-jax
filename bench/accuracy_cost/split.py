"""GAP-18 Si data -> DATA/{train,test}.xyz: labels renamed to energy/forces/virial
(the source mixes dft_* and DFT_* keys), stored GAP predictions dropped, virials
kept where present (fitted by the codes that can), 90/10 stratified by config_type with rare types
kept whole in train (one test frame of a type is noise, not a measurement)."""
import hashlib
import json
import sys
import urllib.request

import numpy as np

from .common import DATA

URL = ("https://raw.githubusercontent.com/libAtoms/silicon-testing-framework/"
       "ec91b17fbbcd6c4a06a257acbfcba77734c0080c/models/GAP/")
SRC, XML, GLUE = "gp_iter6_sparse9k.xml.xyz", "gp_iter6_sparse9k.xml", "glue.xml"
SHA = {SRC: "74ac5f387aa0149ec9ea8f3e275d1087d7287db68c15ba8bc4c28a450de67e2a"}
LABEL_KEYS = {"energy": "dft_energy", "forces": "dft_force", "virial": "dft_virial"}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name in (SRC, XML, GLUE):
        p = DATA / name
        if not p.exists():
            urllib.request.urlretrieve(URL + name, p)
        if name in SHA and hashlib.sha256(p.read_bytes()).hexdigest() != SHA[name]:
            raise RuntimeError(f"{p}: sha256 does not match the pinned commit")


def _get(d, key):
    for k, v in d.items():
        if k.lower() == key:
            return v
    return None


def normalise(f):
    """An ase.Atoms with info energy/config_type and arrays forces, or None when unlabelled."""
    import ase
    E, F = _get(f.info, LABEL_KEYS["energy"]), _get(f.arrays, LABEL_KEYS["forces"])
    if E is None or F is None:
        return None
    at = ase.Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
    at.info.update(energy=float(E), config_type=str(f.info.get("config_type", "default")))
    V = _get(f.info, LABEL_KEYS["virial"])
    if V is not None:
        at.info["virial"] = np.asarray(V, float).reshape(3, 3)
    at.arrays["forces"] = np.asarray(F, float).reshape(-1, 3)
    return at


def stratified(types, frac=0.1, seed=0, min_per_type=10):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for t in sorted(set(types)):
        idx = [i for i, x in enumerate(types) if x == t]
        if len(idx) < min_per_type:
            tr += idx; continue
        idx = list(rng.permutation(idx))
        k = max(1, round(frac * len(idx)))
        te += idx[:k]; tr += idx[k:]
    return sorted(tr), sorted(te)


def _write(path, atoms):
    # plain extxyz, written by hand so no SinglePointCalculator round-trip touches the keys
    with open(path, "w") as f:
        for at in atoms:
            c = at.cell.array.reshape(-1)
            v = at.info.get("virial")
            vs = ' virial="' + " ".join(f"{x:.10f}" for x in np.asarray(v).ravel(order="F")) + '"' if v is not None else ""  # column-major: extxyz reads a 9-vector as Fortran-order 3x3
            f.write(f"{len(at)}\n")
            f.write('Lattice="' + " ".join(f"{x:.10f}" for x in c) + '" Properties=species:S:1:pos:R:3:forces:R:3 '
                    f'energy={at.info["energy"]:.10f}{vs} config_type={at.info["config_type"]} pbc="T T T"\n')
            for s, p, F in zip(at.get_chemical_symbols(), at.positions, at.arrays["forces"]):
                f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {F[0]:.10f} {F[1]:.10f} {F[2]:.10f}\n")


def split_hash(train, test):
    h = hashlib.sha256()
    for p in (train, test):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def _read_plain(path):
    """Reader for exactly what `_write` emits, for the pacemaker venv (python 3.9: no
    ace_jax, no libAtoms extxyz wheel).  Never ase.io.read: it would hide the labels."""
    import shlex

    import ase
    L, out, i = open(path).read().splitlines(), [], 0
    while i < len(L):
        n = int(L[i]); kv = dict(t.split("=", 1) for t in shlex.split(L[i + 1]) if "=" in t)
        rows = [l.split() for l in L[i + 2:i + 2 + n]]
        at = ase.Atoms(symbols=[r[0] for r in rows], positions=np.array([r[1:4] for r in rows], float),
                       cell=np.array(kv["Lattice"].split(), float).reshape(3, 3), pbc=True)
        at.info.update(energy=float(kv["energy"]), config_type=kv["config_type"])
        if "virial" in kv:
            at.info["virial"] = np.array(kv["virial"].split(), float).reshape(3, 3, order="F")
        at.arrays["forces"] = np.array([r[4:7] for r in rows], float)
        out.append(at); i += 2 + n
    return out


def read_atoms(path):
    """ase.Atoms with labels in info/arrays, via libAtoms extxyz (never ase.io.read)."""
    try:
        from ace_jax.fit.xyz import read_extxyz
    except ImportError:
        return _read_plain(path)
    import ase
    out = []
    for f in read_extxyz(path):
        at = ase.Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
        at.info.update(f.info); at.arrays.update({k: np.asarray(v) for k, v in f.arrays.items()})
        out.append(at)
    return out


def ensure_split():
    tr_p, te_p, meta_p = DATA / "train.xyz", DATA / "test.xyz", DATA / "split.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())["hash"]
    fetch()
    from ace_jax.fit.xyz import read_extxyz
    frames = read_extxyz(DATA / SRC)
    atoms = [a for a in map(normalise, frames) if a is not None]
    if len(atoms) != len(frames):
        raise RuntimeError(f"{len(frames) - len(atoms)} frames have no energy/force labels")
    assert all(a.pbc.all() for a in atoms)        # _write states pbc="T T T"
    tr, te = stratified([a.info["config_type"] for a in atoms])
    _write(tr_p, [atoms[i] for i in tr]); _write(te_p, [atoms[i] for i in te])
    h = split_hash(tr_p, te_p)
    meta_p.write_text(json.dumps({"hash": h, "n_train": len(tr), "n_test": len(te), "seed": 0,
                                  "frac": 0.1, "min_per_type": 10}, indent=1))
    return h


if __name__ == "__main__":
    print(ensure_split(), file=sys.stdout)
