"""Reader and writer for the DeepH layout (the exchange format Quoll.jl writes,
shared with DeepH-E3 and MACE-H).

A directory holds `hamiltonians.h5` / `overlaps.h5` (one dataset per atom pair
and cell, key "[Rx, Ry, Rz, i, j]" with 1-based atoms), `lat.dat` and
`site_positions.dat` (3 x 3 and 3 x N, vectors in COLUMNS, angstrom),
`element.dat`, `orbital_types.dat` (l of each shell, one line per atom) and
`info.json`.  Block (i, j, R) couples orbitals of atom i in the home cell with
those of atom j in cell R.

Orbital order.  DeepH orders the 2l+1 orbitals of a shell as `DEEPH_ORDER[l]`
(positions in m = -l..l; Quoll's `default_shconv(DeepHSource)`), with no sign
changes relative to ace-jax's real harmonics; `read` returns blocks in ace-jax's
order (m = -l..l) and `write` puts them back.  tests/test_ham_deeph.py checks
the order and signs on real data: every part of an overlap block must be a
harmonic of the bond direction.

Units are the file's (eV and angstrom in Quoll's output); nothing is converted."""
import json
import pathlib
from typing import NamedTuple

import numpy as np

from .orbitals import OrbitalBasis

# deeph_shell[k] = ace_shell[DEEPH_ORDER[l][k]], ace order m = -l..l
DEEPH_ORDER = {0: (0,), 1: (2, 0, 1), 2: (2, 4, 0, 3, 1), 3: (3, 4, 2, 5, 1, 6, 0)}
FILES = {"H": "hamiltonians.h5", "S": "overlaps.h5"}


class OperatorData(NamedTuple):
    """One structure with its operators.

    cell: (3, 3), lattice vectors in ROWS (ASE's layout); positions (N, 3);
    numbers (N,); basis: `OrbitalBasis`; blocks: {name: {(Rx, Ry, Rz, i, j): (n_i, n_j) array}},
    0-based atoms, ace-jax orbital order; names are operator channels ("H", "S", ...)."""

    cell: np.ndarray
    positions: np.ndarray
    numbers: np.ndarray
    basis: OrbitalBasis
    blocks: dict

    def bond(self, key):
        """Vector from atom i to atom j in cell R, for a block key."""
        Rx, Ry, Rz, i, j = key
        return self.positions[j] + np.array([Rx, Ry, Rz]) @ self.cell - self.positions[i]


def _atom_perm(ls):
    """p with deeph_block = ace_block[p] along one atom's orbitals."""
    out, o = [], 0
    for l in ls:
        if l not in DEEPH_ORDER:
            raise ValueError(f"DeepH layout: no orbital order defined for l = {l}")
        out += [o + k for k in DEEPH_ORDER[l]]
        o += 2 * l + 1
    return np.asarray(out, np.int64)


def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError("reading and writing DeepH files needs h5py: pip install 'ace-jax[ham]'") from e
    return h5py


def read(directory, operators=("H", "S")):
    """`OperatorData` from a DeepH-layout directory.  `operators`: channels to
    read, each a key of `FILES` or a (name, file name) pair."""
    h5py = _h5py()
    d = pathlib.Path(directory)
    cell = np.loadtxt(d / "lat.dat").reshape(3, 3).T.copy()
    positions = np.atleast_2d(np.loadtxt(d / "site_positions.dat")).reshape(3, -1).T.copy()
    numbers = np.atleast_1d(np.loadtxt(d / "element.dat", dtype=np.int64))
    shells = [tuple(int(x) for x in ln.split()) for ln in (d / "orbital_types.dat").read_text().splitlines() if ln.split()]
    if not len(shells) == len(numbers) == len(positions):
        raise ValueError(f"{d}: {len(numbers)} elements, {len(positions)} positions, {len(shells)} orbital lines")
    by_z = {}
    for z, ls in zip(numbers, shells):
        if by_z.setdefault(int(z), ls) != ls:
            raise ValueError(f"{d}: atoms of Z = {int(z)} have different orbital bases")
    basis = OrbitalBasis.of(by_z)
    inv = {z: np.argsort(_atom_perm(basis.ls(z))) for z in basis.species}      # ace = deeph[inv]
    blocks = {}
    for op in operators:
        name, fname = (op, FILES[op]) if isinstance(op, str) else op
        out = {}
        with h5py.File(d / fname, "r") as f:
            for key in f:
                Rx, Ry, Rz, i, j = (int(v) for v in json.loads(key))
                i, j = i - 1, j - 1
                b = np.asarray(f[key], np.float64)
                zi, zj = int(numbers[i]), int(numbers[j])
                if b.shape != (basis.n_orb(zi), basis.n_orb(zj)):
                    raise ValueError(f"{fname} {key}: block shape {b.shape} does not match the orbital bases")
                out[(Rx, Ry, Rz, i, j)] = b[inv[zi]][:, inv[zj]]
        blocks[name] = out
    return OperatorData(cell, positions, numbers, basis, blocks)


def read_spin(directory, overlap=True):
    """Collinear spin: `OperatorData` with the channels ("H_up", "H_dn", "S") from the two
    DeepH directories `<directory>/up` (H and S) and `<directory>/dn` (H) that
    `ham/julia/ingest.jl fhi-aims-spin` writes.  Both must hold one geometry and orbital basis."""
    d = pathlib.Path(directory)
    up = read(d / "up", operators=(("H_up", FILES["H"]),) + ((("S", FILES["S"]),) if overlap else ()))
    dn = read(d / "dn", operators=(("H_dn", FILES["H"]),))
    same = (np.array_equal(up.cell, dn.cell) and np.array_equal(up.positions, dn.positions)
            and np.array_equal(up.numbers, dn.numbers) and up.basis == dn.basis)
    if not same:
        raise ValueError(f"{d}: the up and dn directories hold different structures or orbital bases")
    if set(up.blocks["H_up"]) != set(dn.blocks["H_dn"]):
        raise ValueError(f"{d}: the up and dn Hamiltonians have different blocks")
    blocks = {"H_up": up.blocks["H_up"], "H_dn": dn.blocks["H_dn"]}
    if overlap:
        blocks["S"] = up.blocks["S"]
    return OperatorData(up.cell, up.positions, up.numbers, up.basis, blocks)


def write(directory, data, files=None, spinful=False):
    """Write `data` in the DeepH layout.  `files`: {channel: file name} (default
    `FILES` for the channels present; another name, e.g. "hamiltonians_pred.h5",
    for predictions)."""
    h5py = _h5py()
    d = pathlib.Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    files = {k: FILES[k] for k in data.blocks} if files is None else dict(files)
    perm = {z: _atom_perm(data.basis.ls(z)) for z in data.basis.species}       # deeph = ace[perm]
    np.savetxt(d / "lat.dat", np.asarray(data.cell).T, fmt="%.17g", delimiter="\t")
    np.savetxt(d / "rlat.dat", 2 * np.pi * np.linalg.inv(np.asarray(data.cell)), fmt="%.17g", delimiter="\t")
    np.savetxt(d / "site_positions.dat", np.asarray(data.positions).T, fmt="%.17g", delimiter="\t")
    np.savetxt(d / "element.dat", np.asarray(data.numbers), fmt="%d")
    (d / "orbital_types.dat").write_text(
        "".join("\t".join(str(l) for l in data.basis.ls(z)) + "\n" for z in data.numbers))
    cells = sorted({k[:3] for b in data.blocks.values() for k in b})
    np.savetxt(d / "R_list.dat", np.asarray(cells, np.int64).reshape(-1, 3), fmt="%d", delimiter="\t")
    (d / "info.json").write_text(json.dumps({"isspinful": bool(spinful)}))
    for name, fname in files.items():
        with h5py.File(d / fname, "w") as f:
            for (Rx, Ry, Rz, i, j), b in sorted(data.blocks[name].items()):
                zi, zj = int(data.numbers[i]), int(data.numbers[j])
                f.create_dataset(f"[{Rx}, {Ry}, {Rz}, {i + 1}, {j + 1}]",
                                 data=np.asarray(b, np.float64)[perm[zi]][:, perm[zj]])
