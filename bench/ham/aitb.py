"""Reader for the "aitb" HDF5 files of the ACEhamiltonians paper data (arXiv:2111.13736,
Zenodo 5796204): one aluminium supercell per file from FHI-aims, with dense Gamma-point
H and S (one TB cell) of all 14 orbitals per atom.  H is in HARTREE in the files (the 1s level
is at -55.7 and the metadata's Fermi level, -6.33 eV, is -0.2324 in file units); `load` returns eV.

Orbital layout per atom, as FHI-aims writes it: shells sorted by l then n (Al: 1s 2s 3s 2p 3p
3d), m = -l..l within a shell, with FHI-aims' phases (`FHI_SIGNS`, Quoll.jl's
`default_shconv(FHIaimsSource)`: opposite sign for odd m > 0).  `load` returns blocks in
ace-jax's convention; `bench/ham/onsite_accuracy.py` checks it with the overlap's two-centre form."""
import h5py
import numpy as np

from ace_jax.operators.orbitals import OrbitalBasis

AL = OrbitalBasis.of({13: (0, 0, 0, 1, 1, 2)})
HARTREE = 27.211386245988          # eV
FHI_SIGNS = {0: (1,), 1: (1, 1, -1), 2: (1, 1, 1, -1, 1), 3: (1, 1, 1, 1, -1, 1, -1)}


def atom_signs(ls):
    return np.concatenate([FHI_SIGNS[l] for l in ls]).astype(float)


def load(path, operators=("H", "S"), raw=False):
    """dict: positions (N, 3), cell (3, 3 rows), numbers, and per operator the on-site blocks
    (N, 14, 14).  `raw=True` keeps FHI-aims' phases (what ACEhamiltonians.jl is fitted to) and
    also returns the full matrices under "<op>_full"."""
    with h5py.File(path, "r") as f:
        g = f["aitb"]
        out = {"positions": g["positions"][()], "cell": g["unitcell"][()], "numbers": g["species"][()].astype(int)}
        n = AL.n_orb(13)
        N = len(out["positions"])
        s = np.ones(n) if raw else atom_signs(AL.ls(13))
        for op in operators:
            M = g[op][0] * (HARTREE if op == "H" else 1.0)
            assert M.shape == (N * n, N * n)
            idx = np.arange(N)[:, None, None] * n + np.arange(n)
            out[op] = M[idx, idx.transpose(0, 2, 1)] * (s[:, None] * s[None, :])
            if raw:
                out[op + "_full"] = M
    return out
