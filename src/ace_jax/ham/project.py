"""Core-orbital projection: a valence-only H and S from all-electron data.

The construction of the MACE-H paper (arXiv:2508.15108, Methods, Core-Orbital Projection).  At
each k-point of a grid the orbitals are split into core (c) and valence (v), and the valence
functions are orthogonalised to the core ones,

    chi_v  ->  chi_v - chi_c S_cc^-1 S_cv,

which changes the matrices to

    S_vv  ->  S_vv - S_vc X,                 X = S_cc^-1 S_cv,
    H_vv  ->  H_vv - H_vc X - X^+ H_cv + X^+ H_cc X.

Where the core functions are close to eigenstates the core-valence block of the transformed H is
close to zero, so the valence block alone carries the valence spectrum.  The valence blocks are
then brought back to real space, B(R) = (1 / N_k) sum_k exp(-2 pi i k.R) M(k).

What it buys a model: blocks of the valence shells only (an all-electron basis has core blocks
with elements of thousands of eV, whose small relative errors dominate a band energy), and a
band energy that is the valence one by construction.  What it costs: the transformed valence
functions depend on the environment through S_cv, weakly where the core-valence overlap is small."""
import numpy as np

from ..operators.data import OperatorData
from ..operators.orbitals import OrbitalBasis
from .operator import BlockSum


def core_projection(data, core, nk=None, channels=("H", "S"), cutoff=None):
    """`OperatorData` of the valence shells with the core projected out.

    core: {Z: shell indices} of the core shells of each species (positions in `data.basis.ls(Z)`;
    e.g. {13: (0, 1, 3)} for 1s, 2s and 2p of an Al basis with shells s s s p p d).
    nk: the k-grid (n1, n2, n3) of the transform; default 2 max|R| + 1 along each lattice vector
    for the cells R in the data, the smallest grid on which the back transform of blocks with that
    pattern is exact (the projected blocks reach slightly further; `cutoff` keeps the pattern).
    channels: (Hamiltonian channel, overlap channel).  cutoff: keep blocks of pairs closer than
    this (default: the pairs the data has).  Blocks are real: the imaginary part of the back
    transform is checked to vanish."""
    hname, sname = channels
    basis, numbers = data.basis, np.asarray(data.numbers)
    keys = list(data.blocks[hname])
    R = np.array([k[:3] for k in keys])
    nk = tuple(int(2 * np.abs(R[:, d]).max() + 1) for d in range(3)) if nk is None else tuple(int(v) for v in nk)
    ks = np.array([[a / nk[0], b / nk[1], c / nk[2]] for a in range(nk[0]) for b in range(nk[1]) for c in range(nk[2])])
    bs = BlockSum(keys, numbers, basis)
    H, S = np.asarray(bs.matrices(data.blocks[hname], ks)), np.asarray(bs.matrices(data.blocks[sname], ks))
    # orbital indices of the core and valence shells, and the valence basis
    off = np.concatenate([[0], np.cumsum([basis.n_orb(int(z)) for z in numbers])])
    c_idx, v_idx, v_off = [], [], [0]
    val_ls = {}
    for a, z in enumerate(numbers):
        ls = basis.ls(int(z))
        cz = set(core.get(int(z), ()))
        if not cz <= set(range(len(ls))):
            raise ValueError(f"core shells {sorted(cz)} of Z = {int(z)} are not shells of its basis {ls}")
        val_ls[int(z)] = tuple(l for s, l in enumerate(ls) if s not in cz)
        nv = 0
        for s in range(len(ls)):
            sl = basis.shell_slice(int(z), s)
            idx = off[a] + np.arange(sl.start, sl.stop)
            (c_idx if s in cz else v_idx).extend(idx)
            nv += 0 if s in cz else len(idx)
        v_off.append(v_off[-1] + nv)
    c_idx, v_idx = np.asarray(c_idx, int), np.asarray(v_idx, int)
    vbasis = OrbitalBasis.of(val_ls)
    if len(c_idx) == 0:
        raise ValueError("no core shell given")
    Hv, Sv = np.empty((len(ks), len(v_idx), len(v_idx)), complex), np.empty((len(ks), len(v_idx), len(v_idx)), complex)
    leak = 0.0
    for q in range(len(ks)):
        Scc, Scv, Svv = S[q][np.ix_(c_idx, c_idx)], S[q][np.ix_(c_idx, v_idx)], S[q][np.ix_(v_idx, v_idx)]
        Hcc, Hcv, Hvv = H[q][np.ix_(c_idx, c_idx)], H[q][np.ix_(c_idx, v_idx)], H[q][np.ix_(v_idx, v_idx)]
        X = np.linalg.solve(Scc, Scv)
        Sv[q] = Svv - Scv.conj().T @ X
        Hv[q] = Hvv - Hcv.conj().T @ X - X.conj().T @ Hcv + X.conj().T @ Hcc @ X
        leak = max(leak, float(np.abs(Hcv - Hcc @ X).max()))            # the core-valence block that is dropped
    # back to real space on the pattern of the data (or within the cutoff)
    out = {hname: {}, sname: {}}
    imag = 0.0
    cells = {}
    for key in keys:
        if cutoff is None or np.linalg.norm(data.bond(key)) < cutoff:
            cells.setdefault(key[:3], []).append(key)
    for Rc, ks_R in cells.items():
        ph = np.exp(-2j * np.pi * (ks @ np.asarray(Rc))) / len(ks)
        for name, M in ((hname, Hv), (sname, Sv)):
            B = np.einsum("k,kij->ij", ph, M)
            for key in ks_R:
                i, j = key[3], key[4]
                blk = B[v_off[i]:v_off[i + 1], v_off[j]:v_off[j + 1]]
                imag = max(imag, float(np.abs(blk.imag).max()))
                out[name][key] = np.ascontiguousarray(blk.real)
    if imag > 1e-8 * max(1.0, float(np.abs(Hv).max())):
        raise ValueError(f"the back transform is not real (largest imaginary part {imag:.2e}): the k-grid {nk} is too coarse")
    proj = OperatorData(data.cell, data.positions, data.numbers, vbasis, out)
    return proj, {"nk": nk, "core_valence_coupling": leak, "n_core": len(c_idx), "n_valence": len(v_idx)}
