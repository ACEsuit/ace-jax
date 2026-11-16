"""Stage-A block fit: least squares on the L parts of the blocks.

The real CG stack is orthogonal, so projecting a target block on L splits the
fit into independent regressions per (species, shell pair, L), each with 2L+1
rows per atom, and every operator channel is a right-hand side of the same
design matrix: one factorisation serves H, S and the spin channels."""
import numpy as np

from . import cg
from .onsite import OnsiteModel, design, parts_of, shell_pairs, site_graph, _features


def _solve(A, Y, ridge):
    """argmin |A c - Y|^2 + ridge |c|^2 for each column of Y, by QR of [A; sqrt(ridge) I]."""
    n = A.shape[1]
    Q, R = np.linalg.qr(np.concatenate([A, np.sqrt(ridge) * np.eye(n)]))
    return np.linalg.solve(R, Q.T @ np.concatenate([Y, np.zeros((n, Y.shape[1]))]))


def fit_onsite(basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, atoms=None):
    """An `OnsiteModel` fitted to the on-site blocks of `datasets` (`OperatorData`s).

    Returns (model, report): report[(z, s1, s2, L)] = (rows, columns, rms residual per channel).
    A part with no feature of its parity and L > 0 has no coefficients (predicted zero); its
    report entry has 0 columns and the rms of the target itself.  `atoms`: per dataset, the atom
    indices to fit to (default: every atom); the others still act as neighbours."""
    datasets = list(datasets)
    orbitals = datasets[0].basis if orbitals is None else orbitals
    feats = []
    for d in datasets:
        if d.basis != orbitals:
            raise ValueError("fit_onsite: the datasets have different orbital bases")
        rij, zi, zj, seg, n = site_graph(basis, d.positions, d.cell, d.numbers)
        feats.append({k: np.asarray(v) for k, v in _features(basis, rij, zi, zj, seg, n).items()})
    coef, report = {}, {}
    for z in orbitals.species:
        where = [(k, np.flatnonzero(np.asarray(d.numbers) == z)) for k, d in enumerate(datasets)]
        if atoms is not None:
            where = [(k, np.intersect1d(a, np.asarray(atoms[k], int))) for k, a in where]
        if not any(len(a) for _, a in where):
            continue
        for s1, s2, l1, l2 in shell_pairs(orbitals, z):
            a, b = orbitals.shell_slice(z, s1), orbitals.shell_slice(z, s2)
            for L, p in parts_of(l1, l2, s1 == s2):
                T = cg.real_cg(l1, l2, L)
                rows, rhs = [], []
                for k, idx in where:
                    if not len(idx):
                        continue
                    d = datasets[k]
                    rows.append(design(feats[k].get((L, p)), idx, L).reshape(len(idx) * (2 * L + 1), -1))
                    blocks = np.stack([[d.blocks[c][(0, 0, 0, int(i), int(i))][a, b] for c in channels] for i in idx])
                    rhs.append(np.einsum("acij,ijm->amc", blocks, T).reshape(len(idx) * (2 * L + 1), len(channels)))
                A, Y = np.concatenate(rows), np.concatenate(rhs)
                if A.shape[1] == 0:
                    report[(z, s1, s2, L)] = (A.shape[0], 0, np.sqrt((Y ** 2).mean(0)))
                    continue
                c = _solve(A, Y, ridge)
                coef[(z, s1, s2, L)] = c.T.copy()
                report[(z, s1, s2, L)] = (A.shape[0], A.shape[1], np.sqrt(((A @ c - Y) ** 2).mean(0)))
    return OnsiteModel(basis=basis, coef=coef, orbitals=orbitals, channels=tuple(channels)), report
