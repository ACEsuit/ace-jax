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


# ------------------------------------------------------------------ off-site
def _qr_update(R, A):
    """Triangular factor of [R; A] (R may be None): the streamed least-squares state."""
    return np.linalg.qr(A if R is None else np.concatenate([R, A]), mode="r")


def fit_offsite(bond_basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, chunk=2048):
    """An `OffsiteModel` fitted to the off-site blocks present in `datasets` (`OperatorData`s).

    Every bond within the bond cutoff whose block (R, i, j) is in the data is a sample; to fit
    to a subset, give datasets that hold only those blocks.  Bonds are processed in chunks and
    each problem (species pair, canonical shell pair, L) keeps only a triangular factor of
    [features | targets], so the design matrix is never held.  Returns (model, report) with
    report[(zi, zj, s1, s2, L)] = (rows, columns, rms residual per channel)."""
    from . import bond as B
    from .offsite import OffsiteModel, block_lps, canonical_pairs, site_features

    datasets = list(datasets)
    orbitals = datasets[0].basis if orbitals is None else orbitals
    lps = block_lps(orbitals)
    nc = len(channels)
    state, rows = {}, {}
    for d in datasets:
        if d.basis != orbitals:
            raise ValueError("fit_offsite: the datasets have different orbital bases")
        F = site_features(bond_basis, d.positions, d.cell, d.numbers)
        g = B.bond_graph(bond_basis, d.positions, d.cell, d.numbers)
        env = B.env_lists(bond_basis, d.positions, d.cell)
        keys = [(int(r[0]), int(r[1]), int(r[2]), int(a), int(b)) for r, a, b in zip(g.R, g.i, g.j)]
        have = np.asarray([all(k in d.blocks[c] for c in channels) for k in keys])
        numbers = np.asarray(d.numbers)
        for za in orbitals.species:
            for zb in orbitals.species:
                pairs = canonical_pairs(orbitals, za, zb)
                sel_all = np.flatnonzero(have & (numbers[g.i] == za) & (numbers[g.j] == zb)) if pairs else []
                for lo in range(0, len(sel_all), chunk):
                    sel = sel_all[lo:lo + chunk]
                    f = B.features(bond_basis, F, g, sel, lps, env=env)
                    blocks = np.stack([[d.blocks[c][keys[e]] for c in channels] for e in sel])   # (e, c, ni, nj)
                    for s1, s2, l1, l2, same in pairs:
                        p = 1 if (l1 + l2) % 2 == 0 else -1
                        sub = blocks[:, :, orbitals.shell_slice(za, s1), orbitals.shell_slice(zb, s2)]
                        for L in cg.Ls(l1, l2):
                            A = B.design(f[(L, p)], L, same)                                     # (e, k, M)
                            A = np.swapaxes(A, 1, 2).reshape(len(sel) * (2 * L + 1), -1)
                            Y = np.einsum("ecij,ijm->emc", sub, cg.real_cg(l1, l2, L)).reshape(len(sel) * (2 * L + 1), nc)
                            key = (za, zb, s1, s2, L)
                            state[key] = _qr_update(state.get(key), np.concatenate([A, Y], axis=1))
                            rows[key] = rows.get(key, 0) + len(A)
    coef, report = {}, {}
    for key, R in state.items():
        k = R.shape[1] - nc
        if k == 0:
            report[key] = (rows[key], 0, np.sqrt((R ** 2).sum(0) / rows[key]))
            continue
        R = np.concatenate([R, np.zeros((max(0, k + nc - R.shape[0]), k + nc))])                 # fewer rows than columns
        Rk, Ck = R[:k, :k], R[:k, k:]
        c = _solve(Rk, Ck, ridge)                       # argmin |Rk c - Ck|^2 + ridge |c|^2: the same minimiser
        coef[key] = c.T.copy()
        res2 = ((Rk @ c - Ck) ** 2).sum(0) + (R[k:, k:] ** 2).sum(0)
        report[key] = (rows[key], k, np.sqrt(res2 / rows[key]))
    return OffsiteModel(bond=bond_basis, coef=coef, orbitals=orbitals, channels=tuple(channels)), report
