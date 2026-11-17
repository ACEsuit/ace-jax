"""Stage-A block fit: least squares on the L parts of the blocks, on `ace_jax.fit.linear`.

The real CG stack is orthogonal, so projecting a target block on L splits the fit into
independent regressions.  The design rows of an L part depend only on the species (pair), on
(L, parity) and, off-site, on whether the two shells are the same shell of one species; every
shell pair with that part and every operator channel is a right-hand side of the same rows.  One
"design" is therefore one quantity of the generic row interface (`fit.linear.stats`): its rows
are streamed into a triangular factor R with c = Q^T Y for all its right-hand sides
(`host_qr_stream`, LAPACK tpqrt), and the coefficients are the posterior mean of the
diagonal-prior model (`fit.linear.objective.posterior_qr`) with prior precision `ridge` and unit
noise, i.e. argmin |A c - Y|^2 + ridge |c|^2.  The factors are `fit.linear.stats.Factor`s, so
the evidence and the UQ of `ace_jax.uq` read them as they read a potential's."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..basis import cg
from ..fit.linear import objective as lo
from ..fit.linear import stats as ls
from .onsite import OnsiteModel, _features, design, parts_of, shell_pairs, site_graph


class DesignStats(NamedTuple):
    """The statistics of one design: `factor` (a `fit.linear.stats.Factor` with one right-hand side
    per (coefficient key, channel)) and `keys`, the coefficient keys in the order of its
    right-hand sides (channels innermost)."""
    factor: ls.Factor
    keys: tuple


def _need_x64():
    if not jax.config.jax_enable_x64:
        raise ValueError("the block fit needs float64: jax.config.update('jax_enable_x64', True)")


def _fit(designs, n_chunks, rows, nc, ridge):
    """Stream and solve.  designs: {design key: (n columns, [coefficient keys])}; rows(i) ->
    {design key: (A (n_rows, n_columns), Y (n_rows, n_keys * nc))} for chunk i (a design without
    rows in the chunk is left out).  Returns (coef, report, stats)."""
    live = [d for d, (k, _) in designs.items() if k > 0]
    yy = {d: np.zeros(len(keys) * nc) for d, (_, keys) in designs.items()}
    n = dict.fromkeys(designs, 0)

    def parts(i):
        out = rows(int(i))
        for d, (_A, Y) in out.items():
            yy[d] += (Y ** 2).sum(0)
            n[d] += len(Y)
        return [out.get(d, (np.zeros((0, designs[d][0])), np.zeros((0, len(designs[d][1]) * nc)))) for d in live]

    init = [(np.zeros((designs[d][0],) * 2), np.zeros((designs[d][0], len(designs[d][1]) * nc))) for d in live]
    acc = dict(zip(live, ls.host_qr_stream(parts, np.arange(n_chunks), init))) if live else {}
    for d in designs:                                    # designs without columns: only their scalars
        if d not in acc:
            parts_d = [rows(i).get(d) for i in range(n_chunks)] if not live else []
            for p in parts_d:
                if p is not None:
                    yy[d] += (p[1] ** 2).sum(0); n[d] += len(p[1])
    coef, report, stats = {}, {}, {}
    for d, (k, keys) in designs.items():
        if not n[d]:
            continue
        if k == 0:
            for q, key in enumerate(keys):
                report[key] = (n[d], 0, np.sqrt(yy[d][q * nc:(q + 1) * nc] / n[d]))
            continue
        R, c = acc[d]
        fac = ls.Factor(R, c, jnp.asarray(yy[d]), jnp.asarray(float(n[d])), jnp.zeros(()))
        mu = np.asarray(lo.posterior_qr(jnp.full(k, ridge), [fac], [0.0])[0])            # (k, n_keys * nc)
        res2 = ((np.asarray(R) @ mu - np.asarray(c)) ** 2).sum(0) + np.maximum(yy[d] - (np.asarray(c) ** 2).sum(0), 0.0)
        for q, key in enumerate(keys):
            coef[key] = mu[:, q * nc:(q + 1) * nc].T.copy()
            report[key] = (n[d], k, np.sqrt(res2[q * nc:(q + 1) * nc] / n[d]))
        stats[d] = DesignStats(fac, tuple(keys))
    return coef, report, stats


def fit_onsite(basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, atoms=None, stats=False):
    """An `OnsiteModel` fitted to the on-site blocks of `datasets` (`OperatorData`s).

    Returns (model, report): report[(z, s1, s2, L)] = (rows, columns, rms residual per channel).
    A part with no feature of its parity and L > 0 has no coefficients (predicted zero); its
    report entry has 0 columns and the rms of the target itself.  `atoms`: per dataset, the atom
    indices to fit to (default: every atom); the others still act as neighbours.
    `stats=True`: also {(z, L, p): `DesignStats`}, the QR statistics of each design."""
    _need_x64()
    datasets = list(datasets)
    orbitals = datasets[0].basis if orbitals is None else orbitals
    if any(d.basis != orbitals for d in datasets):
        raise ValueError("fit_onsite: the datasets have different orbital bases")
    nc = len(channels)
    designs, members = {}, {}                           # (z, L, p) -> (columns, keys); -> [(s1, s2, l1, l2)]
    for z in orbitals.species:
        for s1, s2, l1, l2 in shell_pairs(orbitals, z):
            for L, p in parts_of(l1, l2, s1 == s2):
                designs.setdefault((z, L, p), (basis.n(L, p) + (L == 0), []))[1].append((z, s1, s2, L))
                members.setdefault((z, L, p), []).append((s1, s2, l1, l2))

    def rows(k):
        d = datasets[k]
        rij, zi, zj, seg, n = site_graph(basis, d.positions, d.cell, d.numbers)
        F = {key: np.asarray(v) for key, v in _features(basis, rij, zi, zj, seg, n).items()}
        out = {}
        for z in orbitals.species:
            idx = np.flatnonzero(np.asarray(d.numbers) == z)
            if atoms is not None:
                idx = np.intersect1d(idx, np.asarray(atoms[k], int))
            if not len(idx):
                continue
            blocks = np.stack([[d.blocks[c][(0, 0, 0, int(i), int(i))] for c in channels] for i in idx])    # (a, c, n, n)
            for (zz, L, p), mem in members.items():
                if zz != z:
                    continue
                A = design(F.get((L, p)), idx, L).reshape(len(idx) * (2 * L + 1), -1)
                Y = [np.einsum("acij,ijm->amc", blocks[:, :, orbitals.shell_slice(z, s1), orbitals.shell_slice(z, s2)],
                               cg.real_cg(l1, l2, L)).reshape(len(idx) * (2 * L + 1), nc) for s1, s2, l1, l2 in mem]
                out[(z, L, p)] = (A, np.concatenate(Y, axis=1))
        return out

    coef, report, st = _fit(designs, len(datasets), rows, nc, ridge)
    model = OnsiteModel(basis=basis, coef=coef, orbitals=orbitals, channels=tuple(channels))
    return (model, report, st) if stats else (model, report)


def fit_offsite(bond_basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, chunk=2048, stats=False):
    """An `OffsiteModel` fitted to the off-site blocks present in `datasets` (`OperatorData`s).

    Every bond within the bond cutoff whose block (R, i, j) is in the data is a sample; to fit
    to a subset, give datasets that hold only those blocks.  Bonds are processed in chunks and
    each design keeps only its triangular factor, so the design matrix is never held.  Returns
    (model, report) with report[(zi, zj, s1, s2, L)] = (rows, columns, rms residual per channel).
    `stats=True`: also {(zi, zj, L, p, same shell): `DesignStats`}."""
    from . import bond as B
    from .offsite import OffsiteModel, block_lps, canonical_pairs, site_features

    _need_x64()
    datasets = list(datasets)
    orbitals = datasets[0].basis if orbitals is None else orbitals
    if any(d.basis != orbitals for d in datasets):
        raise ValueError("fit_offsite: the datasets have different orbital bases")
    lps = block_lps(orbitals)
    nc = len(channels)
    designs, members = {}, {}                           # (za, zb, L, p, same) -> ...; -> [(s1, s2, l1, l2)]
    for za in orbitals.species:
        for zb in orbitals.species:
            for s1, s2, l1, l2, same in canonical_pairs(orbitals, za, zb):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                for L in cg.Ls(l1, l2):
                    k0, k1 = B.n_features(bond_basis, L, p)
                    k = k0 + (k1 if same else 2 * k1) + B.n_env_features(bond_basis, L, p, same=same)
                    designs.setdefault((za, zb, L, p, same), (k, []))[1].append((za, zb, s1, s2, L))
                    members.setdefault((za, zb, L, p, same), []).append((s1, s2, l1, l2))
    # the chunks: (dataset, species pair, bonds); the graph of each dataset is cheap, its features are not
    graphs, chunks = [], []
    for k, d in enumerate(datasets):
        g = B.bond_graph(bond_basis, d.positions, d.cell, d.numbers)
        keys = [(int(r[0]), int(r[1]), int(r[2]), int(a), int(b)) for r, a, b in zip(g.R, g.i, g.j)]
        have = np.asarray([all(key in d.blocks[c] for c in channels) for key in keys], bool)
        numbers = np.asarray(d.numbers)
        graphs.append((g, keys))
        for za in orbitals.species:
            for zb in orbitals.species:
                if not canonical_pairs(orbitals, za, zb):
                    continue
                sel = np.flatnonzero(have & (numbers[g.i] == za) & (numbers[g.j] == zb)) if len(keys) else []
                chunks += [(k, za, zb, sel[lo:lo + chunk]) for lo in range(0, len(sel), chunk)]
    cache = {}

    def rows(i):
        k, za, zb, sel = chunks[i]
        d, (g, keys) = datasets[k], graphs[k]
        if cache.get("k") != k:                          # chunks come dataset by dataset
            cache.update(k=k, F=site_features(bond_basis, d.positions, d.cell, d.numbers),
                         env=B.env_lists(bond_basis, d.positions, d.cell, numbers=d.numbers))
        f = B.features(bond_basis, cache["F"], g, sel, lps, env=cache["env"])
        blocks = np.stack([[d.blocks[c][keys[e]] for c in channels] for e in sel])               # (e, c, ni, nj)
        out = {}
        for (a, b, L, p, same), mem in members.items():
            if (a, b) != (za, zb):
                continue
            A = np.swapaxes(B.design(f[(L, p)], L, same), 1, 2).reshape(len(sel) * (2 * L + 1), -1)
            Y = [np.einsum("ecij,ijm->emc", blocks[:, :, orbitals.shell_slice(za, s1), orbitals.shell_slice(zb, s2)],
                           cg.real_cg(l1, l2, L)).reshape(len(sel) * (2 * L + 1), nc) for s1, s2, l1, l2 in mem]
            out[(a, b, L, p, same)] = (A, np.concatenate(Y, axis=1))
        return out

    coef, report, st = _fit(designs, len(chunks), rows, nc, ridge)
    model = OffsiteModel(bond=bond_basis, coef=coef, orbitals=orbitals, channels=tuple(channels))
    return (model, report, st) if stats else (model, report)
