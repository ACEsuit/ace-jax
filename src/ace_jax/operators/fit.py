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
import os
import tempfile
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..basis import cg
from ..fit.linear import lml_svd
from ..fit.linear import loo
from ..fit.linear import multistart
from ..fit.linear import objective as lo
from ..fit.linear import stats as ls
from .maps import make_room
from .onsite import OnsiteModel, _features, design, parts_of, shell_pairs, site_graph


class DesignStats(NamedTuple):
    """The statistics of one design: `factor` (a `fit.linear.stats.Factor` with one right-hand side
    per (coefficient key, channel)) and `keys`, the coefficient keys in the order of its
    right-hand sides (channels innermost).  hyper: with sigma_c="evidence", "loo" or "press",
    (n_rhs, 4): for each right-hand side the noise sigma and prior scale sigma_c chosen, the
    criterion there, and the sigma_c the coefficients were solved with (`_evidence`, `_loo`)."""
    factor: ls.Factor
    keys: tuple
    hyper: object = None


class _Prior(NamedTuple):
    """What `fit.linear.lml_svd.svd_statistics` reads of a problem: the smoothness of the leading
    columns and the fixed precisions of the trailing ones."""
    gamma: np.ndarray
    e0_prec: np.ndarray


@jax.jit
def _evidence_vg(x, s2, chat, P, p0, yy, n):
    """Value and gradient in x = (log sigma, log sigma_c) of the evidence of one right-hand side
    (chat (:,), yy): `fit.linear.lml_svd.log_evidence_svd` on the design's SVD."""
    return jax.value_and_grad(lambda x: lml_svd.log_evidence_svd(x[0], x[1], lml_svd.SVDStats(s2, chat, P, p0, yy, n, jnp.zeros(()))))(x)


RHO_STEP = 0.1          # decades: right-hand sides whose sigma^2 / sigma_c^2 agree to this share one posterior solve


def _evidence(fac, gamma, ridge, nc):
    """The evidence fit of one design.  Every right-hand side (a shell pair and a channel: their
    scales differ by orders of magnitude) has its own noise sigma and smoothness-prior scale
    sigma_c (coefficient j ~ N(0, sigma_c^2 / gamma_j^2); columns with gamma_j = 0 keep the
    precision `ridge` at unit noise), the maximum of its evidence: the potential fit's SVD
    evidence (`fit.linear.lml_svd`: one SVD of R D^-1 for the design, O(k) per evaluation) by
    bounded L-BFGS from three starts (`fit.linear.multistart`).  The posterior mean depends on
    rho = sigma^2 / sigma_c^2 alone, so right-hand sides are solved together by rho rounded to
    RHO_STEP decades (`fit.linear.objective.posterior_qr`).  Returns (mean (k, n_rhs) in the
    design's column order, hyper (n_rhs, 4): sigma, sigma_c at the maximum, the log evidence
    there, and the sigma_c used after rounding).

    The solve adds `ridge` to the prior precision of every column, as the fit with a given
    sigma_c does.  A design with fewer rows than columns (a species pair with few bonds) can be
    interpolated, the evidence then takes sigma to its lower bound, and gamma^2 sigma^2 /
    sigma_c^2 alone is orders of magnitude below the ridge: the coefficients of nearly dependent
    columns reached 1e13 (Ti-Ti in NbFeSb; docs/dev/ham-results.md)."""
    gamma = np.asarray(gamma, float)
    free = np.flatnonzero(gamma > 0)
    order = np.concatenate([free, np.flatnonzero(gamma <= 0)])            # smoothness columns first, fixed ones last
    R, c = np.asarray(fac.R, float)[:, order], np.asarray(fac.c, float)
    yy, n = np.asarray(fac.yy, float), float(fac.n)
    prior = _Prior(gamma[free], np.full(len(order) - len(free), ridge))
    sv = lml_svd.svd_statistics([ls.Factor(R, c, jnp.asarray(yy.sum()), fac.n, jnp.zeros(()))], prior)
    hyper = np.zeros((c.shape[1], 4))
    for q in range(c.shape[1]):
        scale = float(np.sqrt(max(yy[q] / n, 1e-300)))                    # rms of this target
        lo_, hi_ = np.log([1e-9 * scale, 1e-3 * scale]), np.log([1e2 * scale, 1e15 * scale])
        vg = lambda x, q=q: _evidence_vg(jnp.asarray(x), sv.s2, sv.chat[:, q], sv.P, sv.p0, jnp.asarray(yy[q]), sv.n)
        best, _ = multistart.multistart_map(vg, [np.log([1e-2 * scale, v * scale]) for v in (1e2, 1e5, 1e8)], lo_, hi_, maxiter=200)
        hyper[q, :3] = np.exp(best["x"][0]), np.exp(best["x"][1]), best["value"]
    rho = np.round(2.0 * np.log10(hyper[:, 0] / hyper[:, 1]) / RHO_STEP) * RHO_STEP          # log10 sigma^2 / sigma_c^2
    hyper[:, 3] = hyper[:, 0] / 10.0 ** (0.5 * rho)
    mean = np.zeros_like(c)
    for r in np.unique(rho):
        q = np.flatnonzero(rho == r)
        lin = jnp.asarray(np.concatenate([ridge + prior.gamma ** 2 * 10.0 ** r, prior.e0_prec]))     # never weaker than the plain ridge
        mu = lo.posterior_qr(lin, [ls.Factor(jnp.asarray(R), jnp.asarray(c[:, q]), jnp.asarray(yy[q]), fac.n, jnp.zeros(()))], [0.0])[0]
        mean[np.ix_(order, q)] = np.asarray(mu)
    return mean, hyper


SELECT = ("evidence", "loo", "press")                  # sigma_c= by a criterion rather than a number
LOO_RHO = np.arange(-24.0, 6.01, 0.5)                  # log10 sigma^2 / sigma_c^2 tried by "loo" and "press"


def _loo(fac, gamma, ridge, blocks, press):
    """The leave-one-structure-out fit of one design.  blocks: per structure (R_g, c_g, e_g), the
    triangular factor of its rows, Q_g^T Y and the squared norm of the part of each target
    outside its columns.  For every rho = sigma^2 / sigma_c^2 of LOO_RHO (prior precision
    gamma_j^2 rho at unit noise; `ridge` where gamma_j = 0) the structures are left out one at a
    time in closed form (`fit.linear.loo.loo_terms`: the block leverages of `aj fit
    --objective loo`, the residuals of the CR3 jackknife), and each right-hand side takes the
    rho of its largest criterion: the log predictive density of the left-out structures at
    the noise that maximises it, or (`press`) minus their squared prediction error.  Residuals
    shared by the atoms of a structure count once per structure, where the evidence reads them
    as independent noise.  A rho at which a structure has leverage 1 within roundoff is skipped.
    The solve adds `ridge` to every column (see `_evidence`); the rho values whose prior is below
    it give the same fit, and of rho values that tie the largest is taken.
    Returns (mean, hyper) as `_evidence` (hyper[:, 2]: the criterion)."""
    gamma, n = np.asarray(gamma, float), float(fac.n)
    blocks = [_Spill.get(b) for b in blocks]
    m = max(len(b[0]) for b in blocks)                   # a structure with fewer rows than columns keeps only those rows
    P, Y = (jnp.asarray(np.stack([np.concatenate([b[i], np.zeros((m - len(b[i]), b[i].shape[1]))]) for b in blocks])) for i in (0, 1))
    ee = np.sum([b[2] for b in blocks], axis=0)
    f = ls.Factor(jnp.asarray(fac.R), jnp.asarray(fac.c), jnp.asarray(fac.yy), fac.n, jnp.zeros(()))
    crit, s2, means = [], [], []
    for r in LOO_RHO:
        quad, hld, pr, mean = (np.asarray(v) for v in loo.loo_terms(jnp.asarray(ridge + gamma ** 2 * 10.0 ** r), f, P, Y))
        s2.append((quad + ee) / n); means.append(mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            crit.append(-(pr + ee) if press else -0.5 * n * np.log(s2[-1]) + hld)
    crit = np.where(np.isfinite(crit), crit, -np.inf)
    top = crit.max(0)                                    # ties (every rho whose prior is below the ridge gives one fit): the largest rho
    tied = crit >= top - 1e-9 * (1.0 + np.abs(np.where(np.isfinite(top), top, 0.0)))
    best, q = len(LOO_RHO) - 1 - tied[::-1].argmax(0), np.arange(crit.shape[1])
    sig = np.sqrt(np.asarray(s2)[best, q])
    sc = sig / 10.0 ** (0.5 * LOO_RHO[best])
    return np.asarray(means)[best, :, q].T, np.stack([sig, sc, crit[best, q], sc], axis=1)


class _Spill:
    """The per-structure factors of `_fit` on disk: over all the designs of an off-site model they
    are structures x columns^2 numbers each, tens of GB, and `_loo` reads one design at a time.
    In `$ACEJAX_SCRATCH` or the system's temporary directory; removed when the fit returns."""

    def __init__(self):
        self.dir, self.n = tempfile.TemporaryDirectory(prefix="acejax-loo-", dir=os.environ.get("ACEJAX_SCRATCH")), 0

    def put(self, R, c, e):
        path, self.n = os.path.join(self.dir.name, f"{self.n}.npz"), self.n + 1
        np.savez(path, R=R, c=c, e=e)
        return path

    @staticmethod
    def get(path):
        with np.load(path) as z:
            return z["R"], z["c"], z["e"]


def _live_rows(R, c):
    """The rows of a factor that carry anything: a structure with fewer rows than the design has
    columns leaves the others at zero (to roundoff), and `_loo` need not hold or solve them."""
    size = np.sqrt((R ** 2).sum(1) + (c ** 2).sum(1))
    keep = size > 1e-13 * max(size.max(), 1e-300)
    return R[keep], c[keep]


def _need_x64():
    if not jax.config.jax_enable_x64:
        raise ValueError("the block fit needs float64: jax.config.update('jax_enable_x64', True)")


def _fit(designs, n_chunks, rows, nc, ridge, device=False, gamma=None, select=None, groups=None):
    """Stream and solve.  designs: {design key: (n columns, [coefficient keys])}; rows(i) ->
    {design key: (A (n_rows, n_columns), Y (n_rows, n_keys * nc))} for chunk i (a design without
    rows in the chunk is left out).  Returns (coef, report, stats).

    device: rows(i) gives {design key: (A, Y, n_rows)} with A a JAX array on the device, padded
    with zero rows to a fixed size (n_rows: the real ones), and the triangular factors are
    merged there (`design.qr_merge`): nothing the size of the design rows comes to the host.

    groups (select "loo" or "press"): the structure of each chunk.  Each structure's rows are
    then merged into a factor of their own, kept on disk for `_loo` (`_Spill`: per design,
    structures x columns^2 numbers), and the factor of all the rows is the merge of those."""
    live = [d for d, (k, _) in designs.items() if k > 0]
    yy = {d: np.zeros(len(keys) * nc) for d, (_, keys) in designs.items()}
    n = dict.fromkeys(designs, 0)
    if device:
        from .design import qr_merge
        Rs, blocks, g0, spill = {}, {}, None, _Spill()

        def close():                                     # the finished structure's factors: to the host, and into the total
            for d in [d for gd, d in Rs if gd == g0 and gd is not None]:
                R, k = Rs.pop((g0, d)), designs[d][0]
                blocks.setdefault(d, []).append(spill.put(*_live_rows(np.asarray(R[:k, :k]), np.asarray(R[:k, k:])), np.asarray((R[k:, k:] ** 2).sum(0))))
                Rs[(None, d)] = jnp.linalg.qr(jnp.concatenate([Rs[(None, d)], R]), mode="r") if (None, d) in Rs else R
        for i in range(n_chunks):
            g = None if groups is None else groups[i]
            if g != g0:
                close(); g0 = g
            make_room(25_000)                            # the row and merge programs of a new shape (`operators.maps`)
            for d, (A, Y, rows_d) in rows(i).items():
                yy[d] += (Y ** 2).sum(0)
                n[d] += rows_d
                if d in live:
                    w = designs[d][0] + Y.shape[1]
                    Rs[(g, d)] = qr_merge(Rs.get((g, d), jnp.zeros((w, w))), A, jnp.asarray(Y))
        close()
        acc = {d: (R[:designs[d][0], :designs[d][0]], R[:designs[d][0], designs[d][0]:]) for (_, d), R in Rs.items()}
        return _solve(designs, acc, yy, n, nc, ridge, gamma, select, blocks)

    def parts(i):
        out = rows(int(i))
        for d, (_A, Y) in out.items():
            yy[d] += (Y ** 2).sum(0)
            n[d] += len(Y)
        return [out.get(d, (np.zeros((0, designs[d][0])), np.zeros((0, len(designs[d][1]) * nc)))) for d in live]

    init = [(np.zeros((designs[d][0],) * 2), np.zeros((designs[d][0], len(designs[d][1]) * nc))) for d in live]
    blocks, spill = {}, _Spill()
    if groups is not None and live:                      # a factor per structure, then their merge
        for g in np.unique(groups):
            y0 = {d: yy[d].copy() for d in live}
            for d, (R, c) in zip(live, ls.host_qr_stream(parts, np.arange(n_chunks), init, batches=np.flatnonzero(np.asarray(groups) == g))):
                R, c = _live_rows(np.asarray(R), np.asarray(c))
                blocks.setdefault(d, []).append(spill.put(R, c, np.maximum(yy[d] - y0[d] - (c ** 2).sum(0), 0.0)))
        acc = dict(zip(live, ls.host_qr_stream(lambda j: [_Spill.get(blocks[d][int(j)])[:2] for d in live],
                                               np.arange(len(blocks[live[0]])), init)))
    else:
        acc = dict(zip(live, ls.host_qr_stream(parts, np.arange(n_chunks), init))) if live else {}
    for d in designs:                                    # designs without columns: only their scalars
        if d not in acc:
            parts_d = [rows(i).get(d) for i in range(n_chunks)] if not live else []
            for p in parts_d:
                if p is not None:
                    yy[d] += (p[1] ** 2).sum(0); n[d] += len(p[1])
    return _solve(designs, acc, yy, n, nc, ridge, gamma, select, blocks)


def _solve(designs, acc, yy, n, nc, ridge, gamma=None, select=None, blocks=None):
    """Coefficients, report and statistics from the factors (R, c) of each design.  gamma:
    {design key: prior precision of each column} added to the ridge (the smoothness prior), or,
    with `select` (one of SELECT), {design key: smoothness of each column} for `_evidence` or,
    with the per-structure `blocks` of `_fit`, `_loo`."""
    coef, report, stats = {}, {}, {}
    for d, (k, keys) in designs.items():
        if not n[d]:
            continue
        if k == 0:
            for q, key in enumerate(keys):
                report[key] = (n[d], 0, np.sqrt(yy[d][q * nc:(q + 1) * nc] / n[d]))
            continue
        R, c = acc[d]
        make_room(40_000 if select else 25_000)          # every design shape is a program of its own (`operators.maps`)
        fac = ls.Factor(R, c, jnp.asarray(yy[d]), jnp.asarray(float(n[d])), jnp.zeros(()))
        hyper = None
        if select == "evidence":
            mu, hyper = _evidence(fac, gamma[d], ridge, nc)
        elif select:
            mu, hyper = _loo(fac, gamma[d], ridge, blocks[d], select == "press")
        else:
            lin = jnp.full(k, ridge) if gamma is None else ridge + jnp.asarray(gamma[d])
            mu = np.asarray(lo.posterior_qr(lin, [fac], [0.0])[0])                       # (k, n_keys * nc)
        res2 = ((np.asarray(R) @ mu - np.asarray(c)) ** 2).sum(0) + np.maximum(yy[d] - (np.asarray(c) ** 2).sum(0), 0.0)
        for q, key in enumerate(keys):
            coef[key] = mu[:, q * nc:(q + 1) * nc].T.copy()
            report[key] = (n[d], k, np.sqrt(res2[q * nc:(q + 1) * nc] / n[d]))
        stats[d] = DesignStats(fac, tuple(keys), hyper)
    return coef, report, stats


def fit_onsite(basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, atoms=None, stats=False, sigma_c=None):
    """An `OnsiteModel` fitted to the on-site blocks of `datasets` (`OperatorData`s).

    Returns (model, report): report[(z, s1, s2, L)] = (rows, columns, rms residual per channel).
    A part with no feature of its parity and L > 0 has no coefficients (predicted zero); its
    report entry has 0 columns and the rms of the target itself.  `atoms`: per dataset, the atom
    indices to fit to (default: every atom); the others still act as neighbours.
    `stats=True`: also {(z, L, p): `DesignStats`}, the QR statistics of each design.

    sigma_c: None, the plain ridge; a number: the algebraic smoothness prior of the potential
    fit (`basis.prior`), precision ridge + (gamma / sigma_c)^2 on the feature of smoothness gamma
    (`SiteBasis.smoothness`; unit noise, so sigma_c is in the units of the blocks).  The constant
    column of the L = 0 parts keeps the ridge alone.  "evidence": sigma_c and the noise of every
    shell pair and channel maximise its evidence (`_evidence`; the values are in the statistics'
    `hyper`).  "loo": sigma_c of every shell pair and channel maximises the predictive density of
    the structures left out one at a time (`_loo`; `aj fit --objective loo` for these rows);
    "press": it minimises their squared prediction error."""
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

    ev = isinstance(sigma_c, str)
    if ev and sigma_c not in SELECT:
        raise ValueError(f"sigma_c: None, a number or one of {SELECT}, got {sigma_c!r}")
    smooth = lambda L, p: np.concatenate([basis.smoothness(L, p), np.zeros(int(L == 0))])
    gamma = None if sigma_c is None else {(z, L, p): smooth(L, p) if ev else smooth(L, p) ** 2 / sigma_c ** 2 for z, L, p in designs}
    coef, report, st = _fit(designs, len(datasets), rows, nc, ridge, gamma=gamma, select=sigma_c if ev else None,
                            groups=np.arange(len(datasets)) if ev and sigma_c != "evidence" else None)
    model = OnsiteModel(basis=basis, coef=coef, orbitals=orbitals, channels=tuple(channels))
    return (model, report, st) if stats else (model, report)


def fit_offsite(bond_basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, chunk=2048, stats=False, compiled=True,
                sigma_c=None):
    """An `OffsiteModel` fitted to the off-site blocks present in `datasets` (`OperatorData`s).

    Every bond within the bond cutoff whose block (R, i, j) is in the data is a sample; to fit
    to a subset, give datasets that hold only those blocks.  Bonds are processed in chunks and
    each design keeps only its triangular factor, so the design matrix is never held.  Returns
    (model, report) with report[(zi, zj, s1, s2, L)] = (rows, columns, rms residual per channel).
    `stats=True`: also {(zi, zj, L, p, same shell): `DesignStats`}.

    compiled (default): the design rows come from `design.bond_designs`, one compiled program per
    species pair, and the factors are merged on the device JAX runs on (a GPU: feature assembly
    and QR there, in float64).  False: the NumPy reference `bond.features` with host merges (the
    same coefficients; tests compare the two).

    sigma_c: as `fit_onsite`, with `bond.smoothness` for the columns of each design.  "loo" and
    "press" hold a triangular factor per structure and design on the host: structures x
    columns^2 numbers each."""
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

    ev = isinstance(sigma_c, str)
    if ev and sigma_c not in SELECT:
        raise ValueError(f"sigma_c: None, a number or one of {SELECT}, got {sigma_c!r}")
    gamma = None if sigma_c is None else {
        key: B.smoothness(bond_basis, key[2], key[3], key[4]) ** (1 if ev else 2) / (1.0 if ev else sigma_c ** 2) for key in designs}
    sel = dict(select=sigma_c if ev else None, groups=[c[0] for c in chunks] if ev and sigma_c != "evidence" else None)
    if compiled:
        from .design import bond_designs
        progs, dev = {}, {}

        def rows_device(i):
            k, za, zb, sel = chunks[i]
            d, (g, keys) = datasets[k], graphs[k]
            if dev.get("k") != k:                        # this dataset's features and lists, on the device
                env = B.env_lists(bond_basis, d.positions, d.cell, numbers=d.numbers)
                dev.update(k=k, F=_features(bond_basis.site, *site_graph(bond_basis.site, d.positions, d.cell, d.numbers)),
                           env=None if env is None else tuple(jnp.asarray(x) for x in env))
            mine = [key for key in members if key[:2] == (za, zb)]
            if (za, zb) not in progs:
                progs[(za, zb)] = bond_designs(bond_basis, [key[2:] for key in mine])
            pad = np.concatenate([sel, np.full(chunk - len(sel), sel[0])])           # fixed size: one compilation
            mask = np.concatenate([np.ones(len(sel)), np.zeros(chunk - len(sel))])
            As = progs[(za, zb)](dev["F"], jnp.asarray(g.rij[pad]), jnp.asarray(g.i[pad]), jnp.asarray(g.j[pad]),
                                 jnp.asarray(mask), dev["env"])
            blocks = np.stack([[d.blocks[c][keys[e]] for c in channels] for e in sel])
            out = {}
            for key, A in zip(mine, As):
                L = key[2]
                Y = np.zeros((chunk * (2 * L + 1), len(members[key]) * nc))
                Y[:len(sel) * (2 * L + 1)] = np.concatenate(
                    [np.einsum("ecij,ijm->emc", blocks[:, :, orbitals.shell_slice(za, s1), orbitals.shell_slice(zb, s2)],
                               cg.real_cg(l1, l2, L)).reshape(len(sel) * (2 * L + 1), nc) for s1, s2, l1, l2 in members[key]], axis=1)
                out[key] = (A, Y, len(sel) * (2 * L + 1))
            return out

        coef, report, st = _fit(designs, len(chunks), rows_device, nc, ridge, device=True, gamma=gamma, **sel)
    else:
        coef, report, st = _fit(designs, len(chunks), rows, nc, ridge, gamma=gamma, **sel)
    model = OffsiteModel(bond=bond_basis, coef=coef, orbitals=orbitals, channels=tuple(channels))
    return (model, report, st) if stats else (model, report)
