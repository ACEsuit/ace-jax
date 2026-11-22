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
import scipy.linalg as sla

from ..basis import cg
from ..fit.linear import lml_svd
from ..fit.linear import loo
from ..fit.linear import multistart
from ..fit.linear import objective as lo
from ..fit.linear import stats as ls
from .maps import make_room
from . import centred as C
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
    gamma: object = None                 # what the fit was given for this design's columns (`_solve`): the prior precision
    ridge: float = 0.0                   # added to the ridge, or with `hyper` the smoothness of each column
    jack: object = None                  # structures kept apart: (n_rhs, columns, structures), the jackknife deviations (`_jackknife`)


class _Prior(NamedTuple):
    """What `fit.linear.lml_svd.svd_statistics` reads of a problem: the smoothness of the leading
    columns and the fixed precisions of the trailing ones."""
    gamma: np.ndarray
    e0_prec: np.ndarray


@jax.jit
def _evidence_vg(x, s2, chat, P, p0, yy, n):
    """Value and gradient in x = (log sigma, log sigma / sigma_c) of the evidence of one right-hand
    side (chat (:,), yy): `fit.linear.lml_svd.log_evidence_svd` on the design's SVD.  The second
    variable is half of log rho, so the floor of rho is a bound of the search (`_evidence`)."""
    return jax.value_and_grad(lambda x: lml_svd.log_evidence_svd(x[0], x[0] - x[1], lml_svd.SVDStats(s2, chat, P, p0, yy, n, jnp.zeros(()))))(x)


def _noise_floor(noise, channels):
    """`noise=` of the fits as one lower bound of sigma per channel: None; a number, for every
    channel but the overlap "S" (which no SCF noise reaches); or {channel: number}."""
    if noise is None:
        return None
    if isinstance(noise, dict):
        unknown = set(noise) - set(channels)
        if unknown:
            raise ValueError(f"noise: no channel {sorted(unknown)} among {tuple(channels)}")
        return tuple(float(noise.get(c, 0.0)) for c in channels)
    return tuple(0.0 if c == "S" else float(noise) for c in channels)


def _rho_floor(gamma, ridge):
    """The smallest rho = sigma^2 / sigma_c^2 a selection may take: gamma_j^2 rho >= ridge for every
    column with a smoothness prior, so a selected prior is nowhere weaker than the fit without one."""
    g = np.asarray(gamma, float)
    return ridge / float(g[g > 0].min()) ** 2


RHO_STEP = 0.1          # decades: right-hand sides whose sigma^2 / sigma_c^2 agree to this share one posterior solve


def _evidence(fac, gamma, ridge, nc, floor=None):
    """The evidence fit of one design.  Every right-hand side (a shell pair and a channel: their
    scales differ by orders of magnitude) has its own noise sigma and smoothness-prior scale
    sigma_c (coefficient j ~ N(0, sigma_c^2 / gamma_j^2); columns with gamma_j = 0 keep the
    precision `ridge` at unit noise), the maximum of its evidence: the potential fit's SVD
    evidence (`fit.linear.lml_svd`: one SVD of R D^-1 for the design, O(k) per evaluation) by
    bounded L-BFGS from three starts (`fit.linear.multistart`).  The posterior mean depends on
    rho = sigma^2 / sigma_c^2 alone, so right-hand sides are solved together by rho rounded to
    RHO_STEP decades (`fit.linear.objective.posterior_qr`).  Returns (mean (k, n_rhs) in
    the design's column order, hyper (n_rhs, 4): sigma, sigma_c at the maximum, the log evidence
    there, and the sigma_c used after rounding).

    floor: per channel, a lower bound of sigma in the data's unit (the noise of the reference
    data, e.g. the scatter of the blocks between SCF runs; `_noise_floor`); None: 1e-9 of each
    target's rms alone.  It keeps the evidence from reading interpolated data as exact, but does
    not by itself keep the prior above the ridge (Ti-Ti, sigma >= 1e-7 eV: coefficients of 1.8e3
    against 51 without a prior), so:

    The search is in (log sigma, log sigma / sigma_c) with rho bounded below by `_rho_floor`: the
    evidence is maximised over priors that are at least the ridge on every column, and the solve
    is the model the evidence was computed for.  Without the bound a design with fewer rows than
    columns (a species pair with few bonds) is interpolated, the evidence takes sigma to its lower
    bound, and gamma^2 rho fell orders of magnitude below the ridge: the coefficients of nearly
    dependent columns reached 1e13 (Ti-Ti in NbFeSb; docs/dev/ham-results.md)."""
    gamma = np.asarray(gamma, float)
    free = np.flatnonzero(gamma > 0)
    order = np.concatenate([free, np.flatnonzero(gamma <= 0)])            # smoothness columns first, fixed ones last
    R, c = np.asarray(fac.R, float)[:, order], np.asarray(fac.c, float)
    yy, n = np.asarray(fac.yy, float), float(fac.n)
    prior = _Prior(gamma[free], np.full(len(order) - len(free), ridge))
    sv = lml_svd.svd_statistics([ls.Factor(R, c, jnp.asarray(yy.sum()), fac.n, jnp.zeros(()))], prior)
    hyper = np.zeros((c.shape[1], 4))
    vmin = 0.5 * np.log(_rho_floor(gamma, ridge))                         # log sigma / sigma_c at the floor of rho
    for q in range(c.shape[1]):
        scale = float(np.sqrt(max(yy[q] / n, 1e-300)))                    # rms of this target
        ls_ = np.log(max(1e-9 * scale, floor[q % nc] if floor is not None else 0.0))    # a floor above the target's own size leaves sigma there
        lo_, hi_ = np.array([ls_, vmin]), np.array([max(np.log(1e2 * scale), ls_ + 1e-6), max(np.log(1e5), vmin + 1.0)])
        vg = lambda x, q=q: _evidence_vg(jnp.asarray(x), sv.s2, sv.chat[:, q], sv.P, sv.p0, jnp.asarray(yy[q]), sv.n)
        starts = [np.array([min(max(np.log(1e-2 * scale), ls_), hi_[0]), min(max(np.log(v), vmin), hi_[1])]) for v in (1e-4, 1e-7, 1e-10)]
        best, _ = multistart.multistart_map(vg, starts, lo_, hi_, maxiter=200)
        hyper[q, :3] = np.exp(best["x"][0]), np.exp(best["x"][0] - best["x"][1]), best["value"]
    rho = np.maximum(np.round(2.0 * np.log10(hyper[:, 0] / hyper[:, 1]) / RHO_STEP) * RHO_STEP, 2.0 * vmin / np.log(10.0))   # log10 sigma^2 / sigma_c^2, not below the floor
    hyper[:, 3] = hyper[:, 0] / 10.0 ** (0.5 * rho)
    mean = np.zeros_like(c)
    for r in np.unique(rho):
        q = np.flatnonzero(rho == r)
        lin = jnp.asarray(np.concatenate([prior.gamma ** 2 * 10.0 ** r, prior.e0_prec]))
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
    rho values below `_rho_floor` are not tried (see `_evidence`).
    Returns (mean, hyper) as `_evidence` (hyper[:, 2]: the criterion)."""
    gamma, n = np.asarray(gamma, float), float(fac.n)
    blocks = [_Spill.get(b) for b in blocks]
    m = max(len(b[0]) for b in blocks)                   # a structure with fewer rows than columns keeps only those rows
    P, Y = (jnp.asarray(np.stack([np.concatenate([b[i], np.zeros((m - len(b[i]), b[i].shape[1]))]) for b in blocks])) for i in (0, 1))
    ee = np.sum([b[2] for b in blocks], axis=0)
    f = ls.Factor(jnp.asarray(fac.R), jnp.asarray(fac.c), jnp.asarray(fac.yy), fac.n, jnp.zeros(()))
    crit, s2, means = [], [], []
    floor = np.log10(_rho_floor(gamma, ridge))
    grid = np.concatenate([[floor], LOO_RHO[LOO_RHO > floor]])
    for r in grid:
        quad, hld, pr, mean = (np.asarray(v) for v in loo.loo_terms(jnp.asarray(np.where(gamma > 0, gamma ** 2 * 10.0 ** r, ridge)), f, P, Y))
        s2.append((quad + ee) / n); means.append(mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            crit.append(-(pr + ee) if press else -0.5 * n * np.log(s2[-1]) + hld)
    crit = np.where(np.isfinite(crit), crit, -np.inf)
    best, q = crit.argmax(0), np.arange(crit.shape[1])
    sig = np.sqrt(np.asarray(s2)[best, q])
    sc = sig / 10.0 ** (0.5 * grid[best])
    mean = np.asarray(means)[best, :, q].T
    return mean, np.stack([sig, sc, crit[best, q], sc], axis=1), _jackknife(fac, gamma, ridge, blocks, grid[best], mean)


def _jackknife(fac, gamma, ridge, blocks, rho, mean):
    """(n_rhs, columns, structures): the change of each right-hand side's coefficients when one
    structure is left out, at its own prior, times sqrt((G - 1) / G): the
    delete-one-structure jackknife (CR3), whose outer product over the structures is the
    covariance of the coefficients that the scatter between structures supports.  In closed form
    from the structure's rows P_g and residual r_g: -A^-1 P_g^T (1 - P_g A^-1 P_g^T)^-1 r_g, with
    A = R^T R + diag(prior).  A leverage of 1 within roundoff is clamped (`uq.jackknife`).

    rho: (n_rhs,) log10 sigma^2 / sigma_c^2 of each right-hand side, the prior precision
    gamma_j^2 rho on the columns with a smoothness gamma_j and `ridge` on the others (a selected
    prior: `_loo`, `_evidence`); None: the fixed prior of `_solve`, ridge + gamma (gamma None: the
    ridge alone), the same for every right-hand side.  blocks: per structure (R_g, c_g, e_g), or
    the `_Spill` paths of them."""
    R = np.asarray(fac.R, float)
    G = len(blocks)
    out = np.zeros((mean.shape[1], R.shape[1], G))
    fixed = rho is None
    rho = np.zeros(mean.shape[1]) if fixed else np.asarray(rho, float)
    for r in np.unique(rho):
        q = np.flatnonzero(rho == r)
        lin = (np.full(R.shape[1], float(ridge)) + (0.0 if gamma is None else np.asarray(gamma, float)) if fixed
               else np.where(gamma > 0, gamma ** 2 * 10.0 ** r, ridge))
        U = np.linalg.qr(np.concatenate([R, np.diag(np.sqrt(lin))]), mode="r")
        for g, b in enumerate(blocks):
            Pg, Yg = (b if not isinstance(b, str) else _Spill.get(b))[:2]
            if not len(Pg):                                  # a structure without rows of this design
                continue
            W = sla.solve_triangular(U, Pg.T, trans="T")                     # U^-T P_g^T: H_g = W^T W
            mu, V = np.linalg.eigh(np.eye(len(Pg)) - W.T @ W)
            res = Yg[:, q] - Pg @ mean[:, q]
            out[q, :, g] = -sla.solve_triangular(U, W @ (V @ ((V.T @ res) / np.maximum(mu, 1e-10)[:, None]))).T
    return out * np.sqrt((G - 1) / G)


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


def _fit(designs, n_chunks, rows, nc, ridge, device=False, gamma=None, select=None, groups=None, floor=None, jackknife=False):
    """Stream and solve.  designs: {design key: (n columns, [coefficient keys])}; rows(i) ->
    {design key: (A (n_rows, n_columns), Y (n_rows, n_keys * nc))} for chunk i (a design without
    rows in the chunk is left out).  Returns (coef, report, stats).

    device: rows(i) gives {design key: (A, Y, n_rows)} with A a JAX array on the device, padded
    with zero rows to a fixed size (n_rows: the real ones), and the triangular factors are
    merged there (`design.qr_merge`): nothing the size of the design rows comes to the host.

    groups (select "loo" or "press", or `jackknife`): the structure of each chunk.  Each
    structure's rows are then merged into a factor of their own, kept on disk for `_loo` and
    `_jackknife` (`_Spill`: per design, structures x columns^2 numbers), and the factor of all the
    rows is the merge of those.

    jackknife: the statistics carry the delete-one-structure jackknife deviations
    (`DesignStats.jack`, `_jackknife`) at the prior the coefficients were solved with, whatever
    `select` is: a fixed prior needs no selection for it.  Needs `groups`.

    gamma[d] may be a function of the diagonal of the design's Gram matrix R^T R (a prior scaled
    to the columns, e.g. r diag(G)); the statistics hold its value."""
    if jackknife and groups is None:
        raise ValueError("_fit: the jackknife needs the structure of each chunk (groups)")
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
        try:
            return _solve(designs, acc, yy, n, nc, ridge, gamma, select, blocks, floor, jackknife)
        finally:
            spill.dir.cleanup()

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
    try:
        return _solve(designs, acc, yy, n, nc, ridge, gamma, select, blocks, floor, jackknife)
    finally:
        spill.dir.cleanup()


def _solve(designs, acc, yy, n, nc, ridge, gamma=None, select=None, blocks=None, floor=None, jackknife=False):
    """Coefficients, report and statistics from the factors (R, c) of each design.  gamma:
    {design key: prior precision of each column} added to the ridge (the smoothness prior), or,
    with `select` (one of SELECT), {design key: smoothness of each column} for `_evidence` or,
    with the per-structure `blocks` of `_fit`, `_loo`.  jackknife: `DesignStats.jack` from the
    blocks at the prior solved with (`_jackknife`; "loo" and "press" always have it)."""
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
        hyper = jack = None
        g = None if gamma is None else gamma[d]
        if callable(g):
            g = g((np.asarray(R, float) ** 2).sum(0))
        if select == "evidence":
            mu, hyper = _evidence(fac, g, ridge, nc, floor)
            if jackknife:                                # at the prior each right-hand side was solved with
                jack = _jackknife(fac, np.asarray(g, float), ridge, blocks[d], 2.0 * np.log10(hyper[:, 0] / hyper[:, 3]), mu)
        elif select:
            mu, hyper, jack = _loo(fac, g, ridge, blocks[d], select == "press")
        else:
            lin = jnp.full(k, ridge) if g is None else ridge + jnp.asarray(g)
            mu = np.asarray(lo.posterior_qr(lin, [fac], [0.0])[0])                       # (k, n_keys * nc)
            if jackknife:
                jack = _jackknife(fac, g, ridge, blocks[d], None, mu)
        res2 = ((np.asarray(R) @ mu - np.asarray(c)) ** 2).sum(0) + np.maximum(yy[d] - (np.asarray(c) ** 2).sum(0), 0.0)
        for q, key in enumerate(keys):
            coef[key] = mu[:, q * nc:(q + 1) * nc].T.copy()
            report[key] = (n[d], k, np.sqrt(res2[q * nc:(q + 1) * nc] / n[d]))
        stats[d] = DesignStats(fac, tuple(keys), hyper, None if g is None else np.asarray(g, float), float(ridge), jack)
    return coef, report, stats


def fit_onsite(basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, atoms=None, stats=False, sigma_c=None, noise=None,
               jackknife=False):
    """An `OnsiteModel` fitted to the on-site blocks of `datasets` (`OperatorData`s).

    Returns (model, report): report[(z, s1, s2, L)] = (rows, columns, rms residual per channel).
    A part with no feature of its parity and L > 0 has no coefficients (predicted zero); its
    report entry has 0 columns and the rms of the target itself.  `atoms`: per dataset, the atom
    indices to fit to (default: every atom); the others still act as neighbours.
    `stats=True`: also {(z, L, p): `DesignStats`}, the QR statistics of each design.

    sigma_c: None, the plain ridge; a number: the algebraic smoothness prior of the potential
    fit (`basis.prior`), precision ridge + (gamma / sigma_c)^2 on the feature of smoothness gamma
    (`SiteBasis.smoothness`; unit noise, so sigma_c is in the units of the blocks).  The constant
    column of the L = 0 parts keeps the ridge alone.  noise: with "evidence", a lower bound of the noise in the data's unit (a
    number: every channel but "S"; or {channel: number}; `_noise_floor`).  "evidence": sigma_c and the noise of every
    shell pair and channel maximise its evidence (`_evidence`; the values are in the statistics'
    `hyper`).  "loo": sigma_c of every shell pair and channel maximises the predictive density of
    the structures left out one at a time (`_loo`; `aj fit --objective loo` for these rows);
    "press": it minimises their squared prediction error.

    jackknife: keep a factor per structure and give the statistics the closed-form
    delete-one-structure jackknife of the coefficients (`DesignStats.jack`, `_jackknife`) at the
    prior of the fit, whichever `sigma_c` ("loo" and "press" have it without asking)."""
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
    coef, report, st = _fit(designs, len(datasets), rows, nc, ridge, gamma=gamma, select=sigma_c if ev else None, floor=_noise_floor(noise, channels),
                            groups=np.arange(len(datasets)) if (ev and sigma_c != "evidence") or jackknife else None, jackknife=jackknife)
    model = OnsiteModel(basis=basis, coef=coef, orbitals=orbitals, channels=tuple(channels))
    return (model, report, st) if stats else (model, report)


def fit_offsite(bond_basis, datasets, orbitals=None, channels=("H", "S"), ridge=1e-10, chunk=2048, stats=False, compiled=True,
                sigma_c=None, noise=None, jackknife=False):
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
    columns^2 numbers each.  jackknife: as `fit_onsite` (it holds the same factors)."""
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
    sel = dict(floor=_noise_floor(noise, channels), select=sigma_c if ev else None, groups=[c[0] for c in chunks] if (ev and sigma_c != "evidence") or jackknife else None,
               jackknife=jackknife)
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


METRICS = ("plain", "overlap")
CLIP = 1e-12                                           # eigenvalues of an overlap below this, of the largest, are raised to it


def overlap_root(S, overwrite=False):
    """(W, clipped): W (n, n) with W^T W = S, the metric of an overlap-weighted fit as a row
    transform (|W x|^2 = x^T S x).  The upper Cholesky factor; for an S that is not positive
    definite to roundoff (nearly dependent functions), sqrt(e) V^T of its eigen-decomposition
    with the eigenvalues raised to CLIP of the largest, and `clipped` how many were.
    overwrite: the factor may take S's memory (S is O(n^2): at 10^4.5 functions, 8 GB)."""
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"an overlap must be square, got {S.shape}")
    if not np.isfinite(S).all():
        raise ValueError("an overlap has entries that are not finite")
    a = S.T if S.flags.c_contiguous else S               # S is symmetric: its transpose is the Fortran-ordered array LAPACK factors in place
    diag = np.diag(a).copy()
    U, info = sla.lapack.dpotrf(a, lower=0, clean=0, overwrite_a=overwrite)
    if info == 0:
        for r in range(0, len(U), 1024):                 # the triangle LAPACK did not touch still holds S
            U[r:r + 1024] = np.triu(U[r:r + 1024], k=r)
        return U, 0
    S = np.tril(U, -1)                                   # not positive definite: S again, from that triangle
    S = S + S.T
    S[np.diag_indices_from(S)] = diag
    e, V = np.linalg.eigh(S)
    lo_ = CLIP * e[-1]
    return np.sqrt(np.maximum(e, lo_))[:, None] * V.T, int((e < lo_).sum())


def fit_centred(basis, datasets, shells=None, channels=("rho",), metric=None, overlaps=None, ridge=1e-10, sigma_c=None, scaled=None,
                noise=None, stats=False, jackknife=False, chunk=4096):
    """A `centred.CentredModel` fitted to the atom-centred coefficients of `datasets`
    (`data.CentredData`s; `shells`: their auxiliary basis).

    metric "plain": least squares on each coefficient.  As `fit_onsite`: one design per (z, l),
    columns the features of (l, (-1)^l) (and the constant for l = 0), rows (atom, m), every shell
    of that l and every channel a right-hand side.  Returns (model, report) with
    report[(z, s)] = (rows, columns, rms residual per channel) and, `stats=True`,
    {(z, l): `DesignStats`}.

    metric "overlap" (the default when every structure has one): the sum over structures of
    (P b - c)^T S (P b - c), S = `overlaps[k]`, the structure's (n_total, n_total) overlap of the
    functions in the coefficients' layout, the same for every channel (an array, or a function
    without arguments that returns it: called when the structure's rows are made, so one overlap
    is in memory at a time).  S couples shells, l and atoms, so there is ONE design, its columns
    the parameters of all species and shells (`centred.columns`), and with S = W^T W
    (`overlap_root`) its rows are W P and its targets W c: ordinary least squares, through the
    same statistics and solves.  P is never formed: an atom's rows touch only the columns of its
    species' shells, so W P is assembled a shell at a time, in chunks of `chunk` rows.  The cost
    is the root of S, O(n_total^3) time and O(n_total^2) memory per structure.
    report[("overlap",)] = (rows, columns, sqrt of the mean over rows of the S-weighted squared
    residual, per channel) and report[("clipped",)] = per structure, the eigenvalues of S that
    were raised (`overlap_root`; 0: the Cholesky factor); statistics {("overlap",): `DesignStats`},
    one right-hand side per channel.

    sigma_c: as `fit_onsite`: None, a number, "evidence", "loo" or "press", with
    `SiteBasis.smoothness(l, p)` for the feature columns and none for the constants.
    scaled: r, a prior precision of r times the diagonal of the design's Gram matrix on every
    feature column, added to the ridge (a ridge relative to the size of each column; the
    constants keep the ridge alone).  Not with `sigma_c`.
    jackknife: the statistics carry the delete-one-structure jackknife (`DesignStats.jack`;
    `centred_jackknife` for the coefficients of each model with a structure left out)."""
    _need_x64()
    datasets = list(datasets)
    shells = datasets[0].basis if shells is None else shells
    if any(d.basis != shells for d in datasets):
        raise ValueError("fit_centred: the datasets have different auxiliary bases")
    nc = len(channels)
    have = overlaps is not None and len(overlaps) == len(datasets) and all(S is not None for S in overlaps)
    metric = ("overlap" if have else "plain") if metric is None else metric
    if metric not in METRICS:
        raise ValueError(f"metric: one of {METRICS}, got {metric!r}")
    if metric == "overlap" and not have:
        raise ValueError("fit_centred: metric='overlap' needs the overlap of every structure (overlaps=)")
    ev = isinstance(sigma_c, str)
    if ev and sigma_c not in SELECT:
        raise ValueError(f"sigma_c: None, a number or one of {SELECT}, got {sigma_c!r}")
    if scaled is not None and sigma_c is not None:
        raise ValueError("fit_centred: give sigma_c or scaled, not both")
    smooth = lambda l: np.concatenate([basis.smoothness(l, C.parity(l)), np.zeros(int(l == 0))])
    feat = lambda l: np.concatenate([np.ones(basis.n(l, C.parity(l))), np.zeros(int(l == 0))])      # 1 on feature columns, 0 on constants
    prior = lambda sm, ft: (lambda diag, r=float(scaled), ft=ft: r * ft * diag) if scaled is not None else None if sigma_c is None else sm if ev else sm ** 2 / sigma_c ** 2
    off = [C.offsets(shells, d.numbers) for d in datasets]
    for d, o in zip(datasets, off):
        if any(np.shape(d.coefficients[c]) != (o[-1],) for c in channels):
            raise ValueError(f"fit_centred: a structure's coefficients are not ({o[-1]},) per channel, the size of its atoms' shells")

    def atom_rows(k, z, sls):
        """(atoms of z in structure k, their rows of the shells' slices: (atoms, shells, 2l+1))"""
        idx = np.flatnonzero(np.asarray(datasets[k].numbers) == z)
        return idx, off[k][idx][:, None, None] + np.asarray([np.arange(sl.start, sl.stop) for sl in sls])[None]

    def features(k):
        d = datasets[k]
        rij, zi, zj, seg, n = site_graph(basis, d.positions, d.cell, d.numbers, getattr(d, "pbc", True))
        if len(rij) == 0:
            return {(L, p): np.zeros((n, basis.n(L, p), 2 * L + 1)) for L, p in basis.lps}
        return {key: np.asarray(v) for key, v in _features(basis, rij, zi, zj, seg, n).items()}

    kw = dict(select=sigma_c if ev else None, floor=_noise_floor(noise, channels), jackknife=jackknife)
    if metric == "plain":
        designs, members = {}, {}                       # (z, l) -> (columns, keys); -> [s]
        for z in shells.species:
            for s, l in enumerate(shells.ls(z)):
                designs.setdefault((z, l), (C.n_columns(basis, l), []))[1].append((z, s))
                members.setdefault((z, l), []).append(s)

        def rows(k):
            F, c = features(k), np.stack([np.asarray(datasets[k].coefficients[ch], float) for ch in channels], axis=-1)   # (n_total, nc)
            out = {}
            for (z, l), mem in members.items():
                idx, r = atom_rows(k, z, [shells.shell_slice(z, s) for s in mem])
                if len(idx):
                    A = design(F.get((l, C.parity(l))), idx, l).reshape(len(idx) * (2 * l + 1), -1)
                    out[(z, l)] = (A, np.moveaxis(c[r], 1, 2).reshape(len(idx) * (2 * l + 1), len(mem) * nc))    # (a, s, m, c) -> rows (a, m), columns (s, c)
            return out

        gamma = None if sigma_c is None and scaled is None else {(z, l): prior(smooth(l), feat(l)) for z, l in designs}
        coef, report, st = _fit(designs, len(datasets), rows, nc, ridge, gamma=gamma, groups=np.arange(len(datasets)) if (ev and sigma_c != "evidence") or jackknife else None, **kw)
    else:
        cols, npar = C.columns(basis, shells)
        chunks = [(k, r0, min(r0 + chunk, o[-1])) for k, o in enumerate(off) for r0 in range(0, o[-1], chunk)]
        cache, clipped = {}, [0] * len(datasets)

        def rows(i):
            k, r0, r1 = chunks[i]
            if cache.get("k") != k:                      # chunks come structure by structure: its features, root and targets once
                cache.clear()
                S = overlaps[k]() if callable(overlaps[k]) else overlaps[k]
                if np.shape(S) != (off[k][-1],) * 2:
                    raise ValueError(f"fit_centred: the overlap of structure {k} is {np.shape(S)}, its coefficients ({off[k][-1]},)")
                own = callable(overlaps[k])              # ours to overwrite: the root in the overlap's memory
                W, clipped[k] = overlap_root(S if own and isinstance(S, np.ndarray) and S.dtype == np.float64 and S.flags.writeable else np.array(S, float), overwrite=True)
                del S
                cache.update(k=k, F=features(k), W=W, Y=W @ np.stack([np.asarray(datasets[k].coefficients[ch], float) for ch in channels], axis=-1))
            F, W = cache["F"], cache["W"][r0:r1]
            A = np.zeros((r1 - r0, npar))
            for z in shells.species:
                for s, l in enumerate(shells.ls(z)):
                    idx, r = atom_rows(k, z, [shells.shell_slice(z, s)])
                    if len(idx) and cols[(z, s)].stop > cols[(z, s)].start:
                        A[:, cols[(z, s)]] = W[:, r.ravel()] @ design(F.get((l, C.parity(l))), idx, l).reshape(len(idx) * (2 * l + 1), -1)
            return {("overlap",): (A, cache["Y"][r0:r1])}

        parts = [(smooth(l), feat(l)) for z in shells.species for l in shells.ls(z)]
        sm, ft = (np.concatenate([p[i] for p in parts]) if parts else np.zeros(0) for i in (0, 1))
        gamma = None if sigma_c is None and scaled is None else {("overlap",): prior(sm, ft)}
        coef, report, st = _fit({("overlap",): (npar, [("overlap",)])}, len(chunks), rows, nc, ridge, gamma=gamma,
                                groups=[c[0] for c in chunks] if (ev and sigma_c != "evidence") or jackknife else None, **kw)
        cache.clear()
        b = coef.pop(("overlap",), None)                # (nc, npar)
        coef = {} if b is None else {key: b[:, sl].copy() for key, sl in cols.items() if sl.stop > sl.start}
        report[("clipped",)] = tuple(clipped)
    model = C.CentredModel(basis=basis, coef=coef, shells=shells, channels=tuple(channels))
    return (model, report, st) if stats else (model, report)


def centred_jackknife(model, stats):
    """[coef, ...], one per training structure: the coefficients of `model` (a `fit_centred` with
    `jackknife=True`, or "loo" / "press", and its statistics) refitted without that structure at
    the prior of the fit, in closed form: the fitted coefficients plus `DesignStats.jack` with
    its factor sqrt((G - 1) / G) taken out.  Each is the `coef` of a `CentredModel`
    (`eqx.tree_at(lambda m: m.coef, model, coef)`): the spread of their predictions about the
    model's, times sqrt((G - 1) / G), is the jackknife standard error."""
    nc = len(model.channels)
    cols, _ = C.columns(model.basis, model.shells)
    out = None
    for st in stats.values():
        if st.jack is None:
            raise ValueError("no jackknife in these statistics: fit with jackknife=True")
        G = st.jack.shape[2]
        out = [{key: np.array(c, float) for key, c in model.coef.items()} for _ in range(G)] if out is None else out
        dev = st.jack / np.sqrt((G - 1) / G)                                 # (n_rhs, columns, G)
        for g in range(G):
            if st.keys == (("overlap",),):
                for key in out[g]:
                    out[g][key] += dev[:, cols[key], g]
            else:
                for i, key in enumerate(st.keys):
                    if key in out[g]:
                        out[g][key] += dev[i * nc:(i + 1) * nc, :, g]
    return out
