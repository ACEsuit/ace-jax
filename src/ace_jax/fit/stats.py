"""Sufficient statistics of the data, per observation type, streamed over
batches.  Memory is one batch's rows plus the (Dt, Dt) accumulators; the scan's
reverse pass recomputes each batch under jax.checkpoint, which is the two-pass
gradient algorithm of the spec with no hand-written VJP."""
from typing import NamedTuple

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .linear.stats import (CPU_SCAN_MAX_L, QR_MERGE_ROWS_PER_L, TPQRT_NB, MPIWorkerDone, QRCheckpoint,  # noqa: F401
                           Factor, Gram, _host_merge, _live, _qr_merge, active_checkpoint, active_comm,
                           device_qr_statistics, device_qr_stream, gram, host_qr_statistics, host_qr_stream,
                           merge_factors, merge_target, mpi_shards, problem_key, qr_checkpoint, shard_batches,
                           tree_reduce, tree_schedule, weighted)
from .rows import QUANTITIES, batch_rows, efv_rows


class Stats(NamedTuple):
    G_E: jnp.ndarray; G_F: jnp.ndarray; G_V: jnp.ndarray
    b_E: jnp.ndarray; b_F: jnp.ndarray; b_V: jnp.ndarray
    yy_E: jnp.ndarray; yy_F: jnp.ndarray; yy_V: jnp.ndarray
    n_E: jnp.ndarray; n_F: jnp.ndarray; n_V: jnp.ndarray
    logw_E: jnp.ndarray; logw_F: jnp.ndarray; logw_V: jnp.ndarray

    quantities = QUANTITIES

    @property
    def grams(self):
        """The generic form: one `linear.stats.Gram` per quantity, in `QUANTITIES` order."""
        return tuple(Gram(getattr(self, f"G_{q}"), getattr(self, f"b_{q}"), getattr(self, f"yy_{q}"),
                          getattr(self, f"n_{q}"), getattr(self, f"logw_{q}")) for q in QUANTITIES)


def _type_stats(Phi, y, w):
    """Phi (n, Dt), y (n,), w (n,) structural weights (0 = absent): linear.stats.gram, as a tuple."""
    return tuple(gram(Phi, y, w))


def _efv_stats(r, batch):
    """Stats of one batch from its rows r (.E, .F, .V): `gram` of each quantity of `rows.efv_rows`."""
    E, F, V = (gram(*q) for q in efv_rows(r, batch))
    return Stats(E[0], F[0], V[0], E[1], F[1], V[1], E[2], F[2], V[2],
                 E[3], F[3], V[3], E[4], F[4], V[4])


def batch_stats(theta, spec, model, ind, cfg, batch):
    return _efv_stats(batch_rows(theta, spec, model, ind, cfg, batch), batch)


def sufficient_statistics(theta, spec, model, ind, cfg, ds):
    Dt = cfg.len_basis + ind.XM.shape[0]
    z2, z1, z0 = jnp.zeros((Dt, Dt)), jnp.zeros(Dt), jnp.zeros(())
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda th, b: batch_stats(th, spec, model, ind, cfg, b))

    def body(carry, batch):
        return jax.tree.map(jnp.add, carry, f(theta, batch)), None

    return jax.lax.scan(body, zero, ds)[0]


# ---------------------------------------------------------------------------
# theta-split statistics.  The linear block G_BB is theta-INDEPENDENT (the ACE
# features do not move with the hyperparameters; sigma_c enters as the prior,
# not the Gram), so it is streamed ONCE and cached, and only the M residual
# columns k_theta(B, B_M) are recomputed per evaluation.  For the Cantor basis
# (L = 6890, M = 500) this cuts the per-gradient Gram cost ~10x -- the expensive
# 6890^2 block leaves the inner loop.  assemble(linear, residual(theta)) equals
# sufficient_statistics(theta) to f64 roundoff (tests/test_gp_stats.py).

class ResidualStats(NamedTuple):
    GBM_E: jnp.ndarray; GBM_F: jnp.ndarray; GBM_V: jnp.ndarray   # (L, M)
    GMM_E: jnp.ndarray; GMM_F: jnp.ndarray; GMM_V: jnp.ndarray   # (M, M)
    bM_E: jnp.ndarray;  bM_F: jnp.ndarray;  bM_V: jnp.ndarray    # (M,)


_linear_type_stats = _type_stats      # G_BB, b_B, yy, n, logw for one observation type from the linear rows


def batch_linear_stats(model, cfg, batch):
    from .rows import linear_rows_bounded
    return linear_stats_from_rows(linear_rows_bounded(model, cfg, batch), batch)


def linear_stats_from_rows(r, batch):
    """batch_linear_stats from already-built linear rows r."""
    return _efv_stats(r, batch)


def linear_statistics(model, cfg, ds):
    """theta-independent linear statistics: Stats with Dt = len_basis."""
    L = cfg.len_basis
    z2, z1, z0 = jnp.zeros((L, L)), jnp.zeros(L), jnp.zeros(())
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda b: batch_linear_stats(model, cfg, b))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(b)), None), zero, ds)[0]


class QRStats(NamedTuple):
    """The linear statistics in QR form: per quantity q, R_q (L, L) upper triangular and
    c_q = Q_q^T (w y_q) from a QR of its weighted rows w Phi_q = Q_q R_q, so R_q^T R_q = G_q
    and R_q^T c_q = b_q without ever forming G_q (whose condition number is kappa(Phi_q)^2).
    yy, n and logw are as in Stats.  objective.log_marginal_likelihood_qr reads it; the Gram
    form (G_q, b_q) is derived on demand for the consumers that need it (POPS, ARD).

    rho_q (float32 rows, row_dtype="float32" on a device): the residual norm |w y_q - Phi_q x_q|
    carried through the merges as the last diagonal of the (L+1) factor of [w Phi_q | w y_q], or
    None.  With it the LML never forms y'y - v'v, which cancels in float32 when the fit nearly
    interpolates (MACE labels: sigma_E 4.8x wrong; docs/dev/fp32-results.md, step 3)."""
    R_E: jnp.ndarray; R_F: jnp.ndarray; R_V: jnp.ndarray
    c_E: jnp.ndarray; c_F: jnp.ndarray; c_V: jnp.ndarray
    yy_E: jnp.ndarray; yy_F: jnp.ndarray; yy_V: jnp.ndarray
    n_E: jnp.ndarray; n_F: jnp.ndarray; n_V: jnp.ndarray
    logw_E: jnp.ndarray; logw_F: jnp.ndarray; logw_V: jnp.ndarray
    rho_E: object = None; rho_F: object = None; rho_V: object = None

    G_E = property(lambda s: s.R_E.T @ s.R_E)
    G_F = property(lambda s: s.R_F.T @ s.R_F)
    G_V = property(lambda s: s.R_V.T @ s.R_V)
    b_E = property(lambda s: s.R_E.T @ s.c_E)
    b_F = property(lambda s: s.R_F.T @ s.c_F)
    b_V = property(lambda s: s.R_V.T @ s.c_V)

    quantities = QUANTITIES
    shared_quantity = "F"            # whose noise scale a shared-noise fit keeps (paramset.SHARED_NOISE)

    @property
    def factors(self):
        """The generic form: one `linear.stats.Factor` per quantity, in `QUANTITIES` order."""
        return tuple(Factor(getattr(self, f"R_{q}"), getattr(self, f"c_{q}"), getattr(self, f"yy_{q}"),
                            getattr(self, f"n_{q}"), getattr(self, f"logw_{q}"), getattr(self, f"rho_{q}"))
                     for q in QUANTITIES)

    @classmethod
    def from_factors(cls, f):
        """QRStats of three factors (E, F, V); rho is kept only when every factor carries one."""
        rho = tuple(x.rho for x in f) if all(x.rho is not None for x in f) else ()
        return cls(*(x.R for x in f), *(x.c for x in f), *(x.yy for x in f), *(x.n for x in f),
                   *(x.logw for x in f), *rho)

    @property
    def grams(self):
        return self.gram().grams

    def gram(self):
        """The equivalent Stats (Gram form), for consumers that need G and b."""
        return Stats(self.G_E, self.G_F, self.G_V, self.b_E, self.b_F, self.b_V,
                     self.yy_E, self.yy_F, self.yy_V, self.n_E, self.n_F, self.n_V,
                     self.logw_E, self.logw_F, self.logw_V)


def _qr_rows(model, cfg, batch):
    """One batch's weighted linear rows per quantity, and their yy, n, logw increments."""
    from .rows import linear_rows_bounded
    return [weighted(*q) for q in efv_rows(linear_rows_bounded(model, cfg, batch), batch)]


# the host path's rows with the model as an argument: one compile per (model structure, cfg, shapes).  Opt-in
# (linear_qr_statistics(cached_rows=True), the ARD stage) -- the default per-call closure is left as it is, since
# the bit-exact pipeline goldens were recorded with it (linear_qr_statistics' docstring)
_qr_rows_jit = jax.jit(_qr_rows, static_argnames=("cfg",))


@partial(jax.jit, static_argnames=("cfg", "merge_rows"))
def _device_qr(model, cfg, ds, merge_rows):
    return device_qr_statistics(lambda b: _qr_rows(model, cfg, b), ds, cfg.len_basis, merge_rows)


@partial(jax.jit, static_argnames=("cfg", "merge_rows"))
def _device_qr32(model, cfg, ds, merge_rows):
    """_device_qr with float32 rows and float32 merges: each quantity's target rides as column L
    of the rows, so the (L+1) factor carries the residual norm too (QRStats.rho_q).  Targets,
    weights and the y'y / n / logw sums stay float64."""
    L = cfg.len_basis
    f32 = jnp.float32
    zeros = [(np.zeros((L + 1, L + 1), np.float32), np.zeros(L + 1, np.float32))] * 3

    def rows_fn(batch):
        out = _qr_rows(model, cfg, batch)
        return ([(jnp.concatenate([Pw, yw[:, None]], 1).astype(f32), jnp.zeros(yw.shape[0], f32))
                 for Pw, yw, *_ in out], [o[2:] for o in out])
    z0 = jnp.zeros(())
    return device_qr_stream(rows_fn, ds, zeros, [(z0, z0, z0)] * 3, merge_rows)


def float32_rows(model, ds):
    """The model and dataset of float32 design rows: floating model leaves and rij in float32
    (cast after any float64 preparation of the model); targets and weights stay as they are."""
    from ..eval.api import with_dtype
    return (with_dtype(model, jnp.float32),
            None if ds is None else ds._replace(rij=jnp.asarray(ds.rij, jnp.float32)))


def linear_qr_statistics(model, cfg, ds, merge_rows=None, host=None, row_dtype="float64", cached_rows=False):
    """linear_statistics in QR form (QRStats): one streaming pass over the same linear rows,
    folded into the per-quantity factors by batched updating QRs (QR_MERGE_ROWS_PER_L) -- on
    the host on the CPU backend (host_qr_stream says why), as a lax.scan on a device
    (device_qr_stream).  host: None picks by backend; merge_rows: see host_qr_stream (0 merges
    every batch).  row_dtype="float32": float32 design rows (`float32_rows`); see the branch below.

    On a device the pass is `_device_qr`, module-level with the model as an argument, so one
    compile serves every call with the same shapes.  A jit of a fresh closure over the model,
    built per call, recompiled the whole streaming scan every time: harmless for a fit's single
    pass, but radial learning calls this at every evaluation with new radials, and on Si o4d16
    the GPU spent its rounds compiling.  The host (CPU) path still jits its row function per
    call: taking the model as an argument there moves the bit-exact pipeline goldens (roundoff,
    amplified by the MAP: 3e-4 in run_linear_pops_auto's coefficients), so it is left for a
    change that re-records them.  cached_rows=True takes the module-level `_qr_rows_jit` there instead
    (the ARD stage, which runs several passes per fit and must not recompile on a repeat)."""
    L = cfg.len_basis
    if host is None:
        host = jax.default_backend() == "cpu"
    if row_dtype not in ("float64", "float32"):
        raise ValueError(f"row_dtype must be 'float64' or 'float32', got {row_dtype!r}")
    if row_dtype == "float32":
        # float32 rows (the bulk of a fit's flops); on a device the merges run in float32 too and
        # carry the residual (QRStats.rho_q).  On the host the merges stay float64 (_host_merge),
        # where y'y - v'v does not cancel, so no rho
        model, ds = float32_rows(model, ds)
        if not host:
            out, ex = _device_qr32(model, cfg, ds, merge_rows)
            fac = [(jnp.asarray(Rt[:L, :L], jnp.float64), jnp.asarray(Rt[:L, L], jnp.float64),
                    jnp.abs(jnp.asarray(Rt[L, L], jnp.float64))) for Rt, _ in out]
            return QRStats(*(f[0] for f in fac), *(f[1] for f in fac),
                           *(ex[i][k] for k in range(3) for i in range(3)), *(f[2] for f in fac))
    if host:
        rows = ((lambda b: _qr_rows_jit(model, cfg, b)) if cached_rows
                else jax.jit(lambda b: _qr_rows(model, cfg, b)))
        return QRStats.from_factors(host_qr_statistics(
            rows, ds, L, QUANTITIES, merge_rows,
            key=lambda: problem_key(model, ds, row_dtype, L, merge_target(L, merge_rows))))
    return QRStats.from_factors(_device_qr(model, cfg, ds, merge_rows))


def _residual_type_stats(Phi_B, Phi_M, y, w):
    """G_BM, G_MM, b_M for one type; the cross block reuses the (theta-indep)
    linear rows but is theta-dependent through Phi_M."""
    Bw, Mw, yw = Phi_B * w[:, None], Phi_M * w[:, None], y * w
    return Bw.T @ Mw, Mw.T @ Mw, Mw.T @ yw


def batch_residual_stats(theta, spec, model, ind, cfg, batch):
    r = batch_rows(theta, spec, model, ind, cfg, batch)
    L, Dt = cfg.len_basis, r.E.shape[-1]
    sl = lambda A: (A[..., :L], A[..., L:])
    parts = []
    for R, y, w, reps in ((r.E, batch.y_E, batch.w_E, 1),
                          (r.F, batch.y_F, batch.w_F, 3),
                          (r.V, batch.y_V, batch.w_V, 6)):
        B, Mrows = sl(R.reshape(-1, Dt))
        parts.append(_residual_type_stats(B, Mrows, y.reshape(-1), jnp.repeat(w, reps)))
    return ResidualStats(parts[0][0], parts[1][0], parts[2][0],
                         parts[0][1], parts[1][1], parts[2][1],
                         parts[0][2], parts[1][2], parts[2][2])


def residual_statistics(theta, spec, model, ind, cfg, ds):
    L, M = cfg.len_basis, ind.XM.shape[0]
    zBM, zMM, zM = jnp.zeros((L, M)), jnp.zeros((M, M)), jnp.zeros(M)
    zero = ResidualStats(zBM, zBM, zBM, zMM, zMM, zMM, zM, zM, zM)
    f = jax.checkpoint(lambda th, b: batch_residual_stats(th, spec, model, ind, cfg, b))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(theta, b)), None), zero, ds)[0]


def assemble_statistics(lin, res):
    """Full Stats (Dt = L + M) from the cached linear part and a residual part."""
    def blk(GBB, GBM, GMM):
        return jnp.block([[GBB, GBM], [GBM.T, GMM]])
    G_E = blk(lin.G_E, res.GBM_E, res.GMM_E)
    G_F = blk(lin.G_F, res.GBM_F, res.GMM_F)
    G_V = blk(lin.G_V, res.GBM_V, res.GMM_V)
    cat = jnp.concatenate
    return Stats(G_E, G_F, G_V,
                 cat([lin.b_E, res.bM_E]), cat([lin.b_F, res.bM_F]), cat([lin.b_V, res.bM_V]),
                 lin.yy_E, lin.yy_F, lin.yy_V, lin.n_E, lin.n_F, lin.n_V,
                 lin.logw_E, lin.logw_F, lin.logw_V)


# ---------------------------------------------------------------------------
# Per-(quantity, TYPE) statistics.  These mirror the single-type functions above
# but split each quantity's rows by their config-type index and accumulate one
# Stats block PER TYPE (a leading n_types axis on every field).  The split is a
# weight MASK (w * (type == t)) -- the structural weights carry NO sigma, so each
# per-(q,t) Gram stays theta-INDEPENDENT and cacheable exactly like the linear
# statistics above.  The per-type noise sigma_{q,t} enters only in the objective
# (objective.combine), NOT the Gram, which is what preserves the streamed
# theta-independent Gram caching.  With a single type these reduce (up to the
# leading size-1 axis) to the functions above, so the fit is bit-identical.

def _linear_type_stats_typed(Phi_B, y, w, tidx, n_types):
    comps = [_linear_type_stats(Phi_B, y, jnp.where(tidx == t, w, 0.0))
             for t in range(n_types)]
    return tuple(jnp.stack([c[k] for c in comps]) for k in range(5))


def batch_linear_stats_typed(model, cfg, batch, n_types):
    from .rows import linear_rows_bounded
    r = linear_rows_bounded(model, cfg, batch)
    L = r.E.shape[-1]
    E = _linear_type_stats_typed(r.E, batch.y_E, batch.w_E, batch.cfg_type, n_types)
    F = _linear_type_stats_typed(r.F.reshape(-1, L), batch.y_F.reshape(-1),
                                 jnp.repeat(batch.w_F, 3), jnp.repeat(batch.node_type, 3), n_types)
    V = _linear_type_stats_typed(r.V.reshape(-1, L), batch.y_V.reshape(-1),
                                 jnp.repeat(batch.w_V, 6), jnp.repeat(batch.cfg_type, 6), n_types)
    return Stats(E[0], F[0], V[0], E[1], F[1], V[1], E[2], F[2], V[2],
                 E[3], F[3], V[3], E[4], F[4], V[4])


def linear_statistics_typed(model, cfg, ds, n_types):
    """theta-independent linear statistics, one Stats block per type
    (leading n_types axis on every field)."""
    L = cfg.len_basis
    z2, z1, z0 = jnp.zeros((n_types, L, L)), jnp.zeros((n_types, L)), jnp.zeros((n_types,))
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda b: batch_linear_stats_typed(model, cfg, b, n_types))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(b)), None), zero, ds)[0]


def _residual_type_stats_typed(Phi_B, Phi_M, y, w, tidx, n_types):
    comps = [_residual_type_stats(Phi_B, Phi_M, y, jnp.where(tidx == t, w, 0.0))
             for t in range(n_types)]
    return tuple(jnp.stack([c[k] for c in comps]) for k in range(3))


def batch_residual_stats_typed(theta, spec, model, ind, cfg, batch, n_types):
    r = batch_rows(theta, spec, model, ind, cfg, batch)
    L, Dt = cfg.len_basis, r.E.shape[-1]
    sl = lambda A: (A[..., :L], A[..., L:])
    parts = []
    for R, y, w, tidx, reps in (
            (r.E, batch.y_E, batch.w_E, batch.cfg_type, 1),
            (r.F, batch.y_F, batch.w_F, jnp.repeat(batch.node_type, 3), 3),
            (r.V, batch.y_V, batch.w_V, jnp.repeat(batch.cfg_type, 6), 6)):
        B, Mrows = sl(R.reshape(-1, Dt))
        parts.append(_residual_type_stats_typed(B, Mrows, y.reshape(-1),
                                                jnp.repeat(w, reps), tidx, n_types))
    return ResidualStats(parts[0][0], parts[1][0], parts[2][0],
                         parts[0][1], parts[1][1], parts[2][1],
                         parts[0][2], parts[1][2], parts[2][2])


def residual_statistics_typed(theta, spec, model, ind, cfg, ds, n_types):
    L, M = cfg.len_basis, ind.XM.shape[0]
    zBM, zMM, zM = (jnp.zeros((n_types, L, M)), jnp.zeros((n_types, M, M)),
                    jnp.zeros((n_types, M)))
    zero = ResidualStats(zBM, zBM, zBM, zMM, zMM, zMM, zM, zM, zM)
    f = jax.checkpoint(lambda th, b: batch_residual_stats_typed(th, spec, model, ind, cfg, b, n_types))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(theta, b)), None), zero, ds)[0]


def assemble_statistics_typed(lin, res):
    """Per-type full Stats: vmap assemble_statistics over the leading type axis."""
    return jax.vmap(assemble_statistics)(lin, res)


# ---------------------------------------------------------------------------
# Streamed POPS statistics (Task 9).  One extra pass over the dataset that
# produces the SAME per-point misspecification corrections `deltas` as Task-8's
# pops_corrections on the whole (whitened) design assembled at once, but never
# materialises more than one batch's rows.
#
# Per batch we build the linear rows (rows.linear_rows, the M=0 arm), whiten
# each quantity's rows/residual by w/sigma_q (Task 7's pops.whiten), and compute
# the per-point correction delta_i and leverage h_i with pops.pointwise_corrections
# (Task 8's formula, NO boolean masking -- jit-safe under scan).  The scan stacks
# every batch's (delta, h); the leverage_pct subselection is applied ONCE,
# eagerly, after the scan (pops.leverage_select), which is where the only
# non-jit-safe boolean lives.  With leverage_pct=0 every point (including padded
# w=0 rows, which carry delta=0, h=0) is kept, so streaming == monolithic
# elementwise (tests/test_gp_stats.py).

def _batch_pops_pointwise(model, cfg, batch, c_star, Sigma0, sigma):
    """One batch's per-point corrections and leverages, quantities concatenated
    in E, F, V order.  `sigma` maps 'E'/'F'/'V' -> per-quantity noise scale."""
    from ..uq.pops import pointwise_corrections, whiten
    from .rows import linear_rows_bounded
    r = linear_rows_bounded(model, cfg, batch)
    L = r.E.shape[-1]
    ds_d, ds_h = [], []
    for phi, y, w, sq in (
            (r.E, batch.y_E, batch.w_E, sigma["E"]),
            (r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3), sigma["F"]),
            (r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6), sigma["V"])):
        phi_t, r_t = whiten(phi, y - phi @ c_star, w, sq)
        d, h = pointwise_corrections(Sigma0, phi_t, r_t)
        ds_d.append(d); ds_h.append(h)
    return jnp.concatenate(ds_d, 0), jnp.concatenate(ds_h, 0)


def _stream_pops_pointwise(model, cfg, ds, c_star, Sigma0, sigma):
    """Stream _batch_pops_pointwise over ds, returning all per-point (deltas, h)
    flattened batch-major.  jit-safe (no boolean masking)."""
    body = lambda carry, batch: (
        carry, _batch_pops_pointwise(model, cfg, batch, c_star, Sigma0, sigma))
    _, (dd, hh) = jax.lax.scan(body, 0.0, ds)        # dd (B, P, L), hh (B, P)
    L = dd.shape[-1]
    return dd.reshape(-1, L), hh.reshape(-1)


def pops_statistics(c_star, Sigma0, prob, ds, sigma, *, leverage_pct=0.0):
    """Streamed POPS pointwise corrections over the dataset ``ds``.

    Returns the SAME ``deltas`` (K, L) as Task-8's ``pops.pops_corrections`` on
    the whole whitened design assembled in one batch, but streams ``ds`` one
    batch at a time (peak memory is one batch's rows plus the accumulated
    per-point (delta, h)).

    Parameters
    ----------
    c_star : (L,) array     fitted linear weights (``objective.posterior`` mean).
    Sigma0 : (L, L) array   epistemic weight covariance ``A^{-1}``
                            (``objective.posterior``'s precision inverse).
    prob   : Problem        supplies ``model``/``cfg`` for the linear rows.
    ds     : Dataset        streamed, leading batch axis (the same ``ds`` fed to
                            the statistics/LML; splitting it into more batches
                            leaves ``deltas`` unchanged).
    sigma  : mapping        per-quantity noise scales, keys ``'E'``/``'F'``/``'V'``
                            (single config-type; per-type sigma is a documented
                            extension, not implemented here).
    leverage_pct : float    keyword-only; percentile in [0, 100], 0 keeps all.
    """
    from ..uq.pops import leverage_select
    deltas, h = _stream_pops_pointwise(prob.model, prob.cfg, ds, c_star, Sigma0, sigma)
    return leverage_select(deltas, h, leverage_pct)


# ---------------------------------------------------------------------------
# Streaming POPS (Swinburne & Perez, structural weights).  Every member is
# delta_i = A phi_i c_i  with  c_i = r_i / h_i,  h_i = phi_i . A . phi_i  (A
# symmetric), so the statistics the predictive needs are Gram-shaped:
#     sum delta delta^T = A (sum c_i^2 phi_i phi_i^T) A,   sum delta = A (sum c_i phi_i)
# and the (K, L) matrix of corrections is never formed.  Memory is O(L^2) plus two
# scalars per observation; each pass holds one batch of rows.
# ---------------------------------------------------------------------------

def _pops_batch_rows(model, cfg, batch, qs=(1.0, 1.0, 1.0)):
    """Loss-weighted rows w*phi, weighted target w*y and raw rows phi for one
    batch, quantities concatenated E, F, V (padded rows carry w = 0).  The weight
    is the structural weight times the per-quantity loss scale qs = (1/sigma_E,
    1/sigma_F, 1/sigma_V)."""
    from .rows import linear_rows_bounded
    r = linear_rows_bounded(model, cfg, batch)
    L = r.E.shape[-1]
    phi = jnp.concatenate([r.E, r.F.reshape(-1, L), r.V.reshape(-1, L)])
    y = jnp.concatenate([batch.y_E, batch.y_F.reshape(-1), batch.y_V.reshape(-1)])
    w = jnp.concatenate([batch.w_E * qs[0], jnp.repeat(batch.w_F, 3) * qs[1], jnp.repeat(batch.w_V, 6) * qs[2]])
    return phi * w[:, None], w * y, phi


def pops_leverage_residual(model, cfg, ds, c_star, A, qs=(1.0, 1.0, 1.0)):
    """Pass 1: whitened leverage h_i = pw_i . A . pw_i and residual r_i = w_i (y_i -
    phi_i . c*) of every row, each (n_batches, rows_per_batch).  Padded rows have
    h = 0 (w = 0)."""
    def body(carry, batch):
        pw, wy, phi = _pops_batch_rows(model, cfg, batch, qs)
        h = jnp.sum((pw @ A) * pw, axis=1)
        r = wy - (pw @ c_star)
        return carry, (h, r)
    _, (h, r) = jax.lax.scan(body, 0.0, ds)
    return h, r


def pops_moment_sums(model, cfg, ds, coef, qs=(1.0, 1.0, 1.0)):
    """Pass 2: W = sum_i coef_i^2 pw_i pw_i^T and s = sum_i coef_i pw_i, with
    coef (n_batches, rows) = r/h on member rows and 0 elsewhere."""
    def body(carry, xs):
        batch, c = xs
        pw, _, _ = _pops_batch_rows(model, cfg, batch, qs)
        pc = pw * c[:, None]
        W, s = carry
        return (W + pc.T @ pc, s + pw.T @ c), None
    return jax.lax.scan(body, _moment_init(model, cfg, ds), (ds, coef))[0]


def _moment_init(model, cfg, ds):
    first = jax.tree.map(lambda a: a[0], ds)
    L = jax.eval_shape(lambda b: _pops_batch_rows(model, cfg, b)[0], first).shape[-1]
    return jnp.zeros((L, L)), jnp.zeros(L)


def pops_projection_bounds(model, cfg, ds, coef, keep, B, qs=(1.0, 1.0, 1.0)):
    """Pass 3: per-axis min / max over member rows of the projected corrections
    (pw_i @ B) * coef_i, with B = A @ support (L, d).  Exact for zero percentile
    clipping (the hypercube default)."""
    d = B.shape[1]
    def body(carry, xs):
        batch, c, k = xs
        pw, _, _ = _pops_batch_rows(model, cfg, batch, qs)
        proj = (pw @ B) * c[:, None]
        lo, hi = carry
        lo = jnp.minimum(lo, jnp.min(jnp.where(k[:, None], proj, jnp.inf), axis=0))
        hi = jnp.maximum(hi, jnp.max(jnp.where(k[:, None], proj, -jnp.inf), axis=0))
        return (lo, hi), None
    init = (jnp.full(d, jnp.inf), jnp.full(d, -jnp.inf))
    return jax.lax.scan(body, init, (ds, coef, keep))[0]


def pops_envelope_streamed(model, cfg, ds, coef, keep, Q, qs=(1.0, 1.0, 1.0)):
    """Member min / max of the prediction shift phi* . delta_i = (Q @ pw_i) * coef_i
    at test rows, with Q = phi* @ A (n, L).  Returns (lo, hi), each (n,)."""
    n = Q.shape[0]
    def body(carry, xs):
        batch, c, k = xs
        pw, _, _ = _pops_batch_rows(model, cfg, batch, qs)
        shift = (Q @ pw.T) * c[None, :]                       # (n, rows)
        lo, hi = carry
        lo = jnp.minimum(lo, jnp.min(jnp.where(k[None, :], shift, jnp.inf), axis=1))
        hi = jnp.maximum(hi, jnp.max(jnp.where(k[None, :], shift, -jnp.inf), axis=1))
        return (lo, hi), None
    init = (jnp.full(n, jnp.inf), jnp.full(n, -jnp.inf))
    return jax.lax.scan(body, init, (ds, coef, keep))[0]


# ---------------------------------------------------------------------------
# Host-cached POPS rows.  Every pass above recomputes the ACE design rows
# (linear_rows: the descriptor Jacobian) for the whole training set; a POPS fit is
# ~5 passes per ridge, so ridge selection over a 13-value grid re-ran the basis ~40
# times (2.8 h at the production Cantor basis, where one pass is ~4 min).  HostRows
# evaluates the rows ONCE, keeps the loss-weighted live rows in host RAM in
# fixed-size blocks, and every pass streams those blocks through small jitted
# kernels.  Memory: n_live_rows * L * 8 B on the host (~85 GB for Cantor-2560 at
# L = 27.6k), one block on the device.
# ---------------------------------------------------------------------------

def _pops_live_rows(model, cfg, batch, qs):
    pw, wy, _ = _pops_batch_rows(model, cfg, batch, qs)
    w = jnp.concatenate([batch.w_E, jnp.repeat(batch.w_F, 3), jnp.repeat(batch.w_V, 6)])
    return pw, wy, w


@jax.jit
def _k_gram(acc, pw, wy):
    M, b = acc
    return M + pw.T @ pw, b + pw.T @ wy


@jax.jit
def _k_leverage(pw, wy, c, A):
    return jnp.sum((pw @ A) * pw, axis=1), wy - pw @ c


@jax.jit
def _k_moments(acc, pw, c):
    W, s = acc
    pc = pw * c[:, None]
    return W + pc.T @ pc, s + pw.T @ c


@jax.jit
def _k_bounds(acc, pw, c, k, B):
    lo, hi = acc
    proj = (pw @ B) * c[:, None]
    return (jnp.minimum(lo, jnp.min(jnp.where(k[:, None], proj, jnp.inf), axis=0)),
            jnp.maximum(hi, jnp.max(jnp.where(k[:, None], proj, -jnp.inf), axis=0)))


@jax.jit
def _k_envelope(acc, pw, c, k, Q):
    lo, hi = acc
    shift = (Q @ pw.T) * c[None, :]
    return (jnp.minimum(lo, jnp.min(jnp.where(k[None, :], shift, jnp.inf), axis=1)),
            jnp.maximum(hi, jnp.max(jnp.where(k[None, :], shift, -jnp.inf), axis=1)))


class HostRows:
    """The POPS row passes over loss-weighted live rows cached in host RAM.

    ``blocks`` are (block, L) float64 host arrays of w*phi (the last one padded with
    zero rows, which have h = 0 and so count as padding everywhere), ``targets`` the
    matching w*y.  Per-row outputs (h, r, coef, keep) are (n_blocks, block) arrays."""

    def __init__(self, model, cfg, ds, qs=(1.0, 1.0, 1.0), block=16384):
        f = jax.jit(lambda b: _pops_live_rows(model, cfg, b, qs))
        self.block, self.blocks, self.targets, self.n_rows = block, [], [], 0
        buf = tbuf = None
        fill = 0
        for i in range(ds.n_batches):
            pw, wy, w = f(jax.tree.map(lambda a: a[i], ds))
            live = np.asarray(w) > 0
            pw, wy = np.asarray(pw)[live], np.asarray(wy)[live]
            self.n_rows += len(pw)
            j = 0
            while j < len(pw):
                if buf is None:
                    buf, tbuf, fill = np.zeros((block, pw.shape[1])), np.zeros(block), 0
                n = min(block - fill, len(pw) - j)
                buf[fill:fill + n], tbuf[fill:fill + n] = pw[j:j + n], wy[j:j + n]
                fill += n; j += n
                if fill == block:
                    self.blocks.append(buf); self.targets.append(tbuf); buf = None
        if buf is not None:
            self.blocks.append(buf); self.targets.append(tbuf)
        self.L = self.blocks[0].shape[1]

    def _stream(self):
        for pw, wy in zip(self.blocks, self.targets):
            yield jax.device_put(pw), jax.device_put(wy)

    def gram(self):
        """M = sum pw^T pw and b = sum pw^T wy (the POPS data Gram and moment)."""
        acc = (jnp.zeros((self.L, self.L)), jnp.zeros(self.L))
        for pw, wy in self._stream():
            acc = _k_gram(acc, pw, wy)
        return acc

    def leverage_residual(self, c_star, A):
        out = [_k_leverage(pw, wy, c_star, A) for pw, wy in self._stream()]
        return jnp.stack([h for h, _ in out]), jnp.stack([r for _, r in out])

    def moment_sums(self, coef):
        acc = (jnp.zeros((self.L, self.L)), jnp.zeros(self.L))
        for (pw, _), c in zip(self._stream(), coef):
            acc = _k_moments(acc, pw, c)
        return acc

    def projection_bounds(self, coef, keep, B):
        d = B.shape[1]
        acc = (jnp.full(d, jnp.inf), jnp.full(d, -jnp.inf))
        for (pw, _), c, k in zip(self._stream(), coef, keep):
            acc = _k_bounds(acc, pw, c, k, B)
        return acc

    def envelope(self, coef, keep, Q):
        n = Q.shape[0]
        acc = (jnp.full(n, jnp.inf), jnp.full(n, -jnp.inf))
        for (pw, _), c, k in zip(self._stream(), coef, keep):
            acc = _k_envelope(acc, pw, c, k, Q)
        return acc


class DeviceRows:
    """The same passes as HostRows, recomputing the rows inside each scan (no host
    memory; one ACE evaluation of the training set per pass)."""

    def __init__(self, model, cfg, ds, qs=(1.0, 1.0, 1.0)):
        self.model, self.cfg, self.ds, self.qs = model, cfg, ds, qs

    def leverage_residual(self, c_star, A):
        return pops_leverage_residual(self.model, self.cfg, self.ds, c_star, A, self.qs)

    def moment_sums(self, coef):
        return pops_moment_sums(self.model, self.cfg, self.ds, coef, self.qs)

    def projection_bounds(self, coef, keep, B):
        return pops_projection_bounds(self.model, self.cfg, self.ds, coef, keep, B, self.qs)

    def envelope(self, coef, keep, Q):
        return pops_envelope_streamed(self.model, self.cfg, self.ds, coef, keep, Q, self.qs)


def host_rows_bytes(ds, L):
    """Host RAM HostRows would take for ds at basis size L (live rows only)."""
    live = int(np.sum(np.asarray(ds.w_E) > 0) + 3 * np.sum(np.asarray(ds.w_F) > 0)
               + 6 * np.sum(np.asarray(ds.w_V) > 0))
    return live * L * 8


def available_host_bytes():
    """The memory limit this process runs under: the cgroup limit (containers) or
    physical RAM."""
    import os
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 1 << 60:
                return int(v)
        except OSError:
            pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
