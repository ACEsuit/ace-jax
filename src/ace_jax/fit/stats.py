"""Sufficient statistics of the data, per observation type, streamed over
batches.  Memory is one batch's rows plus the (Dt, Dt) accumulators; the scan's
reverse pass recomputes each batch under jax.checkpoint, which is the two-pass
gradient algorithm of the spec with no hand-written VJP."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rows import batch_rows


class Stats(NamedTuple):
    G_E: jnp.ndarray; G_F: jnp.ndarray; G_V: jnp.ndarray
    b_E: jnp.ndarray; b_F: jnp.ndarray; b_V: jnp.ndarray
    yy_E: jnp.ndarray; yy_F: jnp.ndarray; yy_V: jnp.ndarray
    n_E: jnp.ndarray; n_F: jnp.ndarray; n_V: jnp.ndarray
    logw_E: jnp.ndarray; logw_F: jnp.ndarray; logw_V: jnp.ndarray


def _type_stats(Phi, y, w):
    """Phi (n, Dt), y (n,), w (n,) structural weights (0 = absent)."""
    Pw = Phi * w[:, None]
    yw = y * w
    live = w > 0
    return (Pw.T @ Pw, Pw.T @ yw, yw @ yw, live.sum(),
            jnp.sum(jnp.where(live, 2.0 * jnp.log(jnp.where(live, w, 1.0)), 0.0)))


def batch_stats(theta, spec, model, ind, cfg, batch):
    r = batch_rows(theta, spec, model, ind, cfg, batch)
    Dt = r.E.shape[-1]
    E = _type_stats(r.E, batch.y_E, batch.w_E)
    F = _type_stats(r.F.reshape(-1, Dt), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3))
    V = _type_stats(r.V.reshape(-1, Dt), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))
    return Stats(E[0], F[0], V[0], E[1], F[1], V[1], E[2], F[2], V[2],
                 E[3], F[3], V[3], E[4], F[4], V[4])


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


def _linear_type_stats(Phi_B, y, w):
    """G_BB, b_B, yy, n, logw for one observation type from the linear rows."""
    Pw = Phi_B * w[:, None]
    yw = y * w
    live = w > 0
    return (Pw.T @ Pw, Pw.T @ yw, yw @ yw, live.sum(),
            jnp.sum(jnp.where(live, 2.0 * jnp.log(jnp.where(live, w, 1.0)), 0.0)))


def batch_linear_stats(model, cfg, batch):
    from .rows import linear_rows_bounded
    return linear_stats_from_rows(linear_rows_bounded(model, cfg, batch), batch)


def linear_stats_from_rows(r, batch):
    """batch_linear_stats from already-built linear rows r."""
    L = r.E.shape[-1]
    E = _linear_type_stats(r.E, batch.y_E, batch.w_E)
    F = _linear_type_stats(r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3))
    V = _linear_type_stats(r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))
    return Stats(E[0], F[0], V[0], E[1], F[1], V[1], E[2], F[2], V[2],
                 E[3], F[3], V[3], E[4], F[4], V[4])


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
    form (G_q, b_q) is derived on demand for the consumers that need it (POPS, ARD)."""
    R_E: jnp.ndarray; R_F: jnp.ndarray; R_V: jnp.ndarray
    c_E: jnp.ndarray; c_F: jnp.ndarray; c_V: jnp.ndarray
    yy_E: jnp.ndarray; yy_F: jnp.ndarray; yy_V: jnp.ndarray
    n_E: jnp.ndarray; n_F: jnp.ndarray; n_V: jnp.ndarray
    logw_E: jnp.ndarray; logw_F: jnp.ndarray; logw_V: jnp.ndarray

    G_E = property(lambda s: s.R_E.T @ s.R_E)
    G_F = property(lambda s: s.R_F.T @ s.R_F)
    G_V = property(lambda s: s.R_V.T @ s.R_V)
    b_E = property(lambda s: s.R_E.T @ s.c_E)
    b_F = property(lambda s: s.R_F.T @ s.c_F)
    b_V = property(lambda s: s.R_V.T @ s.c_V)

    def gram(self):
        """The equivalent Stats (Gram form), for consumers that need G and b."""
        return Stats(self.G_E, self.G_F, self.G_V, self.b_E, self.b_F, self.b_V,
                     self.yy_E, self.yy_F, self.yy_V, self.n_E, self.n_F, self.n_V,
                     self.logw_E, self.logw_F, self.logw_V)


# Batched merges.  A merge QR-factors the stacked [R c ; Pw yw], (L + r) x (L + 1): ~(4/3) L^3
# whatever r is, plus 2 r L^2 for the new rows (LAPACK geqrf does not exploit R's triangle).
# One merge per batch made that L^3 the whole cost at a large basis: Si o4d20 (L = 5477) has
# 788 packed 224-atom batches of E 117 / F 672 / V 702 rows, 3 merges each, 1.2e15 flops per
# pass with the old explicit Q (5e13 now).  So each quantity's live rows (padding and absent
# virials are zero rows, QR-neutral, and are dropped: E and V are ~2% live there) are buffered
# across batches and merged once at least merge_target (QR_MERGE_ROWS_PER_L x L) have
# accumulated, plus a final flush: per row, (4/3) L^3 / buffer + 2 L^2.  The target column
# rides along as an extra column, so only geqrf's R is formed (its last column is
# Q^T [c ; yw]): no Householder product, which used to double the flops.
QR_MERGE_ROWS_PER_L = 1


def merge_target(L, merge_rows=None):
    """Live rows that trigger a merge: merge_rows, default QR_MERGE_ROWS_PER_L * L; 0 merges
    every batch (that has live rows)."""
    return int(QR_MERGE_ROWS_PER_L * L if merge_rows is None else merge_rows)


def _live(Pw, yw):
    return jnp.any(Pw != 0, axis=1) | (yw != 0)


def _qr_merge(R, c, Pw, yw):
    """Fold weighted rows into a triangular factor: the updating QR of [R ; Pw], with c
    carried as an extra column so that only R is formed (no Q).  mode="raw" is geqrf, which JAX
    cannot differentiate: these statistics are not differentiable (the CPU host path never was)."""
    n = R.shape[1]
    A = jnp.concatenate([jnp.concatenate([R, c[:, None]], 1), jnp.concatenate([Pw, yw[:, None]], 1)], 0)
    Ra = jnp.triu(jnp.linalg.qr(A, mode="raw")[0].mT[:n])    # geqrf alone: no Householder product
    return Ra[:, :n], Ra[:, n]


def _host_merge(R, c, Pw, yw):
    """_qr_merge on the host: LAPACK geqrf (blocked: its workspace is queried) of the stacked
    Fortran-ordered matrix, in place; only R is read back."""
    import scipy.linalg.lapack as la
    n = R.shape[1]
    A = np.empty((n + Pw.shape[0], n + 1), order="F")
    A[:n, :n], A[:n, n], A[n:, :n], A[n:, n] = R, c, Pw, yw
    lwork = int(la.dgeqrf_lwork(*A.shape)[0])
    a, _, _, info = la.dgeqrf(A, lwork=max(lwork, n + 1), overwrite_a=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"dgeqrf info={info}")
    Ra = np.triu(a[:n])
    return Ra[:, :n].copy(), Ra[:, n].copy()


def host_qr_stream(batch_rows_fn, ds, init, merge_rows=None):
    """The updating QR of streamed rows, its merges on the host (SciPy, the calling thread).
    batch_rows_fn(batch) -> [(Pw, yw), ...] (jitted by the caller), one pair per factor;
    init = [(R0, c0), ...].  Returns [(R, c), ...] as jax arrays.  Each factor's live rows are
    buffered and merged once merge_target(L, merge_rows) have accumulated, then flushed at the
    end ('Batched merges' above); device_qr_stream merges at the same rows.

    Why not a lax.scan of jnp.linalg.qr on the CPU: XLA:CPU calls LAPACK (scipy's OpenBLAS)
    from its own worker threads, and OpenBLAS's multithreaded geqrf, called that way from
    inside a compiled loop, returned NaN on finite input -- deterministically, with 4 or 8
    OpenBLAS threads (a GitHub runner's default: 4 vCPUs) on a ~650-column basis, never with
    1-2 threads nor on the same matrices called directly.  Merges run here never did (tutorial 3,
    #61).  A device (GPU) backend keeps the scan: its QR is not OpenBLAS."""
    acc = [(np.asarray(R, np.float64), np.asarray(c, np.float64)) for R, c in init]
    buf, fill = [[] for _ in acc], [0] * len(acc)
    target = [max(merge_target(R.shape[1], merge_rows), 1) for R, _ in acc]

    def flush(k):
        if fill[k]:
            acc[k] = _host_merge(*acc[k], np.concatenate([p for p, _ in buf[k]]),
                                 np.concatenate([v for _, v in buf[k]]))
        buf[k], fill[k] = [], 0
    for i in range(ds.n_batches):
        parts = batch_rows_fn(jax.tree.map(lambda a: a[i], ds))
        for k, (Pw, yw) in enumerate(parts):
            Pw, yw = np.asarray(Pw, np.float64), np.asarray(yw, np.float64)
            keep = np.any(Pw != 0, axis=1) | (yw != 0)
            if not keep.all():
                Pw, yw = Pw[keep], yw[keep]
            buf[k].append((Pw, yw)); fill[k] += len(yw)
            if fill[k] >= target[k]:
                flush(k)
    for k in range(len(acc)):
        flush(k)
    return [(jnp.asarray(R), jnp.asarray(c)) for R, c in acc]


CPU_SCAN_MAX_L = 512     # above this, device_qr_stream on the CPU backend warns (#61, host_qr_stream)


def device_qr_stream(rows_fn, ds, init, extra0, merge_rows=None):
    """host_qr_stream as one lax.scan over the batches, for a device backend.  rows_fn(batch)
    -> ([(Pw, yw), ...], extra); extra (a pytree like extra0) is summed.  Shapes are static, so
    each factor carries a (target + r, L) row buffer and a fill count: a batch's r rows are
    compacted live-first (a stable sort) and written at the fill, which then advances by the
    live count only; a lax.cond merges the buffer (rows past the fill masked to zero) once the
    fill reaches the target, so at most target + r rows are ever pending.  The same merge
    points as host_qr_stream.  Returns ([(R, c), ...], extra).

    Not for the CPU backend beyond small L: XLA:CPU's geqrf inside the scan can return NaN with
    multithreaded OpenBLAS (#61; host_qr_stream says why), so callers route the CPU to the host
    and this warns when L > CPU_SCAN_MAX_L there."""
    if jax.default_backend() == "cpu" and max(jnp.shape(R)[1] for R, _ in init) > CPU_SCAN_MAX_L:
        import warnings
        warnings.warn("device_qr_stream on the CPU backend: XLA:CPU's geqrf in a scan can return NaN "
                      "with multithreaded OpenBLAS at this basis size (#61); use host_qr_stream", stacklevel=2)
    shapes = jax.eval_shape(rows_fn, jax.tree.map(lambda a: a[0], ds))[0]
    init = [(jnp.asarray(R), jnp.asarray(c)) for R, c in init]
    plan = [(max(merge_target(R.shape[1], merge_rows), 1), P.shape[0]) for (R, _), (P, _) in zip(init, shapes)]
    bufs0 = [(jnp.zeros((t + r, R.shape[1]), R.dtype), jnp.zeros(t + r, R.dtype), jnp.zeros((), jnp.int32))
             for (t, r), (R, _) in zip(plan, init)]

    def merge(R, c, BP, By, n):
        m = jnp.arange(BP.shape[0]) < n                         # rows past the fill: stale, masked
        return _qr_merge(R, c, jnp.where(m[:, None], BP, 0.0), jnp.where(m, By, 0.0))

    def body(carry, batch):
        fac, bufs, extra = carry
        parts, ex = rows_fn(batch)
        fac2, bufs2 = [], []
        for (t, _), (R, c), (BP, By, n), (Pw, yw) in zip(plan, fac, bufs, parts):
            live = _live(Pw, yw)
            o = jnp.argsort(~live, stable=True)                  # live rows first, in order
            BP = jax.lax.dynamic_update_slice(BP, Pw[o].astype(BP.dtype), (n, jnp.zeros((), jnp.int32)))
            By = jax.lax.dynamic_update_slice(By, yw[o].astype(By.dtype), (n,))
            n = n + live.sum(dtype=jnp.int32)
            R, c, n = jax.lax.cond(n >= t, lambda a: (*merge(*a), jnp.zeros((), jnp.int32)),
                                   lambda a: (a[0], a[1], a[4]), (R, c, BP, By, n))
            fac2.append((R, c)); bufs2.append((BP, By, n))
        return (fac2, bufs2, jax.tree.map(jnp.add, extra, ex)), None

    (fac, bufs, extra), _ = jax.lax.scan(body, (init, bufs0, extra0), ds)
    out = [jax.lax.cond(n > 0, lambda a: merge(*a), lambda a: (a[0], a[1]), (R, c, BP, By, n))
           for (R, c), (BP, By, n) in zip(fac, bufs)]                # the final flush
    return out, extra


def _qr_rows(model, cfg, batch):
    """One batch's weighted linear rows per quantity, and their yy, n, logw increments."""
    from .rows import linear_rows_bounded
    r = linear_rows_bounded(model, cfg, batch)
    L = cfg.len_basis
    out = []
    for P, y, w in ((r.E, batch.y_E, batch.w_E),
                    (r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3)),
                    (r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))):
        yw, live = y * w, w > 0
        out.append((P * w[:, None], yw, yw @ yw, live.sum(),
                    jnp.sum(jnp.where(live, 2.0 * jnp.log(jnp.where(live, w, 1.0)), 0.0))))
    return out


def linear_qr_statistics(model, cfg, ds, merge_rows=None, host=None):
    """linear_statistics in QR form (QRStats): one streaming pass over the same linear rows,
    folded into the per-quantity factors by batched updating QRs (QR_MERGE_ROWS_PER_L) -- on
    the host on the CPU backend (host_qr_stream says why), as a lax.scan on a device
    (device_qr_stream).  host: None picks by backend; merge_rows: see host_qr_stream (0 merges
    every batch)."""
    L = cfg.len_basis
    zeros = [(np.zeros((L, L)), np.zeros(L))] * 3
    if host is None:
        host = jax.default_backend() == "cpu"
    if host:
        rows = jax.jit(lambda b: _qr_rows(model, cfg, b))
        scal = {f"{k}_{q}": 0.0 for k in ("yy", "n", "logw") for q in "EFV"}

        def parts(batch):
            out = rows(batch)
            for q, (_, _, yy, n, lw) in zip("EFV", out):
                scal[f"yy_{q}"] += float(yy); scal[f"n_{q}"] += float(n); scal[f"logw_{q}"] += float(lw)
            return [(Pw, yw) for Pw, yw, *_ in out]
        (RE, cE), (RF, cF), (RV, cV) = host_qr_stream(parts, ds, zeros, merge_rows)
        return QRStats(RE, RF, RV, cE, cF, cV,
                       *(jnp.asarray(scal[f"{k}_{q}"]) for k in ("yy", "n", "logw") for q in "EFV"))

    def rows_fn(batch):
        out = _qr_rows(model, cfg, batch)
        return [(Pw, yw) for Pw, yw, *_ in out], [o[2:] for o in out]
    z0 = jnp.zeros(())
    run = jax.jit(lambda d: device_qr_stream(rows_fn, d, zeros, [(z0, z0, z0)] * 3, merge_rows))
    ((RE, cE), (RF, cF), (RV, cV)), ex = run(ds)
    return QRStats(RE, RF, RV, cE, cF, cV, *(ex[i][k] for k in range(3) for i in range(3)))


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
    from .pops import pointwise_corrections, whiten
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
    from .pops import leverage_select
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
