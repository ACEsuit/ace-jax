"""Native POPS misspecification UQ for the linear (M=0) arm, on the design
whitened per quantity by w/sigma_q so E and F share one homoscedastic scale."""
import jax.numpy as jnp


def whiten(phi, resid, w, sigma):
    """Whiten rows of a design ``phi`` and residual ``resid`` by ``w/sigma``.

    ``phi`` is (n, D), ``resid`` and ``w`` are (n,), ``sigma`` is a scalar (or
    broadcastable to (n,)) per-quantity noise scale. Returns
    ``((w/sigma) * phi, (w/sigma) * resid)`` with the scale broadcast over
    phi's rows.
    """
    s = w / sigma
    return phi * s[:, None], resid * s


def pointwise_corrections(Sigma0, phi_t, r_t):
    """Per-point corrections ``delta_i`` and leverages ``h_i`` on a WHITENED
    design, WITHOUT any leverage subselection.

    This is the jit-safe core of :func:`pops_corrections`: it contains no
    boolean masking, so it can run inside a ``jax.lax.scan`` body (Task 9's
    streamed :func:`ace_jax.fit.stats.pops_statistics`).  The correction for
    training point ``i`` is the minimum-norm Newton step (under the ``A``
    metric) that makes the model fit point ``i`` exactly,
    ``phi_i . (c + delta_i) = y_i``::

        delta_i = Sigma0 @ phi_i * (r_i / h_i),   h_i = phi_i . Sigma0 . phi_i

    Dividing by the (whitened) leverage ``h_i`` is what makes
    ``phi_i . delta_i == r_i`` hold exactly.  Zero-leverage rows (padded /
    structurally-absent observations, ``w = 0``) get ``delta_i = 0`` via the
    ``h_i = inf`` guard.

    Returns ``(deltas (n, L), h (n,))``.
    """
    pc = phi_t @ Sigma0                       # (n, L): rows phi_i . Sigma0
    h = jnp.sum(pc * phi_t, axis=1)           # leverage h_i
    safe_h = jnp.where(h > 0, h, jnp.inf)     # guard zero-leverage points
    deltas = pc * (r_t / safe_h)[:, None]     # delta_i = Sigma0 phi_i r_i / h_i
    return deltas, h


def leverage_select(deltas, h, leverage_pct):
    """Keep only the rows of ``deltas`` whose leverage ``h`` is at or above the
    ``leverage_pct`` percentile (the package's ``leverage_percentile``, for
    tractability at scale).  If that mask is empty, all rows are kept.

    Uses an EAGER ``bool(jnp.any(...))`` fallback, so it is NOT jit-safe and is
    applied ONCE, outside any scan (Task 9's resolution 1).
    """
    thresh = jnp.percentile(h, leverage_pct)
    mask = h >= thresh
    if not bool(jnp.any(mask)):               # fallback: keep everything
        mask = jnp.ones_like(h, dtype=bool)
    return deltas[mask]


def pops_corrections(Sigma0, phi_t, r_t, leverage_pct):
    """Per-training-point "pointwise optimal parameter" corrections.

    Ported from ``popsregression.POPSRegression.fit``. On the WHITENED design
    (``phi_t``, ``r_t`` already whitened by Task 7's :func:`whiten`, so the
    homoscedastic noise scale is 1 and ``Sigma0 = A^{-1}`` needs no ``alpha_``
    rescaling), the correction for training point ``i`` is the minimum-norm
    Newton step (under the ``A`` metric) that makes the model fit point ``i``
    exactly, ``phi_i . (c + delta_i) = y_i``::

        delta_i = Sigma0 @ phi_i * (r_i / h_i),   h_i = phi_i . Sigma0 . phi_i

    where ``h_i`` is the (whitened) leverage. Dividing by ``h_i`` is what makes
    ``phi_i . delta_i == r_i`` hold exactly (the package does this via
    ``pointwise_correction *= errors / safe_leverage``); it is retained here
    even though the brief's inline formula wrote only ``Sigma0 @ phi_i * r_i``.

    Only points whose leverage ``h_i`` is at or above the ``leverage_pct``
    percentile are kept (the package's ``leverage_percentile``, for
    tractability at scale). If that mask is empty, all points are kept.

    Parameters
    ----------
    Sigma0 : (L, L) array   epistemic weight covariance ``A^{-1}``.
    phi_t  : (n, L) array   whitened design rows.
    r_t    : (n,)   array   whitened residuals ``y - phi c``.
    leverage_pct : float    percentile in [0, 100]; 0 keeps every point.

    Returns
    -------
    deltas : (K, L) array    corrections for the K retained points.
    """
    deltas, h = pointwise_corrections(Sigma0, phi_t, r_t)
    return leverage_select(deltas, h, leverage_pct)


def _fit_hypercube_cov(deltas, mode_threshold=1.0e-8, percentile_clipping=0.0):
    """Fit the POPS "hypercube" (PCA axis-aligned box) to ``deltas`` and return
    its misspecification covariance ``(L, L)``.

    Ported from ``popsregression`` ``_fit_hypercube`` + ``_sample_hypercube``.
    The package fits a box in the PCA basis of ``deltas.T @ deltas`` (keeping
    modes above ``mode_threshold * max_eigval``), with per-axis bounds from the
    ``percentile_clipping`` / ``100 - percentile_clipping`` percentiles of the
    projected corrections, then draws a QMC uniform sample from the box and
    reports ``S @ S.T / n`` (the second moment about the origin).

    Here we use the exact large-sample expectation of that estimator instead of
    a finite random draw: a coordinate uniform on ``[low_j, high_j]`` has second
    moment ``(high-low)^2 / 12 + m_j^2`` (with midpoint ``m_j``) on its own axis
    and ``m_j m_k`` off-axis, so::

        cov = support @ (diag((high-low)^2 / 12) + m m^T) @ support.T

    This is deterministic, fp64, and JAX-friendly, and equals the package's
    sampled ``misspecification_sigma_`` as ``n_resample -> inf``.
    """
    gram = deltas.T @ deltas                  # (L, L)
    evals, evecs = jnp.linalg.eigh(gram)
    keep = evals > mode_threshold * jnp.max(evals)
    support = evecs[:, keep]                  # (L, d) principal directions

    projected = deltas @ support              # (K, d)
    low = jnp.percentile(projected, percentile_clipping, axis=0)
    high = jnp.percentile(projected, 100.0 - percentile_clipping, axis=0)

    m = 0.5 * (low + high)                     # per-axis box midpoint
    var_axis = (high - low) ** 2 / 12.0        # per-axis uniform variance
    second_moment = jnp.diag(var_axis) + jnp.outer(m, m)   # E[u u^T]
    return support @ second_moment @ support.T


def pops_posterior(deltas, c_star, form="samples"):
    """Build the POPS misspecification posterior from pointwise ``deltas``.

    Two forms (both ported from ``popsregression._build_posterior``):

    * ``form="samples"`` (DEFAULT): the committee of weight samples
      ``c_star + deltas`` (the package's ``posterior_samples_``, here stored as
      full weight vectors rather than bare perturbations). Returns
      ``{"samples": (K, L)}``.
    * ``form="hypercube"``: the PCA/axis-aligned box over ``deltas`` reduced to
      a covariance (the package's ``misspecification_sigma_``). Returns
      ``{"cov": (L, L)}``.

    NOTE: ``"samples"`` is a CENTERED committee variance (``pops_var`` takes
    ``jnp.var`` of ``phi* . (c_star + deltas)`` across the committee), whereas
    ``"hypercube"`` is an UNCENTERED second moment about the origin (see
    ``_fit_hypercube_cov``'s ``E[u u^T] = diag(var_axis) + outer(m, m)``, not
    ``E[(u - E[u])(u - E[u])^T]``). The two forms therefore differ by
    ``(phi* . mean(deltas))**2`` whenever the pointwise corrections ``deltas``
    have nonzero mean.

    KNOWN LIMITATION — bias/variance conflation (not yet implemented: a clean
    third form).  In BOTH forms the *mean* prediction stays at ``phi* . c_star``
    (see ``predict._pops_predict_batch``): only the variance changes.  When the
    retained ``deltas`` have a nonzero mean ``mbar`` — i.e. the pointwise-optimal
    parameter sets systematically pull ``c_star`` in one direction, the signature
    of a *biased* fit (missing physics), not of scatter — the two forms are:

      * ``samples``:  mean ``phi* . c_star``,  var ``Var_i[phi* . delta_i]``
        (centred on the committee mean).  Drops the systematic term entirely, so
        it under-reports and is overconfident (measured rms-z ~12 on SiGe E).
      * ``hypercube``: mean ``phi* . c_star``,  var ``E_i[(phi* . delta_i)**2]``
        = ``Var_i + (phi* . mbar)**2``.  This is a mean-squared-error ABOUT THE
        RAW FIT, not a variance: it folds ``bias**2`` into ``sigma**2`` while
        leaving the mean at the (knowingly biased) ``phi* . c_star``.  It
        restores *symmetric* coverage (rms-z, Gaussian CRPS) only because the
        interval is widened by exactly enough to straddle the bias; it would
        mislead any sign-sensitive / asymmetric downstream use.

    The statistically clean object is a THIRD form we do not implement:
    bias-correct the mean to ``phi* . (c_star + mbar)`` AND report the centred
    ``Var_i[phi* . delta_i]``.  That separates a mean shift (which belongs in the
    prediction) from spread (which belongs in the variance).  It is a strictly
    better experiment than either form above and is cheap to try:

      * measure BOTH rms-z AND energy RMSE.  If ``mbar`` is a genuine model bias,
        moving the mean should *reduce* RMSE — POPS then improves the fit, not
        just the UQ.
      * CAVEAT: ``mbar`` is taken over the LEVERAGE-SELECTED (high-influence)
        subset — exactly the points most prone to overfitting — so the shift may
        be an overfit direction, not a true bias.  RMSE is the discriminator:
        RMSE down => real bias worth correcting; RMSE up => the "bias" was
        leverage-selection noise, and centred variance + aleatoric is the honest
        report (which is why native POPS did not beat BLR + post-hoc scalar on E).

    This is faithful to the POPS package's definition (``misspecification_sigma_``
    is the uncentred ``E[u u^T]`` by design), so it is a design choice inherited
    from upstream, not a port bug.

    EMPIRICAL NOTE (2026-09-23, spike/pops_1d.py, reference popsregression package).
    A 1D misspecified fit tempers the story above: the mean term (phi.mbar)^2 is
    typically SMALL (a few % of the misspecification variance), so 'samples' vs
    'hypercube' differ only modestly (hypercube a bit better).  The DOMINANT
    calibration factor is having the misspecification/aleatoric variance at all --
    the ordinary posterior variance alone is overconfident by up to ~100x rms-z
    under strong misspecification, and the misspecification term (both forms) is
    what fixes it.  So most of the SiGe/Cantor "samples 19 vs hypercube+alea 1"
    gap was the aleatoric flag (samples-run misspec-thin, hcube-run carried it),
    not centred-vs-uncentred.  The ~5x samples-vs-hypercube gap our port showed
    (aleatoric-off) is NOT reproduced on the benign 1D case, so it is either
    ACE-regime-specific (whitened multi-quantity design inflating mean(delta)) or
    a port quirk -- an open item to pin down directly.
    """
    if form == "samples":
        return {"samples": c_star[None, :] + deltas}
    elif form == "hypercube":
        return {"cov": _fit_hypercube_cov(deltas)}
    raise ValueError(f"unknown POPS posterior form: {form!r}")


def pops_var(phi_star, posterior):
    """Predictive misspecification variance at test rows ``phi_star`` (n, L).

    * samples form: the variance across the committee of ``phi* . c_k``.
    * cov form: the quadratic form ``sum((phi* @ cov) * phi*, axis=1)``.
    """
    if "samples" in posterior:
        samples = posterior["samples"]         # (K, L)
        preds = phi_star @ samples.T           # (n, K)
        return jnp.var(preds, axis=1)
    elif "cov" in posterior:
        cov = posterior["cov"]
        return jnp.sum((phi_star @ cov) * phi_star, axis=1)
    raise ValueError("posterior must contain 'samples' or 'cov'")
