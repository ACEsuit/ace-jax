"""Native POPS misspecification UQ for the linear (M=0) arm, following Swinburne
& Perez (arXiv:2402.01810): pointwise-optimal parameter sets built on the design
weighted by the STRUCTURAL weights only (the intrinsic noise is fixed, never fitted
to the residual).  The predict-side driver is ``predict.PopsRidgePath``."""
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


def pops_posterior(deltas, c_star, form="hypercube"):
    """Build the POPS misspecification posterior from pointwise ``deltas``.

    Two forms (both ported from ``popsregression._build_posterior``; the package
    calls the committee form 'ensemble', which we match):

    * ``form="hypercube"`` (DEFAULT): the PCA/axis-aligned box over ``deltas``
      reduced to a covariance (the package's ``misspecification_sigma_``).
      Returns ``{"cov": (L, L)}``.
    * ``form="ensemble"``: the committee of weight samples ``c_star + deltas``
      (the package's ``posterior_samples_``, stored as full weight vectors).
      Returns ``{"ensemble": (K, L)}``.

    ("samples" is reserved for a possible future route that draws Gaussian
    samples from the hypercube covariance itself.)

    The ensemble variance is centred across the committee; the hypercube is an
    uncentred second moment of a uniform box spanning the full min..max of the
    PCA-projected corrections.  On ACE designs the corrections are heavy-tailed
    (projected kurtosis ~10), so the box is much wider than the committee spread;
    the mean term (phi*.mean(delta))^2 is negligible by comparison.

    Both are Gaussian summaries.  The object of the paper's coverage theorem is the
    member ENVELOPE (:func:`pops_envelope`): the min/max over members brackets
    every training observation as N/P -> inf.  No separate noise term belongs in
    the predictive -- fitting the noise to the residual "recovers standard
    maximum-likelihood inference, which ignores misspecification" (the paper), so
    the members are built with structural weights and a fixed ridge
    (``predict.PopsRidgePath``)."""
    if form == "ensemble":
        return {"ensemble": c_star[None, :] + deltas}
    elif form == "hypercube":
        return {"cov": _fit_hypercube_cov(deltas)}
    raise ValueError(f"unknown POPS posterior form: {form!r}")


def pops_var(phi_star, posterior):
    """Predictive misspecification variance at test rows ``phi_star`` (n, L).

    * ensemble form: the variance across the committee of ``phi* . c_k``.
    * cov form: the quadratic form ``sum((phi* @ cov) * phi*, axis=1)``.
    """
    if "ensemble" in posterior:
        ensemble = posterior["ensemble"]       # (K, L)
        preds = phi_star @ ensemble.T          # (n, K)
        return jnp.var(preds, axis=1)
    elif "cov" in posterior:
        cov = posterior["cov"]
        return jnp.sum((phi_star @ cov) * phi_star, axis=1)
    raise ValueError("posterior must contain 'ensemble' or 'cov'")


def pops_envelope(phi_star, deltas, chunk=1024):
    """Pointwise-optimal ENVELOPE at rows ``phi_star`` (n, L): the min and max of
    the prediction shift ``phi* . delta_i`` over every member ``delta_i`` (K, L).

    This is the object of the Swinburne-Perez coverage theorem (the ensemble
    min/max must bracket every training observation as N/P -> inf), as opposed
    to the Gaussian second-moment summaries of :func:`pops_var`.  Evaluated in
    chunks of ``chunk`` rows so the (n, K) shift matrix is never materialised.
    Returns ``(lo, hi)``, each (n,), as shifts relative to the base prediction."""
    lo, hi = [], []
    for s in range(0, phi_star.shape[0], chunk):
        shift = phi_star[s:s + chunk] @ deltas.T            # (chunk, K)
        lo.append(jnp.min(shift, axis=1)); hi.append(jnp.max(shift, axis=1))
    return jnp.concatenate(lo), jnp.concatenate(hi)
