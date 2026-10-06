"""Predictive mean and variance of E/F/V at fixed theta (spec, "Prediction"),
and the Gaussian mixture over hyperparameter draws (spec, "Hyperposterior
uncertainty propagation").

E_var includes the DTC prior residual k** - q** of the residual GP (summed over
the live site pairs of each configuration); F_var/V_var are SoR (the residual
term needs second kernel derivatives) -- Ruling R29."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .hypers import Hypers, from_array, to_array
from .kernels import K_MM, k_rows
from .summary import site_summary
from .feature import apply as _feat, dwarp
from .objective import posterior, posterior_from_qr, uses_qr
from .data import VOIGT, flat_edges
from .metrics import crps_gaussian
from .pops import leverage_select, pops_var
from .rows import _cat_rows, batch_rows_parts, linear_rows_bounded, residual_inputs
from .stats import (DeviceRows, HostRows, _stream_pops_pointwise, available_host_bytes, host_rows_bytes,
                    sufficient_statistics)


class Prediction(NamedTuple):
    """E_var includes the DTC prior residual k** - q** of the residual GP;
    F_var/V_var are SoR (the residual term needs second kernel derivatives)
    -- Ruling R29."""
    E_mean: np.ndarray; E_var: np.ndarray
    F_mean: np.ndarray; F_var: np.ndarray
    V_mean: np.ndarray; V_var: np.ndarray


def _rows_mean_var(Phi, mu, L):
    """Phi (n, Dt) -> mean Phi mu, var diag(Phi S^-1 Phi^T) with L L^T = S."""
    v = solve_triangular(L, Phi.T, lower=True)            # (Dt, n)
    return Phi @ mu, jnp.sum(v * v, axis=0)


def _dtc_energy_residual(theta, prob, batch, X):
    """DTC prior residual of the energy rows, per configuration a of the batch:
    r_a = sum_{i,j in a, live} [k(x_i, x_j) - k_iM K_MM^-1 k_Mj]  (>= 0: Kxx - Q
    is the Schur complement of a PSD matrix).  M == 0 is the BLR limit -- the
    residual GP is switched OFF, not a GP with no inducing points -- so the term
    is zero (otherwise the unidentified kernel amplitude's prior would leak
    into the energy variance of the linear model).
    X are the compact site descriptors from linear_rows."""
    spec, ind, C = prob.spec, prob.ind, batch.y_E.shape[0]
    if ind.XM.shape[0] == 0:                                    # static shape
        return jnp.zeros(C)
    x, s, _ = residual_inputs(ind, prob.cfg, batch, X)
    z = batch.node_z
    R = k_rows(theta, spec, x, s, z, x, s, z, ind.embed)                   # (Ncap, Ncap)
    if ind.XM.shape[0] > 0:                                     # static: no Cholesky of a (0, 0)
        KxM = k_rows(theta, spec, x, s, z, ind.XM, ind.SM, ind.ZM, ind.embed)          # (Ncap, M)
        L_MM = jnp.linalg.cholesky(K_MM(theta, spec, ind.XM, ind.SM, ind.ZM, ind.embed))
        A = solve_triangular(L_MM, KxM.T, lower=True)                       # (M, Ncap)
        R = R - A.T @ A
    m = batch.node_mask
    R = jnp.where(m[:, None] & m[None, :], R, 0.0)
    seg = lambda a: jax.ops.segment_sum(a, batch.node_cfg, num_segments=C + 1)
    return jnp.diag(seg(seg(R).T))[:C]


def _dtc_D(theta, prob, batch, deL, deR):
    """Two-field DTC residual covariance of the residual GP, per configuration c:

        D_c = sum_{i,j in c} k(U_i^L, U_j^R) - (sum_i k^L_iM) K_MM^-1 (sum_j k^R_Mj)

    where the left sites use edges rij + deL and the right sites rij + deR
    (deL, deR are per-edge displacements, (E, 3)).  deL = deR = 0 is the energy
    residual (== _dtc_energy_residual).  Matched atom-displacement or strain
    perturbations in deL and deR give, by mixed second derivative, the
    force / virial DTC residual -- so F_var, V_var are the position derivative
    of E_var (Ruling R29 lifted: the DTC term is now carried to the gradients).
    Built on the same K_MM Cholesky, so the -Q subtraction is exact."""
    spec, ind, cfg = prob.spec, prob.ind, prob.cfg
    C = batch.y_E.shape[0]
    Ncap = batch.nbr.shape[0]
    rij, send, recv, m = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    z = batch.node_z

    def field(de):
        r = rij + de
        X = prob.model.compact_basis(r, z[send], z[recv], send, Ncap, m)     # (Ncap, D)
        U = _feat(X, ind.Pmap, ind.warp)                                     # (Ncap, d)
        s = site_summary(r, send, Ncap, cfg.r0, cfg.rcut, cfg.p, m)
        return U, s

    UL, sL = field(deL)
    UR, sR = field(deR)
    Kfull = k_rows(theta, spec, UL, sL, z, UR, sR, z, ind.embed)                        # (Ncap, Ncap)
    KLM = k_rows(theta, spec, UL, sL, z, ind.XM, ind.SM, ind.ZM, ind.embed)            # (Ncap, M)
    KRM = k_rows(theta, spec, UR, sR, z, ind.XM, ind.SM, ind.ZM, ind.embed)
    L_MM = jnp.linalg.cholesky(K_MM(theta, spec, ind.XM, ind.SM, ind.ZM, ind.embed))
    AL = solve_triangular(L_MM, KLM.T, lower=True)                           # (M, Ncap)
    AR = solve_triangular(L_MM, KRM.T, lower=True)
    R = Kfull - AL.T @ AR
    mnode = batch.node_mask
    R = jnp.where(mnode[:, None] & mnode[None, :], R, 0.0)
    seg = lambda a: jax.ops.segment_sum(a, batch.node_cfg, num_segments=C + 1)
    return jnp.diag(seg(seg(R).T))[:C]


_DERIV_DTC_WARNED = False


def _warn_deriv_dtc_size(Ncap, K, d):
    """One-time warning when the derivative DTC's whole-batch (Ncap, K, d, 3) arrays (JU, the
    slot velocities vU, the strain velocities cbU -- not node-chunked: a node's velocity gathers
    the edges INTO it, which live in other nodes' chunks) exceed rows.ROWS_EDGE_BUDGET.  They are
    built anyway, as before the rows were chunked; this only says why memory may run out."""
    global _DERIV_DTC_WARNED
    from . import rows as _rows
    n = Ncap * K * d * 3
    if _DERIV_DTC_WARNED or n <= _rows.ROWS_EDGE_BUDGET:
        return
    _DERIV_DTC_WARNED = True
    import warnings
    warnings.warn(
        f"derivative-DTC force/virial variance on a batch of n_cap={Ncap}, k_cap={K} with a "
        f"{d}-wide residual feature map builds whole-batch (n_cap, k_cap, d, 3) arrays of "
        f"{n * 8 / 1e9:.3g} GB each (several are live at once); they are not node-chunked, so "
        f"ACEJAX_ROWS_EDGE_BUDGET does not bound them.  If memory runs out: deriv_dtc=False "
        f"(SoR-only F/V variance; GPCalculator(..., deriv_dtc=False), `aj eval --no-deriv-dtc`, "
        f"predict_fixed/predict_mixture(deriv_dtc=False)), or fit with a narrow feature map "
        f"(--density pair / pca).", stacklevel=3)


DERIV_DTC_NODE_BATCH = 4    # nodes per vmapped chunk of the force double jvp: bounds its temp


def _dtc_deriv_residual(theta, prob, batch, X=None, J=None, res=None, JU0=None):
    """Force and virial DTC prior residuals -- the position/strain derivative of
    the energy DTC residual, so F_var, V_var are consistent with E_var (Ruling
    R30).  For a linear functional o of the residual field,
        Var_res(o) = <o, o>_k - K_oM K_MM^-1 K_Mo   (a Schur complement, >= 0).
    For a force/virial (a derivative functional) K_oM is the residual force/virial
    row (res.F / res.V, already built), so the Q-term is cheap; <o, o>_k is a
    second kernel derivative got by a double jvp of the config-summed Gram along
    the analytic feature/summary velocities dU_i/dr, dU_i/deps -- no autodiff
    through the ACE basis, so it is cheap in the low-d density feature and does
    not build a third-order tape (the naive jacfwd(jacrev) OOMs at scale).
    JU0 = Pmap^T J (Ncap, K, d, 3) may stand in for J (the node-chunked rows path never forms J).
    Returns Fv (Ncap, 3) and Vv (C, 6).  M == 0 (BLR limit) -> zeros."""
    spec, ind, cfg = prob.spec, prob.ind, prob.cfg
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]
    # cosine-SE is twice differentiable at coincidence; matern32's _TINY-floored
    # RMS-sqrt is not, and corrupts the self-pair second derivative (Ruling R30).
    if spec.kind != "cosine":
        raise ValueError(
            f"derivative-DTC (deriv_dtc=True) requires the cosine-SE kernel; "
            f"kind={spec.kind!r} is not twice differentiable at coincidence. "
            f"Pass deriv_dtc=False for SoR-only force/virial variance.")
    if ind.XM.shape[0] == 0:
        return jnp.zeros((Ncap, 3)), jnp.zeros((C, 6))
    _warn_deriv_dtc_size(Ncap, K, ind.Pmap.shape[1])
    if X is None or (J is None and JU0 is None) or res is None:
        _, res, X, JU0 = batch_rows_parts(theta, spec, prob.model, ind, cfg, batch,
                                          with_X=True, with_JU0=True)
    if JU0 is None:
        JU0 = jnp.einsum("nkDa,Dq->nkqa", J.reshape(Ncap, K, cfg.D, 3), ind.Pmap)
    d, M = ind.XM.shape[1], ind.XM.shape[0]
    rij, send, recv, m = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    z = batch.node_z
    U, s, Js = residual_inputs(ind, cfg, batch, X)                        # U (Ncap,d), Js (E,3)
    dw = dwarp(X @ ind.Pmap, ind.warp)                                    # (Ncap, d)
    JU = dw[:, None, :, None] * JU0                                       # (Ncap,K,d,3)
    Jsd = Js.reshape(Ncap, K, 3)
    ar = jnp.arange(Ncap)

    # feature/summary velocities.  rij = r_recv - r_send, so moving node n moves
    # n itself (dU_n/dr_n = -sum_k JU[n,k]) and every site that lists n as a
    # neighbour, i.e. the sender of each edge into n (+JU of that edge).  The
    # velocity vanishes elsewhere, so <o, o>_k needs only those K+1 slots per node,
    # not the whole-batch Gram (3 Ncap velocities x Ncap^2 pairs was cubic in the
    # batch).  A site repeated across slots (periodic images) is exact: the double
    # jvp is bilinear in the slot velocities.  The cutoff list is symmetric and
    # dense_graph raises rather than truncates, so a node has <= K in-edges (_batch
    # asserts it).
    JU_e = JU.reshape(Ncap * K, d, 3); Js_e = Jsd.reshape(Ncap * K, 3)
    key = jnp.where(m, recv, Ncap)
    order = jnp.argsort(key, stable=True)
    slot = jnp.searchsorted(key[order], ar)[:, None] + jnp.arange(K)[None, :]      # (Ncap, K)
    e_in = order[jnp.minimum(slot, Ncap * K - 1)]
    ok = (slot < Ncap * K) & (key[e_in] == ar[:, None])
    sites = jnp.concatenate([ar[:, None], jnp.where(ok, send[e_in], ar[:, None])], 1)          # (Ncap, K+1)
    vU = jnp.concatenate([-JU.sum(1)[:, None], jnp.where(ok[..., None, None], JU_e[e_in], 0.0)], 1)
    vS = jnp.concatenate([-Jsd.sum(1)[:, None], jnp.where(ok[..., None], Js_e[e_in], 0.0)], 1)
    # strain generator W_v(rij) = 1/2 (e_a rij_b + e_b rij_a); dU_i/deps_v via i's edges.
    # A config's strain moves only its own sites, so one velocity per Voigt component
    # serves every config at once against a same-config-masked Gram.
    Wv = jnp.stack([(jnp.zeros((rij.shape[0], 3)).at[:, a].add(0.5 * rij[:, b])
                                                  .at[:, b].add(0.5 * rij[:, a])) for a, b in VOIGT], 1)  # (E,6,3)
    cbU = jnp.where(m[None, :, None], jnp.einsum("eqa,eva->veq", JU_e, Wv), 0.0)   # (6,E,d)
    cbS = jnp.where(m[None], jnp.einsum("ea,eva->ve", Js_e, Wv), 0.0)              # (6,E)
    bU = jnp.zeros((6, Ncap, d)).at[:, send, :].add(cbU)
    bS = jnp.zeros((6, Ncap)).at[:, send].add(cbS)

    # <o, o>_k = d_beta d_gamma sum_{i,j} k(U_i+beta v_i, U_j+gamma v_j) along the
    # matched velocity v (in both the U and the summary s slots), via a double jvp.
    def ffk(Ksum, x, v):
        gy = lambda y: jax.jvp(lambda x: Ksum(x, y), (x,), (v,))[1]
        return jax.jvp(gy, (x,), (v,))[1]
    def node(a):                                          # one node's 3 force components
        site, vU_n, vS_n = a
        x, zx = (U[site], s[site]), z[site]
        Ksum = lambda x, y: jnp.sum(k_rows(theta, spec, *x, zx, *y, zx, ind.embed))
        return jax.vmap(lambda vu, vs: ffk(Ksum, x, (vu, vs)))(jnp.moveaxis(vU_n, -1, 0), vS_n.T)
    FFk = jax.lax.map(node, (sites, vU, vS), batch_size=DERIV_DTC_NODE_BATCH)   # (Ncap, 3)
    same = (batch.node_mask[:, None] & batch.node_mask[None, :]
            & (batch.node_cfg[:, None] == batch.node_cfg[None, :]))
    def Kcfg(x, y):                                       # per-config sums of same-config pairs
        G = jnp.where(same, k_rows(theta, spec, *x, z, *y, z, ind.embed), 0.0)
        return jax.ops.segment_sum(G.sum(1), batch.node_cfg, num_segments=C + 1)[:C]
    VVk = jax.lax.map(lambda v: ffk(Kcfg, (U, s), v), (bU, bS)).T                 # (C, 6)

    # Q-term K_oM K_MM^-1 K_Mo from the residual force/virial rows
    L_MM = jnp.linalg.cholesky(K_MM(theta, spec, ind.XM, ind.SM, ind.ZM, ind.embed))
    vF = solve_triangular(L_MM, res.F.reshape(Ncap * 3, M).T, lower=True)
    vV = solve_triangular(L_MM, res.V.reshape(C * 6, M).T, lower=True)
    Fv = jnp.where(batch.node_mask[:, None], jnp.maximum(FFk - jnp.sum(vF * vF, 0).reshape(Ncap, 3), 0.0), 0.0)
    Vv = jnp.maximum(VVk - jnp.sum(vV * vV, 0).reshape(C, 6), 0.0)
    return Fv, Vv


def _predict_batch(theta, prob, mu, L, batch, dtc=True, deriv_dtc=True):
    want_ju = dtc and deriv_dtc and prob.ind.XM.shape[0] > 0
    lin, res, X, JU0 = batch_rows_parts(theta, prob.spec, prob.model, prob.ind, prob.cfg, batch,
                                        with_X=True, with_JU0=want_ju)   # as rows.batch_rows, keeping X
    r = _cat_rows(lin, res)
    Dt = r.E.shape[-1]
    Em, Ev = _rows_mean_var(r.E, mu, L)
    if dtc:
        Ev = Ev + _dtc_energy_residual(theta, prob, batch, X)
    Fm, Fv = _rows_mean_var(r.F.reshape(-1, Dt), mu, L)
    Vm, Vv = _rows_mean_var(r.V.reshape(-1, Dt), mu, L)
    Fv, Vv = Fv.reshape(-1, 3), Vv.reshape(-1, 6)
    if dtc and deriv_dtc:                                    # F_var, V_var = d E_var (self-consistent)
        dFv, dVv = _dtc_deriv_residual(theta, prob, batch, X, res=res, JU0=JU0)
        Fv, Vv = Fv + dFv, Vv + dVv
    return Em, Ev, Fm.reshape(-1, 3), Fv, Vm.reshape(-1, 6), Vv


def _e0_offset(prob, ds):
    E0 = prob.model.E0
    per_node = jnp.where(ds.node_mask, E0[ds.node_z], 0.0)
    C = ds.y_E.shape[1]
    return jax.vmap(lambda e, c: jax.ops.segment_sum(e, c, num_segments=C + 1)[:C])(per_node, ds.node_cfg)


def _predict_fn(prob, dtc, deriv_dtc):
    """The jitted per-batch predictor with theta/mu/L as ARGUMENTS (not closures)
    so it compiles ONCE and is reused across posterior draws -- predict_mixture
    otherwise recompiled the (expensive) derivative-DTC once per draw."""
    return jax.jit(lambda th, mu, L, b: _predict_batch(th, prob, mu, L, b, dtc, deriv_dtc))


def _pack(outs, prob, ds_test):
    """Stack the per-batch (E, F, V) mean/var tuples, add the E0 offset to the
    energy mean, and drop the padded configs/nodes (shared by the BLR and POPS
    predictive paths)."""
    Em = jnp.stack([o[0] for o in outs]) + _e0_offset(prob, ds_test)     # (nb, C)
    cm, nm = np.asarray(ds_test.cfg_mask).reshape(-1), np.asarray(ds_test.node_mask).reshape(-1)
    cat = lambda k: np.concatenate([np.asarray(o[k]) for o in outs])
    return Prediction(np.asarray(Em).reshape(-1)[cm], cat(1)[cm], cat(2)[nm], cat(3)[nm],
                      cat(4)[cm], cat(5)[cm])


def _train_stats(theta, prob, ds_train, stats):
    """The training sufficient statistics: stats(theta) if the caller supplies them
    (e.g. cached linear statistics), else computed on ds_train -- in QR form (QRStats,
    whose G and b are properties) when prob's LML takes it (objective.uses_qr)."""
    if stats is not None:
        return stats(theta)
    if uses_qr(prob):
        from .stats import linear_qr_statistics
        return linear_qr_statistics(prob.model, prob.cfg, ds_train)
    return sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)


def fit_posterior(theta, st, prob, ds_train):
    """The posterior (mu, L) from training statistics st, by the factorisation the LML used:
    QR for QRStats, else the Cholesky (stable_posterior's QR fallback when it fails)."""
    from .stats import QRStats
    if isinstance(st, QRStats):
        return posterior_from_qr(theta, st, prob)
    return stable_posterior(theta, st, prob, ds_train)


def stable_posterior(theta, st, prob, ds_train):
    """objective.posterior, or posterior_qr on ds_train when the normal-equations Cholesky
    fails: an evidence fit that nearly interpolates its data (noise at its floor, prior
    near flat) leaves G + Lambda too ill-conditioned to factor (kappa ~ 1e16), while the
    design itself, which QR factors, is not.  A Cholesky that succeeds is kept as is."""
    mu, L = posterior(theta, st, prob)
    if bool(jnp.all(jnp.isfinite(L))) and bool(jnp.all(jnp.isfinite(mu))):
        return mu, L
    from .solve import posterior_qr
    return posterior_qr(prob, ds_train, theta)


def theta_key(theta):
    """A hashable key of a Hypers (or its array): its float64 bytes."""
    a = theta if not isinstance(theta, Hypers) else to_array(theta)
    return np.asarray(a, np.float64).tobytes()


def train_posterior(theta, prob, ds_train, stats=None, posts=None):
    """fit_posterior on the training statistics (_train_stats).  posts: an optional dict
    theta_key -> (mu, L) | None; a key present is a posterior worth keeping (the pipeline
    asks for the MAP's, which the model file needs again): a stored one is returned, a None
    is filled in.  Only the requested keys are kept, since each L is (Dt, Dt)."""
    k = theta_key(theta) if posts else None
    if k is not None and posts.get(k) is not None:
        return posts[k]
    post = fit_posterior(theta, _train_stats(theta, prob, ds_train, stats), prob, ds_train)
    if k is not None and k in posts:
        posts[k] = post
    return post


def _run_predict(f, theta, prob, ds_train, ds_test, stats=None, posts=None):
    mu, L = train_posterior(theta, prob, ds_train, stats, posts)
    outs = [f(theta, mu, L, jax.tree.map(lambda a: a[i], ds_test)) for i in range(ds_test.n_batches)]
    return _pack(outs, prob, ds_test)


# ---------------------------------------------------------------------------
# Paper-faithful POPS (Swinburne & Perez, arXiv:2402.01810)
# ---------------------------------------------------------------------------

def _ridge_dict(ridge):
    """A scalar ridge applies to every quantity; a mapping sets each of E/F/V."""
    if isinstance(ridge, dict):
        return {q: _rkey(ridge[q]) for q in "EFV"}
    return {q: _rkey(ridge) for q in "EFV"}


class PopsRidgePath:
    """Paper-faithful POPS members along a ridge path, from ONE factorisation.

    The paper's parameter covariance is ``A = [Sigma_Y Sigma_0^-1 + <F F^T>]^-1``
    with ``Sigma_Y`` an INTRINSIC, fixed noise -- explicitly NOT fitted to the
    residual (fitting it "recovers standard maximum-likelihood inference, which
    ignores misspecification").  So here the member rows carry only the
    structural weights ``w`` (the evidence-fit sigma_q never enter), the data Gram
    is ``M = G_E + G_F + G_V`` from the sufficient statistics (``(w Phi)^T (w Phi)``),
    and the regulariser is ``lam * Gamma^2`` (``Sigma_0`` = the smoothness prior).
    With ``D = diag(Gamma)``, ``D^-1 M D^-1 = U diag(Lambda) U^T`` is factorised
    once and every ridge reuses it::

        A(lam) = D^-1 U diag(1 / (Lambda + lam)) U^T D^-1 ,   lam = ridge * max(Lambda)

    The loss is the BLR's own: rows weighted ``w / sigma_q`` (the fit's relative
    E/F/V weights; their absolute scale cancels against a relative ridge) and the
    regulariser ``lam * Gamma^2``.  ``ridge='blr'`` is ``lam = 1/sigma_c^2``, the
    BLR prior, so ``c*`` IS the BLR mean and ``A`` the BLR posterior covariance; a
    number is relative (``lam = ridge * max Lambda``).  The mean is always the
    minimiser of the SAME loss that defines ``A`` (the pointwise corrections
    assume it), read off the same factorisation::

        c*(lam) = D^-1 U diag(1 / (Lambda + lam)) U^T D^-1 b ,   b = (w Phi)^T (w y)

    sigma_q only weights the FIT -- it never enters the predictive (no noise
    term).  By default each call's mean is its own ridge's ``c*``;
    ``use_mean(r)`` pins one mean for every ridge.  Linear arm (M == 0) only."""

    def __init__(self, theta, prob, ds_train, stats=None, rows="device"):
        """rows: "device" (each pass re-evaluates the ACE rows), "host" (evaluate
        them once into host RAM: stats.HostRows), "auto" (host when it fits in
        half the memory limit), or a HostRows built with this path's qs."""
        if getattr(prob.cfg, "e0_cols", False):
            raise ValueError("POPS builds on the readout's smoothness prior: fit with the least-squares E0 "
                             "fixed (e0='prefit'; the pipeline does this for uq='pops')")
        M_ind = prob.ind.XM.shape[0]
        if M_ind > 0:
            raise ValueError(f"paper-faithful POPS is the linear-arm (M=0) predictive only; "
                             f"this problem has M={M_ind} inducing points.  Pass --arm linear.")
        s2 = {q: jnp.exp(2.0 * getattr(theta, f"log_sigma_{q}")) for q in "EFV"}
        self.sigma = {q: float(jnp.sqrt(s2[q])) for q in "EFV"}
        self.qs = tuple(1.0 / self.sigma[q] for q in "EFV")        # loss weights 1/sigma_q
        self.lam_blr = float(jnp.exp(-2.0 * theta.log_sigma_c))    # the BLR prior Gamma^2 / sigma_c^2
        if rows == "auto":
            rows = ("host" if host_rows_bytes(ds_train, prob.cfg.len_basis) < 0.5 * available_host_bytes()
                    else "device")
        if rows == "host":
            rows = HostRows(prob.model, prob.cfg, ds_train, self.qs)
        if isinstance(rows, HostRows):
            M, b = rows.gram()                       # the same M, b from the cached rows: no stats pass
        elif rows == "device":
            st = _train_stats(theta, prob, ds_train, stats)
            M = sum(getattr(st, f"G_{q}") / s2[q] for q in "EFV")
            b = sum(getattr(st, f"b_{q}") / s2[q] for q in "EFV")
            rows = DeviceRows(prob.model, prob.cfg, ds_train, self.qs)
        else:
            raise ValueError(f"rows must be 'device', 'host', 'auto' or a HostRows, got {rows!r}")
        self.rows = rows
        self.Dinv = 1.0 / jnp.asarray(prob.gamma)
        self._Ms = M * self.Dinv[:, None] * self.Dinv[None, :]      # D^-1 M D^-1, kept for the mean
        self._Db = self.Dinv * b                                     # D^-1 b
        Lam, U = jnp.linalg.eigh(self._Ms)
        self.Lam = jnp.maximum(Lam, 0.0)                    # clip round-off negatives
        self.W = U * self.Dinv[:, None]                     # D^-1 U
        self.mean_ridge = None                              # None: each call uses its own ridge
        self._means = {}                                    # ridge -> c*(ridge)
        self._posts = {}                                    # (mean, ridge, form, lev, thr) -> posterior
        self.prob, self.ds = prob, ds_train

    def ridge_abs(self, ridge):
        if ridge == "blr":
            return self.lam_blr
        return float(ridge) * float(jnp.max(self.Lam))

    def c_star_at(self, ridge):
        """The ridge solution c*(ridge) = (M + lam Gamma^2)^-1 b (memoised), by a
        Cholesky solve of the scaled system (D^-1 M D^-1 + lam) (D c) = D^-1 b.
        Not read off the eigendecomposition: that route loses the directions whose
        eigenvalues sit below eigh's absolute resolution (eps * max Lambda) --
        1e-3 relative error on the Si fixture, 4e-4 eV/A in Cantor forces."""
        from jax.scipy.linalg import cho_factor, cho_solve
        r = _rkey(ridge)
        if r not in self._means:
            lam = self.ridge_abs(r)
            cf = cho_factor(self._Ms + lam * jnp.eye(self._Ms.shape[0]), lower=True)
            self._means[r] = self.Dinv * cho_solve(cf, self._Db)
        return self._means[r]

    def use_mean(self, ridge):
        """Pin the mean to c*(ridge) for every subsequent ridge (None: unpin)."""
        self.mean_ridge = None if ridge is None else _rkey(ridge)

    def _mean_for(self, ridge):
        return _rkey(ridge) if self.mean_ridge is None else self.mean_ridge

    @property
    def c_star(self):
        if self.mean_ridge is None:
            raise ValueError("no mean pinned: call use_mean(ridge) or use c_star_at(ridge)")
        return self.c_star_at(self.mean_ridge)

    def A(self, ridge):
        return (self.W / (self.Lam + self.ridge_abs(ridge))) @ self.W.T

    def _coef(self, ridge, leverage_pct):
        """Pass 1 of the streamed POPS: A, the per-row coefficient c_i = r_i / h_i
        (0 off the member set) and the member mask, each (n_batches, rows).  Members
        are the non-padded rows (h > 0) at or above the leverage percentile -- the
        same set as ``members`` / ``leverage_select``."""
        A = self.A(ridge)
        c = self.c_star_at(self._mean_for(ridge))
        h, r = self.rows.leverage_residual(c, A)
        live = h > 0
        thr = jnp.percentile(h[live], leverage_pct)
        keep = live & (h >= thr)
        if not bool(jnp.any(keep)):                          # leverage_select's fallback
            keep = live
        coef = jnp.where(keep, r / jnp.where(live, h, 1.0), 0.0)
        return A, coef, keep

    def posterior(self, ridge, form="hypercube", leverage_pct=0.0, mode_threshold=1.0e-8):
        """POPS posterior WITHOUT materialising the corrections: O(L^2) memory.
        Memoised per (ridge, form, leverage_pct, mode_threshold): each one is
        several streamed passes over the training set.

        hypercube: box axes from sum delta delta^T = A W A, bounds from a streamed
        min/max of the projected corrections (3 passes).  ensemble: the moments
        E[delta delta^T] = A W A / K and E[delta] = A s / K (2 passes).  Equals
        ``pops_posterior(self.members(...))`` for zero percentile clipping."""
        key = (self._mean_for(ridge), _rkey(ridge), form, float(leverage_pct), float(mode_threshold))
        if key not in self._posts:
            self._posts[key] = self._posterior(ridge, form, leverage_pct, mode_threshold)
        return self._posts[key]

    def _posterior(self, ridge, form, leverage_pct, mode_threshold):
        from .pops import hypercube_cov, hypercube_support
        A, coef, keep = self._coef(ridge, leverage_pct)
        W, s = self.rows.moment_sums(coef)
        if form == "ensemble":
            K = jnp.sum(keep)
            return {"moments": (A @ W @ A / K, A @ s / K)}
        if form != "hypercube":
            raise ValueError(f"unknown POPS posterior form: {form!r}")
        support = hypercube_support(A @ W @ A, mode_threshold)
        lo, hi = self.rows.projection_bounds(coef, keep, A @ support)
        return {"cov": hypercube_cov(support, lo, hi)}

    def envelope(self, phi_star, ridge, leverage_pct=0.0):
        """Member min/max of the prediction shift phi* . delta_i at rows phi_star,
        streamed (equals ``pops.pops_envelope(phi_star, self.members(...))``)."""
        A, coef, keep = self._coef(ridge, leverage_pct)
        return self.rows.envelope(coef, keep, phi_star @ A)

    def members(self, ridge, leverage_pct=0.0):
        """Pointwise-optimal corrections of every member, MATERIALISED (K, L).
        A small-problem oracle for tests: production paths use ``posterior`` /
        ``envelope``, which never form this matrix."""
        deltas, h = _stream_pops_pointwise(self.prob.model, self.prob.cfg, self.ds,     # rows w/sigma_q
                                           self.c_star_at(self._mean_for(ridge)), self.A(ridge), self.sigma)
        keep = np.asarray(h) > 0                            # drop padded (w = 0) rows
        return leverage_select(deltas[keep], h[keep], leverage_pct)


def _pops_batch_phi(prob, batch):
    """The linear rows (phiE (C, L), phiF (3 Ncap, L), phiV (6 C, L)) of one batch."""
    lin = linear_rows_bounded(prob.model, prob.cfg, batch)
    L = lin.E.shape[-1]
    return lin.E, lin.F.reshape(-1, L), lin.V.reshape(-1, L)


def _pops_rows_predict(mu, posts, phiE, phiF, phiV):
    """POPS mean and misspecification variance from precomputed rows."""
    Ev, Fv, Vv = (pops_var(phiE, posts["E"]), pops_var(phiF, posts["F"]),
                  pops_var(phiV, posts["V"]))
    return (phiE @ mu, Ev, (phiF @ mu).reshape(-1, 3), Fv.reshape(-1, 3),
            (phiV @ mu).reshape(-1, 6), Vv.reshape(-1, 6))


def _pops_paper_batch(prob, mu, posts, batch):
    """Paper-faithful POPS predictive for one batch: mean c*.phi*, variance the
    misspecification posterior of each quantity (no noise / epistemic terms)."""
    return _pops_rows_predict(mu, posts, *_pops_batch_phi(prob, batch))


def _pops_predict_fn(prob):
    """The jitted POPS batch predictor with the mean and posteriors as ARGUMENTS:
    closing over them would re-trace per ridge and embed every L x L posterior in
    the compiled program as a constant (6 GB each at the production Cantor basis)."""
    return jax.jit(lambda mu, posts, b: _pops_paper_batch(prob, mu, posts, b))


def _pops_paper_posts(path, ridge, form, leverage_pct):
    r, cache, posts = _ridge_dict(ridge), {}, {}
    for q in "EFV":
        if r[q] not in cache:
            cache[r[q]] = path.posterior(r[q], form=form, leverage_pct=leverage_pct)
        posts[q] = cache[r[q]]
    return posts


def _rkey(ridge):
    """A ridge as a hashable key: 'blr' or a float."""
    return "blr" if ridge == "blr" else float(ridge)


POPS_MEAN = "blr"
"""The POPS mean is pinned to the BLR mean, c*('blr'), whatever the uncertainty
ridge(s): one potential, the one that is fitted and reported.  On SiGe/Cantor a
single consistent (c*, A) either gave a poor mean (theta-free ridge mean, E 10x
worse) or unbalanced E/F uncertainty (the 'blr' ridge for A: E sigma ~150x too
large, F too small), so each quantity's A(ridge_q) is chosen for its own
calibration around the fixed BLR mean."""


def _run_predict_pops_paper(theta, prob, ds_train, ds_test, form, ridge, leverage_pct, path=None,
                            stats=None):
    path = PopsRidgePath(theta, prob, ds_train, stats=stats) if path is None else path
    path.use_mean(POPS_MEAN)
    posts = _pops_paper_posts(path, ridge, form, leverage_pct)
    f = _pops_predict_fn(prob)
    outs = [f(path.c_star, posts, jax.tree.map(lambda a: a[i], ds_test)) for i in range(ds_test.n_batches)]
    return _pack(outs, prob, ds_test)


def select_pops_ridge(theta, prob, ds_fit, ds_val, grid, form="hypercube", leverage_pct=0.0, stats=None,
                      rows="device"):
    """Choose the paper-faithful POPS ridge per quantity on held-out data.

    One :class:`PopsRidgePath` factorisation of the ``ds_fit`` Gram serves the
    whole ``grid``; for each ridge the members are rebuilt and the predictive is
    scored on ``ds_val`` by mean CRPS (energies and virials per
    atom, forces per component).  Returns ``(ridge, scores)``: ``ridge[q]`` is the
    grid value minimising ``scores[q]`` (an array aligned with ``grid``).

    The mean is pinned (POPS_MEAN), so the ridge only moves the uncertainty and
    the mean CRPS (per atom for E and V) ranks calibration+sharpness.

    rows: see PopsRidgePath.  Other than "device", the validation rows are also
    evaluated once and kept on the host, not re-evaluated for every ridge."""
    path = PopsRidgePath(theta, prob, ds_fit, stats=stats, rows=rows)
    path.use_mean(POPS_MEAN)                                 # the ridge only moves A, never the mean
    scores = {q: [] for q in "EFV"}
    batches = [jax.tree.map(lambda a: a[i], ds_val) for i in range(ds_val.n_batches)]
    if isinstance(path.rows, HostRows):
        phi_fn = jax.jit(lambda b: _pops_batch_phi(prob, b))
        val_phi = [tuple(np.asarray(x) for x in phi_fn(b)) for b in batches]
        g = jax.jit(_pops_rows_predict)
        predict = lambda posts, i: g(path.c_star, posts, *val_phi[i])
    else:
        f = _pops_predict_fn(prob)                            # compiled once for the whole grid
        predict = lambda posts, i: f(path.c_star, posts, batches[i])
    for r in grid:
        post = path.posterior(r, form=form, leverage_pct=leverage_pct)
        posts = {q: post for q in "EFV"}
        cols = {q: ([], [], []) for q in "EFV"}               # y, mean, sd (scaled, live rows)
        for i, b in enumerate(batches):
            Em, Ev, Fm, Fv, Vm, Vv = predict(posts, i)
            C = b.y_E.shape[0]
            nat = np.zeros(C + 1)                             # bucket C collects padded nodes
            np.add.at(nat, np.asarray(b.node_cfg), np.asarray(b.node_mask, float))
            nat = np.maximum(nat[:C], 1.0)
            for q, y, m, v, w, sc in (
                    ("E", b.y_E, Em, Ev, b.w_E, nat),
                    ("F", b.y_F.reshape(-1), Fm.reshape(-1), Fv.reshape(-1), np.repeat(np.asarray(b.w_F), 3), 1.0),
                    ("V", b.y_V.reshape(-1), Vm.reshape(-1), Vv.reshape(-1), np.repeat(np.asarray(b.w_V), 6),
                     np.repeat(nat, 6))):
                k = np.asarray(w) > 0
                sc = np.broadcast_to(np.asarray(sc, float), np.asarray(w).shape)[k]
                cols[q][0].append(np.asarray(y)[k] / sc); cols[q][1].append(np.asarray(m)[k] / sc)
                cols[q][2].append(np.sqrt(np.maximum(np.asarray(v)[k], 1e-300)) / sc)
        for q in "EFV":
            y, m, s = (np.concatenate(c) for c in cols[q])
            if y.size == 0:
                scores[q].append(np.nan); continue
            scores[q].append(float(np.mean(crps_gaussian(y, m, s))))
    scores = {q: np.asarray(v) for q, v in scores.items()}
    ridge = {q: (float(grid[int(np.nanargmin(scores[q]))]) if np.isfinite(scores[q]).any()
                 else float(grid[0])) for q in "EFV"}
    return ridge, scores


def predict_fixed(theta, prob, ds_train, ds_test, dtc=True, deriv_dtc=True,
                  uq="blr", pops_form="hypercube", leverage_pct=0.0, pops_ridge="blr", pops_path=None,
                  stats=None, posts=None):
    """dtc=False drops the DTC prior residual from E_var (SoR only; for tests).
    deriv_dtc=False keeps the energy DTC residual but drops its force/virial
    derivative (F_var, V_var stay SoR-only).

    uq='blr' (default) is the linear/GP posterior predictive variance.  uq='pops'
    is the linear-arm (M=0) misspecification predictive of Swinburne & Perez
    (arXiv:2402.01810) as published: the mean is the fitted linear posterior mean,
    the variance is the pointwise-optimal-parameter-set posterior built from
    STRUCTURAL weights only (the evidence-fit sigma_q never enter -- fitting the
    noise to the residual reverts to plain ML) with the regulariser
    pops_ridge * Gamma^2 (relative ridge; a float, or a dict per E/F/V).  There is
    no separate noise/epistemic term.  pops_form in {'hypercube','ensemble'},
    leverage_pct the leverage percentile.  pops_path: a PopsRidgePath already built
    on (theta, ds_train) to reuse -- one factorisation and its memoised posteriors
    serve every test split.  stats: optional theta -> training Stats (e.g. from
    cached linear statistics); default computes them on ds_train.  posts: see
    train_posterior (uq 'blr' only).  See PopsRidgePath / select_pops_ridge."""
    if uq == "blr":
        return _run_predict(_predict_fn(prob, dtc, deriv_dtc), theta, prob, ds_train, ds_test, stats=stats,
                            posts=posts)
    if uq == "pops":
        return _run_predict_pops_paper(theta, prob, ds_train, ds_test, pops_form,
                                       pops_ridge, leverage_pct, path=pops_path, stats=stats)
    raise ValueError(f"uq must be 'blr' or 'pops', got {uq!r}")


def predict_readout(prob, mu, ds_test):
    """Predictions of a fixed readout mu (no posterior: zero variance), linear arm."""
    def batch(mu, b):
        r = linear_rows_bounded(prob.model, prob.cfg, b)       # peak memory bounded for big cells
        Em, Fm, Vm = r.E @ mu, (r.F @ mu).reshape(-1, 3), (r.V @ mu).reshape(-1, 6)
        return Em, jnp.zeros_like(Em), Fm, jnp.zeros_like(Fm), Vm, jnp.zeros_like(Vm)
    f, mu = jax.jit(batch), jnp.asarray(mu)
    return _pack([f(mu, jax.tree.map(lambda a: a[i], ds_test)) for i in range(ds_test.n_batches)], prob, ds_test)


def predict_mixture(draws, prob, ds_train, ds_test, deriv_dtc=True, stats=None, posts=None):
    f = _predict_fn(prob, True, deriv_dtc)   # compile ONCE, reuse across all draws
    preds = [_run_predict(f, from_array(jnp.asarray(d)), prob, ds_train, ds_test, stats=stats, posts=posts)
             for d in np.asarray(draws)]
    out = []
    for k in range(0, 6, 2):
        means = np.stack([p[k] for p in preds]); vars_ = np.stack([p[k + 1] for p in preds])
        out += [means.mean(0), vars_.mean(0) + means.var(0)]
    return Prediction(*out)
