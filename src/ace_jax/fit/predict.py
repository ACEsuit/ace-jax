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

from .hypers import from_array
from .kernels import K_MM, k_rows
from .summary import site_summary
from .feature import apply as _feat, dwarp
from .objective import posterior
from .data import VOIGT, flat_edges
from .metrics import crps_gaussian
from .pops import leverage_select, pops_var
from .rows import Rows, linear_rows, residual_inputs, residual_rows
from .stats import (_stream_pops_pointwise, pops_envelope_streamed, pops_leverage_residual,
                    pops_moment_sums, pops_projection_bounds, sufficient_statistics)


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


def _dtc_deriv_residual(theta, prob, batch, X=None, J=None, res=None):
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
    if X is None or J is None:
        _, X, J = linear_rows(prob.model, cfg, batch)
    if res is None:
        res = residual_rows(theta, spec, ind, cfg, batch, X, J)
    d, M = ind.XM.shape[1], ind.XM.shape[0]
    rij, send, recv, m = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    z = batch.node_z
    U, s, Js = residual_inputs(ind, cfg, batch, X)                        # U (Ncap,d), Js (E,3)
    dw = dwarp(X @ ind.Pmap, ind.warp)                                    # (Ncap, d)
    JU = dw[:, None, :, None] * jnp.einsum("nkDa,Dq->nkqa",
                                           J.reshape(Ncap, K, cfg.D, 3), ind.Pmap)  # (Ncap,K,d,3)
    Jsd = Js.reshape(Ncap, K, 3)
    ar = jnp.arange(Ncap)

    # feature/summary velocities of every site: dU_i/dr_n and dU_i/deps_v.
    # rij = r_recv - r_send, so dU_i/dr_i = -sum_k JU[i,k], dU_i/dr_{nbr} = +JU[i,k].
    JU_e = JU.reshape(Ncap * K, d, 3); Js_e = Jsd.reshape(Ncap * K, 3)
    AU = jnp.zeros((Ncap, 3, Ncap, d)).at[ar, :, ar, :].add(-jnp.swapaxes(JU.sum(1), 1, 2))
    AS = jnp.zeros((Ncap, 3, Ncap)).at[ar, :, ar].add(-Jsd.sum(1))
    AU = AU.at[recv, :, send, :].add(jnp.where(m[:, None, None], jnp.swapaxes(JU_e, 1, 2), 0.0))
    AS = AS.at[recv, :, send].add(jnp.where(m[:, None], Js_e, 0.0))
    aU, aS = AU.reshape(Ncap * 3, Ncap, d), AS.reshape(Ncap * 3, Ncap)
    # strain generator W_v(rij) = 1/2 (e_a rij_b + e_b rij_a); dU_i/deps_v via i's edges
    Wv = jnp.stack([(jnp.zeros((rij.shape[0], 3)).at[:, a].add(0.5 * rij[:, b])
                                                  .at[:, b].add(0.5 * rij[:, a])) for a, b in VOIGT], 1)  # (E,6,3)
    ecfg = batch.node_cfg[send]
    cbU = jnp.where(m[:, None, None], jnp.einsum("eqa,eva->evq", JU_e, Wv), 0.0)   # (E,6,d)
    cbS = jnp.where(m[:, None], jnp.einsum("ea,eva->ev", Js_e, Wv), 0.0)           # (E,6)
    bU = jnp.zeros((C, 6, Ncap, d)).at[ecfg, :, send, :].add(cbU).reshape(C * 6, Ncap, d)
    bS = jnp.zeros((C, 6, Ncap)).at[ecfg, :, send].add(cbS).reshape(C * 6, Ncap)

    # <o, o>_k = d_beta d_gamma sum_{i,j in cfg} k(U_i+beta v_i, U_j+gamma v_j) along the
    # matched velocity v (in both the U and the summary s slots), via a double jvp.
    live = batch.node_mask[:, None] & batch.node_mask[None, :]
    def Ksum(Ux, sx, Uy, sy):
        return jnp.sum(jnp.where(live, k_rows(theta, spec, Ux, sx, z, Uy, sy, z, ind.embed), 0.0))
    def ffk(vU, vS):
        gy = lambda Uy, sy: jax.jvp(lambda Ux, sx: Ksum(Ux, sx, Uy, sy), (U, s), (vU, vS))[1]
        return jax.jvp(gy, (U, s), (vU, vS))[1]
    FFk = jax.vmap(ffk)(aU, aS).reshape(Ncap, 3)
    VVk = jax.vmap(ffk)(bU, bS).reshape(C, 6)

    # Q-term K_oM K_MM^-1 K_Mo from the residual force/virial rows
    L_MM = jnp.linalg.cholesky(K_MM(theta, spec, ind.XM, ind.SM, ind.ZM, ind.embed))
    vF = solve_triangular(L_MM, res.F.reshape(Ncap * 3, M).T, lower=True)
    vV = solve_triangular(L_MM, res.V.reshape(C * 6, M).T, lower=True)
    Fv = jnp.where(batch.node_mask[:, None], jnp.maximum(FFk - jnp.sum(vF * vF, 0).reshape(Ncap, 3), 0.0), 0.0)
    Vv = jnp.maximum(VVk - jnp.sum(vV * vV, 0).reshape(C, 6), 0.0)
    return Fv, Vv


def _predict_batch(theta, prob, mu, L, batch, dtc=True, deriv_dtc=True):
    lin, X, J = linear_rows(prob.model, prob.cfg, batch)        # as rows.batch_rows, keeping X
    res = residual_rows(theta, prob.spec, prob.ind, prob.cfg, batch, X, J)
    r = Rows(jnp.concatenate([lin.E, res.E], 1), jnp.concatenate([lin.F, res.F], 2),
             jnp.concatenate([lin.V, res.V], 2))
    Dt = r.E.shape[-1]
    Em, Ev = _rows_mean_var(r.E, mu, L)
    if dtc:
        Ev = Ev + _dtc_energy_residual(theta, prob, batch, X)
    Fm, Fv = _rows_mean_var(r.F.reshape(-1, Dt), mu, L)
    Vm, Vv = _rows_mean_var(r.V.reshape(-1, Dt), mu, L)
    Fv, Vv = Fv.reshape(-1, 3), Vv.reshape(-1, 6)
    if dtc and deriv_dtc:                                    # F_var, V_var = d E_var (self-consistent)
        dFv, dVv = _dtc_deriv_residual(theta, prob, batch, X, J, res)
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


def _run_predict(f, theta, prob, ds_train, ds_test):
    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)
    mu, L = posterior(theta, st, prob)
    outs = [f(theta, mu, L, jax.tree.map(lambda a: a[i], ds_test)) for i in range(ds_test.n_batches)]
    return _pack(outs, prob, ds_test)


# ---------------------------------------------------------------------------
# Paper-faithful POPS (Swinburne & Perez, arXiv:2402.01810)
# ---------------------------------------------------------------------------

def _ridge_dict(ridge):
    """A scalar ridge applies to every quantity; a mapping sets each of E/F/V."""
    if isinstance(ridge, dict):
        return {q: float(ridge[q]) for q in "EFV"}
    return {q: float(ridge) for q in "EFV"}


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

    ``ridge`` is relative (dimensionless).  The mean ``c_star`` is the fitted
    linear posterior mean at ``theta`` (the same mean as uq='blr'), so POPS only
    supplies the uncertainty.  Linear arm (M == 0) only."""

    def __init__(self, theta, prob, ds_train):
        M_ind = prob.ind.XM.shape[0]
        if M_ind > 0:
            raise ValueError(f"paper-faithful POPS is the linear-arm (M=0) predictive only; "
                             f"this problem has M={M_ind} inducing points.  Pass --arm linear.")
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)
        self.c_star = posterior(theta, st, prob)[0]
        M = st.G_E + st.G_F + st.G_V
        self.Dinv = 1.0 / jnp.asarray(prob.gamma)
        Lam, U = jnp.linalg.eigh(M * self.Dinv[:, None] * self.Dinv[None, :])
        self.Lam = jnp.maximum(Lam, 0.0)                    # clip round-off negatives
        self.W = U * self.Dinv[:, None]                     # D^-1 U
        self.prob, self.ds = prob, ds_train

    def ridge_abs(self, ridge):
        return float(ridge) * float(jnp.max(self.Lam))

    def A(self, ridge):
        return (self.W / (self.Lam + self.ridge_abs(ridge))) @ self.W.T

    def _coef(self, ridge, leverage_pct):
        """Pass 1 of the streamed POPS: A, the per-row coefficient c_i = r_i / h_i
        (0 off the member set) and the member mask, each (n_batches, rows).  Members
        are the non-padded rows (h > 0) at or above the leverage percentile -- the
        same set as ``members`` / ``leverage_select``."""
        A = self.A(ridge)
        h, r = pops_leverage_residual(self.prob.model, self.prob.cfg, self.ds, self.c_star, A)
        live = h > 0
        thr = jnp.percentile(h[live], leverage_pct)
        keep = live & (h >= thr)
        if not bool(jnp.any(keep)):                          # leverage_select's fallback
            keep = live
        coef = jnp.where(keep, r / jnp.where(live, h, 1.0), 0.0)
        return A, coef, keep

    def posterior(self, ridge, form="hypercube", leverage_pct=0.0, mode_threshold=1.0e-8):
        """POPS posterior WITHOUT materialising the corrections: O(L^2) memory.

        hypercube: box axes from sum delta delta^T = A W A, bounds from a streamed
        min/max of the projected corrections (3 passes).  ensemble: the moments
        E[delta delta^T] = A W A / K and E[delta] = A s / K (2 passes).  Equals
        ``pops_posterior(self.members(...))`` for zero percentile clipping."""
        from .pops import hypercube_cov, hypercube_support
        model, cfg = self.prob.model, self.prob.cfg
        A, coef, keep = self._coef(ridge, leverage_pct)
        W, s = pops_moment_sums(model, cfg, self.ds, coef)
        if form == "ensemble":
            K = jnp.sum(keep)
            return {"moments": (A @ W @ A / K, A @ s / K)}
        if form != "hypercube":
            raise ValueError(f"unknown POPS posterior form: {form!r}")
        support = hypercube_support(A @ W @ A, mode_threshold)
        lo, hi = pops_projection_bounds(model, cfg, self.ds, coef, keep, A @ support)
        return {"cov": hypercube_cov(support, lo, hi)}

    def envelope(self, phi_star, ridge, leverage_pct=0.0):
        """Member min/max of the prediction shift phi* . delta_i at rows phi_star,
        streamed (equals ``pops.pops_envelope(phi_star, self.members(...))``)."""
        A, coef, keep = self._coef(ridge, leverage_pct)
        return pops_envelope_streamed(self.prob.model, self.prob.cfg, self.ds, coef, keep,
                                      phi_star @ A)

    def members(self, ridge, leverage_pct=0.0):
        """Pointwise-optimal corrections of every member, MATERIALISED (K, L).
        A small-problem oracle for tests: production paths use ``posterior`` /
        ``envelope``, which never form this matrix."""
        one = {q: 1.0 for q in "EFV"}                       # structural weights only
        deltas, h = _stream_pops_pointwise(self.prob.model, self.prob.cfg, self.ds,
                                           self.c_star, self.A(ridge), one)
        keep = np.asarray(h) > 0                            # drop padded (w = 0) rows
        return leverage_select(deltas[keep], h[keep], leverage_pct)


def _pops_paper_batch(prob, mu, posts, batch):
    """Paper-faithful POPS predictive for one batch: mean c*.phi*, variance the
    misspecification posterior of each quantity (no noise / epistemic terms)."""
    lin, _, _ = linear_rows(prob.model, prob.cfg, batch)
    L = lin.E.shape[-1]
    phiE, phiF, phiV = lin.E, lin.F.reshape(-1, L), lin.V.reshape(-1, L)
    Ev, Fv, Vv = (pops_var(phiE, posts["E"]), pops_var(phiF, posts["F"]),
                  pops_var(phiV, posts["V"]))
    return (phiE @ mu, Ev, (phiF @ mu).reshape(-1, 3), Fv.reshape(-1, 3),
            (phiV @ mu).reshape(-1, 6), Vv.reshape(-1, 6))


def _pops_paper_posts(path, ridge, form, leverage_pct):
    r, cache, posts = _ridge_dict(ridge), {}, {}
    for q in "EFV":
        if r[q] not in cache:
            cache[r[q]] = path.posterior(r[q], form=form, leverage_pct=leverage_pct)
        posts[q] = cache[r[q]]
    return posts


def _run_predict_pops_paper(theta, prob, ds_train, ds_test, form, ridge, leverage_pct):
    path = PopsRidgePath(theta, prob, ds_train)
    posts = _pops_paper_posts(path, ridge, form, leverage_pct)
    f = jax.jit(lambda mu, b: _pops_paper_batch(prob, mu, posts, b))
    outs = [f(path.c_star, jax.tree.map(lambda a: a[i], ds_test)) for i in range(ds_test.n_batches)]
    return _pack(outs, prob, ds_test)


def select_pops_ridge(theta, prob, ds_fit, ds_val, grid, form="hypercube", leverage_pct=0.0):
    """Choose the paper-faithful POPS ridge per quantity on held-out data.

    One :class:`PopsRidgePath` factorisation of the ``ds_fit`` Gram serves the
    whole ``grid``; for each ridge the members are rebuilt and the predictive is
    scored on ``ds_val`` by CRPS / RMSE (scale-free; energies and virials per
    atom, forces per component).  Returns ``(ridge, scores)``: ``ridge[q]`` is the
    grid value minimising ``scores[q]`` (an array aligned with ``grid``)."""
    path = PopsRidgePath(theta, prob, ds_fit)
    scores = {q: [] for q in "EFV"}
    for r in grid:
        post = path.posterior(r, form=form, leverage_pct=leverage_pct)
        posts = {q: post for q in "EFV"}
        f = jax.jit(lambda mu, b: _pops_paper_batch(prob, mu, posts, b))
        cols = {q: ([], [], []) for q in "EFV"}               # y, mean, sd (scaled, live rows)
        for i in range(ds_val.n_batches):
            b = jax.tree.map(lambda a: a[i], ds_val)
            Em, Ev, Fm, Fv, Vm, Vv = f(path.c_star, b)
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
            rmse = np.sqrt(np.mean((y - m) ** 2))
            scores[q].append(float(np.mean(crps_gaussian(y, m, s))) / max(rmse, 1e-300))
    scores = {q: np.asarray(v) for q, v in scores.items()}
    ridge = {q: (float(grid[int(np.nanargmin(scores[q]))]) if np.isfinite(scores[q]).any()
                 else float(grid[0])) for q in "EFV"}
    return ridge, scores


def predict_fixed(theta, prob, ds_train, ds_test, dtc=True, deriv_dtc=True,
                  uq="blr", pops_form="hypercube", leverage_pct=0.0, pops_ridge=1e-3):
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
    leverage_pct the leverage percentile.  See PopsRidgePath / select_pops_ridge."""
    if uq == "blr":
        return _run_predict(_predict_fn(prob, dtc, deriv_dtc), theta, prob, ds_train, ds_test)
    if uq == "pops":
        return _run_predict_pops_paper(theta, prob, ds_train, ds_test, pops_form,
                                       pops_ridge, leverage_pct)
    raise ValueError(f"uq must be 'blr' or 'pops', got {uq!r}")


def predict_mixture(draws, prob, ds_train, ds_test, deriv_dtc=True):
    f = _predict_fn(prob, True, deriv_dtc)   # compile ONCE, reuse across all draws
    preds = [_run_predict(f, from_array(jnp.asarray(d)), prob, ds_train, ds_test)
             for d in np.asarray(draws)]
    out = []
    for k in range(0, 6, 2):
        means = np.stack([p[k] for p in preds]); vars_ = np.stack([p[k + 1] for p in preds])
        out += [means.mean(0), vars_.mean(0) + means.var(0)]
    return Prediction(*out)
