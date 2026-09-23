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
from jax.scipy.linalg import cho_solve, solve_triangular

from .hypers import from_array
from .kernels import K_MM, k_rows
from .summary import site_summary
from .feature import apply as _feat, dwarp
from .objective import posterior
from .data import VOIGT, flat_edges
from .pops import pops_posterior, pops_var
from .rows import Rows, linear_rows, residual_inputs, residual_rows
from .stats import pops_statistics, sufficient_statistics


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


def _pops_predict_batch(prob, mu, post, sigma, aleatoric, batch):
    """Linear-arm POPS predictive for one batch.  The MEAN is the BLR linear
    posterior mean c*.phi* (mu == c_star; unchanged from uq='blr').  The VARIANCE
    is the POPS weight-space misspecification covariance evaluated on the RAW (NOT
    whitened) test linear rows phi* -- Sigma_POPS is already a weight covariance,
    so predictive var = pops_var(phi*, post) directly.  E rows are one L-vector per
    config, F rows one L-vector per force component, V rows per Voigt component.
    aleatoric=True ADDs the per-quantity label noise (sigma_q / w*)^2 (padded rows,
    w=0, contribute 0)."""
    lin, _, _ = linear_rows(prob.model, prob.cfg, batch)
    L = lin.E.shape[-1]
    phiE, phiF, phiV = lin.E, lin.F.reshape(-1, L), lin.V.reshape(-1, L)
    Em, Ev = phiE @ mu, pops_var(phiE, post)
    Fm, Fv = phiF @ mu, pops_var(phiF, post)
    Vm, Vv = phiV @ mu, pops_var(phiV, post)
    if aleatoric:
        alea = lambda w, sq: jnp.where(w > 0, (sq / jnp.where(w > 0, w, 1.0)) ** 2, 0.0)
        Ev = Ev + alea(batch.w_E, sigma["E"])
        Fv = Fv + alea(jnp.repeat(batch.w_F, 3), sigma["F"])
        Vv = Vv + alea(jnp.repeat(batch.w_V, 6), sigma["V"])
    return Em, Ev, Fm.reshape(-1, 3), Fv.reshape(-1, 3), Vm.reshape(-1, 6), Vv.reshape(-1, 6)


def _run_predict_pops(theta, prob, ds_train, ds_test, form, leverage_pct, aleatoric):
    """POPS misspecification predictive for the LINEAR arm (M == 0).  Sigma0 = A^-1
    and c_star = A^-1 b come from objective.posterior; the pointwise corrections
    deltas = pops_statistics(...) (Task 9) build the POPS posterior (Task 8), which
    replaces the BLR variance over the raw test linear rows."""
    M = prob.ind.XM.shape[0]
    if M > 0:
        raise ValueError(
            f"uq='pops' is the linear-arm (M=0) misspecification predictive only; "
            f"this problem has M={M} inducing points (the GP arm, --arm gp, which "
            f"uses the DTC path).  Pass --arm linear, or --uq blr.")
    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)
    c_star, cholA = posterior(theta, st, prob)                     # mean A^-1 b, chol(A)
    Sigma0 = cho_solve((cholA, True), jnp.eye(c_star.shape[0]))    # A^-1
    sigma = {q: float(jnp.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}   # per-quantity sigma_q
    deltas = pops_statistics(c_star, Sigma0, prob, ds_train, sigma, leverage_pct=leverage_pct)
    post = pops_posterior(deltas, c_star, form=form)
    f = jax.jit(lambda mu, b: _pops_predict_batch(prob, mu, post, sigma, aleatoric, b))
    outs = [f(c_star, jax.tree.map(lambda a: a[i], ds_test)) for i in range(ds_test.n_batches)]
    return _pack(outs, prob, ds_test)


def predict_fixed(theta, prob, ds_train, ds_test, dtc=True, deriv_dtc=True,
                  uq="blr", pops_form="hypercube", leverage_pct=0.0, aleatoric=True):
    """dtc=False drops the DTC prior residual from E_var (SoR only; for tests).
    deriv_dtc=False keeps the energy DTC residual but drops its force/virial
    derivative (F_var, V_var stay SoR-only).

    uq='blr' (default) is today's linear/GP posterior predictive variance,
    unchanged.  uq='pops' selects the linear-arm POPS misspecification predictive:
    the mean is untouched, the variance comes from the POPS weight-space posterior
    (pops_form in {'hypercube','ensemble'}, leverage_pct the leverage percentile),
    and aleatoric=True adds the label noise (sigma_q/w)^2.  POPS is linear-arm only
    (raises on M>0).  The POPS defaults ('hypercube' + aleatoric) match the upstream
    popsregression package default; 'ensemble' is the centred committee variance
    (the package's 'ensemble' posterior; see pops.pops_posterior)."""
    if uq == "blr":
        return _run_predict(_predict_fn(prob, dtc, deriv_dtc), theta, prob, ds_train, ds_test)
    if uq == "pops":
        return _run_predict_pops(theta, prob, ds_train, ds_test, pops_form, leverage_pct, aleatoric)
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
