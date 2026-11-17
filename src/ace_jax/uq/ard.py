"""Tempered ARD posterior: calibrated per-atom force uncertainty for linear ACE.

Spec: docs/dev/specs/2026-09-28-tempered-ard-uq-design.md.  For the linear model the weighted design
rows do not depend on the hyperparameters, so one statistics pass (G_q, b_q, y^T y_q, n_q per
quantity) gives the exact evidence for any h = (log sigma_E, log sigma_F, log sigma_V, a_k):

    log p(D|h) = -1/2 sum_q yy_q/s_q^2 + 1/2 b^T A^-1 b - 1/2 log|A| + 1/2 log|Lambda| - sum_q n_q log s_q
    A = sum_q G_q/s_q^2 + Lambda,  Lambda = Gamma^2 exp(a_k(j))

evaluated in the prior-scaled system S = D^-1 A D^-1 (D = diag Gamma), where the Gamma terms of
log|A| and log|Lambda| cancel: cond(A) reached 1e17 on the production Cantor basis, cond(S) 1e13.
"""
import json
import pathlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from ..fit.linear.newton import _projected_gradient, newton_polish  # noqa: F401  (re-exported: tests, callers)
from ..fit.linear.stats import quantities_of


SCHEMA = 3


_NEED3 = "this posterior predates schema 3; refit with --uq ard to get forces_q/forces_cov/forces_group"


E0_GROUP = 0      # body_col marker of a joint-E0 column: a fixed prior, no ARD scale


GP_GROUP = -1     # body_col marker of an inducing column (uq "ard-gp"): one shared scale a_GP, start 0


def gp_body_columns(body_col, M):
    """body_col of the joint design [B | k(B, B_M)]: the linear columns' groups, then M GP_GROUP columns."""
    return np.concatenate([np.asarray(body_col, int), np.full(int(M), GP_GROUP)])


class ARDStats(NamedTuple):
    G: tuple                 # (G_E, G_F, G_V) [joint] or (M,) combined at ls_fixed [sequential]
    b: tuple
    yy: np.ndarray           # (n_q,) [joint]; unused [sequential]
    n: np.ndarray            # (n_q,)
    ls_fixed: np.ndarray | None


class ARDQRStats(NamedTuple):
    """The ARD statistics in QR form (solver "qr", issue #136): data factors R (upper, R^T R = G) and
    c = Q^T (w y), so the evidence factors [diag sqrt(lambda); R D^-1] (kappa(R) = sqrt(kappa(S))) and
    never forms the Gram.  joint: R_q, c_q per quantity, rr_q = |w y_q|^2 - |c_q|^2 (the residual
    outside the column space: rho_q^2 when the rows were float32, else yy_q - c_q.c_q, clamped at 0),
    n_q.  sequential: one factor of the rows combined at the linear MAP noise scales ls_fixed."""
    R: tuple
    c: tuple
    rr: np.ndarray
    n: np.ndarray
    ls_fixed: np.ndarray | None


def ard_qr_stats(qs, ls_fixed=None):
    """ARDQRStats from QR statistics (a statistics object with `.factors`, or a sequence of
    `fit.linear.stats.Factor`: one per quantity).  ls_fixed None: joint (the per-quantity factors as they
    are, shared, not copied); else sequential, the rows R_q / sigma_q stacked and re-factored once to one
    (L, L) factor (a host QR of n_q L x L)."""
    fs = quantities_of(qs)[0]

    def rr(f):
        if f.rho is not None:
            return float(f.rho) ** 2
        c = np.asarray(f.c, np.float64)
        return max(float(f.yy) - float(c @ c), 0.0)
    n = np.array([float(f.n) for f in fs])
    if ls_fixed is None:
        return ARDQRStats(tuple(f.R for f in fs), tuple(f.c for f in fs), np.array([rr(f) for f in fs]), n, None)
    s = np.exp(np.asarray(ls_fixed, np.float64))
    A = np.concatenate([np.asarray(f.R, np.float64) / s[i] for i, f in enumerate(fs)])
    y = np.concatenate([np.asarray(f.c, np.float64) / s[i] for i, f in enumerate(fs)])
    Rt = np.linalg.qr(np.concatenate([A, y[:, None]], 1), mode="r")
    L = A.shape[1]
    return ARDQRStats((jnp.asarray(Rt[:L, :L]),), (jnp.asarray(Rt[:L, L]),), np.zeros(len(fs)), n,
                      np.asarray(ls_fixed, np.float64))


def joint_ard_stats(st):
    """Joint ARDStats from Gram statistics (a statistics object with `.grams`, or a sequence of
    `fit.linear.stats.Gram`): the arrays are shared, not copied, so a caller's cached statistics
    (pipeline objective) serve the full refit as they are."""
    gs = quantities_of(st)[0]
    return ARDStats(tuple(g.G for g in gs), tuple(g.b for g in gs), np.array([float(g.yy) for g in gs]),
                    np.array([float(g.n) for g in gs]), None)


class ARDEvidence:
    """log p(D|h) and its gradient in the prior-scaled system.  h = (log sigma_q [joint only], a_k), one
    noise scale per quantity of `stats`.  quantities: their labels, which name the noise scales `h0` reads
    from theta (`log_sigma_<label>`); default a potential fit's E, F, V.

    Columns with body_col == E0_GROUP (joint E0) have the fixed prior Lambda = Gamma^2 (scaled
    precision 1): no a_k, no group, and a constant 0 in 1/2 log|Lambda~|.

    root (uq "ard-gp"): the prior root R0 = blockdiag(diag Gamma, chol(K_MM)^T) of the joint design
    (prior_root.PriorRoot); S = R0^-T A R0^-1, and the GP_GROUP columns share one scale a_GP, so their
    prior is exp(a_GP) K_MM.  None: PriorRoot.diag(1 / gamma), the linear arm's D, evaluated verbatim.

    stats an ARDQRStats (solver "qr", linear arm only): the evidence, its derivatives and the posterior
    come from a QR of [diag sqrt(lambda); R D^-1] (`_ev_logev_qr`), whose factor R_S has kappa(R_S) =
    sqrt(cond(S)), instead of a Cholesky of S: the prior floor can then sit decades lower (`bounds`,
    `default_cond_max`).  ARDStats: the Cholesky of S, as before."""

    def __init__(self, stats, gamma, body_col, root=None, quantities=("E", "F", "V")):
        from ..fit.linear.prior_root import PriorRoot
        self.solver = "qr" if isinstance(stats, ARDQRStats) else "cholesky"
        self.joint = stats.ls_fixed is None
        self.ls_fixed = stats.ls_fixed
        self.body_col = np.asarray(body_col)
        self.root = PriorRoot.diag(1.0 / np.asarray(gamma)) if root is None else root
        if self.solver == "qr" and self.root.M:
            raise ValueError("ARDEvidence: the QR solver is the linear arm's; a GP-arm (ard-gp) root needs ARDStats")
        if len(np.asarray(gamma)) + self.root.M != len(self.body_col):
            raise ValueError(f"ARDEvidence: gamma has {len(np.asarray(gamma))} columns (+ {self.root.M} "
                             f"inducing), body_col {len(self.body_col)} (a joint-E0 problem: pass ard_gamma(prob); "
                             f"a GP-arm problem: gp_body_columns and prior_root)")
        fixed = self.body_col == E0_GROUP
        self.groups = tuple(int(g) for g in np.unique(self.body_col[~fixed]))
        # E0 columns index a trailing 1 (log 0 = 0) appended to exp(a)
        gidx = jnp.asarray(np.where(fixed, len(self.groups),
                                    np.searchsorted(np.asarray(self.groups), self.body_col)))
        self.dinv = jnp.asarray(self.root.dinv)
        # the unscaled G/b are held as given (no scaled copies: the Gram is L^2 per quantity, and a
        # pipeline caller shares them with its cache); D^-1 (sum_q w_q G_q) D^-1 is formed per call.
        # They are jit ARGUMENTS, not closure constants, which XLA would copy into the executable.
        if self.solver == "qr":
            self._data = (tuple(stats.R), tuple(stats.c), jnp.asarray(stats.rr), jnp.asarray(stats.n))
            self._vg, self._hvp = _evq_value_and_grad, _evq_hvp
        else:
            self._data = (tuple(stats.G), tuple(stats.b), jnp.asarray(stats.yy), jnp.asarray(stats.n))
            self._vg, self._hvp = _ev_value_and_grad, _ev_hvp
        self._gidx = gidx
        self._nls = len(self._data[0]) if self.joint else 0
        self.quantities = tuple(quantities)
        self._parts = lambda h: _ev_parts(h, self._data, self.root, self._gidx, self.joint)

    def h0(self, theta):
        """The MAP's own prior: a_k = -2 log sigma_c on the body orders (Lambda = Gamma^2 / sigma_c^2),
        a_GP = 0 on the inducing columns (Lambda = K_MM), and theta's noise scales [joint]."""
        a_blr = float(-2 * theta.log_sigma_c)
        ls = [float(getattr(theta, f"log_sigma_{q}")) for q in self.quantities] if self.joint else []
        return np.array(ls + [0.0 if g == GP_GROUP else a_blr for g in self.groups])

    def value_and_grad(self, h):
        v, g = self._vg(jnp.asarray(h, float), self._data, self.root, self._gidx, self.joint)
        return float(v), np.asarray(g)

    def hessian(self, h):
        """The exact Hessian (P, P) of log p(D|h): one forward-over-reverse column per hyperparameter
        (a Python loop of one compiled jvp: ~2x a gradient's L^2 workspace, not P x)."""
        h = jnp.asarray(h, float)
        P = h.shape[0]
        H = np.empty((P, P))
        for i in range(P):
            H[:, i] = np.asarray(self._hvp(h, jnp.zeros(P).at[i].set(1.0), self._data, self.root, self._gidx,
                                           self.joint))
        return 0.5 * (H + H.T)

    def bounds(self, h0, cond_max=None):
        """log sigma within h0 +- 3; a in [a_floor, 10] with a_floor = log(lambda_max(Ms(h0)) / cond_max):
        at the noise scales of h0 the prior keeps cond(S) <= ~cond_max (at the unfloored Cantor ARD
        optimum cond(S) was 4e17 and repeated evaluations differed by 0.3 nats).  Not a guarantee over
        the box: lambda_max(Ms) scales with 1/sigma_q^2, so with sigma_q down to its bound h0 - 3 cond(S)
        can reach ~e^6 cond_max (joint mode) -- `cond` reports it at the endpoint.  cond_max None: the
        solver's `default_cond_max` (QR resolves cond(S) ~ its square: kappa(R_S) = sqrt(cond(S)))."""
        cond_max = default_cond_max(self.solver) if cond_max is None else float(cond_max)
        h0 = jnp.asarray(h0, float)
        if self.solver == "qr":                              # Ms = X^T X of the scaled data rows X
            X = _evq_rows(h0, self._data, self.root, self.joint)[0]
            mv = lambda v: X.T @ (X @ v)
        else:
            Ms = self._parts(h0)[0]
            mv = lambda v: Ms @ v
        n = len(np.asarray(self.root.dinv)) + self.root.M
        v = jnp.ones(n) / np.sqrt(n)
        for _ in range(50):                                  # power iteration: lambda_max(Ms)
            v = mv(v); v = v / jnp.linalg.norm(v)
        lam_max = float(v @ mv(v))
        self.cond_max = cond_max
        self.a_floor = float(np.log(max(lam_max, 1e-300) / cond_max))
        k = self._nls
        lo = np.concatenate([np.asarray(h0[:k]) - 3.0, np.full(len(self.groups), self.a_floor)])
        hi = np.concatenate([np.asarray(h0[:k]) + 3.0, np.full(len(self.groups), 10.0)])
        self.lower, self.upper = lo, hi
        return lo, hi

    def factor(self, h):
        """(C, x): C lower triangular with C C^T = S(h) and a positive diagonal (Cholesky of S, or R_S^T
        from the QR), x = S^-1 R0^-T b the scaled posterior mean."""
        h = jnp.asarray(h, float)
        if self.solver == "qr":
            R, z = _evq_factor(h, self._data, self.root, self._gidx, self.joint)
            return R.T, solve_triangular(R, z, lower=False)
        Ms, bv, lam, _ = self._parts(h)
        c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
        return jnp.tril(c), cho_solve((c, low), bv)

    def cond(self, h, iters=50):
        """cond(S(h)) estimated by power iteration on S (lambda_max) and on S^-1 through its factor C
        (lambda_min): O(iters L^2) after one factorisation, deterministic."""
        C = self.factor(h)[0]
        Sv = lambda v: C @ (C.T @ v)
        Si = lambda w: solve_triangular(C, solve_triangular(C, w, lower=True), lower=True, trans="T")
        v = w = jnp.ones(C.shape[0]) / np.sqrt(C.shape[0])
        for _ in range(iters):
            v = Sv(v); v = v / jnp.linalg.norm(v)
            w = Si(w); w = w / jnp.linalg.norm(w)
        return float((v @ Sv(v)) / (w @ Sv(w)))

    def degenerate(self, h, g, gnoise=None):
        """Whether h is the degenerate endpoint of issue #136: every ARD scale a_k on the floor with the
        evidence's gradient pointing below it (or flat to within 10x its roundoff gnoise): the prior there is
        the conditioning guard's, not the evidence's choice, so the posterior is not the ARD optimum."""
        k = self._nls
        a, ga = np.asarray(h, float)[k:], np.asarray(g, float)[k:]
        if not len(a) or not hasattr(self, "lower"):
            return False
        tol = np.zeros_like(ga) if gnoise is None else 10 * np.nan_to_num(np.asarray(gnoise, float)[k:])
        return bool(np.all(np.isclose(a, self.lower[k:])) and np.all(ga <= tol))

    def sigmas(self, h):
        """Noise scales (one per quantity: sigma_E, sigma_F, sigma_V) of hyperparameters h: fitted [joint] or
        the fixed linear MAP ones [sequential]."""
        return np.exp(np.asarray(h[:self._nls], float)) if self.joint else np.exp(np.asarray(self.ls_fixed, float))


def _ev_parts(h, data, root, gidx, joint):
    """(S's data part R0^-T M R0^-1, R0^-T b, the scaled prior precisions, the constant) at h; root a
    prior_root.PriorRoot (M = 0: D^-1 M D^-1 and D^-1 b, as written before the GP arm)."""
    G, b, yy, nq = data
    nls = len(G) if joint else 0
    if joint:
        w = jnp.exp(-2 * h[:nls])
        M, bv = w[0] * G[0], w[0] * b[0]
        for i in range(1, nls):                              # left to right, as w0 G0 + w1 G1 + w2 G2
            M, bv = M + w[i] * G[i], bv + w[i] * b[i]
        const = -0.5 * jnp.sum(yy * w) - jnp.sum(nq * h[:nls])
    else:
        M, bv, const = G[0], b[0], 0.0
    lam = jnp.concatenate([jnp.exp(h[nls:]), jnp.ones(1)])[gidx]
    return root.gram(M), root.dual(bv), lam, const


def _ev_logev(h, data, root, gidx, joint):
    Ms, bv, lam, const = _ev_parts(h, data, root, gidx, joint)
    c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
    x = cho_solve((c, low), bv)
    return const + 0.5 * bv @ x - jnp.sum(jnp.log(jnp.diag(c))) + 0.5 * jnp.sum(jnp.log(lam))


# module-level: one executable per (joint, shapes) for every ARDEvidence, with G/b/root as arguments
_ev_value_and_grad = jax.jit(jax.value_and_grad(_ev_logev), static_argnames="joint")


def _ev_hvp_body(h, t, data, root, gidx, joint):
    return jax.jvp(lambda x: jax.grad(_ev_logev)(x, data, root, gidx, joint), (h,), (t,))[1]


_ev_hvp = jax.jit(_ev_hvp_body, static_argnames="joint")


def default_cond_max(solver):
    """The prior floor's cond(S) target when none is given (FitConfig.ard_cond_max None): 1e14 for the
    Cholesky of S, whose solves lose ~eps cond(S); 1e22 for the QR, whose factor has kappa(R_S) =
    sqrt(cond(S)) = 1e11.  On the Si o3d12 sequential evidence the QR gradient is smooth to cond(S)
    ~1e23 and visibly noisy (2-body component) from ~1e24; the Cholesky evidence is NaN from ~1e18."""
    return {"cholesky": 1e14, "qr": 1e22}[solver]


def _evq_rows(h, data, root, joint):
    """(X, y, const): the scaled data rows X = [R_q R0^-1 / sigma_q] (stacked [joint], one factor
    [sequential]), their target y = [c_q / sigma_q], and the h-dependent constant [joint]
    -1/2 sum_q rr_q / sigma_q^2 - sum_q n_q log sigma_q (0 [sequential])."""
    R, c, rr, nq = data
    if joint:
        k = len(R)
        s = jnp.exp(h[:k])
        X = jnp.concatenate([root.rows(R[i]) / s[i] for i in range(k)], 0)
        y = jnp.concatenate([c[i] / s[i] for i in range(k)])
        return X, y, -0.5 * jnp.sum(rr * jnp.exp(-2 * h[:k])) - jnp.sum(nq * h[:k])
    return root.rows(R[0]), c[0], 0.0


def _evq_qr(h, data, root, gidx, joint):
    """The R factor (L + 1, L + 1) of [[diag sqrt(lambda), 0]; [X, y]], and lambda, const."""
    X, y, const = _evq_rows(h, data, root, joint)
    nls = len(data[0]) if joint else 0
    lam = jnp.concatenate([jnp.exp(h[nls:]), jnp.ones(1)])[gidx]
    L = lam.shape[0]
    A = jnp.concatenate([jnp.concatenate([jnp.diag(jnp.sqrt(lam)), jnp.zeros((L, 1))], 1),
                         jnp.concatenate([X, y[:, None]], 1)], 0)
    return jnp.linalg.qr(A, mode="r"), lam, const


def _ev_logev_qr(h, data, root, gidx, joint):
    """_ev_logev from the QR of [diag sqrt(lambda); X | 0; y]: z = Rt[:L, L] has |z|^2 = b~^T S^-1 b~, and
    Rt[L, L]^2 is the residual |y|^2 - |z|^2 without the cancellation [joint, where the constant
    carries the out-of-span residual rr_q; sequential has no y'y and keeps +1/2 |z|^2, as the Cholesky
    form keeps +1/2 b~^T S^-1 b~].  log|S| = 2 sum log|diag R_S|."""
    Rt, lam, const = _evq_qr(h, data, root, gidx, joint)
    L = lam.shape[0]
    logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(Rt)[:L])))
    fit = -0.5 * Rt[L, L] ** 2 if joint else 0.5 * Rt[:L, L] @ Rt[:L, L]
    return const + fit - logdet + 0.5 * jnp.sum(jnp.log(lam))


def _evq_factor_body(h, data, root, gidx, joint):
    """(R_S, z) with a positive diagonal: R_S^T R_S = S, R_S x = z for the scaled mean x."""
    Rt, lam, _ = _evq_qr(h, data, root, gidx, joint)
    L = lam.shape[0]
    sg = jnp.where(jnp.diag(Rt)[:L] < 0, -1.0, 1.0)
    return Rt[:L, :L] * sg[:, None], Rt[:L, L] * sg


_evq_value_and_grad = jax.jit(jax.value_and_grad(_ev_logev_qr), static_argnames="joint")


_evq_factor = jax.jit(_evq_factor_body, static_argnames="joint")


def _evq_hvp_body(h, t, data, root, gidx, joint):
    return jax.jvp(lambda x: jax.grad(_ev_logev_qr)(x, data, root, gidx, joint), (h,), (t,))[1]


_evq_hvp = jax.jit(_evq_hvp_body, static_argnames="joint")


def fit_ard(ev, h0, cond_max=None, maxiter=500, polish=True, start=None):
    """Type-II ML by L-BFGS-B, bounded (ev.bounds), on the objective relative to its start and divided
    by the initial gradient norm: L-BFGS-B's first bounded step is the full gradient (~3e3 on the
    Cantor problem), which lands on the box corner; a non-finite point is a rejected step.  Then
    (polish) `newton_polish` converges the endpoint with the exact Hessian: L-BFGS-B stops at a
    roundoff-dependent point (a line-search failure on the noisy objective, or a loose scaled gtol),
    which two runs on different BLAS / batch layouts reach differently.  The box comes from h0;
    `start` (default h0) is where L-BFGS-B starts in it.  info["cond_S"]: cond(S) at the endpoint
    (ev.cond), with info["cond_max"] the floor's target (exceeded when sigma ran down, see ev.bounds);
    info["newton"]["noise"] / ["gnoise"]: the measured roundoff of the evidence and of each gradient
    component at the returned endpoint (the resolution of v and of its gradient).  cond_max None: the
    solver's `default_cond_max`.  info["degenerate"]: every a_k on the floor with the gradient pointing
    below it (`ARDEvidence.degenerate`; issue #136) -- the posterior there is the guard's, not ARD's."""
    from scipy.optimize import minimize
    lo, hi = ev.bounds(h0, cond_max)
    cond_max = getattr(ev, "cond_max", cond_max)
    x0 = np.clip(np.asarray(h0 if start is None else start, float), lo, hi)
    v0, g0 = ev.value_and_grad(x0)
    gs = max(1.0, float(np.linalg.norm(g0)))

    def f(z):
        v, g = ev.value_and_grad(z)
        if not (np.isfinite(v) and np.all(np.isfinite(g))):
            return 1e30, np.zeros_like(z)
        return -(v - v0) / gs, -g / gs

    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                 options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6})
    x, success, message = np.clip(r.x, lo, hi), bool(r.success), str(r.message)
    info = {"lbfgs_message": message, "lbfgs_success": success}
    if polish:
        x, pol = newton_polish(ev, x, lo, hi)
        pol.pop("hessian", None)                     # an array; info is written as JSON
        info["newton"] = pol
        success = pol["converged"]
        message = f"{message}; newton: {pol['message']}"
    v, g = ev.value_and_grad(x)
    deg = ev.degenerate(x, g, info.get("newton", {}).get("gnoise")) if hasattr(ev, "degenerate") else False
    return x, v, {**info, "success": success, "message": message, "nit": int(r.nit), "gain": v - v0,
                  "at_bound": ((np.isclose(x, lo)) | (np.isclose(x, hi))).tolist(),
                  "cond_S": ev.cond(x) if hasattr(ev, "cond") else None, "cond_max": float(cond_max),
                  "solver": getattr(ev, "solver", "cholesky"), "a_floor": getattr(ev, "a_floor", None),
                  "degenerate": deg}


def laplace_hypers(ev, h, eps=1e-3):
    """Diagnostic: Laplace approximation of the hyperparameter posterior at h (central differences of
    the exact gradient); the covariance uses interior coordinates only and is PSD by construction."""
    P = len(h)
    H = np.zeros((P, P))
    for i in range(P):
        e = np.zeros(P); e[i] = eps
        H[:, i] = (ev.value_and_grad(h + e)[1] - ev.value_and_grad(h - e)[1]) / (2 * eps)
    H = 0.5 * (H + H.T)
    interior = ~(np.isclose(h, ev.lower) | np.isclose(h, ev.upper))
    cov = np.zeros((P, P))
    w, V = np.linalg.eigh(-H[np.ix_(interior, interior)])
    cov[np.ix_(interior, interior)] = (V / np.clip(w, 1e-12, None)) @ V.T
    return {"std": np.sqrt(np.diag(cov)), "eigs": np.linalg.eigvalsh(-H), "cov": cov,
            "interior": interior.tolist()}


class ARDPosterior(NamedTuple):
    mean: np.ndarray          # (L,) posterior mean coefficients (the readout)
    chol: jax.Array           # (L, L) lower Cholesky factor of S = D^-1 A D^-1 (float64 on the device;
                              # numpy after `load` -- ACECalculator moves it to the device once)
    dinv: np.ndarray          # (L,) 1 / Gamma
    kappa: float              # temperature (sigma inflation)
    h: np.ndarray
    groups: tuple
    body_col: np.ndarray
    meta: dict                # n_B, n_pair, NZ, rcut, elements
    Q: jax.Array | None = None    # (L, n_cfg) S^-1 G~: configuration-clustered sandwich factor (device)
    lam: float = 1.0              # sandwich scale (fitted like kappa)
    R: object = None              # (L, r) schema-3 shape factor (device float64)
    force_shape: str = "iso"
    eps: float = 1e-3
    group_consts: dict | None = None    # {"r1", "z_star", "edges"}
    group_table: dict | None = None     # GroupTable.to_dict()
    cal: dict | None = None             # arrays scores f32, groups i8, cfg i64, src i8
    support: dict | None = None
    root: object = None                 # prior_root.PriorRoot with the GP block (uq "ard-gp"); None: diag(dinv)
    gp_theta: np.ndarray | None = None  # (10,) the theta the GP block's rows and K_MM are evaluated at

    @property
    def prior_root(self):
        """The prior root R0 of this posterior: the stored GP-arm root, else the linear arm's diag(dinv)."""
        from ..fit.linear.prior_root import PriorRoot
        return self.root if self.root is not None else PriorRoot.diag(self.dinv)

    def _tab(self, groups=None, need_groups=True):
        if self.group_table is None:
            raise ValueError(_NEED3)
        if need_groups and groups is None:
            raise ValueError("groups required for a schema-3 posterior; use groups_of(batch) (per node of the "
                             "batch) for the per-atom group ids")
        return self.group_table

    def groups_of(self, batch):
        from .conformal import assign_groups, shell_features
        if self.group_consts is None:
            raise ValueError(_NEED3)
        gc = self.group_consts
        z, d = shell_features(batch, gc["r1"])
        return assign_groups(z, d, gc["z_star"], np.asarray(gc["edges"], float))

    @property
    def n_e0(self):
        """Joint-E0 columns (one per species) after the readout in mean/chol/dinv/R/Q; 0 for a
        prefit-E0 posterior and every file written before joint E0 under ARD."""
        return int(self.meta.get("e0_cols", 0))

    def _force_width(self, n_cols):
        """Width check of force rows: the full design (len(dinv)) or the readout alone -- the model
        file's rows (ACECalculator), whose E0 columns would be zero on a force row anyway.  An ard-gp
        posterior takes the full joint rows only (the E0 columns sit before the inducing columns)."""
        if self.root is not None and self.root.M:
            if n_cols != self.root.width:
                raise ValueError(f"force rows have {n_cols} columns; this ard-gp posterior takes the joint rows "
                                 f"[B | k(B, B_M)] of width {self.root.width} (GPCalculator builds them)")
            return n_cols
        full = len(np.asarray(self.dinv))
        if n_cols not in (full, full - self.n_e0):
            raise ValueError(f"force rows have {n_cols} columns; the posterior has {full}"
                             + (f" ({full - self.n_e0} readout + {self.n_e0} E0)" if self.n_e0 else ""))
        return n_cols

    def atom_shape(self, Frows, chunk=None):
        """Unscaled per-atom shape V (N, 3, 3) from force rows (N, 3, L), in chunks of atoms.  L may
        omit the trailing joint-E0 columns (zero on every force row: R, Q and dinv are truncated,
        the kappa path pads each chunk with zeros), with identical results."""
        from .jackknife import atom_chunk, atom_shape
        L0 = self._force_width(Frows.shape[-1])
        if self.root is not None and self.root.M:                # ard-gp: u = R0^-T phi^T, full joint rows
            return self._atom_shape_gp(Frows, chunk)
        if self.R is not None:
            return atom_shape(self.R[:L0], np.asarray(self.dinv)[:L0], Frows, chunk)
        if self.Q is not None:                                   # schema 2: uncentred sandwich factor
            return atom_shape(self.Q[:L0], np.asarray(self.dinv)[:L0], Frows, chunk)
        N, L = Frows.shape[0], len(np.asarray(self.dinv))        # kappa: V = u^T S^-1 u
        c = atom_chunk(L, chunk)
        dinv = jnp.asarray(self.dinv, jnp.float64)
        chol = jnp.asarray(self.chol)
        out = np.empty((N, 3, 3))
        for i in range(0, N, c):
            U = _pad_cols(jnp.asarray(Frows[i:i + c], jnp.float64), L) * dinv[None, None, :]
            n = U.shape[0]
            W = jnp.swapaxes(solve_triangular(chol, U.reshape(-1, L).T, lower=True), 0, 1).reshape(n, 3, -1)
            out[i:i + c] = np.asarray(jnp.einsum("nar,nbr->nab", W, W))
        return out

    def _atom_shape_gp(self, Frows, chunk=None):
        from .jackknife import atom_chunk, atom_shape
        if self.R is not None:
            return atom_shape(self.R, self.root, Frows, chunk)
        if self.Q is not None:
            raise ValueError("an ard-gp posterior has no legacy sandwich factor Q")
        N, L = Frows.shape[0], self.root.width                   # kappa: V = u^T S^-1 u
        c = atom_chunk(L, chunk)
        chol = jnp.asarray(self.chol)
        out = np.empty((N, 3, 3))
        for i in range(0, N, c):
            U = self.root.rows(jnp.asarray(Frows[i:i + c], jnp.float64))
            n = U.shape[0]
            W = jnp.swapaxes(solve_triangular(chol, U.reshape(-1, L).T, lower=True), 0, 1).reshape(n, 3, -1)
            out[i:i + c] = np.asarray(jnp.einsum("nar,nbr->nab", W, W))
        return out

    def forces_cov(self, Frows, groups):
        return self.served(Frows, groups, ("forces_cov",))["forces_cov"]

    def forces_q(self, Frows, groups):
        return self.served(Frows, groups, ("forces_q",))["forces_q"]

    def forces_q_mahal(self, groups):
        return np.asarray(self._tab(groups)["q"], float)[groups]

    def served(self, Frows, groups, which=("forces_std", "forces_cov", "forces_q")):
        """The schema-3 per-atom quantities in `which` (forces_std, forces_cov, forces_q, forces_q_mahal)
        from ONE shape evaluation V of the force rows (N, 3, L).  forces_q = q_g x the region's radius
        (sqrt(v/3) iso, sqrt(lambda_max(V + eps v/3 I)) aniso): inf wherever q_g = inf (the coverage is
        not attainable from the calibration set -- even at v = 0, never 0 x inf = NaN), 0 where v = 0."""
        need = set(which) & {"forces_std", "forces_cov", "forces_q"}
        return self.served_from_V(self.atom_shape(Frows) if need else None, groups, which)

    def served_from_V(self, V, groups, which=("forces_std", "forces_cov", "forces_q")):
        """`served` from a precomputed unscaled shape V (N, 3, 3) -- the rows' `atom_shape` or the
        committee path (`jackknife.committee_shape`); V may be None when only forces_q_mahal is wanted."""
        t = self._tab(groups)
        lam = np.asarray(t["lam_rms"], float)[groups]
        q = np.asarray(t["q"], float)[groups]
        which = set(which)
        out = {}
        if which & {"forces_std", "forces_cov", "forces_q"}:
            V = np.asarray(V)
            v = np.trace(V, axis1=1, axis2=2)
            if "forces_std" in which:
                out["forces_std"] = lam * np.sqrt(np.maximum(v, 0.0))
            if "forces_cov" in which:
                out["forces_cov"] = lam[:, None, None] ** 2 * V
            if "forces_q" in which:
                if self.force_shape == "aniso":
                    rad = np.linalg.eigvalsh(V + self.eps * (v / 3)[:, None, None] * np.eye(3)).max(1)
                else:
                    rad = v / 3
                rad = np.sqrt(np.maximum(rad, 0.0))
                fin = np.isfinite(q)
                out["forces_q"] = np.where(fin, np.where(fin, q, 0.0) * rad, np.inf)
        if "forces_q_mahal" in which:
            out["forces_q_mahal"] = q
        return out

    def var_rows(self, Phi, chunk=4096, force_rows=False):
        """Untempered posterior variance phi A^-1 phi^T of each row of Phi (n, L).  force_rows: Phi may
        omit the joint-E0 columns (zero on a force row; padded per chunk)."""
        if self.chol is None:
            raise ValueError("this posterior was saved without its Cholesky factor (a sandwich-shape posterior "
                             "serves forces from R alone): the untempered posterior variances need the fit's "
                             "in-memory posterior")
        c = jnp.asarray(self.chol, jnp.float64)
        L = self.prior_root.width
        if force_rows:
            self._force_width(Phi.shape[-1])
        out = []
        for i in range(0, len(Phi), chunk):
            P = jnp.asarray(Phi[i:i + chunk])
            P = _pad_cols(P, L) if force_rows else P
            if self.root is not None and self.root.M:            # ard-gp: phi R0^-1 (L = dinv's width + M)
                v = solve_triangular(c, self.root.rows(P).T, lower=True)
            else:
                v = solve_triangular(c, (P * jnp.asarray(self.dinv)[None, :]).T, lower=True)
            out.append(np.asarray(jnp.sum(v * v, axis=0)))
        return np.concatenate(out) if out else np.zeros(0)

    def misspec_var_rows(self, Phi, chunk=4096, own=None):
        """Unscaled cluster-sandwich variance ||Q^T (D^-1 phi)||^2 = phi A^-1 M A^-1 phi^T per row.

        own: optional (n,) Q column of each row's own training configuration (-1: none); returns
        (all, without_own), where without_own drops that one cluster's term v_own^2 -- the variance a
        genuinely new configuration would get, used only to fit lam on held-out TRAINING configs."""
        L0 = self._force_width(Phi.shape[-1])    # Q, dinv truncated to it: rows may omit the (zero) E0 columns
        Q = jnp.asarray(self.Q, jnp.float64)[:L0]
        dinv = jnp.asarray(self.dinv)[:L0]
        out, loo = [], []
        for i in range(0, len(Phi), chunk):
            v = (jnp.asarray(Phi[i:i + chunk]) * dinv[None, :]) @ Q
            tot = jnp.sum(v * v, axis=1)
            out.append(np.asarray(tot))
            if own is not None:
                o = jnp.asarray(own[i:i + chunk])
                # drop the own column before squaring (no tot - v_own^2 cancellation; >= 0 by construction)
                w = jnp.where(jnp.arange(v.shape[1])[None, :] == o[:, None], 0.0, v)
                loo.append(np.asarray(jnp.sum(w * w, axis=1)))
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
        return cat(out) if own is None else (cat(out), cat(loo))

    def force_var_rows(self, Phi):
        """The served (calibrated) variance of each force-component row: lam^2 x sandwich when Q is
        set, else kappa^2 x the epistemic posterior variance."""
        self._force_width(Phi.shape[-1])
        if self.Q is not None:
            return self.lam ** 2 * self.misspec_var_rows(Phi)
        return self.kappa ** 2 * self.var_rows(Phi, force_rows=True)

    def forces_std(self, Frows, groups=None):
        """Per-atom force std from force rows (N, 3, L), numpy or device.  With a schema-3 group table
        (groups required): lam_rms[g] sqrt(tr V); else sqrt(sum_c force_var_rows) (schema 1/2)."""
        if self.group_table is not None:
            return self.served(Frows, groups, ("forces_std",))["forces_std"]
        v = self.force_var_rows(Frows.reshape(-1, Frows.shape[-1])).reshape(-1, 3)
        return np.sqrt(np.maximum(v.sum(1), 0.0))

    def save(self, path, dtype=np.float32):
        """The Cholesky factor of S (L^2) is written only when a served path reads it: the kappa shape (no R,
        no Q), and then packed (its lower triangle, half the bytes).  The sandwich shapes serve from R or Q
        alone, so their files leave it out (1.14 GB -> 0.24 GB on the Cantor basis); load gives chol None."""
        chol = {}
        if self.R is None and self.Q is None and self.chol is not None:
            c = np.asarray(self.chol, dtype)
            chol = {"chol_packed": c[np.tril_indices(c.shape[0])]}
        np.savez(path, mean=self.mean, **chol, dinv=self.dinv, kappa=self.kappa,
                 h=self.h, groups=np.asarray(self.groups), body_col=self.body_col, schema=SCHEMA,
                 meta_json=np.frombuffer(json.dumps(self.meta).encode(), np.uint8),
                 lam=self.lam, **({} if self.Q is None else {"Q": np.asarray(self.Q, dtype)}),
                 **self._extra_arrays(dtype))

    def _extra_arrays(self, dtype):
        js = lambda d: np.frombuffer(json.dumps(d).encode(), np.uint8)
        out = {"force_shape": np.array(self.force_shape), "eps": self.eps}
        if self.root is not None and self.root.M:
            # always float64: M x M is small, and a float32 chol(K_MM)^T loses the jitter's scale in the solves
            out["gp_U"] = np.asarray(self.root.U, np.float64)
            out["gp_theta"] = np.asarray(self.gp_theta, np.float64)
        if self.R is not None:
            out["R"] = np.asarray(self.R, dtype)
        if self.group_consts is not None:
            out["group_consts_json"] = js(self.group_consts)
        if self.group_table is not None:
            from .conformal import json_safe
            out["group_table_json"] = js(json_safe(self.group_table))       # strict JSON: inf -> "inf"
        if self.cal is not None:
            for k, t in (("scores", np.float32), ("groups", np.int8), ("cfg", np.int64), ("src", np.int16)):
                out[f"cal_{k}"] = np.asarray(self.cal[k], t)
        if self.support is not None:
            from .support import flatten_support
            out.update(flatten_support(self.support, dtype))
        return out

    @staticmethod
    def load(path):
        z = np.load(pathlib.Path(path))
        from .conformal import restore_table
        from ..fit.linear.prior_root import PriorRoot
        from .support import unflatten_support
        js = lambda k: json.loads(bytes(z[k]).decode()) if k in z.files else None
        if int(z["schema"]) not in (1, 2, 3):
            raise ValueError(f"unsupported posterior schema {int(z['schema'])}")
        if "variance" in z.files and str(z["variance"]) == "dtc":
            raise ValueError(f"{path} is an --ard-variance dtc posterior: that shape failed its acceptance and was "
                             "removed; refit with --uq ard-gp (sandwich)")
        if "chol_packed" in z.files:            # the lower triangle, row-major (np.tril_indices)
            k = len(z["chol_packed"])
            n = int(round((np.sqrt(8 * k + 1) - 1) / 2))    # n(n+1)/2 = k: L, or L + M for an ard-gp posterior
            chol = np.zeros((n, n))
            chol[np.tril_indices(n)] = z["chol_packed"]
        else:                                   # a dense factor (files before the packed form), or none
            chol = z["chol"].astype(np.float64) if "chol" in z.files else None
        return ARDPosterior(z["mean"], chol, z["dinv"], float(z["kappa"]), z["h"],
                            tuple(int(g) for g in z["groups"]), z["body_col"],
                            json.loads(bytes(z["meta_json"]).decode()),
                            Q=z["Q"].astype(np.float64) if "Q" in z.files else None,
                            lam=float(z["lam"]) if "lam" in z.files else 1.0,
                            R=jnp.asarray(z["R"], jnp.float64) if "R" in z.files else None,
                            force_shape=str(z["force_shape"]) if "force_shape" in z.files else "iso",
                            eps=float(z["eps"]) if "eps" in z.files else 1e-3,
                            group_consts=js("group_consts_json"), group_table=restore_table(js("group_table_json")),
                            cal={k: z[f"cal_{k}"] for k in ("scores", "groups", "cfg", "src")}
                            if "cal_scores" in z.files else None,
                            support=unflatten_support({k[8:]: z[k] for k in z.files if k.startswith("support_")})
                            if any(k.startswith("support_") for k in z.files) else None,
                            root=PriorRoot(z["dinv"], jnp.asarray(z["gp_U"])) if "gp_U" in z.files else None,
                            gp_theta=np.asarray(z["gp_theta"]) if "gp_theta" in z.files else None)


def _pad_cols(X, width):
    """X (..., n) with trailing zero columns up to width (no copy when already that wide)."""
    k = width - X.shape[-1]
    return X if k == 0 else jnp.concatenate([X, jnp.zeros(X.shape[:-1] + (k,), X.dtype)], -1)


def ard_posterior(ev, h, kappa, meta, theta=None):
    """The ARD posterior at h.  A GP-arm evidence (ev.root.M > 0, uq "ard-gp") also stores its prior root
    and theta (the kernel hyperparameters the inducing columns were built at); pass theta then."""
    c, x = ev.factor(h)
    keep = {k: meta[k] for k in ("n_B", "n_pair", "NZ", "rcut", "elements") if k in meta}
    if "NZ" not in keep and "elements" in keep:          # model meta carries elements, not NZ
        keep["NZ"] = len(keep["elements"])
    n_e0 = int(np.sum(ev.body_col == E0_GROUP))
    if n_e0:                                             # absent (0) on prefit-E0 posteriors, as before
        keep["e0_cols"] = n_e0
    gp = {}
    if ev.root.M:
        if theta is None:
            raise ValueError("ard_posterior: a GP-arm (ard-gp) posterior needs theta, the hyperparameters "
                             "of its inducing columns")
        keep["gp_cols"] = ev.root.M
        gp = {"root": ev.root, "gp_theta": np.stack([np.asarray(v, np.float64) for v in theta])}
    # chol stays a float64 device array: var_rows runs once per batch (predict_ard, _val_errors), and
    # jnp.asarray of a numpy factor would re-upload the L x L matrix (1.8 GB at L = 15k) on every call
    return ARDPosterior(np.asarray(ev.root.primal(x)), jnp.asarray(c, jnp.float64), np.asarray(ev.dinv),
                        float(kappa), np.asarray(h, float), ev.groups, ev.body_col, keep, **gp)


def sandwich_factor(post, G):
    """Q = S^-1 G~ (L, n_cfg), so that phi A^-1 M A^-1 phi^T = ||Q^T (D^-1 phi)||^2.  A float64 device
    array: its consumers (misspec_var_rows, per batch) use it there, no host round trip."""
    c = jnp.asarray(post.chol, jnp.float64)
    return cho_solve((c, True), jnp.asarray(G, jnp.float64))


def kappa_closed_form(e2, s2):
    """NLL-optimal kappa for dF ~ N(0, kappa^2 s^2/3 I_3) per atom: kappa^2 = mean(e2 / (s2/3)) / 3."""
    e2, s2 = np.asarray(e2, float), np.asarray(s2, float)
    ok = s2 > 0
    return float(np.sqrt(np.mean(e2[ok] / (s2[ok] / 3.0)) / 3.0))


def _force_nll(e2, s2, kappa):
    """Mean per-atom Gaussian NLL of dF ~ N(0, kappa^2 s2/3 I_3) (constant 3/2 log 2 pi included)."""
    v = kappa ** 2 * s2 / 3.0
    return float(np.mean(e2 / (2 * v) + 1.5 * np.log(2 * np.pi * v)))


def conformal_scores(e, V, force_shape, eps):
    """iso: s = |e| / sqrt(v/3); aniso: s = sqrt(e^T (V + eps (v/3) I)^-1 e), v = tr V."""
    v = np.trace(V, axis1=1, axis2=2)
    if force_shape == "aniso":
        M = V + eps * (v / 3)[:, None, None] * np.eye(3)
        return np.sqrt(np.einsum("na,na->n", e, np.linalg.solve(M, e[:, :, None])[:, :, 0]))
    return np.sqrt(np.sum(e * e, 1) / (v / 3))
