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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from .newton import _projected_gradient, newton_polish  # noqa: F401  (re-exported: tests, callers)

SCHEMA = 3
_NEED3 = "this posterior predates schema 3; refit with --uq ard to get forces_q/forces_cov/forces_group"


E0_GROUP = 0      # body_col marker of a joint-E0 column: a fixed prior, no ARD scale
GP_GROUP = -1     # body_col marker of an inducing column (uq "ard-gp"): one shared scale a_GP, start 0


def gp_body_columns(body_col, M):
    """body_col of the joint design [B | k(B, B_M)]: the linear columns' groups, then M GP_GROUP columns."""
    return np.concatenate([np.asarray(body_col, int), np.full(int(M), GP_GROUP)])


def body_order_columns(meta, cfg):
    """Body order (2, 3, 4, ...) of every design column, in the `rows._place` layout: species blocks
    of the n_B many-body columns (correlation order nu -> body order nu + 1), then species blocks of
    the n_pair pair columns (body order 2), then (cfg.e0_cols, joint E0) one E0_GROUP column per species:
    not an ACE basis function, so it gets no ARD scale and keeps its fixed prior (`ard_gamma`)."""
    order_B = np.array([len(x) for x in meta["nnll"]]) + 1
    e0 = np.full(cfg.NZ if getattr(cfg, "e0_cols", False) else 0, E0_GROUP)
    return np.concatenate([np.tile(order_B, cfg.NZ), np.full(cfg.n_pair * cfg.NZ, 2), e0]).astype(int)


def ard_gamma(prob):
    """The prior scale Gamma (len_basis,) of the ARD evidence: the readout's smoothness prior prob.gamma,
    then (joint E0) sqrt(e0_prec) per E0 column -- with an E0_GROUP body_col, Lambda = Gamma^2 there is
    exactly the BLR fit's fixed E0 precision (wide, or pinned by an isolated training atom)."""
    g = np.asarray(prob.gamma, np.float64)
    if getattr(prob.cfg, "e0_cols", False):
        g = np.concatenate([g, np.sqrt(np.asarray(prob.e0_prec, np.float64))])
    return g


class ARDStats(NamedTuple):
    G: tuple                 # (G_E, G_F, G_V) [joint] or (M,) combined at ls_fixed [sequential]
    b: tuple
    yy: np.ndarray           # (3,) [joint]; unused [sequential]
    n: np.ndarray            # (3,)
    ls_fixed: np.ndarray | None


def ard_statistics(theta, prob, ds, mode, columns="linear"):
    """joint: per-quantity statistics (3 L^2 matrices).  sequential: the combined Gram at the
    linear MAP noise scales, accumulated in one pass (1 L^2 matrix) -- the low-memory mode.
    columns "linear" (uq "ard"): the L linear columns only -- a hybrid problem's inducing columns never
    enter, and the joint Gram is hyperparameter-independent.  columns "joint" (uq "ard-gp"): the joint
    design [B | k_theta(B, B_M)] at theta (Dt = L + M), streamed by stats.sufficient_statistics; sequential
    mode then combines its per-quantity Grams at theta's noise scales (3 Dt^2 transiently)."""
    from .rows import ROWS_EDGE_BUDGET
    if columns == "joint":
        st = _joint_statistics_jit(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds, ROWS_EDGE_BUDGET)
        if mode == "joint":
            return joint_ard_stats(st)
        if mode != "sequential":
            raise ValueError(f"ard mode must be 'joint' or 'sequential', got {mode!r}")
        ls = np.array([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
        w = np.exp(-2 * ls)
        M = sum(w[i] * getattr(st, f"G_{q}") for i, q in enumerate("EFV"))
        bv = sum(w[i] * getattr(st, f"b_{q}") for i, q in enumerate("EFV"))
        return ARDStats((M,), (bv,), np.zeros(3), np.zeros(3), ls)
    if columns != "linear":
        raise ValueError(f"ard statistics columns must be 'linear' or 'joint', got {columns!r}")
    if mode == "joint":
        return joint_ard_stats(_linear_statistics_jit(prob.model, prob.cfg, ds, ROWS_EDGE_BUDGET))
    if mode != "sequential":
        raise ValueError(f"ard mode must be 'joint' or 'sequential', got {mode!r}")
    ls = np.array([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
    M, bv = _sequential_stats_jit(prob.model, prob.cfg, ds, jnp.asarray(np.exp(-ls)), ROWS_EDGE_BUDGET)
    return ARDStats((M,), (bv,), np.zeros(3), np.zeros(3), ls)


# The stage's compiled programs are module-level eqx.filter_jit functions of (model, cfg, data, ...):
# keyed on the model's structure, cfg and the shapes, with the arrays as arguments, so a repeated
# stage (bench sweeps, tests) reuses them -- a fresh jit closure per call compiled again every time
# and leaked its executable's memory maps (vm.max_map_count after ~17 stages).  `budget`
# (rows.ROWS_EDGE_BUDGET, read while tracing) is a static cache key only.
def _linear_statistics_body(model, cfg, ds, budget):
    from .stats import linear_statistics
    return linear_statistics(model, cfg, ds)


def _sequential_stats_body(model, cfg, ds, inv, budget):
    from .rows import linear_rows_bounded
    L = cfg.len_basis

    def body(acc, batch):
        M, bv = acc
        r = linear_rows_bounded(model, cfg, batch)
        for k, (Phi, y, w) in enumerate(((r.E, batch.y_E, batch.w_E),
                                         (r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3)),
                                         (r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6)))):
            Pw = Phi * (w * inv[k])[:, None]
            M = M + Pw.T @ Pw
            bv = bv + Pw.T @ (y * w * inv[k])
        return (M, bv), None

    return jax.lax.scan(jax.checkpoint(body), (jnp.zeros((L, L)), jnp.zeros(L)), ds)[0]


_linear_statistics_jit = eqx.filter_jit(_linear_statistics_body)


def _joint_statistics_body(theta, spec, model, ind, cfg, ds, budget):
    from .stats import sufficient_statistics
    return sufficient_statistics(theta, spec, model, ind, cfg, ds)


_joint_statistics_jit = eqx.filter_jit(_joint_statistics_body)
_sequential_stats_jit = eqx.filter_jit(_sequential_stats_body)


def joint_ard_stats(st):
    """Joint ARDStats from `stats.linear_statistics` output (a `Stats`): the arrays are shared, not
    copied, so a caller's cached statistics (pipeline objective) serve the full refit as they are."""
    return ARDStats(tuple(getattr(st, f"G_{q}") for q in "EFV"), tuple(getattr(st, f"b_{q}") for q in "EFV"),
                    np.array([float(getattr(st, f"yy_{q}")) for q in "EFV"]),
                    np.array([float(getattr(st, f"n_{q}")) for q in "EFV"]), None)


class ARDEvidence:
    """log p(D|h) and its gradient in the prior-scaled system.  h = (log sigma_q [joint only], a_k).

    Columns with body_col == E0_GROUP (joint E0) have the fixed prior Lambda = Gamma^2 (scaled
    precision 1): no a_k, no group, and a constant 0 in 1/2 log|Lambda~|.

    root (uq "ard-gp"): the prior root R0 = blockdiag(diag Gamma, chol(K_MM)^T) of the joint design
    (prior_root.PriorRoot); S = R0^-T A R0^-1, and the GP_GROUP columns share one scale a_GP, so their
    prior is exp(a_GP) K_MM.  None: PriorRoot.diag(1 / gamma), the linear arm's D, evaluated verbatim."""

    def __init__(self, stats, gamma, body_col, root=None):
        from .prior_root import PriorRoot
        self.joint = stats.ls_fixed is None
        self.ls_fixed = stats.ls_fixed
        self.body_col = np.asarray(body_col)
        self.root = PriorRoot.diag(1.0 / np.asarray(gamma)) if root is None else root
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
        self._data = (tuple(stats.G), tuple(stats.b), jnp.asarray(stats.yy), jnp.asarray(stats.n))
        self._gidx = gidx
        self._nls = 3 if self.joint else 0
        self._parts = lambda h: _ev_parts(h, self._data, self.root, self._gidx, self.joint)

    def h0(self, theta):
        """The MAP's own prior: a_k = -2 log sigma_c on the body orders (Lambda = Gamma^2 / sigma_c^2),
        a_GP = 0 on the inducing columns (Lambda = K_MM), and theta's noise scales [joint]."""
        a_blr = float(-2 * theta.log_sigma_c)
        ls = [float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"] if self.joint else []
        return np.array(ls + [0.0 if g == GP_GROUP else a_blr for g in self.groups])

    def value_and_grad(self, h):
        v, g = _ev_value_and_grad(jnp.asarray(h, float), self._data, self.root, self._gidx, self.joint)
        return float(v), np.asarray(g)

    def hessian(self, h):
        """The exact Hessian (P, P) of log p(D|h): one forward-over-reverse column per hyperparameter
        (a Python loop of one compiled jvp: ~2x a gradient's L^2 workspace, not P x)."""
        h = jnp.asarray(h, float)
        P = h.shape[0]
        H = np.empty((P, P))
        for i in range(P):
            H[:, i] = np.asarray(_ev_hvp(h, jnp.zeros(P).at[i].set(1.0), self._data, self.root, self._gidx,
                                         self.joint))
        return 0.5 * (H + H.T)

    def bounds(self, h0, cond_max):
        """log sigma within h0 +- 3; a in [a_floor, 10] with a_floor = log(lambda_max(Ms(h0)) / cond_max):
        at the noise scales of h0 the prior keeps cond(S) <= ~cond_max (at the unfloored Cantor ARD
        optimum cond(S) was 4e17 and repeated evaluations differed by 0.3 nats).  Not a guarantee over
        the box: lambda_max(Ms) scales with 1/sigma_q^2, so with sigma_q down to its bound h0 - 3 cond(S)
        can reach ~e^6 cond_max (joint mode) -- `cond` reports it at the endpoint."""
        Ms = self._parts(jnp.asarray(h0, float))[0]
        v = jnp.ones(Ms.shape[0]) / np.sqrt(Ms.shape[0])
        for _ in range(50):                                  # power iteration: lambda_max(Ms)
            v = Ms @ v; v = v / jnp.linalg.norm(v)
        lam_max = float(v @ (Ms @ v))
        self.a_floor = float(np.log(max(lam_max, 1e-300) / cond_max))
        k = self._nls
        lo = np.concatenate([np.asarray(h0[:k]) - 3.0, np.full(len(self.groups), self.a_floor)])
        hi = np.concatenate([np.asarray(h0[:k]) + 3.0, np.full(len(self.groups), 10.0)])
        self.lower, self.upper = lo, hi
        return lo, hi

    def cond(self, h, iters=50):
        """cond(S(h)) estimated by power iteration on S (lambda_max) and on S^-1 through its Cholesky
        factor (lambda_min): O(iters L^2) after one factorisation, deterministic."""
        Ms, _, lam, _ = self._parts(jnp.asarray(h, float))
        S = Ms + jnp.diag(lam)
        c = cho_factor(S, lower=True)
        v = w = jnp.ones(S.shape[0]) / np.sqrt(S.shape[0])
        for _ in range(iters):
            v = S @ v; v = v / jnp.linalg.norm(v)
            w = cho_solve(c, w); w = w / jnp.linalg.norm(w)
        return float((v @ (S @ v)) / (w @ (S @ w)))

    def sigmas(self, h):
        """Noise scales (sigma_E, sigma_F, sigma_V) of hyperparameters h: fitted [joint] or the fixed
        linear MAP ones [sequential]."""
        return np.exp(np.asarray(h[:3], float)) if self.joint else np.exp(np.asarray(self.ls_fixed, float))


def _ev_parts(h, data, root, gidx, joint):
    """(S's data part R0^-T M R0^-1, R0^-T b, the scaled prior precisions, the constant) at h; root a
    prior_root.PriorRoot (M = 0: D^-1 M D^-1 and D^-1 b, as written before the GP arm)."""
    G, b, yy, nq = data
    nls = 3 if joint else 0
    if joint:
        w = jnp.exp(-2 * h[:3])
        M = w[0] * G[0] + w[1] * G[1] + w[2] * G[2]
        bv = w[0] * b[0] + w[1] * b[1] + w[2] * b[2]
        const = -0.5 * jnp.sum(yy * w) - jnp.sum(nq * h[:3])
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


def fit_ard(ev, h0, cond_max=1e14, maxiter=500, polish=True, start=None):
    """Type-II ML by L-BFGS-B, bounded (ev.bounds), on the objective relative to its start and divided
    by the initial gradient norm: L-BFGS-B's first bounded step is the full gradient (~3e3 on the
    Cantor problem), which lands on the box corner; a non-finite point is a rejected step.  Then
    (polish) `newton_polish` converges the endpoint with the exact Hessian: L-BFGS-B stops at a
    roundoff-dependent point (a line-search failure on the noisy objective, or a loose scaled gtol),
    which two runs on different BLAS / batch layouts reach differently.  The box comes from h0;
    `start` (default h0) is where L-BFGS-B starts in it.  info["cond_S"]: cond(S) at the endpoint
    (ev.cond), with info["cond_max"] the floor's target (exceeded when sigma ran down, see ev.bounds);
    info["newton"]["noise"] / ["gnoise"]: the measured roundoff of the evidence and of each gradient
    component at the returned endpoint (the resolution of v and of its gradient)."""
    from scipy.optimize import minimize
    lo, hi = ev.bounds(h0, cond_max)
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
    v, _ = ev.value_and_grad(x)
    return x, v, {**info, "success": success, "message": message, "nit": int(r.nit), "gain": v - v0,
                  "at_bound": ((np.isclose(x, lo)) | (np.isclose(x, hi))).tolist(),
                  "cond_S": ev.cond(x) if hasattr(ev, "cond") else None, "cond_max": float(cond_max)}


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
        from .prior_root import PriorRoot
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
        from .prior_root import PriorRoot
        from .support import unflatten_support
        js = lambda k: json.loads(bytes(z[k]).decode()) if k in z.files else None
        if int(z["schema"]) not in (1, 2, 3):
            raise ValueError(f"unsupported posterior schema {int(z['schema'])}")
        if "variance" in z.files and str(z["variance"]) == "dtc":
            raise ValueError(f"{path} is an --ard-variance dtc posterior: that shape failed its acceptance and was "
                             "removed; refit with --uq ard-gp (sandwich)")
        if "chol_packed" in z.files:            # the lower triangle, row-major (np.tril_indices)
            n = len(z["dinv"])
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
    Ms, bv, lam, _ = ev._parts(jnp.asarray(h, float))
    c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
    x = cho_solve((c, low), bv)
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
        from .hypers import to_array
        keep["gp_cols"] = ev.root.M
        gp = {"root": ev.root, "gp_theta": np.asarray(to_array(theta), np.float64)}
    # chol stays a float64 device array: var_rows runs once per batch (predict_ard, _val_errors), and
    # jnp.asarray of a numpy factor would re-upload the L x L matrix (1.8 GB at L = 15k) on every call
    return ARDPosterior(np.asarray(ev.root.primal(x)), jnp.asarray(c, jnp.float64), np.asarray(ev.dinv),
                        float(kappa), np.asarray(h, float), ev.groups, ev.body_col, keep, **gp)


def sandwich_scores(post, prob, ds, sig):
    """Prior-scaled cluster scores G~ (L, n_cfg) of the configurations of ds at the posterior mean:
    g~_c = D^-1 sum_{i in c} rho_i psi_i over every E/F/V row i of config c, psi_i = phi_i w_i/sigma_q,
    rho_i = (y_i - phi_i c) w_i/sigma_q.  Padded configs (cfg_mask) are dropped; padded nodes carry
    node_cfg == C and land in a discarded extra segment."""
    from .rows import ROWS_EDGE_BUDGET
    c, dinv = jnp.asarray(post.mean), jnp.asarray(post.dinv)
    inv = jnp.asarray(1.0 / np.asarray(sig, float))
    out = []
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        g = _sandwich_scores_jit(prob.model, prob.cfg, bt, c, dinv, inv, ROWS_EDGE_BUDGET)
        out.append(np.asarray(g)[np.asarray(bt.cfg_mask)])
    return np.concatenate(out).T


def _sandwich_scores_body(model, cfg, bt, c, dinv, inv, budget):
    from .rows import linear_rows_bounded
    L = cfg.len_basis
    r = linear_rows_bounded(model, cfg, bt)
    C = r.E.shape[0]
    P = jnp.concatenate([r.E, r.F.reshape(-1, L), r.V.reshape(-1, L)])
    y = jnp.concatenate([bt.y_E, bt.y_F.reshape(-1), bt.y_V.reshape(-1)])
    w = jnp.concatenate([bt.w_E * inv[0], jnp.repeat(bt.w_F, 3) * inv[1], jnp.repeat(bt.w_V, 6) * inv[2]])
    cid = jnp.concatenate([jnp.arange(C), jnp.repeat(bt.node_cfg, 3), jnp.repeat(jnp.arange(C), 6)])
    g = jax.ops.segment_sum(P * ((y - P @ c) * w * w)[:, None], cid, num_segments=C + 1)[:C]
    return g * dinv[None, :]


_sandwich_scores_jit = eqx.filter_jit(_sandwich_scores_body)


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


def rows_fn_for(prob, theta, node_chunk=None):
    """The ARD stage's design-row function of one batch: the linear arm's node-chunked rows (M = 0; theta
    unused), or the joint rows [B | k_theta(B, B_M)] at theta (uq "ard-gp"), whose residual block
    rows.batch_rows bounds by ROWS_EDGE_BUDGET as in training."""
    from . import rows as _rows
    if prob.ind.XM.shape[0] == 0:
        return _rows.chunked_rows_fn(prob.model, prob.cfg, node_chunk)
    # ROWS_EDGE_BUDGET is read while tracing: pass it as the cache key, read at call time (as rows._rows_jit)
    return lambda b: _joint_rows_jit(theta, prob.spec, prob.model, prob.ind, prob.cfg, b, _rows.ROWS_EDGE_BUDGET)


def _joint_rows_body(theta, spec, model, ind, cfg, batch, budget):
    from .rows import batch_rows
    return batch_rows(theta, spec, model, ind, cfg, batch)


_joint_rows_jit = eqx.filter_jit(_joint_rows_body)


def predict_ard(post, prob, ds, node_chunk=None, rows_fn=None):
    """Posterior predictive on a Dataset: means from the ARD mean; F_var the served calibrated force
    variance, E_var/V_var the untempered posterior variances.  Schema 3 (post.group_table set):
    F_var = lam_rms[g]^2 diag V per atom (g = post.groups_of(batch)), so sum_a F_var = forces_std^2.
    Schema 1/2: `force_var_rows` (lam^2 x cluster sandwich when post.Q is set, else kappa^2 x the
    posterior variance)."""
    from .predict import _pack
    from .rows import chunked_rows_fn
    if rows_fn is None:
        rows_fn = chunked_rows_fn(prob.model, prob.cfg, node_chunk)
    lam_rms = None if post.group_table is None else np.asarray(post.group_table["lam_rms"], float)
    outs = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = rows_fn(b)
        Fn = np.asarray(r.F)                                                   # (Ncap, 3, L) or (.., L + M)
        L = Fn.shape[-1]
        E, F, V = np.asarray(r.E), Fn.reshape(-1, L), np.asarray(r.V).reshape(-1, L)
        if lam_rms is not None:
            g = post.groups_of(b)
            Vs = post.atom_shape(Fn)
            Fv = lam_rms[g][:, None] ** 2 * np.diagonal(Vs, axis1=1, axis2=2)
            Fv = np.where(np.asarray(b.node_mask)[:, None], Fv, 0.0)
        else:
            Fv = post.force_var_rows(F).reshape(-1, 3)
        # the untempered E/V variances need chol, which a saved sandwich posterior leaves out: NaN then
        # (only F_var is calibrated and served; E/V variances are reported by the fit itself)
        var = post.var_rows if post.chol is not None else (lambda P: np.full(len(P), np.nan))
        outs.append((E @ post.mean, var(E), (F @ post.mean).reshape(-1, 3), Fv,
                     (V @ post.mean).reshape(-1, 6), var(V).reshape(-1, 6)))
    return _pack(outs, prob, ds)


class ARDResult(NamedTuple):
    posterior: ARDPosterior
    report: dict


def _shell_dists(ds, max_batches=200):
    """Live neighbour distances |r_ij| (node_mask & nbr_mask) of up to max_batches evenly spaced
    batches of ds: the sample r1 (`conformal.shell_reference`) is taken from."""
    sel = np.unique(np.linspace(0, ds.n_batches - 1, min(ds.n_batches, max_batches)).round().astype(int))
    out = []
    for i in sel:
        m = np.asarray(ds.nbr_mask[i]) & np.asarray(ds.node_mask[i])[:, None]
        out.append(np.linalg.norm(np.asarray(ds.rij[i]), axis=-1)[m])
    return np.concatenate(out) if out else np.zeros(0)


def _shell_table(ds, r1):
    """(z, d, cfg_index, batch_idx, node_idx) of every live node of ds; cfg_index counts the live
    configurations of ds in order (the order build_dataset received them)."""
    from .conformal import shell_features
    cols, off = [], 0
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        live = np.flatnonzero(np.asarray(b.node_mask))
        z, d = shell_features(b, r1)
        cols.append((z[live], d[live], off + np.asarray(b.node_cfg)[live], np.full(len(live), i), live))
        off += int(np.asarray(b.cfg_mask).sum())
    return tuple(np.concatenate([c[k] for c in cols]) for k in range(5))


class _ValAtoms(NamedTuple):
    e: np.ndarray            # (n, 3) F_label - F_pred(post.mean)
    s2: np.ndarray           # (n,) untempered posterior variance tr(phi A^-1 phi^T)
    V: np.ndarray | None     # (n, 3, 3) post.atom_shape (own cluster left out when own_col is given)
    v_incl: np.ndarray | None    # (n,) tr V with the own cluster kept (own_col only)
    z: np.ndarray | None
    d: np.ndarray | None
    cfg: np.ndarray          # (n,) live-configuration index into ds
    bn: np.ndarray | None = None    # (n, 2) (batch, node) index of each atom in ds


def _val_atoms(post, prob, ds, r1=None, shape=True, own_col=None, rows_fn=None):
    """Per live force-labelled atom (w_F > 0) of ds, in one pass over its rows: the force error at
    post.mean, the untempered s^2, (shape) the unscaled shape V = post.atom_shape, and (r1) the shell
    features z, d.  own_col: (n_cfg(ds),) the Q column of each config of ds -- the #18 own-cluster-out
    rule of the legacy "mixed" ablation (held-out configs are training configs of the full refit): V
    then drops that one cluster's column of Q, and v_incl = tr V with it."""
    from .conformal import shell_features
    from .rows import chunked_rows_fn
    L = post.prior_root.width
    if rows_fn is None:
        rows_fn = chunked_rows_fn(prob.model, prob.cfg)
    acc = {k: [] for k in _ValAtoms._fields}
    dinv = jnp.asarray(post.dinv, jnp.float64)
    off = 0                                          # configs of ds before this batch (padded ones excluded)
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        live = (np.asarray(b.w_F) > 0) & np.asarray(b.node_mask)
        if live.any():
            F = np.asarray(rows_fn(b).F)[live]                                         # (n, 3, L)
            Fr = F.reshape(-1, L)
            acc["e"].append(np.asarray(b.y_F)[live] - (Fr @ post.mean).reshape(-1, 3))
            acc["s2"].append(post.var_rows(Fr).reshape(-1, 3).sum(1))
            cfg = off + np.asarray(b.node_cfg)[live]
            acc["cfg"].append(cfg)
            acc["bn"].append(np.c_[np.full(live.sum(), i), np.flatnonzero(live)])
            if own_col is not None:
                Pr = (jnp.asarray(F, jnp.float64) * dinv[None, None, :]) @ jnp.asarray(post.Q, jnp.float64)
                acc["v_incl"].append(np.asarray(jnp.sum(Pr * Pr, axis=(1, 2))))
                own = jnp.asarray(np.asarray(own_col)[cfg])
                # drop the own column before squaring (no tot - v_own^2 cancellation; >= 0 by construction)
                Pr = jnp.where(jnp.arange(Pr.shape[2])[None, None, :] == own[:, None, None], 0.0, Pr)
                acc["V"].append(np.asarray(jnp.einsum("nar,nbr->nab", Pr, Pr)))
            elif shape:
                acc["V"].append(post.atom_shape(F))
            if r1 is not None:
                z, d = shell_features(b, r1)
                acc["z"].append(z[live])
                acc["d"].append(d[live])
        off += int(np.asarray(b.cfg_mask).sum())
    cat = lambda k, shp: np.concatenate(acc[k]) if acc[k] else np.zeros(shp)
    return _ValAtoms(cat("e", (0, 3)), cat("s2", (0,)),
                     cat("V", (0, 3, 3)) if (shape or own_col is not None) else None,
                     cat("v_incl", (0,)) if own_col is not None else None,
                     cat("z", (0,)).astype(np.int64) if r1 is not None else None,
                     cat("d", (0,)) if r1 is not None else None, cat("cfg", (0,)).astype(np.int64),
                     cat("bn", (0, 2)).astype(np.int64))


def _support_reference(cfg, data, built, ds_val, bn, scores, cfg_ids, grp, log=print):
    """The covariate-shift reference (fit/support.py): per-species descriptor PCA from a random sample of
    training batches (<= ard_support_max_atoms atoms per species), and the T_val atoms (batch, node) = bn
    with their conformal scores and configuration ids.  Site descriptors are evaluated only on those."""
    from .inducing import site_features
    from .support import build_support, explained_variance, fit_pca, n_pass, support_body, support_features
    prob = built.prob
    feat = {"kind": cfg.ard_support_features, "body": support_body({"nnll": data.meta["nnll"],
                                                                     "n_pair": prob.cfg.n_pair})}
    ds = data.ds_train
    rng = np.random.default_rng(cfg.seed)
    cap = int(cfg.ard_support_max_atoms)
    one = lambda ds_, i: jax.tree.map(lambda a: a[i:i + 1], ds_)      # batch i, leading axis kept
    pool = {}
    n_species = len(np.unique(np.asarray(ds.node_z)[np.asarray(ds.node_mask)]))
    for i in rng.permutation(ds.n_batches):
        if pool and len(pool) == n_species and all(sum(len(x) for x in v) >= cap for v in pool.values()):
            break
        sub = one(ds, int(i))
        X = np.asarray(site_features(prob.model, built.gpcfg, sub)[0])[0]
        live, Zs = np.asarray(sub.node_mask)[0], np.asarray(sub.node_z)[0]
        for z in np.unique(Zs[live]):
            pool.setdefault(int(z), []).append(X[live & (Zs == z)])
    Xp = {}
    for z, v in pool.items():
        v = np.concatenate(v)
        Xp[z] = v[rng.permutation(len(v))[:cap]]
    Fp = {z: support_features(v, feat) for z, v in Xp.items()}
    pca = fit_pca(Fp, n_pass=n_pass(feat))
    ev = explained_variance(Fp, pca, n_pass(feat))
    log(f"ARD support ({feat['kind']} features): PCA components per species "
        + ", ".join(f"{z}: {pca[z][2].shape[1] - n_pass(feat)} ({ev[z]:.3f} var)" for z in sorted(pca))
        + (f", plus {n_pass(feat)} log-norm channels" if n_pass(feat) else ""))
    Xc = np.zeros((len(bn), 0))
    for i in np.unique(bn[:, 0]):                    # only the live T_val atoms are kept
        Xi = np.asarray(site_features(prob.model, built.gpcfg, one(ds_val, int(i)))[0])[0]
        sel = bn[:, 0] == i
        if Xc.shape[1] == 0:
            Xc = np.zeros((len(bn), Xi.shape[1]))
        Xc[sel] = Xi[bn[sel, 1]]
    Zc = np.asarray(ds_val.node_z)[bn[:, 0], bn[:, 1]]
    k = np.isin(Zc, list(pca))
    return build_support(pca, Xc[k], Zc[k], np.asarray(scores, float)[k], np.asarray(cfg_ids)[k],
                         cap, cfg.seed, grp=np.asarray(grp)[k], features=None if feat["kind"] == "raw" else feat)


def conformal_scores(e, V, force_shape, eps):
    """iso: s = |e| / sqrt(v/3); aniso: s = sqrt(e^T (V + eps (v/3) I)^-1 e), v = tr V."""
    v = np.trace(V, axis1=1, axis2=2)
    if force_shape == "aniso":
        M = V + eps * (v / 3)[:, None, None] * np.eye(3)
        return np.sqrt(np.einsum("na,na->n", e, np.linalg.solve(M, e[:, :, None])[:, :, 0]))
    return np.sqrt(np.sum(e * e, 1) / (v / 3))


def _ard_fit_warnings(stage, info, names):
    """Log lines for an ARD evidence fit: its convergence record (L-BFGS-B, then the Newton polish), and
    warnings when it did not converge or ended with a hyperparameter on its bound (a scale at a_floor
    is the conditioning guard, not the evidence optimum)."""
    out = []
    if "newton" in info:            # the convergence record of every evidence fit (a log line, not a warning)
        n = info["newton"]
        out.append(f"ARD {stage} evidence fit: L-BFGS-B {info.get('lbfgs_message', '?')} after {info.get('nit')} "
                   f"iterations; Newton polish {n['steps']} steps, {n['hessian_evals']} Hessians, |pg| "
                   f"{n['pg_start']:.2e} -> {n['pg']:.2e}")
    if info.get("cond_S") is not None:
        out.append(f"ARD {stage} evidence fit: cond(S) {info['cond_S']:.1e} at the endpoint")
        if not np.isfinite(info["cond_S"]) or info["cond_S"] > info.get("cond_max", np.inf):
            why = "non-finite: S numerically singular" if not np.isfinite(info["cond_S"]) else "exceeds"
            out.append(f"WARNING: ARD {stage} evidence fit: cond(S) {info['cond_S']:.1e} ({why} ard_cond_max "
                       f"{info['cond_max']:.0e}; the a_floor guard is set at the start's noise scales, which may "
                       f"have moved down); the evidence is resolved only to its roundoff there")
    if not info.get("success", True):
        out.append(f"WARNING: ARD {stage} evidence fit did not converge: {info.get('message', '?')}")
    hit = [n for n, b in zip(names, info.get("at_bound", [])) if b]
    if hit:
        out.append(f"WARNING: ARD {stage} evidence fit ended at a bound for {', '.join(hit)}")
    return out


def _gp_parts(cfg, prob, theta, body_col):
    """The uq-dependent pieces of the stage: the linear arm's (linear statistics, body_col, D, linear rows),
    or ard-gp's joint design at theta (joint statistics, GP_GROUP columns, the prior root with chol(K_MM)^T,
    the joint rows; theta stored with the posterior)."""
    if cfg.uq != "ard-gp":
        return {"columns": "linear", "body": body_col, "root": None, "rows_fn": None, "theta": None}
    from .prior_root import prior_root
    return {"columns": "joint", "body": gp_body_columns(body_col, prob.ind.XM.shape[0]),
            "root": prior_root(prob, theta), "rows_fn": rows_fn_for(prob, theta), "theta": theta}


def _h_names(ev):
    return ((["log_sigma_E", "log_sigma_F", "log_sigma_V"] if ev.joint else [])
            + ["a_GP" if g == GP_GROUP else f"a_{g}body" for g in ev.groups])


def _holdout_posterior(cfg, data, prob, theta, configs, body_col, ell, h0=None, stage="fit-subset", log=print):
    """A hold-out posterior on the training subset `configs` with its own shape (PRESS jackknife R, the
    legacy sandwich Q, or none for ard_variance "kappa" / the "mixed" ablation).  h0: the evidence fit's
    start (default the prior's).  Returns (post, h, logev, info, K, h_names)."""
    from .clusters import row_clusters
    from .data import build_dataset
    from .jackknife import press_scores, shape_factor
    mode = cfg.ard_mode
    gp = _gp_parts(cfg, prob, theta, body_col)
    ds = build_dataset(configs, data.meta, data.E0, cfg.batch, pack=cfg.pack_mode, log=log)
    ev = ARDEvidence(ard_statistics(theta, prob, ds, mode, columns=gp["columns"]), ard_gamma(prob), gp["body"],
                     root=gp["root"])
    names = _h_names(ev)
    h, v, info = fit_ard(ev, ev.h0(theta) if h0 is None else h0, cfg.ard_cond_max)
    for w in _ard_fit_warnings(stage, info, names):
        log(w)
    post = ard_posterior(ev, h, 1.0, data.meta, theta=gp["theta"])
    K = 0
    if cfg.ard_variance == "sandwich" and cfg._score_source == "fit":
        if cfg._shape_variant == "press":
            rc, K = row_clusters(ds, configs, ell)
            Gs, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode=cfg.ard_press, rows_fn=gp["rows_fn"])
            post = post._replace(R=shape_factor(post, Gs, cfg.ard_shape_tau))
        else:
            Gs = sandwich_scores(post, prob, ds, ev.sigmas(h))
            K = int(Gs.shape[1])
            post = post._replace(Q=sandwich_factor(post, Gs))
        del Gs
    return post, h, v, info, K, names


def run_ard_stage(cfg, data, built, theta, log=print, full_stats=None):
    """The schema-3 ARD stage (docs/dev/specs/2026-09-30-conformal-force-sigma-design.md section 4):

    1. shell features of every training atom -> r1, z*, band edges (all of T), configuration strata,
       and the stratified split T = T_fit + T_val;
    2. the hold-out posterior P_fit on T_fit, with its own shape (PRESS jackknife R_fit, or the legacy
       sandwich Q_fit, or A_fit^-1 for ard_variance "kappa");
    3. T_val scores from P_fit's errors and shape (or, for the legacy "mixed" ablation, from the
       served posterior's own-cluster-out sandwich);
    3b. (ard_transfer "exponent") a second hold-out posterior P_fit2 on T_fit2, the same stratified split
       of T_fit, scored on the same T_val atoms: the pooled scales lam1 (P_fit), lam2 (P_fit2) give the
       transfer exponent beta = clip(log(lam1/lam2) / log(N_fit/N_fit2), 0, 1/2) ("sqrt": 1/2, "none": 0);
    4. the served posterior P on all of T (started from the subset optimum) and its stored shape;
    5. per-group configuration-weighted lam_rms and q from the T_val scores times t = (N/N_fit)^beta (the
       hold-out scale carried to P), stored in the posterior.
    kappa and the scalar lam of #18 are still reported for comparison.

    full_stats: the linear statistics of data.ds_train (`stats.linear_statistics`) the caller has
    already cached -- the pipeline objective's.  The joint full refit then uses them as they are
    instead of a second pass over the training set (spec 3); sequential mode ignores them."""
    import time
    from .clusters import row_clusters
    from .conformal import (assign_groups, band_edges, config_strata, group_scales, n_groups, pooled_lam_rms,
                            shell_reference, stratified_split, transfer_exponent, transfer_fixed)
    from .data import build_dataset
    from .jackknife import press_scores, shape_factor
    mode, val_frac, cond_max = cfg.ard_mode, cfg.ard_val_frac, cfg.ard_cond_max
    variance, variant, source = cfg.ard_variance, cfg._shape_variant, cfg._score_source
    t0 = time.time()
    prob = built.prob
    body_col = body_order_columns(data.meta, prob.cfg)
    gp = _gp_parts(cfg, prob, theta, body_col)          # ard-gp: joint design at theta; else the linear arm
    N = len(data.train)

    # 1. groups and strata from T (all live training atoms), then the stratified split
    dists = _shell_dists(data.ds_train)
    if len(dists):
        r1 = shell_reference(dists)
    else:
        r1 = float(prob.cfg.rcut)
        log(f"ARD: no neighbour pairs in the training set; shell reference r1 = r_cut = {r1:.3f}")
    z_all, d_all, cfg_all, _, _ = _shell_table(data.ds_train, r1)
    z_star = int(np.bincount(z_all).argmax()) if len(z_all) else 0
    edges = np.zeros(0)
    if cfg.ard_groups == "distortion":
        if np.isfinite(d_all).any():
            edges = band_edges(d_all)
        else:
            log("ARD: no training atom has a finite shell distortion d; distortion groups disabled (2 groups)")
    G = n_groups(edges)
    g_all = assign_groups(z_all, d_all, z_star, edges)
    o = np.argsort(cfg_all, kind="stable")
    strata = config_strata(np.split(g_all[o], np.searchsorted(cfg_all[o], np.arange(1, N))))
    fit_idx, val_idx = stratified_split(strata, val_frac, cfg.seed)
    val = [data.train[i] for i in val_idx]
    fit_ = [data.train[i] for i in fit_idx]
    if not any(c.forces is not None and c.w_F > 0 for c in val):
        raise ValueError(f"the ARD validation split ({len(val)} configs, ard_val_frac={val_frac}) has "
                         f"no force labels to fit the force-sigma scales on")
    ell = cfg.ard_cluster_size * float(prob.cfg.rcut)

    # 2. P_fit and its own shape
    ds_val = build_dataset(val, data.meta, data.E0, cfg.batch, pack=cfg.pack_mode, log=log)
    post_fit, h_fit, v_fit, info_fit, K_fit, names = _holdout_posterior(
        cfg, data, prob, theta, fit_, body_col, ell, log=log)

    # 3. T_val errors (and, unless "mixed", the scores' shape) from P_fit
    E = _val_atoms(post_fit, prob, ds_val, r1=r1, shape=(source == "fit"), rows_fn=gp["rows_fn"])
    del post_fit                           # free the subset fit's L x L Cholesky factor before the refit

    # 3b. the transfer exponent's second hold-out fit: P_fit2 on T_fit2, the same stratified rule applied
    # to T_fit, scored on the same T_val atoms ("mixed" scores already come from the served posterior)
    transfer = cfg.ard_transfer if source == "fit" else "none"
    if transfer != cfg.ard_transfer:
        log(f"ARD transfer: _score_source='mixed' scores against the served posterior; "
            f"ard_transfer={cfg.ard_transfer!r} ignored (none)")
    E2, n_fit2 = None, None
    if transfer == "exponent":
        fit2_loc, _ = stratified_split(strata[fit_idx], val_frac, cfg.seed + 1)
        fit2 = [fit_[i] for i in fit2_loc]
        n_fit2 = len(fit2)
        post_fit2, _, _, _, _, _ = _holdout_posterior(cfg, data, prob, theta, fit2, body_col, ell, h0=h_fit,
                                                      stage="fit-subset-2", log=log)
        E2 = _val_atoms(post_fit2, prob, ds_val, shape=True, rows_fn=gp["rows_fn"])
        del post_fit2                      # as P_fit: freed before the refit

    # 4. the served posterior on all of T and its shape
    st = (joint_ard_stats(full_stats) if (full_stats is not None and mode == "joint")
          else ard_statistics(theta, prob, data.ds_train, mode, columns=gp["columns"]))
    ev = ARDEvidence(st, ard_gamma(prob), gp["body"], root=gp["root"])
    del st
    v_start = ev.value_and_grad(np.clip(h_fit, *ev.bounds(h_fit, cond_max)))[0]   # refit's start
    h, v, info = fit_ard(ev, h_fit, cond_max)
    for w in _ard_fit_warnings("full", info, names):
        log(w)
    post = ard_posterior(ev, h, 1.0, data.meta, theta=gp["theta"])
    K, lev = 0, np.zeros(0)
    if variance == "sandwich":
        if variant == "press":
            rc, K = row_clusters(data.ds_train, data.train, ell)
            Gf, lev = press_scores(post, prob, data.ds_train, rc, K, ev.sigmas(h), mode=cfg.ard_press,
                                   rows_fn=gp["rows_fn"])
            post = post._replace(R=shape_factor(post, Gf, cfg.ard_shape_tau))
        else:
            # Q is a float64 device array (sandwich_factor): its per-batch consumers call jnp.asarray on
            # it, a no-op there -- as numpy it would re-upload the (L, n_cfg) factor on every call
            Gf = sandwich_scores(post, prob, data.ds_train, ev.sigmas(h))
            K, lev = int(Gf.shape[1]), np.zeros(Gf.shape[1])
            post = post._replace(Q=sandwich_factor(post, Gf))
        del Gf
    # held-out rows against the served posterior: its s^2 (kappa) and, for "mixed", the #18 shape
    Ef = _val_atoms(post, prob, ds_val, shape=False, own_col=val_idx if source == "mixed" else None,
                    rows_fn=gp["rows_fn"])
    V = E.V if source == "fit" else Ef.V
    vtr = np.trace(V, axis1=1, axis2=2)
    ok = (E.s2 > 0) & (vtr > 0)            # atoms with zero force rows (isolated, 1-atom configs): no information
    if not ok.any():
        raise ValueError(f"the ARD validation split (ard_val_frac={val_frac}) has no atom with a "
                         f"non-zero force row to fit the force-sigma scales on")
    e, V, vtr = E.e[ok], V[ok], vtr[ok]
    e2 = np.sum(e * e, 1)
    s2_sub, s2_full = E.s2[ok], Ef.s2[ok]

    # kappa (kept for comparison): the subset model's held-out errors against the subset and the
    # served posterior's untempered s^2
    kappa_subset = kappa_closed_form(e2, s2_sub)
    kappa = kappa_closed_form(e2, s2_full)
    log(f"ARD: fit-subset logev {v_fit:.2f} ({info_fit['message']}, nit {info_fit['nit']}); "
        f"kappa {kappa_subset:.3f} from {len(e2)} held-out atoms; {kappa:.3f} for the full posterior")
    lam, lam_incl_own = 1.0, None
    if variance == "sandwich":                     # the #18 scalar lam, from the new T_val (e^2, v)
        lam = kappa_closed_form(e2, vtr)
        if source == "mixed":
            lam_incl_own = kappa_closed_form(e2, Ef.v_incl[ok])

    # 5. per-group scales on T_val, carried to the served posterior by the transfer factor
    s = conformal_scores(e, V, cfg.ard_force_shape, cfg.ard_shape_eps)
    gv = assign_groups(E.z[ok], E.d[ok], z_star, edges)
    cv = np.asarray(val_idx)[E.cfg[ok]]            # training-set index of each calibration atom's config
    lam1 = pooled_lam_rms(s, gv, cv, G)
    if transfer == "exponent":
        V2 = E2.V[ok]
        m2 = np.trace(V2, axis1=1, axis2=2) > 0    # both scales on the atoms P_fit2 also informs
        s2 = conformal_scores(E2.e[ok][m2], V2[m2], cfg.ard_force_shape, cfg.ard_shape_eps)
        tr = transfer_exponent(pooled_lam_rms(s[m2], gv[m2], cv[m2], G), pooled_lam_rms(s2, gv[m2], cv[m2], G),
                               N, len(fit_), n_fit2, log=log)
        tr["lam1_all"] = lam1                      # lam1 is over the common atoms; lam1_all over every one
        del E2, V2, s2
    else:
        tr = transfer_fixed(transfer, lam1, N, len(fit_))
        log(f"ARD transfer: {transfer}, lam_fit {lam1:.4g} (N {len(fit_)}), factor {tr['factor']:.4g}")
    tr["requested"] = cfg.ard_transfer
    s = s * tr["factor"]
    tab = group_scales(s, gv, cv, G, 1 - cfg.ard_coverage, cfg.ard_n_min, log=log)
    post = post._replace(kappa=kappa, lam=lam, force_shape=cfg.ard_force_shape, eps=cfg.ard_shape_eps,
                         group_consts={"r1": float(r1), "z_star": z_star, "edges": edges.tolist()},
                         group_table=tab.to_dict(),
                         cal={"scores": s.astype(np.float32), "groups": gv.astype(np.int8),
                              "cfg": cv.astype(np.int64), "src": np.zeros(len(s), np.int16)})
    if cfg.ard_support:
        post = post._replace(support=_support_reference(cfg, data, built, ds_val, E.bn[ok], s, cv, gv, log))
    log(f"ARD: {G} groups over {len(np.unique(cv))} held-out configs ({len(s)} atoms); lam_rms "
        f"{np.array2string(tab.lam_rms, precision=3)}, q {np.array2string(tab.q, precision=3)}"
        + (f"; merged {tab.merged}" if tab.merged else ""))

    # 6. report
    from .jackknife import _MU_FLOOR
    n_near1 = int(np.sum(1.0 - lev < 1e-8))
    n_clamp = int(np.sum(1.0 - lev <= _MU_FLOOR)) if cfg.ard_press == "exact" else 0
    if n_near1:
        log(f"WARNING: ARD shape: {n_near1} of {K} clusters have leverage 1 - lambda_max(H_kk) < 1e-8; "
            f"their PRESS scores are accurate only to ~eps/(1 - lambda)"
            + (f"; {n_clamp} at leverage 1 within roundoff had 1 - lambda clamped to {_MU_FLOOR:g}" if n_clamp else ""))
    shp = post.R if post.R is not None else post.Q
    lq = (lambda f: float(f(lev))) if len(lev) else (lambda f: None)
    report = {"mode": mode, "body_groups": list(ev.groups), "gp_cols": ev.root.M, "h": h.tolist(), "h_names": names,
              "logev_full": v, "logev_full_start": v_start, "optimiser": info, "optimiser_fit": info_fit,
              # the evidence's roundoff at the endpoint, as the polish measured it (its spread over 1e-15
              # moves of h): two fits of the same data agree in logev_full only to ~this, not to 1e-15
              "logev_full_noise": info.get("newton", {}).get("noise"),
              "logev_full_gnoise": info.get("newton", {}).get("gnoise"),     # per hyperparameter, likewise
              "a_floor": ev.a_floor,
              "tempered_quantities": ["F"],        # F_var is lam_rms[g]^2 diag V; E_var / V_var untempered
              "kappa": kappa, "kappa_subset": kappa_subset, "n_val_atoms": int(len(e2)),
              "n_val_configs": len(val), "n_fit_configs": len(fit_),
              # held-out errors (subset model) against the served posterior's s^2
              "val_rms_z_untempered": float(np.sqrt(np.mean(e2 / (s2_full / 3)) / 3)),
              "val_rms_z_tempered": float(np.sqrt(np.mean(e2 / (kappa ** 2 * s2_full / 3)) / 3)),
              "val_nll_untempered": _force_nll(e2, s2_full, 1.0), "val_nll_tempered": _force_nll(e2, s2_full, kappa),
              "variance": variance, "lam": lam, "lam_incl_own": lam_incl_own, "n_clusters": K,
              "shape": {"variant": variant if variance == "sandwich" else "kappa", "mode": cfg.ard_press,
                        "ell": ell, "K": K, "K_fit": K_fit,
                        "rank_R": None if shp is None else int(np.shape(shp)[1]),
                        "lev_p50": lq(np.median), "lev_p99": lq(lambda x: np.quantile(x, 0.99)),
                        "lev_max": lq(np.max), "n_lev_near1": n_near1, "n_mu_clamped": n_clamp},
              "groups": tab.to_dict(),
              "split": {"n_fit": len(fit_), "n_val": len(val), "val_idx": np.asarray(val_idx).tolist(),
                        "strata": np.bincount(strata, minlength=G).tolist()},
              "transfer": {"f": val_frac, "N_fit": len(fit_), "N": N, "score_source": source, **tr}}
    if variance == "sandwich":
        report["val_rms_z_sandwich"] = float(np.sqrt(np.mean(e2 / (lam ** 2 * vtr / 3)) / 3))
    if cfg.ard_laplace:
        lap = laplace_hypers(ev, h)
        report["laplace_std_h"] = lap["std"].tolist()
        report["laplace_eigs"] = lap["eigs"].tolist()
    report["seconds"] = time.time() - t0
    return ARDResult(post, report)
