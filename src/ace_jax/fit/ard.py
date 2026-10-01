"""Tempered ARD posterior: calibrated per-atom force uncertainty for linear ACE.

Spec: docs/specs/2026-09-28-tempered-ard-uq-design.md.  For the linear model the weighted design
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

SCHEMA = 3
_NEED3 = "this posterior predates schema 3; refit with --uq ard to get forces_q/forces_cov/forces_group"


def body_order_columns(meta, cfg):
    """Body order (2, 3, 4, ...) of every design column, in the `rows._place` layout: species blocks
    of the n_B many-body columns (correlation order nu -> body order nu + 1), then species blocks of
    the n_pair pair columns (body order 2)."""
    order_B = np.array([len(x) for x in meta["nnll"]]) + 1
    return np.concatenate([np.tile(order_B, cfg.NZ), np.full(cfg.n_pair * cfg.NZ, 2)])


class ARDStats(NamedTuple):
    G: tuple                 # (G_E, G_F, G_V) [joint] or (M,) combined at ls_fixed [sequential]
    b: tuple
    yy: np.ndarray           # (3,) [joint]; unused [sequential]
    n: np.ndarray            # (3,)
    ls_fixed: np.ndarray | None


def ard_statistics(theta, prob, ds, mode):
    """joint: per-quantity statistics (3 L^2 matrices).  sequential: the combined Gram at the
    linear MAP noise scales, accumulated in one pass (1 L^2 matrix) -- the low-memory mode.
    Both are the linear (L-column) statistics only: a hybrid problem's inducing columns (M > 0)
    never enter the ARD posterior, and the joint Gram is hyperparameter-independent."""
    from .stats import linear_statistics
    if mode == "joint":
        return joint_ard_stats(linear_statistics(prob.model, prob.cfg, ds))
    if mode != "sequential":
        raise ValueError(f"ard mode must be 'joint' or 'sequential', got {mode!r}")
    from .rows import linear_rows
    ls = np.array([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
    inv = jnp.asarray(np.exp(-ls))
    L = prob.cfg.len_basis

    def body(acc, batch):
        M, bv = acc
        r = linear_rows(prob.model, prob.cfg, batch)[0]
        for k, (Phi, y, w) in enumerate(((r.E, batch.y_E, batch.w_E),
                                         (r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3)),
                                         (r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6)))):
            Pw = Phi * (w * inv[k])[:, None]
            M = M + Pw.T @ Pw
            bv = bv + Pw.T @ (y * w * inv[k])
        return (M, bv), None

    (M, bv), _ = jax.lax.scan(jax.checkpoint(body), (jnp.zeros((L, L)), jnp.zeros(L)), ds)
    return ARDStats((M,), (bv,), np.zeros(3), np.zeros(3), ls)


def joint_ard_stats(st):
    """Joint ARDStats from `stats.linear_statistics` output (a `Stats`): the arrays are shared, not
    copied, so a caller's cached statistics (pipeline objective) serve the full refit as they are."""
    return ARDStats(tuple(getattr(st, f"G_{q}") for q in "EFV"), tuple(getattr(st, f"b_{q}") for q in "EFV"),
                    np.array([float(getattr(st, f"yy_{q}")) for q in "EFV"]),
                    np.array([float(getattr(st, f"n_{q}")) for q in "EFV"]), None)


class ARDEvidence:
    """log p(D|h) and its gradient in the prior-scaled system.  h = (log sigma_q [joint only], a_k)."""

    def __init__(self, stats, gamma, body_col):
        self.joint = stats.ls_fixed is None
        self.ls_fixed = stats.ls_fixed
        self.groups = tuple(int(g) for g in np.unique(body_col))
        self.body_col = np.asarray(body_col)
        gidx = jnp.asarray(np.searchsorted(np.asarray(self.groups), self.body_col))
        dinv = jnp.asarray(1.0 / np.asarray(gamma))
        self.dinv = dinv
        # the unscaled G/b are held as given (no scaled copies: the Gram is L^2 per quantity, and a
        # pipeline caller shares them with its cache); D^-1 (sum_q w_q G_q) D^-1 is formed per call.
        # They are jit ARGUMENTS, not closure constants, which XLA would copy into the executable.
        self._data = (tuple(stats.G), tuple(stats.b), jnp.asarray(stats.yy), jnp.asarray(stats.n))
        nls = 3 if self.joint else 0

        def parts(h, data):
            G, b, yy, nq = data
            if self.joint:
                w = jnp.exp(-2 * h[:3])
                M = w[0] * G[0] + w[1] * G[1] + w[2] * G[2]
                bv = w[0] * b[0] + w[1] * b[1] + w[2] * b[2]
                const = -0.5 * jnp.sum(yy * w) - jnp.sum(nq * h[:3])
            else:
                M, bv, const = G[0], b[0], 0.0
            return dinv[:, None] * M * dinv[None, :], dinv * bv, jnp.exp(h[nls:])[gidx], const

        def logev(h, data):
            Ms, bv, lam, const = parts(h, data)
            c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
            x = cho_solve((c, low), bv)
            return const + 0.5 * bv @ x - jnp.sum(jnp.log(jnp.diag(c))) + 0.5 * jnp.sum(jnp.log(lam))

        self._parts = lambda h: parts(h, self._data)
        self._vg = jax.jit(jax.value_and_grad(logev))
        self._nls = nls

    def h0(self, theta):
        a_blr = float(-2 * theta.log_sigma_c)
        ls = [float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"] if self.joint else []
        return np.array(ls + [a_blr] * len(self.groups))

    def value_and_grad(self, h):
        v, g = self._vg(jnp.asarray(h, float), self._data)
        return float(v), np.asarray(g)

    def bounds(self, h0, cond_max):
        """log sigma within h0 +- 3; a in [a_floor, 10] with a_floor = log(lambda_max(Ms(h0)) / cond_max):
        the prior never lets cond(S) exceed ~cond_max (at the unfloored Cantor ARD optimum cond(S) was
        4e17 and repeated evaluations differed by 0.3 nats)."""
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

    def sigmas(self, h):
        """Noise scales (sigma_E, sigma_F, sigma_V) of hyperparameters h: fitted [joint] or the fixed
        linear MAP ones [sequential]."""
        return np.exp(np.asarray(h[:3], float)) if self.joint else np.exp(np.asarray(self.ls_fixed, float))


def fit_ard(ev, h0, cond_max=1e14, maxiter=500):
    """Type-II ML by L-BFGS-B, bounded (ev.bounds), on the objective relative to its start and divided
    by the initial gradient norm: L-BFGS-B's first bounded step is the full gradient (~3e3 on the
    Cantor problem), which lands on the box corner; a non-finite point is a rejected step."""
    from scipy.optimize import minimize
    lo, hi = ev.bounds(h0, cond_max)
    x0 = np.clip(np.asarray(h0, float), lo, hi)
    v0, g0 = ev.value_and_grad(x0)
    gs = max(1.0, float(np.linalg.norm(g0)))

    def f(z):
        v, g = ev.value_and_grad(z)
        if not (np.isfinite(v) and np.all(np.isfinite(g))):
            return 1e30, np.zeros_like(z)
        return -(v - v0) / gs, -g / gs

    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                 options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6})
    v, _ = ev.value_and_grad(r.x)
    return r.x, v, {"success": bool(r.success), "message": str(r.message), "nit": int(r.nit), "gain": v - v0,
                    "at_bound": ((np.isclose(r.x, lo)) | (np.isclose(r.x, hi))).tolist()}


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

    def _tab(self):
        if self.group_table is None:
            raise ValueError(_NEED3)
        return self.group_table

    def groups_of(self, batch):
        from .conformal import assign_groups, shell_features
        if self.group_consts is None:
            raise ValueError(_NEED3)
        gc = self.group_consts
        z, d = shell_features(batch, gc["r1"])
        return assign_groups(z, d, gc["z_star"], np.asarray(gc["edges"], float))

    def atom_shape(self, Frows):
        """Unscaled per-atom shape V (N, 3, 3) from force rows (N, 3, L)."""
        from .jackknife import atom_shape
        if self.R is not None:
            return atom_shape(self.R, self.dinv, Frows)
        if self.Q is not None:                                   # schema 2: uncentred sandwich factor
            return atom_shape(self.Q, self.dinv, Frows)
        U = np.asarray(Frows) * np.asarray(self.dinv)[None, None, :]          # kappa: V = u^T S^-1 u
        W = np.asarray(solve_triangular(jnp.asarray(self.chol), jnp.asarray(U.reshape(-1, U.shape[-1]).T),
                                        lower=True)).T.reshape(U.shape[0], 3, -1)
        return np.einsum("nar,nbr->nab", W, W)

    def forces_cov(self, Frows, groups):
        lam = np.asarray(self._tab()["lam_rms"])[groups]
        return lam[:, None, None] ** 2 * self.atom_shape(Frows)

    def forces_q(self, Frows, groups):
        q = np.asarray(self._tab()["q"])[groups]
        V = self.atom_shape(Frows)
        v = np.trace(V, axis1=1, axis2=2)
        if self.force_shape == "aniso":
            lm = np.linalg.eigvalsh(V + self.eps * (v / 3)[:, None, None] * np.eye(3)).max(1)
            return q * np.sqrt(lm)
        return q * np.sqrt(v / 3)

    def forces_q_mahal(self, groups):
        return np.asarray(self._tab()["q"])[groups]

    def var_rows(self, Phi, chunk=4096):
        """Untempered posterior variance phi A^-1 phi^T of each row of Phi (n, L)."""
        c = jnp.asarray(self.chol, jnp.float64)
        out = []
        for i in range(0, len(Phi), chunk):
            v = solve_triangular(c, (jnp.asarray(Phi[i:i + chunk]) * jnp.asarray(self.dinv)[None, :]).T, lower=True)
            out.append(np.asarray(jnp.sum(v * v, axis=0)))
        return np.concatenate(out) if out else np.zeros(0)

    def misspec_var_rows(self, Phi, chunk=4096, own=None):
        """Unscaled cluster-sandwich variance ||Q^T (D^-1 phi)||^2 = phi A^-1 M A^-1 phi^T per row.

        own: optional (n,) Q column of each row's own training configuration (-1: none); returns
        (all, without_own), where without_own drops that one cluster's term v_own^2 -- the variance a
        genuinely new configuration would get, used only to fit lam on held-out TRAINING configs."""
        Q = jnp.asarray(self.Q, jnp.float64)
        out, loo = [], []
        for i in range(0, len(Phi), chunk):
            v = (jnp.asarray(Phi[i:i + chunk]) * jnp.asarray(self.dinv)[None, :]) @ Q
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
        if self.Q is not None:
            return self.lam ** 2 * self.misspec_var_rows(Phi)
        return self.kappa ** 2 * self.var_rows(Phi)

    def forces_std(self, Frows, groups=None):
        """Per-atom force std from force rows (N, 3, L), numpy or device.  With a schema-3 group table
        and groups: lam_rms[g] sqrt(tr V); else sqrt(sum_c force_var_rows) (schema 1/2)."""
        if self.group_table is not None and groups is not None:
            v = np.trace(self.atom_shape(Frows), axis1=1, axis2=2)
            return np.asarray(self.group_table["lam_rms"])[groups] * np.sqrt(v)
        v = self.force_var_rows(Frows.reshape(-1, Frows.shape[-1])).reshape(-1, 3)
        return np.sqrt(np.maximum(v.sum(1), 0.0))

    def save(self, path, dtype=np.float32):
        np.savez(path, mean=self.mean, chol=np.asarray(self.chol, dtype), dinv=self.dinv, kappa=self.kappa,
                 h=self.h, groups=np.asarray(self.groups), body_col=self.body_col, schema=SCHEMA,
                 meta_json=np.frombuffer(json.dumps(self.meta).encode(), np.uint8),
                 lam=self.lam, **({} if self.Q is None else {"Q": np.asarray(self.Q, dtype)}),
                 **self._extra_arrays(dtype))

    def _extra_arrays(self, dtype):
        js = lambda d: np.frombuffer(json.dumps(d).encode(), np.uint8)
        out = {"force_shape": np.array(self.force_shape), "eps": self.eps}
        if self.R is not None:
            out["R"] = np.asarray(self.R, dtype)
        if self.group_consts is not None:
            out["group_consts_json"] = js(self.group_consts)
        if self.group_table is not None:
            out["group_table_json"] = js(self.group_table)
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
        from .support import unflatten_support
        js = lambda k: json.loads(bytes(z[k]).decode()) if k in z.files else None
        if int(z["schema"]) not in (1, 2, 3):
            raise ValueError(f"unsupported posterior schema {int(z['schema'])}")
        return ARDPosterior(z["mean"], z["chol"].astype(np.float64), z["dinv"], float(z["kappa"]), z["h"],
                            tuple(int(g) for g in z["groups"]), z["body_col"],
                            json.loads(bytes(z["meta_json"]).decode()),
                            Q=z["Q"].astype(np.float64) if "Q" in z.files else None,
                            lam=float(z["lam"]) if "lam" in z.files else 1.0,
                            R=jnp.asarray(z["R"], jnp.float64) if "R" in z.files else None,
                            force_shape=str(z["force_shape"]) if "force_shape" in z.files else "iso",
                            eps=float(z["eps"]) if "eps" in z.files else 1e-3,
                            group_consts=js("group_consts_json"), group_table=js("group_table_json"),
                            cal={k: z[f"cal_{k}"] for k in ("scores", "groups", "cfg", "src")}
                            if "cal_scores" in z.files else None,
                            support=unflatten_support({k[8:]: z[k] for k in z.files if k.startswith("support_")})
                            if any(k.startswith("support_") for k in z.files) else None)


def ard_posterior(ev, h, kappa, meta):
    Ms, bv, lam, _ = ev._parts(jnp.asarray(h, float))
    c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
    x = cho_solve((c, low), bv)
    keep = {k: meta[k] for k in ("n_B", "n_pair", "NZ", "rcut", "elements") if k in meta}
    if "NZ" not in keep and "elements" in keep:          # model meta carries elements, not NZ
        keep["NZ"] = len(keep["elements"])
    # chol stays a float64 device array: var_rows runs once per batch (predict_ard, _val_errors), and
    # jnp.asarray of a numpy factor would re-upload the L x L matrix (1.8 GB at L = 15k) on every call
    return ARDPosterior(np.asarray(ev.dinv * x), jnp.asarray(c, jnp.float64), np.asarray(ev.dinv), float(kappa),
                        np.asarray(h, float), ev.groups, ev.body_col, keep)


def sandwich_scores(post, prob, ds, sig):
    """Prior-scaled cluster scores G~ (L, n_cfg) of the configurations of ds at the posterior mean:
    g~_c = D^-1 sum_{i in c} rho_i psi_i over every E/F/V row i of config c, psi_i = phi_i w_i/sigma_q,
    rho_i = (y_i - phi_i c) w_i/sigma_q.  Padded configs (cfg_mask) are dropped; padded nodes carry
    node_cfg == C and land in a discarded extra segment."""
    from .rows import linear_rows
    L = prob.cfg.len_basis
    c, dinv = jnp.asarray(post.mean), jnp.asarray(post.dinv)
    inv = jnp.asarray(1.0 / np.asarray(sig, float))

    @jax.jit
    def scores(bt):
        r = linear_rows(prob.model, prob.cfg, bt)[0]
        C = r.E.shape[0]
        P = jnp.concatenate([r.E, r.F.reshape(-1, L), r.V.reshape(-1, L)])
        y = jnp.concatenate([bt.y_E, bt.y_F.reshape(-1), bt.y_V.reshape(-1)])
        w = jnp.concatenate([bt.w_E * inv[0], jnp.repeat(bt.w_F, 3) * inv[1], jnp.repeat(bt.w_V, 6) * inv[2]])
        cid = jnp.concatenate([jnp.arange(C), jnp.repeat(bt.node_cfg, 3), jnp.repeat(jnp.arange(C), 6)])
        g = jax.ops.segment_sum(P * ((y - P @ c) * w * w)[:, None], cid, num_segments=C + 1)[:C]
        return g * dinv[None, :]

    out = []
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        out.append(np.asarray(scores(bt))[np.asarray(bt.cfg_mask)])
    return np.concatenate(out).T


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


def predict_ard(post, prob, ds, node_chunk=256):
    """Posterior predictive on a Dataset: means from the ARD mean; F_var the served calibrated force
    variance, E_var/V_var the untempered posterior variances.  Schema 3 (post.group_table set):
    F_var = lam_rms[g]^2 diag V per atom (g = post.groups_of(batch)), so sum_a F_var = forces_std^2.
    Schema 1/2: `force_var_rows` (lam^2 x cluster sandwich when post.Q is set, else kappa^2 x the
    posterior variance)."""
    from .predict import _pack
    from .rows import chunked_rows_fn
    L = prob.cfg.len_basis
    rows_fn = chunked_rows_fn(prob.model, prob.cfg, node_chunk)
    lam_rms = None if post.group_table is None else np.asarray(post.group_table["lam_rms"], float)
    outs = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = rows_fn(b)
        Fn = np.asarray(r.F)                                                   # (Ncap, 3, L)
        E, F, V = np.asarray(r.E), Fn.reshape(-1, L), np.asarray(r.V).reshape(-1, L)
        if lam_rms is not None:
            g = post.groups_of(b)
            Fv = lam_rms[g][:, None] ** 2 * np.diagonal(post.atom_shape(Fn), axis1=1, axis2=2)
            Fv = np.where(np.asarray(b.node_mask)[:, None], Fv, 0.0)
        else:
            Fv = post.force_var_rows(F).reshape(-1, 3)
        outs.append((E @ post.mean, post.var_rows(E), (F @ post.mean).reshape(-1, 3), Fv,
                     (V @ post.mean).reshape(-1, 6), post.var_rows(V).reshape(-1, 6)))
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


def _val_atoms(post, prob, ds, r1=None, shape=True, own_col=None):
    """Per live force-labelled atom (w_F > 0) of ds, in one pass over its rows: the force error at
    post.mean, the untempered s^2, (shape) the unscaled shape V = post.atom_shape, and (r1) the shell
    features z, d.  own_col: (n_cfg(ds),) the Q column of each config of ds -- the #18 own-cluster-out
    rule of the legacy "mixed" ablation (held-out configs are training configs of the full refit): V
    then drops that one cluster's column of Q, and v_incl = tr V with it."""
    from .conformal import shell_features
    from .rows import chunked_rows_fn
    L = prob.cfg.len_basis
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
    from .support import build_support, fit_pca
    prob = built.prob
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
    pca = fit_pca(Xp)
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
                         cap, cfg.seed, grp=np.asarray(grp)[k])


def conformal_scores(e, V, force_shape, eps):
    """iso: s = |e| / sqrt(v/3); aniso: s = sqrt(e^T (V + eps (v/3) I)^-1 e), v = tr V."""
    v = np.trace(V, axis1=1, axis2=2)
    if force_shape == "aniso":
        M = V + eps * (v / 3)[:, None, None] * np.eye(3)
        return np.sqrt(np.einsum("na,na->n", e, np.linalg.solve(M, e[:, :, None])[:, :, 0]))
    return np.sqrt(np.sum(e * e, 1) / (v / 3))


def _ard_fit_warnings(stage, info, names):
    """Warning lines for an ARD evidence fit that did not converge or ended with a hyperparameter
    on its bound (a scale at a_floor is the conditioning guard, not the evidence optimum)."""
    out = []
    if not info.get("success", True):
        out.append(f"WARNING: ARD {stage} evidence fit did not converge: {info.get('message', '?')}")
    hit = [n for n, b in zip(names, info.get("at_bound", [])) if b]
    if hit:
        out.append(f"WARNING: ARD {stage} evidence fit ended at a bound for {', '.join(hit)}")
    return out


def run_ard_stage(cfg, data, built, theta, log=print, full_stats=None):
    """The schema-3 ARD stage (docs/specs/2026-09-30-conformal-force-sigma-design.md section 4):

    1. shell features of every training atom -> r1, z*, band edges (all of T), configuration strata,
       and the stratified split T = T_fit + T_val;
    2. the hold-out posterior P_fit on T_fit, with its own shape (PRESS jackknife R_fit, or the legacy
       sandwich Q_fit, or A_fit^-1 for ard_variance "kappa");
    3. T_val scores from P_fit's errors and shape (or, for the legacy "mixed" ablation, from the
       served posterior's own-cluster-out sandwich);
    4. the served posterior P on all of T (started from the subset optimum) and its stored shape;
    5. per-group configuration-weighted lam_rms and q from the T_val scores, stored in the posterior.
    kappa and the scalar lam of #18 are still reported for comparison.

    full_stats: the linear statistics of data.ds_train (`stats.linear_statistics`) the caller has
    already cached -- the pipeline objective's.  The joint full refit then uses them as they are
    instead of a second pass over the training set (spec 3); sequential mode ignores them."""
    import time
    from .clusters import row_clusters
    from .conformal import (assign_groups, band_edges, config_strata, group_scales, n_groups,
                            shell_reference, stratified_split)
    from .data import build_dataset
    from .jackknife import press_scores, shape_factor
    mode, val_frac, cond_max = cfg.ard_mode, cfg.ard_val_frac, cfg.ard_cond_max
    variance, variant, source = cfg.ard_variance, cfg._shape_variant, cfg._score_source
    t0 = time.time()
    prob = built.prob
    body_col = body_order_columns(data.meta, prob.cfg)
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
    ds_fit = build_dataset(fit_, data.meta, data.E0, cfg.batch)
    ds_val = build_dataset(val, data.meta, data.E0, cfg.batch)
    ev = ARDEvidence(ard_statistics(theta, prob, ds_fit, mode), np.asarray(prob.gamma), body_col)
    names = (["log_sigma_E", "log_sigma_F", "log_sigma_V"] if ev.joint else []) + [f"a_{g}body" for g in ev.groups]
    h_fit, v_fit, info_fit = fit_ard(ev, ev.h0(theta), cond_max)
    for w in _ard_fit_warnings("fit-subset", info_fit, names):
        log(w)
    post_fit = ard_posterior(ev, h_fit, 1.0, data.meta)
    K_fit = 0
    if variance == "sandwich" and source == "fit":
        if variant == "press":
            rc, K_fit = row_clusters(ds_fit, fit_, ell)
            Gs, _ = press_scores(post_fit, prob, ds_fit, rc, K_fit, ev.sigmas(h_fit), mode=cfg.ard_press)
            post_fit = post_fit._replace(R=shape_factor(post_fit, Gs, cfg.ard_shape_tau))
        else:
            Gs = sandwich_scores(post_fit, prob, ds_fit, ev.sigmas(h_fit))
            K_fit = int(Gs.shape[1])
            post_fit = post_fit._replace(Q=sandwich_factor(post_fit, Gs))
        del Gs
    del ev

    # 3. T_val errors (and, unless "mixed", the scores' shape) from P_fit
    E = _val_atoms(post_fit, prob, ds_val, r1=r1, shape=(source == "fit"))
    del post_fit                           # free the subset fit's L x L Cholesky factor before the refit

    # 4. the served posterior on all of T and its shape
    st = (joint_ard_stats(full_stats) if (full_stats is not None and mode == "joint")
          else ard_statistics(theta, prob, data.ds_train, mode))
    ev = ARDEvidence(st, np.asarray(prob.gamma), body_col)
    del st
    v_start = ev.value_and_grad(np.clip(h_fit, *ev.bounds(h_fit, cond_max)))[0]   # refit's start
    h, v, info = fit_ard(ev, h_fit, cond_max)
    for w in _ard_fit_warnings("full", info, names):
        log(w)
    post = ard_posterior(ev, h, 1.0, data.meta)
    K, lev = 0, np.zeros(0)
    if variance == "sandwich":
        if variant == "press":
            rc, K = row_clusters(data.ds_train, data.train, ell)
            Gf, lev = press_scores(post, prob, data.ds_train, rc, K, ev.sigmas(h), mode=cfg.ard_press)
            post = post._replace(R=shape_factor(post, Gf, cfg.ard_shape_tau))
        else:
            # Q is a float64 device array (sandwich_factor): its per-batch consumers call jnp.asarray on
            # it, a no-op there -- as numpy it would re-upload the (L, n_cfg) factor on every call
            Gf = sandwich_scores(post, prob, data.ds_train, ev.sigmas(h))
            K, lev = int(Gf.shape[1]), np.zeros(Gf.shape[1])
            post = post._replace(Q=sandwich_factor(post, Gf))
        del Gf
    # held-out rows against the served posterior: its s^2 (kappa) and, for "mixed", the #18 shape
    Ef = _val_atoms(post, prob, ds_val, shape=False, own_col=val_idx if source == "mixed" else None)
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

    # 5. per-group scales on T_val
    s = conformal_scores(e, V, cfg.ard_force_shape, cfg.ard_shape_eps)
    gv = assign_groups(E.z[ok], E.d[ok], z_star, edges)
    cv = np.asarray(val_idx)[E.cfg[ok]]            # training-set index of each calibration atom's config
    tab = group_scales(s, gv, cv, G, 1 - cfg.ard_coverage, cfg.ard_n_min)
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
    n_near1 = int(np.sum(1.0 - lev < 1e-8))
    if n_near1:
        log(f"WARNING: ARD shape: {n_near1} of {K} clusters have leverage 1 - lambda_max(H_kk) < 1e-8; "
            f"their PRESS scores are accurate only to ~eps/(1 - lambda)")
    shp = post.R if post.R is not None else post.Q
    lq = (lambda f: float(f(lev))) if len(lev) else (lambda f: None)
    report = {"mode": mode, "body_groups": list(ev.groups), "h": h.tolist(), "h_names": names,
              "logev_full": v, "logev_full_start": v_start, "optimiser": info, "optimiser_fit": info_fit,
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
                        "lev_max": lq(np.max), "n_lev_near1": n_near1},
              "groups": tab.to_dict(),
              "split": {"n_fit": len(fit_), "n_val": len(val), "val_idx": np.asarray(val_idx).tolist(),
                        "strata": np.bincount(strata, minlength=G).tolist()},
              "transfer": {"f": val_frac, "N_fit": len(fit_), "N": N, "score_source": source}}
    if variance == "sandwich":
        report["val_rms_z_sandwich"] = float(np.sqrt(np.mean(e2 / (lam ** 2 * vtr / 3)) / 3))
    if cfg.ard_laplace:
        lap = laplace_hypers(ev, h)
        report["laplace_std_h"] = lap["std"].tolist()
        report["laplace_eigs"] = lap["eigs"].tolist()
    report["seconds"] = time.time() - t0
    return ARDResult(post, report)
