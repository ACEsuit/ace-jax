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

SCHEMA = 1


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
    chol: np.ndarray          # (L, L) lower Cholesky factor of S = D^-1 A D^-1
    dinv: np.ndarray          # (L,) 1 / Gamma
    kappa: float              # temperature (sigma inflation)
    h: np.ndarray
    groups: tuple
    body_col: np.ndarray
    meta: dict                # n_B, n_pair, NZ, rcut, elements

    def var_rows(self, Phi, chunk=4096):
        """Untempered posterior variance phi A^-1 phi^T of each row of Phi (n, L)."""
        c = jnp.asarray(self.chol, jnp.float64)
        out = []
        for i in range(0, len(Phi), chunk):
            v = solve_triangular(c, (jnp.asarray(Phi[i:i + chunk]) * jnp.asarray(self.dinv)[None, :]).T, lower=True)
            out.append(np.asarray(jnp.sum(v * v, axis=0)))
        return np.concatenate(out) if out else np.zeros(0)

    def forces_std(self, Frows):
        """Tempered per-atom force std kappa * sqrt(sum_c phi_c A^-1 phi_c^T) from force rows (N, 3, L),
        a numpy or a device array: a device array is solved in place, chunk by chunk, never copied
        to the host (the rows are N*3*L*8 bytes)."""
        v = self.var_rows(Frows.reshape(-1, Frows.shape[-1])).reshape(-1, 3)
        return self.kappa * np.sqrt(np.maximum(v.sum(1), 0.0))

    def save(self, path, dtype=np.float32):
        np.savez(path, mean=self.mean, chol=np.asarray(self.chol, dtype), dinv=self.dinv, kappa=self.kappa,
                 h=self.h, groups=np.asarray(self.groups), body_col=self.body_col, schema=SCHEMA,
                 meta_json=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))

    @staticmethod
    def load(path):
        z = np.load(pathlib.Path(path))
        if int(z["schema"]) != SCHEMA:
            raise ValueError(f"unsupported posterior schema {int(z['schema'])}")
        return ARDPosterior(z["mean"], z["chol"].astype(np.float64), z["dinv"], float(z["kappa"]), z["h"],
                            tuple(int(g) for g in z["groups"]), z["body_col"],
                            json.loads(bytes(z["meta_json"]).decode()))


def ard_posterior(ev, h, kappa, meta):
    Ms, bv, lam, _ = ev._parts(jnp.asarray(h, float))
    c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
    x = cho_solve((c, low), bv)
    keep = {k: meta[k] for k in ("n_B", "n_pair", "NZ", "rcut", "elements") if k in meta}
    if "NZ" not in keep and "elements" in keep:          # model meta carries elements, not NZ
        keep["NZ"] = len(keep["elements"])
    return ARDPosterior(np.asarray(ev.dinv * x), np.asarray(c), np.asarray(ev.dinv), float(kappa),
                        np.asarray(h, float), ev.groups, ev.body_col, keep)


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
    """Posterior predictive on a Dataset: means from the ARD mean; F_var tempered by kappa^2 (the
    calibrated quantity), E_var/V_var the untempered posterior variances."""
    from .predict import _pack
    from .rows import linear_rows_chunked
    L, k2 = prob.cfg.len_basis, post.kappa ** 2
    outs = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = linear_rows_chunked(prob.model, prob.cfg, b, node_chunk=node_chunk)
        E, F, V = np.asarray(r.E), np.asarray(r.F).reshape(-1, L), np.asarray(r.V).reshape(-1, L)
        outs.append((E @ post.mean, post.var_rows(E), (F @ post.mean).reshape(-1, 3),
                     k2 * post.var_rows(F).reshape(-1, 3), (V @ post.mean).reshape(-1, 6),
                     post.var_rows(V).reshape(-1, 6)))
    return _pack(outs, prob, ds)


class ARDResult(NamedTuple):
    posterior: ARDPosterior
    report: dict


def _val_errors(post, prob, ds):
    """Per-atom squared force error and untempered s2 on the live force rows of ds."""
    from .rows import linear_rows_chunked
    L, e2, s2 = prob.cfg.len_basis, [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        live = np.asarray(b.w_F) > 0
        if not live.any():
            continue
        F = np.asarray(linear_rows_chunked(prob.model, prob.cfg, b).F)[live]          # (n, 3, L)
        pred = (F.reshape(-1, L) @ post.mean).reshape(-1, 3)
        e2.append(np.sum((np.asarray(b.y_F)[live] - pred) ** 2, 1))
        s2.append(post.var_rows(F.reshape(-1, L)).reshape(-1, 3).sum(1))
    if not e2:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(e2), np.concatenate(s2)


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
    """Fit ARD on a train subset, choose kappa on the held-out rest (never the test set), then refit
    on ALL training data (started from the subset optimum) and keep kappa.

    full_stats: the linear statistics of data.ds_train (`stats.linear_statistics`) the caller has
    already cached -- the pipeline objective's.  The joint full refit then uses them as they are
    instead of a second pass over the training set (spec 3); sequential mode ignores them."""
    import time
    from .data import build_dataset
    mode, val_frac, cond_max = cfg.ard_mode, cfg.ard_val_frac, cfg.ard_cond_max
    t0 = time.time()
    prob = built.prob
    body_col = body_order_columns(data.meta, prob.cfg)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(data.train))
    nval = max(1, int(round(val_frac * len(data.train))))
    val = [data.train[i] for i in idx[:nval]]
    fit_ = [data.train[i] for i in idx[nval:]]
    if not any(c.forces is not None and c.w_F > 0 for c in val):
        raise ValueError(f"the ARD validation split ({nval} configs, ard_val_frac={val_frac}) has "
                         f"no force labels to fit the temperature kappa on")
    ds_fit = build_dataset(fit_, data.meta, data.E0, cfg.batch)
    ds_val = build_dataset(val, data.meta, data.E0, cfg.batch)
    ev = ARDEvidence(ard_statistics(theta, prob, ds_fit, mode), np.asarray(prob.gamma), body_col)
    names = (["log_sigma_E", "log_sigma_F", "log_sigma_V"] if ev.joint else []) + [f"a_{g}body" for g in ev.groups]
    h_fit, v_fit, info_fit = fit_ard(ev, ev.h0(theta), cond_max)
    for w in _ard_fit_warnings("fit-subset", info_fit, names):
        log(w)
    post_fit = ard_posterior(ev, h_fit, 1.0, data.meta)
    e2, s2 = _val_errors(post_fit, prob, ds_val)
    del post_fit                           # free the subset fit's L x L Cholesky factor before the refit
    ok = s2 > 0                            # atoms with zero force rows (isolated, 1-atom configs): no information
    e2, s2 = e2[ok], s2[ok]
    if len(e2) == 0:
        raise ValueError(f"the ARD validation split (ard_val_frac={val_frac}) has no atom with a "
                         f"non-zero force row to fit the temperature kappa on")
    kappa = kappa_closed_form(e2, s2)
    log(f"ARD: fit-subset logev {v_fit:.2f} ({info_fit['message']}, nit {info_fit['nit']}); "
        f"kappa {kappa:.3f} from {len(e2)} held-out atoms")
    del ev
    st = (joint_ard_stats(full_stats) if (full_stats is not None and mode == "joint")
          else ard_statistics(theta, prob, data.ds_train, mode))
    ev = ARDEvidence(st, np.asarray(prob.gamma), body_col)
    del st
    v_start = ev.value_and_grad(np.clip(h_fit, *ev.bounds(h_fit, cond_max)))[0]   # refit's start
    h, v, info = fit_ard(ev, h_fit, cond_max)
    for w in _ard_fit_warnings("full", info, names):
        log(w)
    post = ard_posterior(ev, h, kappa, data.meta)
    report = {"mode": mode, "groups": list(ev.groups), "h": h.tolist(), "h_names": names,
              "logev_full": v, "logev_full_start": v_start, "optimiser": info, "optimiser_fit": info_fit,
              "a_floor": ev.a_floor,
              "tempered_quantities": ["F"],        # E_var / V_var are the untempered posterior variances
              "kappa": kappa, "n_val_atoms": int(len(e2)), "n_val_configs": len(val), "n_fit_configs": len(fit_),
              "val_rms_z_untempered": float(np.sqrt(np.mean(e2 / (s2 / 3)) / 3)),
              "val_rms_z_tempered": float(np.sqrt(np.mean(e2 / (kappa ** 2 * s2 / 3)) / 3)),
              "val_nll_untempered": _force_nll(e2, s2, 1.0), "val_nll_tempered": _force_nll(e2, s2, kappa),
              "seconds": time.time() - t0}
    if cfg.ard_laplace:
        lap = laplace_hypers(ev, h)
        report["laplace_std_h"] = lap["std"].tolist()
        report["laplace_eigs"] = lap["eigs"].tolist()
    return ARDResult(post, report)
