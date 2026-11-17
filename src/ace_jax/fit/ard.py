"""The ARD stage of the E/F/V fit (`aj fit --uq ard`): its statistics, the calibration on a validation
split and the predictions, on the generic evidence and posterior of `ace_jax.uq.ard` (which says what
they are; its names are re-exported here)."""
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..uq.ard import (E0_GROUP, GP_GROUP, SCHEMA, _NEED3, ARDEvidence, ARDPosterior, ARDQRStats,  # noqa: F401
                      ARDStats, _force_nll, _pad_cols, _projected_gradient, ard_posterior, ard_qr_stats,
                      conformal_scores, joint_ard_stats,
                      default_cond_max, fit_ard, gp_body_columns, kappa_closed_form, laplace_hypers,
                      newton_polish, sandwich_factor)
from .linear.prior_root import PriorRoot



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


def ard_statistics(theta, prob, ds, mode, columns="linear", solver="cholesky"):
    """joint: per-quantity statistics (3 L^2 matrices).  sequential: the combined Gram at the
    linear MAP noise scales, accumulated in one pass (1 L^2 matrix) -- the low-memory mode.
    columns "linear" (uq "ard"): the L linear columns only -- a hybrid problem's inducing columns never
    enter, and the joint Gram is hyperparameter-independent.  columns "joint" (uq "ard-gp"): the joint
    design [B | k_theta(B, B_M)] at theta (Dt = L + M), streamed by stats.sufficient_statistics; sequential
    mode then combines its per-quantity Grams at theta's noise scales (3 Dt^2 transiently).
    solver "qr" (columns "linear" only): ARDQRStats from stats.linear_qr_statistics, the factors the
    linear arm's QR LML uses (sequential mode holds 3 L^2 transiently while combining them)."""
    from .rows import ROWS_EDGE_BUDGET
    if solver not in ("cholesky", "qr"):
        raise ValueError(f"ard solver must be 'cholesky' or 'qr', got {solver!r}")
    if solver == "qr":
        if columns != "linear":
            raise ValueError("the ARD QR solver is the linear arm's (columns 'linear'); uq 'ard-gp' uses cholesky")
        if mode not in ("joint", "sequential"):
            raise ValueError(f"ard mode must be 'joint' or 'sequential', got {mode!r}")
        from .stats import linear_qr_statistics
        qs = linear_qr_statistics(prob.model, prob.cfg, ds, cached_rows=True)   # no recompile on a repeat
        ls = None if mode == "joint" else np.array([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
        return ard_qr_stats(qs, ls)
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
    from .gp.predict import _pack
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
    from ..uq.conformal import shell_features
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
    from ..uq.conformal import shell_features
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
    """The covariate-shift reference (uq/support.py): per-species descriptor PCA from a random sample of
    training batches (<= ard_support_max_atoms atoms per species), and the T_val atoms (batch, node) = bn
    with their conformal scores and configuration ids.  Site descriptors are evaluated only on those."""
    from .gp.inducing import site_features
    from ..uq.support import build_support, explained_variance, fit_pca, n_pass, support_body, support_features
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
    if info.get("degenerate"):
        out.append(f"WARNING: ARD {stage} evidence fit is DEGENERATE: every ARD scale is on the floor a_floor "
                   f"{info.get('a_floor', float('nan')):.2f} ({info.get('solver', '?')}, ard_cond_max "
                   f"{info.get('cond_max', float('nan')):.0e}) with the evidence increasing below it; the prior "
                   f"is the conditioning guard's, not the evidence optimum")
    return out


def _check_degenerate(cfg, stage, info):
    """Raise on a degenerate evidence fit (issue #136) unless cfg.ard_allow_degenerate."""
    if info.get("degenerate") and not getattr(cfg, "ard_allow_degenerate", False):
        raise RuntimeError(
            f"ARD {stage} evidence fit is degenerate: every ARD scale ended on the prior floor a_floor = "
            f"{info.get('a_floor', float('nan')):.3f} (solver {info.get('solver')}, cond(S) {info.get('cond_S', float('nan')):.1e}, "
            f"ard_cond_max {info.get('cond_max', float('nan')):.0e}) with the evidence still increasing below it, so the "
            f"posterior would be the floor's, not the ARD optimum. Use ard_solver 'qr' (the default for uq ard) or "
            f"a larger ard_cond_max; pass --ard-allow-degenerate (ard_allow_degenerate) to write it anyway")


def _gp_parts(cfg, prob, theta, body_col):
    """The uq-dependent pieces of the stage: the linear arm's (linear statistics, body_col, D, linear rows),
    or ard-gp's joint design at theta (joint statistics, GP_GROUP columns, the prior root with chol(K_MM)^T,
    the joint rows; theta stored with the posterior)."""
    if cfg.uq != "ard-gp":
        return {"columns": "linear", "body": body_col, "root": None, "rows_fn": None, "theta": None}
    return {"columns": "joint", "body": gp_body_columns(body_col, prob.ind.XM.shape[0]),
            "root": prior_root(prob, theta), "rows_fn": rows_fn_for(prob, theta), "theta": theta}


def _ard_solver(cfg):
    """cfg.ard_solver with "auto" resolved (FitConfig.validate does the same): qr on the linear arm."""
    s = getattr(cfg, "ard_solver", "auto")
    return ("cholesky" if cfg.uq == "ard-gp" else "qr") if s == "auto" else s


def _h_names(ev):
    return ((["log_sigma_E", "log_sigma_F", "log_sigma_V"] if ev.joint else [])
            + ["a_GP" if g == GP_GROUP else f"a_{g}body" for g in ev.groups])


def _stage_hypers(cfg, ev, theta, start, stage, names, log=print):
    """(h, logev, info) of one of the stage's posteriors: the evidence fit from `start` (ard_prior "fit"), or the
    MAP's own prior ev.h0(theta) with no fit (ard_prior "fixed": the BLR posterior, a_k = -2 log sigma_c).  The
    fixed prior can sit below the evidence fit's a_floor (the BLR fits of near-interpolating data do, by ~12
    nats): the QR solver factors the rows there; the Cholesky one may not, so it is refused."""
    if getattr(cfg, "ard_prior", "fit") == "fixed":
        if ev.solver != "qr":
            raise ValueError("ard_prior 'fixed' needs ard_solver 'qr' (the MAP's prior can be far below the "
                             "Cholesky floor)")
        h = np.asarray(ev.h0(theta), float)
        v = ev.value_and_grad(h)[0]
        ev.bounds(h, cfg.ard_cond_max)            # sets ev.a_floor / cond_max for the report; the box is not used
        below = bool(np.any(h[3 if ev.joint else 0:] < ev.a_floor))
        log(f"ARD {stage}: fixed prior (ard_prior 'fixed'): the MAP's own a = {h[-1]:.3f}, no evidence fit; "
            f"logev {v:.2f}" + (f"; below the evidence fit's floor a_floor {ev.a_floor:.3f}" if below else ""))
        return h, v, {"message": "fixed prior (ard_prior 'fixed')", "nit": 0, "success": True, "fixed": True,
                      "at_bound": [False] * len(h), "degenerate": False, "solver": ev.solver,
                      "a_floor": float(ev.a_floor), "below_floor": below}
    h, v, info = fit_ard(ev, start, cfg.ard_cond_max)
    for w in _ard_fit_warnings(stage, info, names):
        log(w)
    _check_degenerate(cfg, stage, info)
    return h, v, info


def _holdout_posterior(cfg, data, prob, theta, configs, body_col, ell, h0=None, stage="fit-subset", log=print):
    """A hold-out posterior on the training subset `configs` with its own shape (PRESS jackknife R, the
    legacy sandwich Q, or none for ard_variance "kappa" / the "mixed" ablation).  h0: the evidence fit's
    start (default the prior's).  Returns (post, h, logev, info, K, h_names)."""
    from ..uq.clusters import row_clusters
    from .data import build_dataset
    from .jackknife import press_scores, shape_factor
    mode = cfg.ard_mode
    gp = _gp_parts(cfg, prob, theta, body_col)
    ds = build_dataset(configs, data.meta, data.E0, cfg.batch, pack=cfg.pack_mode, log=log)
    ev = ARDEvidence(ard_statistics(theta, prob, ds, mode, columns=gp["columns"], solver=_ard_solver(cfg)),
                     ard_gamma(prob), gp["body"], root=gp["root"])
    names = _h_names(ev)
    h, v, info = _stage_hypers(cfg, ev, theta, ev.h0(theta) if h0 is None else h0, stage, names, log=log)
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
    from ..uq.clusters import row_clusters
    from ..uq.conformal import (assign_groups, band_edges, config_strata, group_scales, n_groups, pooled_lam_rms,
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
    from .stats import QRStats
    solver = _ard_solver(cfg)
    if full_stats is not None and mode == "joint":
        if solver == "qr" and isinstance(full_stats, QRStats):
            st = ard_qr_stats(full_stats)
        elif solver == "qr":                 # the objective cached a Gram: one QR pass here
            st = ard_statistics(theta, prob, data.ds_train, mode, columns=gp["columns"], solver=solver)
        else:
            st = joint_ard_stats(full_stats)
    else:
        st = ard_statistics(theta, prob, data.ds_train, mode, columns=gp["columns"], solver=solver)
    ev = ARDEvidence(st, ard_gamma(prob), gp["body"], root=gp["root"])
    del st
    fixed = getattr(cfg, "ard_prior", "fit") == "fixed"
    v_start = (None if fixed else ev.value_and_grad(np.clip(h_fit, *ev.bounds(h_fit, cond_max)))[0])   # refit's start
    h, v, info = _stage_hypers(cfg, ev, theta, h_fit, "full", names, log=log)
    if fixed:
        v_start = v
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
    if cfg.ard_support and ev.root.M == 0:      # only ACECalculator serves the flag, and it refuses ard-gp
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
    report = {"mode": mode, "prior": getattr(cfg, "ard_prior", "fit"), "body_groups": list(ev.groups), "gp_cols": ev.root.M, "h": h.tolist(), "h_names": names,
              "logev_full": v, "logev_full_start": v_start, "optimiser": info, "optimiser_fit": info_fit,
              # the evidence's roundoff at the endpoint, as the polish measured it (its spread over 1e-15
              # moves of h): two fits of the same data agree in logev_full only to ~this, not to 1e-15
              "logev_full_noise": info.get("newton", {}).get("noise"),
              "logev_full_gnoise": info.get("newton", {}).get("gnoise"),     # per hyperparameter, likewise
              "a_floor": ev.a_floor, "solver": ev.solver, "cond_max": ev.cond_max,
              # issue #136: every a_k on the floor with the evidence increasing below it (served only
              # with ard_allow_degenerate); per fit: the full refit, then the fit subset
              "ard_degenerate": bool(info.get("degenerate", False)),
              "ard_degenerate_fit": bool(info_fit.get("degenerate", False)),
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


def prior_root(prob, theta):
    """The ARD prior root of prob at theta: Gamma (ard_gamma) on the linear columns, chol(K_MM(theta))^T on
    the M inducing columns (None when M = 0)."""
    from .gp.kernels import K_MM
    dinv = 1.0 / ard_gamma(prob)
    if prob.ind.XM.shape[0] == 0:
        return PriorRoot.diag(dinv)
    ind = prob.ind
    K = K_MM(theta, prob.spec, ind.XM, ind.SM, ind.ZM, ind.embed)
    return PriorRoot(dinv, jnp.linalg.cholesky(K).T)
