"""Joint VarPro learning of the tensor radials W and P sqrt-density features.

The site energy is E_i = c_z . X_i + sum_p d[p, z] ssqrt(eta[p, z] . (mask * X_i))
(eval.fs_model).  At fixed (W, eta) it is linear in [c | d], so the readout is
projected out exactly as in radial_learn, now on the widened statistics
fit.density.linear_density_statistics; the outer variables are the radial V
and the raw density weights H, with eta = normalise_rho(H, S) fixing each
density's scale gauge (unit mean-square rho on the training sites, S the
per-species second moment at the initial radials -- the analogue of the radial
gauge Gram Q).  d absorbs the scale and the sign of each density.

The optimiser is L-BFGS on u = [vec(V) ; r_eta vec(H)] ("joint"), r_eta =
sqrt(|h_H| / |h_V|) from one Hessian-vector-product curvature probe per block per
round (so a unit step in u moves both blocks by a curvature-matched amount;
clamped to r_eta >= R_ETA_MIN so density steps are never amplified more than
1 / R_ETA_MIN; a failed joint line search falls back to alternating blocks for
the rest of that round), or alternating V / H blocks throughout ("alternating"), both
through radial_learn's single compiled `_lbfgs_step`.  theta is re-profiled on
the widened LML after every round.  After learning, eta is frozen and the
final fit is the ordinary Bayesian linear solve over [c | d].
See docs/specs/2026-09-28-radial-density-varpro-design.md.
"""
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from .data import flat_edges
from .density import compact_gamma, density_gamma, linear_density_statistics
from .hypers import from_array, to_array
from .radial_learn import (_gate_setup, gate, holdout_score, lbfgs_loop, learn_radial,
                           projected_residual_from_stats, relative_lambda, relative_lambda_gap,
                           relative_lambda_spec, require_linear, require_x64, theta_map_linear)
from .radial_model import (data_r_range, gap_penalty, normalise, radial_gram, require_analytic, roughness,
                           roughness_matrix, row_active, spectral_penalty, spectral_weights, uniform_gram,
                           with_radial)
from .stats import linear_statistics

MODES = ("joint", "alternating")


def rho_gram(model, W, mask, cfg, ds):
    """Per-species second moment of the masked compact basis at radials W,
    S[z] = sum_{i: z_i = z} (m X_i)(m X_i)^T / n_z, (NZ, D, D); one streaming pass.
    A species with no sites gets S[z] = 0."""
    m = with_radial(model, W)

    def body(carry, b):
        S, n = carry
        Ncap = b.nbr.shape[0]
        rij, send, recv, msk = flat_edges(b.rij, b.nbr, b.nbr_mask)
        X = m.compact_basis(rij, b.node_z[send], b.node_z[recv], send, Ncap, msk) * mask
        oh = jax.nn.one_hot(b.node_z, cfg.NZ) * b.node_mask[:, None]           # (Ncap, NZ)
        return (S + jnp.einsum("nz,nd,ne->zde", oh, X, X), n + oh.sum(0)), None

    D = cfg.D
    (S, n), _ = jax.lax.scan(body, (jnp.zeros((cfg.NZ, D, D)), jnp.zeros(cfg.NZ)), ds)
    return S / jnp.maximum(n, 1.0)[:, None, None]


def normalise_rho(H, S):
    """eta = H scaled per (p, z) to unit mean-square density eta^T S_z eta = 1;
    exactly zero where the second moment vanishes (absent species / dead weights)."""
    q = jnp.einsum("pzd,zde,pze->pz", H, S, H)
    live = q > 0
    return H * jnp.where(live, jax.lax.rsqrt(jnp.where(live, q, 1.0)), 0.0)[..., None]


def init_density(cfg, mask, P, seed=0):
    """p = 0: equal weights on the pair block for every species (the total pair
    density, the FS starting point) -- zero on the B block even under the full
    mask; p >= 1: that plus a seeded 0.1-scale perturbation over the mask, so the
    P columns start distinct."""
    pair = jnp.concatenate([jnp.zeros(cfg.n_B), jnp.ones(cfg.n_pair)]) * mask
    H = jnp.broadcast_to(pair, (P, cfg.NZ, cfg.D))
    if P > 1:
        noise = 0.1 * jnp.asarray(np.random.default_rng(seed).standard_normal((P - 1, cfg.NZ, cfg.D))) * mask
        H = H.at[1:].add(noise)
    return jnp.asarray(H)


def relative_lambda_eta(lam_eta, r0, pen0):
    """Absolute density-prior weight lam_eta * r0 / pen0 (0 for lam_eta = 0),
    pen0 = sum_{p,z} ||Gamma_m * eta0_{p,z}||^2 at the normalised init eta0 =
    normalise_rho(H0, S) -- the analogue of `relative_lambda`'s r0/rough0
    ratio, but normalised against the penalty's own scale rather than an
    arbitrary count of densities (that count leaves the penalty
    unnormalised: on a real prior_diagonal the penalty at init can be many
    orders of magnitude above r0).  Raises if the init is (numerically)
    penalty-free, where the relative weight would be effectively infinite."""
    if lam_eta < 0:
        raise ValueError(f"lam_eta must be >= 0, got {lam_eta}")
    if not lam_eta:
        return 0.0
    if pen0 < 1e-12 * max(abs(r0), 1.0):
        raise ValueError(f"relative lam_eta={lam_eta:g} is undefined: the density penalty at "
                         f"the init pen0={pen0:.3e} is ~0 relative to r0={r0:.3e}; use lam_eta=0")
    return float(lam_eta) * r0 / pen0


def _pack(V, H, r, block):
    """(xb, other): the optimised block and the frozen rest, preconditioned."""
    v, h = r[0] * V.ravel(), r[1] * H.ravel()
    if block == "joint":
        return jnp.concatenate([v, h]), jnp.zeros(0)
    return (v, h) if block == "V" else (h, v)


def _unpack(xb, other, r, block, shapes):
    (sV, sH) = shapes
    nV = math.prod(sV)
    x = xb if block == "joint" else (jnp.concatenate([xb, other]) if block == "V"
                                     else jnp.concatenate([other, xb]))
    return (x[:nV] / r[0]).reshape(sV), (x[nV:] / r[1]).reshape(sH)


def _objective_rd(xb, other, a, model, ds, gamma_w, Q, active, D2, wn, lam, W_ref, sw, lam_spec,
                  U, lam_gap, S, mask, gc, lam_eta, r, cfg, block, shapes):
    """VarPro residual of the widened design plus the radial priors and the density
    shape prior lam_eta * sum ||Gamma_z * eta_{p,z}||^2.  Module-level so it is a
    stable `f` for `_lbfgs_step` (one compile per block kind); cfg, block and
    shapes are statics, r (the preconditioner) is traced."""
    V, H = _unpack(xb, other, r, block, shapes)
    W = normalise(V, Q, active)
    eta = normalise_rho(H, S)
    lin = linear_density_statistics(with_radial(model, W), eta, mask, cfg, ds)
    return (projected_residual_from_stats(from_array(a), lin, gamma_w) + lam * roughness(W, D2, wn)
            + lam_spec * spectral_penalty(W, W_ref, sw) + lam_gap * gap_penalty(W, W_ref, U)
            + lam_eta * jnp.sum((gc[None] * eta) ** 2))


R_ETA_MIN = 0.1       # never amplify density steps more than 10x (Cantor: gradient-RMS scaling gave 3e-3)


def _hvp(xb, v, other, *rest):
    """Hessian-vector product of _objective_rd in xb along v (forward over reverse)."""
    return jax.jvp(lambda y: jax.grad(_objective_rd)(y, other, *rest), (xb,), (v,))[1]


_hvp_rd = jax.jit(_hvp, static_argnums=(22, 23, 24))   # cfg, block, shapes (after xb, v, other)


def _tangent_probe(X, live, key):
    """Random probe over the live coordinates of X (rows = last axis), with each
    row's component along X itself removed: that direction is the row's scale
    gauge (normalise / normalise_rho), where the curvature is exactly zero and
    would bias the block's curvature estimate low."""
    v = jax.random.normal(key, X.shape, X.dtype) * live
    xx = jnp.sum(X * X, -1, keepdims=True)
    return v - jnp.where(xx > 0, jnp.sum(v * X, -1, keepdims=True) / jnp.where(xx > 0, xx, 1.0), 0.0) * X


def _curvature_scale(V, H, active, mask, common, cfg, shapes, key):
    """(r_eta, h_V, h_H): h_b = v_b' Hess v_b / |v_b|^2 for one tangent probe per
    block at r = (1, 1); r_eta = sqrt(|h_H| / |h_V|) equalises the magnitude of
    the two blocks' curvature in u = [V ; r_eta H], clamped below at R_ETA_MIN.
    Magnitudes, not signs: the VarPro objective is often non-convex along both
    probes at the start (negative h), and the scale mismatch is still what the
    line search trips on (saddle-free-Newton reasoning).  1 when either is zero
    or non-finite."""
    one = jnp.ones(2)
    kV, kH = jax.random.split(key)
    vV = _tangent_probe(V, active[..., None], kV)
    vH = _tangent_probe(H, jnp.broadcast_to(mask, H.shape), kH)
    x, other = _pack(V, H, one, "joint")
    h = []
    for vb, sl in ((jnp.concatenate([vV.ravel(), jnp.zeros(H.size)]), slice(0, V.size)),
                   (jnp.concatenate([jnp.zeros(V.size), vH.ravel()]), slice(V.size, None))):
        Hv = _hvp_rd(x, vb, other, *common, one, cfg, "joint", shapes)
        h.append(float(vb[sl] @ Hv[sl]) / max(float(vb[sl] @ vb[sl]), 1e-300))
    hV, hH = h
    ok = np.isfinite(hV) and np.isfinite(hH) and hV != 0 and hH != 0
    return (max(float(np.sqrt(abs(hH) / abs(hV))), R_ETA_MIN) if ok else 1.0), hV, hH


def learn_radial_density(prob, ds, W0, *, mask, P=1, H0=None, mode="joint", theta0=None, profile=True,
                         lam_rough=0.0, rough_weights=None, lam_spec=0.0, spec_p=4.0, lam_gap=0.0,
                         lam_eta=0.0, steps=40, reprofile_every=10, tol=1e-6, patience=3, map_steps=300,
                         seed=0, log=None, Q=None, D2=None, U=None, S=None, r0=None):
    """VarPro-learn radials W and P density weights eta jointly (M = 0).  Same
    round structure, relative priors and stopping rules as radial_learn.learn_radial;
    lam_eta is relative too: lam_eta_abs = lam_eta * r0 / pen0, pen0 = sum
    ||Gamma_m * eta0||^2 at the normalised init eta0 = normalise_rho(H0, S)
    (relative_lambda_eta).  mode "joint" optimises [V; H] together,
    "alternating" spends each round's steps half on V (H fixed) then half on
    H.  Q, D2, U as learn_radial; S = rho_gram at the normalised init
    (computed when None); r0 = the widened projected residual at the start
    (computed when None; only valid with theta0).  H0, when given, is masked
    (H0 * mask) before normalisation, so a caller cannot smuggle weight
    outside the span.  Returns (V, eta, info), both normalised; steps = 0
    returns the normalised init."""
    require_x64()
    require_analytic(prob.model)
    require_linear(prob)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if P < 1:
        raise ValueError(f"P must be >= 1 (P = 0 is radial_learn.learn_radial), got {P}")
    cfg = prob.cfg
    mask = jnp.asarray(mask, jnp.float64)
    prob_w = prob._replace(gamma=density_gamma(prob.gamma, P, cfg.NZ))
    W0 = jnp.asarray(W0, jnp.float64)
    active = row_active(W0)
    if Q is None:
        Q = radial_gram(prob.model, ds)
    if D2 is None:
        D2 = roughness_matrix(prob.model)
    NZw, n_q = W0.shape[0], W0.shape[-1]
    if U is None:
        if lam_gap:
            r_min, _ = data_r_range(ds)
            U = uniform_gram(prob.model, 0.8 * r_min, cfg.rcut)
        else:
            U = jnp.zeros((NZw, NZw, n_q, n_q), jnp.float64)
    wn = jnp.ones(W0.shape[2]) if rough_weights is None else jnp.asarray(rough_weights, jnp.float64)
    V = normalise(W0, Q, active)
    W_ref, sw = V, spectral_weights(n_q, spec_p)
    if S is None:
        S = rho_gram(prob.model, V, mask, cfg, ds)
    H = normalise_rho(init_density(cfg, mask, P, seed) if H0 is None
                      else jnp.asarray(H0, jnp.float64) * mask, S)
    gc = compact_gamma(prob.gamma, cfg)
    stats = lambda V_, H_: linear_density_statistics(with_radial(prob.model, V_), normalise_rho(H_, S), mask, cfg, ds)
    lin0 = None
    if theta0 is not None:
        a = to_array(theta0)
    elif profile:
        lin0 = stats(V, H)
        a = theta_map_linear(prob_w, ds, None, steps=map_steps, seed=seed, lin=lin0)
    else:
        raise ValueError("learn_radial_density: profile=False needs theta0")
    if r0 is None:
        lin0 = stats(V, H) if lin0 is None else lin0
        r0 = projected_residual_from_stats(from_array(a), lin0, prob_w.gamma)
    r0 = float(r0)
    lam = relative_lambda(lam_rough, r0, float(roughness(V, D2, wn)))
    n_active = int(jnp.sum(active))
    lam_spec_abs = relative_lambda_spec(lam_spec, r0, n_active)
    lam_gap_abs = relative_lambda_gap(lam_gap, r0, n_active)
    pen0 = float(jnp.sum((gc[None] * H) ** 2))
    lam_eta_abs = relative_lambda_eta(lam_eta, r0, pen0)
    info = {"trace": [], "reasons": [], "round_lengths": [], "theta": [np.asarray(a)], "precond": [], "curvature": [], "fallbacks": 0,
            "r0": r0, "lam_abs": lam, "lam_spec_abs": lam_spec_abs, "lam_gap_abs": lam_gap_abs,
            "lam_eta": float(lam_eta), "lam_eta_abs": lam_eta_abs, "pen0": pen0,
            "P": int(P), "mode": mode, "steps": 0}
    if log is not None:
        log(f"learn_radial_density: P={P} mode={mode} lam_eta={float(lam_eta):g} "
            f"lam_eta_abs={lam_eta_abs:.6e} pen0={pen0:.6e} r0={r0:.6e}")
    f64 = lambda v: jnp.asarray(v, jnp.float64)
    shapes = (tuple(V.shape), tuple(H.shape))
    done, round_idx = 0, 0
    while done < int(steps):
        round_idx += 1
        n = min(int(reprofile_every), int(steps) - done)
        t_round = time.perf_counter()
        common = (a, prob.model, ds, prob_w.gamma, Q, active, D2, wn, f64(lam), W_ref, sw, f64(lam_spec_abs),
                  U, f64(lam_gap_abs), S, mask, gc, f64(lam_eta_abs))
        if mode == "joint":
            r_eta, hV, hH = _curvature_scale(V, H, active, mask, common, cfg, shapes,
                                             jax.random.PRNGKey(seed * 1000 + round_idx))
            info["curvature"].append((hV, hH))
        else:
            r_eta = 1.0                   # blocks optimised separately: no cross-block scale to match
        r = f64([1.0, r_eta])
        info["precond"].append(float(r[1]))
        alt = lambda m: [("V", (m + 1) // 2), ("H", m // 2)]
        blocks = [("joint", n)] if mode == "joint" else alt(n)
        reasons, obj, fell_back = [], float("nan"), False
        while blocks:
            block, nb = blocks.pop(0)
            if nb == 0:
                continue
            xb, other = _pack(V, H, r, block)
            xb, obj, trace, reason = lbfgs_loop(_objective_rd, xb, steps=nb, tol=tol, patience=patience,
                                                args=(other, *common, r), statics=(cfg, block, shapes))
            V, H = _unpack(xb, other, r, block, shapes)
            info["trace"].extend(trace)
            info["round_lengths"].append(len(trace))
            reasons.append(reason)
            if block == "joint" and reason == "linesearch" and nb - len(trace) - 1 > 0:
                # the joint line search failed (on Cantor the density coordinates are too
                # non-smooth for any single block scale): spend the rest of this round on
                # alternating blocks instead of ending the run; the next round tries joint again
                blocks = alt(nb - len(trace) - 1)
                info["fallbacks"] += 1
                fell_back = True
        info["reasons"].append(reasons[0] if len(reasons) == 1 else "+".join(reasons))
        done += n
        V, H = normalise(V, Q, active), normalise_rho(H, S)
        if profile:
            a = theta_map_linear(prob_w, ds, None, steps=map_steps, seed=seed, init=a, lin=stats(V, H))
            info["theta"].append(np.asarray(a))
        if log is not None:
            log(f"learn_radial_density: round {round_idx} steps={done}/{int(steps)} obj={obj:.6e} "
                f"reasons={reasons} precond={float(r[1]):.3e} time={time.perf_counter() - t_round:.1f}s")
        if "nonfinite" in reasons or all(x != "steps" for x in (reasons[1:] if fell_back else reasons)):
            break
    info["steps"] = done
    info["theta_final"] = np.asarray(a)
    return V, H, info


def fit_radial_density(prob, ds_fit, ds_val, W0, *, mask, P=1, mode="joint", lam_eta_grid=(0.0,),
                       lam_rough=0.0, lam_spec=0.0, lam_gap=0.0, theta0=None, map_steps=300, log=None,
                       checkpoint=None, H0=None, **learn_kw):
    """Held-out gate over {init, radials_only, density_lam_eta=<l> per l}: the
    radials alone (radial_learn.learn_radial) and the joint radials + density
    (learn_radial_density) from the same start and step budget, each candidate
    scored by radial_learn.fit_radial's one procedure (theta re-MAP on ds_fit
    warm-started at a0, posterior-mean readout, sigma-normalised SSE on ds_val
    with sigma from a0) on its own design width -- so density is kept only when
    it wins on held-out data; ties go to the earlier, simpler candidate.
    H0: optional density warm start for the density candidates only (e.g. a known
    or previously learned eta; masked and normalised by learn_radial_density).
    Returns (W, eta or None, info); info["readout"] is the selected readout,
    [c | d] (len_basis + P * NZ) for a density candidate.  info["readouts"] and
    info["etas"] give every candidate's readout / eta (label -> array or, for
    etas, None on a non-density candidate) and info["cands_W"] every
    candidate's radials, so a caller (bench/learn_radial/run.py) can write a
    model.npz per candidate, not just the selected one."""
    require_x64()
    require_analytic(prob.model)
    require_linear(prob)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if P < 1:
        raise ValueError(f"P must be >= 1 (P = 0 is radial_learn.learn_radial), got {P}")
    for l in lam_eta_grid:
        if l < 0:
            raise ValueError(f"lam_eta must be >= 0, got {l}")
    if learn_kw.pop("learn_sigma_e_mult", 1.0) != 1.0:
        raise ValueError("fit_radial_density: learn_sigma_e_mult is not supported with a density")
    cfg = prob.cfg
    mask = jnp.asarray(mask, jnp.float64)
    W0 = jnp.asarray(W0, jnp.float64)
    n_prior = learn_kw.pop("n_prior", None)
    Q, D2, U, W_init, a0, lin0, r0 = _gate_setup(prob, ds_fit, W0, theta0=theta0, map_steps=map_steps,
                                                 need_U=bool(lam_gap), n_prior=n_prior)
    S = rho_gram(prob.model, W_init, mask, cfg, ds_fit)
    prob_w = prob._replace(gamma=density_gamma(prob.gamma, P, cfg.NZ))
    cands, runs = {"init": (W_init, None)}, {}
    kw = dict(lam_rough=lam_rough, lam_spec=lam_spec, lam_gap=lam_gap, map_steps=map_steps, log=log,
              Q=Q, D2=D2, U=U, **learn_kw)
    W_r, info_r = learn_radial(prob, ds_fit, W0, theta0=from_array(a0), r0=r0, **kw)
    cands["radials_only"], runs["radials_only"] = (W_r, None), info_r
    if checkpoint is not None:
        checkpoint("radials_only", W_r, None, info_r)
    for l in lam_eta_grid:
        key = f"density_lam_eta={float(l):g}"
        if log is not None:
            log(f"fit_radial_density: {key} starting")
        W, eta, info = learn_radial_density(prob, ds_fit, W0, mask=mask, P=P, mode=mode, H0=H0, theta0=from_array(a0),
                                            lam_eta=l, S=S, **kw)
        cands[key], runs[key] = (W, eta), info
        if checkpoint is not None:
            checkpoint(key, W, eta, info)
    theta_fit, map_diag, readouts = {}, {}, {}

    def score(label, cand):
        W, eta = cand
        pw = prob if eta is None else prob_w
        st = ((lambda d: linear_statistics(with_radial(prob.model, W), cfg, d)) if eta is None else
              (lambda d: linear_density_statistics(with_radial(prob.model, W), eta, mask, cfg, d)))
        lin_fit = st(ds_fit)
        a_fit, _, diag = theta_map_linear(pw, ds_fit, None, steps=map_steps, init=a0, return_stats=True,
                                          lin=lin_fit)
        s, c = holdout_score(None, a_fit, a0, pw, ds_fit, ds_val, lin_fit=lin_fit, lin_val=st(ds_val),
                             return_readout=True)
        theta_fit[label], map_diag[label], readouts[label] = np.asarray(a_fit), diag, np.asarray(c)
        if log is not None:
            log(f"fit_radial_density: gate {label} score={s:.6e}")
        return s

    label, scores = gate(cands, score)
    W_sel, eta_sel = cands[label]
    if log is not None:
        log(f"fit_radial_density: selected {label}")
    cands_W = {k: np.asarray(w) for k, (w, _) in cands.items()}
    etas = {k: (None if eta is None else np.asarray(eta)) for k, (_, eta) in cands.items()}
    return W_sel, eta_sel, {"selected": label, "scores": scores, "theta_fit": theta_fit, "map_diag": map_diag,
                            "runs": runs, "theta_init": np.asarray(a0), "readout": readouts[label],
                            "readouts": readouts, "etas": etas, "cands_W": cands_W,
                            "P": 0 if eta_sel is None else int(P), "mask": np.asarray(mask).tolist(),
                            "mode": mode}
