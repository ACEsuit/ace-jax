"""Joint VarPro learning of the tensor radials W and P sqrt-density features.

The site energy is E_i = c_z . X_i + sum_p d[p, z] ssqrt(eta[p, z] . (mask * X_i))
(eval.fs_model).  At fixed (W, eta) it is linear in [c | d], so the readout is
projected out exactly as in radial_learn, now on the widened statistics
fit.density.linear_density_statistics; the outer variables are the radial V
and the raw density weights H, with eta = normalise_rho(H, S) fixing each
density's scale gauge (unit mean-square rho on the training sites, S the
per-species second moment at the initial radials -- the analogue of the radial
gauge Gram Q).  d absorbs the scale and the sign of each density.

The optimiser is L-BFGS on u = [vec(V) ; r_eta vec(H)] ("joint"), r_eta the
ratio of the blocks' RMS gradients recomputed every round (one extra gradient
evaluation), or alternating V / H blocks ("alternating", the fallback), both
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
from .radial_learn import (lbfgs_loop, projected_residual_from_stats, relative_lambda, relative_lambda_gap,
                           relative_lambda_spec, require_linear, require_x64, theta_map_linear)
from .radial_model import (data_r_range, gap_penalty, normalise, radial_gram, require_analytic, roughness,
                           roughness_matrix, row_active, spectral_penalty, spectral_weights, uniform_gram,
                           with_radial)

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


def relative_lambda_eta(lam_eta, r0, n):
    """Absolute density-prior weight lam_eta * r0 / n, n = P * NZ densities."""
    if lam_eta < 0:
        raise ValueError(f"lam_eta must be >= 0, got {lam_eta}")
    return float(lam_eta) * r0 / n if lam_eta else 0.0


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


_grad_rd = jax.jit(jax.grad(_objective_rd), static_argnums=(21, 22, 23))


def _block_scale(g, nV):
    """r_eta = RMS(grad_H) / RMS(grad_V), so both blocks' gradients in u have the same
    RMS; 1 when either is zero or non-finite."""
    gV, gH = np.asarray(g[:nV]), np.asarray(g[nV:])
    rv, rh = float(np.sqrt(np.mean(gV ** 2))), float(np.sqrt(np.mean(gH ** 2)))
    ok = np.isfinite(rv) and np.isfinite(rh) and rv > 0 and rh > 0
    return rh / rv if ok else 1.0


def learn_radial_density(prob, ds, W0, *, mask, P=1, H0=None, mode="joint", theta0=None, profile=True,
                         lam_rough=0.0, rough_weights=None, lam_spec=0.0, spec_p=4.0, lam_gap=0.0,
                         lam_eta=0.0, steps=40, reprofile_every=10, tol=1e-6, patience=3, map_steps=300,
                         seed=0, log=None, Q=None, D2=None, U=None, S=None, r0=None):
    """VarPro-learn radials W and P density weights eta jointly (M = 0).  Same
    round structure, relative priors and stopping rules as radial_learn.learn_radial;
    lam_eta is relative too (relative_lambda_eta).  mode "joint" optimises
    [V; H] together, "alternating" spends each round's steps half on V (H
    fixed) then half on H.  Q, D2, U as learn_radial; S = rho_gram at the
    normalised init (computed when None); r0 = the widened projected residual at
    the start (computed when None; only valid with theta0).  Returns
    (V, eta, info), both normalised; steps = 0 returns the normalised init."""
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
    H = normalise_rho(init_density(cfg, mask, P, seed) if H0 is None else jnp.asarray(H0, jnp.float64), S)
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
    lam_eta_abs = relative_lambda_eta(lam_eta, r0, P * cfg.NZ)
    info = {"trace": [], "reasons": [], "round_lengths": [], "theta": [np.asarray(a)], "precond": [],
            "r0": r0, "lam_abs": lam, "lam_spec_abs": lam_spec_abs, "lam_gap_abs": lam_gap_abs,
            "lam_eta": float(lam_eta), "lam_eta_abs": lam_eta_abs, "P": int(P), "mode": mode, "steps": 0}
    if log is not None:
        log(f"learn_radial_density: P={P} mode={mode} lam_eta={float(lam_eta):g} "
            f"lam_eta_abs={lam_eta_abs:.6e} r0={r0:.6e}")
    f64 = lambda v: jnp.asarray(v, jnp.float64)
    shapes = (tuple(V.shape), tuple(H.shape))
    one = f64([1.0, 1.0])
    done, round_idx = 0, 0
    while done < int(steps):
        round_idx += 1
        n = min(int(reprofile_every), int(steps) - done)
        t_round = time.perf_counter()
        common = (a, prob.model, ds, prob_w.gamma, Q, active, D2, wn, f64(lam), W_ref, sw, f64(lam_spec_abs),
                  U, f64(lam_gap_abs), S, mask, gc, f64(lam_eta_abs))
        g = _grad_rd(*_pack(V, H, one, "joint"), *common, one, cfg, "joint", shapes)
        r = f64([1.0, _block_scale(g, V.size)])
        info["precond"].append(float(r[1]))
        blocks = [("joint", n)] if mode == "joint" else [("V", (n + 1) // 2), ("H", n // 2)]
        reasons, obj = [], float("nan")
        for block, nb in blocks:
            if nb == 0:
                continue
            xb, other = _pack(V, H, r, block)
            xb, obj, trace, reason = lbfgs_loop(_objective_rd, xb, steps=nb, tol=tol, patience=patience,
                                                args=(other, *common, r), statics=(cfg, block, shapes))
            V, H = _unpack(xb, other, r, block, shapes)
            info["trace"].extend(trace)
            info["round_lengths"].append(len(trace))
            reasons.append(reason)
        info["reasons"].append(reasons[0] if len(reasons) == 1 else "+".join(reasons))
        done += n
        V, H = normalise(V, Q, active), normalise_rho(H, S)
        if profile:
            a = theta_map_linear(prob_w, ds, None, steps=map_steps, seed=seed, init=a, lin=stats(V, H))
            info["theta"].append(np.asarray(a))
        if log is not None:
            log(f"learn_radial_density: round {round_idx} steps={done}/{int(steps)} obj={obj:.6e} "
                f"reasons={reasons} precond={float(r[1]):.3e} time={time.perf_counter() - t_round:.1f}s")
        if "nonfinite" in reasons or all(x != "steps" for x in reasons):
            break
    info["steps"] = done
    info["theta_final"] = np.asarray(a)
    return V, H, info
