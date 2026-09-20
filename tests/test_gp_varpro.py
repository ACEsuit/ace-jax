"""VarPro driver (ace_jax.fit.varpro): the outer optimiser learns the nonlinear
FS-density weights on the projected residual, driving the stable inner solve;
it beats the linear-only fit and the learned density transfers frozen to a
disjoint split -- the spike's gate, in ace_jax/JAX."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.fit.varpro import learn, projected_residual, fit_frozen


def _problem(rng, n_cfg=200, sites=6, L=12, n_chan=8, eta_true=None, noise=1e-3):
    """A config-level fit: linear ACE-like block + one sqrt-density column whose
    shape is set by eta_true (weights over per-site density channels)."""
    if eta_true is None:
        eta_true = rng.standard_normal(n_chan)
    seg = np.repeat(np.arange(n_cfg), sites)                 # site -> config
    Phi_lin = rng.standard_normal((n_cfg, L))
    C = rng.standard_normal((sites * n_cfg, n_chan)) + 1.5   # per-site density channels (mostly +)
    ssqrt = lambda z: np.sign(z) * np.sqrt(np.abs(z) + 1e-6)
    dens_true = np.add.reduceat(ssqrt(C @ eta_true), np.arange(0, sites * n_cfg, sites))  # (n_cfg,)
    c_true = rng.standard_normal(L); d_true = 2.0
    y = Phi_lin @ c_true + d_true * dens_true + noise * rng.standard_normal(n_cfg)
    return jnp.asarray(Phi_lin), jnp.asarray(C), seg, jnp.asarray(y), eta_true


def _design_fn(Phi_lin, C, seg, y, n_cfg, sites):
    """design_fn(eta) -> (Phi(eta), y): [linear | summed sqrt-density column]."""
    def f(eta):
        u = jnp.sign(C @ eta) * jnp.sqrt(jnp.abs(C @ eta) + 1e-6)     # (n_sites,)
        col = jax.ops.segment_sum(u, jnp.asarray(seg), num_segments=n_cfg)   # (n_cfg,)
        return jnp.concatenate([Phi_lin, col[:, None]], 1), y
    return f


def test_varpro_learns_density_beats_linear_and_transfers():
    rng = np.random.default_rng(0)
    n_cfg, sites, L, n_chan = 200, 6, 12, 8
    eta_true = rng.standard_normal(n_chan)
    Lam = 1e-3 * jnp.eye(L + 1)                              # light ridge on [c ; d]

    # --- split A: learn eta ---
    Phi, C, seg, y, _ = _problem(rng, n_cfg, sites, L, n_chan, eta_true)
    dfn = _design_fn(Phi, C, seg, y, n_cfg, sites)
    eta0 = jnp.asarray(rng.standard_normal(n_chan))
    lin_only = float(projected_residual(jnp.zeros(n_chan), dfn, Lam))   # dead density column
    start = float(projected_residual(eta0, dfn, Lam))
    eta_star, res = learn(eta0, dfn, Lam, maxiter=300)
    end = float(projected_residual(eta_star, dfn, Lam))
    true = float(projected_residual(jnp.asarray(eta_true), dfn, Lam))

    assert end < start                                       # the outer opt reduced the projected residual
    assert end < 0.2 * lin_only                             # and far below the linear-only fit
    assert end <= 1.5 * true                                # reaching ~ the true-density loss
    # eta identifiable up to sign/scale (sqrt + coefficient absorb it): check direction
    cos = abs(float(eta_star @ eta_true) / (jnp.linalg.norm(eta_star) * np.linalg.norm(eta_true)))
    assert cos > 0.9

    # --- split B (disjoint): freeze eta_star, convex fit; must beat linear-only ---
    PhiB, CB, segB, yB, _ = _problem(np.random.default_rng(1), n_cfg, sites, L, n_chan, eta_true)
    dfnB = _design_fn(PhiB, CB, segB, yB, n_cfg, sites)
    _, r_frozen = fit_frozen(eta_star, dfnB, Lam)
    r_linB = float(projected_residual(jnp.zeros(n_chan), dfnB, Lam))
    assert r_frozen < 0.2 * r_linB                          # the learned density transfers
