"""Plumbing test on a synthetic Gaussian target: every rung must recover a
known mode and width.  No ACE here; the ACE smoke test is in test_gp_cli.py."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.fit.hypers import Hypers, Prior, to_array
from ace_jax.fit.ladder import run_laplace, run_map, run_nuts, run_vi

STAR = jnp.linspace(-1.0, 1.0, 10)
WIDTH = 0.2


def lml(a):
    return -0.5 * jnp.sum(((a - STAR) / WIDTH) ** 2)


PRIOR = Prior(Hypers(*[0.0] * 10), Hypers(*[10.0] * 10))   # essentially flat


def test_map_finds_mode():
    h = run_map(lml, PRIOR, steps=2000, lr=0.05)
    assert np.abs(np.asarray(to_array(h)) - np.asarray(STAR)).max() < 1e-2


def test_laplace_width():
    draws, h = run_laplace(lml, PRIOR, n_draws=4000, steps=2000, lr=0.05)
    assert draws.shape == (4000, 10)
    assert np.abs(draws.mean(0) - np.asarray(STAR)).max() < 0.02
    assert np.abs(draws.std(0) - WIDTH).max() < 0.02


def test_vi_width():
    draws, _ = run_vi(lml, PRIOR, n_draws=4000, steps=3000, lr=0.02)
    assert np.abs(draws.mean(0) - np.asarray(STAR)).max() < 0.05
    assert np.abs(draws.std(0) - WIDTH).max() < 0.05


def test_nuts_recovers_target():
    draws, summary = run_nuts(lml, PRIOR, num_warmup=300, num_samples=500, num_chains=2)
    assert draws.shape == (1000, 10)
    assert np.abs(draws.mean(0) - np.asarray(STAR)).max() < 0.05
    assert np.abs(draws.std(0) - WIDTH).max() < 0.05
    assert max(summary["r_hat"].values()) < 1.05


def test_laplace_fd_matches_exact_width():
    from ace_jax.fit.ladder import run_laplace_fd
    h = run_map(lml, PRIOR, steps=2000, lr=0.05)
    draws, info = run_laplace_fd(lml, PRIOR, h, n_draws=4000)
    assert draws.shape == (4000, 10) and info["n_floored"] == 0
    assert np.abs(draws.mean(0) - np.asarray(STAR)).max() < 0.02
    assert np.abs(np.array(info["std"]) - WIDTH).max() < 1e-3      # quadratic target: FD Hessian is exact
