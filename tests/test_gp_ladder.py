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


def test_run_map_ps_matches_flat_run_map(tiny_linear_problem):
    from ace_jax.fit.paramset import from_hypers
    from ace_jax.fit.ladder import run_map, run_map_ps
    from ace_jax.fit.objective import make_lml
    prob, ds = tiny_linear_problem
    lml = make_lml(prob, ds)
    flat = run_map(lml, prob.prior, steps=20, seed=0)          # existing path
    ps = from_hypers(prob.prior.mu, prob.prior)                # all-LML default
    ps_out = run_map_ps(ps, prob, ds, steps=20, seed=0)
    assert jnp.allclose(ps_out.lml_vector(), to_array(flat), atol=1e-6)  # same optimum


def test_laplace_fd_matches_exact_width():
    from ace_jax.fit.ladder import run_laplace_fd
    h = run_map(lml, PRIOR, steps=2000, lr=0.05)
    draws, info = run_laplace_fd(lml, PRIOR, h, n_draws=4000)
    assert draws.shape == (4000, 10) and info["n_floored"] == 0
    assert np.abs(draws.mean(0) - np.asarray(STAR)).max() < 0.02
    assert np.abs(np.array(info["std"]) - WIDTH).max() < 1e-3      # quadratic target: FD Hessian is exact


import pytest


def _tiny_embed_problem(mps=4, ncfg=3):
    """SiGe (unfitted) problem with the species kernel factor live."""
    import equinox as eqx
    from conftest import FIXTURE_DIR
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, build_pmap, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem
    model, meta, z = load(FIXTURE_DIR / "sige_nofit.npz")
    els = [int(e) for e in meta["elements"]]
    cfgs = load_configs(FIXTURE_DIR / "si_tiny_train.xyz", "dft_energy", "dft_force", "dft_virial")[:ncfg]
    counts = np.array([[np.sum(c.numbers == e) for e in els] for c in cfgs], float)
    E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in cfgs]), rcond=None)
    model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
    ds = build_dataset(cfgs, meta, E0, ncfg)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(els), C=ncfg)
    X, S = site_features(model, cfg, ds); scale = descriptor_scale(X, ds.node_mask)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, mps, scale,
                          Pmap=build_pmap(cfg, scale, None), nz=len(els))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.ones(cfg.len_basis), default_prior(2.35))   # unfitted model: no exported gamma
    return prob, ds, els


@pytest.mark.slow
def test_run_map_ps_normalizes_embed_for_inner_map():
    # An embed-carrying ParamSet stores the RAW E; the kernel's coregionalization
    # needs normalize_rows(E).  run_map_ps must normalise centrally, so a raw and
    # a pre-normalised (frozen) E give the same inner theta-MAP.
    from ace_jax.eval import highest_precision
    from ace_jax.fit.paramset import from_hypers
    from ace_jax.fit.ladder import run_map_ps
    from ace_jax.fit.embedding import normalize_rows
    with highest_precision():
        prob, ds, els = _tiny_embed_problem()
        NZ = len(els)
        rng = np.random.default_rng(0)
        E = jnp.asarray(np.eye(NZ) + 0.1 * rng.normal(size=(NZ, NZ)))     # rows NOT unit-norm
        assert not np.allclose(np.asarray(E), np.asarray(normalize_rows(E)))   # E is genuinely raw
        ps = from_hypers(prob.prior.mu, prob.prior, embed=E, embed_route="fixed")
        a_ps = run_map_ps(ps, prob, ds, steps=50, seed=0).block("hypers").value
        ps_n = from_hypers(prob.prior.mu, prob.prior, embed=normalize_rows(E), embed_route="fixed")
        a_norm = run_map_ps(ps_n, prob, ds, steps=50, seed=0).block("hypers").value
        assert np.allclose(np.asarray(a_ps), np.asarray(a_norm), atol=1e-10)
