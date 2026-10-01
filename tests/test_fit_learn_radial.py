"""aj fit --learn-radial: learned tensor radials as a fit-pipeline stage
(docs/specs/2026-10-01-fit-learned-radials-design.md)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_ace_model.npz"
QUIET = lambda *a, **k: None
KEYS = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")


def _cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(MODEL), arm="linear", m_per_species=0, rungs=("map",), map_steps=20,
                opt="adam", batch=4, r0=2.35, e0="model", predict_train=False, **KEYS)
    base.update(kw)
    return FitConfig(**base)


def test_config_radial_defaults_and_validation():
    c = _cfg().validate()
    assert (c.learn_radial, c.radial_n_q, c.radial_steps, c.radial_lam_grid, c.radial_val_frac) == \
        (False, 12, 40, (0.0, 1e-2), 0.2)
    for bad in (dict(radial_val_frac=0.0), dict(radial_val_frac=1.0), dict(radial_n_q=0),
                dict(radial_steps=-1), dict(radial_lam_grid=())):
        with pytest.raises(ValueError, match="radial"):
            _cfg(learn_radial=True, **bad).validate()


def _data(cfg):
    from ace_jax.fit.pipeline import load_fit_data
    return load_fit_data(cfg, train=str(XYZ), log=QUIET)


@pytest.fixture
def fast(monkeypatch):
    import ace_jax.fit.pipeline.radials as R
    monkeypatch.setattr(R, "RADIAL_MAP_STEPS", 20)


def test_stage_matches_fit_radial_on_the_same_split(fast):
    """The stage is exactly bench/learn_radial/run.py's recipe on the ARD-style split."""
    import jax.numpy as jnp
    from ace_jax.basis.prior import prior_diagonal
    from ace_jax.fit.data import build_dataset
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem
    from ace_jax.fit.pipeline.radials import learn_radials
    from ace_jax.fit.radial_learn import fit_radial
    from ace_jax.fit.radial_model import rnl_degrees, to_analytic
    cfg = _cfg(learn_radial=True, radial_steps=3, radial_lam_grid=(0.0,), radial_val_frac=0.25)
    d = _data(cfg)
    d2, rr = learn_radials(cfg, d, log=QUIET)
    # reference: the driver's construction on the same split
    idx = np.random.default_rng(cfg.seed).permutation(len(d.train))
    nval = max(1, int(round(0.25 * len(d.train))))
    val, fit_ = [d.train[i] for i in idx[:nval]], [d.train[i] for i in idx[nval:]]
    m, _ = to_analytic(d.model, 12)
    ds_fit, ds_val = (build_dataset(cs, d.meta, d.E0, cfg.batch) for cs in (fit_, val))
    gc = GPConfig(r0=2.35, rcut=float(d.meta["rcut"]), n_B=d.meta["n_B"], n_pair=d.meta["n_pair"],
                  NZ=len(d.meta["elements"]), C=cfg.batch)
    X, S = site_features(m, gc, ds_fit)
    ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
    prob = Problem(KernelSpec("cosine", True, gc.D), m, ind, gc,
                   jnp.asarray(prior_diagonal(d.z, d.meta, d.source)), default_prior(2.35))
    W, info = fit_radial(prob, ds_fit, ds_val, m.rnl_Wnlq, lam_grid=(0.0,), steps=3, map_steps=20,
                         rough_weights=1.0 / (1.0 + rnl_degrees(d.meta)) ** 2)
    assert (rr.n_fit, rr.n_val) == (len(fit_), nval)
    assert rr.info["selected"] == info["selected"]
    np.testing.assert_allclose(rr.W, np.asarray(W), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(d2.model.rnl_Wnlq), np.asarray(W), rtol=0, atol=1e-12)
    assert np.asarray(d2.z["rnl_Wnlq"]).shape == np.asarray(W).shape


def test_stage_gate_keeps_init(fast):
    """radial_steps=0: learned == init, the tie goes to init, nothing is marked learned."""
    from ace_jax.fit.pipeline.radials import learn_radials
    from ace_jax.fit.radial_model import to_analytic
    cfg = _cfg(learn_radial=True, radial_steps=0, radial_lam_grid=(0.0,))
    d = _data(cfg)
    d2, rr = learn_radials(cfg, d, log=QUIET)
    assert rr.info["selected"] == "init"
    assert not d2.model.radial_learned and d2.meta.get("radial_learned") is False
    np.testing.assert_allclose(np.asarray(d2.model.rnl_Wnlq), rr.W, rtol=0, atol=1e-12)
    # init is fit_radial's normalise(W0): the widened radials up to a positive per-radial gauge scale
    W0 = np.asarray(to_analytic(d.model, 12)[0].rnl_Wnlq)
    nrm = lambda W: np.linalg.norm(W, axis=-1, keepdims=True)
    live = nrm(W0)[..., 0] > 0
    np.testing.assert_allclose((rr.W / nrm(rr.W))[live], (W0 / nrm(W0))[live], rtol=0, atol=1e-12)


def test_stage_keeps_fitted_e0(fast):
    """e0='lsq' E0 (set on the model by load_fit_data) survives the reload."""
    from ace_jax.fit.pipeline.radials import learn_radials
    cfg = _cfg(e0="lsq", learn_radial=True, radial_steps=2, radial_lam_grid=(0.0,))
    d = _data(cfg)
    d2, _ = learn_radials(cfg, d, log=QUIET)
    np.testing.assert_allclose(np.asarray(d2.model.E0), d.E0, rtol=0, atol=0)
    np.testing.assert_allclose(d2.E0, d.E0, rtol=0, atol=0)


def test_stage_refuses_factorised_radial():
    import dataclasses
    from ace_jax.fit.pipeline.radials import learn_radials
    cfg = _cfg(learn_radial=True)
    d = _data(cfg)
    d = d._replace(model=dataclasses.replace(d.model, radial_kind="spline_factorised"))
    with pytest.raises(ValueError, match="#31"):
        learn_radials(cfg, d, log=QUIET)


def test_stage_refuses_an_empty_split():
    from ace_jax.fit.pipeline.radials import learn_radials
    cfg = _cfg(learn_radial=True, radial_val_frac=0.999)
    d = _data(cfg)
    with pytest.raises(ValueError, match="radial_val_frac"):
        learn_radials(cfg, d, log=QUIET)
