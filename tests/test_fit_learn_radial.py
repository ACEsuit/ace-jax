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
    """radial_steps=0: learned == init, the tie goes to init, nothing is marked learned.
    With e0='lsq' (written into the model by load_fit_data) the fitted E0 survives the
    reload from the patched arrays (the reload does not depend on the step count)."""
    from ace_jax.fit.pipeline.radials import learn_radials
    from ace_jax.fit.radial_model import to_analytic
    cfg = _cfg(e0="lsq", learn_radial=True, radial_steps=0, radial_lam_grid=(0.0,))
    d = _data(cfg)
    d2, rr = learn_radials(cfg, d, log=QUIET)
    np.testing.assert_allclose(np.asarray(d2.model.E0), d.E0, rtol=0, atol=0)
    np.testing.assert_allclose(d2.E0, d.E0, rtol=0, atol=0)
    assert rr.info["selected"] == "init"
    assert not d2.model.radial_learned and d2.meta.get("radial_learned") is False
    np.testing.assert_allclose(np.asarray(d2.model.rnl_Wnlq), rr.W, rtol=0, atol=1e-12)
    # init is fit_radial's normalise(W0): the widened radials up to a positive per-radial gauge scale
    W0 = np.asarray(to_analytic(d.model, 12)[0].rnl_Wnlq)
    nrm = lambda W: np.linalg.norm(W, axis=-1, keepdims=True)
    live = nrm(W0)[..., 0] > 0
    np.testing.assert_allclose((rr.W / nrm(rr.W))[live], (W0 / nrm(W0))[live], rtol=0, atol=1e-12)


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


def test_fit_learn_radial_end_to_end(fast, tmp_path):
    """fit() runs the stage, refits on the full training set, saves the learned
    model (marked when learned) and radial_info.json."""
    import json
    from ace_jax import ACECalculator
    from ace_jax.fit.pipeline import fit, write_outputs
    cfg = _cfg(learn_radial=True, radial_steps=4, radial_lam_grid=(0.0,))
    d = _data(cfg)
    staged = {}
    res = fit(cfg, d, log=QUIET, on_stage=lambda k, v: staged.__setitem__(k, v))
    assert res.radial is staged["radial"]
    assert len(res.data.ds_train.y_E.reshape(-1)) >= len(d.train)            # full train, not the fit split
    write_outputs(res, tmp_path, layout=("cli",), log=QUIET)
    info = json.loads((tmp_path / "radial_info.json").read_text())
    assert set(info) >= {"selected", "scores", "n_q", "lam_grid", "val_frac", "to_analytic_relres_max",
                         "n_fit", "n_val", "seconds"}
    z = np.load(tmp_path / "model.npz")
    np.testing.assert_allclose(z["rnl_Wnlq"], res.radial.W, rtol=0, atol=1e-12)
    assert info["selected"] != "init"            # a recoverable case: the learned candidate wins the gate
    assert json.loads(bytes(z["meta_json"]).decode())["radial_learned"] is True
    assert ACECalculator(str(tmp_path / "model.npz"), lean=True).splined


@pytest.mark.slow                    # ~1 min each; the CI slow job runs on every PR
def test_fit_learn_radial_on_built_basis(fast, tmp_path, monkeypatch):
    from test_basis_build import _primed_cache
    from ace_jax.basis.model import BasisSpec
    from ace_jax.fit.pipeline import fit
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cfg = _cfg(model=BasisSpec(order=3, max_degree=10, coupling_cache_dir=_primed_cache(tmp_path)),
               r0=None, e0="lsq", learn_radial=True, radial_steps=2, radial_lam_grid=(0.0,))
    res = fit(cfg, _data(cfg), log=QUIET)
    assert res.radial is not None and res.radial.n_val >= 1


@pytest.mark.slow                    # ~1 min each; the CI slow job runs on every PR
def test_fit_learn_radial_then_gp(fast, tmp_path):
    """The radial stage is linear; a GP final fit after it still runs and saves."""
    from ace_jax.fit.pipeline import fit, write_outputs
    cfg = _cfg(arm="gp", m_per_species=2, map_steps=5, learn_radial=True, radial_steps=2,
               radial_lam_grid=(0.0,))
    res = fit(cfg, _data(cfg), log=QUIET)
    write_outputs(res, tmp_path, layout=("cli",), log=QUIET)
    z = np.load(tmp_path / "gp_model.npz")
    np.testing.assert_allclose(z["ace/rnl_Wnlq"], res.radial.W, rtol=0, atol=1e-12)


FAST_CLI = ["--model", str(MODEL), "--r0", "2.35", "--train", str(XYZ), "--energy-key", "dft_energy",
            "--force-key", "dft_force", "--virial-key", "dft_virial", "--m-per-species", "0",
            "--rungs", "map", "--map-steps", "20", "--opt", "adam", "--configs-per-batch", "4"]


@pytest.mark.slow                    # two full CLI runs (~80 s)
def test_cli_learn_radial_writes_info_and_reproduces(fast, tmp_path):
    import json
    import yaml
    from ace_jax.cli import main
    a = tmp_path / "a"
    main(["fit", *FAST_CLI, "--learn-radial", "--radial-steps", "3", "--radial-lam-grid", "0",
          "--out", str(a)])
    info = json.loads((a / "radial_info.json").read_text())
    assert info["lam_grid"] == [0.0] and info["n_q"] == 12
    y = yaml.safe_load((a / "fit.yaml").read_text())
    assert y["learn_radial"] is True and y["radial_steps"] == 3 and y["radial_lam_grid"] in ("0", [0.0], "0.0")
    b = tmp_path / "b"
    main(["fit", "--config", str(a / "fit.yaml"), "--out", str(b)])
    np.testing.assert_allclose(np.load(b / "model.npz")["rnl_Wnlq"], np.load(a / "model.npz")["rnl_Wnlq"],
                               rtol=0, atol=1e-10)   # two separate runs


def test_cli_radial_option_needs_learn_radial(capsys):
    from ace_jax.cli import _parse
    with pytest.raises(SystemExit):
        _parse(["fit", *FAST_CLI, "--radial-steps", "3", "--out", "o"])
    assert "--radial-steps needs --learn-radial" in capsys.readouterr().err


def test_yaml_radial_defaults_without_learning_are_fine(tmp_path):
    """A resolved fit.yaml records every dest, radial defaults included; re-running
    it with learn_radial false must not trip the stray-option check."""
    import yaml
    from ace_jax.cli import _parse
    f = tmp_path / "fit.yaml"
    f.write_text(yaml.safe_dump({"model": str(MODEL), "r0": 2.35, "train": str(XYZ), "out": "o",
                                 "learn_radial": False, "radial_steps": 40, "radial_n_q": 12,
                                 "radial_lam_grid": "0,0.01", "radial_val_frac": 0.2}))
    a = _parse(["fit", "--config", str(f)])
    assert a.learn_radial is False and a.radial_steps == 40


def test_cli_lam_grid_parses_to_floats():
    from ace_jax.cli import _fit_config, _parse
    a = _parse(["fit", *FAST_CLI, "--learn-radial", "--radial-lam-grid", "0,1e-3,1e-2", "--out", "o"])
    assert _fit_config(a).radial_lam_grid == (0.0, 1e-3, 1e-2)


def test_config_refuses_learn_radial_with_a_baseline():
    """A baseline fit saves no model file, so the learned radials would be lost."""
    for kw in (dict(baseline="dimer_mean.npz"), dict(base_npz="mu0.npz")):
        with pytest.raises(ValueError, match="learn_radial.*baseline"):
            _cfg(learn_radial=True, **kw).validate()
