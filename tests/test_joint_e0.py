"""e0='lsq' fits the reference energies jointly with the readout: one column per species
(its atom count on the energy rows, zero on forces and virials) with a fixed wide prior
around the pre-fit E0, folded back into E0 on export."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR  # noqa: E402

from ace_jax.eval import load  # noqa: E402
from ace_jax.fit.data import build_dataset, load_configs  # noqa: E402
from ace_jax.fit.hypers import Hypers, default_prior  # noqa: E402
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features  # noqa: E402
from ace_jax.fit.kernels import KernelSpec  # noqa: E402
from ace_jax.fit.objective import Problem, prior_precision  # noqa: E402
from ace_jax.fit.rows import linear_rows  # noqa: E402

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
KEYS = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")


def _theta(**kw):
    d = dict(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=0.0, log_r0=np.log(2.35),
             log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
             log_sigma_E=np.log(1e-3), log_sigma_F=np.log(2e-2), log_sigma_V=np.log(2e-2))
    d.update(kw)
    return Hypers(**d)


@pytest.fixture(scope="module")
def m0():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(str(XYZ), **KEYS)[:12]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 4)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=4)
    return model, meta, z, configs, ds, cfg


def test_e0_columns_hold_the_species_counts_on_energy_rows_only(m0):
    model, meta, z, configs, ds, cfg = m0
    cfg_e0 = dataclasses.replace(cfg, e0_cols=True)
    assert cfg_e0.len_basis == cfg.len_basis + cfg.NZ and cfg_e0.D == cfg.D
    b = jax.tree.map(lambda a: a[0], ds)
    r0, r1 = linear_rows(model, cfg, b)[0], linear_rows(model, cfg_e0, b)[0]
    L0 = cfg.len_basis
    np.testing.assert_array_equal(np.asarray(r1.E[:, :L0]), np.asarray(r0.E))
    counts = np.array([[np.sum((np.asarray(b.node_z) == s) & (np.asarray(b.node_cfg) == c) & np.asarray(b.node_mask))
                        for s in range(cfg.NZ)] for c in range(r1.E.shape[0])], float)
    np.testing.assert_array_equal(np.asarray(r1.E[:, L0:]), counts)
    assert not np.any(np.asarray(r1.F[..., L0:])) and not np.any(np.asarray(r1.V[..., L0:]))


def test_e0_columns_take_a_fixed_prior_independent_of_sigma_c(m0):
    model, meta, z, configs, ds, cfg = m0
    cfg_e0 = dataclasses.replace(cfg, e0_cols=True)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg_e0, jnp.asarray(z["gamma"]),
                   default_prior(2.35), e0_prec=jnp.full(cfg.NZ, 1.0))
    for sc in (0.3, 30.0):
        Lam, logdet = prior_precision(_theta(log_sigma_c=np.log(sc)), prob)
        d = np.diag(np.asarray(Lam))
        assert d.shape == (cfg_e0.len_basis,)
        np.testing.assert_allclose(d[:cfg.len_basis], np.asarray(z["gamma"]) ** 2 / sc ** 2)
        np.testing.assert_array_equal(d[cfg.len_basis:], 1.0)
        np.testing.assert_allclose(float(logdet), np.sum(np.log(d)))


def _fit(e0, data=None, **kw):
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), ntrain=24, ntest=12, batch=4, r0=2.35, arm="linear",
                    m_per_species=0, rungs=("map",), opt="lbfgs", map_steps=30, e0=e0, predict_train=True,
                    predict_stats="recompute", **KEYS, **kw)
    d = load_fit_data(cfg, data=data or str(XYZ), log=lambda *a: None)
    return cfg, d, fit(cfg, d, log=lambda *a: None)


@pytest.fixture(scope="module")
def joint():
    return _fit("lsq")


def test_joint_e0_is_exported_and_reproduced_by_the_calculator(joint, tmp_path):
    from ase import Atoms
    from ace_jax import ACECalculator
    from ace_jax.fit.pipeline.export import linear_model_arrays
    cfg, d, res = joint
    arr = linear_model_arrays(res)
    L0 = res.built.prob.cfg.len_basis - res.built.prob.cfg.NZ
    assert arr["WB"].size + arr["Wpair"].size == L0                        # the E0 columns are not readout
    assert not np.allclose(arr["E0"], d.E0)                                 # the fit moved E0 off the pre-fit
    p = tmp_path / "model.npz"; np.savez(p, **arr)
    calc = ACECalculator(str(p), skin=0)
    pred = res.preds.arrays["test/map"]["E_mean"]
    for c, e in zip(d.test_o[:4], pred):
        a = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc); a.calc = calc
        assert abs(a.get_potential_energy() - e) < 1e-6 * max(1.0, abs(e))


def test_joint_e0_never_loses_to_the_prefit_on_training_energies(joint):
    _, _, res = joint
    _, _, ref = _fit("prefit")
    assert res.preds.metrics["train/map"]["E"]["rmse"] <= ref.preds.metrics["train/map"]["E"]["rmse"] * 1.05


def test_joint_e0_absorbs_a_constant_energy_shift(tmp_path):
    # shift every training energy by 2 eV/atom: the pre-fit E0 absorbs it, and so must the joint fit,
    # with the readout unchanged
    from ase.io import read, write
    frames = read(str(XYZ), ":")
    for a in frames:
        a.info["dft_energy"] = a.info["dft_energy"] + 2.0 * len(a)
    p = tmp_path / "shifted.xyz"; write(str(p), frames)
    _, _, res = _fit("lsq", data=str(p))
    _, _, ref = _fit("lsq")
    from ace_jax.fit.pipeline.export import linear_model_arrays
    a, b = linear_model_arrays(res), linear_model_arrays(ref)
    np.testing.assert_allclose(a["E0"] - b["E0"], 2.0, atol=1e-4)    # the rewrite rounds energies
    pa, pb = res.preds.arrays["test/map"], ref.preds.arrays["test/map"]      # same predictions, shifted
    np.testing.assert_allclose((pa["E_mean"] - pb["E_mean"]) / pa["nat"], 2.0, atol=1e-4)


def test_pops_keeps_the_prefit_e0_and_ard_fits_it_jointly():
    from ace_jax.fit.pipeline import FitConfig as _F
    assert _F(model="m.npz", e0="lsq", arm="linear", learn_radial=True).validate().joint_e0 is True
    from ace_jax.fit.pipeline import FitConfig
    assert FitConfig(model="m.npz", e0="lsq", uq="pops", arm="linear").validate().joint_e0 is False
    assert FitConfig(model="m.npz", e0="lsq", uq="ard", arm="linear").validate().joint_e0 is True
    assert FitConfig(model="m.npz", e0="prefit", uq="ard", arm="linear").validate().joint_e0 is False
    assert FitConfig(model="m.npz", e0="lsq", arm="linear").validate().joint_e0 is True
    assert FitConfig(model="m.npz", e0="prefit", arm="linear").validate().joint_e0 is False


def test_gp_arm_exports_the_joint_e0_and_the_calculator_reproduces_it(tmp_path):
    from ase import Atoms
    from ace_jax import GPCalculator
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, save_model
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), ntrain=16, ntest=6, batch=4, r0=2.35, arm="gp",
                    m_per_species=4, rungs=("map",), opt="lbfgs", map_steps=15, e0="lsq", predict_train=False,
                    predict_stats="recompute", **KEYS)
    d = load_fit_data(cfg, data=str(XYZ), log=lambda *a: None)
    res = fit(cfg, d, log=lambda *a: None)
    assert res.built.prob.cfg.e0_cols
    calc = GPCalculator.from_file(str(save_model(res, tmp_path)))
    pred = res.preds.arrays["test/map"]
    for c, e in zip(d.test_o[:3], pred["E_mean"]):
        a = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc); a.calc = calc
        assert abs(a.get_potential_energy() - e) < 1e-6 * max(1.0, abs(e))


def test_an_isolated_atom_in_training_still_fixes_its_species_e0(tmp_path):
    # an isolated atom is predicted as E0 alone, so its energy defines E0 exactly (as pre-fit E0 did)
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    from ace_jax.fit.pipeline.export import linear_model_arrays
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), batch=4, r0=2.35, arm="linear", m_per_species=0,
                    rungs=("map",), opt="lbfgs", map_steps=30, e0="lsq", predict_train=False,
                    predict_stats="recompute", **KEYS)
    cfgs = load_configs(str(XYZ), **KEYS)
    iso = [c.energy for c in cfgs if len(c.numbers) == 1]
    assert iso, "fixture has an isolated atom"
    d = load_fit_data(cfg, train=str(XYZ), log=lambda *a: None)
    E0 = linear_model_arrays(fit(cfg, d, log=lambda *a: None))["E0"]
    assert abs(E0[0] - np.mean(iso)) < 1e-6


def test_patch_radial_npz_accepts_a_readout_with_joint_e0_columns(tmp_path):
    from ace_jax.basis.export import patch_radial_npz
    from ace_jax.fit.radial_model import to_analytic
    src = FIXTURE_DIR / "si_fitted.npz"
    model, meta, z = load(src)
    m, _ = to_analytic(model, 8)
    L0 = (meta["n_B"] + meta["n_pair"]) * len(meta["elements"])
    r = np.arange(L0 + len(meta["elements"]), dtype=float)
    patch_radial_npz(str(src), tmp_path / "m.npz", m, readout=r)
    out = np.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(out["E0"], z["E0"])
    assert out["WB"].size + out["Wpair"].size == L0


def test_gp_json_omits_e0_cols_unless_set():
    # older ace-jax reads gp_json with GPConfig(**gpcfg): an unknown key would break it
    from ace_jax.fit.pipeline.export import _gpcfg_json
    cfg = GPConfig(r0=2.35, rcut=5.0, n_B=3, n_pair=2, NZ=1, C=4)
    assert "e0_cols" not in _gpcfg_json(cfg) and _gpcfg_json(dataclasses.replace(cfg, e0_cols=True))["e0_cols"]


def test_run_config_records_the_fitted_e0(joint, tmp_path):
    import json
    from ace_jax.fit.pipeline import write_outputs
    from ace_jax.fit.pipeline.export import linear_model_arrays
    cfg, d, res = joint
    write_outputs(res, tmp_path, layout=("run",), save_model=False, log=lambda *a: None)
    c = json.loads((tmp_path / "config.json").read_text())
    assert abs(c["E0"]["14"] - linear_model_arrays(res)["E0"][0]) < 1e-9
    assert c["len_basis"] == res.built.prob.cfg.len_readout


def test_patch_radial_npz_warns_when_it_drops_an_e0_shift(tmp_path):
    from ace_jax.basis.export import patch_radial_npz
    from ace_jax.fit.radial_model import to_analytic
    src = FIXTURE_DIR / "si_fitted.npz"
    model, meta, z = load(src)
    m, _ = to_analytic(model, 8)
    L0 = (meta["n_B"] + meta["n_pair"]) * len(meta["elements"])
    with pytest.warns(UserWarning, match="E0"):
        patch_radial_npz(str(src), tmp_path / "m.npz", m, readout=np.ones(L0 + 1))


def _ard_e0_problem(m0, e0_std=(0.7,)):
    model, meta, z, configs, ds, cfg = m0
    cfg_e0 = dataclasses.replace(cfg, e0_cols=True)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    return Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg_e0, jnp.asarray(z["gamma"]),
                   default_prior(2.35), e0_prec=jnp.asarray(e0_std, float) ** -2)


@pytest.mark.parametrize("mode", ["joint", "sequential"])
def test_ard_evidence_gives_the_e0_columns_their_fixed_prior(m0, mode):
    """Joint E0 under ARD: the E0 columns are in the design with BLR's fixed precision e0_prec, outside
    the body-order groups (no a_k).  Brute force: the dense evidence and posterior mean with
    Lambda = diag(gamma^2 exp(a_body(j)), e0_prec)."""
    from ace_jax.fit.ard import (E0_GROUP, ARDEvidence, ard_gamma, ard_posterior, ard_statistics,
                                 body_order_columns)
    from ace_jax.fit.rows import linear_rows
    model, meta, z, configs, ds, cfg = m0
    prob = _ard_e0_problem(m0)
    L = prob.cfg.len_basis
    bc = body_order_columns(meta, prob.cfg)
    assert len(bc) == L and np.all(bc[prob.cfg.len_readout:] == E0_GROUP)
    np.testing.assert_array_equal(bc[:prob.cfg.len_readout], body_order_columns(meta, cfg))
    with pytest.raises(ValueError, match="ard_gamma"):
        ARDEvidence(ard_statistics(_theta(), prob, ds, mode), np.asarray(prob.gamma), bc)
    from ace_jax.eval import highest_precision
    with highest_precision():
        ev = ARDEvidence(ard_statistics(_theta(), prob, ds, mode), ard_gamma(prob), bc)
        assert E0_GROUP not in ev.groups and len(ev.groups) == len(np.unique(body_order_columns(meta, cfg)))
        h = ev.h0(_theta()) + 0.3 * np.arange(len(ev.h0(_theta())))
        v = ev.value_and_grad(h)[0]
        post = ard_posterior(ev, h, 1.0, meta)
        # dense reference
        sig = ev.sigmas(h)
        P, y = [], []
        for i in range(ds.n_batches):
            b = jax.tree.map(lambda a, i=i: a[i], ds)
            r = linear_rows(model, prob.cfg, b)[0]
            for k, (Phi, yy, w) in enumerate(((r.E, b.y_E, b.w_E), (r.F.reshape(-1, L), b.y_F.reshape(-1),
                                                                     jnp.repeat(b.w_F, 3)),
                                               (r.V.reshape(-1, L), b.y_V.reshape(-1), jnp.repeat(b.w_V, 6)))):
                P.append(np.asarray(Phi) * (np.asarray(w) / sig[k])[:, None]); y.append(np.asarray(yy) * np.asarray(w) / sig[k])
        P, y = np.concatenate(P), np.concatenate(y)
    nls = 3 if ev.joint else 0
    a = dict(zip(ev.groups, h[nls:]))
    gam = np.asarray(prob.gamma)
    lam = np.r_[gam ** 2 * np.exp([a[int(g)] for g in bc[:prob.cfg.len_readout]]), np.asarray(prob.e0_prec)]
    A = P.T @ P + np.diag(lam)
    mu = np.linalg.solve(A, P.T @ y)
    np.testing.assert_allclose(post.mean, mu, rtol=0, atol=1e-7 * np.abs(mu).max())
    assert post.n_e0 == cfg.NZ and post.meta["e0_cols"] == cfg.NZ
    if ev.joint:
        n = len(y)    # Gaussian evidence up to the constant -n/2 log 2 pi
        ref = (-0.5 * y @ y + 0.5 * (P.T @ y) @ mu - 0.5 * np.linalg.slogdet(A)[1] + 0.5 * np.sum(np.log(lam))
               - np.sum(np.asarray(ev._data[3]) * h[:3]))
        assert v == pytest.approx(ref, abs=1e-6 * max(1.0, abs(ref))), n


def test_ard_force_quantities_ignore_the_e0_columns(m0):
    """The E0 columns are zero on every force row: force rows WITHOUT them (the model file's, as
    ACECalculator builds them) give the same shape V, forces_std and force variance as the full rows,
    on the R (PRESS), Q (sandwich) and kappa (Cholesky) paths."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.ard import ARDEvidence, ard_gamma, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.rows import linear_rows
    model, meta, z, configs, ds, cfg = m0
    prob = _ard_e0_problem(m0)
    with highest_precision():
        ev = ARDEvidence(ard_statistics(_theta(), prob, ds, "joint"), ard_gamma(prob),
                         body_order_columns(meta, prob.cfg))
        post = ard_posterior(ev, ev.h0(_theta()), 1.3, meta)
        b = jax.tree.map(lambda a: a[0], ds)
        F = np.asarray(linear_rows(model, prob.cfg, b)[0].F)
        L0 = prob.cfg.len_readout
        assert not np.any(F[..., L0:])
        rng = np.random.default_rng(0)
        G = rng.standard_normal((prob.cfg.len_basis, 5))
        from ace_jax.fit.ard import sandwich_factor
        for p in (post, post._replace(R=jnp.asarray(G)), post._replace(Q=sandwich_factor(post, G))):
            np.testing.assert_allclose(p.atom_shape(F[:, :, :L0]), p.atom_shape(F), rtol=1e-12, atol=1e-300)
            np.testing.assert_allclose(p.forces_std(F[:, :, :L0]), p.forces_std(F), rtol=1e-12, atol=1e-300)
        with pytest.raises(ValueError, match="columns"):
            post.atom_shape(F[:, :, :L0 - 1])
        # misspec_var_rows checks the width too (no silent truncation of Q/dinv to a wrong width)
        pq = post._replace(Q=sandwich_factor(post, G))
        Fr = F.reshape(-1, F.shape[-1])
        np.testing.assert_allclose(pq.misspec_var_rows(Fr[:, :L0]), pq.misspec_var_rows(Fr), rtol=1e-12, atol=1e-300)
        with pytest.raises(ValueError, match="columns"):
            pq.misspec_var_rows(Fr[:, :L0 - 1])


def test_ard_posterior_without_e0_columns_loads_as_before(ard_setup, tmp_path):
    """A schema-3 posterior written before joint E0 under ARD (no e0_cols in meta) has n_e0 = 0 and
    serves from its full-width rows unchanged."""
    prob, ds, ev, h, post = ard_setup
    assert "e0_cols" not in post.meta and post.n_e0 == 0
    post.save(tmp_path / "p.npz")
    from ace_jax.fit.ard import ARDPosterior
    p2 = ARDPosterior.load(tmp_path / "p.npz")
    assert p2.n_e0 == 0
    b = jax.tree.map(lambda a: a[0], ds)
    F = np.asarray(linear_rows(prob.model, prob.cfg, b)[0].F)
    np.testing.assert_allclose(p2.forces_std(F), post.forces_std(F), rtol=1e-5)


def _ard_cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear",
                uq="ard", opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False, ard_n_min=1,
                **KEYS)
    return FitConfig(**{**base, **kw}).validate()


def _ard_fit(cfg, **data):
    from ace_jax.fit.pipeline import fit, load_fit_data
    d = load_fit_data(cfg, log=lambda *a: None, **data)
    return fit(cfg, d, log=lambda *a: None)


def test_ard_joint_e0_with_an_isolated_atom_matches_the_prefit_fit():
    """An isolated training atom pins its species' E0 column (as under BLR): the joint-E0 ARD fit is the
    pre-fit one, and the exported E0 is the isolated-atom energy."""
    from ace_jax.fit.pipeline.export import linear_model_arrays
    cfgs = load_configs(str(XYZ), **KEYS)
    iso = [c.energy for c in cfgs if len(c.numbers) == 1]
    j = _ard_fit(_ard_cfg(), data=str(XYZ))
    p = _ard_fit(_ard_cfg(e0="prefit"), data=str(XYZ))
    assert j.built.prob.cfg.e0_cols and not p.built.prob.cfg.e0_cols
    assert any(len(c.numbers) == 1 for c in j.data.train), "the split keeps the isolated atom in training"
    assert abs(linear_model_arrays(j)["E0"][0] - np.mean(iso)) < 1e-6
    a, b = j.preds.arrays["test/map"], p.preds.arrays["test/map"]
    # equal to the evidence optimiser's resolution along its flat directions (~3e-5 eV/atom, ~4e-4 eV/A;
    # the test RMSEs are ~0.2 eV/atom and ~1.9 eV/A)
    # two separate evidence fits: their L-BFGS endpoints drift by ~0.4 meV/atom across BLAS/CPU platforms (CI);
    # 2 meV/atom still separates "same fit" from any real joint-vs-prefit difference (~100 meV/atom)
    np.testing.assert_allclose(a["E_mean"] / a["nat"], b["E_mean"] / b["nat"], rtol=0, atol=2e-3)
    np.testing.assert_allclose(a["F_mean"], b["F_mean"], rtol=0, atol=2e-3)


def test_ard_joint_e0_fit_exports_and_serves_the_e0_shift(tmp_path):
    """Without an isolated atom the E0 column is free (prior N(pre-fit E0, 1 eV^2)): the model file folds
    the fitted shift into E0, so ACECalculator's energy is the fit's prediction, and its forces_std (from
    readout-only force rows) is the stage's served value."""
    from ase import Atoms
    from ase.io import read, write
    from ace_jax import ACECalculator
    from ace_jax.fit.pipeline import write_outputs
    from ace_jax.fit.pipeline.export import linear_model_arrays
    frames = [a for a in read(str(XYZ), ":") if len(a) > 1]
    write(tmp_path / "bulk.xyz", frames)
    cfg = _ard_cfg(ard_variance="sandwich")
    res = _ard_fit(cfg, data=str(tmp_path / "bulk.xyz"))
    pc = res.built.prob.cfg
    shift = np.asarray(res.ard.posterior.mean)[pc.len_readout:]
    assert pc.e0_cols and shift.shape == (pc.NZ,) and np.all(np.abs(shift) > 1e-6)
    arr = linear_model_arrays(res)
    np.testing.assert_allclose(arr["E0"], res.data.E0 + shift, rtol=0, atol=1e-12)
    write_outputs(res, tmp_path / "run", log=lambda *a: None)
    calc = ACECalculator(str(tmp_path / "run" / "model.npz"), posterior=str(tmp_path / "run" / "posterior.npz"))
    assert calc.posterior.n_e0 == pc.NZ
    pred = res.preds.arrays["test/map"]
    off = np.r_[0, np.cumsum(pred["nat"])]
    for k, c in enumerate(res.data.test_o[:3]):
        at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
        at.calc = calc
        assert abs(at.get_potential_energy() - pred["E_mean"][k]) < 1e-6 * max(1.0, abs(pred["E_mean"][k]))
        sd = np.asarray(calc.get_property("forces_std", at))
        # posterior.npz stores R in float32
        np.testing.assert_allclose(sd ** 2, pred["F_var"][off[k]:off[k + 1]].sum(1), rtol=1e-4, atol=1e-14)
