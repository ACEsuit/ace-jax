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


def test_ard_and_pops_keep_the_prefit_e0():
    from ace_jax.fit.pipeline import FitConfig as _F
    assert _F(model="m.npz", e0="lsq", arm="linear", learn_radial=True).validate().joint_e0 is True
    from ace_jax.fit.pipeline import FitConfig
    assert FitConfig(model="m.npz", e0="lsq", uq="pops", arm="linear").validate().joint_e0 is False
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
