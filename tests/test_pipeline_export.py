"""Fitted-model files written by the pipeline: a linear fit saves an ACE npz that
ACECalculator loads directly; a GP fit saves a self-contained gp_model.npz that
GPCalculator.from_file reloads.  Both must reproduce the pipeline's own test-set
predictions, which is what makes the file the fitted model and not a lookalike."""
import jax
import numpy as np
import pytest
from ase import Atoms

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR  # noqa: E402

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")

BASE = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
            virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, rungs=("map",),
            predict_train=False, e0="lsq",
            # the CLI's statistics path, which the export uses too; the cached path
            # (run.py) differs by summation order, ~1e-6 in the GP forces
            predict_stats="recompute")


def _fit(**kw):
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(**{**BASE, **kw}).validate()
    return fit(cfg, load_fit_data(cfg, data=str(XYZ)), log=lambda *a: None)


def _atoms(c):
    return Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)


@pytest.fixture(scope="module")
def linear_res():
    return _fit(arm="linear", map_steps=5)


@pytest.fixture(scope="module")
def gp_res():
    return _fit(arm="gp", m_per_species=6, density="pair", opt="lbfgs", map_steps=10)


def test_linear_fit_saves_an_ace_model_that_reproduces_the_fit(linear_res, tmp_path):
    from ace_jax import ACECalculator
    from ace_jax.fit.pipeline import save_model
    path = save_model(linear_res, tmp_path)
    assert path.name == "model.npz"
    pred = linear_res.preds.arrays["test/map"]
    calc = ACECalculator(str(path))
    E, F = [], []
    for c in linear_res.data.test:
        at = _atoms(c); at.calc = calc
        E.append(at.get_potential_energy()); F.append(at.get_forces())
    assert np.allclose(E, pred["E_mean"], rtol=1e-9, atol=1e-9)
    assert np.allclose(np.concatenate(F), pred["F_mean"], rtol=1e-8, atol=1e-9)


def test_gp_fit_saves_a_file_gpcalculator_reloads(gp_res, tmp_path):
    from ace_jax import GPCalculator
    from ace_jax.fit.pipeline import save_model
    path = save_model(gp_res, tmp_path)
    assert path.name == "gp_model.npz"
    pred = gp_res.preds.arrays["test/map"]
    calc = GPCalculator.from_file(str(path))
    E, Es, F = [], [], []
    for c in gp_res.data.test:
        at = _atoms(c); at.calc = calc
        E.append(at.get_potential_energy()); F.append(at.get_forces())
        Es.append(calc.results["energy_std"])
    assert np.allclose(E, pred["E_mean"], rtol=1e-9, atol=1e-9)
    assert np.allclose(np.concatenate(F), pred["F_mean"], rtol=1e-8, atol=1e-9)
    assert np.allclose(np.square(Es), pred["E_var"], rtol=1e-6)


def test_baseline_fits_are_not_saved(linear_res, tmp_path):
    """The dimer baseline is added back outside the model, so a saved model would
    silently omit it: refuse rather than write a wrong potential."""
    from ace_jax.fit.pipeline import save_model
    res = linear_res._replace(config=linear_res.config.__class__(**{**BASE, "arm": "linear",
                                                                     "baseline": "dimer_mean.npz"}))
    assert save_model(res, tmp_path) is None
    assert not (tmp_path / "model.npz").exists()


def test_write_outputs_saves_the_model_by_default(linear_res, tmp_path):
    from ace_jax.fit.pipeline import write_outputs
    write_outputs(linear_res, tmp_path, layout=("cli",), argv={})
    assert (tmp_path / "model.npz").exists()
    write_outputs(linear_res, tmp_path / "no", layout=("cli",), argv={}, save_model=False)
    assert not (tmp_path / "no" / "model.npz").exists()



def test_cli_eval_reads_a_gp_model(gp_res, tmp_path):
    """`ace-jax eval` on a gp_model.npz goes through GPCalculator and adds the
    predictive ace_energy_std and ace_forces_std."""
    from ase.io import write
    from ace_jax.cli import main
    from ace_jax.fit.pipeline import save_model
    from ace_jax.fit.xyz import read_extxyz
    path = save_model(gp_res, tmp_path)
    data = tmp_path / "te.xyz"
    write(data, [_atoms(c) for c in gp_res.data.test])
    assert main(["eval", "--model", str(path), "--data", str(data), "--out", str(tmp_path / "p.xyz")]) == 0
    rows = read_extxyz(tmp_path / "p.xyz")
    assert len(rows) == len(gp_res.data.test) and float(rows[0].info["ace_energy_std"]) > 0
    assert rows[0].arrays["ace_forces_std"].shape == (len(rows[0].numbers), 3)


def test_cli_eval_no_deriv_dtc_reaches_the_gp_calculator(gp_res, tmp_path, monkeypatch):
    """`aj eval --no-deriv-dtc` builds the GPCalculator with deriv_dtc=False."""
    from ase.io import write
    from ace_jax.calc import gp
    from ace_jax.cli import main
    from ace_jax.fit.pipeline import save_model
    seen = []
    orig = gp.GPCalculator.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        seen.append(self.deriv_dtc)

    monkeypatch.setattr(gp.GPCalculator, "__init__", spy)
    path = save_model(gp_res, tmp_path)
    data = tmp_path / "te.xyz"
    write(data, [_atoms(c) for c in gp_res.data.test])
    for flag, want in ((["--no-deriv-dtc"], False), ([], True)):
        assert main(["eval", "--model", str(path), "--data", str(data),
                     "--out", str(tmp_path / "p.xyz")] + flag) == 0
        assert seen[-1] is want


def test_pops_fit_saves_the_pops_mean_exactly(tmp_path):
    """A POPS run predicts with the ridge path's pinned mean c*; the saved model must
    hold exactly that vector.  A second, independent solve agreed only to the
    conditioning (4e-4 eV/A in forces at the production Cantor basis, L = 27.6k)."""
    from ace_jax.fit.pipeline import save_model
    res = _fit(arm="linear", map_steps=5, uq="pops", opt="lbfgs")
    z = np.load(save_model(res, tmp_path))
    cfg = res.built.prob.cfg
    mu = np.concatenate([z["WB"].T.reshape(-1), z["Wpair"].T.reshape(-1)])
    assert mu.shape == (cfg.len_basis,)
    assert np.array_equal(mu, res.preds.pops["mean"])
    # ... and that mean is the one the POPS predictions were made with
    from ace_jax import ACECalculator
    calc = ACECalculator(str(tmp_path / "model.npz"))
    E = []
    for c in res.data.test:
        at = _atoms(c); at.calc = calc; E.append(at.get_potential_energy())
    assert np.allclose(E, res.preds.arrays["test/map"]["E_mean"], rtol=1e-10, atol=1e-10)


def test_gpcalculator_evaluates_an_isolated_atom(gp_res, tmp_path, monkeypatch):
    """No edges at all: E0 plus the model's empty-environment site term, zero
    forces.  The empty neighbour list must not change the answer, so it is
    checked against the same structure padded to 8 (all masked) neighbour slots."""
    import ace_jax.calc.gp as G
    from ace_jax import GPCalculator
    from ace_jax.fit.pipeline import save_model
    path = save_model(gp_res, tmp_path)

    def energies(k_cap=None):
        calc = GPCalculator.from_file(str(path))
        at = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 20.0, pbc=True)
        at.calc = calc
        return at.get_potential_energy(), at.get_forces(), calc.results["energy_std"]

    E, F, Es = energies()
    real = G.build_dataset
    monkeypatch.setattr(G, "build_dataset", lambda *a, **k: real(*a, **{**k, "k_cap": 8}))
    E8, F8, Es8 = energies()
    assert np.isfinite(E) and np.isfinite(Es) and np.array_equal(F, np.zeros((1, 3)))
    assert np.isclose(E, E8, rtol=1e-12, atol=1e-12) and np.isclose(Es, Es8, rtol=1e-9)


def test_cli_eval_gp_model_on_the_si_fixture(gp_res, tmp_path):
    """`aj eval` of a GP model over si_tiny_train.xyz, whose first frame is an
    isolated atom."""
    from ace_jax.cli import main
    from ace_jax.fit.pipeline import save_model
    from ace_jax.fit.xyz import read_extxyz
    path = save_model(gp_res, tmp_path)
    assert main(["eval", "--model", str(path), "--data", str(XYZ), "--energy-key", "dft_energy",
                 "--force-key", "dft_force", "--virial-key", "dft_virial",
                 "--out", str(tmp_path / "p.xyz")]) == 0
    rows = read_extxyz(tmp_path / "p.xyz")
    assert len(rows) == 53 and len(rows[0].numbers) == 1
    assert all(np.isfinite(r.info["ace_energy"]) and np.isfinite(r.info["ace_energy_std"]) for r in rows)
