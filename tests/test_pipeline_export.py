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


def _cli_fit(tmp_path, *extra):
    from ase.io import read, write
    from ace_jax.cli import main
    cfgs = read(XYZ, ":")
    tr, te = tmp_path / "tr.xyz", tmp_path / "te.xyz"
    write(tr, cfgs[:12]); write(te, cfgs[12:16])
    out = tmp_path / "out"
    main(["fit", "--model", BASE["model"], "--train", str(tr), "--test", str(te), "--energy-key", "dft_energy",
          "--force-key", "dft_force", "--virial-key", "dft_virial", "--configs-per-batch", "4",
          "--rungs", "map", "--map-steps", "5", "--r0", "2.35", "--out", str(out), *extra])
    return te, out


def _cli_eval(model, te, csv_path):
    import csv
    from ace_jax.cli import main
    main(["eval", "--model", str(model), "--data", str(te), "--energy-key", "dft_energy",
          "--force-key", "dft_force", "--forces", "--out", str(csv_path)])
    return list(csv.DictReader(open(csv_path)))


def test_cli_fit_then_eval_linear_model(tmp_path):
    te, out = _cli_fit(tmp_path, "--m-per-species", "0")
    pred = np.load(out / "model.npz")                       # an ordinary ACE npz
    assert "WB" in pred.files and "meta_json" in pred.files
    rows = _cli_eval(out / "model.npz", te, tmp_path / "p.csv")
    assert len(rows) == 4 and "energy_std" not in rows[0]


def test_cli_fit_then_eval_gp_model(tmp_path):
    te, out = _cli_fit(tmp_path, "--m-per-species", "6", "--opt", "lbfgs", "--model-draws", "1")
    assert (out / "gp_model.npz").exists() and not (out / "model.npz").exists()
    rows = _cli_eval(out / "gp_model.npz", te, tmp_path / "p.csv")
    assert len(rows) == 4 and float(rows[0]["energy_std"]) > 0


def test_cli_no_save_model(tmp_path):
    _, out = _cli_fit(tmp_path, "--m-per-species", "0", "--no-save-model")
    assert (out / "metrics.csv").exists() and not (out / "model.npz").exists()
