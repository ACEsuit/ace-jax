import jax
import numpy as np
import pytest
from ase import Atoms

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    from ace_jax.cli import main
    out = tmp_path_factory.mktemp("ard")
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "30",
                 "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                 "dft_virial", "--m-per-species", "0", "--uq", "ard", "--opt", "lbfgs", "--map-steps", "5",
                 "--configs-per-batch", "4", "--r0", "2.35", "--out", str(out)]) == 0
    return out


def test_calculator_forces_std_matches_pipeline(fitted):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior, predict_ard
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.pipeline import load_fit_data
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    cfgs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:3]
    post = ARDPosterior.load(fitted / "posterior.npz")
    cfg = FitConfig(model=str(fitted / "model.npz"), arm="linear", r0=2.35, batch=1, energy_key="dft_energy",
                    force_key="dft_force", virial_key="dft_virial", e0="model").validate()
    d = load_fit_data(cfg, train=str(XYZ), test=str(XYZ))
    prob = build_problem(cfg, d).prob
    ds = build_dataset(cfgs, d.meta, d.E0, 1)
    ref = np.sqrt(np.asarray(predict_ard(post, prob, ds).F_var).sum(1))
    got = []
    for c in cfgs:
        at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc); at.calc = calc
        at.get_forces(); got.append(calc.results["forces_std"])
    np.testing.assert_allclose(np.concatenate(got), ref, rtol=1e-6, atol=1e-12)


def test_isolated_atom_has_zero_finite_std(fitted):
    from ace_jax import ACECalculator
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at = Atoms("Si", positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=False); at.calc = calc
    at.get_forces()
    s = calc.results["forces_std"]
    assert s.shape == (1,) and np.all(np.isfinite(s)) and s[0] == 0.0


def test_mismatched_posterior_is_refused(fitted, tmp_path):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    p = ARDPosterior.load(fitted / "posterior.npz")
    bad = p._replace(mean=p.mean[:-1], dinv=p.dinv[:-1], chol=p.chol[:-1, :-1], body_col=p.body_col[:-1])
    bad.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="posterior"):
        ACECalculator(str(fitted / "model.npz"), posterior=str(tmp_path / "bad.npz"))


def test_cli_eval_with_posterior_writes_per_atom_std(fitted, tmp_path):
    import csv
    from ase.io import read, write
    from ace_jax.cli import main
    data = tmp_path / "d.xyz"
    write(data, read(XYZ, ":3"))
    assert main(["eval", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
                 "--data", str(data), "--energy-key", "dft_energy", "--force-key", "dft_force", "--forces",
                 "--out", str(tmp_path / "p.csv"), "--per-atom", str(tmp_path / "atoms.xyz")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "p.csv")))
    assert len(rows) == 3
    # config 0 of the fixture is an isolated Si atom: no neighbours, so its force std is exactly 0
    assert int(rows[0]["natoms"]) == 1 and float(rows[0]["fmax_std"]) == 0.0
    assert all(float(r["fmax_std"]) > 0 for r in rows[1:])
    ats = read(tmp_path / "atoms.xyz", ":")
    assert len(ats) == 3 and ats[0].arrays["forces_std"].shape == (len(ats[0]),)


def test_forces_std_only_on_request_when_not_every_call(fitted):
    """Spec: forces_std runs only when requested, or when set to compute every call (the default)."""
    from ace_jax import ACECalculator
    from ase.io import read
    at = read(XYZ, "1")
    lazy = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"),
                         forces_std_every_call=False)
    at.calc = lazy
    F = at.get_forces()
    assert "forces_std" not in lazy.results
    s = lazy.get_property("forces_std", at)
    np.testing.assert_allclose(lazy.results["forces"], F)          # the cached forces were kept
    eager = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at2 = at.copy(); at2.calc = eager
    at2.get_forces()
    np.testing.assert_allclose(s, eager.results["forces_std"], rtol=1e-12)
    assert s.shape == (len(at),) and s.max() > 0
