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


def test_cli_fit_ard_variance_default_sandwich_and_kappa_flag(fitted, tmp_path):
    """--ard-variance: default sandwich (the fixture fit, Q in posterior.npz); `kappa` once."""
    import json
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    assert json.load(open(fitted / "ard.json"))["variance"] == "sandwich"
    assert ARDPosterior.load(fitted / "posterior.npz").Q is not None
    out = tmp_path / "kappa"
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "30",
                 "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                 "dft_virial", "--m-per-species", "0", "--uq", "ard", "--ard-variance", "kappa", "--opt",
                 "lbfgs", "--map-steps", "5", "--configs-per-batch", "4", "--r0", "2.35",
                 "--out", str(out)]) == 0
    assert json.load(open(out / "ard.json"))["variance"] == "kappa"
    assert ARDPosterior.load(out / "posterior.npz").Q is None


def test_calculator_caches_Q_on_device(fitted):
    """The sandwich factor is moved to the device once at construction, not on every forces_std."""
    from ace_jax import ACECalculator
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    assert calc.posterior.Q is not None and isinstance(calc.posterior.Q, jax.Array)
    assert calc.posterior.Q.dtype == np.float64


def test_calculator_caches_chol_on_device(fitted):
    """The L x L Cholesky factor (numpy from posterior.npz) is moved to the device once at
    construction: the kappa path's var_rows would otherwise re-upload it on every call."""
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    assert isinstance(ARDPosterior.load(fitted / "posterior.npz").chol, np.ndarray)   # load stays numpy
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    assert isinstance(calc.posterior.chol, jax.Array) and calc.posterior.chol.dtype == np.float64


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
        got.append(calc.get_property("forces_std", at))
    np.testing.assert_allclose(np.concatenate(got), ref, rtol=1e-6, atol=1e-12)


def test_isolated_atom_has_zero_finite_std(fitted):
    from ace_jax import ACECalculator
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at = Atoms("Si", positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=False); at.calc = calc
    s = calc.get_property("forces_std", at)
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



def test_forces_std_only_on_request_by_default(fitted):
    """Spec 4: forces_std runs only when requested (default), or every call when opted in."""
    from ace_jax import ACECalculator
    from ase.io import read
    at = read(XYZ, "1")
    lazy = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at.calc = lazy
    F = at.get_forces()
    assert "forces_std" not in lazy.results                       # a plain get_forces() does not pay for it
    s = lazy.get_property("forces_std", at)
    np.testing.assert_allclose(lazy.results["forces"], F)          # the cached forces were kept
    eager = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"),
                          forces_std_every_call=True)
    at2 = at.copy(); at2.calc = eager
    at2.get_forces()
    np.testing.assert_allclose(s, eager.results["forces_std"], rtol=1e-12)
    assert s.shape == (len(at),) and s.max() > 0


def test_posterior_with_other_elements_is_refused(fitted, tmp_path):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    p = ARDPosterior.load(fitted / "posterior.npz")
    els = list(p.meta["elements"])
    for bad_els in ([14 if e != 14 else 6 for e in els], els[::-1] if len(els) > 1 else None):
        if bad_els is None:
            continue
        p._replace(meta={**p.meta, "elements": bad_els}).save(tmp_path / "bad.npz")
        with pytest.raises(ValueError, match="elements"):
            ACECalculator(str(fitted / "model.npz"), posterior=str(tmp_path / "bad.npz"))


def test_cli_eval_refuses_posterior_with_gp_model(fitted, tmp_path):
    from ase.io import read, write
    from ace_jax.cli import main
    data = tmp_path / "d.xyz"
    write(data, read(XYZ, ":2"))
    gp_stub = tmp_path / "gp_model.npz"
    np.savez(gp_stub, gp_json=np.frombuffer(b"{}", np.uint8))      # cmd_eval keys GP models on gp_json
    with pytest.raises(ValueError, match="posterior"):
        main(["eval", "--model", str(gp_stub), "--posterior", str(fitted / "posterior.npz"),
              "--data", str(data), "--energy-key", "dft_energy", "--force-key", "dft_force"])


def test_posterior_without_x64_raises(fitted):
    """M1: forces_std needs float64 (A^-1 at cond ~1e13); with x64 off the calculator must refuse,
    not silently solve in float32.  x64 is global state: run in a fresh interpreter."""
    import os
    import subprocess
    import sys
    code = ("import jax\n"
            "assert not jax.config.jax_enable_x64\n"
            "from ace_jax import ACECalculator\n"
            "try:\n"
            f"    ACECalculator({str(fitted / 'model.npz')!r}, posterior={str(fitted / 'posterior.npz')!r})\n"
            "except RuntimeError as e:\n"
            "    print('RAISED', e)\n"
            "else:\n"
            "    print('NO ERROR', jax.config.jax_enable_x64)\n")
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "RAISED" in out.stdout and "jax_enable_x64" in out.stdout, out.stdout


def test_posterior_from_another_fit_is_refused(fitted, tmp_path):
    """M5: the posterior mean must be this model's readout (same `_place` column layout): a model
    whose coefficients differ -- another fit's, or an edited file -- is refused; the true pair loads."""
    from ace_jax import ACECalculator
    ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))      # genuine pair
    z = dict(np.load(fitted / "model.npz"))
    for key in ("WB", "Wpair"):
        bad = {**z, key: z[key].copy()}
        bad[key][0, 0] *= 1 + 1e-6
        np.savez(tmp_path / "other.npz", **bad)
        with pytest.raises(ValueError, match="coefficients"):
            ACECalculator(str(tmp_path / "other.npz"), posterior=str(fitted / "posterior.npz"))
