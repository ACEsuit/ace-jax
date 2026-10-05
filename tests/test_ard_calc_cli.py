import jax
import numpy as np
import pytest
from ase import Atoms

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"


def test_cli_fit_ard_variance_default_sandwich_and_kappa_flag(fitted, tmp_path):
    """--ard-variance: default sandwich (the fixture fit, the PRESS shape factor R in posterior.npz); `kappa` once."""
    import json
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    assert json.load(open(fitted / "ard.json"))["variance"] == "sandwich"
    p = ARDPosterior.load(fitted / "posterior.npz")
    assert p.R is not None and p.Q is None and p.group_table is not None
    out = tmp_path / "kappa"
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "30",
                 "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                 "dft_virial", "--m-per-species", "0", "--uq", "ard", "--ard-variance", "kappa", "--opt",
                 "lbfgs", "--map-steps", "5", "--configs-per-batch", "4", "--r0", "2.35",
                 "--out", str(out)]) == 0
    assert json.load(open(out / "ard.json"))["variance"] == "kappa"
    pk = ARDPosterior.load(out / "posterior.npz")
    assert pk.Q is None and pk.R is None and pk.group_table is not None


def test_calculator_caches_shape_factor_on_device(fitted):
    """The shape factor (R, schema 3) is on the device once at construction, not moved on every forces_std."""
    from ace_jax import ACECalculator
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    assert calc.posterior.R is not None and isinstance(calc.posterior.R, jax.Array)
    assert calc.posterior.R.dtype == np.float64


def test_calculator_caches_chol_on_device(fitted):
    """The L x L Cholesky factor (numpy from posterior.npz) is moved to the device once at
    construction: the kappa path's var_rows would otherwise re-upload it on every call."""
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    assert isinstance(ARDPosterior.load(fitted / "posterior.npz").chol, np.ndarray)   # load stays numpy
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    assert isinstance(calc.posterior.chol, jax.Array) and calc.posterior.chol.dtype == np.float64


# How far lean=True may drift from lean=False.  The lean form (#16) is exact to roundoff today.  Once
# lean() also splines learned radials (#19, spline_tol "auto" = 1e-10), the lean MEAN may differ from
# the full model by up to that tolerance; sigma always comes from the full model.  Raise it then.
LEAN_RTOL = 1e-8


def test_posterior_lean_and_full_models_agree(fitted):
    """With posterior=, lean=True and lean=False give the same forces and forces_std: E/F come from
    eval_model (the lean, energy-only form when lean=True), sigma's design rows from the full model."""
    from ase.io import read
    from ace_jax import ACECalculator
    at = max(read(XYZ, ":8"), key=len)                                # the largest fixture cell
    out = {}
    for lean in (True, False):
        calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), lean=lean)
        if lean:
            assert calc.eval_model is not calc.model                  # the lean path is really exercised
        a = at.copy(); a.calc = calc
        out[lean] = (a.get_forces(), np.asarray(calc.get_property("forces_std", a)))
    (F1, s1), (F0, s0) = out[True], out[False]
    np.testing.assert_allclose(F1, F0, rtol=LEAN_RTOL, atol=LEAN_RTOL * np.abs(F0).max())
    np.testing.assert_allclose(s1, s0, rtol=LEAN_RTOL, atol=LEAN_RTOL * s0.max())
    assert s0.max() > 0


def test_model_swap_is_refused_with_a_posterior(fitted):
    """The posterior, its design-row model and the device factors all belong to the model FILE the
    calculator was built with: swapping calc.model would leave forces_std serving the old model's
    sigma beside the new model's forces, so it raises.  Without a posterior a swap still works."""
    from ace_jax import ACECalculator
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    with pytest.raises(ValueError, match="posterior"):
        calc.model = calc.model
    plain = ACECalculator(str(fitted / "model.npz"))
    plain.model = plain.model                                         # no posterior: allowed


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
    from ase.io import read, write
    from ace_jax.cli import main
    from ace_jax.fit.xyz import read_extxyz
    data = tmp_path / "d.xyz"
    write(data, read(XYZ, ":3"))
    assert main(["eval", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
                 "--data", str(data), "--energy-key", "dft_energy", "--force-key", "dft_force",
                 "--out", str(tmp_path / "p.xyz"), "--per-atom", str(tmp_path / "atoms.xyz")]) == 0
    ats = read(tmp_path / "atoms.xyz", ":")
    assert len(ats) == 3 and ats[0].arrays["forces_std"].shape == (len(ats[0]),)
    rows = read_extxyz(tmp_path / "p.xyz")
    assert len(rows) == 3 and all(r.arrays["ace_forces_std"].shape == (len(r.numbers),) for r in rows)
    # config 0 of the fixture is an isolated Si atom: no neighbours, so its force std is exactly 0
    assert len(rows[0].numbers) == 1 and float(rows[0].arrays["ace_forces_std"].max()) == 0.0
    assert all(float(r.arrays["ace_forces_std"].max()) > 0 for r in rows[1:])



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


def test_calculator_served_quantities_and_lean_parity(fitted):
    from ase.io import read
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    tab = ARDPosterior.load(fitted / "posterior.npz").group_table
    at = max(read(XYZ, ":8"), key=len)
    out = {}
    for lean in (True, False):
        calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), lean=lean)
        a = at.copy()
        a.calc = calc
        a.get_forces()
        sd = np.asarray(calc.get_property("forces_std", a))
        cov = np.asarray(calc.get_property("forces_cov", a))
        q = np.asarray(calc.get_property("forces_q", a))
        g = np.asarray(calc.get_property("forces_group", a))
        assert sd.shape == (len(a),) and cov.shape == (len(a), 3, 3) and g.dtype.kind == "i"
        np.testing.assert_allclose(np.trace(cov, axis1=1, axis2=2), sd ** 2, rtol=1e-10)
        v = sd ** 2 / np.asarray(tab["lam_rms"])[g] ** 2
        np.testing.assert_allclose(q, np.asarray(tab["q"])[g] * np.sqrt(v / 3), rtol=1e-8)
        out[lean] = (sd, q, g)
    np.testing.assert_allclose(out[True][0], out[False][0], rtol=LEAN_RTOL)
    np.testing.assert_allclose(out[True][1], out[False][1], rtol=LEAN_RTOL)
    assert np.array_equal(out[True][2], out[False][2])


def test_calculator_forces_support(fitted):
    from ase.io import read
    from ace_jax import ACECalculator
    at = max(read(XYZ, ":8"), key=len)
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at.calc = calc
    sup = calc.get_property("forces_support", at)
    ok = sup["support_ok"]
    assert ok.shape == (len(at),) and ok.dtype == bool and all(v > 0 for v in sup["n_eff"].values())
    assert np.isfinite(sup["support_q"][ok]).all() and np.isinf(sup["support_q"][~ok]).all()
    # the level: the (1 - alpha) weighted quantile with alpha the posterior's miscoverage (0.1), not alpha itself
    from ace_jax.fit.support import support_check
    post = calc.posterior
    alpha = float(np.asarray(post.group_table["alpha"]))
    assert alpha < 0.5
    X, Z = calc.support_descriptors(at)
    ref = support_check(post.support, X, Z, alpha)
    np.testing.assert_array_equal(sup["support_ok"], ref["support_ok"])
    np.testing.assert_array_equal(sup["support_q"], ref["support_q"])
    lo = support_check(post.support, X, Z, 1 - alpha)["support_q"]
    assert np.all(sup["support_q"] >= lo) and np.any(sup["support_q"] > lo)


def test_calculator_schema2_new_properties_raise(fitted, tmp_path):
    from ase.io import read
    from ace_jax import ACECalculator
    z = dict(np.load(fitted / "posterior.npz"))
    z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("R", "group_", "cal_", "support", "force_shape", "eps"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    at = read(XYZ, "0")
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(tmp_path / "p2.npz"))
    at.calc = calc
    assert np.isfinite(calc.get_property("forces_std", at)).all()
    with pytest.raises(ValueError, match="refit with --uq ard"):
        calc.get_property("forces_q", at)


def _calib_args(fitted, data, extra=()):
    return ["calibrate", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
            "--data", str(data), "--energy-key", "dft_energy", "--force-key", "dft_force",
            "--virial-key", "dft_virial", *extra]


@pytest.fixture(scope="module")
def calib_set(tmp_path_factory):
    """25 rattled copies of one training config (index 1: config 0 is the isolated atom, whose shape is zero)."""
    from ase.io import read, write
    base = read(XYZ, "1")
    out = []
    for i in range(25):
        a = base.copy()
        a.rattle(0.01, seed=i)
        out.append(a)
    p = tmp_path_factory.mktemp("cal") / "U.xyz"
    write(p, out)
    return p


def test_calibrate_per_group_replace(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    base = ARDPosterior.load(fitted / "posterior.npz")
    assert main(_calib_args(fitted, calib_set, ["--out", str(tmp_path / "d.npz")])) == 0
    new = ARDPosterior.load(tmp_path / "d.npz")
    t = new.group_table
    replaced = [g for g, nc in enumerate(t["n_cfg_cal"]) if nc >= t["n_min"]]
    assert replaced                                                       # the 25 copies cover >= 1 group
    for g in replaced:
        assert t["n_cfg_val"][g] == 0, g
    for g, (nv0, nc) in enumerate(zip(base.group_table["n_cfg_val"], t["n_cfg_cal"])):
        if nc < t["n_min"]:
            assert t["n_cfg_val"][g] == nv0, g                             # short groups keep T_val
    np.testing.assert_array_equal(new.mean, base.mean)
    assert t["sources"] and t["sources"][-1]["n_cfg"] == 25
    assert {"path", "sha256", "n_cfg", "n_atoms", "mode"} <= set(t["sources"][-1])


def test_calibrate_append_and_replace(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    assert main(_calib_args(fitted, calib_set, ["--append", "--out", str(tmp_path / "a.npz")])) == 0
    assert main(_calib_args(fitted, calib_set, ["--replace", "--out", str(tmp_path / "r.npz")])) == 0
    a, r = ARDPosterior.load(tmp_path / "a.npz"), ARDPosterior.load(tmp_path / "r.npz")
    assert set(np.unique(a.cal["src"]).tolist()) == {0, 1} and set(np.unique(r.cal["src"]).tolist()) == {1}
    assert sum(r.group_table["n_cfg_val"]) == 0


def test_calibrate_rebuilds_support(fitted, calib_set, tmp_path):
    """The support reference is rebuilt on the pooled atoms: the replace-mode reference is U only, append is T_val + U."""
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    base = ARDPosterior.load(fitted / "posterior.npz")
    assert base.support is not None
    assert main(_calib_args(fitted, calib_set, ["--append", "--out", str(tmp_path / "a.npz")])) == 0
    assert main(_calib_args(fitted, calib_set, ["--replace", "--out", str(tmp_path / "r.npz")])) == 0
    a, r = ARDPosterior.load(tmp_path / "a.npz"), ARDPosterior.load(tmp_path / "r.npz")
    zs = [z for z in base.support if z != "pca"]
    assert a.support is not None and r.support is not None
    for z in zs:
        assert len(a.support[z]["s"]) == len(base.support[z]["s"]) + len(r.support[z]["s"])
        assert len(a.support[z]["g"]) == len(a.support[z]["s"]) == len(a.support[z]["m"])
    assert a.support["pca"].keys() == base.support["pca"].keys()
    from ase.io import read
    from ace_jax import ACECalculator
    at = read(XYZ, "1")
    at.calc = ACECalculator(str(fitted / "model.npz"), posterior=str(tmp_path / "a.npz"))
    assert "support_ok" in at.calc.get_property("forces_support", at)


def test_calibrate_refuses_schema2(fitted, calib_set, tmp_path, capsys):
    from ace_jax.cli import main
    z = dict(np.load(fitted / "posterior.npz"))
    z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("R", "group_", "cal_", "support", "force_shape", "eps"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    args = _calib_args(fitted, calib_set, ["--out", str(tmp_path / "o.npz")])
    args[args.index("--posterior") + 1] = str(tmp_path / "p2.npz")
    with pytest.raises(SystemExit) as ex:                  # m3: the user-facing error path, no traceback
        main(args)
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("aj calibrate: error:") and "refit with --uq ard" in err


def test_calibrate_requires_forces(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    args = _calib_args(fitted, calib_set, ["--out", str(tmp_path / "o.npz")])
    args[args.index("--force-key") + 1] = "no_such_force"
    with pytest.raises(ValueError, match="no_such_force"):
        main(args)


def test_calibrate_append_replace_exclusive(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    with pytest.raises(SystemExit):
        main(_calib_args(fitted, calib_set, ["--append", "--replace", "--out", str(tmp_path / "o.npz")]))


def test_eval_per_atom_writes_served_arrays(fitted, tmp_path):
    from ase.io import read
    from ace_jax.cli import main
    assert main(["eval", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
                 "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
                 "--per-atom", str(tmp_path / "pa.xyz"), "--support"]) == 0
    a = read(tmp_path / "pa.xyz", "0")
    for k in ("forces_std", "forces_q", "forces_group", "support_ok", "support_q"):
        assert k in a.arrays, k


def test_fit_flags_map_to_config():
    from ace_jax.cli import _fit_config, _parser
    a = _parser().parse_args(["fit", "--model", "m.npz", "--data", "d.xyz", "--r0", "2.3", "--out", "o",
                              "--uq", "ard", "--m-per-species", "0", "--force-shape", "aniso",
                              "--ard-coverage", "0.8", "--ard-groups", "none", "--ard-cluster-size", "inf",
                              "--ard-press", "block", "--ard-n-min", "7", "--no-ard-support",
                              "--ard-transfer", "sqrt"])
    c = _fit_config(a)
    assert (c.ard_force_shape, c.ard_coverage, c.ard_groups, c.ard_press, c.ard_n_min, c.ard_support) == \
        ("aniso", 0.8, "none", "block", 7, False)
    assert c.ard_transfer == "sqrt"
    assert c.ard_cluster_size == float("inf")
    d = _fit_config(_parser().parse_args(["fit", "--model", "m.npz", "--data", "d.xyz", "--r0", "2.3", "--out", "o"]))
    assert (d.ard_force_shape, d.ard_coverage, d.ard_groups, d.ard_press, d.ard_n_min, d.ard_support) == \
        ("aniso", 0.9, "distortion", "exact", 20, True)
    assert d.ard_transfer == "exponent"


def test_calibrate_per_group_keeps_support_consistent(fitted, calib_set, tmp_path):
    """Stored (T_val) support points survive exactly when their conformal group is not replaced."""
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    base = ARDPosterior.load(fitted / "posterior.npz")
    assert main(_calib_args(fitted, calib_set, ["--out", str(tmp_path / "d.npz")])) == 0
    new = ARDPosterior.load(tmp_path / "d.npz")
    assert new.support is not None
    t = new.group_table
    replaced = np.array([nc >= t["n_min"] for nc in t["n_cfg_cal"]])
    assert replaced.any()
    zs = [z for z in base.support if z != "pca"]
    grp = lambda ref: np.concatenate([ref[z]["grp"] for z in zs])
    n_old_kept = int((~replaced[grp(base.support)]).sum())
    u_in_kept = int(((new.cal["src"] > 0) & ~replaced[new.cal["groups"].astype(int)]).sum())
    gn = grp(new.support)
    assert int((~replaced[gn]).sum()) == n_old_kept + u_in_kept          # stored points of kept groups, all of them
    n_u_replaced = int(((new.cal["src"] > 0) & replaced[new.cal["groups"].astype(int)]).sum())
    assert int(replaced[gn].sum()) == n_u_replaced                        # replaced groups hold U points only


def test_keep_for_mode_two_groups():
    from ace_jax.fit.support import fit_pca, build_support, recalibrate_support
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    Z = np.zeros(120, int)
    grp = np.arange(120) % 3                                              # groups 0, 1, 2
    ref = build_support(fit_pca({0: X}), X, Z, rng.random(120), np.arange(120) // 4, 1000, 0, grp=grp)
    u_cfg = np.array([25, 3, 0])                                          # n_min 20: only group 0 is replaced
    Xu = rng.normal(size=(10, 4))
    new = recalibrate_support(ref, "per-group", u_cfg, 20, Xu, np.zeros(10, int), rng.random(10),
                              np.arange(10), np.zeros(10, int))
    assert (new[0]["grp"] == 0).sum() == 10                               # group 0: U only
    assert (new[0]["grp"] == 1).sum() == (grp == 1).sum() and (new[0]["grp"] == 2).sum() == (grp == 2).sum()
    assert len(new[0]["g"]) == len(new[0]["m"]) == len(new[0]["s"]) == len(new[0]["Xc"])
    app = recalibrate_support(ref, "append", u_cfg, 20, Xu, np.zeros(10, int), rng.random(10),
                              np.arange(10), np.zeros(10, int))
    assert len(app[0]["s"]) == 130
    rep = recalibrate_support(ref, "replace", u_cfg, 20, Xu, np.zeros(10, int), rng.random(10),
                              np.arange(10), np.zeros(10, int))
    assert len(rep[0]["s"]) == 10
    legacy = {z: {k: v for k, v in r.items() if k != "grp"} if z != "pca" else r for z, r in ref.items()}
    assert recalibrate_support(legacy, "per-group", u_cfg, 20, Xu, np.zeros(10, int), rng.random(10),
                               np.arange(10), np.zeros(10, int)) is None


def test_aniso_fit_eval_and_calibrate(aniso_fit, calib_set):
    from ase.io import read
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    out = aniso_fit
    assert ARDPosterior.load(out / "posterior.npz").force_shape == "aniso"
    assert main(["eval", "--model", str(out / "model.npz"), "--posterior", str(out / "posterior.npz"),
                 "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
                 "--per-atom", str(out / "pa.xyz")]) == 0
    a = read(out / "pa.xyz", "1")
    assert a.arrays["forces_cov"].shape == (len(a), 9) and "forces_q_mahal" in a.arrays
    assert main(_calib_args(out, calib_set, ["--out", str(out / "c.npz")])) == 0
    assert ARDPosterior.load(out / "c.npz").force_shape == "aniso"


def test_resolved_yaml_records_ard_flags_and_roundtrips(tmp_path):
    import yaml
    from ace_jax.cli import _fit_config, _parse, main
    out = tmp_path / "r"
    ard = ["--force-shape", "aniso", "--ard-coverage", "0.8", "--ard-groups", "none", "--ard-cluster-size",
           "inf", "--ard-press", "block", "--ard-n-min", "5", "--no-ard-support", "--ard-transfer", "none"]
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "30",
                 "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                 "dft_virial", "--m-per-species", "0", "--uq", "ard", "--opt", "lbfgs", "--map-steps", "5",
                 "--configs-per-batch", "4", "--r0", "2.35", "--out", str(out), *ard]) == 0
    d = yaml.safe_load((out / "fit.yaml").read_text())
    assert (d["force_shape"], d["ard_coverage"], d["ard_groups"], d["ard_press"], d["ard_n_min"]) == \
        ("aniso", 0.8, "none", "block", 5)
    assert d["ard_cluster_size"] == float("inf") and d["no_ard_support"] is True and d["ard_transfer"] == "none"
    a = _parse(["fit", "--config", str(out / "fit.yaml")])
    b = _parse(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--r0", "2.35",
                "--out", "x", "--uq", "ard", "--m-per-species", "0", *ard])
    keys = ("ard_force_shape", "ard_coverage", "ard_groups", "ard_cluster_size", "ard_press", "ard_n_min",
            "ard_support", "ard_transfer")
    ca, cb = _fit_config(a), _fit_config(b)
    assert all(getattr(ca, k) == getattr(cb, k) for k in keys) and ca.ard_support is False
    # default fit: the negative flag is written false, and reads back as support on
    out2 = tmp_path / "d"
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "30",
                 "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                 "dft_virial", "--m-per-species", "0", "--uq", "ard", "--opt", "lbfgs", "--map-steps", "5",
                 "--configs-per-batch", "4", "--r0", "2.35", "--out", str(out2)]) == 0
    d2 = yaml.safe_load((out2 / "fit.yaml").read_text())
    assert d2["no_ard_support"] is False and d2["ard_cluster_size"] == 3.0 and d2["ard_transfer"] == "exponent"
    assert _fit_config(_parse(["fit", "--config", str(out2 / "fit.yaml")])).ard_support is True


def test_calibrate_loads_legacy_int8_cal_arrays(fitted, calib_set, tmp_path):
    """Posteriors saved with int8 cal_src/cal_groups still load and recalibrate; the output src is int16."""
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    z = dict(np.load(fitted / "posterior.npz"))
    n = len(z["cal_src"])
    src = (np.arange(n) % 5).astype(np.int8)                              # max 4 -> U source id 5
    z["cal_src"] = src
    z["cal_groups"] = z["cal_groups"].astype(np.int8)
    old = tmp_path / "old.npz"
    np.savez(old, **z)
    assert np.load(old)["cal_src"].dtype == np.int8
    post = ARDPosterior.load(old)
    np.testing.assert_array_equal(np.asarray(post.cal["src"]), src)
    args = ["calibrate", "--model", str(fitted / "model.npz"), "--posterior", str(old),
            "--data", str(calib_set), "--energy-key", "dft_energy", "--force-key", "dft_force",
            "--virial-key", "dft_virial", "--append", "--out", str(tmp_path / "new.npz")]
    assert main(args) == 0
    assert np.load(tmp_path / "new.npz")["cal_src"].dtype == np.int16
    new = ARDPosterior.load(tmp_path / "new.npz")
    s = np.asarray(new.cal["src"])
    np.testing.assert_array_equal(s[:n], src)                             # --append keeps every old score
    assert len(s) > n and (s[n:] == 5).all()


def test_posterior_properties_share_one_rows_and_shape_pass(fitted, monkeypatch):
    """I4: forces_std / forces_cov / forces_q / forces_group on the same atoms build the (N, 3, L) force rows
    and the shape V once; moving the atoms invalidates and recomputes them once more."""
    from ase.io import read
    import ace_jax.fit.rows as rows_mod
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    n_rows, n_shape = [], []
    real_rows, real_shape = rows_mod.chunked_rows_fn, ARDPosterior.atom_shape

    def spy_rows(*a, **k):
        f = real_rows(*a, **k)
        return lambda b: (n_rows.append(1), f(b))[1]

    monkeypatch.setattr(rows_mod, "chunked_rows_fn", spy_rows)
    monkeypatch.setattr(ARDPosterior, "atom_shape", lambda self, F: (n_shape.append(1), real_shape(self, F))[1])
    at = read(XYZ, "1")
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at.calc = calc
    at.get_forces()
    props = {p: np.asarray(calc.get_property(p, at)) for p in ("forces_std", "forces_cov", "forces_q", "forces_group")}
    assert len(n_rows) == 1 and len(n_shape) == 1
    np.testing.assert_allclose(np.trace(props["forces_cov"], axis1=1, axis2=2), props["forces_std"] ** 2, rtol=1e-10)
    at.positions[0] += 0.01
    for p in ("forces_q", "forces_std", "forces_cov"):
        calc.get_property(p, at)
    assert len(n_rows) == 2 and len(n_shape) == 2


def test_forces_q_mahal_on_schema2_raises_need3(fitted, tmp_path):
    """m3: forces_q_mahal on a schema-2 posterior is the schema error, not "needs an aniso posterior"."""
    from ase.io import read
    from ace_jax import ACECalculator
    z = dict(np.load(fitted / "posterior.npz"))
    z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("R", "group_", "cal_", "support", "force_shape", "eps"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    at = read(XYZ, "1")
    at.calc = ACECalculator(str(fitted / "model.npz"), posterior=str(tmp_path / "p2.npz"))
    with pytest.raises(ValueError, match="refit with --uq ard"):
        at.calc.get_property("forces_q_mahal", at)


def _rattled_si(n=2, seed=0):
    from ase.build import bulk
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, n, n))
    at.rattle(0.05, seed=seed)
    return at


@pytest.mark.parametrize("tau", [1.0, 0.9])
def test_calculator_committee_shape_path_matches_rows(fitted, tau):
    """shape_path='committee' (the r-output linear ACE) serves what the design rows serve, to roundoff,
    for the exact R and for a truncated one (shape_tau applies to both paths)."""
    from ace_jax import ACECalculator
    kw = dict(posterior=str(fitted / "posterior.npz"), shape_tau=tau)
    at = _rattled_si()
    props = ("forces_std", "forces_cov", "forces_q", "forces_group")
    rows = ACECalculator(str(fitted / "model.npz"), **kw)
    com = ACECalculator(str(fitted / "model.npz"), shape_path="committee", **kw)
    assert com.posterior.R.shape == rows.posterior.R.shape
    for p in props:
        a, b = np.asarray(rows.get_property(p, at)), np.asarray(com.get_property(p, at))
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-13 * max(np.abs(a).max(), 1.0), err_msg=p)


def test_calculator_shape_tau_truncates_rank(fitted):
    from ace_jax import ACECalculator
    full = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    r = full.posterior.R.shape[1]
    half = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), shape_tau=0.5)
    one = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), shape_rank=1)
    assert 1 <= half.posterior.R.shape[1] < r and one.posterior.R.shape[1] == 1
    at = _rattled_si(seed=1)
    assert np.all(half.get_property("forces_std", at) <= full.get_property("forces_std", at) * (1 + 1e-12))


def test_calculator_shape_path_validated(fitted):
    from ace_jax import ACECalculator
    with pytest.raises(ValueError, match="shape_path"):
        ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), shape_path="fast")
    with pytest.raises(ValueError, match="shape_tau"):
        ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), shape_tau=1.5)


def test_cli_eval_and_calibrate_shape_path_committee(fitted, calib_set, tmp_path):
    """--shape-path committee serves and recalibrates exactly what the design-row path does."""
    from ase.io import read, write
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.xyz import read_extxyz
    data = tmp_path / "d.xyz"
    write(data, read(XYZ, ":4"))
    std = {}
    for path in ("rows", "committee"):
        out = tmp_path / f"p_{path}.xyz"
        assert main(["eval", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
                     "--data", str(data), "--energy-key", "dft_energy", "--force-key", "dft_force",
                     "--out", str(out), "--shape-path", path]) == 0
        std[path] = np.concatenate([r.arrays["ace_forces_std"] for r in read_extxyz(out)])
        assert main(_calib_args(fitted, calib_set, ["--out", str(tmp_path / f"c_{path}.npz"),
                                                    "--shape-path", path])) == 0
    np.testing.assert_allclose(std["committee"], std["rows"], rtol=1e-10, atol=1e-14)
    assert std["rows"][0] == 0.0 and std["committee"][0] == 0.0          # the isolated atom
    a, b = (ARDPosterior.load(tmp_path / f"c_{p}.npz").group_table for p in ("rows", "committee"))
    np.testing.assert_allclose(b["q"], a["q"], rtol=1e-6)
    np.testing.assert_allclose(b["lam_rms"], a["lam_rms"], rtol=1e-6)
