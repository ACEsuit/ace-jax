"""Smoke test: MAP + Laplace through the CLI on Si_tiny, metrics written; eval."""
import csv
import json

import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)

from ace_jax.cli import main

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def _cli_fit_args(tmp_path):
    """Trimmed by default (12 train / 4 test configs of Si_tiny) so the test is a
    fast plumbing check; ACEJAX_CLI_FULL=1 runs the physically meaningful version
    by hand (all 53 configs, tested on the training file, 150 MAP steps)."""
    import os
    if os.environ.get("ACEJAX_CLI_FULL"):
        train, test = XYZ, XYZ
    else:
        from ase.io import read, write
        cfgs = read(XYZ, ":")
        train, test = tmp_path / "train.xyz", tmp_path / "test.xyz"
        write(train, cfgs[:12]); write(test, cfgs[12:16])
    return ["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--train", str(train), "--test", str(test),
            "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
            # linear arm: the Laplace plumbing and R32 are the point; the GP residual
            # path is covered by the pipeline/export tests at a third of the cost
            "--configs-per-batch", "4", "--m-per-species", "0", "--rungs", "map,laplace",
            # 150 Adam MAP steps: at 30 the MAP is far from the mode and the Laplace Hessian is
            # indefinite (singular-Hessian warning, constant draws) -- Ruling R32.  Adam, not the
            # default L-BFGS: on this 12-config subset the converged MAP runs sigma_F to its lower
            # bound (the basis interpolates the forces), where the SVI Laplace Hessian is singular
            "--n-draws", "5", "--opt", "adam", "--map-steps", "150", "--r0", "2.35", "--out", str(tmp_path / "out")]


def test_cli_map_laplace(tmp_path):
    main(_cli_fit_args(tmp_path))
    tmp_path = tmp_path / "out"
    with open(tmp_path / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["rung"] for r in rows} == {"map", "laplace"}
    assert {r["quantity"] for r in rows} == {"E", "F", "V"}
    assert all(float(r["rmse"]) >= 0 for r in rows)
    assert (tmp_path / "draws_laplace.npy").exists() and (tmp_path / "theta_map.json").exists()
    with open(tmp_path / "config.json") as fh:
        cfg = json.load(fh)
    assert cfg["M"] == 0
    # Ruling R32: the noise parameters log_sigma_E/F/V are identified by data, so the
    # Laplace draws must have non-zero spread in those columns even at smoke settings.
    draws = np.load(tmp_path / "draws_laplace.npy")
    assert draws.shape[1] == 10 and draws[:, 7:10].std(0).max() > 0


def test_cli_map_laplace_on_the_default_map(tmp_path):
    """The same fit on the default (converged, polished) MAP: sigma_F ends on its lower bound there
    (the basis interpolates 12 configs' forces), so the Laplace rung holds it fixed instead of
    taking a singular Hessian; the draws are finite and the other noise scales still spread."""
    import warnings
    args = _cli_fit_args(tmp_path)
    i = args.index("--opt"); del args[i:i + 2]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        main(args)
    assert not [x for x in w if "singular" in str(x.message).lower()], [str(x.message) for x in w]
    draws = np.load(tmp_path / "out" / "draws_laplace.npy")
    theta = json.load(open(tmp_path / "out" / "theta_map.json"))
    fixed = json.load(open(tmp_path / "out" / "map_convergence.json"))
    assert np.all(np.isfinite(draws)) and draws.shape[1] == 10
    assert abs(theta["log_sigma_F"] - np.log(1e-4)) < 1e-8 and draws[:, 8].std() == 0     # held on its bound
    assert draws[:, [7, 9]].std(0).min() > 0
    assert fixed["converged"]


def test_cli_eval(tmp_path, capsys):
    """The eval subcommand evaluates a fitted model on a dataset, prints the RMSE table
    vs the labels and writes the predictions to extxyz.  main() returns 0: the console
    script passes its value to sys.exit, and returning the rows once made every
    successful run exit 1."""
    from ase.io import read, write
    from ace_jax.fit.xyz import read_extxyz
    data = tmp_path / "d.xyz"
    write(data, read(XYZ, ":4"))                  # each config size compiles anew: keep it small
    out = tmp_path / "pred.xyz"
    assert main(["eval", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(data),
                 "--energy-key", "dft_energy", "--force-key", "dft_force", "--out", str(out)]) == 0
    frames = read_extxyz(out)
    assert len(frames) == 4 and all("ace_energy" in f.info and "ace_forces" in f.arrays for f in frames)
    assert "config type" in capsys.readouterr().out


def test_cli_eval_matches_the_exact_model_per_config(tmp_path):
    """aj eval of a linear model.npz (the jitted calculator path) reproduces the exact
    per-config E/F of the model itself, isolated atom included."""
    import jax.numpy as jnp
    from ace_jax.eval import load, sparse_graph, species_indices
    from ace_jax.fit.data import load_configs
    from conftest import small_si_xyz
    from ace_jax.fit.xyz import read_extxyz
    xyz = small_si_xyz(tmp_path / "si12.xyz")                 # the isolated atom + 11 cells
    out = tmp_path / "p.xyz"
    assert main(["eval", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(xyz),
                 "--energy-key", "dft_energy", "--force-key", "dft_force", "--out", str(out)]) == 0
    rows = read_extxyz(out)
    model, meta, _ = load(FIXTURE_DIR / "si_fitted.npz")
    cs = load_configs(xyz, energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
    assert len(rows) == len(cs) == 12
    for r, c in list(zip(rows, cs))[::3]:
        g = sparse_graph(c.positions, c.cell, c.pbc, float(meta["rcut"]))
        nz = jnp.asarray(species_indices(meta, c.numbers)); s, v = jnp.asarray(g.senders), jnp.asarray(g.receivers)
        E, F, _ = model.energy_forces_virial(jnp.asarray(g.rij), nz[s], nz[v], s, v, g.n_nodes, nz)
        assert abs(float(r.info["ace_energy"]) - float(E)) <= 1e-8 * max(1.0, abs(float(E)))   # %16.8f
        np.testing.assert_allclose(r.arrays["ace_forces"], np.asarray(F), rtol=0, atol=1e-8)
