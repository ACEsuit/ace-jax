"""Smoke test: MAP + Laplace on Si_tiny with a tiny inducing set, metrics written."""
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


def test_cli_map_laplace(tmp_path):
    main(["--model", str(FIXTURE_DIR / "si_fitted.npz"), "--train", str(XYZ), "--test", str(XYZ),
          "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
          "--configs-per-batch", "4", "--m-per-species", "6", "--rungs", "map,laplace",
          # 150 MAP steps: at 30 the MAP is far from the mode and the Laplace Hessian is
          # indefinite (singular-Hessian warning, constant draws) -- Ruling R32.
          "--n-draws", "5", "--map-steps", "150", "--r0", "2.35", "--out", str(tmp_path)])
    with open(tmp_path / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["rung"] for r in rows} == {"map", "laplace"}
    assert {r["quantity"] for r in rows} == {"E", "F", "V"}
    assert all(float(r["rmse"]) >= 0 for r in rows)
    assert (tmp_path / "draws_laplace.npy").exists() and (tmp_path / "theta_map.json").exists()
    with open(tmp_path / "config.json") as fh:
        cfg = json.load(fh)
    assert cfg["M"] == 6
    # Ruling R32: the noise parameters log_sigma_E/F/V are identified by data, so the
    # Laplace draws must have non-zero spread in those columns even at smoke settings.
    draws = np.load(tmp_path / "draws_laplace.npy")
    assert draws.shape[1] == 10 and draws[:, 7:10].std(0).max() > 0
