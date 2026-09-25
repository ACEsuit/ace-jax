# tests/test_pipeline_parity.py
"""The drivers must reproduce their pre-refactor outputs exactly (Task 0 goldens)."""
import csv, json, os, pathlib, subprocess, sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLD = ROOT / "fixtures" / "pipeline_golden"
sys.path.insert(0, str(ROOT / "tests" / "pipeline_golden"))
from make_golden import SCENARIOS, cli_argv, run_argv  # noqa: E402

RTOL = 1e-12


def _same(a, b, where):
    if isinstance(a, dict):
        assert set(a) == set(b), where
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert np.isclose(a, b, rtol=RTOL, atol=0, equal_nan=True), (where, a, b)
    else:
        assert a == b, (where, a, b)


def _compare(got, gold):
    names = sorted(p.name for p in gold.iterdir())
    assert names == sorted(p.name for p in got.iterdir() if p.name in names), names
    for n in names:
        g, r = got / n, gold / n
        if n == "config.json":
            continue                                   # holds the (temp) paths it was run with
        if n.endswith(".json"):
            _same(json.load(open(g)), json.load(open(r)), n)
        elif n.endswith(".npy"):
            assert np.allclose(np.load(g), np.load(r), rtol=RTOL, atol=0, equal_nan=True), n
        elif n.endswith(".npz"):
            zg, zr = np.load(g), np.load(r)
            assert sorted(zg.files) == sorted(zr.files), n
            for k in zr.files:
                assert np.allclose(zg[k], zr[k], rtol=RTOL, atol=0, equal_nan=True), (n, k)
        elif n.endswith(".csv"):
            rg, rr = list(csv.DictReader(open(g))), list(csv.DictReader(open(r)))
            _same([{k: (float(v) if k not in ("rung", "quantity") else v) for k, v in d.items()} for d in rg],
                  [{k: (float(v) if k not in ("rung", "quantity") else v) for k, v in d.items()} for d in rr], n)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_driver_reproduces_golden(name, tmp_path):
    driver, argv = SCENARIOS[name]
    out = tmp_path / name; out.mkdir()
    cmd = cli_argv(out) if driver == "cli" else run_argv(argv, out)
    env = dict(os.environ, JAX_ENABLE_X64="1", PYTHONPATH=str(ROOT))
    subprocess.run(cmd, check=True, env=env, cwd=ROOT, capture_output=True)
    _compare(out, GOLD / name)
