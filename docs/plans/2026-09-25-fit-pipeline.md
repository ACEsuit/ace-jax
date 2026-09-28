# Fit Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One library fitting pipeline (`ace_jax.fit.pipeline`) behind both the `ace-jax fit` CLI and the research driver `bench/acegp_cantor/run.py`, plus `ace-jax construct --embedding`, with each driver's current outputs reproduced exactly.

**Architecture:** `run.py` (495 lines, imperative, module-level state) is cut into pipeline stages with explicit inputs and outputs: data → problem → objective → MAP → rungs → prediction/POPS → outputs. A `FitConfig` dataclass carries every option; `fit(config, data)` runs the stages and returns a `FitResult`; `write_outputs` writes the files. `run.py` and `cli.run` become thin argument-parsing drivers that keep their own defaults and file layouts. The one global side effect today (`run.py` monkeypatching `ace_jax.fit.predict.sufficient_statistics`) becomes an explicit `stats=` argument. Golden outputs captured from the *unchanged* drivers in Task 0 pin exact equivalence.

**Tech Stack:** Python 3.11+, JAX (x64), numpy, scipy (L-BFGS-B), numpyro (Adam/SVI, NUTS), existing `ace_jax.fit.*` modules. Tests: pytest, fixtures in `fixtures/` (`si_fitted.npz`, `si_tiny_train.xyz`, `sige_nofit.npz`).

**Spec:** this plan (no separate spec). Decisions agreed in session 2026-09-25:
- the pipeline lives in `ace_jax.fit.pipeline`; `ace-jax fit` and `run.py` are thin wrappers;
- experimental flags (delta floor, pinned rho, routing / sigma-type) stay available to `run.py`; the CLI exposes the stable options;
- `ace-jax construct --embedding <table> --d-max` exposes `construct.model.build_embedding_model`;
- a parity test proves `ace-jax fit` and `run.py` reproduce their pre-refactor results.

**Base:** branch `feat/fit-pipeline`, stacked on `feat/multistart-map` (PR #8; uses `ace_jax.fit.multistart`).

## Global Constraints

- Default behaviour of **both** drivers is unchanged: same numbers (theta_map, draws, metrics, prediction arrays) to `rtol=1e-12` on the golden scenarios, same output files and names.
- Each driver keeps its own defaults (they differ, and that is deliberate): CLI = model `E0`, Adam MAP (`--map-steps 500`), SVI Laplace (`run_laplace`), `--train/--test` files, `metrics.csv`; run.py = least-squares `E0`, L-BFGS MAP (`--map-steps 150`), FD Laplace (`run_laplace_fd`), `--data` split by seeded permutation, `metrics.json` + `pred_*.npz`.
- No new required dependencies. `construct --embedding` needs the `authoring` extra only on a coupling-cache miss (as `construct` today).
- No global state mutation in library code (no monkeypatching modules).
- `pytest -m "not slow"` stays green after every task; commit per task.

## Review Focus

0. **POPS ridge selection must not see its validation configs** (run.py's monkeypatch leaked them into the fit-subset Gram) — pinned in Task 7, golden regenerated in Task 9.
1. **Configs missing E, F or V labels** (Si_tiny's isolated atom has no virial): metrics must drop exactly those rows (CLI's `_metrics` does; run.py's metrics do not) — pinned in Task 7.
2. **`--weights` means two things** (CLI: ACEfit weights *dict*; run.py: weight-factor *list*): the CLI must accept both JSON shapes and route them correctly — pinned in Task 10.
3. **Host cache with an incompatible option** (`--arm linear`, `--density none`, rungs other than map, `--opt adam`): a clear `ValueError` from the pipeline, not a crash deep in JAX — pinned in Task 4.
4. **POPS with the GP arm** (`uq="pops"`, M > 0): clear `ValueError` naming `--arm linear` — pinned in Task 4.
5. **Multi-device (`--devices > 1`) with the host cache**: unsupported combination must be rejected clearly — pinned in Task 4.

---

## File Structure

- Create `src/ace_jax/fit/pipeline/__init__.py` — public API: `FitConfig`, `FitData`, `FitResult`, `fit`, `write_outputs`, `load_fit_data`, `split_configs`.
- Create `src/ace_jax/fit/pipeline/config.py` — `FitConfig` dataclass (all options + validation).
- Create `src/ace_jax/fit/pipeline/data.py` — `FitData`, `split_configs`, `load_fit_data` (labels, weights, baseline / base-npz, E0 modes, datasets).
- Create `src/ace_jax/fit/pipeline/problem.py` — `build_problem` (features, feature map, embedding, inducing, delta floor, prior diagonal).
- Create `src/ace_jax/fit/pipeline/objective.py` — `make_objective` (device / host-cache / sharded LML or LOO; value-and-grad; prediction statistics).
- Create `src/ace_jax/fit/pipeline/mapfit.py` — `fit_map` (sigma-type ParamSet, L-BFGS multi-start with bounds / fix-rho, Adam).
- Create `src/ace_jax/fit/pipeline/rungs.py` — `run_rungs` (laplace fd|svi, pathfinder, vi, nuts).
- Create `src/ace_jax/fit/pipeline/predict.py` — `predict_splits` (BLR/GP mixture or POPS, baseline add-back, metrics robust to missing labels, POPS ridge + envelope).
- Create `src/ace_jax/fit/pipeline/outputs.py` — `write_outputs` (run.py files and/or CLI `metrics.csv`).
- Create `src/ace_jax/fit/pipeline/run.py` — `fit(config, data) -> FitResult` wiring the stages.
- Modify `src/ace_jax/fit/predict.py` — `stats=` argument on `predict_mixture`, `predict_fixed`, `_run_predict`, `PopsRidgePath`.
- Modify `bench/acegp_cantor/run.py` — thin driver.
- Modify `src/ace_jax/cli.py` — `fit` on the pipeline + new flags; `construct --embedding`.
- Create `tests/pipeline_golden/make_golden.py`, `fixtures/pipeline_golden/<scenario>/…` (Task 0).
- Create `tests/test_pipeline_parity.py`, `tests/test_pipeline_units.py`, `tests/test_cli_construct_embedding.py`.

---

### Task 0: Capture golden outputs from the unchanged drivers

**Files:**
- Create: `tests/pipeline_golden/make_golden.py`
- Create: `fixtures/pipeline_golden/` (generated, committed)
- Create: `tests/test_pipeline_parity.py`

**Interfaces:**
- Produces: `fixtures/pipeline_golden/<scenario>/` holding each scenario's `theta_map.json`, `metrics.json` or `metrics.csv`, `draws_*.npy`, `pred_*.npz` (run.py), `map_restarts.json` / `pops_ridge.json` / `pops_envelope_test.npz` where written; `SCENARIOS` dict in `make_golden.py` (name -> (driver, argv)).

- [ ] **Step 1: Write the golden generator** (runs the drivers exactly as they are now)

```python
# tests/pipeline_golden/make_golden.py
"""Record the outputs of the UNCHANGED drivers (run.py and `ace-jax fit`) on the
fixtures, so the pipeline refactor can be checked for exact equivalence.
    uv run --extra gp python tests/pipeline_golden/make_golden.py
Re-run only if a behaviour change is intended (and say so in the commit)."""
import os, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures"
OUT = FIX / "pipeline_golden"
X = str(FIX / "si_tiny_train.xyz")
KEYS = ["--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial"]
RUN = [sys.executable, str(ROOT / "bench/acegp_cantor/run.py"), "--model", str(FIX / "si_fitted.npz"),
       "--data", X, *KEYS, "--r0", "2.35", "--ntrain", "16", "--test-start", "16", "--ntest", "6",
       "--batch", "4", "--no-predict-train"]

SCENARIOS = {
    "run_linear_laplace": ("run", ["--arm", "linear", "--rungs", "map,laplace", "--map-steps", "15",
                                   "--n-draws", "4"]),
    "run_gp_pair_lbfgs": ("run", ["--arm", "gp", "--m-per-species", "6", "--density", "pair",
                                  "--rungs", "map", "--map-steps", "15"]),
    "run_gp_pca_hostcache_restarts": ("run", ["--arm", "gp", "--m-per-species", "6", "--density", "pca",
                                              "--pca-d", "8", "--lml", "host-cache", "--rungs", "map",
                                              "--map-steps", "10", "--map-restarts", "2"]),
    "run_linear_pops_auto": ("run", ["--arm", "linear", "--uq", "pops", "--rungs", "map",
                                     "--map-steps", "10", "--pops-ridge-grid", "1e-3,1e-6",
                                     "--pops-env-nf", "20"]),
    "run_gp_adam_ood": ("run", ["--arm", "gp", "--m-per-species", "6", "--opt", "adam", "--rungs", "map",
                                "--map-steps", "10", "--ood", X]),
    "cli_map_laplace": ("cli", None),     # the trimmed tests/test_gp_cli.py settings
}


def cli_argv(out):
    from ase.io import read, write
    cfgs = read(X, ":")
    tr, te = out / "_train.xyz", out / "_test.xyz"
    write(tr, cfgs[:12]); write(te, cfgs[12:16])
    return [sys.executable, "-m", "ace_jax.cli", "fit", "--model", str(FIX / "si_fitted.npz"),
            "--train", str(tr), "--test", str(te), *KEYS, "--configs-per-batch", "4",
            "--m-per-species", "6", "--rungs", "map,laplace", "--n-draws", "5",
            "--map-steps", "150", "--r0", "2.35", "--out", str(out)]


def main():
    env = dict(os.environ, JAX_ENABLE_X64="1", PYTHONPATH=str(ROOT))
    for name, (driver, argv) in SCENARIOS.items():
        out = OUT / name
        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        cmd = cli_argv(out) if driver == "cli" else [*RUN, *argv, "--out", str(out)]
        subprocess.run(cmd, check=True, env=env, cwd=ROOT)
        for junk in ("timings.json", "_train.xyz", "_test.xyz", "gpu_mem_mib.log"):
            (out / junk).unlink(missing_ok=True)
        print("golden:", name, sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate and inspect**

Run: `uv run --extra gp python tests/pipeline_golden/make_golden.py`
Expected: six `golden: <name> [...]` lines; each run.py scenario lists `config.json`, `draws_map.npy`, `metrics.json`, `pred_test_map.npz`, `split_perm.npy`, `theta_map.json` (+ `map_restarts.json` for L-BFGS, `laplace_info.json`/`draws_laplace.npy`/`pred_test_laplace.npz` for the Laplace scenario, `pops_ridge.json`/`pops_envelope_test.npz` for POPS, `pred_ood_map.npz` for OOD); `cli_map_laplace` lists `config.json`, `draws_laplace.npy`, `draws_map.npy`, `metrics.csv`, `theta_map.json`.

- [ ] **Step 3: Write the parity test (passes now: the drivers are unchanged)**

```python
# tests/test_pipeline_parity.py
"""The drivers must reproduce their pre-refactor outputs exactly (Task 0 goldens)."""
import csv, json, os, pathlib, subprocess, sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLD = ROOT / "fixtures" / "pipeline_golden"
sys.path.insert(0, str(ROOT / "tests" / "pipeline_golden"))
from make_golden import RUN, SCENARIOS, cli_argv  # noqa: E402

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
    cmd = cli_argv(out) if driver == "cli" else [*RUN, *argv, "--out", str(out)]
    env = dict(os.environ, JAX_ENABLE_X64="1", PYTHONPATH=str(ROOT))
    subprocess.run(cmd, check=True, env=env, cwd=ROOT, capture_output=True)
    _compare(out, GOLD / name)
```

- [ ] **Step 4: Run it**

Run: `uv run --extra gp pytest tests/test_pipeline_parity.py -v`
Expected: 6 PASS (unchanged code reproduces itself — this also proves the runs are deterministic on this machine; if any scenario is not, drop it from `SCENARIOS` with a comment rather than loosening `RTOL`).

- [ ] **Step 5: Commit**

```bash
git add tests/pipeline_golden/make_golden.py tests/test_pipeline_parity.py fixtures/pipeline_golden
git commit -m "test: golden outputs of run.py and ace-jax fit before the pipeline refactor"
```

---

### Task 1: Explicit prediction statistics (`stats=`) instead of monkeypatching

**Files:**
- Modify: `src/ace_jax/fit/predict.py` (`_run_predict`, `predict_fixed`, `predict_mixture`, `PopsRidgePath.__init__`, `_run_predict_pops_paper`, `select_pops_ridge`)
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Produces: `predict_mixture(draws, prob, ds_train, ds_test, deriv_dtc=True, stats=None)`, `predict_fixed(..., pops_path=None, stats=None)`, `PopsRidgePath(theta, prob, ds_train, stats=None)` where `stats(theta: Hypers) -> Stats` returns the training sufficient statistics; `None` keeps today's `sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_units.py
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_predict_uses_explicit_stats(tiny_linear_problem):
    from ace_jax.fit.hypers import from_array, to_array
    from ace_jax.fit.predict import predict_mixture
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds = tiny_linear_problem
    theta = prob.prior.mu
    calls = []
    def stats(th):
        calls.append(1)
        return sufficient_statistics(th, prob.spec, prob.model, prob.ind, prob.cfg, ds)
    d = np.asarray(to_array(theta))[None]
    ref = predict_mixture(d, prob, ds, ds)
    got = predict_mixture(d, prob, ds, ds, stats=stats)
    assert calls == [1]
    for f in ref._fields:
        assert np.allclose(np.asarray(getattr(got, f)), np.asarray(getattr(ref, f)), rtol=1e-12)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra gp pytest tests/test_pipeline_units.py::test_predict_uses_explicit_stats -v`
Expected: FAIL with `TypeError: predict_mixture() got an unexpected keyword argument 'stats'`

- [ ] **Step 3: Implement** — thread `stats` through; the default is the current call.

```python
# src/ace_jax/fit/predict.py
def _train_stats(theta, prob, ds_train, stats):
    return (stats(theta) if stats is not None
            else sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train))


def _run_predict(f, theta, prob, ds_train, ds_test, stats=None):
    st = _train_stats(theta, prob, ds_train, stats)
    mu, L = posterior(theta, st, prob)
    outs = [f(theta, mu, L, jax.tree.map(lambda a: a[i], ds_test)) for i in range(ds_test.n_batches)]
    return _pack(outs, prob, ds_test)
```

In `PopsRidgePath.__init__(self, theta, prob, ds_train, stats=None)` replace
`st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)` with
`st = _train_stats(theta, prob, ds_train, stats)`. Add `stats=None` to
`_run_predict_pops_paper(..., path=None, stats=None)` (pass `stats=stats` where it builds
`PopsRidgePath`), to `predict_fixed(..., pops_path=None, stats=None)` (pass to `_run_predict(..., stats=stats)`
and `_run_predict_pops_paper(..., stats=stats)`), to `predict_mixture(draws, prob, ds_train, ds_test,
deriv_dtc=True, stats=None)` (pass to every `_run_predict` call) and to
`select_pops_ridge(..., leverage_pct=0.0, stats=None)` (pass to its `PopsRidgePath`).

- [ ] **Step 4: Run tests**

Run: `uv run --extra gp pytest tests/test_pipeline_units.py tests/test_gp_predict.py tests/test_gp_pops_paper.py tests/test_pipeline_parity.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/predict.py tests/test_pipeline_units.py
git commit -m "refactor(predict): explicit stats= for the training statistics (no monkeypatching)"
```

---

### Task 2: `FitConfig`, `FitData`, `split_configs`, `load_fit_data`

**Files:**
- Create: `src/ace_jax/fit/pipeline/__init__.py`, `src/ace_jax/fit/pipeline/config.py`, `src/ace_jax/fit/pipeline/data.py`
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Produces:
  - `FitConfig` (dataclass, fields listed below; `FitConfig.validate()` raises `ValueError`).
  - `FitData` (NamedTuple): `train, test, ood` (lists of `Config`, labels reduced by the baseline), `train_o, test_o, ood_o` (original labels), `base_train, base_test, base_ood` (per-config `(E, F (n,3), V (3,3))` baselines), `E0` (np.ndarray, per species), `ds_train, ds_test, ds_ood` (datasets; `ds_ood` None if no OOD), `model, meta, z` (from `eval.load`), `perm` (np.ndarray or None).
  - `split_configs(configs, ntrain, ntest, test_start=None, seed=0) -> (train, test, perm)`.
  - `load_fit_data(cfg: FitConfig, *, data=None, train=None, test=None, ood=None) -> FitData` — either `data` (one file split by `split_configs`) or `train`[/`test`] files.

- [ ] **Step 1: Write the failing tests**

```python
def test_split_configs_matches_run_py_permutation():
    from ace_jax.fit.pipeline import split_configs
    cfgs = list(range(30))
    tr, te, perm = split_configs(cfgs, 10, 5, test_start=20, seed=3)
    ref = np.random.default_rng(3).permutation(30)
    assert tr == [cfgs[i] for i in ref[:10]] and te == [cfgs[i] for i in ref[20:25]]
    assert np.array_equal(perm, ref)


def test_fitconfig_rejects_bad_combinations():
    from ace_jax.fit.pipeline import FitConfig
    with pytest.raises(ValueError, match="arm linear"):
        FitConfig(model="m.npz", arm="gp", uq="pops").validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="linear", lml="host-cache").validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="gp", density="none", lml="host-cache").validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map", "laplace")).validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map",),
                  opt="adam").validate()
    with pytest.raises(ValueError, match="devices"):
        FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map",),
                  devices=2).validate()
    FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map",)).validate()
    with pytest.raises(ValueError, match="map_restarts"):
        FitConfig(model="m.npz", map_restarts=0).validate()


def test_load_fit_data_file_split_and_lsq_e0():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy",
                    force_key="dft_force", virial_key="dft_virial", ntrain=16, ntest=6,
                    test_start=16, batch=4, e0="lsq")
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    assert len(d.train) == 16 and len(d.test) == 6 and d.ds_ood is None
    counts = np.array([[np.sum(c.numbers == 14)] for c in d.train], float)
    E0ref, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in d.train]), rcond=None)
    assert np.allclose(d.E0, E0ref)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra gp pytest tests/test_pipeline_units.py -k "split_configs or fitconfig or load_fit_data" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.pipeline'`

- [ ] **Step 3: Implement**

```python
# src/ace_jax/fit/pipeline/config.py
from dataclasses import dataclass, field


@dataclass
class FitConfig:
    """Every option of the fitting pipeline.  Defaults are run.py's; the CLI
    overrides the ones where it differs (see cli.py)."""
    model: str
    arm: str = "gp"                      # "linear" (M = 0) | "gp"
    # data
    energy_key: str = "energy"; force_key: str = "forces"; virial_key: str = "virial"
    ntrain: int = 800; ntest: int = 200; test_start: int | None = None
    seed: int = 0; batch: int = 4
    weights: dict | None = None          # ACEfit weights dict (per config type)
    factors: list | None = None          # ace_jax.fit.weights factor instances
    sigma_type: bool = False             # per-config-type noise block (diagnostic)
    route: dict | None = None            # ParamSet route overrides
    baseline: str | None = None          # dimer_mean.npz (mu_0 subtracted, added back)
    base_npz: str | None = None          # precomputed per-config mu_0 offsets
    e0: str = "lsq"                      # "lsq" (fit on train energies) | "model" (z["E0"])
    # GP
    m_per_species: int = 100
    kernel: str = "cosine"; bump: bool = True
    density: str = "none"                # "none" | "pair" | "pca"
    pca_d: int = 128
    warp: str = "none"
    embedding: str | None = None         # MACE table JSON: frozen coregionalization
    delta_s_floor_q: float | None = None
    fix_rho: str | None = None           # "auto" or a number (L-BFGS only)
    r0: float = 2.5
    # objective
    objective: str = "lml"               # "lml" | "loo"
    lml: str = "device"                  # "device" | "host-cache"
    lml_chunk: int = 64
    devices: int = 1
    # MAP
    opt: str = "lbfgs"                   # "lbfgs" | "adam"
    map_steps: int = 150; map_lr: float = 0.02
    map_restarts: int = 1
    init: dict | None = None             # Hypers field -> value
    # rungs
    rungs: tuple = ("map", "laplace")
    laplace: str = "fd"                  # "fd" (run_laplace_fd) | "svi" (run_laplace)
    n_draws: int = 64
    vi_steps: int = 1000
    nuts_warmup: int = 100; nuts_samples: int = 100; nuts_chains: int = 1
    pf_samples: int = 4; pf_maxiter: int = 10
    # prediction / UQ
    uq: str = "blr"                      # "blr" | "pops"
    deriv_dtc: bool = True
    predict_train: bool = True
    pops_posterior: str = "hypercube"; pops_leverage_pct: float = 0.0
    pops_ridge: object = "auto"          # "auto" | "blr" | float | {"E":..,"F":..,"V":..}
    pops_ridge_grid: tuple = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11,
                              1e-12, 1e-13, 1e-14)
    pops_val_frac: float = 0.2; pops_env_nf: int = 2000

    def validate(self):
        if self.arm not in ("linear", "gp"):
            raise ValueError(f"arm must be 'linear' or 'gp', got {self.arm!r}")
        if self.uq == "pops" and self.arm != "linear":
            raise ValueError("uq='pops' is the linear-arm misspecification predictive: use arm linear")
        if self.lml == "host-cache":
            if (self.arm != "gp" or self.density not in ("pair", "pca") or tuple(self.rungs) != ("map",)
                    or self.opt != "lbfgs"):
                raise ValueError("lml='host-cache' needs arm gp, density pair|pca, rungs ('map',) and "
                                 "opt lbfgs (the cached LML exposes value_and_grad for L-BFGS)")
            if self.devices > 1:
                raise ValueError("lml='host-cache' is single-device: set devices=1")
        if self.map_restarts < 1:
            raise ValueError(f"map_restarts must be >= 1, got {self.map_restarts}")
        if self.fix_rho is not None and self.opt != "lbfgs":
            raise ValueError("fix_rho is implemented for opt lbfgs only")
        return self
```

```python
# src/ace_jax/fit/pipeline/data.py
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ...eval import load
from ..data import build_dataset, load_configs


class FitData(NamedTuple):
    train: list; test: list; ood: list
    train_o: list; test_o: list; ood_o: list
    base_train: list; base_test: list; base_ood: list
    E0: np.ndarray
    ds_train: object; ds_test: object; ds_ood: object
    model: object; meta: dict; z: object
    perm: object


def _zero_base(c):
    return (0.0, np.zeros((len(c.numbers), 3)), np.zeros((3, 3)))


def split_configs(configs, ntrain, ntest, test_start=None, seed=0):
    """run.py's split: a seeded permutation; train = perm[:ntrain], test =
    perm[test_start : test_start + ntest] (test_start defaults to ntrain)."""
    perm = np.random.default_rng(seed).permutation(len(configs))
    ts = ntrain if test_start is None else test_start
    return [configs[i] for i in perm[:ntrain]], [configs[i] for i in perm[ts:ts + ntest]], perm


def _config_type_weights(path):
    """sigma_type: a weight-neutral named-weights dict over the file's config_type
    labels, so load_configs sets each config's type index (run.py behaviour)."""
    from ase.io import read
    cts = []
    for at in read(path, index=":"):
        ct = str(at.info.get("config_type", ""))
        if ct and ct not in cts:
            cts.append(ct)
    return {"default": {"E": 1.0, "F": 1.0, "V": 1.0}, **{ct: {"E": 1.0, "F": 1.0, "V": 1.0} for ct in cts}}


def load_fit_data(cfg, *, data=None, train=None, test=None, ood=None):
    """Configs, baselines, E0 and datasets.  Either `data` (one file, split with
    split_configs) or `train` (+ optional `test`; defaults to train) files."""
    if (data is None) == (train is None):
        raise ValueError("pass exactly one of data= (split) or train= (+ test=)")
    model, meta, z = load(cfg.model)
    keys = dict(energy_key=cfg.energy_key, force_key=cfg.force_key, virial_key=cfg.virial_key)
    if cfg.factors:
        keys["factors"] = cfg.factors
    if cfg.sigma_type:
        keys["weights"] = _config_type_weights(data or train)
    elif cfg.weights is not None:
        keys["weights"] = cfg.weights
    perm = None
    if data is not None:
        train_o, test_o, perm = split_configs(load_configs(data, **keys), cfg.ntrain, cfg.ntest,
                                              cfg.test_start, cfg.seed)
    else:
        train_o = load_configs(train, **keys)
        test_o = load_configs(test, **keys) if test else train_o
    ood_o = load_configs(ood, **keys) if ood else []

    if cfg.baseline:
        from ..baseline import load_mean, subtract_baseline
        mean = load_mean(cfg.baseline)
        tr, base_train = subtract_baseline(train_o, mean)
        te, base_test = subtract_baseline(test_o, mean)
        od, base_ood = subtract_baseline(ood_o, mean) if ood_o else ([], [])
    elif cfg.base_npz:
        if perm is None:
            raise ValueError("base_npz offsets are indexed by the data file: use data=")
        zb = np.load(cfg.base_npz)
        off = np.concatenate([[0], np.cumsum(zb["natoms"])])
        ts = cfg.ntrain if cfg.test_start is None else cfg.test_start
        base = lambda idx: [(float(zb["E"][i]), zb["F"][off[i]:off[i + 1]], zb["V"][i]) for i in idx]
        sub = lambda cs, bs: [c._replace(energy=None if c.energy is None else c.energy - b[0],
                                         forces=None if c.forces is None else c.forces - b[1],
                                         virial=None if c.virial is None else c.virial - b[2])
                              for c, b in zip(cs, bs)]
        base_train, base_test = base(perm[:cfg.ntrain]), base(perm[ts:ts + cfg.ntest])
        tr, te, od = sub(train_o, base_train), sub(test_o, base_test), ood_o
        base_ood = [_zero_base(c) for c in ood_o]
    else:
        tr, te, od = train_o, test_o, ood_o
        base_train = [_zero_base(c) for c in train_o]
        base_test = [_zero_base(c) for c in test_o]
        base_ood = [_zero_base(c) for c in ood_o]

    els = [int(e) for e in meta["elements"]]
    if cfg.e0 == "lsq":
        counts = np.array([[np.sum(c.numbers == e) for e in els] for c in tr], float)
        E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in tr]), rcond=None)
        model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
    elif cfg.e0 == "model":
        E0 = np.asarray(z["E0"])
    else:
        raise ValueError(f"e0 must be 'lsq' or 'model', got {cfg.e0!r}")
    ds_train = build_dataset(tr, meta, E0, cfg.batch)
    ds_test = build_dataset(te, meta, E0, cfg.batch)
    ds_ood = build_dataset(od, meta, E0, cfg.batch) if od else None
    return FitData(tr, te, od, train_o, test_o, ood_o, base_train, base_test, base_ood,
                   np.asarray(E0), ds_train, ds_test, ds_ood, model, meta, z, perm)
```

```python
# src/ace_jax/fit/pipeline/__init__.py
"""The fitting pipeline behind `ace-jax fit` and bench/acegp_cantor/run.py."""
from .config import FitConfig
from .data import FitData, load_fit_data, split_configs

__all__ = ["FitConfig", "FitData", "load_fit_data", "split_configs"]
```

- [ ] **Step 4: Run tests**

Run: `uv run --extra gp pytest tests/test_pipeline_units.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/pipeline tests/test_pipeline_units.py
git commit -m "feat(pipeline): FitConfig (validated) and data stage (split, baselines, E0, datasets)"
```

---

### Task 3: Problem stage (`build_problem`)

**Files:**
- Create: `src/ace_jax/fit/pipeline/problem.py`
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Consumes: `FitConfig`, `FitData` (Task 2).
- Produces: `build_problem(cfg: FitConfig, d: FitData) -> Built` with `Built(NamedTuple): prob, gpcfg, X, S, s_floor, timings: dict`.

- [ ] **Step 1: Failing test**

```python
def test_build_problem_linear_has_no_inducing_and_gp_has_m():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy",
                force_key="dft_force", virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35)
    for arm, m in (("linear", 0), ("gp", 6)):
        cfg = FitConfig(**base, arm=arm, m_per_species=6, density="pair")
        b = build_problem(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")))
        assert b.prob.ind.XM.shape[0] == m
```

- [ ] **Step 2: Verify failure** — `uv run --extra gp pytest tests/test_pipeline_units.py -k build_problem -v` → FAIL `ModuleNotFoundError: ... pipeline.problem`.

- [ ] **Step 3: Implement** (moved verbatim in behaviour from run.py lines 202–230)

```python
# src/ace_jax/fit/pipeline/problem.py
import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ...construct.prior import prior_diagonal
from ...eval import highest_precision
from ..embedding import load_mace_embedding
from ..hypers import default_prior
from ..inducing import GPConfig, build_pmap, descriptor_scale, select_inducing, site_features
from ..kernels import KernelSpec
from ..objective import Problem


class Built(NamedTuple):
    prob: object; gpcfg: object; X: object; S: object; s_floor: object; timings: dict


def build_problem(cfg, d):
    t = time.time()
    meta = d.meta
    els = [int(e) for e in meta["elements"]]
    gpcfg = GPConfig(r0=cfg.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                     NZ=len(els), C=cfg.batch)
    with highest_precision():
        X, S = site_features(d.model, gpcfg, d.ds_train)
        scale = descriptor_scale(X, d.ds_train.node_mask)
        m = cfg.m_per_species if cfg.arm == "gp" else 0
        Pmap = build_pmap(gpcfg, scale, density=None if cfg.density == "none" else cfg.density,
                          d=cfg.pca_d, X=np.asarray(X), mask=np.asarray(d.ds_train.node_mask))
        raw = None if cfg.embedding is None else np.asarray(load_mace_embedding(cfg.embedding, els))
        embed = None if raw is None else jnp.asarray(raw)
        ind = select_inducing(X, S, d.ds_train.node_z, d.ds_train.node_mask, m, scale,
                              Pmap=Pmap, warp=cfg.warp, embed=embed, nz=len(els),
                              de=(int(embed.shape[1]) if embed is not None else None))
        s_floor = None
        if cfg.delta_s_floor_q is not None:
            s_live = np.asarray(S)[np.asarray(d.ds_train.node_mask)]
            s_floor = float(np.quantile(s_live, cfg.delta_s_floor_q))
        prob = Problem(KernelSpec(cfg.kernel, cfg.bump, gpcfg.D, s_floor=s_floor), d.model, ind, gpcfg,
                       jnp.asarray(prior_diagonal(d.z, meta, cfg.model)), default_prior(cfg.r0))
    return Built(prob, gpcfg, X, S, s_floor, {"inducing": time.time() - t})
```

- [ ] **Step 4: Run** — `uv run --extra gp pytest tests/test_pipeline_units.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/ace_jax/fit/pipeline/problem.py tests/test_pipeline_units.py && git commit -m "feat(pipeline): problem stage (features, feature map, embedding, inducing, prior)"`

---

### Task 4: Objective stage (`make_objective`)

**Files:**
- Create: `src/ace_jax/fit/pipeline/objective.py`
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Consumes: `FitConfig`, `FitData`, `Built`.
- Produces: `make_objective(cfg, d, b) -> Objective` with `Objective(NamedTuple): lik (callable a -> LML), vg (a -> (logpost, grad) as jax arrays), stats (theta -> training Stats, for Task 1's stats=), host_cache (HostCachedLML or None), timings: dict)`; `release()` clears the JAX compile caches and garbage-collects; callers drop their own references to the likelihood first (the pre-POPS memory release).

- [ ] **Step 1: Failing tests** (also pins Review Focus 3–5 via `FitConfig.validate`, already covered in Task 2; here the objective values):

```python
def test_objective_device_and_hostcache_agree_on_the_fixture():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, arm="gp",
                m_per_species=6, density="pair", rungs=("map",))
    vals = []
    for lml in ("device", "host-cache"):
        cfg = FitConfig(**base, lml=lml).validate()
        d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
        o = make_objective(cfg, d, build_problem(cfg, d))
        v, g = o.vg(to_array(o.prior_mu))
        vals.append((float(v), np.asarray(g)))
    assert np.isclose(vals[0][0], vals[1][0], rtol=1e-7)
    assert np.allclose(vals[0][1], vals[1][1], rtol=1e-5, atol=1e-6 * np.abs(vals[0][1]).max())
```

- [ ] **Step 2: Verify failure** — `-k objective_device` → FAIL `ModuleNotFoundError: ... pipeline.objective`.

- [ ] **Step 3: Implement** (run.py lines 233–254 + the log-posterior value-and-grad from lines 295–302 + CLI's sharded `make_log_density`)

```python
# src/ace_jax/fit/pipeline/objective.py
import gc
import time
from typing import NamedTuple

import jax
import numpy as np
from jax.sharding import Mesh

from ..hypers import from_array, log_prior, to_array
from ..objective import make_lml, make_log_density
from ..stats import assemble_statistics, linear_statistics, residual_statistics


class Objective(NamedTuple):
    lik: object; vg: object; stats: object; host_cache: object; prior_mu: object; timings: dict


def _pad_to_multiple(ds, n):
    """Append inert copies of the last batch (weights, cfg_mask, node_mask and
    nbr_mask zeroed) so ds.n_batches is a multiple of n -- the sharded
    sufficient statistics need n_batches % n_devices == 0.  Moved verbatim from
    cli.py (which now imports it from here)."""
    import jax.numpy as jnp
    k = (-ds.n_batches) % n
    if k == 0:
        return ds
    last = jax.tree.map(lambda a: a[-1:], ds)
    last = last._replace(w_E=jnp.zeros_like(last.w_E), w_F=jnp.zeros_like(last.w_F),
                         w_V=jnp.zeros_like(last.w_V), cfg_mask=jnp.zeros_like(last.cfg_mask),
                         node_mask=jnp.zeros_like(last.node_mask),
                         nbr_mask=jnp.zeros_like(last.nbr_mask))
    pad = jax.tree.map(lambda a: jnp.concatenate([a] * k, axis=0), last)
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), ds, pad)


def make_objective(cfg, d, b):
    prob, t = b.prob, time.time()
    if cfg.lml == "host-cache":
        from ..hostcache import HostCachedLML
        lik = HostCachedLML(prob, d.ds_train, chunk=cfg.lml_chunk)
        prior_vg = jax.jit(jax.value_and_grad(lambda a: log_prior(from_array(a), prob.prior)))
        def vg(a):
            v, g = lik.value_and_grad(a); pv, pg = prior_vg(a)
            return v + pv, g + pg
        stats = lambda th: assemble_statistics(lik.lin, lik._residual_stats(to_array(th)))
        host = lik
    else:
        mesh = None
        if cfg.devices > 1:
            devs = jax.devices()[:cfg.devices]
            if len(devs) != cfg.devices:
                raise ValueError(f"devices={cfg.devices} requested, only {len(devs)} available")
            mesh = Mesh(np.array(devs), ("data",))
        if mesh is None and cfg.objective == "lml":
            lik = make_lml(prob, d.ds_train, cache_linear=True)
        else:
            ds_fit = _pad_to_multiple(d.ds_train, cfg.devices) if mesh is not None else d.ds_train
            lik = make_log_density(prob, ds_fit, cfg.objective, mesh=mesh).likelihood
        jax.block_until_ready(lik(to_array(prob.prior.mu)))
        logpost = jax.jit(lambda a: lik(a) + log_prior(from_array(a), prob.prior))
        vg = jax.jit(jax.value_and_grad(logpost))
        lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, d.ds_train))()
        stats = lambda th: assemble_statistics(lin, residual_statistics(th, prob.spec, prob.model,
                                                                        prob.ind, prob.cfg, d.ds_train))
        host = None
    return Objective(lik, vg, stats, host, prob.prior.mu, {"stats_once": time.time() - t})


def release():
    """Clear JAX's compile caches and collect: once the caller has dropped its
    references to the likelihood, this frees its Gram-sized buffers (POPS then
    needs only the linear statistics; on the lossless Cantor base the LML holds
    ~150 GB and starved POPS's kernels)."""
    jax.clear_caches(); gc.collect()
```

Note for the implementer: CLI's current `lik` is `make_log_density(prob, ds_fit, objective, mesh).likelihood` in **all** cases. The single-device `objective == "lml"` branch above uses `make_lml(..., cache_linear=True)` (run.py's) — `make_log_density` with `mesh=None`, `objective="lml"` calls the same `make_lml(prob, ds, None, True)`, so the CLI golden (Task 0) must still match; the parity test in Task 10 is the check.

- [ ] **Step 4: Run** — `uv run --extra gp pytest tests/test_pipeline_units.py -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(pipeline): objective stage (device / host-cache / sharded; value_and_grad; stats)"` (after `git add src/ace_jax/fit/pipeline/objective.py`).

---

### Task 5: MAP stage (`fit_map`)

**Files:**
- Create: `src/ace_jax/fit/pipeline/mapfit.py`
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Consumes: `FitConfig`, `FitData`, `Built`, `Objective`; `ace_jax.fit.multistart.{multistart_map, prior_starts}` (PR #8).
- Produces: `fit_map(cfg, d, b, obj, log=print) -> MapFit` with `MapFit(NamedTuple): theta (Hypers), restarts (list of dicts as run.py's map_restarts.json, or None), sigma_type_ratios (ndarray or None), timings`. Constants `LBFGS_LO`, `LBFGS_HI` (np arrays, run.py's bounds).

- [ ] **Step 1: Failing test**

```python
def test_fit_map_restarts_one_equals_single_lbfgs():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, arm="linear",
                rungs=("map",), map_steps=8)
    thetas = []
    for n in (1, 2):
        cfg = FitConfig(**base, map_restarts=n)
        d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
        b = build_problem(cfg, d)
        r = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a, **k: None)
        thetas.append(r)
    assert len(thetas[0].restarts) == 1 and len(thetas[1].restarts) == 2
    assert thetas[1].restarts[0]["x"] == thetas[0].restarts[0]["x"]      # start 0 identical
```

- [ ] **Step 2: Verify failure** — `-k fit_map` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement** (run.py lines 256–342, logging through `log`)

```python
# src/ace_jax/fit/pipeline/mapfit.py
import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..hypers import Hypers, from_array, to_array
from ..ladder import run_map
from ..multistart import multistart_map, prior_starts

# log-space boxes: generous, but keep the Cholesky away from sigma -> 0; A up to
# 1e3 (the full/PCA-descriptor GP sat at the old bound of 10, Cantor-1k)
LBFGS_LO = np.log([0.05, 1e-3, 0.1, 1.5, 1e-3, 0.1, 1e-2, 1e-4, 1e-4, 1e-4])
LBFGS_HI = np.log([50.0, 1e3, 50.0, 4.0, 50.0, 100.0, 1e4, 10.0, 10.0, 10.0])


class MapFit(NamedTuple):
    theta: object; restarts: object; sigma_type_ratios: object; timings: dict


def _rho_fix(cfg, prob):
    if cfg.fix_rho == "auto":
        XM = np.asarray(prob.ind.XM)
        d2 = ((XM[:, None, :] - XM[None, :, :]) ** 2).mean(-1)
        np.fill_diagonal(d2, np.inf)
        return float(np.median(np.sqrt(d2.min(1))))
    return float(cfg.fix_rho)


def fit_map(cfg, d, b, obj, log=print):
    prob, t = b.prob, time.time()
    init = None if cfg.init is None else Hypers(**cfg.init)
    if cfg.sigma_type:
        from ..ladder import run_map_ps
        from ..paramset import build_fit_paramset
        n_types = int(np.asarray(d.ds_train.cfg_type).max()) + 1
        ps0 = build_fit_paramset(init or prob.prior.mu, prob.prior, n_types=n_types,
                                 sigma_type=True, route=cfg.route)
        ps = run_map_ps(ps0, prob, d.ds_train, steps=cfg.map_steps, lr=cfg.map_lr, seed=cfg.seed)
        ratios = ps.sigma_type_ratios()
        return MapFit(from_array(ps.block("hypers").value), None,
                      None if ratios is None else np.asarray(ratios), {"map": time.time() - t})
    if cfg.opt == "adam":
        theta = run_map(obj.lik, prob.prior, steps=cfg.map_steps, lr=cfg.map_lr, seed=cfg.seed, init=init)
        return MapFit(theta, None, None, {"map": time.time() - t})
    x0 = np.asarray(to_array(init or prob.prior.mu), float)
    lo, hi = LBFGS_LO.copy(), LBFGS_HI.copy()
    if cfg.fix_rho is not None:
        lo[5] = hi[5] = x0[5] = np.log(_rho_fix(cfg, prob))
        log(f"fix-rho: rho pinned at {float(np.exp(lo[5])):.4f}")
    tick = [time.time()]
    def vg_host(x):
        v, g = obj.vg(jnp.asarray(x)); g.block_until_ready()
        return float(v), np.asarray(g, float)
    def log_eval(k, i, v):
        log(f"  lbfgs start {k} eval {i}  logpost = {v:.6g}  ({time.time() - tick[0]:.0f} s)")
        tick[0] = time.time()
    starts = prior_starts(prob.prior, cfg.map_restarts, lo, hi, x0, seed=cfg.seed)
    best, runs = multistart_map(vg_host, starts, lo, hi, cfg.map_steps, log=log_eval)
    for r in runs:
        log(f"L-BFGS start {r['start']}: logpost {r['value']:.6g}  nfev {r['nfev']}  {r['message']}")
    restarts = [{"start": r["start"], "logpost": r["value"], "nfev": r["nfev"], "message": r["message"],
                 "x0": r["x0"].tolist(), "x": r["x"].tolist()} for r in runs]
    log(f"L-BFGS: best of {len(runs)} start(s) = start {best['start']}, logpost {best['value']:.6g}")
    return MapFit(Hypers(*[float(v) for v in best["x"]]), restarts, None, {"map": time.time() - t})
```

- [ ] **Step 4: Run** — `uv run --extra gp pytest tests/test_pipeline_units.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/ace_jax/fit/pipeline/mapfit.py tests/test_pipeline_units.py && git commit -m "feat(pipeline): MAP stage (sigma-type, L-BFGS multi-start with bounds/fix-rho, Adam)"`

---

### Task 6: Rungs stage (`run_rungs`)

**Files:**
- Create: `src/ace_jax/fit/pipeline/rungs.py`
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Produces: `run_rungs(cfg, b, obj, theta, log=print) -> Rungs` with `Rungs(NamedTuple): draws (dict rung -> (n, 10) ndarray; always includes "map"), info (dict: "laplace" -> laplace_info dict for fd, "nuts" -> summary), timings`.

- [ ] **Step 1: Failing test**

```python
def test_rungs_map_is_theta_and_unknown_rung_raises():
    from ace_jax.fit.hypers import default_prior, to_array
    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.rungs import run_rungs
    th = default_prior(2.35).mu
    r = run_rungs(FitConfig(model="m", rungs=("map",)), None, None, th, log=lambda *a: None)
    assert np.array_equal(r.draws["map"], np.asarray(to_array(th))[None])
    with pytest.raises(ValueError, match="unknown rung"):
        run_rungs(FitConfig(model="m", rungs=("map", "hmc")), None, None, th, log=lambda *a: None)
```

- [ ] **Step 2: Verify failure** — `-k rungs_map` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement** (run.py lines 344–367; CLI's SVI Laplace as `laplace="svi"`)

```python
# src/ace_jax/fit/pipeline/rungs.py
import time
from typing import NamedTuple

import numpy as np

from ..hypers import to_array

KNOWN = ("map", "laplace", "pathfinder", "vi", "nuts")


class Rungs(NamedTuple):
    draws: dict; info: dict; timings: dict


def run_rungs(cfg, b, obj, theta, log=print):
    bad = [r for r in cfg.rungs if r not in KNOWN]
    if bad:
        raise ValueError(f"unknown rung(s) {bad}; known: {list(KNOWN)}")
    from ..ladder import run_laplace, run_laplace_fd, run_nuts, run_pathfinder, run_vi
    draws, info, tm = {"map": np.asarray(to_array(theta))[None]}, {}, {}
    prior = None if b is None else b.prob.prior
    if "laplace" in cfg.rungs:
        t = time.time()
        if cfg.laplace == "fd":
            draws["laplace"], info["laplace"] = run_laplace_fd(obj.lik, prior, theta, n_draws=cfg.n_draws,
                                                               seed=cfg.seed)
        else:
            draws["laplace"], _ = run_laplace(obj.lik, prior, n_draws=cfg.n_draws, steps=cfg.map_steps,
                                              seed=cfg.seed, init=theta)
        tm["laplace"] = time.time() - t
    if "pathfinder" in cfg.rungs:
        t = time.time()
        draws["pathfinder"], info["pathfinder"] = run_pathfinder(obj.lik, prior, theta, n_draws=cfg.n_draws,
                                                                 seed=cfg.seed, num_samples=cfg.pf_samples,
                                                                 maxiter=cfg.pf_maxiter)
        tm["pathfinder"] = time.time() - t
    if "vi" in cfg.rungs:
        t = time.time()
        draws["vi"], _ = run_vi(obj.lik, prior, n_draws=cfg.n_draws, steps=cfg.vi_steps, seed=cfg.seed,
                                init=theta)
        tm["vi"] = time.time() - t
    if "nuts" in cfg.rungs:
        t = time.time()
        draws["nuts"], info["nuts"] = run_nuts(obj.lik, prior, num_warmup=cfg.nuts_warmup,
                                               num_samples=cfg.nuts_samples, num_chains=cfg.nuts_chains,
                                               seed=cfg.seed, init=theta)
        tm["nuts"] = time.time() - t
    if "map" not in cfg.rungs:
        draws.pop("map")
    return Rungs(draws, info, tm)
```

The CLI puts `"map"` in `draws` only when requested (`if "map" in rungs`), run.py always; `draws.pop("map")` above reproduces the CLI, and run.py's driver (Task 9) always passes `"map"` in `rungs`.

- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add src/ace_jax/fit/pipeline/rungs.py tests/test_pipeline_units.py && git commit -m "feat(pipeline): rungs stage (laplace fd|svi, pathfinder, vi, nuts)"`

---

### Task 7: Prediction stage (`predict_splits`) with POPS and robust metrics

**Files:**
- Create: `src/ace_jax/fit/pipeline/predict.py`
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Consumes: `FitConfig`, `FitData`, `Built`, `Objective.stats`, `MapFit.theta`, `Rungs.draws`; Task 1's `stats=`.
- Produces: `predict_splits(cfg, d, b, stats, theta, draws, log=print) -> Preds` with `Preds(NamedTuple): arrays (dict "split/rung" -> dict of npz arrays as run.py's pred_*.npz), metrics (dict "split/rung" -> {"E","F","V"} summaries), pops (dict: "ridge", "ridge_scores", "envelope" (npz dict), "envelope_metrics"), timings`. `label_metrics(E, E_mean, E_var, ..., nat, has)` helper.

- [ ] **Step 1: Failing test** (Review Focus 1: missing labels)

```python
def test_metrics_drop_configs_without_labels():
    from ace_jax.fit.pipeline.predict import label_metrics
    nat = np.array([1, 2, 2])
    E = np.array([np.nan, -1.0, -2.0]); Em = np.array([5.0, -1.1, -2.1]); Ev = np.full(3, 0.01)
    F = np.zeros((5, 3)); Fm = np.zeros((5, 3)) + 0.1; Fv = np.full((5, 3), 0.01)
    V = np.full((3, 6), np.nan); V[1:] = 0.0
    Vm = np.zeros((3, 6)); Vv = np.full((3, 6), 0.01)
    m = label_metrics(E, Em, Ev, F, Fm, Fv, V, Vm, Vv, nat)
    assert np.isfinite(m["E"]["rmse"]) and np.isfinite(m["V"]["rmse"])
    assert np.isclose(m["E"]["rmse"], 1e3 * np.sqrt(np.mean(((E[1:] - Em[1:]) / nat[1:]) ** 2)))
```

- [ ] **Step 2: Verify failure** — `-k metrics_drop` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement** (run.py lines 369–488; labels absent are carried as NaN and masked)

```python
# src/ace_jax/fit/pipeline/predict.py
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..data import build_dataset
from ..metrics import summarise
from ..predict import POPS_MEAN, PopsRidgePath, predict_fixed, predict_mixture, select_pops_ridge

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


class Preds(NamedTuple):
    arrays: dict; metrics: dict; pops: dict; timings: dict


def label_metrics(E, Em, Ev, F, Fm, Fv, V, Vm, Vv, nat):
    """Summaries on the observed labels only: NaN rows (no label) are dropped."""
    e = np.isfinite(E); f = np.isfinite(F).all(1); v = np.isfinite(V).all(1)
    return {"E": summarise(1e3 * E[e] / nat[e], 1e3 * Em[e] / nat[e], 1e3 * np.sqrt(Ev[e]) / nat[e]),
            "F": summarise(F[f].reshape(-1), Fm[f].reshape(-1), np.sqrt(Fv[f]).reshape(-1)),
            "V": summarise(V[v].reshape(-1), Vm[v].reshape(-1), np.sqrt(Vv[v]).reshape(-1))}


def _labels(cfgs):
    E = np.array([np.nan if c.energy is None else c.energy for c in cfgs])
    F = np.concatenate([np.full((len(c.numbers), 3), np.nan) if c.forces is None else c.forces for c in cfgs])
    V = np.array([np.full(6, np.nan) if c.virial is None else [c.virial[i, j] for i, j in VOIGT]
                  for c in cfgs])
    return E, F, V


def _pops_setup(cfg, d, b, stats, theta, log):
    """Ridge (auto: selected on a train hold-out), the BLR-mean path, and the test
    envelope -- run.py's POPS block verbatim in behaviour."""
    from ..rows import linear_rows
    prob, out = b.prob, {}
    if cfg.pops_ridge == "auto":
        nval = max(1, int(cfg.pops_val_frac * len(d.train)))
        ds_fit = build_dataset(d.train[:-nval], d.meta, d.E0, cfg.batch)
        ds_val = build_dataset(d.train[-nval:], d.meta, d.E0, cfg.batch)
        ridge, scores = select_pops_ridge(theta, prob, ds_fit, ds_val, list(cfg.pops_ridge_grid),
                                          form=cfg.pops_posterior, leverage_pct=cfg.pops_leverage_pct)
        out["ridge_scores"] = {"grid": list(cfg.pops_ridge_grid), "ridge": ridge, "n_val": nval,
                               "scores_crps": {q: [float(x) for x in v] for q, v in scores.items()}}
    else:
        ridge = cfg.pops_ridge
    out["ridge"] = ridge
    log(f"POPS (paper) ridge: {ridge}")
    rd = ridge if isinstance(ridge, dict) else {q: ridge for q in "EFV"}
    path = PopsRidgePath(theta, prob, d.ds_train, stats=stats)
    path.use_mean(POPS_MEAN)
    cst = np.asarray(path.c_star)
    rowsE, rowsF = ([], [], []), ([], [])
    for i in range(d.ds_test.n_batches):
        bt = jax.tree.map(lambda x_: x_[i], d.ds_test)
        lin, _, _ = linear_rows(prob.model, prob.cfg, bt)
        Lb, C = lin.E.shape[-1], bt.y_E.shape[0]
        nat_b = np.zeros(C + 1); np.add.at(nat_b, np.asarray(bt.node_cfg), np.asarray(bt.node_mask, float))
        kE = np.asarray(bt.w_E) > 0
        phE = np.asarray(lin.E)[kE]
        rowsE[0].append(phE); rowsE[1].append(np.asarray(bt.y_E)[kE] - phE @ cst); rowsE[2].append(nat_b[:C][kE])
        kF = np.repeat(np.asarray(bt.w_F) > 0, 3)
        phF = np.asarray(lin.F).reshape(-1, Lb)[kF]
        rowsF[0].append(phF); rowsF[1].append(np.asarray(bt.y_F).reshape(-1)[kF] - phF @ cst)
    phE, rE, natE = (np.concatenate(x_) for x_ in rowsE)
    phF, rF = (np.concatenate(x_) for x_ in rowsF)
    sel = np.random.default_rng(cfg.seed).choice(len(rF), size=min(cfg.pops_env_nf, len(rF)), replace=False)
    phF, rF = phF[np.sort(sel)], rF[np.sort(sel)]
    env, env_m = {}, {}
    for q, ph, r, sc in (("E", phE, rE, natE), ("F", phF, rF, np.ones(len(rF)))):
        lo, hi = path.envelope(jnp.asarray(ph), rd[q], cfg.pops_leverage_pct)
        lo, hi = np.asarray(lo), np.asarray(hi)
        unit = 1e3 if q == "E" else 1.0
        env_m[q] = {"env_cover": float(np.mean((lo <= r) & (r <= hi))),
                    "env_width_median": float(np.median(unit * (hi - lo) / sc)), "env_n": int(len(r))}
        env.update({f"{q}_lo": lo / sc * unit, f"{q}_hi": hi / sc * unit, f"{q}_resid": r / sc * unit})
    env["F_index"] = np.sort(sel)
    out.update(envelope=env, envelope_metrics=env_m, path=path)
    return out


def predict_splits(cfg, d, b, stats, theta, draws, log=print):
    prob, arrays, metrics, tm = b.prob, {}, {}, {}
    pops = {}
    if cfg.uq == "pops":
        t = time.time()
        pops = _pops_setup(cfg, d, b, stats, theta, log)
        tm["pops_paper_setup"] = time.time() - t
    splits = [("test", d.test_o, d.ds_test, d.base_test)]
    if cfg.predict_train:
        splits.append(("train", d.train_o, d.ds_train, d.base_train))
    if d.ds_ood is not None:
        splits.append(("ood", d.ood_o, d.ds_ood, d.base_ood))
    for rung, dr in draws.items():
        sub = dr if len(dr) <= cfg.n_draws else dr[np.linspace(0, len(dr) - 1, cfg.n_draws).astype(int)]
        for split, cfgs, ds, base in splits:
            t = time.time()
            if cfg.uq == "pops":
                pred = predict_fixed(theta, prob, d.ds_train, ds, deriv_dtc=cfg.deriv_dtc, uq="pops",
                                     pops_form=cfg.pops_posterior, leverage_pct=cfg.pops_leverage_pct,
                                     pops_ridge=pops["ridge"], pops_path=pops["path"], stats=stats)
            else:
                pred = predict_mixture(sub, prob, d.ds_train, ds, deriv_dtc=cfg.deriv_dtc, stats=stats)
            tm[f"predict_{split}_{rung}"] = time.time() - t
            nat = np.array([len(c.numbers) for c in cfgs])
            bE = np.array([x[0] for x in base])
            bF = np.concatenate([x[1] for x in base]) if base else np.zeros((0, 3))
            bV6 = np.array([[x[2][i, j] for i, j in VOIGT] for x in base])
            E, F, V = _labels(cfgs)
            Em = np.asarray(pred.E_mean) + bE; Fm = np.asarray(pred.F_mean) + bF
            Vm = np.asarray(pred.V_mean) + bV6
            s2 = {k: float(np.mean(np.exp(2 * sub[:, i]))) for k, i in (("E", 7), ("F", 8), ("V", 9))}
            arrays[f"{split}/{rung}"] = dict(
                nat=nat, E=E, E_mean=Em, E_var=np.asarray(pred.E_var), F=F, F_mean=Fm,
                F_var=np.asarray(pred.F_var), V=V, V_mean=Vm, V_var=np.asarray(pred.V_var),
                noise_E=s2["E"] * nat, noise_F=np.full(len(F), s2["F"]), noise_V=s2["V"] * nat)
            m = label_metrics(E, Em, np.asarray(pred.E_var), F, Fm, np.asarray(pred.F_var),
                              V, Vm, np.asarray(pred.V_var), nat)
            if split == "test":
                for q, v in pops.get("envelope_metrics", {}).items():
                    m[q].update(v)
            metrics[f"{split}/{rung}"] = m
            log(f"{split} {rung} " + str({q: {k: round(x, 4) for k, x in v.items()
                                              if k in ("rmse", "crps", "coverage", "rho", "rms_z")}
                                          for q, v in m.items()}))
    pops.pop("path", None)
    return Preds(arrays, metrics, pops, tm)
```

**Deliberate behaviour change (POPS ridge selection).** run.py's monkeypatched
`sufficient_statistics` ignored its `ds` argument and always returned the *full
training set's* linear statistics, so `select_pops_ridge` -- which builds its path on
the fit subset `train[:-nval]` -- was scored with the held-out validation configs
inside the Gram (validation leakage).  The pipeline calls `select_pops_ridge` without
`stats=`, i.e. on the fit subset's own statistics, which is what the hold-out means.
Pin it with this test (add to `tests/test_pipeline_units.py`):

```python
def test_pops_ridge_selection_uses_the_fit_subset_statistics(tiny_linear_problem, monkeypatch):
    import ace_jax.fit.predict as P
    prob, ds = tiny_linear_problem
    seen = []
    real = P.sufficient_statistics
    def spy(theta, spec, model, ind, cfg, dsx):
        seen.append(int(dsx.n_batches)); return real(theta, spec, model, ind, cfg, dsx)
    monkeypatch.setattr(P, "sufficient_statistics", spy)
    fit_ds = jax.tree.map(lambda a: a[:1], ds)
    P.select_pops_ridge(prob.prior.mu, prob, fit_ds, ds, [1e-3])
    assert seen and set(seen) == {1}            # the fit subset (1 batch), never the full set
```

and regenerate that one golden with the fix, saying so in the commit:

```bash
uv run --extra gp python -c "import sys; sys.path.insert(0, 'tests/pipeline_golden'); \
  import make_golden as g; g.SCENARIOS = {'run_linear_pops_auto': g.SCENARIOS['run_linear_pops_auto']}; g.main()"
```

(run it after Task 9, when run.py uses the pipeline; until then keep the parity test for
`run_linear_pops_auto` expected-failing with `pytest.mark.xfail(reason="POPS ridge
selection leak fixed in the pipeline; golden regenerated in Task 9")`.)

Implementer note: on configs that **do** carry every label, `label_metrics` gives exactly run.py's numbers (the masks are all-true), so the run.py goldens still pass; where labels are missing run.py computed NaN summaries, the CLI dropped them — the pipeline drops them (CLI behaviour).

- [ ] **Step 4: Run** — `uv run --extra gp pytest tests/test_pipeline_units.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/ace_jax/fit/pipeline/predict.py tests/test_pipeline_units.py && git commit -m "feat(pipeline): prediction stage (mixture or POPS, baseline add-back, label-robust metrics)"`

---

### Task 8: `fit()` and `write_outputs`

**Files:**
- Create: `src/ace_jax/fit/pipeline/run.py`, `src/ace_jax/fit/pipeline/outputs.py`
- Modify: `src/ace_jax/fit/pipeline/__init__.py`
- Test: `tests/test_pipeline_units.py`

**Interfaces:**
- Produces: `fit(cfg: FitConfig, data: FitData, log=print) -> FitResult`; `FitResult(NamedTuple): config, data, built, theta, map (MapFit), rungs (Rungs), preds (Preds), timings (dict)`; `write_outputs(res: FitResult, out, layout=("run",), argv=None)` where `layout` ⊆ {"run", "cli"}: "run" writes run.py's files (split_perm.npy when data was split, theta_map.json, map_restarts.json, sigma_type_ratios.json, laplace_info.json, nuts_summary.json, draws_<rung>.npy, pred_<split>_<rung>.npz, pops_ridge.json, pops_envelope_test.npz, metrics.json, timings.json, config.json); "cli" writes theta_map.json, draws_<rung>.npy, nuts_summary.json, metrics.csv (test split, rows rung, quantity, summary keys), config.json (CLI keys: argv + M, len_basis, n_train, n_test).

- [ ] **Step 1: Failing test**

```python
def test_fit_end_to_end_writes_run_layout(tmp_path):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, write_outputs
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, arm="linear",
                    rungs=("map",), map_steps=5, predict_train=False)
    res = fit(cfg.validate(), load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")),
              log=lambda *a: None)
    write_outputs(res, tmp_path, layout=("run", "cli"), argv={"arm": "linear"})
    for n in ("theta_map.json", "metrics.json", "metrics.csv", "pred_test_map.npz", "draws_map.npy",
              "map_restarts.json", "timings.json", "config.json", "split_perm.npy"):
        assert (tmp_path / n).exists(), n
```

- [ ] **Step 2: Verify failure** — `-k end_to_end` → FAIL `ImportError: cannot import name 'fit'`.

- [ ] **Step 3: Implement**

```python
# src/ace_jax/fit/pipeline/run.py
import time
from typing import NamedTuple

from ...eval import highest_precision
from .mapfit import fit_map
from .objective import make_objective, release
from .predict import predict_splits
from .problem import build_problem
from .rungs import run_rungs


class FitResult(NamedTuple):
    config: object; data: object; built: object; theta: object
    map: object; rungs: object; preds: object; timings: dict


def fit(cfg, data, log=print):
    T0 = time.time()
    cfg.validate()
    b = build_problem(cfg, data)
    with highest_precision():
        obj = make_objective(cfg, data, b)
        mf = fit_map(cfg, data, b, obj, log=log)
        rg = run_rungs(cfg, b, obj, mf.theta, log=log)
        stats = obj.stats
        if cfg.uq == "pops":
            # POPS needs only the linear statistics: free the LML's Gram-sized buffers first
            from ..stats import assemble_statistics, linear_statistics, residual_statistics
            import jax
            prob, ds = b.prob, data.ds_train
            obj = obj._replace(lik=None, vg=None, stats=None, host_cache=None)
            release()
            lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, ds))()
            stats = lambda th: assemble_statistics(lin, residual_statistics(th, prob.spec, prob.model,
                                                                            prob.ind, prob.cfg, ds))
        pr = predict_splits(cfg, data, b, stats, mf.theta, rg.draws, log=log)
    tm = {**b.timings, **obj.timings, **mf.timings, **rg.timings, **pr.timings, "total": time.time() - T0}
    return FitResult(cfg, data, b, mf.theta, mf, rg, pr, tm)
```

```python
# src/ace_jax/fit/pipeline/outputs.py
import csv
import dataclasses
import json
import pathlib

import numpy as np


def _dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def write_outputs(res, out, layout=("run",), argv=None):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg, d, b = res.config, res.data, res.built
    _dump(out / "theta_map.json", res.theta._asdict())
    for rung, dr in res.rungs.draws.items():
        np.save(out / f"draws_{rung}.npy", dr)
    if "nuts" in res.rungs.info:
        _dump(out / "nuts_summary.json", res.rungs.info["nuts"])
    if "run" in layout:
        if d.perm is not None:
            np.save(out / "split_perm.npy", d.perm)
        if res.map.restarts is not None:
            _dump(out / "map_restarts.json", res.map.restarts)
        if res.map.sigma_type_ratios is not None:
            _dump(out / "sigma_type_ratios.json", np.asarray(res.map.sigma_type_ratios).tolist())
        if "laplace" in res.rungs.info:
            _dump(out / "laplace_info.json", res.rungs.info["laplace"])
        for key, arr in res.preds.arrays.items():
            split, rung = key.split("/")
            np.savez(out / f"pred_{split}_{rung}.npz", **arr)
        if "ridge_scores" in res.preds.pops:
            _dump(out / "pops_ridge.json", res.preds.pops["ridge_scores"])
        if "envelope" in res.preds.pops:
            np.savez(out / "pops_envelope_test.npz", **res.preds.pops["envelope"])
        _dump(out / "metrics.json", res.preds.metrics)
        _dump(out / "timings.json", res.timings)
        els = [int(e) for e in d.meta["elements"]]
        _dump(out / "config.json", {**(argv or dataclasses.asdict(cfg)), "M": int(b.prob.ind.XM.shape[0]),
                                    "len_basis": b.gpcfg.len_basis,
                                    "E0": dict(zip(map(str, els), map(float, d.E0)))})
    if "cli" in layout:
        rows = [{"rung": key.split("/")[1], "quantity": q, **m}
                for key, per_q in res.preds.metrics.items() if key.startswith("test/")
                for q, m in per_q.items()]
        with open(out / "metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        if "run" not in layout:
            _dump(out / "config.json", {**(argv or {}), "M": int(b.prob.ind.XM.shape[0]),
                                        "len_basis": b.gpcfg.len_basis, "n_train": len(d.train),
                                        "n_test": len(d.test)})
```

Add to `__init__.py`: `from .run import FitResult, fit` and `from .outputs import write_outputs`, extending `__all__`.

Implementer note: the CLI's metrics.csv rows must carry only `summarise`'s keys (no POPS envelope keys) — the CLI never runs POPS, so rows are unchanged; the Task 10 parity test is the check.

- [ ] **Step 4: Run** — `uv run --extra gp pytest tests/test_pipeline_units.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/ace_jax/fit/pipeline && git commit -m "feat(pipeline): fit() and write_outputs (run.py and CLI layouts)"`

---

### Task 9: `run.py` on the pipeline

**Files:**
- Modify: `bench/acegp_cantor/run.py` (replace the body after argument parsing)
- Test: `tests/test_pipeline_parity.py` (run.py scenarios)

**Interfaces:**
- Consumes: `FitConfig`, `load_fit_data`, `fit`, `write_outputs`.

- [ ] **Step 1: Confirm the parity tests pass before the change** — `uv run --extra gp pytest tests/test_pipeline_parity.py -k run_ -q` → 5 PASS.

- [ ] **Step 2: Replace run.py's pipeline with the library** — keep every `p.add_argument` exactly (help text included); after `a = p.parse_args()` the whole file becomes:

```python
from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, write_outputs
from ace_jax.fit.paramset import parse_route
from ace_jax.fit.weights import ConfigType, PerConfig, Quantity, Structural

_FACTOR_CLASSES = {"Structural": Structural, "Quantity": Quantity,
                   "ConfigType": ConfigType, "PerConfig": PerConfig}
factors = None
if a.weights:
    factors = [_FACTOR_CLASSES[name](**kw) for entry in json.loads(a.weights) for name, kw in entry.items()]
    print("weight factors:", [type(f).__name__ for f in factors], flush=True)
if a.pops_ridge in ("auto", "blr"):
    ridge = a.pops_ridge
elif "=" in a.pops_ridge:
    ridge = {k.strip(): (v.strip() if v.strip() == "blr" else float(v))
             for k, v in (kv.split("=") for kv in a.pops_ridge.split(","))}
    if set(ridge) != set("EFV"):
        p.error("--pops-ridge per-quantity form needs E=,F=,V=")
else:
    ridge = float(a.pops_ridge)
cfg = FitConfig(
    model=a.model, arm=a.arm, energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key,
    ntrain=a.ntrain, ntest=a.ntest, test_start=a.test_start, seed=a.seed, batch=a.batch,
    factors=factors, sigma_type=a.sigma_type, route=parse_route(a.route), baseline=a.baseline,
    base_npz=a.base_npz, e0="lsq", m_per_species=a.m_per_species, kernel=a.kernel, bump=not a.no_bump,
    density=a.density, pca_d=a.pca_d, warp=a.warp, embedding=a.embedding,
    delta_s_floor_q=a.delta_s_floor_q, fix_rho=a.fix_rho, r0=a.r0, lml=a.lml, lml_chunk=a.lml_chunk,
    opt=a.opt, map_steps=a.map_steps, map_lr=a.map_lr, map_restarts=a.map_restarts,
    init=json.load(open(a.init)) if a.init else None,
    rungs=tuple(dict.fromkeys(["map", *[r.strip() for r in a.rungs.split(",")]])),
    laplace="fd", n_draws=a.n_draws, vi_steps=a.vi_steps, nuts_warmup=a.nuts_warmup,
    nuts_samples=a.nuts_samples, nuts_chains=a.nuts_chains, pf_samples=a.pf_samples,
    pf_maxiter=a.pf_maxiter, uq=a.uq, deriv_dtc=not a.no_deriv_dtc, predict_train=not a.no_predict_train,
    pops_posterior=a.pops_posterior, pops_leverage_pct=a.pops_leverage_pct, pops_ridge=ridge,
    pops_ridge_grid=tuple(float(x) for x in a.pops_ridge_grid.split(",")),
    pops_val_frac=a.pops_val_frac, pops_env_nf=a.pops_env_nf)
try:
    cfg.validate()
except ValueError as e:
    p.error(str(e))
data = load_fit_data(cfg, data=a.data, ood=a.ood)
res = fit(cfg, data, log=lambda *s: print(*s, flush=True))
write_outputs(res, a.out, layout=("run",), argv=vars(a))
print("done", {k: round(v, 1) for k, v in res.timings.items()}, flush=True)
```

and remove the now-unused imports at the top (keep `argparse, json`).

- [ ] **Step 3: Run parity**

Run: `uv run --extra gp pytest tests/test_pipeline_parity.py -k run_ -v`
Expected: 4 PASS and `run_linear_pops_auto` XFAIL (the ridge-selection leak fix, Task 7). If any
other scenario differs, compare the first differing file (`_compare` names it) and fix the
stage that produces it; do not regenerate those goldens.

- [ ] **Step 3b: Regenerate the POPS golden with the fix and drop the xfail**

Run the one-scenario regeneration from Task 7, remove the `xfail` mark, and confirm
`uv run --extra gp pytest tests/test_pipeline_parity.py -v` → 6 PASS.  Commit the golden
separately: `git add fixtures/pipeline_golden/run_linear_pops_auto tests/test_pipeline_parity.py &&
git commit -m "test: regenerate the POPS golden -- ridge selection no longer sees the validation set"`.

- [ ] **Step 4: Full suite** — `uv run --extra gp pytest -m "not slow" -q` → PASS.
- [ ] **Step 5: Commit** — `git add bench/acegp_cantor/run.py && git commit -m "refactor(bench): run.py is a thin driver on ace_jax.fit.pipeline (outputs unchanged)"`

---

### Task 10: `ace-jax fit` on the pipeline, with the new options

**Files:**
- Modify: `src/ace_jax/cli.py` (`_add_fit_args`, `run`; move `_pad_to_multiple` into `pipeline/objective.py` and import it back)
- Test: `tests/test_pipeline_parity.py` (cli scenario), `tests/test_gp_cli.py` (unchanged, must pass), `tests/test_pipeline_units.py`

**Interfaces:**
- Consumes: the pipeline API.
- Produces: new `ace-jax fit` flags: `--data` (with `--ntrain/--ntest/--test-start`, mutually exclusive with `--train`), `--ood`, `--opt {adam,lbfgs}` (default adam), `--map-restarts`, `--laplace {svi,fd}` (default svi), `--lml {device,host-cache}`, `--density {none,pair,pca}`, `--pca-d`, `--embedding`, `--baseline`, `--uq {blr,pops}`, `--pops-ridge`, `--e0 {model,lsq}` (default model), `--init`. `--weights` accepts a JSON **dict** (ACEfit weights, as today) or a JSON **list** (weight factors, as run.py).

- [ ] **Step 1: Failing tests** (Review Focus 2)

```python
def test_cli_weights_accepts_dict_or_factor_list():
    from ace_jax.cli import _parse_weights
    w, f = _parse_weights('{"default": {"E": 30, "F": 1, "V": 1}}')
    assert w == {"default": {"E": 30, "F": 1, "V": 1}} and f is None
    w, f = _parse_weights('[{"Structural": {}}]')
    assert w is None and [type(x).__name__ for x in f] == ["Structural"]
    with pytest.raises(ValueError, match="weights"):
        _parse_weights('"nonsense"')


def test_cli_rejects_train_and_data_together(tmp_path):
    from ace_jax.cli import main
    with pytest.raises(SystemExit):
        main(["fit", "--model", "m.npz", "--train", "a.xyz", "--data", "b.xyz", "--r0", "2.35",
              "--out", str(tmp_path)])
```

- [ ] **Step 2: Verify failure** — `-k "cli_weights or cli_rejects"` → FAIL `ImportError: cannot import name '_parse_weights'`.

- [ ] **Step 3: Implement** — extend `_add_fit_args` (existing flags and defaults unchanged; `--train` no longer `required=True`; a mutually exclusive group `--train` / `--data`, one required):

```python
def _add_fit_args(p):
    p.add_argument("--model", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--train", help="training extxyz (with --test, or tested on itself)")
    src.add_argument("--data", help="one extxyz split by a seeded permutation (--ntrain/--ntest/--test-start)")
    p.add_argument("--test"); p.add_argument("--ood", help="extra out-of-distribution test extxyz")
    p.add_argument("--ntrain", type=int, default=800); p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--test-start", type=int, default=None)
    p.add_argument("--energy-key", default="energy")
    p.add_argument("--force-key", default="forces"); p.add_argument("--virial-key", default="virial")
    p.add_argument("--weights", default=None,
                   help='JSON: an ACEfit weights dict {"default": {"E":..,"F":..,"V":..}, <config_type>: ..} '
                        'or a list of weight factors [{"Structural": {}}, {"ConfigType": {...}}]')
    p.add_argument("--e0", choices=["model", "lsq"], default="model",
                   help="per-species E0: the model's (default) or least squares on the training energies")
    p.add_argument("--baseline", default=None, help="dimer_mean.npz: fit the residual to this pair mean")
    p.add_argument("--configs-per-batch", type=int, default=8)
    p.add_argument("--m-per-species", type=int, default=500)
    p.add_argument("--kernel", default="cosine", choices=["cosine", "matern32"])
    p.add_argument("--no-bump", action="store_true")
    p.add_argument("--density", choices=["none", "pair", "pca"], default="none",
                   help="GP feature map: full descriptor, pair densities, or a PCA view (--pca-d)")
    p.add_argument("--pca-d", type=int, default=128)
    p.add_argument("--embedding", default=None, help="MACE element table (JSON): frozen species coregionalization")
    p.add_argument("--objective", default="lml", choices=["lml", "loo"])
    p.add_argument("--lml", choices=["device", "host-cache"], default="device",
                   help="host-cache: cache the linear design rows in host RAM (GP, pair|pca, L-BFGS, map only)")
    p.add_argument("--opt", choices=["adam", "lbfgs"], default="adam")
    p.add_argument("--map-restarts", type=int, default=1, help="L-BFGS multi-start (best log-posterior)")
    p.add_argument("--init", default=None, help="theta_map.json to start the MAP from")
    p.add_argument("--rungs", default="map,laplace"); p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--laplace", choices=["svi", "fd"], default="svi")
    p.add_argument("--map-steps", type=int, default=500); p.add_argument("--vi-steps", type=int, default=2000)
    p.add_argument("--nuts-warmup", type=int, default=500); p.add_argument("--nuts-samples", type=int, default=500)
    p.add_argument("--nuts-chains", type=int, default=4); p.add_argument("--r0", type=float, required=True)
    p.add_argument("--uq", choices=["blr", "pops"], default="blr", help="pops: linear arm (--m-per-species 0)")
    p.add_argument("--pops-ridge", default="auto")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
    p.add_argument("--devices", type=int, default=1,
                   help="shard sufficient statistics over this many devices (Task 15, stretch)")
    return p


def _parse_weights(s):
    """(ACEfit weights dict, factor list): exactly one is set, from one JSON string."""
    from .fit.weights import ConfigType, PerConfig, Quantity, Structural
    if not s:
        return None, None
    v = json.loads(s)
    if isinstance(v, dict):
        return v, None
    if isinstance(v, list):
        cls = {"Structural": Structural, "Quantity": Quantity, "ConfigType": ConfigType, "PerConfig": PerConfig}
        return None, [cls[name](**kw) for entry in v for name, kw in entry.items()]
    raise ValueError("--weights must be a JSON object (ACEfit weights) or a JSON list (weight factors)")


def run(a):
    from .fit.pipeline import FitConfig, fit, load_fit_data, write_outputs
    weights, factors = _parse_weights(a.weights)
    rungs = tuple(r.strip() for r in a.rungs.split(","))
    ridge = a.pops_ridge if a.pops_ridge in ("auto", "blr") else float(a.pops_ridge)
    cfg = FitConfig(
        model=a.model, arm="gp" if a.m_per_species > 0 else "linear", energy_key=a.energy_key,
        force_key=a.force_key, virial_key=a.virial_key, ntrain=a.ntrain, ntest=a.ntest,
        test_start=a.test_start, seed=a.seed, batch=a.configs_per_batch, weights=weights, factors=factors,
        baseline=a.baseline, e0=a.e0, m_per_species=a.m_per_species, kernel=a.kernel, bump=not a.no_bump,
        density=a.density, pca_d=a.pca_d, embedding=a.embedding, r0=a.r0, objective=a.objective,
        lml=a.lml, devices=a.devices, opt=a.opt, map_steps=a.map_steps, map_restarts=a.map_restarts,
        init=json.load(open(a.init)) if a.init else None, rungs=rungs, laplace=a.laplace,
        n_draws=a.n_draws, vi_steps=a.vi_steps, nuts_warmup=a.nuts_warmup, nuts_samples=a.nuts_samples,
        nuts_chains=a.nuts_chains, uq=a.uq, predict_train=False, pops_ridge=ridge)
    cfg.validate()
    data = (load_fit_data(cfg, data=a.data, ood=a.ood) if a.data
            else load_fit_data(cfg, train=a.train, test=a.test, ood=a.ood))
    res = fit(cfg, data)
    write_outputs(res, a.out, layout=("cli",), argv=vars(a))
    return {key.split("/")[1]: m for key, m in res.preds.metrics.items() if key.startswith("test/")}
```

Keep `cmd_eval`, `cmd_construct` and `main` unchanged in this task. Delete `cli.py`'s own `_pad_to_multiple` (Task 4 moved it into `pipeline/objective.py`) and add `from .fit.pipeline.objective import _pad_to_multiple  # noqa: F401` so existing imports of `ace_jax.cli._pad_to_multiple` keep working.

- [ ] **Step 4: Run** — `uv run --extra gp pytest tests/test_pipeline_parity.py tests/test_gp_cli.py tests/test_pipeline_units.py -q` → all PASS (the `cli_map_laplace` golden reproduces: `e0="model"`, Adam, SVI Laplace, `metrics.csv` rows identical).
- [ ] **Step 5: Full suite and commit** — `uv run --extra gp pytest -m "not slow" -q` → PASS; `git add src/ace_jax/cli.py tests && git commit -m "feat(cli): ace-jax fit on the pipeline; --data/--ood/--opt/--map-restarts/--lml/--density/--embedding/--uq pops"`

---

### Task 11: `ace-jax construct --embedding`

**Files:**
- Modify: `src/ace_jax/cli.py` (`construct` parser, `cmd_construct`)
- Create: `tests/test_cli_construct_embedding.py`

**Interfaces:**
- Consumes: `construct.model.build_embedding_model(elements, order, totaldegree, embedding, *, d_max, wL, maxl, rcut, reduction, coupling_cache_dir, ...)` (on main from PR #4).
- Produces: `ace-jax construct --embedding <table.json|identity> [--d-max N] [--maxl L] [--reduction pca|truncate]`; without `--embedding` behaviour is unchanged (`build_model`). `--rcut` default stays 5.5 for `build_model`; with `--embedding` an omitted `--rcut` means the embedded model's default (2.5 x mean bond length).

- [ ] **Step 1: Failing test** (uses the committed coupling cache from PR #4, restamped in-process as in `tests/test_construct_embedding_model.py`)

```python
# tests/test_cli_construct_embedding.py
import json, pathlib
import numpy as np
import pytest

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"


def _restamped_cache(tmp):
    from ace_jax.construct import coupling as C
    out = tmp / "cpl"
    for f in sorted((FIX / "coupling_cache_embedding").glob("cpl-*.npz")):
        z = np.load(f); m = json.loads(bytes(z["meta_json"]).decode())
        n = sum(1 for k in z.files if k.startswith("aa_spec_"))
        cpl = C.Coupling(A2B=np.asarray(z["A2B"]),
                         aa_sig=tuple(tuple(tuple(int(v) for v in t) for t in s) for s in m["aa_sig"]),
                         aspec=tuple((int(r), int(y)) for r, y in m["aspec"]),
                         aa_specs=tuple(np.asarray(z[f"aa_spec_{k + 1}"]) for k in range(n)),
                         nnll_spec=tuple(tuple((int(b[0]), int(b[1])) for b in bb) for bb in m["nnll_spec"]))
        C._write_entry(C._entry_path(out, m["key"]), cpl, m["key"], [[tuple(b) for b in bb] for bb in m["mb"]],
                       [tuple(r) for r in m["rnl"]], [tuple(y) for y in m["ylm"]])
    return out


def test_construct_embedding_matches_the_library(tmp_path, monkeypatch):
    import jax
    jax.config.update("jax_enable_x64", True)
    monkeypatch.setenv("ACEJAX_NO_JULIA", "1")
    from ace_jax.cli import main
    from ace_jax.construct.model import build_embedding_model
    from ace_jax.eval import load
    z = np.load(FIX / "embedding_ref_mh1_SiGe.npz")
    table = tmp_path / "emb.json"
    table.write_text(json.dumps({"Z": [14, 32], "emb": z["table"].tolist()}))
    cache = _restamped_cache(tmp_path)
    out = tmp_path / "m.npz"
    main(["construct", "--elements", "Si,Ge", "--order", "2", "--max-degree", "6", "--embedding", str(table),
          "--maxl", "6", "--coupling-cache-dir", str(cache), "--out", str(out)])
    model, meta, _ = load(out)
    ref = build_embedding_model([14, 32], 2, 6, embedding=([14, 32], z["table"]), maxl=6,
                                coupling_cache_dir=str(cache))
    for k in ("n_B", "n_pair", "n_rnl", "len_basis"):
        assert meta[k] == ref.meta[k], k
    assert json.loads(meta["embedding"])["widths"] == [2, 3]
```

- [ ] **Step 2: Verify failure** — `uv run --extra gp pytest tests/test_cli_construct_embedding.py -v` → FAIL `error: unrecognized arguments: --embedding`.

- [ ] **Step 3: Implement** — add to the `construct` parser (keep existing flags; change `--rcut` default to `None` and resolve it in `cmd_construct`):

```python
    con.add_argument("--embedding", default=None,
                     help="frozen element embedding: a JSON table {Z, emb} or 'identity' "
                          "(builds ace_embedding_model: ace1-compatible, factorised radial)")
    con.add_argument("--d-max", type=int, default=None, help="cap on per-order channel widths (default lossless)")
    con.add_argument("--maxl", type=int, default=None)
    con.add_argument("--reduction", choices=["pca", "truncate"], default="pca")
```

```python
def cmd_construct(a):
    from .construct.export import save_npz
    els = [int(e) if e.strip().isdigit() else e.strip() for e in a.elements.split(",")]
    if a.embedding:
        from .construct.model import build_embedding_model
        auth = build_embedding_model(els, a.order, a.max_degree, embedding=a.embedding, d_max=a.d_max,
                                     wL=a.wL, maxl=a.maxl, rcut=a.rcut, reduction=a.reduction,
                                     with_gamma=not a.no_gamma, coupling_cache=not a.no_coupling_cache,
                                     coupling_cache_dir=a.coupling_cache_dir)
    else:
        from .construct.model import build_model
        auth = build_model(els, a.order, a.max_degree, wL=a.wL, rcut=5.5 if a.rcut is None else a.rcut,
                           rin=a.rin, radial_mode=a.radial_mode, pair_mode=a.pair_mode,
                           seed=a.seed, with_gamma=not a.no_gamma,
                           coupling_cache=not a.no_coupling_cache,
                           coupling_cache_dir=a.coupling_cache_dir)
    out = pathlib.Path(a.out).expanduser()
    if out.parent and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, auth)
    m, meta = auth.model, auth.meta
    print(f"authored {m.A2B.shape[0]} B functions ({meta['n_AA']} AA), "
          f"{meta['n_pair']} pair, {meta['len_basis']} basis entries, "
          f"lmax {meta['lmax']}, rcut {meta['rcut']} -> {a.out}")
    return auth
```

`build_embedding_model` accepts a path string; `"identity"` is handled by it.

- [ ] **Step 4: Run** — `uv run --extra gp pytest tests/test_cli_construct_embedding.py tests/test_python_authoring.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/ace_jax/cli.py tests/test_cli_construct_embedding.py && git commit -m "feat(cli): ace-jax construct --embedding (embedded ace1-compatible models)"`

---

### Task 12: Documentation and final checks

**Files:**
- Modify: `README.md` (the `ace-jax` CLI section)

- [ ] **Step 1: Document** — add under the CLI section:

```markdown
### Fitting (`ace-jax fit`)

`ace-jax fit` and the research driver `bench/acegp_cantor/run.py` share one pipeline
(`ace_jax.fit.pipeline`). Common options:

- data: `--train/--test` files, or `--data` split with `--ntrain/--ntest/--test-start`; `--ood` for an extra test set
- GP features: `--density none|pair|pca` (`--pca-d`), `--embedding` (frozen species coregionalization)
- likelihood: `--lml host-cache` caches the linear design rows in host RAM (GP arm, pair/pca features, L-BFGS, MAP only)
- MAP: `--opt adam|lbfgs`, `--map-restarts N` (best of N L-BFGS starts; the joint LML is multimodal)
- UQ: `--rungs map,laplace,...` (`--laplace svi|fd`), or `--uq pops` on the linear arm (`--m-per-species 0`)

### Building an embedded model (`ace-jax construct --embedding`)

    ace-jax construct --elements Cr,Mn,Fe,Co,Ni --order 3 --max-degree 10 \
        --embedding mace_embedding.json --out cantor_embed.npz          # lossless widths
    ... --d-max 16                                                      # capped widths
```

- [ ] **Step 2: Full suite, slow suite, parity** — `uv run --extra gp pytest -q` (includes `slow`) → PASS.
- [ ] **Step 3: Commit** — `git add README.md && git commit -m "docs: ace-jax fit / run.py share the pipeline; construct --embedding"`
