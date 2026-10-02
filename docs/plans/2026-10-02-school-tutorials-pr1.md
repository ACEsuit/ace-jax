# School tutorials, PR 1 (E1, E1x, C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert MLIP-school-2026's E1, E1x and C notebooks into ace-jax tutorials 3, 4 and 7. They run locally on shipped MACE-MPA-0 / MACE-MP-0b3 labels, with no servers. Add the four small ace-jax features they need.

**Architecture:**

- **Fitting from data in memory:** `load_configs`/`load_fit_data` accept lists of `ase.Atoms` as well as paths, and an optional `stress_key` that converts stress to virial.
- **Evidence on the result:** `fit_map` returns the log marginal likelihood (`MapFit.log_evidence`).
- **E1x's least-squares arm:** chosen in a measured spike (Task 3).
- **Labels:** the tutorial-support module `ace_jax.tutorials.labels` serves labels from shipped content-hash caches, and falls back to a live MACE model only when mace-torch is installed.
- **Label generator:** `docs/user/tutorials/data/school/make_labels.py` builds the caches. It runs in a separate MACE environment.
- **Pages:** each tutorial is a marimo notebook rendered by `docs/build_tutorials.py`, like tutorials 1 and 2.

**Tech Stack:**

- Python 3.11+, JAX (float64), ASE, marimo ≥ 0.25 and matplotlib.
- Generating labels only (never an ace-jax dependency): mace-torch 0.3.16 on CPU torch.

**Spec:** `docs/specs/2026-10-02-school-tutorials-design.md`.

**School source:**

- Repo and commit: ACEsuit/MLIP-school-2026 at `093ebde`.
- Clone it with `git clone https://github.com/ACEsuit/MLIP-school-2026 /tmp/<scratch>/mlip`.
- Notebooks: `notebooks/source/e1/e1_oracle_and_first_fit.py`, `notebooks/source/e1x/e1x_basis_and_overfitting.py` and `notebooks/source/c/c_truth_about_the_truth.py`.
- Reference set: `notebooks/reference/e1x-bulk-reference.xyz`.

## Global Constraints

**Dependencies and labels**
- No torch or mace-torch, either in ace-jax's dependencies or in any notebook's PEP 723 header. Live MACE is an optional extra, documented separately.
- Every shipped label comes from **MACE-MPA-0** (`mace_mp(model="medium-mpa-0")`, MIT). The only exception is C's second labeller, **MACE-MP-0b3** (`mace_mp(model="medium-0b3")`, MIT).
  - Label in float64 on CPU (`default_dtype="float64", device="cpu"`).
  - MH-1 and other ASL models are never used.
- Label files store `energy` (info, eV), `forces` (arrays, eV/Å) and `virial` (info, 3×3, eV; `virial = −stress·V`). They also carry `config_type` and `label_model`. Write them with libAtoms extxyz through `ace_jax.fit.xyz` conventions, so that `load_configs` reads them with the default keys.

**Notebooks**
- Each notebook follows `docs/user/tutorials/notebooks/first_fit_si.py`:
  - a PEP 723 header that installs ace-jax from git, with `TODO(pypi)`, plus `marimo>=0.25` and `matplotlib`;
  - an `app = marimo.App(width="medium")`;
  - the title cell with Goals, then a "Run this notebook" block (uvx command, molab link, static-copy note and runtime);
  - checkpoints as `mo.callout(kind="success"/"warn")`;
  - exercise hints in `mo.accordion`;
  - `jax.config.update("jax_enable_x64", True)` in the imports cell.
- Each runs in a few CPU minutes, under 3 GB of RAM. Data is fetched from `https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/...`, with a fallback to the local checkout when run from one, as tutorial 1 does.
- No mograder imports, API keys, budgets enforced by a server, Julia snippets, Pyodide code or server `load_model` calls.
- No checkpoint asserts a number copied from the school. Every threshold is re-measured on ace-jax plus MPA-0 labels and set with margin.

**Code conventions**
- The code style is that of CLAUDE.md: dense code with short names; trailing comments explain why. No PEP 701 f-strings in `src/` or `tests/`.
- Tests are plain pytest with `jax.config.update("jax_enable_x64", True)` at the top. Heavy tests are `@pytest.mark.slow`.

**Environment and process**
- Memory: the session cgroup is 16 GiB, swap-free, and shared. Run one heavy job at a time, with `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false` and at most 4 test workers.
- The MACE environment is `/storage/eng/essswb/venvs/mace-labels`, with CPU torch 2.14.1 and mace-torch 0.3.16; it is already built. Checkpoints are in `~/.cache/mace`.
- End every commit message with the trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Atoms carrying a SinglePointCalculator** (MACE or ASE output) must yield the same `Config` labels as the same structure written to extxyz and read back. Calculator results must not be lost, as `ase.io` once lost them.
2. **The `stress_key` conversion** must be correct for a non-cubic, triclinic cell, giving virial = −σ·V. The 3×3, Voigt-6 and flat-9 stress forms must all work. A frame with neither virial nor stress must stay label-free, not become zero.
3. **The label cache key** must not depend on how a structure was built: atom order is part of the identity, while float noise up to 1e-7 Å, or wrapping by a lattice vector, must hit the cache. A cache miss without MACE installed raises an actionable error and never silently returns unlabelled structures.
4. **`MapFit.log_evidence`** must be comparable across basis sizes on the same data: one definition for both the Adam and L-BFGS paths, excluding the hyperprior.
5. **Every tutorial must run its default path** from shipped labels with mace-torch *not* installed. That is how the docs build runs it.

---

### Task 1: Fitting from in-memory Atoms, and stress as virial

**Files:**
- Modify: `src/ace_jax/fit/data.py` (`load_configs`)
- Modify: `src/ace_jax/fit/pipeline/data.py` (`load_fit_data`)
- Modify: `src/ace_jax/fit/pipeline/config.py` (add `stress_key: str | None = None` after `virial_key`)
- Modify: `src/ace_jax/cli.py`: add `--stress-key` to `aj fit` and `aj eval`, map it in `_fit_config` and `cmd_eval`, and add it to the run-file dests automatically.
- Test: `tests/test_fit_atoms_input.py` (new)

**Interfaces:**
- **Produces:**
  - `load_configs(source, energy_key="energy", force_key="forces", virial_key="virial", stress_key=None, weights=None, weight_key="config_type", factors=None)`, where `source` is a path or an iterable of `ase.Atoms`.
  - `load_fit_data(cfg, data=..., train=..., test=..., ood=...)`, where each argument is a path or a list of Atoms.
  - `FitConfig.stress_key`.
- **Behaviour:**
  - An Atoms' labels are taken from `atoms.info` and `atoms.arrays` under their own names. Any `atoms.calc.results` (a SinglePointCalculator) are added as `info["energy"]`, `arrays["forces"]` and `info["stress"]` when not already present.
  - When `virial_key` finds nothing for a periodic frame and `stress_key` is set and present, use `virial = -stress_full(3x3) * volume`. Stress can be 3×3, Voigt-6 (xx yy zz yz xz xy) or flat 9 (row-major).

- [ ] **Step 1: Write the failing tests** in `tests/test_fit_atoms_input.py`:

```python
"""load_configs / load_fit_data take ase.Atoms lists as well as files; stress labels
convert to virials (virial = -stress * volume)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from conftest import FIXTURE_DIR

from ace_jax.fit.data import load_configs

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
KEYS = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")


def _atoms_from_file(n=6):
    from ace_jax.fit.xyz import read_extxyz
    from ase import Atoms
    out = []
    for f in read_extxyz(XYZ)[:n]:
        a = Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
        a.info.update(f.info); [a.set_array(k, v) for k, v in f.arrays.items()]
        out.append(a)
    return out


def test_atoms_list_gives_the_same_configs_as_the_file():
    a, b = load_configs(_atoms_from_file(), **KEYS), load_configs(XYZ, **KEYS)[:6]
    for x, y in zip(a, b):
        assert x.energy == y.energy and x.config_type == y.config_type and (x.w_E, x.w_F) == (y.w_E, y.w_F)
        np.testing.assert_array_equal(x.forces, y.forces)
        np.testing.assert_array_equal(x.positions, y.positions)
        assert (x.virial is None) == (y.virial is None)


def test_singlepoint_calculator_results_are_labels():
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    F = np.random.default_rng(0).normal(size=(len(at), 3))
    at.calc = SinglePointCalculator(at, energy=-40.0, forces=F)
    (c,) = load_configs([at])
    assert c.energy == -40.0
    np.testing.assert_array_equal(c.forces, F)


@pytest.mark.parametrize("form", ["voigt", "full", "flat"])
def test_stress_converts_to_virial_on_a_triclinic_cell(form):
    at = bulk("Si", "diamond", a=5.43)                       # fcc primitive: not orthogonal
    at.set_cell(at.cell.array @ np.array([[1.0, 0.03, 0.0], [0.0, 1.0, 0.02], [0.0, 0.0, 1.0]]),
                scale_atoms=True)
    s = np.array([[0.01, 0.002, -0.003], [0.002, -0.02, 0.004], [-0.003, 0.004, 0.015]])
    at.info["stress"] = {"voigt": full_3x3_to_voigt_6_stress(s), "full": s, "flat": s.ravel()}[form]
    at.info["energy"] = -1.0
    (c,) = load_configs([at], stress_key="stress")
    np.testing.assert_allclose(c.virial, -s * at.get_volume(), rtol=0, atol=1e-12)


def test_no_virial_and_no_stress_stays_unlabelled():
    at = bulk("Si", "diamond", a=5.43, cubic=True); at.info["energy"] = -40.0
    (c,) = load_configs([at], stress_key="stress")
    assert c.virial is None


def test_load_fit_data_accepts_atoms_lists():
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), arm="linear", m_per_species=0, r0=2.35, **KEYS)
    atoms = _atoms_from_file(12)
    d = load_fit_data(cfg, train=atoms[:8], test=atoms[8:], log=lambda *a: None)
    assert (len(d.train), len(d.test)) == (8, 4)
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_fit_atoms_input.py -v`
  Expected: FAIL (`load_configs` reads only paths; it has no `stress_key`).

- [ ] **Step 3: Implement.** In `fit/data.py`, add the following before `load_configs`:

```python
def _atoms_frames(source):
    """ase.Atoms -> fit.xyz.Frame-like records, keys as named; SinglePointCalculator
    results (what MACE and ASE leave behind) become info energy/stress and arrays forces."""
    from .xyz import Frame
    out = []
    for a in source:
        info, arrays = dict(a.info), {k: np.asarray(v) for k, v in a.arrays.items() if k not in ("numbers", "positions")}
        res = getattr(a.calc, "results", None) or {}
        if "energy" in res: info.setdefault("energy", float(res["energy"]))
        if "free_energy" in res and "energy" not in res: info.setdefault("energy", float(res["free_energy"]))
        if "stress" in res: info.setdefault("stress", np.asarray(res["stress"]))
        if "forces" in res: arrays.setdefault("forces", np.asarray(res["forces"]))
        cell = np.asarray(a.cell.array, float)
        out.append(Frame(np.asarray(a.numbers), np.asarray(a.positions, float), cell,
                         np.asarray(a.pbc, bool) if cell.any() else np.zeros(3, bool), info, arrays))
    return out


def _stress_to_virial(s, cell):
    """virial = -stress * volume; stress as 3x3, Voigt-6 (xx yy zz yz xz xy) or flat 9."""
    from ase.stress import voigt_6_to_full_3x3_stress
    s = np.asarray(s, float)
    S = voigt_6_to_full_3x3_stress(s) if s.shape == (6,) else s.reshape(3, 3)
    return -S * abs(np.linalg.det(np.asarray(cell, float)))
```

Then change `load_configs`:
- Signature: `load_configs(source, ..., virial_key="virial", stress_key=None, ...)`.
- The frame source: `frames = read_extxyz(source) if isinstance(source, (str, os.PathLike)) else _atoms_frames(source)`.
- The `where` label: `where = f"{source} config {i}"` for paths, and `f"structure {i}"` otherwise.
- After `V = _get(at.info, virial_key)`, add:

```python
        if V is None and stress_key is not None and np.any(at.pbc):
            S = _get(at.info, stress_key)
            if S is not None:
                V = _stress_to_virial(S, at.cell); found.add(virial_key)
```

- Keep the typo check as it is, but treat `virial_key` as found when a stress was converted (above).
- Add `import os` at the top of `fit/data.py` if it is not already imported.

In `fit/pipeline/data.py::load_fit_data`:
- Add `stress_key=cfg.stress_key` to `keys`.
- Update the docstring to say that `data`/`train`/`test`/`ood` take a path or a list of Atoms.
- `_config_type_weights(data or train)` reads a path. When `train` is a list, compute the config types from the Atoms' `info.get("config_type")` instead:

```python
def _config_type_weights(src):
    if not isinstance(src, (str, os.PathLike)):
        names = sorted({str(a.info["config_type"]) for a in src if "config_type" in a.info})
        return {"default": {"E": 1.0, "F": 1.0, "V": 1.0}, **{n: {"E": 1.0, "F": 1.0, "V": 1.0} for n in names}}
    ...existing body...
```

Read the existing `_config_type_weights` first, and keep its exact return form for the path case.

In `config.py`, add the field:

```python
    stress_key: str | None = None        # ASE/MACE stress label, converted to virial = -stress*V when no virial
```

In `cli.py`:
- In `_add_fit_args`, after the virial key: `p.add_argument("--stress-key", default=None, help="stress label (eV/A^3) to use as the virial (virial = -stress * volume) when a config has no virial label")`.
- Add the same argument to the eval parser.
- `_fit_config`: `stress_key=a.stress_key`.
- `cmd_eval` keys: `stress_key=a.stress_key`.

- [ ] **Step 4: Run the tests.**
  Run: `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_fit_atoms_input.py tests/test_gp_data.py tests/test_runfile.py tests/test_runfile_roundtrip.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit.**
  `feat(data): load_configs/load_fit_data take ase.Atoms lists; stress_key converts stress to virial`

---

### Task 2: The log-evidence on the fit result

**Files:**
- Modify: `src/ace_jax/fit/pipeline/mapfit.py` (`MapFit`, `fit_map`)
- Test: `tests/test_pipeline_units.py` (append)

**Interfaces:**
- **Produces:** `MapFit.log_evidence: float | None` (the last field, default None), equal to `float(obj.lik(to_array(theta)))`: the log marginal likelihood at the chosen θ, excluding the hyperprior. It is set by both the Adam and L-BFGS paths. The `sigma_type` path leaves it None.
- `FitResult.map.log_evidence` then gives it to callers.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_pipeline_units.py`:

```python
def test_fit_reports_the_log_evidence_for_both_optimisers():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=16, ntest=6, batch=4, r0=2.35, arm="linear",
                m_per_species=0, rungs=("map",), map_steps=5, predict_train=False)
    for opt in ("adam", "lbfgs"):
        cfg = FitConfig(opt=opt, **base)
        d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None)
        res = fit(cfg, d, log=lambda *a: None)
        b = build_problem(cfg, d)
        ref = float(make_objective(cfg, d, b).lik(to_array(res.theta)))
        assert np.isfinite(res.map.log_evidence) and abs(res.map.log_evidence - ref) <= 1e-8 * abs(ref), opt
```

- [ ] **Step 2: Run it to verify it fails.** `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_pipeline_units.py -k log_evidence -v`. Expected: `AttributeError: 'MapFit' object has no attribute 'log_evidence'`.

- [ ] **Step 3: Implement.**
  - Add `log_evidence: object = None` as the last `MapFit` field.
  - In `fit_map`, for the Adam path, compute `ev = float(obj.lik(to_array(theta)))` before returning, and pass it.
  - For the L-BFGS path, compute it the same way at the best θ: the logged `logpost` includes the hyperprior, so it is a different quantity.
  - Leave `sigma_type` as None.
  - Import `to_array` from `..hypers` if it is not already imported.
  - `obj.lik` is the LML in `make_objective`. Under `lml="host-cache"` it is a `HostCachedLML` object: call `obj.lik(a)` there too, or use `obj.lik.value_and_grad(a)[0]`, whichever exists. Check `hostcache.py`.

- [ ] **Step 4: Run the test and its neighbours.** `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_pipeline_units.py tests/test_pipeline_parity.py -q`. Expected: PASS. The parity goldens are bit-exact, and this adds a field without changing any computed value.

- [ ] **Step 5: Commit.** `feat(pipeline): MapFit.log_evidence, the log marginal likelihood at the chosen hyperparameters`

---

### Task 3: A least-squares arm for E1x (a spike, then the chosen implementation)

**Files:**
- Create: `scratch/e1x_lstsq_spike.py`. It lives in the scratchpad and is not committed.
- Then one of:
  - (A) `src/ace_jax/fit/pipeline/config.py` and `mapfit.py`: `fix_sigma_c`;
  - (B) `src/ace_jax/fit/pipeline/run.py`, `config.py` and a new `src/ace_jax/fit/pipeline/lstsq.py`: `solver="lstsq"`.
- Test: `tests/test_pipeline_units.py` (append)

**Interfaces:**
- **Produces, either:**
  - (A) `FitConfig(fix_sigma_c=<float>)`, which pins the coefficient prior scale; or
  - (B) `FitConfig(solver="lstsq")`, which returns the weighted least-squares readout with no prior (minimum-norm when underdetermined), as ACEfit's QR does.
- Task 6 (E1x) consumes whichever exists; the ledger records which.

- [ ] **Step 1: The spike.** Clone the school repo. Relabel nothing yet: use the existing e1x reference file, which has `energy`/`forces` keys and no stress.
  1. Split it as the school does: `default_rng(11)`, 80/20.
  2. For `max_degree` in (8, 12, 14, 20), build `BasisSpec(order=3, max_degree=d, rcut=5.5, elements=("Si",))`.
  3. Fit with the evidence (default) and record the train/test E RMSE in meV/atom, the basis size and `res.map.log_evidence`.
  4. Then measure the least-squares candidates:
     - (A) Fix `log_sigma_c` at `ln(1e4)`, with `BasisSpec(no_gamma=True)`. Implement it temporarily, like `fix_rho`: pin `lo[6] = hi[6] = x0[6] = ln(1e4)` in `fit_map`'s L-BFGS path; `sigma_c` is index 6 in `Hypers`.
     - (B) A direct weighted least squares. Use `ace_jax.fit.solve.stacked_design(prob, ds, theta)` and drop the prior block (rows after the data rows), then call `np.linalg.lstsq(A, y, rcond=None)` with weights E 30, F 1, V 0, as the school's workbench does. Compare its train/test RMSE with A's.
  5. Write the four RMSE curves and basis sizes to the task ledger.

  **Decision rule:**
  - Choose **A** if it shows the school's shape and stays numerically stable at degree 20: train RMSE falling monotonically, test RMSE with an interior minimum, and the test/train ratio above 5 at the largest basis, all finite. A reuses the pipeline.
  - Otherwise choose **B**.
  - Record the measured numbers either way. Task 6 uses them to set its checkpoints.

- [ ] **Step 2: Write the failing test** for the chosen arm. Append to `tests/test_pipeline_units.py`:
  - (A):

```python
def test_fix_sigma_c_pins_the_prior_scale():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=16, ntest=6, batch=4, r0=2.35, arm="linear", m_per_species=0,
                    rungs=("map",), opt="lbfgs", map_steps=10, predict_train=False, fix_sigma_c=1e4)
    res = fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None),
              log=lambda *a: None)
    assert abs(np.exp(res.theta.log_sigma_c) - 1e4) < 1e-6 * 1e4
```

  - (B): the same config with `solver="lstsq"` (and no `fix_sigma_c`). Assert that the saved readout equals `np.linalg.lstsq` on `stacked_design` without the prior block, to `rtol=1e-8`, as in the spike.

- [ ] **Step 3: Run it to verify it fails** (an unknown `FitConfig` field).

- [ ] **Step 4: Implement the chosen arm.**
  - (A):
    - A `FitConfig` field `fix_sigma_c: float | None = None`.
    - `validate()`: requires `opt == "lbfgs"`, as `fix_rho` does.
    - `fit_map`: `if cfg.fix_sigma_c is not None: lo[6] = hi[6] = x0[6] = np.log(cfg.fix_sigma_c)`, plus a log line.
  - (B):
    - A `FitConfig` field `solver: str = "evidence"`, with values `"evidence"` or `"lstsq"`. `validate()` requires `arm == "linear"` and `uq == "blr"` for lstsq.
    - In `run.py`, when `solver == "lstsq"`, skip the MAP and rungs. Compute the readout with `lstsq.py::fit_lstsq(prob, ds, theta=prob.prior.mu)` and return it in the `FitResult` shape the exporters expect. Prediction uses the readout as the mean, with zero variance: mark it with `res.preds` metrics computed from the mean only.
    - The design is held in host memory, so the docstring states it is for small tutorial-scale problems.
  - In both cases, document the option in SKILL.md's fit-options table and the CLI `--help`. Expose `--fix-sigma-c` or `--solver` in `cli.py` and `_fit_config`.

- [ ] **Step 5: Run the tests.** `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_pipeline_units.py tests/test_pipeline_parity.py tests/test_runfile.py -q`. Expected: PASS.

- [ ] **Step 6: Commit.** Either `feat(pipeline): fix_sigma_c pins the coefficient prior scale (a least-squares limit for teaching)` or `feat(pipeline): solver="lstsq", plain weighted least squares for teaching`.

---

### Task 4: `ace_jax.tutorials.labels`, shipped label caches with a live-MACE fallback

**Files:**
- Create: `src/ace_jax/tutorials/__init__.py` (a docstring: tutorial support, not stable API)
- Create: `src/ace_jax/tutorials/labels.py`
- Test: `tests/test_tutorial_labels.py` (new)

**Interfaces:**
- **Produces:**
  - `structure_key(atoms) -> str`: a SHA-256 over the numbers, the pbc, the cell rounded to 1e-6 Å, and the positions wrapped into the cell (fractional coordinates mod 1, rounded to 1e-6). Atom order is part of the key.
  - `class LabelCache`:
    - `LabelCache.from_file(path_or_url)` reads an extxyz with `energy`, `forces`, `virial`, `label_model` and `config_type`;
    - `.get(atoms) -> ase.Atoms | None`;
    - `.models` is the set of `label_model` values present.
  - `label(structures, *, model="mpa-0", cache=None, live=True) -> list[ase.Atoms]`:
    - It returns copies with `info["energy"]`, `arrays["forces"]`, `info["virial"]` (periodic cells only) and `info["label_model"]`. Each input's own `info` (e.g. `config_type`) is kept.
    - It tries `cache` first, keyed on `structure_key` and `model`. On a miss with `live=True` and mace-torch importable, it runs `mace_mp(model=MODELS[model], default_dtype="float64", device="cpu")`.
    - Otherwise it raises `LabelsUnavailable`, naming how many structures missed and the pip line for the extra.
  - `MODELS = {"mpa-0": "medium-mpa-0", "mp-0b3": "medium-0b3"}`.
  - `labels_used() -> int`, `reset_labels_used()`: a voluntary budget counter. It counts live and cached structures alike, because a structure is a label either way.
  - A `calculator=` keyword on `label()` for tests and power users: an ASE calculator used instead of mace_mp on a miss.

- [ ] **Step 1: Write the failing tests** in `tests/test_tutorial_labels.py`:

```python
"""Tutorial labels: shipped caches keyed on structure content, with a live labeller only on a miss."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from ace_jax.tutorials import labels as L


def _cells():
    out = []
    for s in (0.98, 1.0, 1.02):
        a = bulk("Cu", "fcc", a=3.6 * s, cubic=True); a.rattle(0.01, seed=1); a.info["config_type"] = "bulk"
        out.append(a)
    return out


def _cache(tmp_path, structures):
    lab = L.label(structures, model="mpa-0", calculator=EMT())        # EMT stands in for MACE in tests
    p = tmp_path / "labels.xyz"
    L.write_cache(p, lab)
    return L.LabelCache.from_file(p), lab


def test_cache_hit_returns_the_shipped_labels_and_keeps_info(tmp_path):
    cache, lab = _cache(tmp_path, _cells())
    got = L.label(_cells(), model="mpa-0", cache=cache, live=False)
    for g, r in zip(got, lab):
        assert g.info["energy"] == r.info["energy"] and g.info["config_type"] == "bulk"
        assert g.info["label_model"] == "mpa-0"
        np.testing.assert_array_equal(g.arrays["forces"], r.arrays["forces"])
        np.testing.assert_allclose(g.info["virial"], r.info["virial"], rtol=0, atol=1e-12)


def test_key_ignores_float_noise_and_wrapping_but_not_order():
    a = _cells()[0]
    b = a.copy(); b.positions += 1e-8
    c = a.copy(); c.positions[0] += a.cell[0]                          # same structure, atom wrapped
    d = a.copy(); d.positions[[0, 1]] = d.positions[[1, 0]]           # atoms swapped: a different labelling
    assert L.structure_key(a) == L.structure_key(b) == L.structure_key(c) != L.structure_key(d)


def test_miss_without_a_labeller_raises_and_names_the_extra(tmp_path, monkeypatch):
    cache, _ = _cache(tmp_path, _cells()[:1])
    monkeypatch.setattr(L, "_mace_calculator", lambda model: (_ for _ in ()).throw(ImportError("no mace")))
    with pytest.raises(L.LabelsUnavailable, match="2 of 3.*mace-torch"):
        L.label(_cells(), model="mpa-0", cache=cache)


def test_wrong_model_is_a_miss(tmp_path):
    cache, _ = _cache(tmp_path, _cells())
    with pytest.raises(L.LabelsUnavailable):
        L.label(_cells(), model="mp-0b3", cache=cache, live=False)


def test_counter_counts_every_labelled_structure(tmp_path):
    cache, _ = _cache(tmp_path, _cells())
    L.reset_labels_used()
    L.label(_cells(), model="mpa-0", cache=cache, live=False)
    assert L.labels_used() == 3


def test_virial_is_minus_stress_times_volume():
    (a,) = L.label(_cells()[:1], calculator=EMT())
    at = _cells()[0]; at.calc = EMT()
    from ase.stress import voigt_6_to_full_3x3_stress
    np.testing.assert_allclose(a.info["virial"], -voigt_6_to_full_3x3_stress(at.get_stress()) * at.get_volume(),
                               rtol=1e-10, atol=1e-12)
```

- [ ] **Step 2: Run them to verify they fail** (no module named `ace_jax.tutorials`).

- [ ] **Step 3: Implement `labels.py`**:

```python
"""Labels for the school-derived tutorials (tutorial support, not stable API).

The tutorials' default structures are labelled once, offline, by an MIT-licensed
MACE foundation model (docs/user/tutorials/data/school/make_labels.py) and shipped
as extxyz caches keyed on structure content. `label()` serves those, so a tutorial's
default path needs neither torch nor mace-torch; only a structure the cache does not
hold (a student moved a slider, or brought their own system) is labelled live, and
only if mace-torch is installed."""
import hashlib
import pathlib
import urllib.request

import numpy as np

MODELS = {"mpa-0": "medium-mpa-0", "mp-0b3": "medium-0b3"}      # tutorial name -> mace_mp(model=...)
EXTRA = ('pip install mace-torch --extra-index-url https://download.pytorch.org/whl/cpu '
         '(CPU torch; about 1 GB installed)')
_used = [0]


class LabelsUnavailable(RuntimeError):
    pass


def labels_used():
    return _used[0]


def reset_labels_used():
    _used[0] = 0


def structure_key(atoms):
    """Content hash: numbers, pbc, cell (1e-6 A) and wrapped fractional positions (1e-6);
    atom order is part of the identity (labels are per atom)."""
    cell = np.asarray(atoms.cell.array, float)
    if np.any(atoms.pbc) and abs(np.linalg.det(cell)) > 1e-12:
        frac = np.linalg.solve(cell.T, np.asarray(atoms.positions, float).T).T
        frac = np.where(np.asarray(atoms.pbc), np.mod(np.round(frac, 6), 1.0), frac)
        pos = np.round(np.mod(np.round(frac, 6), 1.0), 6)
    else:
        pos = np.round(np.asarray(atoms.positions, float), 6)
    h = hashlib.sha256()
    for part in (np.asarray(atoms.numbers, np.int64), np.asarray(atoms.pbc, bool),
                 np.round(cell, 6) + 0.0, pos + 0.0):          # + 0.0: no -0.0
        h.update(np.ascontiguousarray(part).tobytes())
    return h.hexdigest()


def _labelled_copy(atoms, energy, forces, stress, model):
    from ase.stress import voigt_6_to_full_3x3_stress
    a = atoms.copy(); a.calc = None
    a.info["energy"] = float(energy); a.arrays["forces"] = np.asarray(forces, float)
    if np.all(a.pbc) and stress is not None:
        s = np.asarray(stress, float)
        S = voigt_6_to_full_3x3_stress(s) if s.shape == (6,) else s.reshape(3, 3)
        a.info["virial"] = -S * a.get_volume()
    a.info["label_model"] = model
    return a


class LabelCache:
    def __init__(self, entries):
        self._d = entries                                    # (model, key) -> labelled Atoms

    @classmethod
    def from_file(cls, path_or_url):
        from ase import Atoms
        from ..fit.xyz import read_extxyz
        src = str(path_or_url)
        if src.startswith(("http://", "https://")):
            local = pathlib.Path.home() / ".cache" / "ace-jax" / "tutorial-labels" / src.rsplit("/", 1)[-1]
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                urllib.request.urlretrieve(src, local)
            src = local
        d = {}
        for f in read_extxyz(src):
            a = Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
            a.info.update(f.info); a.arrays["forces"] = np.asarray(f.arrays["forces"])
            d[(str(f.info["label_model"]), structure_key(a))] = a
        return cls(d)

    @property
    def models(self):
        return {m for m, _ in self._d}

    def get(self, atoms, model):
        hit = self._d.get((model, structure_key(atoms)))
        if hit is None:
            return None
        out = atoms.copy(); out.calc = None
        out.info["energy"] = hit.info["energy"]; out.arrays["forces"] = hit.arrays["forces"].copy()
        if "virial" in hit.info:
            out.info["virial"] = np.asarray(hit.info["virial"], float).reshape(3, 3)
        out.info["label_model"] = model
        return out


def _mace_calculator(model):
    from mace.calculators import mace_mp                    # optional: the live path only
    return mace_mp(model=MODELS[model], default_dtype="float64", device="cpu")


def label(structures, *, model="mpa-0", cache=None, live=True, calculator=None):
    """Labelled copies of `structures` (energy, forces, virial for periodic cells,
    label_model), from `cache` where it holds them, else from the live labeller."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    structures = list(structures)
    out = [None if cache is None else cache.get(a, model) for a in structures]
    missing = [i for i, o in enumerate(out) if o is None]
    if missing:
        if calculator is None:
            if not live:
                raise LabelsUnavailable(f"{len(missing)} of {len(structures)} structures are not in the "
                                        f"shipped {model} labels and live labelling is off")
            try:
                calculator = _mace_calculator(model)
            except ImportError as e:
                raise LabelsUnavailable(
                    f"{len(missing)} of {len(structures)} structures are not in the shipped {model} labels; "
                    f"labelling them needs mace-torch: {EXTRA}. (The tutorial's default settings need "
                    f"nothing extra.)") from e
        for i in missing:
            a = structures[i].copy(); a.calc = calculator
            stress = a.get_stress() if np.all(a.pbc) else None
            out[i] = _labelled_copy(structures[i], a.get_potential_energy(), a.get_forces(), stress, model)
    _used[0] += len(structures)
    return out


def write_cache(path, labelled):
    """Write labelled Atoms (from label()) as a cache file the tutorials ship."""
    from ase.io import write
    frames = []
    for a in labelled:
        b = a.copy(); b.calc = None
        b.info = {k: v for k, v in a.info.items()}
        frames.append(b)
    write(str(path), frames, format="extxyz")
```

  Implementation notes for the executor:
  - `write_cache` uses ASE's writer with the labels in `info`/`arrays` under non-calculator names (`energy` goes to info here). ASE ≥ 3.23 moves `energy`/`forces` keys into a calculator *on read*, not on write. Check that `read_extxyz` returns `energy`, `forces` and `virial` for a written file; the round trip is tested above.
  - If ASE's writer rejects the 3×3 `virial` info value, write `virial` flat (9 values) and reshape it in `get`, which already reshapes.
  - Keep the `structure_key` rounding *after* the mod, so values near 1.0 wrap consistently. The test with a wrapped atom must pass.

- [ ] **Step 4: Run the tests.** `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_tutorial_labels.py tests/test_core_deps.py -q`. Expected: PASS. `test_core_deps` confirms mace stays a lazy import.

- [ ] **Step 5: Commit.** `feat(tutorials): ace_jax.tutorials.labels, shipped label caches keyed on structure content`

---

### Task 5: Label generator and shipped labels for E1, E1x and C

**Files:**
- Create: `docs/user/tutorials/data/school/make_labels.py`
- Create: `docs/user/tutorials/data/school/README.md` (provenance and licence)
- Create (generated, committed): `docs/user/tutorials/data/school/e1/labels-mpa-0.xyz`, `docs/user/tutorials/data/school/e1x/reference.xyz` and `docs/user/tutorials/data/school/c/labels-{mpa-0,mp-0b3}.xyz`
- Create: `src/ace_jax/tutorials/structures.py`, the structure builders shared by the notebooks and the generator, so that both build *identical* structures (the cache keys depend on it).

**Interfaces:**
- **Produces, in `ace_jax.tutorials.structures`:**
  - `e1_cells(strain, rattle, n=10, seed=0) -> list[Atoms]`: cubic 8-atom Si cells, each scaled by `1 + u·strain` (u uniform in [−1, 1], from `default_rng(seed)`) and rattled with σ = `rattle`. Port the school E1 Q1 code exactly, but with a fixed seed.
  - `E1_STRAINS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)` and `E1_RATTLES = (0.0, 0.02, 0.04, 0.06, 0.08)`: the slider grids. The notebook's sliders use exactly these steps.
  - `e1_vacancy_pair() -> (supercell64, vacancy63)`.
  - `c_structures() -> dict[str, Atoms]`: C's bulk, its (111) shuffle slab, and E2's recipe (8 rattled bulk cells plus 4-layer (100), (110) and (111) slabs). Port them from the school's `c` notebook and E2 recipe.
- Shipped files that the notebooks read through `LabelCache.from_file(RAW + "school/<t>/labels-<m>.xyz")`.

- [ ] **Step 1: Write `structures.py`.** Port the builders from the school notebooks (E1 Q1 and Q3, and C's structures), with every random draw seeded. Add a test, `tests/test_tutorial_structures.py`, that pins:
  - the counts and atom numbers (E1: 10 × 8; vacancy pair 64/63; C: bulk + 48-atom (111) slab + 8 bulk + three 32-atom slabs);
  - determinism: the same call twice gives an identical `structure_key` list.

  Run the test; commit `feat(tutorials): seeded structure builders shared by the school tutorials and their label generator`.
- [ ] **Step 2: Write `make_labels.py`.** Run it with the MACE environment (`/storage/eng/essswb/venvs/mace-labels/bin/python`, with `PYTHONPATH=src`). It builds:
  - **E1:** `e1_cells(s, r)` for every (s, r) on the grid, plus `e1_vacancy_pair()`, labelled with mpa-0. That is 30 × 10 + 2 structures.
  - **E1x:** the school's `e1x-bulk-reference.xyz` structures (250), relabelled with mpa-0. Keep `config_type`. Read the source file's structures with `ace_jax.fit.xyz.read_extxyz` and drop the old labels.
  - **C:** `c_structures()` labelled with both mpa-0 and mp-0b3.

  It writes them with `labels.write_cache` and prints counts and timings. Use `labels.label(..., calculator=_mace_calculator(m))` so that the stored format is exactly what `label()` produces.
- [ ] **Step 3: Run it.** Expect about 5–10 minutes on CPU at ~0.7 s per 8-atom structure. Check the outputs:
  - the sizes (each file well under 5 MB);
  - `LabelCache.from_file` loads every file, and every builder structure is a cache hit (a short script asserts zero misses).
- [ ] **Step 4: Write the README.** It covers:
  - the models and versions (mace-torch 0.3.16, `medium-mpa-0`, `medium-0b3`; the MIT licence; citations);
  - the generation command;
  - that E1x's structures come from the MLIP-school-2026 reference set (MIT data) and were relabelled;
  - the energy/forces/virial conventions.
- [ ] **Step 5: Commit** the script, README and data: `docs(tutorials): shipped MACE-MPA-0 / MP-0b3 labels for the E1, E1x and C tutorials`.

---

### Task 6: Tutorial 3 (from E1), "Your data, your property"

**Files:**
- Create: `docs/user/tutorials/notebooks/school_dataset_si.py`
- Modify: `docs/build_tutorials.py` (`NOTEBOOKS["school_dataset_si"] = "dataset-and-properties"`), `mkdocs.yml` (nav) and `docs/user/tutorials/index.md` (table)

**Consumes:** Task 1 (Atoms lists), Task 4 (`label`, `LabelCache`, `labels_used`) and Task 5 (`e1_cells`, `e1_vacancy_pair`, shipped E1 labels).

- [ ] **Step 1: Port the storyline** (one cell group per school section; the school file is `notebooks/source/e1/e1_oracle_and_first_fit.py`):
  - **Title cell.** Goals, the run block, and "Builds on Tutorial 1".
  - **The labeller cell.** Explain that MACE-MPA-0 (MIT) stands in for DFT, and that its labels for this notebook's settings ship with the tutorial. Moving off the sliders' grid needs `EXTRA`; give the install line from `labels.EXTRA`.
  - **Q1, build the data.** Sliders: strain `mo.ui.slider(steps=E1_STRAINS)` and rattle `mo.ui.slider(steps=E1_RATTLES)`, with the school's defaults. Then `e1_cells(strain, rattle)` and `label(..., cache=cache)`.
    - Checkpoint: 10 periodic 8-atom cells, each with energy, forces and virial.
  - **Equation of state.** Birch–Murnaghan on the labelled cells (ASE `EquationOfState`), with a0 and B shown.
    - Checkpoint: a0 within a window measured in Step 2 (the PBE family gives ~5.47 Å; MPA-0 is measured).
  - **Fit.** `FitConfig(model=BasisSpec(order=3, max_degree=8, elements=("Si",)), arm="linear", m_per_species=0, e0="lsq", opt="lbfgs", r0=2.35, rungs=("map",), predict_train=False)`, with `load_fit_data(cfg, train=labelled, test=labelled)`.
    - Show the RMSE table that `fit` logs: capture `log` lines into a list and display the table block.
    - Show the equivalent `aj fit` command with `--order 3 --max-degree 8 ...`.
  - **Q2, R² and parity.** R² of energy per atom over the training set; the parity plot as in tutorial 1.
    - Checkpoint: R² above a measured threshold.
  - **Q3, vacancy.** Label `e1_vacancy_pair()` (shipped), and compute E_vac with the labeller and with the model. Both use `E_vac = E(63) − 63/64·E(64)`.
    - Checkpoint: the labeller's E_vac is physical (window measured).
    - Show the model's error, and the banner "RMSE measures interpolation; properties measure the dataset".
  - **Q4.** The surface-energy question, as a free-text reflection in an accordion with the model answer. No grading.
  - **Exercises.** Hints only.
  - Add a voluntary "labels used: N" line (`labels_used()`) after each labelling call.
- [ ] **Step 2: Measure.**
  1. Run the notebook as a script (`JAX_PLATFORMS=cpu python <nb>`) at the defaults, and at two other grid points.
  2. Record a0, B, R², the labeller's and model's E_vac, and the runtime in the ledger.
  3. Set each checkpoint threshold with margin, from the extremes across the three runs.
  4. Write the measured "typical results" into the title cell's run block.
- [ ] **Step 3: Register the page** in `build_tutorials.NOTEBOOKS`, the `mkdocs.yml` nav (as "3. Your data, your property") and the index table. Run `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync mkdocs build --strict`. Expected: the page is rendered, no warnings, no `marimo-` left in the page (grep).
- [ ] **Step 4: Commit.** `docs(tutorials): tutorial 3, building a dataset and testing a property (from MLIP-school E1)`

---

### Task 7: Tutorial 4 (from E1x), "Basis size, overfitting and the evidence"

**Files:**
- Create: `docs/user/tutorials/notebooks/school_basis_si.py`
- Modify: `docs/build_tutorials.py`, `mkdocs.yml` and `docs/user/tutorials/index.md`

**Consumes:** Task 2 (`res.map.log_evidence`), Task 3 (the chosen least-squares arm), Task 4 and Task 5 (`school/e1x/reference.xyz`).

- [ ] **Step 1: Port the storyline** (the school file is `notebooks/source/e1x/e1x_basis_and_overfitting.py`):
  - **Title and run block.**
  - **Q1, the split.** The seeded 80/20 split of the 250-frame reference, `default_rng(11)`.
  - **Q2, the sweep.** Basis sizes over `max_degree` in (8, 12, 14, 20). Each degree is fitted twice:
    - (i) least squares, with the Task 3 arm;
    - (ii) the evidence fit.

    Table: degree, basis size, train/test E RMSE for both, and the log-evidence.
  - **Q3.**
    - Plot train and test RMSE against basis size, for both arms.
    - Plot the log-evidence against basis size, marking the evidence's choice and least squares' test minimum.
    - Checkpoints:
      - least squares overfits (its test/train ratio at the largest basis exceeds a measured threshold);
      - the evidence fit's test error at the largest basis stays within a measured factor of its best;
      - the evidence-selected degree is among the best by test RMSE.
  - **"Where next".** `order` and `wL` at matched basis size, ungraded.
  - **Exercises.**
- [ ] **Step 2: Measure.**
  1. Run the notebook as a script and record every number and the runtime. The degree-20 fits dominate the runtime.
  2. If the total exceeds about 4 CPU minutes, drop degree 20 for 18 or reduce the training frames, and record the decision as a ruling.
  3. Set the checkpoints with margin.
- [ ] **Step 3: Register and build** (as in Task 6, as "4. Basis size, overfitting and the evidence").
- [ ] **Step 4: Commit.** `docs(tutorials): tutorial 4, basis size, overfitting and Bayesian model selection (from MLIP-school E1x)`

---

### Task 8: Tutorial 7 (from C), "The truth about the truth"

**Files:**
- Create: `docs/user/tutorials/notebooks/school_truth_si.py`
- Modify: `docs/build_tutorials.py`, `mkdocs.yml` and `docs/user/tutorials/index.md`

**Consumes:** Tasks 1, 4 and 5 (`c_structures()`, `school/c/labels-{mpa-0,mp-0b3}.xyz`).

- [ ] **Step 1: Port the storyline** (the school file is `notebooks/source/c/c_truth_about_the_truth.py`):
  - **Title and run block.** It notes that it reuses the surface recipe that tutorial 5 builds; tutorial 5 is in PR 2, so link to it once it exists.
  - **Two labellers.** Label bulk and the (111) slab with mpa-0 and with mp-0b3, and compute γ(111) = (E_slab − N·E_bulk/atom) / 2A for each. Show the gap in percent.
  - **The model's truth.** Fit the surface recipe (`c_structures()["recipe"]`) to mpa-0 labels at total degree 10, and compute the model's γ.
    - Q1: the distance from the model's γ to each labeller.
    - Checkpoint: the model sits closer to its teacher, mpa-0.
  - **Refit against the other labeller.** Fit the same recipe to mp-0b3 labels.
    - Checkpoint: this fit is as faithful to mp-0b3 as the first was to mpa-0 (each distance under a measured tolerance).
  - **Discussion: where references disagree.** Same functional family, different training data and architecture. Foundation models are themselves fits, and the uncertainty in the reference is part of the error bar.
  - **Q2.** A reflection in an accordion.
- [ ] **Step 2: Measure.**
  1. Record γ for both labellers and both fits, the gap, the distances and the runtime.
  2. Set the checkpoints with margin.
  3. If the two labellers' γ(111) differ by less than about 3%, the lesson is too weak at that size: make it a ruling, and add (100) and (110) to make the point across the three facets. Their structures are already in the recipe.
- [ ] **Step 3: Register and build** (as "7. The truth about the truth").
- [ ] **Step 4: Commit.** `docs(tutorials): tutorial 7, the reference is a model choice (from MLIP-school C)`

---

### Task 9: Docs, the full check, and the build budget

**Files:**
- Modify: `docs/user/tutorials/index.md` (the tutorial table, and a note on labels and the optional MACE extra)
- Modify: `docs/user/howto/` (a short "Labels from a foundation model" section in an existing how-to, or a new `howto/foundation-labels.md` in the nav): `ace_jax.tutorials.labels`, `make_labels.py`, `--stress-key`, `load_fit_data` with Atoms
- Modify: `skills/ace-jax/SKILL.md` (`--stress-key`, Atoms input, `log_evidence`, the Task 3 option) and `README.md` (one line for the new tutorials)
- Modify: `CLAUDE.md` (layout: `src/ace_jax/tutorials/`, `docs/user/tutorials/data/school/`)

- [ ] **Step 1: Write the docs.** No backend names in user docs; `tests/test_no_backend_names.py` checks this.
- [ ] **Step 2: Lint and name guards.** `uv run pre-commit run --all-files && JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_names.py tests/test_no_backend_names.py tests/test_docs_tutorials.py -q`. Expected: PASS.
- [ ] **Step 3: The strict docs build, from scratch** (delete the generated tutorial pages first): `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync mkdocs build --strict`.
  - Record the total time. The docs CI job's timeout is 40 minutes; keep the build under about 20.
  - Confirm mace-torch is not installed in that environment: `python -c "import mace"` must fail.
- [ ] **Step 4: The full suite, once.** `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ACEJAX_TEST_WORKERS=4 uv run pytest -q`. Expected: all pass. Refresh `.test_durations` with the new test files' entries if they are slow.
- [ ] **Step 5: Commit.** `docs: school tutorials 3, 4 and 7; foundation-model labels how-to`
