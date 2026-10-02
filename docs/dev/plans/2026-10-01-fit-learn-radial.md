# `aj fit --learn-radial` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One `aj fit --learn-radial …` run learns the tensor radials (VarPro with a held-out gate), then does the usual fit on the learned model.

**Architecture:** A new pipeline stage, `fit/pipeline/radials.py::learn_radials(cfg, data, log)`, runs first in `fit()` when `cfg.learn_radial` is set.
- **Split:** a seeded `radial_val_frac` of `data.train` is held out for the gate.
- **Learn:** `to_analytic`, a linear learning problem, then `fit_radial`.
- **Hand-off:** the selected radials and their readout are patched into the model's npz arrays in memory (`patch_radial_npz` into BytesIO), and `model`, `meta` and `z` in `FitData` are replaced, reapplying the fitted E0.
- **Final fit:** everything downstream is unchanged and refits on the full training set.

The CLI adds five flags, mapped to `FitConfig`; `fit.yaml` records them automatically.

**Tech Stack:**
- Python 3.11+ and JAX in float64;
- `ace_jax.fit.radial_learn` (`fit_radial`) and `fit.radial_model` (`to_analytic`, `rnl_degrees`, `with_radial`);
- `basis.export.patch_radial_npz`;
- argparse with the run-file layer (`ace_jax/runfile.py`);
- pytest.

**Spec:** `docs/dev/specs/2026-10-01-fit-learned-radials-design.md`

## Global Constraints

- **Flags and defaults:** `--learn-radial` (off), `--radial-n-q 12`, `--radial-steps 40`, `--radial-lam-grid 0,1e-2` and `--radial-val-frac 0.2`. These are the only CLI options added.
- **`FitConfig` fields:** `learn_radial: bool = False`, `radial_n_q: int = 12`, `radial_steps: int = 40`, `radial_lam_grid: tuple = (0.0, 1e-2)` and `radial_val_frac: float = 0.2`.
- **Gate split:** the same convention as ARD (`fit/ard.py:409-413`): `idx = np.random.default_rng(cfg.seed).permutation(len(data.train))`, `nval = max(1, int(round(frac * len(data.train))))`, `val = idx[:nval]` and `fit = idx[nval:]`.
- **Learning is linear only** (M = 0). The final fit may use any arm or UQ.
- **Marking:** the saved model is marked `radial_learned` unless the gate selected `"init"`.
- **Outputs:** `out/radial_info.json` holds `selected`, `scores`, `n_q`, `lam_grid`, `val_frac`, `to_analytic_relres_max`, `n_fit`, `n_val` and `seconds`.
- **Unsupported radials:** a radial `to_analytic` can't handle (e.g. `radial_kind == "spline_factorised"`, from embedding models) raises a `ValueError` that names issue #31.
- **Repo style** (CLAUDE.md):
  - dense code, short names, `#` comments that explain why;
  - no PEP 701 f-strings (no nested same-type quotes in f-strings);
  - tests put `jax.config.update("jax_enable_x64", True)` at module top;
  - commits use conventional prefixes with a scope and end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- **Memory:** run `JAX_PLATFORMS=cpu`, and only one heavy pytest process at a time (the session cap is 16 GiB with no swap). The full suite runs at most once, with `ACEJAX_TEST_WORKERS=4`.

## Review Focus

1. **E0 after the reload.** With `e0="lsq"`, `load_fit_data` writes the fitted E0 into the model. The stage reloads from the patched arrays and must put it back, or the final fit silently uses the npz E0. *Pinned in Task 2 (`test_stage_keeps_fitted_e0`).*
2. **Built basis.** `--learn-radial` with the inline basis (`--order/--max-degree`, `FitConfig(model=BasisSpec)`) must work as with `--model`. *Pinned in Task 3 (`test_fit_learn_radial_on_built_basis`).*
3. **Non-linear final fit.** The radial stage is linear, but a GP final fit (`arm="gp"`) after it must still run and save `gp_model.npz` with the learned radials. *Pinned in Task 3 (`test_fit_learn_radial_then_gp`).*
4. **Re-running a resolved `fit.yaml` with `learn_radial: false`.** It records the radial defaults, and `aj fit --config` on it must not raise the stray-option error, which applies only to options typed on the command line. *Pinned in Task 4 (`test_yaml_radial_defaults_without_learning_are_fine`).*
5. **Gate keeps `init`.** With `radial_steps=0` the gate selects `"init"`. The saved model must not be marked learned, and its radials must equal the analytic (widened) starting radials. *Pinned in Task 2 (`test_stage_gate_keeps_init`).*

---

## File structure

- **Create `src/ace_jax/fit/pipeline/radials.py`:** `RadialResult`, `RADIAL_MAP_STEPS` and `learn_radials(cfg, data, log)`. This file has one responsibility: the learned-radial stage.
- **Modify `src/ace_jax/fit/pipeline/config.py`:** the five fields, plus validation.
- **Modify `src/ace_jax/fit/pipeline/run.py`:** call the stage; `FitResult.radial`.
- **Modify `src/ace_jax/fit/pipeline/outputs.py`:** write `radial_info.json`.
- **Modify `src/ace_jax/cli.py`:** the flags, the `_fit_config` mapping, and the stray-option check in `_parse`.
- **Modify `src/ace_jax/runfile.py`:** `radial_lam_grid` as a comma key.
- **Create `tests/test_fit_learn_radial.py`:** every test for this feature.
- **Docs:** `skills/ace-jax/SKILL.md`, `README.md` and `CLAUDE.md` (the layout line for `pipeline/radials.py`).

---

### Task 1: `FitConfig` fields and validation

**Files:**
- Modify: `src/ace_jax/fit/pipeline/config.py` (the fields, after the `pops_*` fields near line 63; `validate()` near line 66)
- Test: `tests/test_fit_learn_radial.py` (create)

**Interfaces:**
- Produces: `FitConfig.learn_radial: bool`, `radial_n_q: int`, `radial_steps: int`, `radial_lam_grid: tuple` and `radial_val_frac: float`. `validate()` raises `ValueError` for `radial_val_frac` outside (0, 1), `radial_n_q < 1`, `radial_steps < 0`, or an empty `radial_lam_grid`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fit_learn_radial.py`:

```python
"""aj fit --learn-radial: learned tensor radials as a fit-pipeline stage
(docs/dev/specs/2026-10-01-fit-learned-radials-design.md)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_ace_model.npz"
QUIET = lambda *a, **k: None
KEYS = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")


def _cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(MODEL), arm="linear", m_per_species=0, rungs=("map",), map_steps=20,
                opt="adam", batch=4, r0=2.35, e0="model", predict_train=False, **KEYS)
    base.update(kw)
    return FitConfig(**base)


def test_config_radial_defaults_and_validation():
    c = _cfg().validate()
    assert (c.learn_radial, c.radial_n_q, c.radial_steps, c.radial_lam_grid, c.radial_val_frac) == \
        (False, 12, 40, (0.0, 1e-2), 0.2)
    for bad in (dict(radial_val_frac=0.0), dict(radial_val_frac=1.0), dict(radial_n_q=0),
                dict(radial_steps=-1), dict(radial_lam_grid=())):
        with pytest.raises(ValueError, match="radial"):
            _cfg(learn_radial=True, **bad).validate()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_fit_learn_radial.py::test_config_radial_defaults_and_validation -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'learn_radial'` or an `AttributeError`.

- [ ] **Step 3: Implement**

In `config.py`, after the `pops_rows` field:

```python
    # learned tensor radials (fit/pipeline/radials.py): VarPro over a held-out-gated
    # training split, before the fit; defaults are bench/learn_radial/run.py's
    learn_radial: bool = False
    radial_n_q: int = 12                 # tensor-radial polynomial span after widening
    radial_steps: int = 40               # L-BFGS steps per roughness weight
    radial_lam_grid: tuple = (0.0, 1e-2)  # relative roughness weights the gate picks among
    radial_val_frac: float = 0.2         # train hold-out for the gate (as ard_val_frac)
```

In `validate()`, before `return self`:

```python
        if self.learn_radial:
            if not 0.0 < self.radial_val_frac < 1.0:
                raise ValueError(f"radial_val_frac must be in (0, 1), got {self.radial_val_frac}")
            if self.radial_n_q < 1:
                raise ValueError(f"radial_n_q must be >= 1, got {self.radial_n_q}")
            if self.radial_steps < 0:
                raise ValueError(f"radial_steps must be >= 0, got {self.radial_steps}")
            if not tuple(self.radial_lam_grid):
                raise ValueError("radial_lam_grid must hold at least one roughness weight")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_fit_learn_radial.py::test_config_radial_defaults_and_validation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/pipeline/config.py tests/test_fit_learn_radial.py
git commit -m "feat(pipeline): FitConfig learned-radial fields and validation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The `learn_radials` stage

**Files:**
- Create: `src/ace_jax/fit/pipeline/radials.py`
- Test: `tests/test_fit_learn_radial.py` (append)

**Interfaces:**
- Consumes: the `FitConfig` fields from Task 1; `FitData` (`fit/pipeline/data.py:14`) with `train`, `meta`, `z`, `model`, `E0`, `r0` and `source`.
- Produces:
  - `RadialResult(NamedTuple)`: `W` (the selected `rnl_Wnlq`, an np.ndarray), `info` (dict, `fit_radial`'s info minus `readout`), `n_fit: int`, `n_val: int`, `relres_max: float` and `seconds: float`.
  - `RADIAL_MAP_STEPS: int = 300`, the per-candidate θ-MAP steps inside `fit_radial`, as the driver uses.
  - `learn_radials(cfg, data, log=print) -> (FitData, RadialResult)`. The returned `FitData` has `model`, `meta` and `z` replaced by the learned ones, the fitted `E0` reapplied, and `source` suffixed `" + learned radials"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fit_learn_radial.py`:

```python
def _data(cfg):
    from ace_jax.fit.pipeline import load_fit_data
    return load_fit_data(cfg, train=str(XYZ), log=QUIET)


@pytest.fixture
def fast(monkeypatch):
    import ace_jax.fit.pipeline.radials as R
    monkeypatch.setattr(R, "RADIAL_MAP_STEPS", 20)


def test_stage_matches_fit_radial_on_the_same_split(fast):
    """The stage is exactly bench/learn_radial/run.py's recipe on the ARD-style split."""
    import jax.numpy as jnp
    from ace_jax.basis.prior import prior_diagonal
    from ace_jax.fit.data import build_dataset
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem
    from ace_jax.fit.pipeline.radials import learn_radials
    from ace_jax.fit.radial_learn import fit_radial
    from ace_jax.fit.radial_model import rnl_degrees, to_analytic
    cfg = _cfg(learn_radial=True, radial_steps=3, radial_lam_grid=(0.0,), radial_val_frac=0.25)
    d = _data(cfg)
    d2, rr = learn_radials(cfg, d, log=QUIET)
    # reference: the driver's construction on the same split
    idx = np.random.default_rng(cfg.seed).permutation(len(d.train))
    nval = max(1, int(round(0.25 * len(d.train))))
    val, fit_ = [d.train[i] for i in idx[:nval]], [d.train[i] for i in idx[nval:]]
    m, _ = to_analytic(d.model, 12)
    ds_fit, ds_val = (build_dataset(cs, d.meta, d.E0, cfg.batch) for cs in (fit_, val))
    gc = GPConfig(r0=2.35, rcut=float(d.meta["rcut"]), n_B=d.meta["n_B"], n_pair=d.meta["n_pair"],
                  NZ=len(d.meta["elements"]), C=cfg.batch)
    X, S = site_features(m, gc, ds_fit)
    ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
    prob = Problem(KernelSpec("cosine", True, gc.D), m, ind, gc,
                   jnp.asarray(prior_diagonal(d.z, d.meta, d.source)), default_prior(2.35))
    W, info = fit_radial(prob, ds_fit, ds_val, m.rnl_Wnlq, lam_grid=(0.0,), steps=3, map_steps=20,
                         rough_weights=1.0 / (1.0 + rnl_degrees(d.meta)) ** 2)
    assert (rr.n_fit, rr.n_val) == (len(fit_), nval)
    assert rr.info["selected"] == info["selected"]
    np.testing.assert_allclose(rr.W, np.asarray(W), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(d2.model.rnl_Wnlq), np.asarray(W), rtol=0, atol=1e-12)
    assert np.asarray(d2.z["rnl_Wnlq"]).shape == np.asarray(W).shape


def test_stage_gate_keeps_init(fast):
    """radial_steps=0: learned == init, the tie goes to init, nothing is marked learned."""
    from ace_jax.fit.pipeline.radials import learn_radials
    from ace_jax.fit.radial_model import to_analytic
    cfg = _cfg(learn_radial=True, radial_steps=0, radial_lam_grid=(0.0,))
    d = _data(cfg)
    d2, rr = learn_radials(cfg, d, log=QUIET)
    assert rr.info["selected"] == "init"
    assert not d2.model.radial_learned and d2.meta.get("radial_learned") is False
    m0, _ = to_analytic(d.model, 12)
    np.testing.assert_allclose(np.asarray(d2.model.rnl_Wnlq), np.asarray(m0.rnl_Wnlq), rtol=0, atol=1e-12)


def test_stage_keeps_fitted_e0(fast):
    """e0='lsq' E0 (set on the model by load_fit_data) survives the reload."""
    from ace_jax.fit.pipeline.radials import learn_radials
    cfg = _cfg(e0="lsq", learn_radial=True, radial_steps=2, radial_lam_grid=(0.0,))
    d = _data(cfg)
    d2, _ = learn_radials(cfg, d, log=QUIET)
    np.testing.assert_allclose(np.asarray(d2.model.E0), d.E0, rtol=0, atol=0)
    np.testing.assert_allclose(d2.E0, d.E0, rtol=0, atol=0)


def test_stage_refuses_factorised_radial():
    import dataclasses
    from ace_jax.fit.pipeline.radials import learn_radials
    cfg = _cfg(learn_radial=True)
    d = _data(cfg)
    d = d._replace(model=dataclasses.replace(d.model, radial_kind="spline_factorised"))
    with pytest.raises(ValueError, match="#31"):
        learn_radials(cfg, d, log=QUIET)


def test_stage_refuses_an_empty_split():
    from ace_jax.fit.pipeline.radials import learn_radials
    cfg = _cfg(learn_radial=True, radial_val_frac=0.999)
    d = _data(cfg)
    with pytest.raises(ValueError, match="radial_val_frac"):
        learn_radials(cfg, d, log=QUIET)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_fit_learn_radial.py -k stage -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.pipeline.radials'`.

- [ ] **Step 3: Implement**

Create `src/ace_jax/fit/pipeline/radials.py`:

```python
"""Learned tensor radials as a fit-pipeline stage (`aj fit --learn-radial`;
docs/dev/specs/2026-10-01-fit-learned-radials-design.md).

bench/learn_radial/run.py's recipe, run before the fit: a seeded hold-out of the
training configs gates fit_radial's candidates (init + one learned per roughness
weight), and the selected radials, with the readout fitted for them, are patched
into the model's npz arrays in memory.  FitData's model/meta/z are swapped for the
patched ones, the fitted E0 reapplied, and the rest of the pipeline then refits on
the FULL training set exactly as for any other model file."""
import io
import time
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ...basis.export import patch_radial_npz
from ...basis.prior import prior_diagonal
from ...eval import load
from ..data import build_dataset
from ..hypers import default_prior
from ..inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ..kernels import KernelSpec
from ..objective import Problem
from ..radial_learn import fit_radial
from ..radial_model import rnl_degrees, to_analytic, with_radial

RADIAL_MAP_STEPS = 300               # per-candidate theta-MAP steps in fit_radial (the driver's)


class RadialResult(NamedTuple):
    W: np.ndarray                    # the selected rnl_Wnlq
    info: dict                       # fit_radial's info, minus the readout
    n_fit: int; n_val: int
    relres_max: float                # to_analytic's worst relative projection residual
    seconds: float


def learn_radials(cfg, data, log=print):
    """(FitData with the learned model, RadialResult)."""
    t0 = time.time()
    n = len(data.train)
    idx = np.random.default_rng(cfg.seed).permutation(n)       # the ARD split convention
    nval = max(1, int(round(cfg.radial_val_frac * n)))
    if nval >= n:
        raise ValueError(f"radial_val_frac={cfg.radial_val_frac} holds out {nval} of {n} training "
                         f"configs and leaves none to learn the radials on")
    val, fit_ = [data.train[i] for i in idx[:nval]], [data.train[i] for i in idx[nval:]]
    try:
        model, relres = to_analytic(data.model, cfg.radial_n_q)
    except ValueError as e:
        raise ValueError(f"learned radials need an analytic or spline tensor radial, got "
                         f"radial_kind={data.model.radial_kind!r} (embedding models are not supported "
                         f"yet, see issue #31): {e}") from e
    relres_max = float(np.max(relres)) if np.size(relres) else 0.0
    meta = data.meta
    ds_fit, ds_val = (build_dataset(cs, meta, data.E0, cfg.batch) for cs in (fit_, val))
    r0 = cfg.r0 if cfg.r0 is not None else data.r0
    gc = GPConfig(r0=r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                  NZ=len(meta["elements"]), C=cfg.batch)
    X, S = site_features(model, gc, ds_fit)
    ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
    prob = Problem(KernelSpec("cosine", True, gc.D), model, ind, gc,
                   jnp.asarray(prior_diagonal(data.z, meta, data.source)), default_prior(r0))
    log(f"learn radials: n_q={cfg.radial_n_q} on {len(fit_)} configs, gate on {nval} "
        f"(to_analytic relres_max={relres_max:.2e})")
    W, info = fit_radial(prob, ds_fit, ds_val, model.rnl_Wnlq, lam_grid=tuple(cfg.radial_lam_grid),
                         steps=cfg.radial_steps, map_steps=RADIAL_MAP_STEPS,
                         rough_weights=1.0 / (1.0 + rnl_degrees(meta)) ** 2, log=log)
    # patch the source arrays in memory: same file-shaped hand-off as a built basis
    src, dst = io.BytesIO(), io.BytesIO()
    np.savez(src, **{k: data.z[k] for k in data.z.files})
    src.seek(0)
    patch_radial_npz(src, dst, with_radial(model, W, learned=info["selected"] != "init"),
                     readout=info["readout"])
    dst.seek(0)
    new_model, new_meta, new_z = load(dst)
    new_model = eqx.tree_at(lambda m: m.E0, new_model, jnp.asarray(data.E0))   # e0='lsq' survives
    rr = RadialResult(np.asarray(W), {k: v for k, v in info.items() if k != "readout"}, len(fit_), nval,
                      relres_max, time.time() - t0)
    log(f"learn radials: selected {info['selected']} in {rr.seconds:.1f}s")
    return data._replace(model=new_model, meta=new_meta, z=new_z,
                         source=(data.source or "model") + " + learned radials"), rr
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -n 0 tests/test_fit_learn_radial.py -k stage -v`
Expected: PASS (5 tests).

If `load(dst)` rejects a BytesIO, check how `_model_source` hands one on (`fit/pipeline/data.py`, the `buf` path). It must behave the same.

If `d2.meta.get("radial_learned")` is missing rather than False, check that `patch_radial_npz` writes `meta["radial_learned"]` (it does, `basis/export.py:143`) and that `load` keeps meta keys.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/pipeline/radials.py tests/test_fit_learn_radial.py
git commit -m "feat(pipeline): learn_radials stage -- held-out-gated VarPro, in-memory patch

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire the stage into `fit()` and the outputs

**Files:**
- Modify: `src/ace_jax/fit/pipeline/run.py` (`FitResult`, about line 12; `fit()`, about line 17)
- Modify: `src/ace_jax/fit/pipeline/outputs.py` (`write_outputs`, about line 60)
- Test: `tests/test_fit_learn_radial.py` (append)

**Interfaces:**
- Consumes: `learn_radials(cfg, data, log) -> (FitData, RadialResult)` and `RadialResult` from Task 2.
- Produces:
  - `FitResult.radial` (`RadialResult | None`, the last field, default None);
  - `on_stage("radial", RadialResult)`;
  - `write_outputs` writes `out/radial_info.json` with the keys `selected`, `scores`, `n_q`, `lam_grid`, `val_frac`, `to_analytic_relres_max`, `n_fit`, `n_val` and `seconds`, whenever `res.radial is not None`, in every layout.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_fit_learn_radial_end_to_end(fast, tmp_path):
    """fit() runs the stage, refits on the full training set, saves the learned
    model (marked when learned) and radial_info.json."""
    import json
    from ace_jax import ACECalculator
    from ace_jax.fit.pipeline import fit, write_outputs
    cfg = _cfg(learn_radial=True, radial_steps=4, radial_lam_grid=(0.0,))
    d = _data(cfg)
    staged = {}
    res = fit(cfg, d, log=QUIET, on_stage=lambda k, v: staged.__setitem__(k, v))
    assert res.radial is staged["radial"]
    assert len(res.data.ds_train.y_E.reshape(-1)) >= len(d.train)            # full train, not the fit split
    write_outputs(res, tmp_path, layout=("cli",), log=QUIET)
    info = json.loads((tmp_path / "radial_info.json").read_text())
    assert set(info) >= {"selected", "scores", "n_q", "lam_grid", "val_frac", "to_analytic_relres_max",
                         "n_fit", "n_val", "seconds"}
    z = np.load(tmp_path / "model.npz")
    np.testing.assert_allclose(z["rnl_Wnlq"], res.radial.W, rtol=0, atol=1e-12)
    learned = json.loads(bytes(z["meta_json"]).decode())["radial_learned"]
    assert learned == (info["selected"] != "init")
    if learned:
        assert ACECalculator(str(tmp_path / "model.npz"), lean=True).splined


def test_fit_learn_radial_on_built_basis(fast, tmp_path, monkeypatch):
    from test_basis_build import _primed_cache
    from ace_jax.basis.model import BasisSpec
    from ace_jax.fit.pipeline import fit
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cfg = _cfg(model=BasisSpec(order=3, max_degree=10, coupling_cache_dir=_primed_cache(tmp_path)),
               r0=None, e0="lsq", learn_radial=True, radial_steps=2, radial_lam_grid=(0.0,))
    res = fit(cfg, _data(cfg), log=QUIET)
    assert res.radial is not None and res.radial.n_val >= 1


def test_fit_learn_radial_then_gp(fast, tmp_path):
    """The radial stage is linear; a GP final fit after it still runs and saves."""
    from ace_jax.fit.pipeline import fit, write_outputs
    cfg = _cfg(arm="gp", m_per_species=2, map_steps=5, learn_radial=True, radial_steps=2,
               radial_lam_grid=(0.0,))
    res = fit(cfg, _data(cfg), log=QUIET)
    write_outputs(res, tmp_path, layout=("cli",), log=QUIET)
    z = np.load(tmp_path / "gp_model.npz")
    np.testing.assert_allclose(z["ace/rnl_Wnlq"], res.radial.W, rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_fit_learn_radial.py -k "fit_learn_radial" -v`
Expected: FAIL with `AttributeError: 'FitResult' object has no attribute 'radial'`.

- [ ] **Step 3: Implement**

In `run.py`, add the last field of `FitResult`:

```python
    radial: object = None                # RadialResult when cfg.learn_radial (last: positional use unaffected)
```

In `fit()`, right after `stage("data", data)` and before `b = build_problem(cfg, data)`:

```python
    radial = None
    if cfg.learn_radial:
        from .radials import learn_radials
        data, radial = learn_radials(cfg, data, log=log)
        stage("radial", radial)
```

Pass `radial` as the last `FitResult(...)` argument in the `return`, after `ard`. Add `"radial"` to the timings if `radial` is set: `tm["radial"] = radial.seconds` next to `tm["ard"]`. Extend the docstring's stage list with `"radial" -> RadialResult when cfg.learn_radial`.

In `outputs.py`, `write_outputs`, right after the `theta_map.json` dump (so every layout gets it):

```python
    if res.radial is not None:
        r, c = res.radial, res.config
        _dump(out / "radial_info.json", {
            "selected": r.info["selected"], "scores": r.info["scores"], "n_q": c.radial_n_q,
            "lam_grid": list(c.radial_lam_grid), "val_frac": c.radial_val_frac,
            "to_analytic_relres_max": r.relres_max, "n_fit": r.n_fit, "n_val": r.n_val,
            "seconds": r.seconds})
```

(`_dump` already JSON-serialises numpy scalars and arrays; it's used the same way for `metrics.json`. If `scores` holds numpy floats it stays serialisable.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -n 0 tests/test_fit_learn_radial.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/pipeline/run.py src/ace_jax/fit/pipeline/outputs.py tests/test_fit_learn_radial.py
git commit -m "feat(pipeline): fit() runs the learned-radial stage; radial_info.json

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The CLI flags, the `FitConfig` mapping, the stray-option check, and the run file

**Files:**
- Modify: `src/ace_jax/cli.py`:
  - `_add_fit_args` (about line 20): the flags, after `--ard-val-frac`;
  - `_fit_config` (about line 101): the mapping;
  - `_parse`: the stray-option check.
- Modify: `src/ace_jax/runfile.py:18` (`COMMA_KEYS`)
- Test: `tests/test_fit_learn_radial.py` (append)

**Interfaces:**
- Consumes: the `FitConfig` fields from Task 1; `radial_info.json` from Task 3; `runfile.explicit_dests(parser, argv)` (it exists: the dests given explicitly in argv).
- Produces:
  - **Flags** (dest names): `--learn-radial` (`learn_radial`, store_true), `--radial-n-q` (`radial_n_q`, int, 12), `--radial-steps` (`radial_steps`, int, 40), `--radial-lam-grid` (`radial_lam_grid`, str, `"0,1e-2"`) and `--radial-val-frac` (`radial_val_frac`, float, 0.2).
  - **Error:** a `--radial-*` option typed without `--learn-radial` is a `SystemExit` from `parser.error` with the message `--radial-<x> needs --learn-radial`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
FAST_CLI = ["--model", str(MODEL), "--r0", "2.35", "--train", str(XYZ), "--energy-key", "dft_energy",
            "--force-key", "dft_force", "--virial-key", "dft_virial", "--m-per-species", "0",
            "--rungs", "map", "--map-steps", "20", "--opt", "adam", "--configs-per-batch", "4"]


def test_cli_learn_radial_writes_info_and_reproduces(fast, tmp_path):
    import json
    import yaml
    from ace_jax.cli import main
    a = tmp_path / "a"
    main(["fit", *FAST_CLI, "--learn-radial", "--radial-steps", "3", "--radial-lam-grid", "0",
          "--out", str(a)])
    info = json.loads((a / "radial_info.json").read_text())
    assert info["lam_grid"] == [0.0] and info["n_q"] == 12
    y = yaml.safe_load((a / "fit.yaml").read_text())
    assert y["learn_radial"] is True and y["radial_steps"] == 3 and y["radial_lam_grid"] in ("0", [0.0], "0.0")
    b = tmp_path / "b"
    main(["fit", "--config", str(a / "fit.yaml"), "--out", str(b)])
    np.testing.assert_allclose(np.load(b / "model.npz")["rnl_Wnlq"], np.load(a / "model.npz")["rnl_Wnlq"],
                               rtol=0, atol=1e-10)   # two separate runs


def test_cli_radial_option_needs_learn_radial(capsys):
    from ace_jax.cli import _parse
    with pytest.raises(SystemExit):
        _parse(["fit", *FAST_CLI, "--radial-steps", "3", "--out", "o"])
    assert "--radial-steps needs --learn-radial" in capsys.readouterr().err


def test_yaml_radial_defaults_without_learning_are_fine(tmp_path):
    """A resolved fit.yaml records every dest, radial defaults included; re-running
    it with learn_radial false must not trip the stray-option check."""
    import yaml
    from ace_jax.cli import _parse
    f = tmp_path / "fit.yaml"
    f.write_text(yaml.safe_dump({"model": str(MODEL), "r0": 2.35, "train": str(XYZ), "out": "o",
                                 "learn_radial": False, "radial_steps": 40, "radial_n_q": 12,
                                 "radial_lam_grid": "0,0.01", "radial_val_frac": 0.2}))
    a = _parse(["fit", "--config", str(f)])
    assert a.learn_radial is False and a.radial_steps == 40


def test_cli_lam_grid_parses_to_floats():
    from ace_jax.cli import _fit_config, _parse
    a = _parse(["fit", *FAST_CLI, "--learn-radial", "--radial-lam-grid", "0,1e-3,1e-2", "--out", "o"])
    assert _fit_config(a).radial_lam_grid == (0.0, 1e-3, 1e-2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_fit_learn_radial.py -k "cli or yaml" -v`
Expected: FAIL with `SystemExit` (unrecognised `--learn-radial`) or an `AttributeError` on `a.learn_radial`.

- [ ] **Step 3: Implement**

In `_add_fit_args`, after the `--ard-val-frac` argument:

```python
    p.add_argument("--learn-radial", action="store_true",
                   help="learn the tensor radials (VarPro, held-out gate) before the fit; the saved model "
                        "is marked radial_learned and splined at deploy time")
    p.add_argument("--radial-n-q", type=int, default=12,
                   help="tensor-radial polynomial span after widening (with --learn-radial)")
    p.add_argument("--radial-steps", type=int, default=40,
                   help="L-BFGS steps per roughness weight (with --learn-radial)")
    p.add_argument("--radial-lam-grid", default="0,1e-2",
                   help="comma-separated relative roughness weights the gate picks among, "
                        "alongside the initial radials (with --learn-radial)")
    p.add_argument("--radial-val-frac", type=float, default=0.2,
                   help="fraction of the training configs held out for the gate (with --learn-radial)")
```

In `_fit_config`, add these to the `FitConfig(...)` call:

```python
        learn_radial=a.learn_radial, radial_n_q=a.radial_n_q, radial_steps=a.radial_steps,
        radial_lam_grid=tuple(float(x) for x in str(a.radial_lam_grid).split(",") if x.strip()),
        radial_val_frac=a.radial_val_frac,
```

In `_parse`, right before `if a.cmd == "fit": _check_fit_args(subs["fit"], a)`, add the check against the options **typed**, so file defaults don't count:

```python
    if a.cmd == "fit" and not a.learn_radial:
        from . import runfile
        typed = runfile.explicit_dests(subs["fit"], argv[argv.index(cmd) + 1:])
        stray = sorted(d for d in typed if d.startswith("radial_"))
        if stray:
            subs["fit"].error(f"--{stray[0].replace('_', '-')} needs --learn-radial")
```

In `runfile.py`:

```python
COMMA_KEYS = ("rungs", "elements", "radial_lam_grid")
```

(The run-file writer, `runfile.resolved`, writes dests as they are on the namespace. `radial_lam_grid` is a string there, `"0"` or `"0,1e-2"`, and `_value` joins a list back into a comma string on read, so both forms reproduce.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -n 0 tests/test_fit_learn_radial.py -v`
Expected: PASS (all tests).

Then run the neighbouring CLI and run-file tests (one process): `JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_runfile.py tests/test_runfile_roundtrip.py tests/test_cli_fit_basis.py tests/test_names.py tests/test_no_backend_names.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/cli.py src/ace_jax/runfile.py tests/test_fit_learn_radial.py
git commit -m "feat(cli): aj fit --learn-radial and the --radial-* options

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Docs and the full check

**Files:**
- Modify: `skills/ace-jax/SKILL.md` (the `aj fit` options section, and the learned-radials notes around "Learned radials are splined")
- Modify: `README.md` (the `aj fit` usage section)
- Modify: `CLAUDE.md` (the Layout bullet for `fit/pipeline/`: add `radials.py`, the learned-radial stage)

- [ ] **Step 1: Write the docs**

**SKILL.md** gets a short "Learned radials in `aj fit`" block:

```markdown
- **Learned radials: `aj fit --learn-radial`.** Learns the tensor radials before
  the fit (VarPro over the training configs, gated on a seeded
  `--radial-val-frac` hold-out, default 0.2), then fits as usual on the full
  training set. Options: `--radial-n-q 12`, `--radial-steps 40`,
  `--radial-lam-grid 0,1e-2`. Writes `out/radial_info.json` (the gate's
  selection and scores). The saved model is marked `radial_learned`, so
  `ACECalculator`/`export_lammps` spline it. Works with `--model` and with
  `--order/--max-degree`; not with embedding models (issue #31). Advanced priors:
  `ace_jax.fit.radial_learn.fit_radial`.
```

**README.md:** add one line to the `aj fit` example block: `aj fit --order 3 --max-degree 10 --train train.xyz --learn-radial --out fit/   # + learned radials`.

**CLAUDE.md:** in the Layout `fit/` bullet, change "`pipeline/` is the `aj fit` pipeline." to "`pipeline/` is the `aj fit` pipeline (`radials.py`: the `--learn-radial` stage)."

- [ ] **Step 2: Lint and the name guards**

Run: `uv run pre-commit run --all-files && JAX_PLATFORMS=cpu uv run pytest -n 0 tests/test_names.py tests/test_no_backend_names.py -q`
Expected: all pass. User docs must not mention Julia; `test_no_backend_names` checks this.

- [ ] **Step 3: The full suite, once, within the memory budget**

Run: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ACEJAX_TEST_WORKERS=4 uv run pytest -q`
Expected: all pass. The only new tests are in `tests/test_fit_learn_radial.py`.

- [ ] **Step 4: Commit**

```bash
git add skills/ace-jax/SKILL.md README.md CLAUDE.md
git commit -m "docs: aj fit --learn-radial in SKILL.md, README and CLAUDE.md

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
