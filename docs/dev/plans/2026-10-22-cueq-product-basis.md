# cuequivariance product basis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `aa_dag="cueq"`: an opt-in GPU product-basis form for lean linear ACE models (single species, then TrACE),
in ASE/JAX evaluation and lammps-jax GPU bundles, with the Si linear ACE ladder re-benchmarked on the docs site.

**Architecture:** A new module `ace_jax/eval/cueq.py` builds cuequivariance `SegmentedPolynomial`s from a lean
model's product specs and folded readout (chunked at 1000 monomials, atoms in the u mode, U = 32) and attaches
them as static plan fields on `ACEModel`, read by `_readout_folded` exactly where the DAG plan is read today.
`with_aa_dag(model, "cueq")` dispatches to it; `export_lammps` records cueq's plan compiles with lammps-jax's
`contrib/ffi-replay` recorder, ships `<bundle>.ffi-replay` and a warmed NVRTC cache, and sizes row blocks for cueq.

**Tech Stack:** cuequivariance / cuequivariance-jax / cuequivariance-ops-jax-cu12 0.13.x, JAX 0.11.x (x64),
lammps-jax (bundle export, `contrib/ffi-replay`), pytest.

**Spec:** `docs/dev/specs/2026-10-22-cueq-product-basis-design.md`. Reference implementation (throwaway, but
correct and parity-tested): branch `exp/cueq-product`, `bench/cueq_product/cueq_ace.py` (`build_poly`,
`build_polys`, `product_energy` layout "uU"), `trace_cueq.py` (TrACE), `lammps_hook.py` (recorder),
README.md (all numbers).

## Global Constraints

- `aa_dag="auto"` never selects "cueq"; cueq is opt-in only.
- Supported: lean linear `ACEModel` with one species (phase 1); species-embedding (`rnl_embedding`) models (phase 2).
  Refused with a clear error: cueq not installed, PACE/other models, categorical multi-species, non-CUDA platform.
- No module-level import of cuequivariance anywhere (`tests/test_core_deps.py`); import inside functions.
- `cueq` extra pinned to the tested 0.13.x; Linux x86_64 + CUDA 12 only; not in `cuda` or the dev group.
- Kernels of at most `CHUNK = 1000` monomials; atoms in the u mode with `U = 32`.
- Bundle record: `ace_jax.aa_dag = "cueq"` and `ace_jax.cueq = {"version", "targets": ["uniform_1d_cuda"],
  "replay": "<bundle>.ffi-replay", "cache": "<bundle>.cueq-cache", "handlers": "<LAMMPS_JAX_FFI_HANDLERS line>"}`.
- Row blocks for cueq bundles: the largest of 8192, 4096 (halving, at least MIN_BLOCK_ROWS) that fits the budget.
- Parity: float64 site energies ≤ 1e-12, forces ≤ 1e-10 relative to the full model; bundle gate at the bench's
  tolerances (1e-10 eV/atom, 1e-8 eV/Å).
- float64 is the library caller's choice (CLAUDE.md); tests set `jax_enable_x64` at module top.
- User docs follow ASD-STE100 (CLAUDE.md "User docs style").
- GPU tests are opt-in: they skip without a CUDA device or cueq (`conftest.require_optional`), and are run by hand
  on lestrade / Sulis, not in GitHub CI.

## Review Focus

1. **A readout swap on `ACECalculator` with `aa_dag="cueq"`** must re-plan (baked coefficients), not keep the old
   kernels: Task 4 tests energies after `calc.model = <new readout>`.
2. **Atom counts not a multiple of U = 32** (including fewer than 32 atoms): padded rows must not change energies
   or forces — Task 3 tests 1, 31, 33 and 216 atoms.
3. **The NVRTC cache directory must be set before the first cueq compile in the export process**, or the bundle
   ships an empty cache: Task 5 asserts the cache directory is non-empty after export.
4. **A model the form does not support** (two-species categorical, PACE) must raise, never silently fall back:
   Task 2 tests each refusal.
5. **The blocked dense path** (`blk_aa_specs`, what bundles evaluate) and the unblocked path must both use cueq
   plans: Task 3 tests a lean model through both `site_energies` and the blocked dense evaluation.

---

### Task 1: `cueq` extra and availability

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Create: `src/ace_jax/eval/cueq.py`
- Test: `tests/test_cueq.py`

**Interfaces:**
- Produces: `cueq.available() -> bool`, `class CueqUnavailable(ImportError)`, `cueq.require() -> None` (raises
  `CueqUnavailable` naming the extra), constants `CHUNK = 1000`, `U = 32`, `TARGET = "uniform_1d_cuda"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cueq.py
import jax
jax.config.update("jax_enable_x64", True)
import pytest  # noqa: E402


def test_require_names_the_extra_when_cueq_is_missing(monkeypatch):
    import builtins
    from ace_jax.eval import cueq
    real = builtins.__import__
    def fake(name, *a, **k):
        if name.startswith("cuequivariance"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert not cueq.available()
    with pytest.raises(cueq.CueqUnavailable, match=r"ace-jax\[cueq\]"):
        cueq.require()
```

- [ ] **Step 2: Run it:** `uv run pytest tests/test_cueq.py -q` — Expected: FAIL (no module `ace_jax.eval.cueq`).
- [ ] **Step 3: Implement**

```python
# src/ace_jax/eval/cueq.py
"""cuequivariance (NVIDIA cueq) as a GPU form of the lean ACE product basis (`aa_dag="cueq"`).

Design: docs/dev/specs/2026-10-22-cueq-product-basis-design.md.  The folded site energy
e_i = sum_v ctilde[v, z_i] prod_t A_i[s_v,t] is a cueq SegmentedPolynomial: one SegmentedTensorProduct per order
k ([A]*k -> out), one path per monomial, evaluated by cuequivariance_jax.segmented_polynomial(method="uniform_1d")
with atoms in the u mode (U atoms per group: the obvious batch-axis layout is 6x slower) and the polynomial split
into kernels of at most CHUNK monomials (NVRTC compile grows superlinearly with paths).  cueq is optional:
nothing here imports it at module level."""
CHUNK = 1000
U = 32
TARGET = "uniform_1d_cuda"


class CueqUnavailable(ImportError):
    pass


def available():
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_jax  # noqa: F401
    except ImportError:
        return False
    return True


def require():
    if not available():
        raise CueqUnavailable("aa_dag='cueq' needs cuequivariance: pip install 'ace-jax[cueq]' (Linux, CUDA 12)")
```

and in `pyproject.toml`, after `cuda = [...]`:

```toml
# aa_dag="cueq": NVIDIA cuequivariance as the GPU product basis (single-species and species-embedding models;
# docs/dev/specs/2026-10-22-cueq-product-basis-design.md).  Linux x86_64 + CUDA 12 only; tested at 0.13.x.
cueq = ["cuequivariance>=0.13,<0.14", "cuequivariance-jax>=0.13,<0.14",
        "cuequivariance-ops-jax-cu12>=0.13,<0.14; sys_platform == 'linux'"]
```

Then `uv lock` (the extra is not installed by default; check `uv sync` still resolves without it).
- [ ] **Step 4: Run** `uv run pytest tests/test_cueq.py tests/test_core_deps.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(eval): cueq extra and availability check`

### Task 2: Polynomial plans and refusals (`polynomial`, `check_supported`)

**Files:**
- Modify: `src/ace_jax/eval/cueq.py`
- Test: `tests/test_cueq.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `cueq.check_supported(model) -> None` (raises `ValueError` for: not an `ACEModel`; categorical model
  with `ctilde.shape[1] > 1`; not folded); `cueq.polynomial(specs, coef, u=U, chunk=CHUNK) -> tuple[(poly, cols)]`
  (port of the spike's `build_polys` with baked coefficients); `class _Static` (hash-by-identity holder);
  `cueq.n_A_of(specs) -> int`.

- [ ] **Step 1: Write the failing tests** (CPU: cueq's descriptor package is pure Python; skip if absent)

```python
def _si():
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.eval.model import lean
    m, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    return lean(m)


def test_refuses_multi_species_categorical_and_pace():
    from conftest import FIXTURE_DIR, pace_fixture
    from ace_jax import load
    from ace_jax.eval import cueq
    from ace_jax.eval.model import lean
    m2, _, _ = load(str(FIXTURE_DIR / "sige_fitted.npz"))            # two species, categorical
    with pytest.raises(ValueError, match="one species"):
        cueq.check_supported(lean(m2))
    with pytest.raises(ValueError, match="linear ACEModel"):
        cueq.check_supported(load(str(pace_fixture()))[0])


def test_polynomial_chunks_cover_every_monomial_once():
    from conftest import require_optional
    require_optional("cuequivariance")
    import numpy as np
    from ace_jax.eval import cueq
    m = _si()
    specs, coef = m.aa_specs, np.asarray(m.ctilde[:, 0])
    plans = cueq.polynomial(specs, coef, chunk=7)
    cols = np.concatenate([c for _, c in plans])
    assert sorted(cols.tolist()) == list(range(sum(np.shape(s)[0] for s in specs if np.size(s))))
    assert all(len(c) <= 7 for _, c in plans)
```

(Check `tests/conftest.py` and `fixtures/` for the exact fixture names — the two-species categorical fixture and
the PACE fixture helper — and use those; the names above are the expected ones.)
- [ ] **Step 2: Run** `uv run pytest tests/test_cueq.py -q` — Expected: FAIL (no `check_supported` / `polynomial`).
- [ ] **Step 3: Implement** in `cueq.py`: `_Static` and `polynomial` copied from the spike's `_Static`,
  `build_poly` (the `coef is not None` branch only, `wu=False`) and `build_polys` (drop the `coef.ndim == 2`
  branch), renamed; and

```python
def check_supported(model):
    from .model import ACEModel
    if not isinstance(model, ACEModel):
        raise ValueError(f"aa_dag='cueq' needs a linear ACEModel, got {type(model).__name__}")
    if not model.folded:
        raise ValueError("aa_dag='cueq' needs the folded (lean) model: apply lean() first")
    if getattr(model, "rnl_embedding", None) is None and model.ctilde.shape[1] != 1:
        raise ValueError("aa_dag='cueq' supports one species, or species-embedding (TrACE) models; "
                         f"this categorical model has {model.ctilde.shape[1]} species")


def n_A_of(specs):
    import numpy as np
    return 1 + max(int(np.asarray(s).max()) for s in specs if np.size(s))
```

(Until Task 8, an embedding model also raises: add `if getattr(model, "rnl_embedding", None) is not None: raise
ValueError("... TrACE support: not yet")` and remove it in Task 8.)
- [ ] **Step 4: Run** — Expected: PASS (the polynomial test skips without cuequivariance).
- [ ] **Step 5: Commit** `feat(eval): cueq polynomial plans and refusals`

### Task 3: cueq evaluation in the folded readout (`with_cueq`, `with_aa_dag(..., "cueq")`)

**Files:**
- Modify: `src/ace_jax/eval/cueq.py`, `src/ace_jax/eval/model.py:180-183, 408-414`, `src/ace_jax/eval/aa_dag.py:261-296`
- Test: `tests/test_cueq.py` (GPU, opt-in)

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `cueq.with_cueq(model) -> ACEModel` (sets `cueq_plan`, `blk_cueq_plan`, clears `aa_dag`,
  `blk_aa_dag`); `cueq.product_energy(plans, A, u=U) -> (n,)`; `ACEModel.cueq_plan` / `blk_cueq_plan`
  (`eqx.field(static=True, default=None)`, `_Static` holders); `with_aa_dag(model, "cueq")`;
  `resolve_aa_dag(model, "auto")` never returns "cueq".

- [ ] **Step 1: Write the failing tests**

```python
def test_auto_never_picks_cueq():
    from ace_jax.eval.aa_dag import resolve_aa_dag
    assert resolve_aa_dag(_si(), "auto") != "cueq"


def _gpu():
    from conftest import require_optional
    require_optional("cuequivariance_jax")
    if jax.default_backend() != "gpu":
        pytest.skip("cueq evaluation needs a CUDA device")


@pytest.mark.parametrize("n_atoms", [1, 31, 33, 216])
def test_cueq_matches_the_full_model(n_atoms):
    _gpu()
    import numpy as np
    from ase.build import bulk
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    full, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat(3)[:n_atoms] if n_atoms <= 216 else None
    at.rattle(0.05, seed=1)
    ref = ACECalculator((full, meta), lean=False, aa_dag=None)
    cq = ACECalculator((full, meta), aa_dag="cueq")
    at.calc = ref; e0, f0 = at.get_potential_energy(), at.get_forces()
    at.calc = cq; e1, f1 = at.get_potential_energy(), at.get_forces()
    assert abs(e1 - e0) <= 1e-12 * max(abs(e0), 1)
    assert np.abs(f1 - f0).max() <= 1e-10 * max(np.abs(f0).max(), 1)


def test_cueq_uses_the_blocked_dense_plan_too():
    _gpu()
    from ace_jax.eval import cueq
    m = cueq.with_cueq(_si())
    assert m.cueq_plan is not None and (m.blk_aa_specs is None or m.blk_cueq_plan is not None)
    assert m.aa_dag is None and m.blk_aa_dag is None
```

(Build the `n_atoms` cells so each is a periodic or cluster structure the calculator accepts — a 1-atom periodic
cell and 31/33-atom clusters are fine for parity; the point is the padding to U.)
- [ ] **Step 2: Run** `uv run pytest tests/test_cueq.py -q` on a GPU host (lestrade) — Expected: FAIL.
- [ ] **Step 3: Implement**
  - `model.py`: after `blk_aa_dag`, add
    ```python
    # cuequivariance product-basis plans (`eval.cueq.with_cueq`, aa_dag="cueq"), None = ace-jax's own form; static
    # holders of the chunked polynomials for aa_specs / blk_aa_specs, read by the folded readout only.
    cueq_plan: object = eqx.field(static=True, default=None)
    blk_cueq_plan: object = eqx.field(static=True, default=None)
    ```
    and in `_readout_folded`, before the `dag` branch:
    ```python
    plan = self.blk_cueq_plan if specs is not None else self.cueq_plan
    if plan is not None:                      # cuequivariance product basis (`eval.cueq`)
        from .cueq import product_energy
        e = product_energy(plan.v, A)
    elif dag is not None:
    ```
  - `cueq.py`: `product_energy` = the spike's layout "uU" branch without weights (pad to G·U rows, reshape to
    (G, n_A·U), sum the chunk kernels, reshape, drop padding), and
    ```python
    def with_cueq(model):
        require(); check_supported(model)
        import dataclasses, numpy as np
        coef = np.asarray(model.ctilde[:, 0])
        plan = lambda specs: _Static(polynomial(specs, coef))
        return dataclasses.replace(model, aa_dag=None, blk_aa_dag=None, cueq_plan=plan(model.aa_specs),
                                   blk_cueq_plan=None if model.blk_aa_specs is None else plan(model.blk_aa_specs))
    ```
  - `cueq.py`, in `with_cueq`: if `CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR` is unset, `warnings.warn` once ("cueq kernels
    compile on first use (minutes for large models); set CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR to reuse them") — the
    spec's missing-cache edge case; test it with `pytest.warns` in the CPU test file (monkeypatch `require` and
    `polynomial` so no GPU is needed).
  - `aa_dag.py`: in `with_aa_dag`, `if mode == "cueq": from .cueq import with_cueq; return with_cueq(model)`
    before the `PLAN_MODES` check; on `mode is None` also clear `cueq_plan`/`blk_cueq_plan`; docstrings name "cueq".
- [ ] **Step 4: Run** on lestrade: `uv run pytest tests/test_cueq.py -q` (GPU tests run) and on moriarty CPU
  `uv run pytest tests/test_cueq.py tests/test_aa_dag.py tests/test_lean.py -q` — Expected: PASS / GPU tests skip on CPU.
- [ ] **Step 5: Commit** `feat(eval): aa_dag="cueq" product basis in the folded readout`

### Task 4: `ACECalculator(aa_dag="cueq")` and readout swaps

**Files:**
- Modify: `src/ace_jax/calc/point.py:130-140` (docstring), the `model` setter (already re-applies `with_aa_dag`)
- Test: `tests/test_cueq.py`

**Interfaces:**
- Consumes: Task 3. Produces: `calc.aa_dag == "cueq"` when used.

- [ ] **Step 1: Write the failing test**

```python
def test_readout_swap_replans():
    _gpu()
    import dataclasses
    import numpy as np
    from ase.build import bulk
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    m, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    calc = ACECalculator((m, meta), aa_dag="cueq")
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2); at.rattle(0.05, seed=2); at.calc = calc
    e_a = at.get_potential_energy()
    m2 = dataclasses.replace(m, WB=2.0 * m.WB)                  # a different readout, same basis
    calc.model = m2; at.calc = calc
    ref = ACECalculator((m2, meta), lean=False, aa_dag=None)
    e_b = at.get_potential_energy()
    at.calc = ref
    assert calc.aa_dag == "cueq" and abs(e_b - at.get_potential_energy()) <= 1e-12 * abs(e_b) and e_b != e_a
```

- [ ] **Step 2: Run** on lestrade — Expected: FAIL only if the setter keeps stale plans (it should already pass
  once Task 3 lands; if it passes, keep the test as the regression pin and update the docstring only).
- [ ] **Step 3: Implement** the docstring: list "cueq" among `aa_dag` values (opt-in GPU form, `eval.cueq`; a
  readout swap re-plans and recompiles its kernels — minutes at o4d20 without the NVRTC cache).
- [ ] **Step 4: Run** `uv run pytest tests/test_cueq.py tests/test_calc_layout.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(calc): ACECalculator(aa_dag="cueq")`

### Task 5: GPU bundles: replay recording, cache, metadata, row blocks, `aj export --aa-dag`

**Files:**
- Modify: `src/ace_jax/export/lammps.py` (`export_lammps`, `block_rows_for` call site), `src/ace_jax/eval/cueq.py`,
  `src/ace_jax/cli.py` (`cmd_export`, its parser)
- Test: `tests/test_export_lammps.py`, `tests/test_cli_export.py`, `tests/test_cueq.py`

**Interfaces:**
- Consumes: Tasks 1-3; lammps-jax `export_model(..., custom_call_targets=...)`; lammps-jax
  `contrib/ffi-replay/ffi_replay_record.py` (`install(library, symbol)`, `dump(path, compile_symbol, execute_symbol)`).
- Produces: `cueq.recorder() -> module` (finds `ffi_replay_record` via `$LAMMPS_JAX_CONTRIB`, else next to the
  installed lammps-jax checkout; raises `CueqUnavailable` with instructions); `cueq.CUEQ_BLOCK_ROWS = (8192, 4096)`;
  bundle keys per Global Constraints; `aj export --aa-dag {auto,chain,greedy,prefix,cueq}`.

- [ ] **Step 1: Write the failing tests**
  - CPU (`tests/test_export_lammps.py`, lammps-jax required, cueq mocked): monkeypatch `cueq.with_cueq` to return
    the model unchanged, `cueq.recorder` to a stub with `install`/`dump`, and assert the bundle's `ace_jax.aa_dag ==
    "cueq"`, the `ace_jax.cueq` keys, `custom_call_targets` passed through, the sidecar paths, and `block_rows` the
    cueq rule (8192 when it fits a large `device_memory`, 4096 when only that fits).
  - CPU (`tests/test_cli_export.py`): `aj export --aa-dag cueq` on a two-species fixture exits non-zero with the
    refusal message; `--aa-dag chain` maps to `aa_dag=None`.
  - GPU (`tests/test_cueq.py`, lammps-jax + cueq + CUDA): export the Si fixture with `aa_dag="cueq"` to a tmp path;
    assert `<bundle>.ffi-replay` exists, `<bundle>.cueq-cache/` is non-empty (Review Focus 3).
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** (follow PR #112's `ace_jax.ffi` plumbing and the spike's `lammps_hook.py`):
  - At the top of `export_lammps` when `aa_dag == "cueq"`: `cueq.require()`, `cueq.check_supported(model)`, refuse
    a non-CUDA `platform`; set `os.environ["CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR"]` to `<path stem>.cueq-cache`
    (created) **before** any cueq import in the process; `rec = cueq.recorder(); rec.install(cuequivariance_ops_jax.
    _common.library, "compile_uniform_1d_ctypes")`. Document that one bundle is exported per process (plan handles
    are registry indices) and raise if the recorder already holds plans from an earlier export.
  - Replace the `with_aa_dag` call for "cueq" by `cueq.with_cueq` (same place, after `lean`, before `with_dtype`).
  - `block_rows`: when `aa_dag == "cueq"` and no explicit `block_rows`, try `CUEQ_BLOCK_ROWS` (each capped at rows)
    then the halvings down to `MIN_BLOCK_ROWS`, taking the first whose estimate fits the budget (reuse
    `block_rows_for(..., block_rows=b)` per candidate).
  - Pass `custom_call_targets=(cueq.TARGET,)` to `export_model`; after it returns, `rec.dump(<stem>.ffi-replay,
    "compile_uniform_1d_ctypes", "execute_uniform_1d_cuda_handler")`; write `bundle["ace_jax"]["cueq"]`
    = {version (`cuequivariance.__version__`), targets, replay, cache, handlers:
    `"uniform_1d_cuda=<shim>.so:ffi_replay_handler"`}.
  - `cli.py`: `--aa-dag` choice (default `auto`; `chain` → None); for cueq print the environment lines
    (`LAMMPS_JAX_FFI_HANDLERS`, `FFI_REPLAY_FILE`, `FFI_REPLAY_LIB`, `CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR`,
    `LD_LIBRARY_PATH` note), as #112 prints its handler line.
- [ ] **Step 4: Run** on moriarty `uv run pytest tests/test_export_lammps.py tests/test_cli_export.py tests/test_cueq.py -q`
  (CPU parts) and on lestrade the GPU export test; then a 64-atom LAMMPS gate by hand with the bench (Task 7's driver).
- [ ] **Step 5: Commit** `feat(export): cueq GPU bundles (ffi-replay sidecar, shipped NVRTC cache, cueq row blocks, aj export --aa-dag)`

### Task 6: Documentation

**Files:**
- Modify: `docs/user/howto/lammps.md`, `skills/ace-jax/SKILL.md`, `CLAUDE.md` (Architecture: the new `aa_dag` value,
  `eval/cueq.py`, the bundle record), `docs/user/concepts.md` Terms table only if a new term is introduced.

- [ ] **Step 1:** `howto/lammps.md`: a section "cuequivariance on the GPU", ASD-STE style, numbered procedure:
  install `ace-jax[cueq]`; build lammps-jax's `contrib/ffi-replay` shim (its README command); export with
  `aj export ... --aa-dag cueq`; copy the bundle, `.ffi-replay` and `.cueq-cache` together; set the printed
  environment lines; run. A `!!! warning` on export time (~13 min at o4d20) and cold start without the cache, and a
  note that "auto" never selects it and that it supports one species (TrACE after phase 2).
- [ ] **Step 2:** `SKILL.md` and `CLAUDE.md`: `aa_dag="cueq"`, refusals, the bundle record, the opt-in GPU tests.
- [ ] **Step 3:** `uv run pytest tests/test_no_backend_names.py -q`; strict docs build
  (`ACEJAX_DOCS_NOTEBOOKS=skip uv run --no-sync mkdocs build --strict`). Expected: PASS.
- [ ] **Step 4: Commit** `docs: cuequivariance product basis (aa_dag="cueq")`

### Task 7: Re-benchmark the Si linear ACE ladder with cueq; docs Performance page

**Files:**
- Create: `bench/accuracy_cost/envs/sulis-gpu-cueq.sbatch`, `bench/accuracy_cost/envs/sulis-gpu-cueq.json`
- Modify: `bench/accuracy_cost/timing.py` (an `acejax-ace` code variant with `aa_dag="cueq"`: one export per model,
  the gate cells reusing it), `bench/scaling/run_lammps.py` (env for the replay sidecar per bundle),
  `bench/si_qoi/plot_cost.py` (`FAM["ace_cueq"]`, `COST_NAME`), `bench/si_qoi/plot_cost_web.py` (`PRESETS`),
  `docs/dev/accuracy-cost-results.md` (reproduce block, method), `docs/user/benchmarks.md`, `docs/dev/benchmarks.md`
- Test: `tests/test_bench_accuracy_cost.py`, `tests/test_bench_user_page.py`

- [ ] **Step 1: Failing tests:** the timing code list includes the cueq variant only for single-species `ace` sizes;
  `plot_cost.FAM` has `ace_cueq` and `plot_cost_web.PRESETS["Same model, other code"]` contains it;
  `make_cost` maps a `{"code": "acejax-ace", "aa_dag": "cueq"}` row to `ACEcueq_<size>`.
- [ ] **Step 2:** implement the bench changes (port the spike's `export_one` pattern so the ladder exports each
  model once: o4d20 ~13 min); run the tests.
- [ ] **Step 3:** Sulis A100 job (`ssh -O check sulis` first): every `ace` size o3d8-o4d20, float64, 32,768 atoms,
  each gated (64-atom cells) before timing; rows to `results/sulis/sulis-gpu-cueq.jsonl`; budget ≈ 2 A100-h (the
  exports dominate); report the estimate first if more.
- [ ] **Step 4:** add `$R/sulis-gpu-cueq.jsonl` to the GPU `make_cost.py` call in the reproduce block; regenerate
  every figure from the reproduce block (`plot_cost.py` three outputs, `plot_cost_web.py`); `PRESETS["Headline"]`
  gets `ace_cueq` only if it is the fastest linear ACE code on the A100 at o4d20. Update `docs/user/benchmarks.md`
  (the line, a "How the cost was measured" bullet, its o4d20 row in "At the target") in ASD-STE style, and
  `docs/dev/benchmarks.md`.
- [ ] **Step 5:** `uv run pytest tests/test_bench_user_page.py tests/test_bench_accuracy_cost.py tests/test_no_backend_names.py -q`;
  `pre-commit run --all-files` (after `git add` of new files); strict docs build. Commit
  `feat(bench): Si linear ACE ladder with cueq on the A100; Performance page line`.

### Task 8: Phase 2 — TrACE (species-embedding) models

**Files:**
- Modify: `src/ace_jax/eval/cueq.py`, `src/ace_jax/eval/model.py` (`_readout_folded` cueq branch passes `node_z`)
- Test: `tests/test_cueq.py`

**Interfaces:**
- Consumes: Tasks 1-5. Port of the spike's `trace_cueq.py` (`trace_plan`, layout "ach").
- Produces: `cueq.trace_polynomial(model) -> (_Static(plans), gather)` where paths are the channel-free monomials,
  weights W[path, k, z] fold the coupling and per-centre readout; `product_energy(plan, A, node_z=...)` with u =
  U · d (atoms × channels), weights gathered per atom; `check_supported` accepts embedding models.

- [ ] **Step 1: Failing tests (GPU):** parity (site energies ≤ 1e-12, forces ≤ 1e-10 relative, float64) against the
  full model for a two-species and a five-species embedding model built with `build_embedding_model` from the
  repository's embedding fixtures (as the spike's `trace_*` models; small sizes); `check_supported` accepts them and
  still refuses categorical multi-species.
- [ ] **Step 2: Run** on lestrade — Expected: FAIL.
- [ ] **Step 3: Implement** from `trace_cueq.py`: A' gathered by `rnl_emb_nidx` / `rnl_emb_kidx` from the lean blocked
  A; W[path, k, z] = ctilde summed onto paths; the "ach" layout; remove Task 2's temporary TrACE refusal.
- [ ] **Step 4: Run** the cueq tests on lestrade and a lestrade LAMMPS gate on a SiGe and a Cantor embedding bundle
  (use cells large enough for the bench's ghost capacity at ace1's rcut — 512 / 1,372 atoms, as the spike did).
- [ ] **Step 5:** docs: drop "single species only" from the howto and SKILL/CLAUDE lines; commit
  `feat(eval): aa_dag="cueq" for species-embedding (TrACE) models`.
