# `aj fit` builds the basis; `fit.yaml`; no visible backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `aj fit --order 3 --max-degree 10 --train t.xyz --out fit/` builds the basis in memory and fits it; the run is fully described by (and reproducible from) one `fit.yaml`; nothing the user sees mentions the Julia backend.

**Architecture:** A `BasisSpec` (frozen dataclass of the basis definition) can stand in for the model path in `FitConfig.model`; `load_fit_data` reads the configs first, infers the species, builds the `Basis` via the single entry point `build_basis`, serialises it to a `BytesIO` and continues on the existing file-shaped path (approach A). The CLI adds basis flags to `aj fit` from a helper shared with `aj basis`, and a `--config fit.yaml` layer (YAML values → argparse defaults; explicit flags win). The package `ace_jax.construct` becomes `ace_jax.basis`, `Authoring` becomes `Basis`, and `ace-jax-coupling` becomes a core dependency.

**Tech Stack:** Python 3.11–3.13, argparse, PyYAML (already a core dep), numpy, JAX; uv; the compiled `ace-jax-coupling` from PR #20.

**Spec:** `docs/specs/2026-09-30-fit-builds-basis-design.md` (approved 2026-09-30).

**Branch:** `feat/fit-basis` in worktree `.worktrees/fit-basis`, stacked on `feat/trim-coupling` (PR #20, head `879fd7c`). The PR targets `feat/trim-coupling`.

## Global Constraints

- No backward-compatibility shims or aliases (ace-jax is unreleased, single user): renamed names are renamed everywhere except historical docs (`docs/plans/`, `docs/specs/` other than this spec, `docs/tier2-plan.md`).
- Renames: package `ace_jax.construct` → `ace_jax.basis`; class `Authoring` → `Basis`; model meta key `meta["authoring"]` → `meta["basis"]`; `tests/test_python_authoring.py` → `tests/test_basis_build.py`; `docs/python-authoring.md` → `docs/basis.md`; env var `ACEJAX_NO_JULIA` → `ACEJAX_COUPLING_CACHE_ONLY`; the `basis` extra is removed.
- `ace-jax-coupling==0.1.0` is a core dependency with marker `(sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')) or (sys_platform == 'darwin' and platform_machine == 'arm64')`; the uv path source stays.
- `fit.yaml`: top-level keys = `aj fit` argparse `dest`s; nested `basis:` keys = `aj basis` `dest`s (so `embedding`, not `basis_embedding`); `model:` xor `basis:`; reserved read-ignored key `provenance`. Precedence: explicit CLI > YAML > default. Unknown keys error with a `difflib` hint.
- `aj fit` basis flags: `--order --max-degree --elements --wL --rcut --rin --maxl --d-max --reduction --radial-mode --pair-mode --basis-embedding --no-gamma --no-coupling-cache --coupling-cache-dir`; `--seed` is shared with the fit.
- `--r0` defaults, when the basis is built, to the mean of `meta["basis"]["r0"]`; it stays required with `--model` (unless the model's meta has `basis.r0`).
- No user-facing string (argparse help, log lines, exceptions, README, SKILL.md, CLAUDE.md, `docs/basis.md`) contains "Julia"; `coupling/**` and `docs/coupling-etshim-spec.md` may.
- A definition-built fit writes a `model.npz` bit-identical to a fit of the same basis from a file.
- Spec amendment (Task 3 edits the spec): the "library unavailable" error is raised when the basis is built — after the configs are read (species come from them), before any fitting work — not before all data loading.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A basis-needing species appears only in the test or OOD file** — with inferred elements the basis must include it; with explicit `--elements` it must be an error, not a silent crash in evaluation. Tests in Task 4 (`test_inferred_elements_union_of_splits`, `test_explicit_elements_missing_test_species_errors`).
2. **A `fit.yaml` moved to another directory, or run from another cwd** — relative data paths must resolve against the YAML file, and the resolved `out/fit.yaml` must reproduce from anywhere. Tests in Task 6 (`test_relative_paths_resolve_against_yaml_dir`) and Task 7 (`test_resolved_yaml_reproduces_from_other_cwd`).
3. **A YAML typo** (`m_per_specie`, a basis key at top level, `basis:` and `model:` together) — must fail before any work with a helpful message, never be silently ignored. Tests in Task 6.
4. **Boolean/list flags from YAML** (`no_gamma: true`, `rungs: [map, laplace]`, `elements: [Si, Ge]`, `forces: false`) — must behave exactly like their CLI forms. Tests in Task 6 (`test_yaml_lists_and_bools_equal_cli_forms`).
5. **An old model file saved before the rename** (`meta["authoring"]`) — no shim by design; a user's own pre-rename `.npz` still loads for fitting (the key is only read for `wL` in `fit/radial_model.py` and `r0` defaulting). Test in Task 1 (`test_model_without_basis_meta_still_fits_radial`) pins that a missing `meta["basis"]` degrades to defaults rather than crashing.

---

### Task 1: Renames (`construct` → `basis`, `Authoring` → `Basis`, meta key, env var, files)

**Files:**
- Move: `src/ace_jax/construct/` → `src/ace_jax/basis/` (`git mv`)
- Move: `tests/test_python_authoring.py` → `tests/test_basis_build.py`; `docs/python-authoring.md` → `docs/basis.md`
- Modify: every importer (`git grep -l "ace_jax.construct\|from .construct\|from ..construct\|from ...construct\|construct\."`): `src/ace_jax/cli.py`, `src/ace_jax/fit/**`, `src/ace_jax/eval/**` (if any), `tests/*.py`, `bench/**/*.py`, `coupling/tools/gen_cases.py`
- Modify: `src/ace_jax/basis/model.py` (class + meta key), `src/ace_jax/basis/export.py` (docstrings, type name), `src/ace_jax/fit/radial_model.py:237` (meta key), `src/ace_jax/basis/coupling.py` (env var)
- Modify: tests using `ACEJAX_NO_JULIA`, `.github/workflows/coupling-wheels.yml` (paths `src/ace_jax/construct/**`, `tests/test_python_authoring.py`), `CLAUDE.md`, `README.md`, `skills/ace-jax/SKILL.md`, `docs/coupling-etshim-spec.md`, `docs/basis.md`
- Create: `tests/test_names.py`

**Interfaces:**
- Produces: `ace_jax.basis.{coupling, spec, model, export, prior, embedding, radial_init, radial_ace1}`; `ace_jax.basis.model.Basis` (the renamed NamedTuple, same fields); `meta["basis"]` (same content as the old `meta["authoring"]`); env var `ACEJAX_COUPLING_CACHE_ONLY`.

- [ ] **Step 1: Write the failing legacy-name test**

`tests/test_names.py`:
```python
"""No legacy names survive the construct -> basis rename (historical docs excepted)."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
HISTORICAL = ("docs/plans/", "docs/specs/", "docs/tier2-plan.md")
LEGACY = ["ace_jax.construct", "from .construct", "from ..construct", "from ...construct",
          "Authoring", '"authoring"', "ACEJAX_NO_JULIA", "python-authoring", "test_python_authoring",
          "[basis]", "--extra basis", "src/ace_jax/construct"]


def _grep(pattern):
    r = subprocess.run(["git", "grep", "-n", "-F", pattern, "--", ".", ":!uv.lock"],
                       cwd=ROOT, capture_output=True, text=True)
    return [l for l in r.stdout.splitlines()
            if not l.startswith(HISTORICAL) and not l.startswith("tests/test_names.py")]


def test_no_legacy_names():
    hits = {p: _grep(p) for p in LEGACY}
    assert not any(hits.values()), {p: h[:5] for p, h in hits.items() if h}
```
(`"[basis]"` / `"--extra basis"` go away in Task 3; this test goes green at the end of Task 3 — run it at the end of Task 1 expecting only those two patterns to remain.)

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_names.py -q -n 0`
Expected: FAIL listing hits for every pattern.

- [ ] **Step 3: Move and rewrite**

```bash
git mv src/ace_jax/construct src/ace_jax/basis
git mv tests/test_python_authoring.py tests/test_basis_build.py
git mv docs/python-authoring.md docs/basis.md
git ls-files 'src/**/*.py' 'tests/*.py' 'tests/**/*.py' 'bench/**/*.py' 'coupling/tools/*.py' | xargs perl -pi -e '
  s/\bace_jax\.construct\b/ace_jax.basis/g;
  s/from \.construct\b/from .basis/g; s/from \.\.construct\b/from ..basis/g; s/from \.\.\.construct\b/from ...basis/g;
  s/\bAuthoring\b/Basis/g;
  s/"authoring"/"basis"/g;
  s/ACEJAX_NO_JULIA/ACEJAX_COUPLING_CACHE_ONLY/g;
  s/\btest_python_authoring\b/test_basis_build/g'
git ls-files README.md CLAUDE.md skills docs/basis.md docs/coupling-etshim-spec.md coupling/python/README.md .github/workflows/coupling-wheels.yml | xargs perl -pi -e '
  s/\bace_jax\.construct\b/ace_jax.basis/g; s#src/ace_jax/construct#src/ace_jax/basis#g; s#construct/#basis/#g;
  s/\bAuthoring\b/Basis/g; s/ACEJAX_NO_JULIA/ACEJAX_COUPLING_CACHE_ONLY/g;
  s/python-authoring\.md/basis.md/g; s/test_python_authoring/test_basis_build/g'
```
Then review `git diff` by hand for over-matches (a docs `construct/` that referred to something else; `"authoring"` in prose). In `src/ace_jax/basis/coupling.py` rewrite the env-var error to backend-neutral text:
```python
    if os.environ.get("ACEJAX_COUPLING_CACHE_ONLY"):
        raise RuntimeError("ACEJAX_COUPLING_CACHE_ONLY is set but the coupling cache missed: "
                           "this basis shape has not been built before")
```
Update `.test_durations` keys: `perl -pi -e 's/tests\/test_python_authoring\.py/tests\/test_basis_build.py/g' .test_durations`.

- [ ] **Step 4: Pin the pre-rename-file behaviour (Review Focus 5)**

Append to `tests/test_basis_build.py`:
```python
def test_model_without_basis_meta_still_fits_radial():
    """A model file without meta['basis'] (e.g. a Julia export, or a pre-rename
    save) takes the radial-model defaults instead of failing."""
    from ace_jax.fit import radial_model as rm
    import inspect
    src = inspect.getsource(rm)
    assert 'meta.get("basis", {})' in src
```
(`fit/radial_model.py:237` already uses `.get(..., {})`; the rename keeps that form.)

- [ ] **Step 5: Run everything**

Run: `uv run pytest -q` → PASS (the suite count matches the base branch). Run: `uv run pytest tests/test_names.py -q -n 0` → FAIL only for `"[basis]"` and `"--extra basis"`. Run: `uv run ruff check` → clean.

- [ ] **Step 6: Commit**

```bash
git add -A src tests bench coupling/tools README.md CLAUDE.md skills docs .github .test_durations
git commit -m "refactor!: ace_jax.construct -> ace_jax.basis, Authoring -> Basis, meta['basis'], ACEJAX_COUPLING_CACHE_ONLY"
```

---

### Task 2: `BasisSpec` + `build_basis` (one entry point) and `aj basis` on it

**Files:**
- Modify: `src/ace_jax/basis/model.py` (add `BasisSpec`, `build_basis`, `basis_r0`)
- Modify: `src/ace_jax/cli.py` (`cmd_basis` calls `build_basis`; `add_basis_args` helper)
- Test: `tests/test_basis_spec.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class BasisSpec:
      order: int; max_degree: int
      elements: tuple | None = None          # Z ints or symbols; None = infer from data
      wL: float = 1.5; rcut: float | None = None; rin: float = 0.0
      maxl: int | None = None; d_max: int | None = None; reduction: str = "pca"
      radial_mode: str = "glorot_normal"; pair_mode: str = "onehot"
      embedding: str | None = None           # JSON table path | "identity" | None (categorical)
      no_gamma: bool = False
      no_coupling_cache: bool = False; coupling_cache_dir: str | None = None
      FIELDS: ClassVar[tuple]                # the dataclass field names, = aj basis dests
  def build_basis(spec: BasisSpec, *, seed: int = 0) -> Basis        # spec.elements must be set
  def basis_r0(meta: dict) -> float | None                            # mean of meta["basis"]["r0"]
  ```
  `add_basis_args(p, *, fit: bool)` in `cli.py`: adds the basis flags; with `fit=True` the embedding flag is `--basis-embedding` (dest `basis_embedding`), `--seed` is not added, and `--order`/`--max-degree` are not `required`.

- [ ] **Step 1: Failing tests**

`tests/test_basis_spec.py`:
```python
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_basis_build import _primed_cache


def test_build_basis_equals_build_model(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec, build_basis, build_model
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    a = build_basis(BasisSpec(order=3, max_degree=10, elements=(14,), coupling_cache_dir=cache), seed=0)
    b = build_model([14], 3, 10, coupling_cache_dir=cache, seed=0)
    assert np.array_equal(np.asarray(a.model.A2B), np.asarray(b.model.A2B))
    assert np.array_equal(np.asarray(a.model.rnl_Wnlq), np.asarray(b.model.rnl_Wnlq))
    assert a.meta == b.meta


def test_build_basis_requires_elements():
    from ace_jax.basis.model import BasisSpec, build_basis
    with pytest.raises(ValueError, match="elements"):
        build_basis(BasisSpec(order=2, max_degree=4))


def test_basis_r0_is_mean_of_meta_r0():
    from ace_jax.basis.model import basis_r0
    assert basis_r0({"basis": {"r0": [[2.0, 3.0], [3.0, 4.0]]}}) == 3.0
    assert basis_r0({"basis": {"r0": 2.5}}) == 2.5
    assert basis_r0({}) is None


def test_symbols_and_numbers_are_equivalent(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec, build_basis
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    a = build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",), coupling_cache_dir=cache))
    assert a.meta["elements"] == [14]
```
Run: `uv run pytest tests/test_basis_spec.py -q -n 0` → FAIL (`ImportError: cannot import name 'BasisSpec'`).

- [ ] **Step 2: Implement in `src/ace_jax/basis/model.py`**

Add after the `Basis` class:
```python
@dataclasses.dataclass(frozen=True)
class BasisSpec:
    """A basis definition: the `aj basis` flags (and the `basis:` block of a
    fit.yaml).  `elements=None` means "the species of the training data"."""
    order: int
    max_degree: int
    elements: tuple | None = None
    wL: float = 1.5
    rcut: float | None = None
    rin: float = 0.0
    maxl: int | None = None
    d_max: int | None = None
    reduction: str = "pca"
    radial_mode: str = "glorot_normal"
    pair_mode: str = "onehot"
    embedding: str | None = None
    no_gamma: bool = False
    no_coupling_cache: bool = False
    coupling_cache_dir: str | None = None

    FIELDS: typing.ClassVar[tuple] = ()


BasisSpec.FIELDS = tuple(f.name for f in dataclasses.fields(BasisSpec))


def _zs(elements):
    from ase.data import atomic_numbers
    return [int(e) if isinstance(e, (int, np.integer)) or str(e).strip().isdigit()
            else atomic_numbers[str(e).strip()] for e in elements]


def build_basis(spec, *, seed=0):
    """The one basis builder (`aj basis`, `aj fit`, Python): a `Basis` from a
    `BasisSpec`.  Categorical (`build_model`) unless `spec.embedding` is set
    (`build_embedding_model`)."""
    if not spec.elements:
        raise ValueError("build_basis: spec.elements is not set (aj fit infers it from the data)")
    els = _zs(spec.elements)
    common = dict(with_gamma=not spec.no_gamma, coupling_cache=not spec.no_coupling_cache,
                  coupling_cache_dir=spec.coupling_cache_dir)
    if spec.embedding:
        return build_embedding_model(els, spec.order, spec.max_degree, embedding=spec.embedding,
                                     d_max=spec.d_max, wL=spec.wL, maxl=spec.maxl, rcut=spec.rcut,
                                     reduction=spec.reduction, **common)
    return build_model(els, spec.order, spec.max_degree, wL=spec.wL,
                       rcut=5.5 if spec.rcut is None else spec.rcut, rin=spec.rin,
                       radial_mode=spec.radial_mode, pair_mode=spec.pair_mode, seed=seed, **common)


def basis_r0(meta):
    """Mean radial length scale the basis was built with (per-pair table or a
    scalar): the default GP hyperprior centre for a fit of this basis."""
    r0 = meta.get("basis", {}).get("r0")
    return None if r0 is None else float(np.mean(np.asarray(r0, float)))
```
(add `import dataclasses`, `import typing` at the top if absent.)

- [ ] **Step 3: Rewire `aj basis`**

In `cli.py` replace the body of `cmd_basis` down to `out = pathlib.Path(a.out)` with:
```python
    from .basis.model import build_basis
    auth = build_basis(_basis_spec(a, embedding=a.embedding), seed=a.seed)
```
and add:
```python
def add_basis_args(p, *, fit):
    """The basis-definition flags, shared by `aj basis` and `aj fit`.  On `aj fit`
    the embedding flag is --basis-embedding (--embedding is the GP species
    table there) and --seed is the fit's own."""
    p.add_argument("--elements", default=None, help="comma-separated Z numbers or symbols"
                   + (" (default: the species in the data)" if fit else ""))
    p.add_argument("--order", type=int, required=not fit, help="correlation order")
    p.add_argument("--max-degree", type=int, required=not fit, help="TotalDegree level bound")
    p.add_argument("--wL", type=float, default=1.5)
    p.add_argument("--rcut", type=float, default=None,
                   help="cutoff (default 5.5; with an embedding, 2.5 x mean bond length)")
    p.add_argument("--rin", type=float, default=0.0)
    p.add_argument("--radial-mode", default="glorot_normal")
    p.add_argument("--pair-mode", default="onehot")
    p.add_argument("--no-gamma", action="store_true", help="skip the smoothness prior")
    p.add_argument("--no-coupling-cache", action="store_true",
                   help="always compute the coupling instead of using the per-shape cache")
    p.add_argument("--coupling-cache-dir", default=None,
                   help="coupling cache directory (default: $ACEJAX_COUPLING_CACHE or ~/.cache/ace-jax/coupling)")
    p.add_argument("--basis-embedding" if fit else "--embedding", default=None,
                   help="frozen element embedding of the basis: a JSON table {Z, emb} or 'identity'")
    p.add_argument("--d-max", type=int, default=None, help="cap on per-order channel widths (default lossless)")
    p.add_argument("--maxl", type=int, default=None)
    p.add_argument("--reduction", choices=["pca", "truncate"], default="pca")
    if not fit:
        p.add_argument("--seed", type=int, default=0)


def _basis_spec(a, *, embedding):
    from .basis.model import BasisSpec
    kw = {f: getattr(a, f) for f in BasisSpec.FIELDS if f not in ("embedding", "elements")}
    els = None if a.elements is None else tuple(
        e.strip() for e in (a.elements if isinstance(a.elements, (list, tuple)) else a.elements.split(",")))
    return BasisSpec(elements=els, embedding=embedding, **kw)
```
Replace the `aj basis` argument block in `_parser` (from `con.add_argument("--elements"...` to `con.add_argument("--reduction"...`) with `add_basis_args(con, fit=False)`, keeping `con.add_argument("--out", required=True)`.

- [ ] **Step 4: Tests**

Run: `uv run pytest tests/test_basis_spec.py tests/test_cli_basis_embedding.py tests/test_basis_build.py -q -n 0` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/basis/model.py src/ace_jax/cli.py tests/test_basis_spec.py
git commit -m "feat(basis): BasisSpec + build_basis, the one basis entry point (aj basis uses it)"
```

---

### Task 3: Core dependency; backend-neutral "unavailable" message; spec amendment

**Files:**
- Modify: `pyproject.toml` (dependency, remove `basis` extra), `uv.lock`
- Modify: `src/ace_jax/basis/coupling.py` (`_lib()` message)
- Modify: `tests/test_coupling_cache.py` (`test_backend_id_matches_extra_pin` → reads `dependencies`)
- Modify: `docs/specs/2026-09-30-fit-builds-basis-design.md` (§4 wording)
- Test: `tests/test_basis_unavailable.py`

**Interfaces:**
- Produces: `ace_jax.basis.coupling.BasisUnavailable(RuntimeError)`, raised by `_lib()` when the coupling library cannot be used.

- [ ] **Step 1: Failing tests**

`tests/test_basis_unavailable.py`:
```python
import builtins
import sys

import pytest


def test_missing_library_message_is_backend_neutral(monkeypatch):
    from ace_jax.basis import coupling as C
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "ace_jax_coupling":
            raise ModuleNotFoundError("No module named 'ace_jax_coupling'")
        return real_import(name, *a, **k)
    monkeypatch.delenv("ACEJAX_COUPLING_CACHE_ONLY", raising=False)
    monkeypatch.setitem(sys.modules, "ace_jax_coupling", None)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(C.BasisUnavailable) as e:
        C._lib()
    msg = str(e.value)
    assert "not available on this platform" in msg and "--model" in msg
    assert "julia" not in msg.lower()
```
In `tests/test_coupling_cache.py::test_backend_id_matches_extra_pin`, replace the extra lookup with
```python
    deps = " ".join(pp["project"]["dependencies"])
    m = re.search(r"ace-jax-coupling==([0-9][^;\s]*)", deps)
```
and rename it `test_backend_id_matches_dependency_pin`.
Run: `uv run pytest tests/test_basis_unavailable.py tests/test_coupling_cache.py -q -n 0` → FAIL (`AttributeError: BasisUnavailable`; dependency not found).

- [ ] **Step 2: Implement**

`pyproject.toml`: add to `dependencies`
```toml
  # coupling tables for new basis shapes (compiled; no runtime toolchain). Platform wheels only.
  "ace-jax-coupling==0.1.0; (sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')) or (sys_platform == 'darwin' and platform_machine == 'arm64')",
```
and delete the `basis = [...]` extra and its comment. Update the `[tool.uv.sources]` comment's command to `uv sync --reinstall-package ace-jax-coupling` (no `--extra`). Run `uv lock`.

`coupling.py` `_lib()`:
```python
class BasisUnavailable(RuntimeError):
    """Building a new basis needs the compiled coupling library, which is not
    installed (unsupported platform) or has no compiled payload (dev build)."""


def _lib():
    if os.environ.get("ACEJAX_COUPLING_CACHE_ONLY"):
        raise RuntimeError("ACEJAX_COUPLING_CACHE_ONLY is set but the coupling cache missed: "
                           "this basis shape has not been built before")
    import platform
    try:
        import ace_jax_coupling
    except ModuleNotFoundError as e:
        raise BasisUnavailable(
            f"building a new basis is not available on this platform ({sys.platform} {platform.machine()}); "
            "fit from an existing basis with --model <file.npz> built elsewhere") from e
    try:
        ace_jax_coupling.build_info()
    except ace_jax_coupling.CouplingLibError as e:
        raise BasisUnavailable(f"the ace-jax-coupling install has no compiled library: {e}") from e
    if ace_jax_coupling.__version__ != COUPLING_LIB_VERSION:
        raise BasisUnavailable(f"ace-jax-coupling {ace_jax_coupling.__version__} is installed; this ace-jax "
                               f"needs =={COUPLING_LIB_VERSION}")
    return ace_jax_coupling
```
(`import sys` at the top of `coupling.py`.) Spec §4: replace "It is raised **before** any data is loaded." with "It is raised when the basis is built — after the configs are read (the species come from them), before any fitting work."

- [ ] **Step 3: Tests**

Run: `uv sync && uv run pytest tests/test_basis_unavailable.py tests/test_coupling_cache.py tests/test_names.py -q -n 0` → PASS (test_names now fully green). Run `uv run pytest -q` → PASS.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/ace_jax/basis/coupling.py tests/test_basis_unavailable.py tests/test_coupling_cache.py docs/specs/2026-09-30-fit-builds-basis-design.md
git commit -m "feat!: ace-jax-coupling is a core dependency; backend-neutral BasisUnavailable"
```

---

### Task 4: The pipeline accepts a basis (`FitConfig.model: path | Basis | BasisSpec`)

**Files:**
- Modify: `src/ace_jax/fit/pipeline/config.py` (`model` type, `r0: float | None`), `data.py` (reorder; species; build; buffer; `FitData.r0`, `FitData.basis`), `problem.py` (resolved r0; prior label), `export.py` (`_ace_arrays` from `res.data.z`; `.yace` check), `src/ace_jax/fit/pipeline/__init__.py` (nothing new exported unless needed)
- Test: `tests/test_fit_basis_pipeline.py`

**Interfaces:**
- Consumes: `BasisSpec`, `build_basis`, `basis_r0` (Task 2); `save_npz(file_or_path, basis)` (existing; `np.savez` accepts a file object).
- Produces: `FitData` gains fields `r0: float | None` (the basis's mean r0 when known) and `source: str` (a log label: the path, or `"basis Si order 3 max-degree 10"`); `load_fit_data` builds the basis when `cfg.model` is a `BasisSpec` (elements inferred when `None`) or serialises a `Basis`; `problem.build_problem` uses `r0 = cfg.r0 if cfg.r0 is not None else d.r0` and raises `ValueError("r0 is not set and the model has no basis r0: pass r0")` when both are `None`.

- [ ] **Step 1: Failing tests**

`tests/test_fit_basis_pipeline.py`:
```python
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR
from test_basis_build import _primed_cache

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"


def _cfg(model, **kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(arm="linear", m_per_species=0, rungs=("map",), map_steps=20, opt="adam",
                batch=4, r0=2.35, e0="lsq", predict_train=False)
    base.update(kw)
    return FitConfig(model=model, **base)


def _fit(cfg, tmp_path, name):
    from ace_jax.fit.pipeline import fit, load_fit_data
    from ace_jax.fit.pipeline.export import save_model
    d = load_fit_data(cfg, train=str(XYZ))
    res = fit(cfg, d, log=lambda *a: None)
    return res, np.load(save_model(res, tmp_path / name, log=lambda *a: None))


def test_basis_spec_fit_equals_file_fit(tmp_path, monkeypatch):
    """A basis built inside the pipeline fits and exports bit-identically to the
    same basis saved to a file first (the in-memory hand-off changes nothing)."""
    from ace_jax.basis.export import save_npz
    from ace_jax.basis.model import BasisSpec, build_basis
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    spec = BasisSpec(order=3, max_degree=10, coupling_cache_dir=cache)
    f = tmp_path / "si.npz"
    save_npz(f, build_basis(BasisSpec(order=3, max_degree=10, elements=(14,), coupling_cache_dir=cache)))
    _, za = _fit(_cfg(spec), tmp_path, "a")
    _, zb = _fit(_cfg(str(f)), tmp_path, "b")
    assert sorted(za.files) == sorted(zb.files)
    for k in za.files:
        assert np.array_equal(za[k], zb[k]), k


def test_basis_object_is_accepted(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec, build_basis
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    b = build_basis(BasisSpec(order=3, max_degree=10, elements=(14,), coupling_cache_dir=cache))
    res, z = _fit(_cfg(b), tmp_path, "c")
    assert "WB" in z.files and res.data.source.startswith("basis")


def test_inferred_elements_union_of_splits(tmp_path, monkeypatch):
    from ace_jax.fit.pipeline.data import infer_elements
    from ace_jax.fit.data import load_configs
    cs = load_configs(str(XYZ))
    assert infer_elements(cs[:2], cs[2:4], []) == [14]


def test_explicit_elements_missing_test_species_errors(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec
    from ace_jax.fit.pipeline import load_fit_data
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    spec = BasisSpec(order=3, max_degree=10, elements=("Ge",), coupling_cache_dir=_primed_cache(tmp_path))
    with pytest.raises(ValueError, match="not in the basis elements.*Si"):
        load_fit_data(_cfg(spec), train=str(XYZ))


def test_r0_defaults_to_basis_mean(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec, basis_r0
    from ace_jax.fit.pipeline import fit, load_fit_data
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cfg = _cfg(BasisSpec(order=3, max_degree=10, coupling_cache_dir=_primed_cache(tmp_path)), r0=None)
    d = load_fit_data(cfg, train=str(XYZ))
    assert d.r0 == pytest.approx(basis_r0(d.meta))
    res = fit(cfg, d, log=lambda *a: None)
    assert res.built.gpcfg.r0 == pytest.approx(d.r0)
```
Run: `uv run pytest tests/test_fit_basis_pipeline.py -q -n 0` → FAIL (`load` of a `BasisSpec`; `infer_elements` missing).

- [ ] **Step 2: Implement**

`config.py`: `model: object` (docstring: `str | os.PathLike | Basis | BasisSpec`), `r0: float | None = 2.5` (unchanged default; `None` = the basis's mean).

`data.py`: add
```python
import io


def infer_elements(*splits):
    """Sorted atomic numbers present in any of the config lists."""
    return sorted({int(z) for s in splits for c in s for z in np.unique(c.numbers)})


def _model_source(cfg, train_o, test_o, ood_o, log):
    """(load target, label, basis or None) for cfg.model: a path stays a path; a
    Basis or BasisSpec becomes an in-memory npz (approach A of the spec)."""
    from ...basis.export import save_npz
    from ...basis.model import Basis, BasisSpec, build_basis
    m = cfg.model
    if isinstance(m, BasisSpec):
        seen = infer_elements(train_o, test_o, ood_o)
        if m.elements is None:
            m = dataclasses.replace(m, elements=tuple(seen))
            inferred = True
        else:
            inferred = False
        from ...basis.model import _zs
        missing = sorted(set(seen) - set(_zs(m.elements)))
        if missing:
            from ase.data import chemical_symbols
            raise ValueError(f"species {', '.join(chemical_symbols[z] for z in missing)} in the data are "
                             f"not in the basis elements {list(m.elements)}")
        b = build_basis(m, seed=cfg.seed)
        label = (f"basis {','.join(_symbols(m.elements))} order {m.order} max-degree {m.max_degree}")
        log(f"{label} -> {b.meta['n_B']} B functions" + (" (elements from data)" if inferred else ""))
        m = b
    if isinstance(m, Basis):
        buf = io.BytesIO()
        save_npz(buf, m)
        buf.seek(0)
        label = locals().get("label") or f"basis {','.join(map(str, m.meta['elements']))}"
        return buf, label, m
    return m, str(m), None
```
with a small `_symbols(elements)` helper (Z → symbol via `ase.data.chemical_symbols`, symbols unchanged). Change `load_fit_data(cfg, *, data=None, train=None, test=None, ood=None, log=print)`: move `model, meta, z = load(cfg.model)` and the PACE check **after** `ood_o = ...`, replaced by
```python
    src, label, basis = _model_source(cfg, train_o, test_o, ood_o, log)
    model, meta, z = load(src)
    if isinstance(model, PACEModel):
        raise ValueError(f"{label}: a PACE .yace model is evaluate-only and cannot be fitted; "
                         "fit a linear ACE model (.npz, e.g. from `aj basis`) instead")
```
(`_config_type_weights(data or train)` and `split_configs` do not depend on the model, so the reorder is safe.) Add to `FitData` (at the end, with defaults): `r0: float | None = None; source: str = ""; basis: object = None`, and fill them in the return: `r0=basis_r0(meta), source=label, basis=basis`. (`import dataclasses` and `from ...basis.model import basis_r0`.)

`problem.py`: at the top of `build_problem`:
```python
    r0 = cfg.r0 if cfg.r0 is not None else d.r0
    if r0 is None:
        raise ValueError("r0 is not set and the model carries no basis r0: pass r0 (--r0)")
```
and use `r0` for `GPConfig(r0=...)` and `default_prior(r0)`; pass `d.source` instead of `cfg.model` to `prior_diagonal`.

`export.py`: `_ace_arrays(res)` → `z = res.data.z` (the arrays `load` read; a `NpzFile` over the path or the buffer); `save_model`: `if isinstance(cfg.model, (str, os.PathLike)) and str(cfg.model).endswith(".yace"):`.

- [ ] **Step 3: Tests**

Run: `uv run pytest tests/test_fit_basis_pipeline.py -q -n 0` → PASS. Run: `uv run pytest -q` → PASS (existing pipeline tests unchanged: `FitData` fields added at the end with defaults).

- [ ] **Step 4: Commit**

```bash
git add src/ace_jax/fit/pipeline tests/test_fit_basis_pipeline.py
git commit -m "feat(fit): FitConfig.model may be a Basis or BasisSpec (in-memory basis, species from data, r0 default)"
```

---

### Task 5: `aj fit` basis flags

**Files:**
- Modify: `src/ace_jax/cli.py` (`_add_fit_args`, `_fit_config`, `run`)
- Test: `tests/test_cli_fit_basis.py`

**Interfaces:**
- Consumes: `add_basis_args(p, fit=True)`, `_basis_spec(a, embedding=a.basis_embedding)` (Task 2); `FitConfig.model` accepting `BasisSpec`; `FitData.r0` (Task 4).
- Produces: `aj fit` accepts `--model` xor basis flags (`--order`/`--max-degree` together); `--r0` optional when a basis is built; `_fit_config(a)` returns a `FitConfig` whose `model` is a path or a `BasisSpec`.

- [ ] **Step 1: Failing tests**

`tests/test_cli_fit_basis.py`:
```python
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR
from test_basis_build import _primed_cache

from ace_jax.cli import main

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
FAST = ["--m-per-species", "0", "--rungs", "map", "--map-steps", "20", "--opt", "adam",
        "--configs-per-batch", "4"]


def test_cli_basis_flags_fit_equals_model_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    main(["basis", "--elements", "Si", "--order", "3", "--max-degree", "10",
          "--coupling-cache-dir", cache, "--out", str(tmp_path / "si.npz")])
    main(["fit", "--model", str(tmp_path / "si.npz"), "--train", str(XYZ), "--r0", "2.35",
          "--out", str(tmp_path / "a"), *FAST])
    main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
          "--train", str(XYZ), "--r0", "2.35", "--out", str(tmp_path / "b"), *FAST])
    za, zb = np.load(tmp_path / "a" / "model.npz"), np.load(tmp_path / "b" / "model.npz")
    assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)


def test_cli_r0_defaults_when_building(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
          "--train", str(XYZ), "--out", str(tmp_path / "c"), *FAST])
    assert "r0" in capsys.readouterr().out


@pytest.mark.parametrize("argv,msg", [
    (["fit", "--train", "x.xyz", "--out", "o", "--r0", "2"], "one of --model"),
    (["fit", "--model", "m.npz", "--order", "3", "--max-degree", "5", "--train", "x.xyz", "--out", "o"],
     "--model and"),
    (["fit", "--order", "3", "--train", "x.xyz", "--out", "o"], "--max-degree"),
    (["fit", "--model", "m.npz", "--train", "x.xyz", "--out", "o"], "--r0"),
])
def test_cli_basis_source_errors(argv, msg, capsys):
    with pytest.raises(SystemExit):
        main(argv)
    assert msg in capsys.readouterr().err
```
Run: `uv run pytest tests/test_cli_fit_basis.py -q -n 0` → FAIL (unrecognized `--order`).

- [ ] **Step 2: Implement**

In `_add_fit_args(p)`: `p.add_argument("--model", default=None, help="an ACE basis/model .npz; or give the basis flags below")`; change `--r0` to `default=None` with help `"typical nearest-neighbour distance (A); centres the GP hyperprior (default when building a basis: its mean bond length)"`; make the `--train/--data` group `required=False` (checked in `_check_fit_args`, Task 6 moves it after the YAML merge); at the end call `add_basis_args(p.add_argument_group("basis (instead of --model)"), fit=True)`.

Add:
```python
def _check_fit_args(p, a):
    """Cross-flag rules argparse cannot express (run after any config merge)."""
    building = a.order is not None or a.max_degree is not None
    if a.model is None and not building:
        p.error("give one of --model, --order/--max-degree, or --config")
    if a.model is not None and building:
        p.error("--model and --order/--max-degree are alternatives: give one")
    if building and (a.order is None or a.max_degree is None):
        p.error("--order and --max-degree go together")
    if a.train is None and a.data is None:
        p.error("one of the arguments --train --data is required")
    if a.out is None:
        p.error("the following arguments are required: --out")
    if a.model is not None and a.r0 is None:
        p.error("--r0 is required with --model")
```
In `_fit_config(a)`: `model = a.model if a.model is not None else _basis_spec(a, embedding=a.basis_embedding)`, pass `model=model` and `r0=a.r0`. In `main`, after `parse_args`, for `a.cmd == "fit"` call `_check_fit_args(fit_parser, a)` (keep a reference to the `fit` subparser in `_parser`, e.g. `top._fit = sub_fit`). In `run(a)`: after `load_fit_data`, log `r0` when defaulted: `if a.r0 is None: print(f"r0 {data.r0:.3f} A (mean bond length of the basis; pass --r0 to override)")`.

- [ ] **Step 3: Tests**

Run: `uv run pytest tests/test_cli_fit_basis.py tests/test_gp_cli.py tests/test_cli_alias.py -q -n 0` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ace_jax/cli.py tests/test_cli_fit_basis.py
git commit -m "feat(cli): aj fit builds the basis from --order/--max-degree (no aj basis step)"
```

---

### Task 6: `--config fit.yaml` (whole run; precedence; validation; paths)

**Files:**
- Create: `src/ace_jax/runfile.py` (load, validate, merge; no argparse knowledge beyond a parser passed in)
- Modify: `src/ace_jax/cli.py` (`--config` on `fit` and `basis`; two-phase parse in `main`)
- Test: `tests/test_runfile.py`

**Interfaces:**
- Produces (`ace_jax.runfile`):
  - `PATH_KEYS = ("train", "test", "ood", "data", "model", "baseline", "init", "embedding")`; `BASIS_PATH_KEYS = ("embedding", "coupling_cache_dir")`
  - `read(path) -> dict` — YAML mapping, relative path values resolved against `path`'s directory (`"identity"` embedding left alone); raises `ValueError` for a non-mapping.
  - `defaults_for(parser, cfg: dict, *, basis_dest_map: dict) -> dict` — validates keys (top-level against the parser's dests; `basis:` against `BasisSpec.FIELDS`; `model`+`basis` conflict; `provenance` ignored), returns a flat `dest -> value` dict for `parser.set_defaults`. `basis_dest_map` maps `BasisSpec` fields to fit dests (`{"embedding": "basis_embedding"}` for `aj fit`, `{}` for `aj basis`). Lists for `rungs`/`elements` are joined with commas where the CLI expects a comma string.
  - `explicit_dests(parser, argv) -> set` — dests given on the command line (a SUPPRESS-default re-parse).
- `main(argv)`: for `fit`/`basis`, if `--config` is present: `set_defaults(**defaults_for(...))` on that subparser, re-parse, log `override: <key> <yaml> -> <cli> (command line)` for keys in both, then `_check_fit_args`.

- [ ] **Step 1: Failing tests**

`tests/test_runfile.py`:
```python
import pathlib

import pytest
import yaml

from ace_jax.cli import _parser, main


def _write(p, d):
    p.write_text(yaml.safe_dump(d)); return p


def _fit_ns(argv):
    from ace_jax.cli import _parse
    return _parse(["fit", *argv])


def test_yaml_supplies_required_and_cli_overrides(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "m_per_species": 0,
                                       "basis": {"order": 3, "max_degree": 10}})
    a = _fit_ns(["--config", str(f), "--m-per-species", "5"])
    assert a.m_per_species == 5 and a.order == 3 and a.max_degree == 10
    assert pathlib.Path(a.train) == tmp_path / "t.xyz"
    assert "override: m_per_species 0 -> 5 (command line)" in capsys.readouterr().out


def test_unknown_key_did_you_mean(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "m_per_specie": 0,
                                       "basis": {"order": 3, "max_degree": 10}})
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])
    err = capsys.readouterr().err
    assert "unknown key 'm_per_specie'" in err and "did you mean 'm_per_species'" in err


def test_basis_key_at_top_level_is_unknown(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "max_degree": 10})
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])


def test_model_and_basis_conflict(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "model": "m.npz",
                                       "basis": {"order": 3, "max_degree": 10}})
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])
    assert "model" in capsys.readouterr().err


def test_relative_paths_resolve_against_yaml_dir(tmp_path, monkeypatch):
    sub = tmp_path / "runs"; sub.mkdir()
    f = _write(sub / "fit.yaml", {"train": "../data/t.xyz", "test": "t2.xyz", "out": "o", "r0": 2.3,
                                  "model": "m.npz"})
    monkeypatch.chdir(tmp_path)
    a = _fit_ns(["--config", str(f)])
    assert pathlib.Path(a.train).resolve() == (tmp_path / "data" / "t.xyz").resolve()
    assert pathlib.Path(a.model) == sub / "m.npz"


def test_yaml_lists_and_bools_equal_cli_forms(tmp_path):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "rungs": ["map", "laplace"],
                                       "no_bump": True,
                                       "basis": {"order": 3, "max_degree": 10, "elements": ["Si", "Ge"],
                                                 "no_gamma": True}})
    a = _fit_ns(["--config", str(f)])
    b = _fit_ns(["--train", str(tmp_path / "t.xyz"), "--out", "o", "--rungs", "map,laplace", "--no-bump",
                 "--order", "3", "--max-degree", "10", "--elements", "Si,Ge", "--no-gamma"])
    for k in ("rungs", "no_bump", "elements", "no_gamma", "order", "max_degree"):
        assert getattr(a, k) == getattr(b, k), k


def test_aj_basis_reads_the_basis_block(tmp_path, monkeypatch):
    from test_basis_build import _primed_cache
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "seed": 0,
                                       "basis": {"elements": ["Si"], "order": 3, "max_degree": 10,
                                                 "coupling_cache_dir": cache}})
    main(["basis", "--config", str(f), "--out", str(tmp_path / "si.npz")])
    assert (tmp_path / "si.npz").exists()
```
Run: `uv run pytest tests/test_runfile.py -q -n 0` → FAIL (`_parse` missing / `--config` unrecognized).

- [ ] **Step 2: Implement `src/ace_jax/runfile.py`**

```python
"""fit.yaml: a whole `aj fit` run in one file.  Top-level keys are `aj fit`
flag dests; a nested `basis:` block holds `aj basis` dests (or `model:` a path).
YAML values become argparse defaults, so explicit command-line flags win."""
import argparse
import difflib
import pathlib

import yaml

PATH_KEYS = ("train", "test", "ood", "data", "model", "baseline", "init", "embedding")
BASIS_PATH_KEYS = ("embedding", "coupling_cache_dir")
IGNORED = ("provenance",)
COMMA_KEYS = ("rungs", "elements")


def _resolve(v, base):
    if v is None or not isinstance(v, str) or v == "identity":
        return v
    p = pathlib.Path(v).expanduser()
    return str(p if p.is_absolute() else base / p)


def read(path):
    path = pathlib.Path(path)
    d = yaml.safe_load(path.read_text()) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a mapping of flag names to values")
    base = path.resolve().parent
    for k in PATH_KEYS:
        if k in d:
            d[k] = _resolve(d[k], base)
    if isinstance(d.get("basis"), dict):
        for k in BASIS_PATH_KEYS:
            if k in d["basis"]:
                d["basis"][k] = _resolve(d["basis"][k], base)
    return d


def _unknown(where, key, known):
    hint = difflib.get_close_matches(key, known, n=1)
    return f"{where}: unknown key '{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")


def _dests(parser):
    return {a.dest for a in parser._actions if a.dest not in ("help", argparse.SUPPRESS)}


def defaults_for(parser, cfg, *, basis_fields, basis_dest_map):
    """Flat dest -> value defaults for `parser`, or ValueError naming the bad key."""
    dests = _dests(parser)
    top_known = sorted(dests - set(basis_fields) - {"config"}) + ["basis", "model"]
    if "model" in cfg and "basis" in cfg:
        raise ValueError("fit.yaml: 'model' and 'basis' are alternatives: give one")
    out = {}
    for k, v in cfg.items():
        if k in IGNORED or k == "basis":
            continue
        if k not in top_known:
            raise ValueError(_unknown("fit.yaml", k, top_known))
        out[k] = ",".join(map(str, v)) if k in COMMA_KEYS and isinstance(v, list) else v
    for k, v in (cfg.get("basis") or {}).items():
        if k not in basis_fields:
            raise ValueError(_unknown("fit.yaml basis", k, list(basis_fields)))
        dest = basis_dest_map.get(k, k)
        if dest not in dests:
            continue                        # e.g. seed lives at top level
        out[dest] = ",".join(map(str, v)) if k in COMMA_KEYS and isinstance(v, list) else v
    return out


def explicit_dests(parser, argv):
    """Dests given explicitly on the command line (a SUPPRESS-default re-parse)."""
    shadow = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    for a in parser._actions:
        if a.option_strings and a.dest != "help":
            kw = {"dest": a.dest, "default": argparse.SUPPRESS}
            if a.nargs == 0:
                kw["action"] = "store_const"; kw["const"] = a.const
            else:
                kw["nargs"] = a.nargs
            shadow.add_argument(*a.option_strings, **kw)
    ns, _ = shadow.parse_known_args(argv)
    return set(vars(ns))
```
(`store_true` actions have `nargs=0` and `const=True`; the shadow parser records presence only.)

`cli.py`: add `--config` to both subparsers (`help="a fit.yaml run file (command-line flags override it)"`; on `basis` it reads only the `basis:` block plus top-level `seed`). Split `main` into `_parse(argv)` (returns the namespace, performing the two-phase parse and `_check_fit_args`) and dispatch. In `_parse`: parse once; if `a.config`: `cfg = runfile.read(a.config)`; for `basis`, flatten to `flat = {**cfg.get("basis", {}), **({"seed": cfg["seed"]} if "seed" in cfg else {})}` (the rest of a fit.yaml is ignored by `aj basis`) and call `defaults_for(basis_parser, flat, basis_fields=(), basis_dest_map={})`, so every key is validated against `aj basis`'s own dests; for `fit`, `basis_fields=BasisSpec.FIELDS`, `basis_dest_map={"embedding": "basis_embedding"}`. Wrap `ValueError` from `runfile` into `subparser.error(str(e))`. Then `sub.set_defaults(**defaults)`, re-parse `argv`, compute `explicit_dests`, and print `override: {k} {defaults[k]!r} -> {getattr(a, k)!r} (command line)` for `k in defaults and k in explicit`. Then `_check_fit_args`.

- [ ] **Step 3: Tests**

Run: `uv run pytest tests/test_runfile.py tests/test_cli_fit_basis.py tests/test_gp_cli.py -q -n 0` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ace_jax/runfile.py src/ace_jax/cli.py tests/test_runfile.py
git commit -m "feat(cli): --config fit.yaml for aj fit / aj basis (CLI > file > default; did-you-mean; paths relative to the file)"
```

---

### Task 7: Resolved `out/fit.yaml` with provenance; reproduce round-trip

**Files:**
- Modify: `src/ace_jax/runfile.py` (`resolved(a, data, *, parser, basis_fields, basis_dest_map) -> dict`, `write(path, d)`)
- Modify: `src/ace_jax/cli.py` (`run` writes `out/fit.yaml`)
- Test: `tests/test_runfile_roundtrip.py`

**Interfaces:**
- Consumes: the namespace after `_parse`; `FitData` (`meta["elements"]`, `r0`, `basis`); `ace_jax_coupling.build_info()` (optional).
- Produces: `<out>/fit.yaml` = every fit dest with its effective value (paths absolute, `config` dropped), plus `basis:` (effective `BasisSpec` fields, `elements` as symbols, explicit) or `model:` (absolute path), `r0` explicit, and `provenance: {ace_jax, coupling: {version, et_repo, et_rev} | null, command, created}`.

- [ ] **Step 1: Failing tests**

`tests/test_runfile_roundtrip.py`:
```python
import jax
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR
from test_basis_build import _primed_cache

from ace_jax.cli import main

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
FAST = ["--m-per-species", "0", "--rungs", "map", "--map-steps", "20", "--opt", "adam",
        "--configs-per-batch", "4"]


def test_resolved_yaml_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
          "--train", str(XYZ), "--out", str(tmp_path / "a"), *FAST])
    d = yaml.safe_load((tmp_path / "a" / "fit.yaml").read_text())
    assert d["basis"]["elements"] == ["Si"] and d["basis"]["order"] == 3
    assert isinstance(d["r0"], float) and "model" not in d
    assert d["provenance"]["command"].startswith("aj fit --order 3 --max-degree 10")
    assert "config" not in d


def test_resolved_yaml_reproduces_from_other_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
          "--train", str(XYZ), "--out", str(tmp_path / "a"), *FAST])
    other = tmp_path / "elsewhere"; other.mkdir(); monkeypatch.chdir(other)
    main(["fit", "--config", str(tmp_path / "a" / "fit.yaml"), "--out", str(tmp_path / "b")])
    za, zb = np.load(tmp_path / "a" / "model.npz"), np.load(tmp_path / "b" / "model.npz")
    assert all(np.array_equal(za[k], zb[k]) for k in za.files)
    ya = yaml.safe_load((tmp_path / "a" / "fit.yaml").read_text())
    yb = yaml.safe_load((tmp_path / "b" / "fit.yaml").read_text())
    ya.pop("provenance"); yb.pop("provenance"); ya.pop("out"); yb.pop("out")
    assert ya == yb
```
(`provenance.command` is `" ".join(["aj", *argv])`, argv as passed to `main`.)
Run: `uv run pytest tests/test_runfile_roundtrip.py -q -n 0` → FAIL (no `fit.yaml`).

- [ ] **Step 2: Implement**

`runfile.resolved(...)`: start from `{k: v for k, v in vars(a).items() if k in fit_dests and k not in basis dests and k not in ("config", "cmd")}`; make `PATH_KEYS` values absolute; if a basis was built: `basis = {f: getattr(spec, f) for f in BasisSpec.FIELDS}` with `elements` → symbols list from `data.meta["elements"]`, `embedding` absolute unless `identity`/None; else `model` absolute. Set `r0 = float(a.r0 if a.r0 is not None else data.r0)`. `rungs` → list. Add `provenance` (`ace_jax.__version__` or `importlib.metadata.version("ace-jax")`; `coupling`: `build_info()` subset or `None` if import fails; `command`; `created`: UTC ISO seconds). `write(path, d)`: `yaml.safe_dump(d, sort_keys=False)` with a leading comment line `# resolved run file written by aj fit: aj fit --config <this file> reproduces it`.

`cli.run(a)`: `runfile.write(pathlib.Path(a.out) / "fit.yaml", runfile.resolved(a, data, ...))` after `write_outputs`. Store the raw argv in `a._argv` in `_parse` for the provenance command.

- [ ] **Step 3: Tests**

Run: `uv run pytest tests/test_runfile_roundtrip.py tests/test_runfile.py tests/test_cli_fit_basis.py -q -n 0` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ace_jax/runfile.py src/ace_jax/cli.py tests/test_runfile_roundtrip.py
git commit -m "feat(cli): every fit writes a resolved out/fit.yaml (+ provenance) that reproduces it"
```

---

### Task 8: No visible backend; docs; CI paths

**Files:**
- Create: `tests/test_no_backend_names.py`
- Modify: `README.md` (quickstart one-liner, install, remove `[basis]`), `skills/ace-jax/SKILL.md` (workflow: fit builds the basis; `fit.yaml`; `aj basis` optional), `CLAUDE.md` (layout: `basis/`, `runfile.py`; extras list; skip list; env var), `docs/basis.md` (API: `BasisSpec`, `build_basis`, fitting a `Basis` object), `coupling/python/README.md`, `.github/workflows/coupling-wheels.yml` (paths: `src/ace_jax/basis/**`, `pyproject.toml`, `uv.lock`; parity job test list with renamed files)
- Modify: any user-facing string found by the new test

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Failing test**

`tests/test_no_backend_names.py`:
```python
"""Users never meet the Julia backend: not in help, messages or user docs."""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
USER_DOCS = ["README.md", "skills/ace-jax/SKILL.md", "docs/basis.md"]


def test_cli_help_has_no_backend_names():
    from ace_jax.cli import _parser
    top = _parser()
    texts = [top.format_help()] + [p.format_help() for p in top._subparsers._group_actions[0].choices.values()]
    assert not any(re.search(r"julia", t, re.I) for t in texts)


def test_user_docs_have_no_backend_names():
    hits = [f for f in USER_DOCS if re.search(r"julia", (ROOT / f).read_text(), re.I)]
    assert not hits, hits


def test_source_strings_have_no_backend_names():
    r = subprocess.run(["git", "grep", "-n", "-i", "julia", "--", "src/ace_jax"], cwd=ROOT,
                       capture_output=True, text=True)
    offenders = [l for l in r.stdout.splitlines()
                 if re.search(r"""(raise |print\(|log\(|help=|warn)""", l)]
    assert not offenders, offenders
```
Run: `uv run pytest tests/test_no_backend_names.py -q -n 0` → FAIL (README/SKILL/`docs/basis.md` mention Julia; possibly help strings).

- [ ] **Step 2: Rewrite docs and strings**

- README: the intro "Author … compiled ahead of time so no Julia is needed" → describe what the user gets ("fit an ACE model straight from data: `aj fit` builds the basis"); Install: single `pip install ace-jax` line for all platforms with the note "building a new basis (`aj fit --order …`, `aj basis`) needs Linux x86_64/aarch64 or macOS arm64; elsewhere fit from an existing `.npz` with `--model`" and the pre-release note reworded (no `[basis]`). Quickstart: `aj fit --order 3 --max-degree 10 --train train.xyz --out fit/` then `aj fit --config fit/fit.yaml`. Keep the ACEpotentials parity explanation, but speak of "the coupling" / "EquivariantTensors", not Julia. The "Julia parity (maintainers / CI only)" section stays — it is maintainer-facing — **but** rename it "Reference parity (maintainers / CI)" and drop the word Julia from the user-facing sentences; references to `julia/*.jl` file paths are fine only if the test allows paths — adjust `test_user_docs_have_no_backend_names` to ignore lines containing `julia/` or `.jl` (maintainer file references), not prose.
- SKILL.md: workflow section: 1. `aj fit` directly (with `--order/--max-degree`), `fit.yaml` reproduction; `aj basis` only to save a basis; pitfalls: first build of a new shape takes seconds and is cached.
- `docs/basis.md`: title "Building a basis (Python and CLI)"; `BasisSpec`/`build_basis`; `FitConfig(model=Basis|BasisSpec)`; remove backend narrative (point to `docs/coupling-etshim-spec.md` for maintainers).
- CLAUDE.md: layout (`src/ace_jax/basis/`, `src/ace_jax/runfile.py`), setup (no extras for basis; dev: `uv sync --reinstall-package ace-jax-coupling` with the bundle env vars), skip list (`require_coupling_lib`), env var `ACEJAX_COUPLING_CACHE_ONLY`. CLAUDE.md is maintainer-facing: Julia may stay where it describes building the library.
- Workflow: update the `paths:` filter and the parity job's test list (`tests/test_basis_build.py`, add `tests/test_cli_fit_basis.py`, `tests/test_fit_basis_pipeline.py`).

- [ ] **Step 3: Tests**

Run: `uv run pytest tests/test_no_backend_names.py tests/test_names.py -q -n 0` → PASS. Run: `uv run pytest -q` → PASS. Run: `uv run pre-commit run --all-files` → PASS. With the compiled library installed (`ACEJAX_COUPLING_BUNDLE=... uv sync --reinstall-package ace-jax-coupling`): `uv run pytest -q` → PASS.

- [ ] **Step 4: Commit and open the stacked PR**

```bash
git add -A README.md skills CLAUDE.md docs coupling/python/README.md .github tests src
git commit -m "docs: aj fit builds the basis; fit.yaml; no backend vocabulary in user-facing text"
git push -u origin feat/fit-basis
gh pr create --draft --base feat/trim-coupling --head feat/fit-basis \
  --title "aj fit builds the basis; fit.yaml run file; ace_jax.basis" --body-file <body>
```
(PR body: the spec summary, the renames table, the test list, stacked-on-#20 note, and the Claude Code attribution line.)

---

## Self-review notes

- Spec coverage: §1 CLI → Tasks 2, 5; §2 fit.yaml → Tasks 6, 7; §3 hand-off → Task 4; §4 backend → Tasks 3, 8; §5 renames → Task 1; §6 errors → Tasks 3–6; §7 tests → each task.
- The spec's "before any data is loaded" is amended in Task 3 (Global Constraints note).
