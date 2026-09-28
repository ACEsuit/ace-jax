# CLAUDE.md

ace-jax fits and evaluates ACE and PACE interatomic potentials in Python/JAX,
with a hybrid ACE + GP fit and a calibrated UQ ladder. No Julia is needed to
fit or evaluate; Julia is only used to author new coupling tables, and in the
parity CI jobs. User docs: `README.md`. Agent-facing usage guide:
`skills/ace-jax/SKILL.md` (keep it in step with CLI and API changes).

## Layout

- `src/ace_jax/eval/`: the model core.
  - `edge_model.py`: `EdgeSiteModel`, the shared base.
  - `model.py`: `ACEModel`.
  - `pace_model.py`, `pace_io.py`, `pace_radial.py`, `pace_build.py`: `PACEModel`, `.yace` read/write.
  - `io.py`: `load`. `nlist.py`: neighbour lists. `harmonics.py`, `radial.py`.
- `src/ace_jax/calc/`: ASE calculators. `point.py` has `ACECalculator`; `gp.py` has `GPCalculator`.
- `src/ace_jax/fit/`: linear and GP fitting.
  - `data.py` reads extxyz into `Config`s and padded `Dataset` batches.
  - The core modules are `rows`, `stats`, `objective`, `ladder`, `predict`, `pops`, `solve`, `hostcache`.
  - `pipeline/` is the `aj fit` pipeline.
  - `radial_learn.py` and `varpro.py` learn radials.
- `src/ace_jax/construct/`: Python model authoring.
  - `spec.py`, `coupling.py`: the EquivariantTensors shim via juliacall.
  - `model.py`: `build_model`, `build_embedding_model`.
  - `prior.py`: the smoothness prior. `export.py`: `save_npz`.
- `src/ace_jax/export/lammps.py`: `export_lammps`, a lammps-jax bundle.
- `src/ace_jax/cli.py`: `ace-jax`/`aj` with `construct`, `fit` and `eval`.
- `tests/`: the pytest suite. `conftest.py` holds the shared fixtures and helpers.
- `fixtures/`: committed reference data. These are bit-exact goldens: never hand-edit or reformat them.
- `bench/`: benchmark and research drivers.
  - `acegp_cantor/run.py`: the research fit driver.
  - `scaling/`: the benchmark suite, tested by `tests/test_bench_scaling.py`.
  - `learn_radial/`.
- `julia/`: ACEpotentials reference generators.
- `pace_ref/`: ML-PACE and python-ace reference tooling.
- `spike/`: throwaway experiments, not linted.
- `docs/`: specs, plans and results. `docs/benchmarks.md` has the performance numbers.

## Setup and tests

```bash
uv sync                               # core + dev group (pytest, xdist, split, numpyro, optax, blackjax, ruff, pre-commit)
uv run pytest                         # full fast suite, 6 xdist workers (~4 min); `slow` excluded
uv run pytest tests/test_efv.py       # targeted runs stay single-process
uv run pytest -m slow                 # real-model MCMC ladder, bit-exact driver goldens
uv run ruff check                     # lint; `uv run pre-commit run --all-files` = the CI lint job
```

- **Extras:**
  - `gp`: numpyro, optax, blackjax. The dev group mirrors it.
  - `authoring`: juliacall and juliapkg. They provision the Julia pinned in `juliapkg.json`.
  - `cuda`.
  - `fast-neighbours`: matscipy-neighbours, a C++ source build.
- **Tests that silently skip** when an optional dependency is missing:
  - `authoring`: `test_coupling_etshim`, `test_coupling_parity`. The coupling cache tests still run: they use `ACEJAX_NO_JULIA=1` and the committed cache.
  - lammps-jax: `test_export_lammps` and `test_bench_scaling`'s export test. Make lammps-jax importable with `PYTHONPATH=<lammps-jax>/python` or `uv pip install -e <lammps-jax>`.
  - `fast-neighbours` (matscipy-neighbours): `test_calc_jit`'s native `neighbour_matrix` test and `test_efv`'s dense-vs-neighbour_matrix check; without it the skin list and dense layout also take their fallback neighbour path.
  - `pyace` (python-ace, in its own venv under `pace_ref/`).
  - `sphericart`, `psutil`.
  - PACE fixtures.
- **Test environment variables:**
  - `ACEJAX_REQUIRE_FIXTURES=1` turns a missing-fixture skip into a failure. CI parity jobs set it.
  - `ACEJAX_REQUIRE_OPTIONAL=1` does the same for the lammps-jax and matscipy-neighbours tests (sites use `conftest.require_optional`, not `pytest.importorskip`). The `optional-deps` CI job sets it.
  - `ACEJAX_TEST_WORKERS` sets the xdist worker count. `-n 0` runs serially.
  - `ACEJAX_FIXTURE_DIR` points the suite at other exports.
  - `ACEJAX_CLI_FULL=1` runs the full CLI test.
- `tests/conftest.py` sets `XLA_FLAGS=--xla_force_host_platform_device_count=2`, for the sharding tests, and a persistent compile cache in `.jax_cache/`. Both must be set before any jax import, so conftest must not import jax at the top level.
- **CI** (`.github/workflows/`):
  - `test.yml`: 3 pytest-split shards on Python 3.12, a smoke job on 3.11 and 3.13, the `slow` ladder, and `optional-deps` (matscipy-neighbours plus lammps-jax pinned to a commit, with `ACEJAX_REQUIRE_OPTIONAL=1`).
  - `lint.yml`.
  - Path-gated parity jobs: `julia-parity` (ACEfit rows/QR), `coupling-parity` (ET vs ACEpotentials), `prior-parity`, `pace-parity` (ML-PACE C++ + python-ace).
  - pytest-split balances on `.test_durations`. Refresh it with `pytest --store-durations` when adding slow tests.

## Conventions

- **float64:** the library never calls `jax.config.update`; the caller chooses precision. The exceptions are `cli.py` and `fit/baseline.py`, which enable x64 at import.
  - Tests and scripts put `jax.config.update("jax_enable_x64", True)` at the top of the module. CI also sets `JAX_ENABLE_X64=1`.
  - Fitting and learned radials require float64: `radial_learn` raises without it.
  - Use `highest_precision()` (a matmul-precision context) around numerics that are compared to references.
- **Model files:**
  - `.npz` models follow the export schema: `meta_json` with `schema_version: 1`. The writers are `julia/export_model.jl` and `construct.export.save_npz`.
  - `aj.load(path)` returns `(model, meta, z)`. For a `.yace` it returns a `PACEModel` with `(model, meta, spec)`, and `write_yace(model, spec, path)` writes it back.
  - A fitted GP is `gp_model.npz`, identified by its `gp_json` key. Load it with `GPCalculator.from_file`.
  - Large models are release assets, not commits.
- **Architecture:**
  - A model maps edge vectors `rij`, never positions and a cell, to site energies. That is what lets one core serve ASE and LAMMPS.
  - `EdgeSiteModel` owns E/F/virial: one `value_and_grad` over `rij`. It also owns the A-basis product (`edge_a`, `edge_a_kind` "gather" or "matmul"), the dense and sparse layouts (`LAYOUTS`) and `estimate_a_bytes`.
  - `ACEModel` and `PACEModel` subclass `EdgeSiteModel` and provide `site_energies` and `site_energies_dense`.
  - `ACECalculator(path, layout="auto", edge_a_kind="auto", skin=1.0)` picks dense when the padding fill is at least `MIN_DENSE_FILL` (a pure fill test: the dense model runs in `CHUNK_NODES` blocks, so memory is not a criterion). The sparse edge list is padded to a power-of-two bucket, so MD reuses the jitted function, and gather is calibrated against matmul per edge bucket.
  - `skin > 0` (dense only, `calc/skin.py`) reuses a Verlet list built for cutoff + skin until an atom moves skin / 2 or the cell, pbc, species or atom count change; `last_timing["rebuilds"]` counts builds. `skin=0` rebuilds every call and must match the skin path to 1e-12. Setting `calc.model` or `calc.skin` drops the list; the compiled step takes the model's arrays per call, so new weights do not retrace.
  - `export_lammps(..., k_dense=, max_owned=)`: dense bundles evaluate owned rows only (recorded as `ace_jax.owned_rows`, never `max_owned`, which lammps-jax claims) in `BUNDLE_BLOCK_ROWS` blocks.
  - **Lean evaluation form** (`eval/model.py::lean`). It composes three exact, load-time transforms of a folded `ACEModel`:
    - `prune_columns`: drop the R_nl columns A never reads, and the Y_lm above the used l.
    - `fold_pair`: fold Wpair into the pair radial.
    - `block_dense`: an l-blocked, species-compact, feature-major dense A, with `blk_aa_specs`; `aa_specs` stay for the sparse path.
    - `ACECalculator(lean=True)` and `export_lammps(lean=True)` apply it; `calc.eval_model` is the result, and `calc.model` stays as given.
    - `load` never applies it. Fitting, descriptors and learned radials need the full basis, and a lean model is `energy_only`: its basis methods raise.
    - `tests/test_lean.py` holds each transform to 1e-12.
  - `PACEModel.sbessel_form` is `"matmul"` (`pace_radial._sbessel_mm`) when `nradbase >= SBESSEL_MATMUL_MIN_K` (12), else the rotation recurrence. `load_yace` fixes it per model.
- **GP fit:**
  - At fixed hyperparameters θ the model is Bayesian linear regression over `[B | k_θ(B, B_M)]`, with streamed sufficient statistics (`fit/stats`, `fit/objective`).
  - The LML is maximised by Adam or L-BFGS (multi-start is `map_restarts`).
  - Ladder rungs: MAP, Laplace, Pathfinder, VI, NUTS. They mix numpyro and blackjax on purpose; see the `fit/ladder.py` docstring.
  - POPS is available for linear-arm misspecification UQ.
  - Hyperparameter blocks are routed fixed or LML via `fit/paramset.py`.
  - `FitConfig` defaults are the research driver's; `cli._fit_config` applies the CLI's own defaults.
- **Code style:** dense, often one-line statements and short math names (`Phi`, `Dt`, `nz`, `l`). Trailing `#` comments explain why, not what. Docstrings record design constraints and the reference being matched (file:line in ACEpotentials, ML-PACE). There is no formatter. Ruff ignores the rules that fight this style (see `pyproject.toml`), so don't reformat unrelated code.
- **Test style:** plain pytest functions.
  - Parity against committed reference fixtures, usually to float noise or bit-for-bit.
  - Finite-difference and autodiff cross-checks.
  - Import shared helpers with `from conftest import FIXTURE_DIR, pace_fixture`.
  - Mark heavy tests `@pytest.mark.slow`.
- **Commits:** conventional-commit prefixes with a scope on branch commits, for example `feat(bench):`, `fix(nlist):`, `refactor(gp)!:`, `test(ladder):`, `docs(...)`, `ci:`, `chore:`. PRs are squash-merged under a plain title with `(#N)`.

## Pitfalls

- **Reading data:**
  - Training and eval data are read with libAtoms `extxyz` (`fit/xyz.py`), never `ase.io.read`.
  - ASE ≥ 3.23 moves extxyz `energy`/`forces`/`stress` into a `SinglePointCalculator`, so an `atoms.info` lookup finds nothing. That loss was silent.
  - `read_extxyz` keeps ASE's value conventions (special 3×3 keys, flat 9-vectors, `_JSON`) so the committed fixtures read bit-identically. Keep it that way.
- **Closures:** `jax.tree.map(lambda a: a[i], ds)` inside a loop is the batch-slicing idiom. It is safe because it is consumed in the same iteration. Ruff B023 is suppressed per file for exactly this.
- **Python 3.11:** `requires-python` is `>=3.11`, so no PEP 701 f-strings (`f"{d["k"]}"`) in `src/` or `tests/`.
- **Laplace compile time:** the Laplace rung (a Hessian through the whole LML) can take tens of minutes to compile. Keep `--rungs map` in quick checks.
- **Coupling cache:** the first `construct` of a new basis shape runs Julia. The per-shape cache lives in `~/.cache/ace-jax/coupling`, or `$ACEJAX_COUPLING_CACHE`.
