# CLAUDE.md

ace-jax fits and evaluates ACE and PACE interatomic potentials in Python/JAX,
with a hybrid ACE + GP fit and a calibrated UQ ladder. No Julia is needed to
fit or evaluate. New coupling tables come from `ace-jax-coupling`, a
`juliac --trim` compiled EquivariantTensors shipped as a platform wheel (no
Julia at runtime); Julia itself runs only when building that library and in the
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
  - `pipeline/` is the `aj fit` pipeline (`radials.py`: the `--learn-radial` stage).
  - `radial_learn.py` learns radials. `varpro.py` (the FS-density VarPro driver), `density.py` and `block_lbfgs.py` are research tools kept from PR #22 and exercised only by their tests; no pipeline path uses them.
- `src/ace_jax/basis/`: building an ACE basis (a frozen, zero-readout model).
  - `spec.py`, `coupling.py`: the EquivariantTensors shim via the compiled `ace-jax-coupling` library; `BasisUnavailable` when it cannot run.
  - `model.py`: `BasisSpec`, `build_basis` (the one entry point: `aj basis`, `aj fit`, Python), over `build_model` / `build_embedding_model`; `Basis` (NamedTuple), `basis_r0`.
  - `prior.py`: the smoothness prior. `export.py`: `save_npz`.
- `src/ace_jax/export/lammps.py`: `export_lammps`, a lammps-jax bundle.
- `src/ace_jax/tutorials/`: tutorial support, not stable API. `labels.py` serves shipped MACE labels from content-keyed extxyz caches (live MACE only on a miss, if mace-torch is installed); `structures.py` builds the school tutorials' structures deterministically; `campaign.py` (reference-basis descriptors, seeded Langevin MD pools, novelty scores) and `curation.py` (the MD-select-label-refit loop of tutorial 7, shared with `make_labels.py e3`).
- `src/ace_jax/cli.py`: `ace-jax`/`aj` with `basis`, `fit` and `eval`. `aj fit` builds the basis from `--order/--max-degree` (flags shared with `aj basis` via `add_basis_args`) or takes `--model`; `_parse` layers a `--config fit.yaml` under the command line.
- `src/ace_jax/runfile.py`: `fit.yaml` read/validate/merge (`defaults_for`, `explicit_dests`) and the resolved `out/fit.yaml` writer (`resolved`, `write`).
- `tests/`: the pytest suite. `conftest.py` holds the shared fixtures and helpers.
- `fixtures/`: committed reference data. These are bit-exact goldens: never hand-edit or reformat them.
- `bench/`: benchmark and research drivers.
  - `acegp_cantor/run.py`: the research fit driver.
  - `scaling/`: the benchmark suite, tested by `tests/test_bench_scaling.py`.
  - `learn_radial/`.
- `coupling/`: the `ace-jax-coupling` distribution.
  - `julia/src/ETCouple.jl`: the C ABI over EquivariantTensors (fork rev pinned in `julia/Project.toml` `[sources]`); `julia/build.jl` compiles it with JuliaC (`--trim=safe`, Julia 1.13); `julia/reference/` is the unpatched-upstream oracle.
  - `python/`: the ctypes package `ace_jax_coupling` and its wheel hook; `tools/`: bundle check, trace-based pruning, clean-env wheel test.
  - Spec: `docs/dev/coupling-etshim-spec.md`.
- `julia/`: ACEpotentials reference generators.
- `pace_ref/`: ML-PACE and python-ace reference tooling.
- `docs/`: `docs/user/` is the site; `docs/dev/` holds specs, plans, benchmark reports and research results (`docs/dev/benchmarks.md` has the performance numbers). Removed research prototypes: tag `archive/research-prototypes` (`bench/ARCHIVED.md`).
  - `docs/user/`: the user documentation site (MkDocs Material, `mkdocs.yml`, toolchain pinned in `docs/requirements.txt`; build with `uv pip install -r docs/requirements.txt && uv run --no-sync mkdocs build --strict`). `docs/mkdocs_hooks.py` renders the CLI reference from `aj --help`; `docs/snippets/` holds shared fragments (the install line). Tutorials are marimo notebooks in `docs/user/tutorials/notebooks/` with PEP 723 headers; keep each a few CPU minutes (tutorial 5's basis sweep, ~4 min, and tutorial 7's three campaigns, ~5 min, are the exceptions). The MLIP-school-derived tutorials (`school_*.py`) read labels from `docs/user/tutorials/data/school/` (MIT, MACE-MPA-0 / MP-0b3; regenerate with its `make_labels.py` in a separate mace-torch environment), falling back to GitHub `main` when not run from a checkout; the docs build never needs torch. Their pages (`tutorials/<page>.md` + `<page>_files/`, gitignored) are rendered by `docs/build_tutorials.py`, run from the mkdocs hook: it runs each notebook (`marimo export ipynb --include-outputs`) only when its source changed, maps marimo callouts/accordions/controls to admonitions/details/notes, and fails the build if a cell raises; `ACEJAX_DOCS_NOTEBOOKS=skip` writes placeholders for a quick local build. The run command and molab link live in each notebook's first cell. `tests/test_no_backend_names.py` also covers `docs/user` (except `licence.md`).

## Setup and tests

```bash
uv sync                               # core + dev group (pytest, xdist, split, numpyro, optax, blackjax, ruff, pre-commit)
uv run pytest                         # full fast suite, 6 xdist workers (~4 min); `slow` excluded
uv run pytest tests/test_efv.py       # targeted runs stay single-process
uv run pytest -m slow                 # real-model MCMC ladder, bit-exact driver goldens
uv run ruff check                     # lint; `uv run pre-commit run --all-files` = the CI lint job
```

- **Extras:**
  - `gp`: blackjax, for the pathfinder rung only (numpyro and optax are core: the MAP and the optimisers). The dev group includes it. `tests/test_core_deps.py` fails if a module-level import is not a core dependency: import optional packages inside the function that needs them.
  - (no extra for building bases: `ace-jax-coupling` from PyPI is a core dependency on Linux x86_64/aarch64, macOS arm64 and Windows x64; no Julia.) To try a locally built library without reinstalling, point the installed package at it with `ACEJAX_COUPLING_LIB=<bundle>/lib/libetcouple.<so|dylib>` (Windows: `<bundle>/bin/libetcouple.dll`), or `uv pip install` its wheel and use `uv run --no-sync`. Building the library: `coupling/RELEASING.md` and `coupling-wheels.yml` (JuliaC on Julia 1.13.1, then `coupling/tools/prune_bundle.py` and `check_bundle.py`).
  - `cuda`.
  - `fast-neighbours`: matscipy-neighbours, a C++ source build.
- **Tests that silently skip** when an optional dependency is missing:
  - compiled coupling library: `test_coupling_etshim`, `test_coupling_parity` and the basis bridge tests skip unless a compiled `ace_jax_coupling` is installed (`conftest.require_coupling_lib`). The coupling cache tests still run: they use `ACEJAX_COUPLING_CACHE_ONLY=1` and the committed cache.
  - lammps-jax: `test_export_lammps` and `test_bench_scaling`'s export test. Make lammps-jax importable with `PYTHONPATH=<lammps-jax>/python` or `uv pip install -e <lammps-jax>`.
  - `fast-neighbours` (matscipy-neighbours): `test_calc_jit`'s native `neighbour_matrix` test and `test_efv`'s dense-vs-neighbour_matrix check; without it the skin list and dense layout also take their fallback neighbour path.
  - `pyace` (python-ace, in its own venv under `pace_ref/`).
  - `sphericart`, `psutil`.
  - `marimo` (a docs dependency): `test_tutorial_notebooks`, run by the `docs` CI job instead.
  - PACE fixtures.
- **Test environment variables:**
  - `ACEJAX_REQUIRE_FIXTURES=1` turns a missing-fixture skip into a failure. CI parity jobs set it.
  - `ACEJAX_REQUIRE_OPTIONAL=1` does the same for the lammps-jax and matscipy-neighbours tests (sites use `conftest.require_optional`, not `pytest.importorskip`). The `optional-deps` CI job sets it.
  - `ACEJAX_TEST_WORKERS` sets the xdist worker count. `-n 0` runs serially.
  - `ACEJAX_FIXTURE_DIR` points the suite at other exports.
  - `ACEJAX_CLI_FULL=1` runs the full CLI test.
- `tests/conftest.py` sets `XLA_FLAGS=--xla_force_host_platform_device_count=2`, for the sharding tests, and a persistent compile cache in `.jax_cache/`. Both must be set before any jax import, so conftest must not import jax at the top level.
- **CI** (`.github/workflows/`):
  - `test.yml`: 3 shards on Python 3.12 (whole files each: `ACEJAX_SHARD=i/n`, `conftest.shard_files`), a smoke job on 3.11, 3.13 and 3.14, the `slow` ladder, and `optional-deps` (matscipy-neighbours plus lammps-jax pinned to a commit, with `ACEJAX_REQUIRE_OPTIONAL=1`).
  - `lint.yml`.
  - `docs.yml`: each tutorial notebook rendered in its own matrix job (`build_tutorials.py --matrix`; the page is cached on a hash of the notebook, converter, `src/`, lock and tutorial data), then the strict site build from those pages (`ACEJAX_DOCS_NOTEBOOKS=require`: a page not current fails rather than re-renders); path-gated; on `main` it deploys to GitHub Pages.
  - Path-gated parity jobs: `julia-parity` (ACEfit rows/QR), `coupling-wheels` (builds + clean-env-tests the coupling wheels, then parity vs ACEpotentials), `prior-parity`, `pace-parity` (ML-PACE C++ + python-ace).
  - The shards balance whole files on `.test_durations` (per-test splitting scattered modules, and each shard repaid their fixtures and first compiles). Refresh it from the `durations` artifact of a recent main run, or `pytest --store-durations`, when adding slow tests. Every shard restores all three groups' JAX caches, so a file moving between shards stays warm. The JAX caches are cleared per module (conftest `_release_jax_memory`), not per test: share expensive fits as module/session fixtures.

## Conventions

- **float64:** the library never calls `jax.config.update`; the caller chooses precision. The exceptions are `cli.py` and `fit/baseline.py`, which enable x64 at import.
  - Tests and scripts put `jax.config.update("jax_enable_x64", True)` at the top of the module. CI also sets `JAX_ENABLE_X64=1`.
  - Fitting and learned radials require float64: the pipeline's `fit` and `radial_learn` raise without it.
  - Use `highest_precision()` (a matmul-precision context) around numerics that are compared to references.
- **Model files:**
  - `.npz` models follow the export schema: `meta_json` with `schema_version: 1`. The writers are `julia/export_model.jl` and `basis.export.save_npz`.
  - `aj.load(path)` returns `(model, meta, z)`. For a `.yace` it returns a `PACEModel` with `(model, meta, spec)`, and `write_yace(model, spec, path)` writes it back.
  - A fitted GP is `gp_model.npz`, identified by its `gp_json` key. Load it with `GPCalculator.from_file`.
  - Large models are release assets, not commits.
- **Architecture:**
  - A model maps edge vectors `rij`, never positions and a cell, to site energies. That is what lets one core serve ASE and LAMMPS.
  - `EdgeSiteModel` owns E/F/virial: one `value_and_grad` over `rij`. It also owns the A-basis product (`edge_a`, `edge_a_kind` "gather" or "matmul"), the dense and sparse layouts (`LAYOUTS`) and `estimate_a_bytes`.
  - `ACEModel` and `PACEModel` subclass `EdgeSiteModel` and provide `site_energies` and `site_energies_dense`.
  - `ACECalculator(path, layout="auto", edge_a_kind="auto", skin=1.0)` picks dense when the padding fill is at least `MIN_DENSE_FILL` (a pure fill test: the dense model runs in `CHUNK_NODES` blocks, so memory is not a criterion). The sparse edge list is padded to a power-of-two bucket, so MD reuses the jitted function, and gather is calibrated against matmul per edge bucket.
  - `skin > 0` (dense only, `calc/skin.py`) reuses a Verlet list built for cutoff + skin until an atom moves skin / 2 or the cell, pbc, species or atom count change; `last_timing["rebuilds"]` counts builds. `skin=0` rebuilds every call and must match the skin path to 1e-12. Setting `calc.model` or `calc.skin` drops the list; the compiled step takes the model's arrays per call, so new weights do not retrace. A new analytic radial is re-splined, and it retraces only if its interval count lands in a different `splinify.BUCKETS` bucket (quarter-octave steps; `spline_intervals` pins it).
  - `export_lammps(..., k_dense=, max_owned=, max_neighbors=)`: dense and matrix bundles evaluate owned rows only (recorded as `ace_jax.owned_rows`, never `max_owned`, which lammps-jax claims: its reader takes a key name from anywhere in the file, and `tests/test_export_lammps.py::LAMMPS_JAX_KEYS` lists them) in `BUNDLE_BLOCK_ROWS` blocks.
  - Bundle layouts: `sparse`, `dense` (lammps-jax's edge buffer, argsort-packed into slots every step) and `matrix` (lammps-jax's neighbour-matrix input: the LAMMPS full list, copied only on list rebuilds, so it holds skin pairs the model masks at rcut; no per-step packing; `k_dense < max_neighbors` compacts in-cutoff pairs into tighter model slots). `layout="matrix"` requires `max_neighbors` (list slots; a `k_dense` sized for rcut is too small for the rcut + skin list). `layout="auto"` picks `matrix` over `dense` only when `max_neighbors` is passed, `matrix_supported()`, and one block + `matrix_prep_bytes` fits. The bundle records `ace_jax.lammps_jax`; an older LAMMPS plugin rejects matrix bundles (rebuild it, or `layout="dense"`).
  - `neighbour_capacity(atoms, rcut, skin, slots="skin"|"cutoff")` sizes the buffers. `"skin"` is the safe default; `"cutoff"` (tight model slots, 1.2-1.4x on Cantor) is opt-in for stable MD only. Overflow is always NaN or a LAMMPS abort, never truncation.
  - **Lean evaluation form** (`eval/model.py::lean`). It composes three exact, load-time transforms of a folded `ACEModel`:
    - `prune_columns`: drop the R_nl columns A never reads, and the Y_lm above the used l.
    - `fold_pair`: fold Wpair into the pair radial.
    - `block_dense`: an l-blocked, species-compact, feature-major dense A, with `blk_aa_specs`; `aa_specs` stay for the sparse path.
    - `ACECalculator(lean=True)` and `export_lammps(lean=True)` apply it; `calc.eval_model` is the result, and `calc.model` stays as given.
    - `load` never applies it. Fitting, descriptors and learned radials need the full basis, and a lean model is `energy_only`: its basis methods raise.
    - A lean model holds the radial twice (`rnl_coefs` sparse, `blk_rnl_coefs` dense). Radial editors (`fit/radial_model.py`, `patch_radial_npz`, `save_npz`) call `model.require_full()`. Edit the full model and re-apply `lean`.
    - `export_lammps(layout="auto")` sizes memory on the full model: `estimate_a_bytes` underestimates the blocked path's temp on lean widths.
    - `tests/test_lean.py` holds each transform to 1e-12.
    - **Learned radials** (`ACEModel.radial_learned`, a static field): set by `fit.radial_model.with_radial` (`to_analytic` clears it; `widen_radial` keeps it), stored as meta_json `"radial_learned"` by `patch_radial_npz` (hence `radial_learn.save_result`, unless the gate kept `"init"`), `save_npz`/`eval_pair`, and `mark_radial_learned(path)`; `load` reads it, defaulting to False for old files. `lean(model, spline_tol="auto", spline_intervals=None)` decides in `splinify.spline_plan` alone: `"auto"` splines an analytic R_nl only when `radial_learned`, at `DEFAULT_SPLINE_TOL` (1e-10), and never the pair radial (never learned); a float splines every analytic radial (the opt-in for Julia `ace_model` exports and authored models, which stay exact by default); None never splines. The conversion `eval/splinify.py::to_spline`, which tabulates sum_q W P_q(x) as the cubic B-spline `spline_eval` reads, taking the smallest quarter-octave bucket of intervals whose relative R_nl error (on the columns A reads) is <= tol. `tol` bounds values; the S' error (O(h^3), what forces see) is reported and optionally gated (`deriv_tol`). That is an approximation: lean-splined agrees with full to about tol, not roundoff (at 1e-10: energies up to ~1e-9 relative, forces up to ~2.3e-8 of max|F|; `tests/test_to_spline.py`). `calc.splined` and `model.splining(before, after, tol)` report what happened, and the bundle records it from the lean result (looking through `.base`). `ACECalculator` and `export_lammps` take `spline_tol` (every bundle layout, matrix included, traces the lean, splined model; the bundle records `ace_jax.spline_tol`). The Julia-parity tests run analytic exports with the defaults and are exact (not learned, so not splined). `to_spline` is cached (content-keyed, 256 MB, thread-locked LRU in `splinify`, `clear_cache()`), so readout-only `calc.model` swaps skip it; prune/fold/block cost <5 ms and are not cached. `_rnl_owner` then sees ACE1's per-z_j column pattern, which learned radials keep, so compaction applies. Fitting keeps the analytic model; UQ variances come from the full model.
    - **Radial tables in r** (`splinify.radial_table`, opt-in): `radial_table=` on `lean`, `ACECalculator` and `export_lammps` (None off, the default; True = `DEFAULT_RADIAL_TABLE` 4000 intervals; an int) tabulates the whole per-edge radial stage as one cubic B-spline per species pair on a uniform grid in r over [`DEFAULT_TABLE_R_MIN` 0.5 Å, max per-pair rcut]: ACE R_nl with transform and envelope plus the (folded) pair radial (`rtab_coefs`; `blk_rtab_coefs` for the compact blocked path), PACE g_k (`rtab_coefs`; core repulsion stays analytic, tabulating it gained nothing). A CPU speed-up (removes per-edge transcendentals). Samples and end curvature come from the model's own exact code (`radial_table_values`, autodiff), so no radial maths is duplicated; values are masked to exact zeros from each pair's cutoff on (ACE: `pair_envelope[..., 0]`, and building refuses an R_nl that does not vanish there; PACE: the bond rcut), so skin edges contribute nothing; below r_min the end cubic extrapolates (the ACE pair envelope is singular at r -> 0). An approximation: at 4000 intervals energies ~1e-12 relative, forces <= 1.5e-8 of max|F|, virial <= 4.7e-7 of max|V| on the fixtures (`tests/test_radial_table.py`). `radial()`, `pair_radial()`, `_blocked_radial()` and PACE `edge_basis_factors()` are the single entry points that read it. Applied last, by `model.with_radial_table` (the one helper `lean`, the calculator and the exporter call; wrappers refuse); `calc.radial_table` / `last_timing["radial_table"]` and the bundle's `ace_jax.radial_table` report it. Cached in the `to_spline` LRU (`_memo`) on `radial_table_key()` (never the readout), so readout-only swaps reuse it.
    - **Wrapper models** with `.base` and `with_base(new_base)` (e.g. `FSModel(base, ...)`): `lean` returns `model.with_base(lean_keep_basis(model.base))`. `lean_keep_basis` applies only `to_spline` and `prune_columns`, never `fold_pair` or `block_dense`, so `site_basis`/`site_basis_dense`/`_readout` stay valid (B to roundoff, not always bitwise).
    - The spline radial goes through `radial.spline_eval_pairs`: under a trace it inlines `vmap(spline_eval)(x, coefs[zi, zj])` (XLA fuses it into a 4-row gather), and an eager call goes through a jitted copy, because run eagerly that expression materialises an (E, ncoef, F) per-edge table, tens of GB on a fine `to_spline` grid. Don't wrap the traced call in its own jit: the nested jit changes the caller's fusion and moved the bit-exact `run_linear_pops_auto` golden. Explicit row gathers were 1-12% slower on an A100.
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
- **Coupling cache:** the first build of a new basis shape (`aj fit --order ...`, `aj basis`) calls the coupling library; later ones hit the cache (stamped with `coupling.backend_id()`). The per-shape cache lives in `~/.cache/ace-jax/coupling`, or `$ACEJAX_COUPLING_CACHE`.
