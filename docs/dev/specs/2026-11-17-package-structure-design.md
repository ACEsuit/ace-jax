# Package structure: framework, UQ, applications and bench (design)

Status: design, awaiting approval (2026-11-17). Reviewed against `main` at dfd5373 (0.6.0) and
`feat/hamiltonians` at 6d5bbeb. Builds on `docs/dev/acehamiltonians-port-scoping.md` §9 and
`docs/dev/specs/2026-11-16-hamiltonians-design.md` §2; where they differ, this one wins.

## Intent

Make the framework/application boundary real inside one repository and one distribution, so that
Hamiltonians and other operators reuse the basis, the linear fit and the UQ instead of copying
them, and so that `fit/` stops being one flat directory of 43 modules. Decide what happens to
`bench/` at the same time, because the refactor moves the modules it imports.

## Findings

Sizes (Python lines):

| Tree | Lines | Notes |
|---|---|---|
| `src/ace_jax` | 23.9k | plus 1.4k C++/CUDA; `fit/` 9.9k in 43 flat modules, `fit/pipeline/` 1.9k, `eval/` 5.1k |
| `src/ace_jax/ham` (branch) | 1.4k | 11 modules |
| `tests/` | 30.0k | 176 files |
| `bench/` | 23.8k | 18 MB with results |

Imports:

- `ham` imports only `basis.{coupling,spec,model}` and `eval.{harmonics,nlist,model}`. Nothing
  imports `ham`.
- The two framework refactors of the Hamiltonian design are not done. `ham/features.py` holds a
  carrier `ACEModel` and calls its private `_aa`; `ham/fit.py` has its own NumPy QR and uses none
  of `fit/stats.py`, the evidence, the jackknife, conformal or POPS. The import-boundary test is
  not written.
- `calc` depends on `fit` internals: `calc/point.py` imports `fit.ard`, `jackknife`, `support`,
  `rows`, `inducing` and `data` (lazily, 11 sites); `calc/gp.py` imports five `fit` modules at
  module level. `fit/pipeline/export.py` imports `calc.gp` back.
- `eval` has lazy back-edges: `dense.py` → `calc.skin.compact`, `pace_io.py` → `export.yace`,
  `splinify.py` → `basis.radial_ace1`. `basis/model.py` imports `eval` (it builds an `ACEModel`).

Bench:

- It is three different things in one directory (table in §5).
- It is not a CI cost: the `test_bench_*` files are 32 s of the 3815 s in `.test_durations` (1%).
- It is not in the wheel or the sdist.
- It is tied to the package in both directions. It imports internals in about 150 places, eight
  of them underscore names (`fit.support._proj`, `fit.conformal._pooled_q`,
  `fit.predict._dtc_deriv_residual`, `fit.rows._cat_rows`, `calc.point._round_k`). Eleven source
  files cite `bench/` paths in docstrings, 56 docs files reference it, `tests/test_calc_layout.py`
  reads `bench/perf/results/microbench_memory_float64.json`, and three test files and the pipeline
  golden generator run `bench/acegp_cantor/run.py` or `bench/learn_radial/run.py` as subprocesses.
- Only the scripts under test are checked when a library module moves. The rest break silently.

## Goals

1. An enforced layering of `ace_jax` (§1), checked by a test on every PR.
2. `fit/` split into generic linear fitting, UQ, the GP arm, learned radials and the E/F/V
   pipeline (§2), with `uq` usable by `calc` and by operator models.
3. `eval`'s A/AA product usable without the scalar-energy path (§3); the carrier model gone from
   `ham`.
4. A home for operators other than H and S (§4).
5. `bench/` split by role (§5), with every remaining script's `ace_jax` imports checked.

## Non-goals

- Separate PyPI distributions for the framework, UQ or `ham`. Revisit when `ham` and TT-ACE both
  consume the framework through an interface that has survived a release.
- Renaming the potential-facing packages (`eval`, `calc`, `export`, `basis`) to `core` /
  `potentials`.
- Any change of behaviour, defaults, file formats or the CLI.
- Reducing the size of `tests/`.

## Anti-goals

- A move and a behaviour change in the same PR.
- A `core` package that is a renamed copy of today's energy-specific internals.
- Compatibility shims for underscore names.
- Moving `bench/` to another repository wholesale.

## Constraints

- Pipeline goldens, `.yace` and LAMMPS parity must not move; goldens re-checked on lestrade after
  every PR that touches `fit/` or `eval/`.
- Public names keep working: `ace_jax.{ACEModel, load, ACECalculator, BatchCalculator,
  GPCalculator, site_descriptors, …}`, `ace_jax.fit.pipeline`, `ace_jax.export`, the `aj` CLI,
  everything `skills/ace-jax/SKILL.md` and `docs/user/` name.
- Moved public modules leave a re-export stub at the old path that warns, removed in the next
  release but one (the policy already used for the `fast-neighbours` extra).
- No new runtime dependency. The boundary test uses `ast`, not import-linter.
- `feat/hamiltonians` is long-lived; the order of work must not force a painful rebase.

## Approaches

**Where the boundary lives.**
- *Separate distributions now.* Rejected: the framework API is not settled (the carrier
  `ACEModel._aa` call shows it), one wheel carries the compiled kernel, and a cross-package
  change costs two releases.
- *Rename everything into `core` / `potentials` / `ham`.* Rejected, as in scoping §9: it churns
  every import, the docs and the skill for no gain to `ham`.
- *Layer contract plus targeted moves* (chosen): keep the top-level names, split `fit/`, add
  `uq/` and `operators/`, and enforce the direction of imports.

**Where UQ goes.**
- *`fit/uq/`.* Keeps everything under `fit`, but `calc` and `ham` would still reach into `fit`.
- *Top-level `ace_jax.uq`* (chosen): it is consumed by calculators and by operator models, it is
  the part of the code with a paper behind it, and a top-level name gives it a documented API.

**Bench.**
- *Separate repository for all of it.* Rejected: see Findings. It would trade a 1% CI saving for
  silent breakage on every internal move, starting with this refactor.
- *Leave as is.* Rejected: test drivers, the published benchmark, a cross-code comparison and
  paper analyses have different lifetimes and different audiences.
- *Split by role* (chosen), §5.

## Design

### 1. Layers and the boundary test

A module may import its own layer and the layers above it in this table, never below.

| Layer | Modules |
|---|---|
| 0 framework | `basis.{spec, coupling, prior, export, embedding, radial_init, radial_ace1}`, `eval.{harmonics, radial, splinify, nlist, aa_dag, product, io}` |
| 1 linear fit | `fit.linear.*` |
| 2 UQ | `uq.*` |
| 3 applications | potentials: `basis.model`, `eval.{model, edge_model, pace_*, ffi, dense, api}`, `fit.{rows, data, gp, radial, pipeline, …}`, `calc`, `export`, `batch`, `jaxmd`; operators: `operators`, `ham` |
| 4 front ends | `cli`, `runfile`, `tutorials` |

Within layer 3, potentials and operators do not import each other.

`tests/test_import_layers.py` parses every file under `src/ace_jax` with `ast`, including imports
inside functions, and fails on an upward edge. Existing violations are listed in an allow-list in
the test, each with the step of this spec that removes it; the list may only shrink.

The same test file checks `bench/`: every `ace_jax` import in every bench script must name a
module and an attribute that exist (resolved by `ast` on the source, nothing is executed). This
is what lets the later steps move modules safely.

### 2. `fit/` split

| Destination | Modules today |
|---|---|
| `fit/linear/` | `stats`, `solve`, `objective`, `lml_host`, `lml_svd`, `hostcache`, `prior_root`, `block_lbfgs`, `newton`, `multistart`, `varpro`, `sharding`, `progress` |
| `uq/` | `ard`, `jackknife`, `conformal`, `pops`, `support`, `loo`, `clusters` |
| `fit/gp/` | `kernels`, `inducing`, `ladder`, `predict`, `feature`, `density` |
| `fit/radial/` | `radial_learn`, `radial_model`, `embedding` |
| stays in `fit/` | `rows`, `data`, `xyz`, `weights`, `baseline`, `zbl`, `metrics`, `summary`, `report`, `fit_api`, `pipeline/` |

`hypers`, `paramset` and `rowcheck` are placed by the builder after reading their callers.

Two changes of substance, each in its own PR after the moves:

- **Generic rows** (Hamiltonian design §2): a row generator yields (Φ, Y, w) per quantity, Y with
  any number of right-hand sides; `fit.linear.stats` and the QR evidence take it. E/F/V becomes
  one generator, in `fit/rows.py`. `uq.ard`, `jackknife`, `conformal` and `pops` are written
  against the generator and a "quantity" label, not against energies and forces.
- **`calc` consumes `uq` through an interface.** The UQ branches of `calc/point.py` (the ARD
  posterior, committee forces, the support check) become calls on an object `uq` provides, loaded
  from the model file. `calc/gp.py` keeps its `fit.gp` imports (same layer). The
  `fit/pipeline/export.py` → `calc.gp` edge is removed by moving `FittedGP` beside its loader.

### 3. `eval`

- `eval/product.py`: the A/AA product, its layouts, `estimate_a_bytes` and `pool_sparse`, taken
  out of `edge_model.py` and `model.py`. `EdgeSiteModel` uses it; behaviour and goldens unchanged.
- `ham/features.py` builds on `eval.product` and `basis.coupling`; the carrier `ACEModel` and
  `basis.model.build_model` leave `ham`.
- Back-edges: `compact` moves from `calc/skin.py` to `eval/dense.py`; `ace_ctilde_tree` moves
  beside the PACE reader or the `.yace` writer takes it as an argument; `cubic_bspline_coefs`
  moves to `eval/radial.py`. `eval/__init__`'s lazy `ACECalculator` goes (the top-level
  `ace_jax.ACECalculator` already covers it).

### 4. Operators

`ace_jax.operators`: everything about an equivariant one- and two-centre block model that does
not know it is a Hamiltonian: `cg`, `features`, `bond`, `onsite`, `offsite`, `offsite_fast`,
`orbitals` (shell lists define the block structure of any atomic-orbital operator) and the block
fit. Operator channels stay a leading coefficient axis.

`ace_jax.ham`: what is specific to H and S: `deeph`, `cache`, the Bloch sum and bands in
`operator.py`, channel names, the Quoll ingestion, and later the spectral losses.

A second operator (density matrix, friction tensor) is then a sibling of `ham`, not a fork of it.
The `ham` extra (`h5py`) is unchanged.

### 5. Bench

| Role | Today | Decision |
|---|---|---|
| Test drivers | `acegp_cantor/run.py`, `learn_radial/{run, rmse_npz}.py` | Move under `tests/drivers/` (or call `aj fit` where it is equivalent). They are run by the test suite and generate the pipeline goldens; they are not benchmarks. |
| Published benchmark | `scaling/`, `gpu_memory/`, `batch/`, `aa_dag/`, `perf/results/` | Stay in `bench/`. They back `docs/user/benchmarks.md`, are tested, and their results are cited by library docstrings and read by `tests/test_calc_layout.py`. |
| Cross-code comparison | `accuracy_cost/`, `si_qoi/` | Stay for now; recommended to move to their own repository after step 4 (separate decision, below). |
| Paper and study analyses | `defect_uq/`, `fp32/`, `learn_radial/` (the rest), `perf/` one-off scripts | Finished studies are archived by tag, as `bench/ARCHIVED.md` already does; analyses for a paper live in that paper's repository pinned to a release. What `defect_uq` needs from `_proj`, `_pooled_q` and the like becomes public `uq` API first. |

The cross-code comparison is the one part with an outside audience: it carries four comparator
environments with their own `pyproject.toml` and lock files, 143 result files and 43 batch
scripts, uses `ace_jax` mostly through calculators and loaders, and other codes' authors are being
asked to check the protocol and send changes. A neutral repository that depends on a released
`ace-jax` suits that better than a directory in one competitor's source tree. It should not move
before the refactor, because `tests/test_bench_accuracy_cost.py` (1065 lines) and the figures on
the docs site move with it.

## Order of work

Two streams run in parallel. Each step is one squash-merged PR.

**Stream A, on `main` (this refactor):**

1. `tests/test_import_layers.py` with the allow-list and the bench import check. No source change.
2. `fit/` moves (§2 table) with stubs; bench, tests and docs imports updated in the same PR.
3. `eval/product.py` and the back-edge fixes (§3). `eval.model` keeps re-exporting `pool_sparse`.
4. Generic rows; `uq` written against them. Goldens on lestrade.
5. `calc` → `uq` interface.

**Stream B, `feat/hamiltonians` (feature work, unchanged in scope except):**

- It does **not** do the framework refactor of the Hamiltonian design §2 (splitting
  `EdgeSiteModel`, generic rows). Those belong to stream A, steps 3 and 4. Until it merges, `ham`
  keeps the carrier `ACEModel` and its own QR.
- It merges `main` in after each stream A PR. Its tests are the check that the stubs hold.

**Why they do not collide.** Outside `ham/`, the branch edits only `basis/coupling.py`,
`basis/spec.py`, `basis/model.py`, `eval/nlist.py`, `pyproject.toml` and `uv.lock`. Stream A
edits none of the first four: they are frozen on `main` until the branch merges. `ham` imports
no `fit` module, and the two `eval.model` names it uses stay importable.

**After `feat/hamiltonians` merges** (its `ham → eval.model` and `ham → basis.model` edges go on
the allow-list in the merge PR):

6. `ham` onto the new `main`: drop the carrier for `eval.product`; the block fit moves onto
   `fit.linear` and `uq`; `operators`/`ham` split (§4).
7. Bench: test drivers to `tests/drivers/`; archive finished studies; update `CLAUDE.md`,
   `bench/ARCHIVED.md` and the docstring citations.
8. Decide on the cross-code repository.

Steps 1 to 3 are moves. Steps 4 and 5 are the real work; if the branch merges before they are
done, step 6 splits in two: the carrier and the package split first, the fit after step 4.

## Risks

- Open branches: the remote has about 110 heads. Renames in `fit/` will conflict with any that
  touch the moved files. Delete merged branches before step 2 and land or rebase the live ones.
- A stub that imports eagerly can create a cycle; stubs use module `__getattr__`.
- The bench import check sees names, not signatures. Scripts outside the tested set can still
  break on a changed argument; that is accepted for study code.

## Left to the builder

Names inside `fit/linear`, `uq` and `operators`; the placement of `hypers`, `paramset` and
`rowcheck`; the shape of the object `uq` hands to `calc`; whether `tests/drivers/` or `aj fit`
replaces each bench driver; the allow-list format.

## Calls made here that the owner has not confirmed

- `uq` as a top-level package rather than `fit/uq`.
- `operators` as the generic package with `ham` as a thin layer; not checked against a concrete
  friction or density-matrix model.
- `density` and `feature` in `fit/gp/`, from their names and neighbours rather than a full read.
- Steps 1 to 3 on `main` before the Hamiltonian branch merges, rather than on that branch.
- Moving `accuracy_cost/` and `si_qoi/` to a separate repository later.
- Archiving `defect_uq/` once its paper analyses are reproducible from a pinned release.
