# ace-jax

Fit and evaluate **Atomic Cluster Expansion (ACE)** interatomic potentials in
pure **Python/JAX**: build a basis, fit it and run it, everything from
`pip install ace-jax`.

- **Evaluate** exported ACE models and pacemaker **PACE `.yace`** potentials
  (energy / forces / stress) and site descriptors; `.yace` files can be written back.
- **Fast forces** on CPU and GPU: a shared edge-vector core with sparse and dense
  (per-node outer-product) layouts picked automatically, jitted per padded shape.
- **Fit** the linear (M=0) model with stable solvers (Cholesky / QR / streaming
  QR / LSQR), reproducing ACEfit to machine precision.
- **Hybrid GP** fits with a calibrated uncertainty ladder (MAP → Laplace →
  Pathfinder / VI → NUTS), multi-start L-BFGS MAP, frozen species
  coregionalization, and predictive `energy_std` / `forces_std`; **POPS**
  misspecification uncertainty for the linear model. Fitted models are saved
  (`model.npz` / `gp_model.npz`).
- **ASE calculators** (`ACECalculator`, `GPCalculator`) and **LAMMPS** deployment
  through [lammps-jax](https://github.com/abhijeetgangan/lammps-jax)
  (`ace_jax.export.lammps.export_lammps`).
- **Build the basis and fit it in one step**: `aj fit --order 3 --max-degree 10
  --train train.xyz --out fit/` builds the symmetry-adapted ACE basis (coupling
  from [EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl),
  precompiled into the `ace-jax-coupling` wheel) and fits it; the `fit.yaml` it
  writes reproduces the run. Species-embedded bases and the smoothness prior
  included.
- Research: learned radial basis by variable projection (`bench/learn_radial/`).

## Install

```bash
pip install ace-jax             # core: evaluate, fit, build new bases, ASE calculator
pip install ace-jax[gp]         # + `ace-jax fit` pipeline: GP/UQ hyperparameter ladder
pip install ace-jax[cuda]       # + CUDA 12 JAX
pip install ace-jax[fast-neighbours]  # + matscipy-neighbours (C++ source build; ASE's list is the fallback)
```

Building a new basis (`aj basis`, or `aj fit` with `--order/--max-degree`) is
available on Linux x86_64/aarch64 and macOS arm64; elsewhere, fit and evaluate
from an existing `.npz` basis (`--model`).

> **Pre-release:** the `ace-jax-coupling` wheel that ace-jax depends on is not
> on PyPI yet, so `pip install ace-jax` does not resolve outside this
> repository. Build that wheel from `coupling/` (see CLAUDE.md "Setup" and
> `docs/coupling-etshim-spec.md`) and install it next to ace-jax.

Training and evaluation data (extxyz) are read with libAtoms
[`extxyz`](https://github.com/libAtoms/extxyz), a core dependency: labels come
back under the names they were written with, including `energy` / `forces`.
LAMMPS export needs [lammps-jax](https://github.com/abhijeetgangan/lammps-jax),
which is not on PyPI: install it from a clone (`pip install -e <lammps-jax>`).

A model **definition** (basis + radials) is an `.npz` that ace-jax consumes.
There are three ways to get one:

- **as part of a fit**: `aj fit --order 3 --max-degree 10 ...` builds the basis
  in memory; the fitted `model.npz` contains it;
- **on its own**: `aj basis ... --out si.npz`, to save, share, inspect or fit it
  several times (`aj fit --model si.npz`), including species-embedded bases;
- **exported from ACEpotentials** (`julia/export_model.jl`), still fully
  supported.

The symmetry-adapted A→B coefficients come from EquivariantTensors, shipped
precompiled in the `ace-jax-coupling` wheel (nothing else to install, nothing
downloaded at runtime); radials, pair basis and embedding are built in Python. A
per-shape coupling cache means a basis shape is computed only once.

Models come as **unfitted definitions** (coefficients to be fit here) or
**fitted potentials** (ready to evaluate). Large model files are not committed
to the repository.

## Quickstart

```python
import ace_jax as aj
model, meta, z = aj.load("si_fitted.npz")      # model, meta dict, raw npz
from ace_jax import ACECalculator
atoms.calc = ACECalculator("si_fitted.npz")
atoms.get_potential_energy(); atoms.get_forces()
```

From labelled data to a fitted model in one command, and back from one file:

```bash
aj fit --order 3 --max-degree 10 --train train.xyz --test test.xyz --out fit/   # builds the basis, fits it
aj fit --config fit/fit.yaml --out fit2/                                        # reproduce (or edit) the run
```

CLI: `ace-jax` (short alias `aj`) with subcommands `basis`, `fit` and `eval`;
see [Command line](#command-line-ace-jax--aj) below. An agent-oriented guide lives in
[`skills/ace-jax/SKILL.md`](skills/ace-jax/SKILL.md).

### PACE (pacemaker) potentials

`.yace` files load directly and evaluate in JAX (ASE calculator, lammps-jax):

```python
atoms.calc = ACECalculator("model.yace")
```

Supported: ChebExpCos / ChebPow / ChebLinear / SBessel radials, FinnisSinclair
and FinnisSinclairShiftedScaled embeddings, `density` / `distance` / `zbl` inner
cutoffs. `aj.load("model.yace")` returns `(PACEModel, meta, spec)`, and
`write_yace(model, spec, path)` (`ace_jax.eval`) writes a (possibly modified)
model back. `.yace` models are for evaluation and export: `aj fit` needs an
`.npz` ACE model. Checked against the ML-PACE C++, python-ace and LAMMPS; see
`docs/pace-yace-spec.md` and `docs/pace-yace-results.md`.

### Speed options and LAMMPS

`ACECalculator(path, layout="auto", edge_a_kind="auto", skin=1.0, lean=True, spline_tol="auto", spline_intervals=None)`:

- **Lean evaluation form** (ACE `.npz` models). With `lean=True` (the
  default), energies, forces and stress are evaluated with
  `ace_jax.eval.lean(model)`, which is exact to roundoff. It drops the radial
  columns and harmonics the basis never reads, folds the pair weights into the
  pair radial, and pools the dense A per l-block. Forces are 1.1–3.3× faster on
  an A100 (`docs/ace-vs-pace-gap.md` §8).
  - `calc.eval_model` is the lean form, and `calc.model` stays the model as
    given. Descriptors use `calc.model`.
  - A lean model is energy-only: never edit or fit it. Edit the full model and
    re-apply `lean`.
  - **Learned radials are splined.** A learned tensor radial (a
    `bench/learn_radial` model.npz, marked `radial_learned` in its meta) is
    first converted to a spline at 1e-10 per radial (`ace_jax.eval.to_spline`).
    That recovers the splined models' lean speed, but it is not roundoff: the
    lean energies agree with the full model to up to ~1e-9 relative and forces
    to up to ~2.3e-8 of the largest force on the benchmark models.
    - ACEpotentials `ace_model` exports and built bases (`aj basis`, `aj fit`) are analytic but not
      learned, so they stay exact. `spline_tol=1e-10` opts them in.
    - `spline_tol=None` never splines.
    - Old learned-radial files written before the flag existed load as not learned: mark one with `ace_jax.basis.export.mark_radial_learned("model.npz")`, or pass `spline_tol=1e-10`.
    - `calc.splined` (and `calc.last_timing["spline_tol"]`) says what was
      splined, None when nothing was.
    - The spline is cached on the radial's content, so swapping in new readout
      weights (`calc.model = ...`) does not redo it.
    - Its interval count is rounded up to a quarter-octave bucket, so a swapped
      radial usually reuses the compiled step. `spline_intervals` pins it.
    - See `docs/learned-radial-splining.md`.

- **Layout.** `"auto"` picks the dense layout (A per node by a batched outer
  product, several times faster forces on GPU) when the neighbour padding is
  efficient, i.e. edges / (atoms × max neighbours) ≥ `MIN_DENSE_FILL` (0.5),
  else the sparse edge list. Memory is not a criterion: the dense model runs in
  blocks of 16,384 rows (`CHUNK_NODES`), so its peak is bounded per block.
  Sparse edge lists are padded to power-of-two buckets so MD reuses the
  compiled function.
- **Skin (Verlet) neighbour list,** dense layout only. The list is built for
  cutoff + `skin` (Å) and reused across MD-like calls, one compiled step each.
  It is rebuilt automatically when an atom has moved more than skin / 2 since
  the build, when the cell, pbc, species or atom count change, or when a row
  outgrows its capacity. `calc.last_timing["rebuilds"]` counts the calls that
  built a list, and `last_timing["nlist_s"]` is 0 on a reuse. For one-shot
  evaluation of unrelated structures (a dataset, a screening loop) pass
  `skin=0`, which builds a list for the cutoff alone on every call. Setting
  `calc.model` or `calc.skin` drops the current list.

To run in LAMMPS (`pair_style jax/kk`, GPU):

```python
from ace_jax.export.lammps import export_lammps, neighbour_capacity
model, meta, _ = aj.load("model.npz")                    # or a .yace
cap = neighbour_capacity(atoms, meta["rcut"], skin=1.0)  # the LAMMPS `neighbor` skin
export_lammps(model, meta, "bundle", max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
              k_dense=cap["k_dense"], max_neighbors=cap["max_neighbors"],
              max_owned=cap["max_owned"],                # dense rows: owned atoms only
              type_elements=[14, 32])                    # Z of LAMMPS types 1, 2, ...
```

`layout="auto"` exports a dense-family layout when `k_dense` (max neighbours
per atom) is given and one block's `estimate_a_bytes` fits the device budget,
else sparse. The dense-family layout is `"matrix"` only when `max_neighbors`
is passed, the installed lammps-jax has its neighbour-matrix input (4a7f4fb
and later), and the block plus the matrix's unblocked pre-processing
(`matrix_prep_bytes`) fits. Otherwise it is `"dense"`, so a call without
`max_neighbors` keeps the packed dense bundle.
`"matrix"` reads the LAMMPS neighbour list rows directly (copied only when
LAMMPS rebuilds its list; the model drops the skin pairs), so there is no
per-step packing; `"dense"` packs lammps-jax's edge buffer into slots every
step. `layout="matrix"` requires `max_neighbors`, the list slots per row. The
list holds the rcut + skin pairs, so a `k_dense` sized for pairs within rcut
is too small for it. `max_owned` bounds the rows to the owned atoms (LAMMPS
numbers them first), so ghost rows cost nothing; it is recorded as
`ace_jax.owned_rows`. Too many neighbours within the cutoff (past `k_dense`)
gives NaN, never a silent truncation. So does an owned atom past `max_owned`
in a dense bundle. For a matrix bundle, owned atoms past `max_owned` or a
list row wider than `max_neighbors` abort the run in LAMMPS.

A LAMMPS lammps-jax plugin older than the Python package rejects a matrix
bundle, with a message that suggests re-exporting. Re-exporting does not
help: rebuild the plugin at the Python package's commit, or export
`layout="dense"`. The bundle records the exporting lammps-jax as
`ace_jax.lammps_jax`.

`neighbour_capacity(atoms, rcut, skin=1.0, slots="skin", margin=8,
list_headroom=0.5)` sizes all of these for a structure. The ghost shell is
sized on the face spacings, so triclinic cells are covered. It gives the
matrix list 50% headroom over the rcut + skin coordination, and the model
slots are compacted from the list. The headroom is there because that count
grows fastest when a structure compresses. The 0.5 is calibrated on a single
observed overflow.
The default `slots="skin"` sizes `k_dense` for the rcut + skin coordination,
which is safe between list rebuilds.
`slots="cutoff"` sizes it for rcut pairs only (the matrix list keeps its
rcut + skin width, and the in-cutoff pairs are compacted into the tighter
model slots): 1.2-1.4x faster than the default on Cantor (A100), and safe for stable MD with a
fitted model, whose coordination stays within `margin` of the start. A
structure that compresses (a random-weight model, a collapse) overflows it,
and the step is NaN. Above 32,768 rows
(`BUNDLE_BLOCK_ROWS`) the dense bundle evaluates in blocks, which bounds
memory at large N. `lean=True` (the default) exports the lean form of an ACE
model, as the calculator does, and records it as `ace_jax.lean`. `spline_tol`
and `spline_intervals` work as for the calculator. What was actually splined is
recorded as `ace_jax.spline_tol` and `ace_jax.spline_intervals`, both None when
nothing was. The `"auto"` layout is still sized on the full model.

## Performance

[`docs/benchmarks.md`](docs/benchmarks.md) has throughput-vs-size, model-size,
memory and precision scaling for ace-jax (standalone and in LAMMPS via
lammps-jax), ML-PACE and MACE, on SiGe and Cantor, on CPU (moriarty) and GPU
(RTX A4500, A100), each run behind a parity gate. The harness is in
`bench/scaling/`. The ace-jax production model (linear + species + density
embedding) and ACEpotentials.jl rows will be added in a follow-up.

## The coupling table (EquivariantTensors)

`ace_jax.basis` builds an ACE basis's symmetry-adapted coupling coefficients in
Python: `basis.spec.build_spec(...)` enumerates the admissible `(n,l)` many-body
specification (total-degree / `wL`), and `basis.coupling.couple(...)` returns
the A→B symmetrisation matrix computed by EquivariantTensors' own construction,
which ships precompiled in the `ace-jax-coupling` wheel (`coupling/`).

The coupling matches EquivariantTensors (main, the fork rev in
`ace_jax_coupling.build_info()`) exactly: bit-identical on macOS arm64 and Linux
aarch64; on x86_64 identical structure with values within a few ulp (BLAS/sparse
LU rounding in degenerate blocks). Against ACEpotentials 0.10.1 exports (which pin
ET 0.4.3) it has the same `nnll` blocks and the same per-block row spaces for
orders 2–4, including the degenerate blocks, but each B function is normalised
differently (row-norm ratio 0.225–1): the same function space, so fitted
coefficients are not interchangeable with 0.10.1 exports.
See `docs/coupling-etshim-spec.md` for the design and `tests/test_coupling_parity.py`
for usage.

## Building a basis (CLI and Python)

`aj fit --order 3 --max-degree 10 ...` and
`aj basis --elements Si --order 3 --max-degree 10 --out si.npz` both call
`build_basis(BasisSpec(...))` (`ace_jax.basis.model`); in Python a `Basis` can be
inspected or modified and handed straight to the fit
(`FitConfig(model=basis_or_spec)`, no file). It builds a complete frozen model in memory:
coupling via the shim, seeded radial/pair init, zero readout, and the algebraic
smoothness prior — then packages it in the export format, so the saved file
evaluates with the plain eval path. `save_npz` derives the branch-selector meta
keys from the tree being saved, which is what lets fitted coefficients be
injected via `dataclasses.replace` and round-trip through the loader. The
bridge test verifies the whole chain against the committed Si fixture: `A2B`
up to that per-row scale, then (with the fixture's coefficients rescaled)
energies, forces, stress and descriptors to float noise.
See `docs/basis.md`.

### Embedded (species-compressed) models: `ace-jax basis --embedding`

    ace-jax basis --elements Cr,Mn,Fe,Co,Ni --order 3 --max-degree 10 \
        --embedding mace_embedding.json --out cantor_embed.npz          # lossless widths
    ace-jax basis ... --d-max 16                                    # capped widths
    ace-jax basis ... --embedding identity                          # identity (one-hot) element table

builds the frozen-element-embedding model (`basis.model.build_embedding_model`,
the ace1-compatible `ace_embedding_model`), parity-tested against ACEpotentials'
exports. On `aj fit` the same flag is `--basis-embedding` (`--embedding` there
is the GP's frozen species table).

## Command line (`ace-jax` / `aj`)

`aj` is the same entry point as `ace-jax`. Three subcommands cover the workflow:
**fit** fits labelled data (building the basis itself from `--order/--max-degree`,
or taking one with `--model`), **basis** saves a basis on its own, and **eval**
evaluates a fitted model. Every fit writes `fit.yaml`, the whole resolved run
(`aj fit --config fit.yaml` reproduces it; command-line flags override the file).
`si.npz` below is any model definition, e.g. from
`aj basis --elements Si --order 3 --max-degree 10 --out si.npz`; replacing
`--model si.npz` with `--order 3 --max-degree 10` builds it inside the fit. The examples
below were checked on the Si test fixture (`si_fitted.npz`, with `si_tiny_train.xyz`
split into train/test/ood files). The `--*-key` flags name the extxyz fields that
hold the labels. Data is read with libAtoms `extxyz`, so any label name works as
written, including `energy`/`forces`.

```bash
K="--energy-key dft_energy --force-key dft_force --virial-key dft_virial"

# linear ACE (M = 0): Bayesian linear regression, fitted model -> out_linear/model.npz
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --rungs map --r0 2.35 --out out_linear
aj eval --model out_linear/model.npz --data test.xyz $K --forces

# hybrid ACE + GP: 6 inducing sites per species, best of 3 L-BFGS MAP starts
# -> out_gp/gp_model.npz  (--rungs map,laplace adds hyperparameter draws)
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 6 --opt lbfgs --map-restarts 3 --map-steps 40 \
    --rungs map --r0 2.35 --out out_gp
aj eval --model out_gp/gp_model.npz --data test.xyz $K --forces --out pred.csv   # + energy_std

# large data: PCA density features, design rows cached in host RAM, an OOD set
aj fit --model si.npz --train train.xyz --test test.xyz --ood ood.xyz $K \
    --m-per-species 6 --density pca --pca-d 8 --lml host-cache --opt lbfgs \
    --map-steps 40 --rungs map --r0 2.35 --out out_hc

# POPS misspecification uncertainty on the linear model
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --uq pops --opt lbfgs --rungs map --r0 2.35 --out out_pops

# calibrated per-atom force uncertainty: ARD posterior (linear model)
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --uq ard --opt lbfgs --r0 2.35 --out out_ard
aj eval --model out_ard/model.npz --posterior out_ard/posterior.npz --data big.xyz $K \
    --forces --per-atom atoms_std.xyz          # per-atom forces_std, e.g. to colour a crack tip

# one file split by a seeded permutation, E0 by least squares
aj fit --model si.npz --data all.xyz --ntrain 40 --ntest 10 --e0 lsq $K \
    --m-per-species 0 --rungs map --r0 2.35 --out out_split
```

`--uq ard` fits prior scales per body order and the noise scales by evidence (joint type-II ML).
The default `--ard-variance sandwich` serves the configuration-clustered sandwich variance,
σ² = λ²·φA⁻¹MA⁻¹φᵀ, the misspecification-robust covariance, with λ from the train hold-out (fitted
the same way as κ, but with each held-out atom's own training-configuration cluster left out of M: a
new configuration has no such term; `ard.json` also reports `lam_incl_own`, the λ with it). On the bench365 prototype it ranked local errors better than the tempered
posterior (Spearman ρ 0.26–0.37 against 0.15–0.26) at the same calibration and OOD detection.
`--ard-variance kappa` keeps the single-temperature posterior variance κ²φA⁻¹φᵀ instead. Only
the force variance is calibrated (λ or κ; `ard.json` `tempered_quantities: ["F"]`); energy and virial
variances are the uncalibrated posterior ones. The
prototype of this method (`bench/defect_uq`, PR #12) held rms-z 0.91–1.02 on held-out Cantor
defect combinations; the acceptance run of this implementation is pending. `--uq ard` changes the
mean as well as the uncertainty: `model.npz` holds the ARD posterior mean, not the BLR/MAP mean.
`posterior.npz` stores the float32 posterior factor, ~0.9 GB at L = 15k, and (for the default
sandwich variance) an additional (L, n_train_configs) float32 factor. `--ard-mode sequential` is
the low-memory fallback. The calculator's `forces_std` holds the whole cell's force design rows,
about N·3·L·8 bytes (N atoms, L columns; 7 GB for 100k atoms at L = 3k), so size cells to fit them.

`aj fit` and the research driver `bench/acegp_cantor/run.py` share one pipeline
(`ace_jax.fit.pipeline`: `FitConfig`, `load_fit_data`, `fit`, `write_outputs`,
`save_model`). Common options:

- data: `--train/--test` files, or `--data` split with `--ntrain/--ntest/--test-start`;
  `--ood` for an extra test set; `--weights` takes an ACEfit weights dict or a list of
  weight factors; `--r0` is the typical nearest-neighbour distance (required with
  `--model`; when the basis is built it defaults to the basis's mean bond length)
- model: `--m-per-species 0` is the linear model, `> 0` the hybrid GP (default 500)
- GP features: `--density none|pair|pca` (`--pca-d`), `--embedding` (frozen species
  coregionalization)
- likelihood: `--lml host-cache` caches the linear design rows in host RAM (GP arm,
  pair/pca features, L-BFGS, MAP only)
- MAP: `--opt adam|lbfgs` (default adam, 500 steps; L-BFGS is much faster on small
  data), `--map-restarts N` (best of N L-BFGS starts; the joint LML is multimodal)
- UQ: `--rungs map,laplace,pathfinder,vi,nuts` (`--laplace svi|fd`), or `--uq pops`/`--uq ard`
  on the linear model. The default is `--rungs map`. The other rungs add
  hyperparameter draws and cost far more than the MAP: the Laplace rung takes a
  Hessian through the whole LML, which had not finished compiling after 30 min on a
  laptop CPU for 40 Si configs

Outputs in `--out`: `metrics.csv` (test RMSE, MAE, CRPS, coverage, rms z per rung and
quantity), `metrics_ood.csv`, `theta_map.json`, `draws_<rung>.npy`, `config.json`,
and the **fitted model**:

- linear: `model.npz`, an ordinary ACE model file (`aj.load`, `ACECalculator`,
  `aj eval`); `--uq ard` also writes `posterior.npz` and `ard.json` (see above);
- GP: `gp_model.npz`, self-contained, loaded by
  `GPCalculator.from_file("gp_model.npz")` (energy, forces, stress, `energy_std`,
  `forces_std`) or `aj eval`. It stores one (Dt, Dt) posterior factor per
  hyperparameter draw (Dt = basis size + M): `--model-draws N` keeps N draws of the
  last rung (default 1, the MAP), and `--no-save-model` skips the file.

A fit with `--baseline` saves no model, because its pair baseline is added outside the model.

## Running the tests

```bash
uv run pytest                        # whole fast suite, 6 parallel workers (~2-4 min)
uv run pytest tests/test_efv.py      # a targeted run stays single-process
uv run pytest -m slow                # opt-in: real-model MCMC ladder, bit-exact driver goldens
```

A bare `pytest` uses pytest-xdist when it is installed (`ACEJAX_TEST_WORKERS`
sets the worker count; `-n 0` forces a single process, as CI does). The `slow`
marker is excluded by default. Tests needing an optional package (ace-jax-coupling,
lammps-jax, python-ace, sphericart, matscipy-neighbours, psutil) skip without
it; for the lammps-jax tests put a clone on the path
(`PYTHONPATH=<lammps-jax>/python`). See `CLAUDE.md` for the test switches.

## Linting and pre-commit

```bash
uv run ruff check                    # lint (config in pyproject.toml [tool.ruff])
uv run pre-commit install            # once per clone: ruff + whitespace/YAML/large-file hooks on commit
uv run pre-commit run --all-files    # what the `lint` CI job runs
```

No formatter is enforced: the code keeps its dense one-line style, and the ruff
rules that fight it (semicolon statements, short math names, import sorting,
line length) are off. Bump the ruff pin in the dev group and the `rev` in
`.pre-commit-config.yaml` together.

## Reference parity (maintainers / CI only)

The everyday test suite is pip-only, run against committed npz fixtures. Path-gated CI jobs regenerate the references and guard the seams to
the reference codes:

- **ACEfit parity** (`.github/workflows/julia-parity.yml`) — the design-matrix rows match ACEfit to 1e-8 and the linear
  solve matches ACEfit's `solve(QR)` to 1e-9, checked against the committed
  fixtures (`julia/export_model.jl`, `julia/acefit_qr_reference.jl`).
- **coupling-wheels** — builds the `ace-jax-coupling` wheels (manylinux_2_28
  x86_64/aarch64, macOS arm64, Windows x64), tests them in clean environments
  (matching upstream EquivariantTensors: exact indices, values within 4 ulp),
  and runs the coupling parity
  against the ACEpotentials references (`julia/coupling_reference.jl`).
- **prior-parity** — the smoothness prior (`basis/prior.py`) matches
  ACEpotentials' `algebraic_smoothness_prior` bit-for-bit
  (`julia/smoothness_reference.jl`), and the committed fixtures match a fresh run.
- **pace-parity** — the PACE path against the ML-PACE C++ (pinned
  `lammps-user-pace`) and python-ace, built from source in their own
  environments; see `pace_ref/README.md`.
