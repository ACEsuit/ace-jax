# ace-jax

Fit and evaluate **Atomic Cluster Expansion (ACE)** interatomic potentials in
pure **Python/JAX** — no Julia needed to fit or run.

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
- **Author** whole models from Python — the symmetry-adapted coupling via
  [EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl),
  compiled ahead of time so no Julia is needed at all (optional `authoring`
  extra) — including species-embedded models and the smoothness prior.
- Research: learned radial basis by variable projection (`bench/learn_radial/`).

## Install

```bash
pip install ace-jax             # core: evaluate + linear fit + ASE calculator
pip install ace-jax[gp]         # + `ace-jax fit` pipeline: GP/UQ hyperparameter ladder
pip install ace-jax[authoring]  # + Python basis coupling (compiled EquivariantTensors; Linux x86_64/aarch64, macOS arm64)
pip install ace-jax[cuda]       # + CUDA 12 JAX
pip install ace-jax[fast-neighbours]  # + matscipy-neighbours (C++ source build; ASE's list is the fallback)
```

Training and evaluation data (extxyz) are read with libAtoms
[`extxyz`](https://github.com/libAtoms/extxyz), a core dependency: labels come
back under the names they were written with, including `energy` / `forces`.
LAMMPS export needs [lammps-jax](https://github.com/abhijeetgangan/lammps-jax),
which is not on PyPI: install it from a clone (`pip install -e <lammps-jax>`).

No Julia is required to **use, fit, or evaluate** a model. A model **definition**
(basis + splined radials) is exported to an `.npz` that ace-jax consumes. The
model-authoring seam has three paths:

- **use / fit / evaluate an existing model** → only the `.npz` (no Julia);
- **author a new model in Python** → the `authoring` extra builds the whole
  model (`ace-jax construct`, including species-embedded models): the `(n,l)`
  specification and the symmetry-adapted A→B coefficients come from
  EquivariantTensors, shipped as the `ace-jax-coupling` platform wheel (a
  `juliac --trim` compiled library: no Julia install, nothing downloaded at
  runtime, no ACEpotentials stack); radials, pair basis and embedding are built
  in Python. A per-shape coupling cache means the library runs only for a basis
  shape not seen before;
- **export a whole new basis from Julia** → the original path
  (`julia/export_model.jl`), still fully supported.

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

CLI: `ace-jax` (short alias `aj`) with subcommands `construct`, `fit` and `eval`;
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

`ACECalculator(path, layout="auto", edge_a_kind="auto", skin=1.0, lean=True)`:

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
from ace_jax.export.lammps import export_lammps
model, meta, _ = aj.load("model.npz")                    # or a .yace
export_lammps(model, meta, "bundle", max_atoms=4096, max_edges=200_000,
              k_dense=64, max_owned=2048,                # dense rows: owned atoms only
              type_elements=[14, 32])                    # Z of LAMMPS types 1, 2, ...
```

`layout="auto"` exports the dense layout when `k_dense` (max neighbours per
atom) is given and one block's `estimate_a_bytes` fits the device budget, else
sparse. `max_owned` bounds the dense rows to the owned atoms (LAMMPS numbers
them first), so ghost rows cost nothing. It is recorded as
`ace_jax.owned_rows` in the bundle, and an atom past it, or past `k_dense`
neighbours, gives NaN, never a silent truncation. Above 32,768 rows
(`BUNDLE_BLOCK_ROWS`) the dense bundle evaluates in blocks, which bounds
memory at large N. `lean=True` (the default) exports the lean form of an ACE
model, as the calculator does, and records it as `ace_jax.lean`. The `"auto"`
layout is still sized on the full model.

## Performance

[`docs/benchmarks.md`](docs/benchmarks.md) has throughput-vs-size, model-size,
memory and precision scaling for ace-jax (standalone and in LAMMPS via
lammps-jax), ML-PACE and MACE, on SiGe and Cantor, on CPU (moriarty) and GPU
(RTX A4500, A100), each run behind a parity gate. The harness is in
`bench/scaling/`. The ace-jax production model (linear + species + density
embedding) and ACEpotentials.jl rows will be added in a follow-up.

## Authoring the coupling table in Python (EquivariantTensors)

With the `authoring` extra, `ace_jax.construct` builds an ACE basis's
symmetry-adapted coupling coefficients without a Julia export step:
`construct.spec.build_spec(...)` enumerates the admissible `(n,l)` many-body
specification (total-degree / `wL`), and `construct.coupling.couple(...)` calls
EquivariantTensors to produce the A→B symmetrisation matrix — through
`ace-jax-coupling`, EquivariantTensors' coupling construction compiled with
`juliac --trim=safe` into a self-contained library (`coupling/`), so you never
install or manage Julia yourself.

The coupling is **bit-identical** to EquivariantTensors (main, the fork rev in
`ace_jax_coupling.build_info()`). Against ACEpotentials 0.10.1 exports (which pin
ET 0.4.3) it has the same `nnll` blocks and the same per-block row spaces for
orders 2–4, including the degenerate blocks, but each B function is normalised
differently (row-norm ratio 0.225–1): the same function space, so fitted
coefficients are not interchangeable with 0.10.1 exports.
See `docs/coupling-etshim-spec.md` for the design and `tests/test_coupling_parity.py`
for usage.

## Authoring a whole model in Python (Tier 1)

`ace-jax construct --elements Si --order 3 --max-degree 10 --out si.npz`
(`construct.model.build_model`) authors a complete frozen model in memory:
coupling via the shim, seeded radial/pair init, zero readout, and the algebraic
smoothness prior — then packages it in the export format, so the saved file
evaluates with the plain eval path. `save_npz` derives the branch-selector meta
keys from the tree being saved, which is what lets fitted coefficients be
injected via `dataclasses.replace` and round-trip through the loader. The
bridge test verifies the whole chain against the committed Si fixture: `A2B`
up to that per-row scale, then (with the fixture's coefficients rescaled)
energies, forces, stress and descriptors to float noise.
See `docs/python-authoring.md`.

### Embedded (species-compressed) models: `ace-jax construct --embedding`

    ace-jax construct --elements Cr,Mn,Fe,Co,Ni --order 3 --max-degree 10 \
        --embedding mace_embedding.json --out cantor_embed.npz          # lossless widths
    ace-jax construct ... --d-max 16                                    # capped widths
    ace-jax construct ... --embedding identity                          # identity (one-hot) element table

builds the frozen-element-embedding model (`construct.model.build_embedding_model`,
the ace1-compatible `ace_embedding_model`) without Julia, parity-tested against
ACEpotentials' exports.

## Command line (`ace-jax` / `aj`)

`aj` is the same entry point as `ace-jax`. Three subcommands cover the workflow:
**construct** a model definition, **fit** it to labelled data, and **eval** the
fitted model. `si.npz` is any model definition, e.g. from
`aj construct --elements Si --order 3 --max-degree 10 --out si.npz`. The examples
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

# one file split by a seeded permutation, E0 by least squares
aj fit --model si.npz --data all.xyz --ntrain 40 --ntest 10 --e0 lsq $K \
    --m-per-species 0 --rungs map --r0 2.35 --out out_split
```

`aj fit` and the research driver `bench/acegp_cantor/run.py` share one pipeline
(`ace_jax.fit.pipeline`: `FitConfig`, `load_fit_data`, `fit`, `write_outputs`,
`save_model`). Common options:

- data: `--train/--test` files, or `--data` split with `--ntrain/--ntest/--test-start`;
  `--ood` for an extra test set; `--weights` takes an ACEfit weights dict or a list of
  weight factors; `--r0` (required) is the typical nearest-neighbour distance
- model: `--m-per-species 0` is the linear model, `> 0` the hybrid GP (default 500)
- GP features: `--density none|pair|pca` (`--pca-d`), `--embedding` (frozen species
  coregionalization)
- likelihood: `--lml host-cache` caches the linear design rows in host RAM (GP arm,
  pair/pca features, L-BFGS, MAP only)
- MAP: `--opt adam|lbfgs` (default adam, 500 steps; L-BFGS is much faster on small
  data), `--map-restarts N` (best of N L-BFGS starts; the joint LML is multimodal)
- UQ: `--rungs map,laplace,pathfinder,vi,nuts` (`--laplace svi|fd`), or `--uq pops` on
  the linear model. The default is `--rungs map`. The other rungs add
  hyperparameter draws and cost far more than the MAP: the Laplace rung takes a
  Hessian through the whole LML, which had not finished compiling after 30 min on a
  laptop CPU for 40 Si configs

Outputs in `--out`: `metrics.csv` (test RMSE, MAE, CRPS, coverage, rms z per rung and
quantity), `metrics_ood.csv`, `theta_map.json`, `draws_<rung>.npy`, `config.json`,
and the **fitted model**:

- linear: `model.npz`, an ordinary ACE model file (`aj.load`, `ACECalculator`,
  `aj eval`);
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

## Julia parity (maintainers / CI only)

The everyday test suite is pip-only (no Julia), run against committed npz
fixtures. Path-gated CI jobs regenerate the references and guard the seams to
the reference codes:

- **julia-parity** — the design-matrix rows match ACEfit to 1e-8 and the linear
  solve matches ACEfit's `solve(QR)` to 1e-9, checked against the committed
  fixtures (`julia/export_model.jl`, `julia/acefit_qr_reference.jl`).
- **coupling-wheels** — builds the `ace-jax-coupling` wheels (manylinux_2_28
  x86_64/aarch64, macOS arm64), tests them in clean environments with no Julia
  (bit-identical to upstream EquivariantTensors), and runs the coupling parity
  against the ACEpotentials references (`julia/coupling_reference.jl`).
- **prior-parity** — the smoothness prior (`construct/prior.py`) matches
  ACEpotentials' `algebraic_smoothness_prior` bit-for-bit
  (`julia/smoothness_reference.jl`), and the committed fixtures match a fresh run.
- **pace-parity** — the PACE path against the ML-PACE C++ (pinned
  `lammps-user-pace`) and python-ace, built from source in their own
  environments; see `pace_ref/README.md`.
