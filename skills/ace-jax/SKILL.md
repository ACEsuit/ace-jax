---
name: ace-jax
description: Build, fit and evaluate Atomic Cluster Expansion (ACE) interatomic potentials in Python/JAX with the ace-jax package — linear ACE, hybrid ACE+GP with calibrated uncertainty, POPS, ASE calculators, PACE (.yace) import/export, LAMMPS (lammps-jax) export, and the `ace-jax`/`aj` CLI. Use when asked to fit an ACE potential to extxyz data, evaluate or deploy a fitted ACE/GP model, load or write a .npz or .yace ACE model, or quantify force/energy uncertainty with ace-jax.
---

# ace-jax

ACE potentials in pure Python/JAX. No Julia is needed anywhere: fitting and
evaluation use only the core package, and authoring a *new* basis shape (`aj basis`)
uses the `basis` extra, a compiled EquivariantTensors wheel.

## Install

```bash
pip install ace-jax              # evaluate, linear fit, ASE calculator
pip install "ace-jax[gp]"        # + `aj fit` (all arms: MAP optimisers, GP, UQ ladder, POPS)
pip install "ace-jax[basis]" # + `aj basis` (compiled EquivariantTensors wheel; no Julia; Linux x86_64/aarch64, macOS arm64)
pip install "ace-jax[cuda]"      # + CUDA 12 JAX
```

`ace-jax-coupling` (the `basis` extra's wheel) is not on PyPI yet: until it is,
`aj basis` for a new basis shape needs a locally built wheel (`coupling/` in the
ace-jax repo); fit and eval are unaffected.

`ace-jax` and `aj` are the same CLI. `aj <cmd> --help` lists every flag.

## Workflow: basis → fit → eval

```bash
# 1. a model definition (unfitted). Skip this step if you already have a .npz.
aj basis --elements Si --order 3 --max-degree 10 --out si.npz
#    multi-element with a frozen species embedding (MACE table JSON {Z, emb},
#    or `identity`); --d-max caps the channel widths (default lossless):
aj basis --elements Cr,Mn,Fe,Co,Ni --order 3 --max-degree 10 \
    --embedding mace_embedding.json [--d-max 16] --out cantor.npz
#    the smoothness prior (Gamma) is built in; --no-gamma skips it. --rcut
#    defaults to 5.5 (with --embedding: 2.5 x mean bond length).

# 2. fit. Label keys default to energy/forces/virial; pass yours explicitly.
K="--energy-key dft_energy --force-key dft_force --virial-key dft_virial"
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --r0 2.35 --out out_linear                      # linear ACE
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 6 --opt lbfgs --map-restarts 3 --map-steps 40 \
    --r0 2.35 --out out_gp                                            # ACE + GP

# 3. evaluate the fitted model on any extxyz
aj eval --model out_linear/model.npz --data new.xyz $K --forces --out pred.csv
aj eval --model out_gp/gp_model.npz  --data new.xyz $K --forces --out pred.csv  # adds energy_std
aj eval --model model.yace --data new.xyz $K --forces                            # PACE works too
```

`aj fit` always needs `--model`, `--out`, `--r0` (the typical nearest-neighbour
distance in Å, which centres the GP hyperprior), and either `--train` [+ `--test`]
or `--data` [+ `--ntrain/--ntest/--test-start`, a seeded split]. Without
`--test`, the fit is scored on its own training set.

## Choosing options

| Goal | Flags |
|---|---|
| Plain linear ACE (fast baseline) | `--m-per-species 0 --rungs map` |
| Hybrid GP with uncertainty | `--m-per-species M` (default 500; start small, e.g. 6–100) |
| Faster, more reliable MAP | `--opt lbfgs` (default is adam with 500 steps, which is slow) |
| Multimodal hyperparameter posterior | `--map-restarts N` (**requires `--opt lbfgs`**) |
| Hyperparameter uncertainty | `--rungs map,laplace` (`--laplace svi` default, or `fd`); also `pathfinder`, `vi`, `nuts`. **Slow**: see Gotchas |
| Useful OOD sigma for the GP | `--density pca --pca-d 128` (pair-only features give anti-informative OOD sigma) |
| Multi-element GP sharing across species | `--embedding mace_embedding.json` (frozen species coregionalization in the GP kernel) |
| GP kernel / objective | `--kernel cosine` (default) or `matern32`, `--no-bump`; `--objective lml` (default) or `loo` |
| Warm-start the MAP | `--init out_old/theta_map.json` |
| Fit a residual over a pair baseline | `--baseline dimer_mean.npz` (then no model file is saved) |
| Big data on limited GPU memory | `--lml host-cache` (**GP arm, `--density pair` or `pca`, `--opt lbfgs`, `--rungs map` only, single device**) |
| Misspecification UQ for linear ACE | `--uq pops` (**linear only: `--m-per-species 0`**) |
| Per-config-type weights | `--weights '{"default":{"E":30,"F":1,"V":1},"bulk":{"E":100,"F":1,"V":1}}'` or a factor list |
| E0 from data, not the model | `--e0 lsq` (default `model`) |
| Out-of-distribution check | `--ood ood.xyz` (writes `metrics_ood.csv`) |

These constraints are validated up front. A bad combination raises a
`ValueError` that names the fix, so read it rather than retrying variants.

## Outputs (`--out DIR`)

- `metrics.csv`, `metrics_ood.csv`: one row per rung × quantity (E in meV/atom,
  F in eV/Å, V). Columns are `rmse`, `mae`, `crps`, `coverage` (fraction
  within ±1σ, ≈0.68 when calibrated), `rho` (error–sigma rank correlation), `rms_z` (≈1 when
  calibrated), `sigma_ratio`, `median_sigma`.
- `theta_map.json`: the MAP hyperparameters. `draws_<rung>.npy` holds the
  hyperparameter draws. `config.json` records the run.
- **Fitted model:**
  - `model.npz` (linear): an ordinary ACE file, loaded by `ace_jax.load`,
    `ACECalculator` and `aj eval`.
  - `gp_model.npz` (GP): self-contained, loaded by `GPCalculator.from_file` and
    `aj eval`. Its size is about 8·Dt²·(model draws) bytes, where Dt = basis
    size + M. The default stores 1 draw (the MAP); `--model-draws N` stores N
    draws of the last rung.
  - `--no-save-model` skips the model file.
  - A fit with `--baseline` saves no model, because its baseline lives
    outside the model.

## Python API

```python
import jax; jax.config.update("jax_enable_x64", True)   # fitting needs float64
import ace_jax as aj
from ace_jax import ACECalculator, GPCalculator

model, meta, z = aj.load("si_fitted.npz")                 # ACEModel, meta dict, raw npz
atoms.calc = ACECalculator("out_linear/model.npz")        # energy, forces, stress
atoms.calc = GPCalculator.from_file("out_gp/gp_model.npz")
atoms.get_forces(); atoms.calc.results["forces_std"]      # also energy_std

# the pipeline behind `aj fit`
from ace_jax.fit.pipeline import FitConfig, load_fit_data, fit, write_outputs, save_model
cfg = FitConfig(model="si.npz", arm="gp", m_per_species=6, opt="lbfgs", r0=2.35,
                rungs=("map",), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", predict_stats="recompute").validate()
data = load_fit_data(cfg, train="train.xyz", test="test.xyz")   # or data="all.xyz" (split)
res = fit(cfg, data)             # res.preds.metrics, res.theta, res.rungs.draws
write_outputs(res, "out", layout=("cli",))                       # metrics + model file
```

In Python, `FitConfig`'s defaults are the research driver's
(`bench/acegp_cantor/run.py`). They differ from the CLI's: the arm is set
explicitly (`arm="linear"|"gp"`, default `"gp"`), `e0="lsq"`, `opt="lbfgs"`
(150 steps), `m_per_species=100`, `batch=4`, `laplace="fd"`, and
`predict_stats="cached"`. Use `"recompute"` to match the CLI and the saved
model file exactly. Python-only options:
- `factors=[...]`: composable weights from `ace_jax.fit.weights`
  (`Structural()`, `Quantity({"E":..,"F":..,"V":..})`,
  `ConfigType({type: {"E":..,"F":..,"V":..}})`, `PerConfig(key="weight")`).
  The CLI's `--weights '[{"Structural": {}}, ...]'` builds the same list.
- `sigma_type=True`: a per-config-type noise block. `route={"sigma_type": "lml"}`
  routes hyperparameter blocks `fixed` or `lml` (`ace_jax.fit.paramset`).
- `pops_rows="auto"|"host"|"device"`: where POPS keeps its design rows
  (`host` caches them in RAM; `auto` picks).

### PACE (`.yace`) models

```python
pm, pmeta, spec = aj.load("model.yace")                  # PACEModel, meta, parsed spec
atoms.calc = ACECalculator("model.yace")
from ace_jax.eval import write_yace
write_yace(pm, spec, "copy.yace")                        # numeric leaves from pm, layout from spec
```

Supported: ChebExpCos, ChebPow, ChebLinear and SBessel radials,
FinnisSinclair(ShiftedScaled) embeddings, and `density`, `distance` and `zbl`
inner cutoffs. `.yace` models are for evaluation and export only: `aj fit` needs
an `.npz` ACE model. An SBessel radial with `nradbase >= 12` is evaluated as one
sin per basis function and a constant matrix (`PACEModel.sbessel_form`, fixed at
load; faster only at large `nradbase`). Values are unchanged to roundoff.

### Calculator performance options

`ACECalculator(path, dtype=None, layout="auto", edge_a_kind="auto", skin=1.0, lean=True)`:
- `lean` (ACE `.npz` models): energies, forces and stress are evaluated with
  `ace_jax.eval.lean(model)`, exact to roundoff. It drops radial columns and
  harmonics the basis never reads, folds the pair weights into the pair
  radial, and pools the dense A per l-block. Forces are 1.1–3.3× faster (A100, docs/ace-vs-pace-gap.md §8).
  `calc.eval_model` is that form. `calc.model` stays the model as given, and
  descriptors use it. A lean model is energy-only: its `site_basis` and
  descriptor methods raise. Never fit from it or edit it (the radial helpers
  call `require_full()` and raise); edit the full model and re-apply `lean`.
  `aj.load` returns the full model. Setting `calc.model` recomputes the lean
  form on the host, a device-to-host copy per swap.
- `layout`: `"sparse"` (edge list) or `"dense"` (padded per-node blocks, A by a
  batched outer product, several times faster forces on GPU). `"auto"` picks
  dense when the padding fill, edges / (atoms × max neighbours), is at least
  `MIN_DENSE_FILL` (0.5). Memory is not a criterion: the dense model runs in
  blocks of `CHUNK_NODES` (16,384) rows.
- `skin` (Å, dense layout only): a Verlet neighbour list built for cutoff +
  skin and reused across MD-like calls, one compiled step each. It rebuilds
  automatically when an atom moves more than skin / 2, when the cell, pbc,
  species or atom count change, or when a row outgrows its capacity.
  `calc.last_timing["rebuilds"]` counts the calls that built a list, and
  `last_timing["nlist_s"]` is 0 on a reuse. Use `skin=0` for one-shot
  evaluation of unrelated structures (it rebuilds on every call). Setting
  `calc.model` or `calc.skin` drops the current list.
- `edge_a_kind`: `"gather"` or `"matmul"`, the two A-basis forms. `"auto"` times
  both once per edge-count bucket.
- Edge lists are padded to power-of-two buckets, so MD reuses the jitted
  function. `pip install "ace-jax[fast-neighbours]"` (matscipy-neighbours,
  a C++ source build) gives faster neighbour lists; ASE's list is the fallback.

### LAMMPS

```python
from ace_jax.export.lammps import export_lammps          # needs lammps-jax installed
model, meta, _ = aj.load("model.npz")                     # or a .yace
export_lammps(model, meta, "bundle", max_atoms=4096, max_edges=200_000,
              dtype="float64", layout="auto", k_dense=64, max_owned=2048,
              type_elements=[14, 32])                     # Z in LAMMPS type order
```

This writes a lammps-jax bundle for `pair_style jax/kk` (GPU only).
- `layout="auto"` chooses dense only when `k_dense` (max neighbours per atom) is
  given and one dense block's `estimate_a_bytes` fits `dense_budget_bytes()`.
- `max_owned`: owned-row capacity of the dense bundle. LAMMPS numbers owned
  atoms first, so only rows below it are evaluated and ghosts cost nothing.
  The bundle records it as `ace_jax.owned_rows`; the key is not `max_owned`
  because lammps-jax reads that name from anywhere in the file. An atom past
  `max_owned`, or with more than `k_dense` neighbours, gives NaN, never a
  silent truncation.
- The dense bundle runs in blocks of `BUNDLE_BLOCK_ROWS` (32,768) rows above
  one block, bounding memory at large N.
- `lean=True` (default) exports `lean(model)` for an ACE model, as
  `ACECalculator` does. It is recorded as `ace_jax.lean`. `layout="auto"` is
  sized on the full model, so it chooses the same layout either way.

Other entry points:
- `aj.site_descriptors(...)`: per-atom ACE descriptors.
- `ace_jax.construct.model.build_model` and `build_embedding_model`: author a
  model in memory.
- `ace_jax.construct.export.save_npz`: write an authored model to `.npz`.
- Learned radial basis (research, linear arm): `bench/learn_radial/run.py
  --model M.npz --data D.xyz --out DIR --r0 2.35 [--n-q 12] [--lam-grid 0,1e-2]`
  learns the tensor radials by VarPro (`ace_jax.fit.radial_learn.learn_radial`)
  and writes `DIR/model.npz` with the radials a held-out gate selects. Fit the GP
  on that file as usual. Needs float64.
- Benchmarks: `docs/benchmarks.md` (harness in `bench/scaling/`).

## Gotchas

- **Data is read with libAtoms `extxyz`, not `ase.io`.** Every label comes
  back under the name it was written with, `energy`/`forces`/`stress`
  included (ASE would move those into a calculator). ASE's `_JSON` 2-D info
  values are decoded. A label that is present but not numeric raises a
  `ValueError` naming the file, config and key.
- **Run time.** The default Adam MAP is 500 steps. On small data use
  `--opt lbfgs --map-steps 40–150`.
- **`--rungs` defaults to `map`.** Adding `laplace` (or `pathfinder`, `vi`,
  `nuts`) gives hyperparameter draws but costs far more than the MAP. The Laplace
  rung differentiates the LML twice (a Hessian): on a laptop CPU, 40 Si configs
  with M=6 had not finished compiling after 30 min.
  - Add those rungs only when you need hyperparameter uncertainty, and try them
    on a small subset first.
  - A silent process after the last `lbfgs ... logpost` line means it is
    compiling the Laplace rung, not hung.
- **Memory.** It scales with Dt² (Dt = basis size + M), and prediction adds
  per-batch kernel tensors.
  - Lower `--configs-per-batch` (default 8) or M.
  - Or use `--lml host-cache`.
- **POPS.** `--uq pops` changes only the uncertainty. The mean is pinned to the
  BLR mean, and the ridge is selected per quantity by CRPS on a training
  hold-out (`--pops-ridge auto`; `blr` or a number fixes it).
- **First `aj basis` of a new basis shape** calls the compiled coupling
  library (`ace-jax-coupling`, milliseconds) to build the coupling table, then
  caches it in `~/.cache/ace-jax/coupling`. Later runs do not need the library.
- **Exit status.** `aj` exits 0 on success. Treat any non-zero exit as a
  failure and read the last lines of stderr.
