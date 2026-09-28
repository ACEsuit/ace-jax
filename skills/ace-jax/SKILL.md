---
name: ace-jax
description: Build, fit and evaluate Atomic Cluster Expansion (ACE) interatomic potentials in Python/JAX with the ace-jax package — linear ACE, hybrid ACE+GP with calibrated uncertainty, POPS, ASE calculators, PACE (.yace) import/export, LAMMPS (lammps-jax) export, and the `ace-jax`/`aj` CLI. Use when asked to fit an ACE potential to extxyz data, evaluate or deploy a fitted ACE/GP model, load or write a .npz or .yace ACE model, or quantify force/energy uncertainty with ace-jax.
---

# ace-jax

ACE potentials in pure Python/JAX. You can fit and evaluate models without Julia.
Julia is only needed to author a *new* coupling table on a coupling-cache miss,
and the `authoring` extra provisions it automatically.

## Install

```bash
pip install ace-jax              # evaluate, linear fit, ASE calculator
pip install "ace-jax[gp]"        # + `aj fit` (all arms: MAP optimisers, GP, UQ ladder, POPS)
pip install "ace-jax[authoring]" # + `construct` (juliacall; first run provisions Julia)
pip install "ace-jax[cuda]"      # + CUDA 12 JAX
```

`ace-jax` and `aj` are the same CLI. `aj <cmd> --help` lists every flag.

## Workflow: construct → fit → eval

```bash
# 1. a model definition (unfitted). Skip this step if you already have a .npz.
aj construct --elements Si --order 3 --max-degree 10 --out si.npz
#    multi-element with a frozen species embedding (MACE table JSON {Z, emb},
#    or `identity`); --d-max caps the channel widths (default lossless):
aj construct --elements Cr,Mn,Fe,Co,Ni --order 3 --max-degree 10 \
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
| Calibrated per-atom force uncertainty (e.g. big-cell fracture) | `--m-per-species 0 --uq ard` (`posterior.npz`; `ACECalculator(model, posterior=...)`) |
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
  - `--uq ard` (linear only) also writes `posterior.npz` (float32 Cholesky factor of
    the ARD posterior, plus the (L, n_cfg) sandwich factor Q by default) and `ard.json`
    (evidence, prior scales, κ, λ, held-out NLL and rms-z; `lam_incl_own` is the λ the
    held-out atoms' own training clusters would give, for comparison only).
    `ACECalculator(model, posterior="out_ard/posterior.npz")` and
    `aj eval --posterior out_ard/posterior.npz` add a `forces_std` result: per-atom
    calibrated force uncertainty. `--uq ard` also changes the mean: `model.npz` is the
    ARD posterior mean, not the BLR/MAP mean.
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
an `.npz` ACE model.

### Calculator performance options

`ACECalculator(path, dtype=None, layout="auto", edge_a_kind="auto")`:
- `layout`: `"sparse"` (edge list) or `"dense"` (padded per-node blocks, A by a
  batched outer product, several times faster forces on GPU). `"auto"` picks
  dense when its memory estimate fits the device budget.
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
              dtype="float64", layout="auto", type_elements=[14, 32])  # Z in LAMMPS type order
```

This writes a lammps-jax bundle for `pair_style jax/kk` (GPU only). For
`layout="auto"` to choose dense, pass `k_dense` (max neighbours).

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
- **ARD `forces_std` is computed on request, not on every call.** A plain
  `atoms.get_forces()` does not compute it; call
  `calc.get_property("forces_std", atoms)` (reuses the cached E/F/stress), or
  pass `forces_std_every_call=True` — costly for per-step MD on big cells. Only
  the force σ is calibrated: by default it is λ × the configuration-clustered
  sandwich σ, with `--ard-variance kappa` κ × the posterior σ. Energy and virial
  variances are the uncalibrated posterior ones (`ard.json` `tempered_quantities: ["F"]`
  names the calibrated quantity, whichever scale was used).
  `ACECalculator(model, posterior=...)` raises `ValueError` if the posterior
  doesn't match the model (basis size, species count, element list, or a mean
  that is not the model's coefficients, i.e. a posterior from another fit), and
  `RuntimeError` unless `jax_enable_x64` is on. `forces_std` holds the whole
  cell's force design rows, ~N·3·L·8 bytes.
- **The default sandwich variance (`--ard-variance sandwich`) needs the training data at fit
  time and stores an (L, n_cfg) factor.** Use `--ard-variance kappa` for the smaller posterior.
- **First `construct` of a new basis shape** runs Julia (via juliacall) to
  build the coupling table, then caches it in `~/.cache/ace-jax/coupling`.
  Later runs are pure Python.
- **Exit status.** `aj` exits 0 on success. Treat any non-zero exit as a
  failure and read the last lines of stderr.
