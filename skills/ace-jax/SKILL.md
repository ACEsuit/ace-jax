---
name: ace-jax
description: Build, fit and evaluate Atomic Cluster Expansion (ACE) interatomic potentials in Python/JAX with the ace-jax package — linear ACE, hybrid ACE+GP with calibrated uncertainty, POPS, ASE calculators, and the `ace-jax`/`aj` CLI. Use when asked to fit an ACE potential to extxyz data, evaluate or deploy a fitted ACE/GP model, load a .npz or .yace ACE model, or quantify force/energy uncertainty with ace-jax.
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
#    multi-element with a frozen species embedding (MACE table JSON {Z, emb}):
aj construct --elements Cr,Mn,Fe,Co,Ni --order 3 --max-degree 10 \
    --embedding mace_embedding.json [--d-max 16] --out cantor.npz

# 2. fit. Label keys default to energy/forces/virial; pass yours explicitly.
K="--energy-key dft_energy --force-key dft_force --virial-key dft_virial"
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --rungs map --r0 2.35 --out out_linear          # linear ACE
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 6 --opt lbfgs --map-restarts 3 --map-steps 40 \
    --rungs map --r0 2.35 --out out_gp                                # ACE + GP

# 3. evaluate the fitted model on any extxyz
aj eval --model out_linear/model.npz --data new.xyz $K --forces --out pred.csv
aj eval --model out_gp/gp_model.npz  --data new.xyz $K --forces --out pred.csv  # adds energy_std
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

model, meta = aj.load("si_fitted.npz")                    # .npz or PACE .yace
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
explicitly (`arm="linear"|"gp"`, default `"gp"`), `e0="lsq"`, `opt="lbfgs"`, and
`predict_stats="cached"`. Use `"recompute"` to match the CLI and the saved
model file exactly.

Other entry points:
- `aj.site_descriptors(...)`: per-atom ACE descriptors.
- `ace_jax.construct.model.build_model` and `build_embedding_model`: author a
  model in memory.
- `ace_jax.construct.export.save_npz`: write an authored model to `.npz`.

## Gotchas

- **Run time.** The default Adam MAP is 500 steps. On small data use
  `--opt lbfgs --map-steps 40–150`.
- **The CLI default `--rungs` is `map,laplace`.** The Laplace rung
  differentiates the LML twice (a Hessian). Compiling it can take far longer
  than the MAP: on a laptop CPU, 40 Si configs with M=6 had not finished after
  30 min.
  - Pass `--rungs map` unless you need hyperparameter draws.
  - A silent process after the last `lbfgs ... logpost` line means it is
    compiling the Laplace rung, not hung.
- **Memory.** It scales with Dt² (Dt = basis size + M), and prediction adds
  per-batch kernel tensors.
  - Lower `--configs-per-batch` (default 8) or M.
  - Or use `--lml host-cache`.
- **POPS.** `--uq pops` changes only the uncertainty. The mean is pinned to the
  BLR mean, and the ridge is selected per quantity by CRPS on a training
  hold-out.
- **First `construct` of a new basis shape** runs Julia (via juliacall) to
  build the coupling table, then caches it in `~/.cache/ace-jax/coupling`.
  Later runs are pure Python.
- **Exit status.** `aj` exits 0 on success. Treat any non-zero exit as a
  failure and read the last lines of stderr.
