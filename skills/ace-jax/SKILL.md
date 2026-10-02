---
name: ace-jax
description: Build, fit and evaluate Atomic Cluster Expansion (ACE) interatomic potentials in Python/JAX with the ace-jax package — linear ACE, hybrid ACE+GP with calibrated uncertainty, POPS, ASE calculators, PACE (.yace) import/export, LAMMPS (lammps-jax) export, and the `ace-jax`/`aj` CLI. Use when asked to fit an ACE potential to extxyz data, evaluate or deploy a fitted ACE/GP model, load or write a .npz or .yace ACE model, or quantify force/energy uncertainty with ace-jax.
---

# ace-jax

ACE potentials in pure Python/JAX. `aj fit` builds the basis from
`--order/--max-degree` and fits it in one command; everything installs with
`pip install ace-jax`.

## Install

```bash
pip install ace-jax              # build bases, `aj fit` (every arm), evaluate, ASE calculator
pip install "ace-jax[gp]"        # + blackjax: the pathfinder rung of --rungs
pip install "ace-jax[cuda]"      # + CUDA 12 JAX
```

Building a new basis shape works on Linux x86_64/aarch64, macOS arm64 and
Windows x64; elsewhere fit from an existing `.npz` with `--model`. Pre-release:
ace-jax itself is not on PyPI yet, so install it from the repo (git URL).

`ace-jax` and `aj` are the same CLI. `aj <cmd> --help` lists every flag.

## Workflow: fit → eval (the basis is built inside the fit)

```bash
K="--energy-key dft_energy --force-key dft_force --virial-key dft_virial"
# 1. fit straight from data: the basis (species from the data) is built in memory.
aj fit --order 3 --max-degree 10 --train train.xyz --test test.xyz $K \
    --m-per-species 0 --out out_linear                                 # linear ACE
#    every fit writes out_linear/fit.yaml: the whole resolved run. Reproduce or
#    vary it (command-line flags override the file):
aj fit --config out_linear/fit.yaml --m-per-species 6 --out out_gp     # same run, + GP

# Optional: save a basis on its own (to share it, or fit it several times).
aj basis --elements Si --order 3 --max-degree 10 --out si.npz
#    multi-element with a frozen species embedding (MACE table JSON {Z, emb},
#    or `identity`); --d-max caps the channel widths (default lossless):
aj basis --elements Cr,Mn,Fe,Co,Ni --order 3 --max-degree 10 \
    --embedding mace_embedding.json [--d-max 16] --out cantor.npz
#    the smoothness prior (Gamma) is built in; --no-gamma skips it. --rcut
#    defaults to 5.5 (with --embedding: 2.5 x mean bond length).

# 2. fit a saved basis. Label keys default to energy/forces/virial; pass yours explicitly.
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --r0 2.35 --out out_linear                      # linear ACE
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 6 --opt lbfgs --map-restarts 3 --map-steps 40 \
    --r0 2.35 --out out_gp                                            # ACE + GP

# 3. evaluate the fitted model on any extxyz
#    prints an E/F/V RMSE table per config_type (labels present); --out writes the
#    structures back as extxyz, every label kept, plus ace_energy, ace_forces, ace_stress
#    (written by the ase-extxyz plugin; read it with ase.io.read(f, ':', format='cextxyz'))
aj eval --model out_linear/model.npz --data new.xyz $K --out pred.xyz
aj eval --model out_gp/gp_model.npz  --data new.xyz $K --out pred.xyz   # adds ace_energy_std, ace_forces_std
aj eval --model model.yace --data new.xyz $K                            # PACE works too; --prefix renames ace_
```

`aj fit` needs a basis — `--order/--max-degree` (built in the fit; `--elements`
defaults to the species in the data, `--basis-embedding` adds a frozen element
embedding) or `--model <file.npz>` — plus `--out` and either `--train` [+ `--test`]
or `--data` [+ `--ntrain/--ntest/--test-start`, a seeded split]. `--r0` (the
typical nearest-neighbour distance in Å, centring the GP hyperprior) defaults to
the built basis's mean bond length and is required with `--model`. Without
`--test`, the fit is scored on its own training set. `--config fit.yaml`
supplies any of these (keys = flag names with underscores, the basis in a
`basis:` block); typos and bad values in the file are errors naming the key.
Command-line flags win, including switching an alternative: `--model m.npz`
over a file's `basis:`, `--train` over its `data:` (each logged as an override).

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
| Learn the tensor radials before the fit | `--learn-radial` (writes `radial_info.json`; not with embedding models) |
| Per-config-type weights | `--weights '{"default":{"E":30,"F":1,"V":1},"bulk":{"E":100,"F":1,"V":1}}'` or a factor list |
| E0 from data, not the model | `--e0 lsq` (default `model`) |
| Stress labels (MACE, ASE, DFT codes) | `--stress-key stress` (virial = −stress × volume for periodic configs without a virial label; also on `aj eval`) |
| Plain least squares, no prior (teaching: shows overfitting) | `--m-per-species 0 --solver lstsq` (no evidence, no UQ: zero predictive variance; weights from `--weights`) |
| Out-of-distribution check | `--ood ood.xyz` (writes `metrics_ood.csv`) |

- **Learned radials: `aj fit --learn-radial`.** Learns the tensor radials before
  the fit (VarPro over the training configs, gated on a seeded
  `--radial-val-frac` hold-out, default 0.2), then fits as usual on the full
  training set. Options: `--radial-n-q 12`, `--radial-steps 40`,
  `--radial-lam-grid 0,1e-2`. Writes `out/radial_info.json` (the gate's
  selection and scores). The saved model is marked `radial_learned`, so
  `ACECalculator`/`export_lammps` spline it. Works with `--model` and with
  `--order/--max-degree`; not with embedding models (issue #31). Advanced priors:
  `ace_jax.fit.radial_learn.fit_radial`.

These constraints are validated up front. A bad combination raises a
`ValueError` that names the fix, so read it rather than retrying variants.

## Outputs (`--out DIR`)

- **Log:** after the fit, an E/F/V RMSE table per `config_type` for each split
  (E and V in meV/atom, F in eV/Å); `aj eval` prints the same table.

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
    `aj eval --posterior out_ard/posterior.npz` (per-atom `ace_forces_std` in its `--out` extxyz) add a `forces_std` result: per-atom
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
# train=/test= also take lists of ase.Atoms: labels from info/arrays or the attached
# calculator's results (a calculator shared by several Atoms raises: its results are the last one's)
res = fit(cfg, data)             # res.preds.metrics, res.theta, res.rungs.draws
res.map.log_evidence             # the LML at the MAP: compare bases fitted to the same data
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

`ACECalculator(path, dtype=None, layout="auto", edge_a_kind="auto", skin=1.0, lean=True, spline_tol="auto", spline_intervals=None)`:
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
  - **Learned radials are splined.** With `spline_tol="auto"` (the default),
    an analytic tensor radial marked `radial_learned` (what `radial_learn`
    writes, e.g. a `bench/learn_radial` model.npz) is splined at 1e-10 by
    `ace_jax.eval.to_spline`.
    - ACEpotentials `ace_model` exports and built bases (`aj basis`, `aj fit`) are analytic but not
      learned, so they stay exact unless you pass a float, e.g.
      `spline_tol=1e-10`.
    - Old learned-radial files written before the flag existed load as not learned: mark one with `ace_jax.basis.export.mark_radial_learned("model.npz")`, or pass `spline_tol=1e-10`.
    - The spline gather replaces the polynomial recursion, and the
      species-compact blocks apply again (learned radials keep ACE1's
      one-neighbour-species-per-column pattern).
    - It is not roundoff: at 1e-10, energies agree with the full model to up to
      ~1e-9 relative and forces to up to ~2.3e-8 of max|F|. `spline_tol=None`
      (on `lean`, `ACECalculator` or `export_lammps`) never splines.
    - `calc.splined` and `calc.last_timing["spline_tol"]` report it.
    - The spline is cached on the radial's content, so a readout-only
      `calc.model` swap does not redo it.
    - The interval count is bucketed (quarter-octave, <= 20% extra), so a
      radial swap usually reuses the compiled step. `spline_intervals=N`
      pins it.
    - Take UQ variances from the full model.
  - A wrapper model with `.base` and `with_base(new_base)` (e.g. an
    `FSModel(base, ...)`) gets `model.with_base(lean_keep_basis(model.base))`:
    `to_spline` and `prune_columns` only, which keep `site_basis` and the
    unfolded readout valid.
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
from ace_jax.export.lammps import export_lammps, neighbour_capacity  # needs lammps-jax
model, meta, _ = aj.load("model.npz")                     # or a .yace
cap = neighbour_capacity(atoms, meta["rcut"], skin=1.0)   # slots="cutoff": tight, opt-in
export_lammps(model, meta, "bundle", max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
              dtype="float64", layout="auto", k_dense=cap["k_dense"],
              max_neighbors=cap["max_neighbors"], max_owned=cap["max_owned"],
              type_elements=[14, 32])                     # Z in LAMMPS type order
```

This writes a lammps-jax bundle for `pair_style jax/kk` (GPU only).
- Layouts: `"sparse"`, `"dense"` (lammps-jax's edge buffer, packed into
  `k_dense` slots by an argsort every step) and `"matrix"` (lammps-jax's
  neighbour-matrix input, `matrix_supported()`: the LAMMPS full list itself,
  `(max_neighbors, max_owned)` slot-major, copied only on list rebuilds; the
  model drops skin pairs at rcut, and compacts in-cutoff pairs when
  `k_dense < max_neighbors`). No per-step packing.
- `layout="auto"` chooses a dense-family layout only when `k_dense` (max
  neighbours per atom) is given and one dense block's `estimate_a_bytes` fits
  `dense_budget_bytes()`. It is `"matrix"` only if `max_neighbors` is passed,
  `matrix_supported()`, and the block plus `matrix_prep_bytes` (the unblocked
  list pre-processing) fits; otherwise `"dense"`. The old signature, without
  `max_neighbors`, stays packed dense. `layout="matrix"` requires
  `max_neighbors`: the list holds rcut + skin pairs, so a `k_dense` sized for
  rcut is too small. `max_edges` is needed by sparse and dense only.
- An older lammps-jax LAMMPS plugin rejects a matrix bundle ("re-export"). To
  fix it, rebuild the plugin, or export `layout="dense"`. The bundle records
  the exporting lammps-jax as `ace_jax.lammps_jax`.
- `max_owned`: owned-row capacity of the dense and matrix bundles. LAMMPS
  numbers owned atoms first, so only rows below it are evaluated and ghosts
  cost nothing. The bundle records it as `ace_jax.owned_rows`; the key is not
  `max_owned` because lammps-jax reads that name from anywhere in the file
  (so ace-jax metadata never reuses a lammps-jax contract key). More than
  `k_dense` neighbours within rcut gives NaN, never a silent truncation. So
  does an atom past `max_owned` in a dense bundle. A matrix bundle instead
  aborts the run in LAMMPS for an owned atom past `max_owned`, or for a list
  row wider than `max_neighbors`.
- `neighbour_capacity(atoms, rcut, skin=1.0, slots="skin", margin=8,
  list_headroom=0.5)`: buffer sizes. The ghost shell uses the face spacings,
  so triclinic cells are fine. The matrix list gets 50% headroom over
  k(rcut + skin), because a compressing structure grows that count fastest.
  The 0.5 comes from one observed overflow, 34 to 45; the benchmark has
  `--list-headroom`. Model slots are compacted from the list. `slots="skin"` (default) is safe between list rebuilds.
  `slots="cutoff"` sizes model slots for rcut pairs, 1.2-1.4x faster on Cantor; use it
  only for stable MD with a fitted model (coordination within `margin` of the
  start). An overflow is a NaN step. The benchmark's `run_lammps.py
  --tight-slots` opts in; the main suite never does.
- The dense bundle runs in blocks of `BUNDLE_BLOCK_ROWS` (32,768) rows above
  one block, bounding memory at large N.
- `lean=True` (default) exports `lean(model, spline_tol, spline_intervals)` for
  an ACE model, as `ACECalculator` does. The bundle records what `lean` actually
  did, looking through a wrapper's `.base`:
  - `ace_jax.lean`: False when `lean` returned the model as given, e.g. a PACE
    or unfolded model.
  - `ace_jax.spline_tol` and `ace_jax.spline_intervals`: both None when nothing
    was splined. `layout="auto"` is
  sized on the full model, so it chooses the same layout either way.

Other entry points:
- `aj.site_descriptors(...)`: per-atom ACE descriptors.
- `ace_jax.basis.model.build_model` and `build_embedding_model`: author a
  model in memory.
- `ace_jax.basis.export.save_npz`: write an authored model to `.npz`.
- Learned radial basis (research, linear arm): `bench/learn_radial/run.py
  --model M.npz --data D.xyz --out DIR --r0 2.35 [--n-q 12] [--lam-grid 0,1e-2]`
  learns the tensor radials by VarPro (`ace_jax.fit.radial_learn.learn_radial`)
  and writes `DIR/model.npz` with the radials a held-out gate selects. Fit the GP
  on that file as usual. Needs float64. For MD, `ACECalculator` and
  `export_lammps` spline the learned radial through `lean` (see above);
  `ace_jax.eval.to_spline(model, n_intervals=None, tol=1e-10)` returns
  `(spline model, max_rel_err)` directly.
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
- **First `aj basis` of a new basis shape** calls the compiled coupling
  library (`ace-jax-coupling`, milliseconds) to build the coupling table, then
  caches it in `~/.cache/ace-jax/coupling`. Later runs do not need the library.
- **Exit status.** `aj` exits 0 on success. Treat any non-zero exit as a
  failure and read the last lines of stderr.
