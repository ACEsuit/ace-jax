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
Windows x64; elsewhere fit from an existing `.npz` with `--model`.
`aj --version` prints the installed version.

`ace-jax` and `aj` are the same CLI. `aj <cmd> --help` lists every flag.

## Workflow: fit → eval (the basis is built inside the fit)

```bash
K="--energy-key dft_energy --force-key dft_force --virial-key dft_virial"
# 1. fit straight from data: the basis (species from the data) is built in memory.
aj fit --order 3 --max-degree 10 --train train.xyz --test test.xyz $K \
    --out out_linear                                                   # linear ACE (the default)
#    every fit writes out_linear/fit.yaml: the whole resolved run. Reproduce or
#    vary it (command-line flags override the file):
aj fit --config out_linear/fit.yaml --m-per-species 6 --out out_gp     # same run, + GP (experimental)

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
    --r0 2.35 --out out_linear                                        # linear ACE
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 6 --opt lbfgs --map-restarts 3 --map-steps 40 \
    --r0 2.35 --out out_gp                                            # ACE + GP (experimental)

# 3. evaluate the fitted model on any extxyz
#    prints an E/F/V RMSE table per config_type (labels present); --out writes the
#    structures back as extxyz, every label kept, plus ace_energy, ace_forces, ace_stress
#    (written by the ase-extxyz plugin; read it with ase.io.read(f, ':', format='cextxyz'))
aj eval --model out_linear/model.npz --data new.xyz $K --out pred.xyz
aj eval --model out_gp/gp_model.npz  --data new.xyz $K --out pred.xyz   # adds ace_energy_std, ace_forces_std
aj eval --model model.yace --data new.xyz $K                            # PACE works too; --prefix renames ace_
aj export --model out_linear/model.npz --out model.yace                # ML-PACE .yace (exact; tabulated radials)
aj export --model out_linear/model.npz --out bundle.json --structure start.xyz   # lammps-jax bundle
```

Per-order caps: `--nmax-by-order 15,6,4,3,2,2 --lmax-by-order 0,4,3,2,1,0` (pacemaker's
`nradmax_by_orders`/`lmax_by_orders`, from correlation order 1, the last entry repeating;
`BasisSpec.nmax_by_order`/`lmax_by_order`, `build_model(nmax_by_order=, lmax_by_order=)`) prune
the TotalDegree basis per order (categorical bases only, not `--basis-embedding`); with `--wL 1`
the degree is pacemaker's power order.

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
| Plain linear ACE (the default; what most fits should use) | no flags (`--m-per-species 0 --rungs map`) |
| Hybrid GP with uncertainty (experimental, advanced) | `--m-per-species M` (default 0 = linear; start small, e.g. 6–100). `--kernel`, `--no-bump`, `--density`, `--pca-d`, `--embedding` need `M > 0` (an error otherwise) |
| Converged MAP | the default `--opt lbfgs` (bounded L-BFGS-B, one restart after a line-search stop; linear arm: + exact-Hessian Newton polish by HVP columns, `--map-polish auto/on/off`; GP: `on` is opt-in, a GP gradient can take a minute). Writes `map_convergence.json`; warns `MAP did not converge`, or fails with `--strict`. `--opt adam` (`--map-steps` steps) often stops short |
| Multimodal hyperparameter posterior | `--map-restarts N` (**requires `--opt lbfgs`**) |
| Hyperparameter uncertainty | `--rungs map,laplace` (`--laplace svi` default, or `fd`); also `pathfinder`, `vi`, `nuts`. **Slow**: see Gotchas |
| Useful OOD sigma for the GP | `--density pca --pca-d 128` (pair-only features give anti-informative OOD sigma) |
| Multi-element GP sharing across species | `--embedding mace_embedding.json` (frozen species coregionalization in the GP kernel) |
| GP kernel / objective | `--kernel cosine` (default) or `matern32`, `--no-bump`; `--objective lml` (default) or `loo` |
| Linear-arm evidence solver | `--lml-solver qr` (default: stable when the fit nearly interpolates) or `cholesky` (faster at several thousand functions; can lose accuracy silently) |
| Fit faster on a float32 GPU | `--row-precision float32` (linear arm, QR only; **not yet supported**: the GP arm and `--learn-radial`, refused at validation): design rows and, on a GPU, their QR merges in float32 with the residual carried in the factor (`QRStats.rho_q`); evidence/MAP/posterior stay float64. RTX A4500: stats pass 3x, o4d16 Si fit 265 -> 169 s, RMSE within 1%. σ_c can move more than predictions do; `--row-check` (noise-injection MAP refit) records the relative training-RMSE change per quantity as `row_rounding_check` in `map_convergence.json` (stable <= 5%, unstable > 10%; it overestimates the real fp32 change ~1.5-4x) |
| Warm-start the MAP | `--init out_old/theta_map.json` |
| Fit a residual over a pair baseline | `--baseline dimer_mean.npz` (then no model file is saved) |
| Big data on limited GPU memory | `--lml host-cache` (**GP arm, `--density pair` or `pca`, `--opt lbfgs`, `--rungs map` only, single device**) |
| Misspecification UQ for linear ACE | `--uq pops` (**linear only**, the default arm) |
| Calibrated per-atom force uncertainty (e.g. big-cell fracture) | `--uq ard` (`posterior.npz`; `ACECalculator(model, posterior=...)`) |
| The same on the hybrid ACE + GP model (experimental) | `--m-per-species 100 --uq ard-gp` (`gp_model.npz` + `posterior.npz`; `GPCalculator.from_file(gp_model, posterior=...)`) |
| Learn the tensor radials before the fit | `--learn-radial` (writes `radial_info.json`; not with embedding models) |
| Per-config-type weights | `--weights '{"default":{"E":30,"F":1,"V":1},"bulk":{"E":100,"F":1,"V":1}}'` or a factor list; also accepts the path of a `.json` file |
| Make the weights set the E:F:V balance (ACEpotentials BLR) | Give `--weights`: the default `--noise auto` then resolves to `shared`, one noise σ for every weighted row (σ_E = σ_F = σ_V, a tie in the MAP; recorded in `map_convergence.json`; `fit.yaml` records the resolved mode). Without weights `auto` is `per-quantity` (three σ's, each cancelling its weight; at 1:1:1 weights shared noise cost +79 % energy RMSE on Cantor). `auto` also falls back to `per-quantity` with `sigma_type`, joint ARD or `--solver lstsq`; an explicit `--noise shared` there is refused. Not with `sigma_type`, joint ARD (use `--ard-mode sequential`) or `--solver lstsq`. Compare fits across modes by `log_evidence`, not `logpost` |
| E0 from data, not the model | `--e0 lsq` (default `model`): fitted jointly with the readout (wide prior around a least-squares start; isolated atoms pin their species); `--e0 prefit` fixes the least-squares E0 first (POPS always does; ARD fits E0 jointly under `--e0 lsq`, as BLR) |
| Stress labels (MACE, ASE, DFT codes) | `--stress-key stress` (virial = −stress × volume for periodic configs without a virial label; also on `aj eval`) |
| Plain least squares, no prior (teaching: shows overfitting) | `--solver lstsq` (no evidence, no UQ: zero predictive variance; weights from `--weights`) |
| Out-of-distribution check | `--ood ood.xyz` (writes `metrics_ood.csv`) |

- **Learned radials: `aj fit --learn-radial`.** Learns the tensor radials before
  the fit (VarPro over the training configs, gated on a seeded
  `--radial-val-frac` hold-out, default 0.2), then fits as usual on the full
  training set. Options: `--radial-n-q 12` (only widens: a wider analytic model keeps its span; the log gives the one used), `--radial-steps 300`, `--radial-reprofile-every 100` (theta re-fit, which restarts L-BFGS), `--radial-opt lbfgs|newton` (trust-region Newton with exact Hessian-vector products, `--radial-cg-iters 20` per iteration),
  `--radial-lam-grid 0,1e-2`. Writes `out/radial_info.json` (the gate's
  selection and scores: validation SSE_E/sigma_E^2 + SSE_F/sigma_F^2, the objective's weighting) and, as each roughness weight finishes and before the gate,
  `out/radial_lam=<w>.npz` (`rnl_Wnlq`, `steps`, `theta_final`): hours of learning
  survive a later failure. The saved model is marked `radial_learned`, so
  `ACECalculator`/`export_lammps` spline it. Works with `--model` and with
  `--order/--max-degree`; not with embedding models (issue #31). Advanced priors:
  `ace_jax.fit.radial_learn.fit_radial`.
- **Radial basis: `--radial-basis poly|sbessel`** (a basis flag, no
  `--learn-radial` needed; `BasisSpec.radial_basis`, `build_model(rnl_basis=)`,
  `ACEModel.rnl_basis`). `sbessel` is pacemaker's simplified spherical Bessel
  basis g_k(r) on [0, rc), zero from rc, no transform or envelope:
  R_nl = sum_k W_nlk g_k. The radial learner works on either. `lean`'s
  `spline_tol="auto"` never splines an sbessel radial (`to_spline` refuses it;
  `radial_table` tabulates it instead). Saved in meta_json as `rnl_basis`.

These constraints are validated up front. A bad combination raises a
`ValueError` that names the fix, so read it rather than retrying variants.

## Outputs (`--out DIR`)

- **Log:** after the fit, an E/F/V RMSE table per `config_type` for each split
  (E and V in meV/atom, F in eV/Å); `aj eval` prints the same table. stdout is
  line-buffered, so a log redirected to a file follows a running fit.
- `progress.jsonl`, written live: one JSON object per event (`time`, `elapsed_s`,
  `event`, fields), for following or parsing a running fit (`tail -f`, or
  `jq 'select(.event=="stage")'`). Events: `fit` (start; done with the test
  metrics), `stage` (`data`, `radial`, `problem`, `objective` = the statistics pass,
  `map`, `rungs`, `ard`, `predict`, `outputs`; start/done/failed with `seconds`),
  `map_eval` (each L-BFGS evaluation), `map_polish`, `map_result` (theta,
  `log_evidence`), and with `--learn-radial` `radial_candidate`, `radial_round`
  and `radial_gate`/`radial_selected`. A `radial_round`'s `obj_start`/`obj_end` are at that
  round's fixed theta (not comparable across rounds); `lml`, after the re-fit of theta,
  is the progress measure across rounds.

- `metrics.csv`, `metrics_ood.csv`: one row per rung × quantity (E in meV/atom,
  F in eV/Å, V). Columns are `rmse`, `mae`, `crps`, `coverage` (fraction
  within ±1σ, ≈0.68 when calibrated), `rho` (error–sigma rank correlation), `rms_z` (≈1 when
  calibrated), `sigma_ratio`, `median_sigma`.
- `theta_map.json`: the MAP hyperparameters. `draws_<rung>.npy` holds the
  hyperparameter draws. `config.json` records the run.
- **Fitted model:**
  - `model.npz` (linear): an ordinary ACE file, loaded by `ace_jax.load`,
    `ACECalculator` and `aj eval`.
  - `--uq ard` (linear only) also writes `posterior.npz` (for --ard-variance kappa only, a packed float32 Cholesky factor of
    the ARD posterior, the jackknife cluster factors and the per-group conformal table) and
    `ard.json` (evidence, prior scales, per-group scales, held-out NLL and rms-z).
    `ACECalculator(model, posterior="out_ard/posterior.npz")` and
    `aj eval --posterior out_ard/posterior.npz` add per-atom calibrated force
    uncertainty: `forces_std`, plus `forces_cov`, `forces_q`, `forces_group`, `forces_support`
    (see `docs/dev/ard-force-uq.md`). `aj eval --out` writes `ace_forces_std` into its extxyz;
    `aj eval --per-atom pa.xyz` writes the served arrays (`--support` adds `support_ok`/`support_q`).
    `--uq ard` also changes the mean: `model.npz` is the
    ARD posterior mean, not the BLR/MAP mean. The hold-out scales are carried to the served
    posterior by (N/N_fit)^β, β fitted per run from a second hold-out fit and clipped to [0, ½]
    (`--ard-transfer exponent|sqrt|none`; it costs one more evidence fit + PRESS on ~(1−f)²N
    configs; `ard.json` `transfer` records λ1, λ2, β and the factor).
  - `--uq ard-gp` (GP arm, experimental): the same stage over the joint design [B | k(B, B_M)] at the
    MAP theta (one ARD scale on the GP block, prior root blockdiag(diag Γ, chol(K_MM)ᵀ)). It writes
    `gp_model.npz` holding the ARD mean and `posterior.npz` (with `gp_U`, `gp_theta`); serve with
    `GPCalculator.from_file("gp_model.npz", posterior="posterior.npz")` or
    `aj eval --model gp_model.npz --posterior posterior.npz`. Cantor acceptance: tip coverage 0.902 vs 0.893
    for `--uq ard`, Spearman ρ 0.006-0.009 lower, so it stays experimental. The support flag and
    `aj calibrate` stay linear-only; a calculator refuses a posterior of the other arm.
  - `gp_model.npz` (GP): self-contained, loaded by `GPCalculator.from_file` and
    `aj eval`. Its size is about 8·Dt²·(model draws) bytes, where Dt = basis
    size + M. The default stores 1 draw (the MAP); `--model-draws N` stores N
    draws of the last rung.
  - `--no-save-model` skips the model file.
  - A fit with `--baseline` saves no model, because its baseline lives
    outside the model.

## Python API

```python
import jax; jax.config.update("jax_enable_x64", True)   # fitting needs float64 (fit raises without it)
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
                virial_key="dft_virial", predict_stats="auto").validate()
data = load_fit_data(cfg, train="train.xyz", test="test.xyz")   # or data="all.xyz" (split)
# train=/test= also take lists of ase.Atoms: labels from info/arrays or the attached
# calculator's results (a calculator shared by several Atoms raises: its results are the last one's)
res = fit(cfg, data)             # res.preds.metrics, res.theta, res.rungs.draws
res.map.log_evidence             # the LML at the MAP: compare bases fitted to the same data
write_outputs(res, "out", layout=("cli",))                       # metrics + model file
```

In Python, `FitConfig`'s defaults are the research driver's
(`bench/acegp_cantor/run.py`). They differ from the CLI's: the arm is set
explicitly (`arm="linear"|"gp"`, default `"gp"`, whereas `aj fit` defaults to linear), `e0="lsq"`, `map_steps=150`
(both use `opt="lbfgs"`, `map_polish="auto"`), `m_per_species=100`, `batch=4`, `laplace="fd"`, and
`predict_stats="cached"`. Use `"auto"` (the CLI's) to match the CLI and the saved
model file exactly: it reuses the fit's QR statistics (bitwise a recompute's) and
recomputes per draw elsewhere; `"recompute"` gives the same numbers with extra passes. Python-only options:
- `factors=[...]`: composable weights from `ace_jax.fit.weights`
  (`Structural()`, `Quantity({"E":..,"F":..,"V":..})`,
  `ConfigType({type: {"E":..,"F":..,"V":..}})`, `PerConfig(key="weight")`).
  The CLI's `--weights '[{"Structural": {}}, ...]'` builds the same list.
- `noise="auto"` (default): `validate()` resolves it to `"shared"` (one noise scale for the E, F and V rows)
  when `weights`/`factors` are given and shared noise is available, else `"per-quantity"`.
- `sigma_type=True`: a per-config-type noise block. `route={"sigma_type": "lml"}`
  routes hyperparameter blocks `fixed` or `lml` (`ace_jax.fit.paramset`).
- `pops_rows="auto"|"host"|"device"`: where POPS keeps its design rows
  (`host` caches them in RAM; `auto` picks).

### PACE (`.yace`) models

```python
pm, pmeta, spec = aj.load("model.yace")                  # PACEModel, meta, parsed spec
atoms.calc = ACECalculator("model.yace")
from ace_jax.eval import write_yace
write_yace(pm, "copy.yace", spec=spec)                   # numeric leaves from pm, layout from spec
model, meta, _ = aj.load("model.npz")
info = write_yace(model, "ace.yace")                     # a linear ACE model, exactly (Tabulated radials:
                                                         # ML-PACE needs its tabulated-radial addition)
```

Supported: ChebExpCos, ChebPow, ChebLinear and SBessel radials,
FinnisSinclair(ShiftedScaled) embeddings, and `density`, `distance` and `zbl`
inner cutoffs. `.yace` models are for evaluation and export only: `aj fit` needs
an `.npz` ACE model. An SBessel radial with `nradbase >= 12` is evaluated as one
sin per basis function and a constant matrix (`PACEModel.sbessel_form`, fixed at
load; faster only at large `nradbase`). Values are unchanged to roundoff.

### Calculator performance options

`ACECalculator(path, dtype=None, layout="auto", edge_a_kind="auto", skin=1.0, lean=True, spline_tol="auto", spline_intervals=None, radial_table=None, ffi=False, ffi_threads=0)`:
- `ffi=True` (CPU backend, float64, lean, linear ACEModel incl. species embedding): the C++ kernel
  (`eval.ffi.KernelModel`, entry point `AceEdges`) evaluates the dense layout's site energies and dE/drij;
  JAX assembles forces and stress, so skin lists and `calc.model` swaps work as before. Rows run on XLA's
  intra-op thread pool (`ffi_threads` caps it): ~17x the JAX model at 1.7k-4k atoms on 32 cores (o4d20).
  Radial table on (energies ~1e-12, forces ~1.5e-8 of max|F|). Compiled on first use (C++17 compiler,
  cached). `ffi="auto"` falls back to the JAX model with `calc.ffi_reason`. Derivatives in positions only:
  for parameter gradients or second derivatives keep `ffi=False` (the default). Pair it with the
  `native` extra (matscipy-neighbours; formerly `fast-neighbours`): list rebuilds 4-12x faster than ASE's (14 ms at
  4096 atoms). `aj native` reports the compiler, flags, CPU, kernel cache and neighbour list; `aj native --build`
  compiles the kernel now (exit 1 if it cannot; on a cluster run it on a compute node).
- `dtype` (e.g. `np.float32`): under x64 the model is read and made lean in float64 and only
  `calc.eval_model` is cast (`eval.with_dtype`); a pre-loaded (model, meta) is cast too.
  `calc.precision.precision_check(model, meta, atoms)` measures the float32 error of a model.
- `lean` (ACE `.npz` models): energies, forces and stress are evaluated with
  `ace_jax.eval.lean(model)`, exact to roundoff. It drops radial columns and
  harmonics the basis never reads, folds the pair weights into the pair
  radial, and pools the dense A per l-block. Forces are 1.1–3.3× faster (A100, docs/dev/ace-vs-pace-gap.md §8).
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
- `radial_table` (ACE and PACE; default None = off): `True` (sized from 4000
  intervals up to a 1e-9 value error, `splinify.DEFAULT_TABLE_TOL`, checked on
  r >= 0.4 rcut; one extra fit when 4000 is not enough; it stops early, with a
  warning, when a finer table's derivative is no more accurate), a float (that tolerance) or an int (that many
  intervals) tabulates the per-edge radial stage in r after `lean`: ACE R_nl
  (transform and envelope included) and the pair radial, PACE g_k. One cubic
  B-spline per species pair on [0.5 Å, rcut], exactly zero beyond each pair's
  cutoff; below 0.5 Å the end cubic extrapolates. A CPU speed-up (see
  CHANGELOG), not roundoff: with True energies agree to ~1e-12 relative,
  forces to ~1e-8 of max|F| (4000 fixed intervals gave 3.6e-7 on a
  species-embedding SiGe o4d16). `calc.radial_table` (n_intervals,
  r_min, r_max, max_rel_err, max_rel_deriv_err) and
  `calc.last_timing["radial_table"]` report it; `lean(model,
  radial_table=True)` gives the same model. Wrapper models refuse it.
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
  function. `pip install "ace-jax[native]"` (matscipy-neighbours,
  a C++ source build) gives faster neighbour lists; ASE's list is the fallback.

### Batched evaluation (many small structures)

`ace_jax.batch.BatchCalculator(model, **ACECalculator evaluation options, max_batch_atoms=None,
n_device=None, builder="auto").calculate(atoms_list)` -> `BatchResults` (`energy` (n,), `forces` list,
`stress` (n, 6) NaN unless fully periodic, `virial` (n, 3, 3), `e0_offset`, `timing`; `.to_atoms(atoms_list)`).
User page: `docs/user/howto/batch.md`; numbers: `docs/dev/batched-evaluation-results.md`.

- Pads structures into size buckets (`batch.pack`), builds each batch's lists in one step, evaluates
  B x N rows as one dense block (`batch.efv`). Agrees with `ACECalculator` to roundoff; same option
  resolution (it builds one inside, `bc.calc`).
- Use it for screening, phonon displacements, committee pools; `aj eval` already uses it (no
  posterior/GP). Not for one or two structures (slower), not for MD (no integrators in ace-jax).
- GPU: 28-54x over one-structure-per-call for 2-8 atom Si cells, 5-7x at 64. CPU: only with
  `ffi=True` (19-52x small cells, 2.4x at 64); the JAX model barely gains on a CPU, so there
  `n_device` defaults to 16 (bigger structures go through `ACECalculator`).
- `builder`: "msn" (matscipy-neighbours' batched XLA FFI, when installed), "jax" (all pairs over
  images, GPU), "host" (cell list per structure); "auto" picks.
- Batched MD / relaxation = JAX-MD's own `simulate.*` / `minimize.*` under `jax.vmap` (ace-jax hosts no
  integrators): `nb = bc.neighbour_lists(structures, skin)` (one padded batch; `nb.positions`, `nb.live`,
  `nb.unpad`), `nl = nb.allocate()`, `energy = nb.energy_fn(nl)` (`energy(R, neighbor=nl_slice)`; K slots from
  the in-cutoff capacity x1.2, row blocks sized for the batch; `bc.energy_fn()` evaluates every list slot;
  on a CPU use `ffi=True`: the kernel vmaps into one threaded call, 37-48x the JAX model), `space.free()`; jit(vmap) a fori_loop chunk, then `nl, rebuilt = nb.update(state.position, nl)`
  between chunks (skin/2 test, K grows on overflow -> recompile), or inside the jitted loop with
  `update = nb.update_fn()`: `nl, rebuilt, ov = update(state.position, nl)` then
  `jax.vmap(apply)(state, neighbor=nl)`; check `ov` after the loop and `nb.allocate` on overflow.
  On a GPU a per-step lax.cond syncs with the host (cheap on an idle GPU, costly on a shared one:
  `update_fn(branchless=True)` then); checking every n steps (nested fori_loops) is cheapest. Padding: pass
  `momenta=batch.initial_momenta(key, nb.live, kT, mass, dtype=nb.positions.dtype)` to `init` (JAX-MD's own draw gives padding
  momenta and removes the COM over all slots, drifting real atoms), `batch.zero_padding` after Langevin,
  `batch.temperature(p, mass, live)`; Nose-Hoover counts padding as DOF -> only unpadded (one-size)
  batches. Tests `tests/test_batch_md.py`.

### JAX-MD (MD fully on the device)

`pip install "ace-jax[jaxmd]"`. User page: `docs/user/howto/jax-md.md`.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax_md import partition, simulate, space
from ace_jax import jaxmd

box = jnp.asarray(atoms.cell.array.T)              # JAX-MD: lattice vectors as columns
disp, shift = space.periodic_general(box, fractional_coordinates=True)
R = jnp.asarray(atoms.get_scaled_positions())
rc = jaxmd.cutoff("model.npz")                      # from the file's metadata
nf = partition.neighbor_list(disp, box, rc, dr_threshold=1.0, format=partition.Dense,
                             fractional_coordinates=True)
nbrs = jaxmd.allocate(nf, R)                        # not nf.allocate: JAX 0.11 workaround
pot = jaxmd.potential("model.npz", disp, atoms.numbers, K=jaxmd.capacity(R, nbrs, disp, rc + 1.0))
init, apply = simulate.nve(pot.force, shift, 1.0 / 10.18)   # dt 1 fs (time unit 10.18 fs)
# jitted lax.fori_loop body: nbrs = nbrs.update(state.position); state = apply(state, neighbor=nbrs)
# after each chunk: did_buffer_overflow -> allocate the list again; pot.overflow(...) or non-finite
#   positions (an overflow mid-chunk) -> make the potential with a larger K (and re-jit), then
#   state = state.set(force=pot.force(state.position, nbrs)) (the state caches forces); redo the chunk
```

- `pot.force` is `-grad(pot.energy)` (Cartesian, also with fractional positions);
  `quantity.stress` and NPT integrators take `pot.energy`. JAX-MD's stress sign is
  the opposite of ASE's.
- `K` slots per atom: `capacity(..., rc + skin)` never overflows between list
  builds; `capacity(..., rc)` (tight) is ~1.5x faster but can overflow as atoms
  move. Too small a K, or an overflowed list (Dense or Sparse), makes
  energy and forces NaN, never truncated; `pot.overflow` is the explicit check.
- Sparse JAX-MD lists work without `K` (K is refused for them).
- CPU: `potential(..., ffi=True)` runs the C++ kernel (linear ACE, float64; stress fine, no second
  derivatives): 10-18x the JAX model for 216-1728 Si atoms on 32 cores. The dense energy core is
  shared with the batched path (`eval/dense.py`).
- Memory: rows run in blocks (`chunk="auto"`: `jaxmd.block_rows`, the lammps-jax
  bundles' `block_rows_for` rule, lax.map + checkpoint above one block). An int
  `chunk` fixes it: one large block avoids the checkpoint recompute (faster for
  small float64 models) at several times the memory.
- Calculator keywords (`spline_tol`, `lean`, `aa_dag`, `dtype`, ...) pass through
  `potential`: it evaluates exactly `ACECalculator`'s model.
- JAX-MD's own list is minimum-image: every box width must exceed 2 (rc + skin)
  (`jaxmd.min_image_ok`), else it silently misses neighbours.

### LAMMPS

```python
from ace_jax.export.lammps import export_lammps, neighbour_capacity  # needs `pip install "ace-jax[lammps]"`
model, meta, _ = aj.load("model.npz")                     # or a .yace
cap = neighbour_capacity(atoms, meta["rcut"], skin=1.0)   # slots="cutoff": tight, opt-in
export_lammps(model, meta, "bundle", max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
              dtype="float64", layout="auto", k_dense=cap["k_dense"],
              max_neighbors=cap["max_neighbors"], max_owned=cap["max_owned"],
              type_elements=[14, 32])                     # Z in LAMMPS type order
```

`dtype="float32"`: the traced model is cast to float32 (`ace_jax.eval.with_dtype`, after lean / splines in
float64; spline tables in a local, difference form so their derivatives keep float32 precision), so the bundle
is a float32 program even under x64; 2-4.5x faster than float64 on fp64-poor GPUs (RTX A4500), only 1.3-1.5x on an
A100 (not matmul-bound: TF32 gains nothing and costs 1-14% force error).
`precision_structure=atoms` records `calc.precision.precision_check` (float32 vs float64 E/F/stress on
`atoms`) as `ace_jax.precision_check` and warns above `PRECISION_WARN` (0.1 meV/atom, 1e-4 relative force);
`aj export --dtype float32` checks on `--structure`.  The float32 error is the model's (readout cancellation),
1e-5..1e-3 relative force; lammps-jax passes float32 absolute positions, adding ~ulp(|x|) that grows with the box.
When float32 is safe for a workflow (force-only optimisers such as ODE12r, fmax >= ~3x the force error; NVE at
<= 1 fs drifts no more than float64; energy line searches fail): docs/user/howto/precision.md.

This writes a lammps-jax bundle for `pair_style jax/kk` (GPU; `platform="cpu"`
for its CPU build, which refuses a cuda bundle and vice versa).
- Layouts: `"sparse"`, `"dense"` (lammps-jax's edge buffer, packed into
  `k_dense` slots by an argsort every step) and `"matrix"` (lammps-jax's
  neighbour-matrix input, `matrix_supported()`: the LAMMPS full list itself,
  `(max_neighbors, max_owned)` slot-major, copied only on list rebuilds; the
  model drops skin pairs at rcut, and compacts in-cutoff pairs when
  `k_dense < max_neighbors`). No per-step packing.
- `layout="auto"` chooses a dense-family layout only when `k_dense` (max
  neighbours per atom) is given and a dense block's `estimate_a_bytes` fits the
  budget. It is `"matrix"` only if `max_neighbors` is passed,
  `matrix_supported()`, and the block plus `matrix_prep_bytes` (the unblocked
  list pre-processing) fits; otherwise `"dense"`.
- Memory: dense and matrix rows run in blocks (`block_rows_for`): the largest of
  `BUNDLE_BLOCK_ROWS` (32768) halved down to `MIN_BLOCK_ROWS` (1024) that fits
  `DENSE_BUDGET_FRACTION` (0.5) of `device_memory`, the bytes XLA can allocate
  on the GPU LAMMPS runs on (default: the exporting process's device; with
  XLA's default preallocation, 75% of the GPU, so ~30e9 on a 40 GB A100).
  Pass `device_memory=` when exporting on another machine; `block_rows=`
  overrides. Recorded as `ace_jax.block_rows`, `device_memory`, `dense_budget`. The old signature, without
  `max_neighbors`, stays packed dense. `layout="matrix"` requires
  `max_neighbors`: the list holds rcut + skin pairs, so a `k_dense` sized for
  rcut is too small. `max_edges` is needed by sparse and dense only.
- `ffi` (default `"auto"`): on a CPU bundle the C++ XLA FFI kernel `ace_jax.eval.ffi` replaces the XLA
  program on the matrix layout (positions in, energies and forces out, `newton on`; radial table on) when it
  can: linear ACEModel (categorical or species embedding), float64, lean, layout auto/matrix with
  `max_neighbors`, lammps-jax >= 0.2.1, and the kernel compiles. Otherwise the XLA program, with the reason in
  `bundle["ace_jax"]["ffi_reason"]` (`export.lammps.choose_ffi`). `ffi=True` requires it (raises when it cannot),
  `ffi=False` is always the XLA program, `ffi=<path>` uses a prebuilt library.
  The export compiles it on first use (`ffi.library()`: `$CXX`/c++/g++/clang++, `-O3 -march=native` or
  `ACEJAX_FFI_FLAGS`, cached in `~/.cache/ace-jax/ffi` or `$ACEJAX_FFI_CACHE`; no compiler: "auto" falls back,
  `ffi=True` raises `ffi.FFIUnavailable`). CLI: `aj export --platform cpu` (auto), `--ffi [--ffi-flags="..."]`
  (required), `--no-ffi` (XLA); it prints the line to set, and `aj export --print-env BUNDLE` prints it on the
  LAMMPS machine (compiling the kernel there if the recorded library is missing). A kernel bundle needs
  `LAMMPS_JAX_FFI_HANDLERS` = `bundle["ace_jax"]["ffi"]["handlers"]` in LAMMPS (without it LAMMPS stops),
  newton on, one MPI rank per core. `-march=native` on a login node may not run on compute nodes of another CPU:
  export there, or pass generic flags. Needs a lammps-jax plugin that registers FFI handlers for the CPU
  (lammps-jax >= 0.2.1, the `lammps` extra's floor; 0.2.0 registers them for CUDA only). Build the plugin at the
  version of the Python package.
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
- `aa_dag="cueq"` (opt-in; `auto` never picks it; `aj export --aa-dag cueq`): the product basis as NVIDIA
  cuequivariance kernels (`eval/cueq.py`), CUDA only, `pip install "ace-jax[cueq]"`. One species, or species-embedding
  (TrACE) models (a linear `ACEModel`). Refused: cueq missing, PACE or non-ACE models, categorical multi-species, a non-CUDA
  platform or device (`ACECalculator` also needs a CUDA backend). Needs `LAMMPS_JAX_CONTRIB` = lammps-jax's
  `contrib/ffi-replay` directory at export (the recorder is not in the package) and at run time (its built
  `ffi_replay_shim.so`; README gcc command, `-lffi` on libffi.so.6 systems). One cueq bundle per process.
  The bundle, `<stem>.ffi-replay` and `<stem>.cueq-cache/` go together (recorded by name, relative to the
  bundle, as `ace_jax.cueq`); `aj export` prints the environment lines (`eval.cueq.run_env(record, bundle_path)`)
  resolved on the export host; on another LAMMPS machine (with `ace-jax[cueq]` installed, the shim built and
  `LAMMPS_JAX_CONTRIB` set there) run `aj export --print-env BUNDLE.json` to print them for that machine.
  `export_lammps(lean=False, aa_dag="cueq")` is refused.
  Export of o4d20 13-26 min on an A100; without the cache the start compiles kernels (tens of minutes); the cache serves
  the same GPU architecture only. A `calc.model` swap rebuilds and recompiles.
- `lean=True` (default) exports `lean(model, spline_tol, spline_intervals)` for
  an ACE model, as `ACECalculator` does. The bundle records what `lean` actually
  did, looking through a wrapper's `.base`:
  - `ace_jax.lean`: False when `lean` returned the model as given, e.g. a PACE
    or unfolded model.
  - `ace_jax.spline_tol` and `ace_jax.spline_intervals`: both None when nothing
    was splined.
  - `radial_table=None` (as on `ACECalculator`; applied after `lean`, to every
    layout, PACE included) is recorded as `ace_jax.radial_table`: the table's
    info, None when off. `layout="auto"` is
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
- Benchmarks: `docs/dev/benchmarks.md` (harness in `bench/scaling/`).

## Gotchas

- **Data is read with libAtoms `extxyz`, not `ase.io`.** Every label comes
  back under the name it was written with, `energy`/`forces`/`stress`
  included (ASE would move those into a calculator). ASE's `_JSON` 2-D info
  values are decoded. A label that is present but not numeric raises a
  `ValueError` naming the file, config and key.
- **MAP convergence.** The default `--opt lbfgs` converges in tens of
  evaluations; the linear arm then takes a Newton polish with the exact
  Hessian, built one Hessian-vector product per free hyperparameter
  (~2.3x the gradient's memory; seconds on 2,000 Si configs).
  `map_convergence.json` records the predicted gain of another Newton step
  (converged: <= 1e-3 nats) and the final |gradient|; an unconverged
  MAP logs `WARNING: MAP did not converge` (`--strict`: an error). Adam
  (`--opt adam`, the CLI default in ace-jax <= 0.2.1) left GAP-18 Si about 10⁷ nats
  short of the optimum at 500 steps.
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
  - Mixed bulk + big cells: `--batch-pack auto` (default) packs batches by an
    atom budget (the largest config) instead of padding every batch to
    C x the biggest cell; `on` forces it, `off` keeps fixed groups of C.
- **POPS.** `--uq pops` changes only the uncertainty. The mean is pinned to the
  BLR mean, and the ridge is selected per quantity by CRPS on a training
  hold-out (`--pops-ridge auto`; `blr` or a number fixes it).
- **ARD `forces_std` is computed on request, not on every call.** A plain
  `atoms.get_forces()` does not compute it; call
  `calc.get_property("forces_std", atoms)` (reuses the cached E/F/stress), or
  pass `forces_std_every_call=True` — costly for per-step MD on big cells. Only
  the force uncertainty is calibrated (`ard.json` `tempered_quantities: ["F"]`);
  energy and virial variances are the uncalibrated posterior ones.
  `ACECalculator(model, posterior=...)` raises `ValueError` if the posterior
  doesn't match the model (basis size, species count, element list, or a mean
  that is not the model's coefficients, i.e. a posterior from another fit), and
  `RuntimeError` unless `jax_enable_x64` is on. `forces_std` holds the whole
  cell's force design rows, ~N·3·L·8 bytes.
- **ARD properties.** `--ard-variance sandwich` (default) is the PRESS jackknife shape
  (`--ard-press exact|block`). `forces_cov` is always the full 3x3; `forces_q_mahal` exists for
  `--force-shape aniso` (the default; `--force-shape iso` gives the spherical radius, and for aniso
  `forces_q` is the largest semi-axis q_g sqrt(lambda_max(M)) while `forces_q_mahal` is the Mahalanobis
  radius q_g). `forces_std`/`forces_cov` use a per-group rms scale on the PRESS jackknife
  shape (`--ard-cluster-size`, `--force-shape`); `forces_q` is the per-group conformal
  radius at `--ard-coverage` (default 0.9); `forces_group` names each atom's group.
  `--ard-variance kappa` keeps the A⁻¹ shape and also gets the per-group scales.
- **Coverage is conditional on exchangeability.** It holds for atoms exchangeable with their
  group's calibration configurations. For a new regime (cracks, interfaces), run
  `aj calibrate --model M --posterior P --data labelled.xyz --out new_posterior.npz` on labelled
  cells like it, then use `posterior="new_posterior.npz"` / `aj eval --posterior new_posterior.npz`.
  `--out` is a file. Default: groups where the new cells have >= `--ard-n-min` (20; ⌈(1−α)/α⌉ if
  larger) configurations use them alone; with fewer, the new scores only pool with the stored
  hold-out, so a few cells barely move that group. `--append` pools in every group, `--replace` uses
  the new cells alone in every group. The labelled cells must not be training configurations;
  `aj calibrate` does not check.
- **An unattainable coverage gives `forces_q = inf`.** A pool needs ⌈(1−α)/α⌉ configurations (9 at
  0.9, 99 at 0.99) for a finite conformal radius; below that the fit logs a WARNING and `q` stays
  infinite (stored as the string "inf" in `ard.json` / `posterior.npz`).
- **`support_ok = False` marks candidates for labelling** (`forces_support`; `aj eval --posterior P
  --per-atom out.xyz --support`; `--no-ard-support` at fit time skips it). `--ard-support-features
  normalised` builds the reference on the unit-norm descriptor plus log-norm channels per body order
  (default `raw`). In 0.2.0 and 0.2.1 the calculator computed `support_q` at the wrong level (the α, not the
  1 − α, quantile), so its `support_ok` flags almost never fired.
- **Very big cells: `--shape-path committee`** (`aj eval`, `aj calibrate`; `ACECalculator(...,
  shape_path="committee")`) gives the same `forces_std`/`forces_q`/`forces_cov` without the whole
  cell's design rows (arrays ~N·3·r·8 bytes, r the shape factor's rank, instead of N·3·L·8). At 3–4k
  atoms the default rows path is better (6.4 GB, ~20 s on an A100-40GB at L = 15k).
  `shape_tau=`/`shape_rank=` (Python) truncate the shape factor: rankings survive, coverage does not
  (the scales are fitted at full rank), so use a truncated shape for ranking only.
- **Schema-2 posteriors serve only the old scalar `forces_std`.** The new properties raise and ask
  you to refit with `--uq ard`. The validation of the new scales has been run
  (`bench/defect_uq/results/2026-10-03_rev2_acceptance.md`): the default (aniso + transfer
  exponent) meets every coverage target; a posterior from `aj calibrate` is specific to the
  regime it was calibrated on (crack-calibrated: tip 0.90, in distribution 0.87). User docs:
  <https://acesuit.github.io/ace-jax/howto/force-uncertainty/>.
- **First `aj basis` of a new basis shape** calls the compiled coupling
  library (`ace-jax-coupling`, milliseconds) to build the coupling table, then
  caches it in `~/.cache/ace-jax/coupling`. Later runs do not need the library.
- **Exit status.** `aj` exits 0 on success. Treat any non-zero exit as a
  failure and read the last lines of stderr.
