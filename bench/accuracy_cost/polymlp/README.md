# Polynomial MLPs (pypolymlp) in the Si accuracy-cost benchmark

This directory documents how Seko's polynomial machine-learning potentials enter the silicon accuracy-cost
benchmark of ace-jax (`docs/dev/accuracy-cost-spec.md`, results in `docs/dev/accuracy-cost-results.md`). It is a
first pass under a fixed protocol. **PRs that change the protocol or the fits are welcome.**

## What is compared

Every family in the benchmark is fitted to the same training split of the GAP-18 silicon database, scored on the
same test split and on 21 quantities of interest (QoIs) against DFT, and timed in LAMMPS on one CPU node.
No family gets per-system hyperparameter optimisation in this pass.

## Protocol

| Item | Choice |
|---|---|
| Code | pypolymlp 0.22.1 (conda-forge), `lammps-polymlp-package` 046eaa4 (VERSION 0.3.4) |
| Data | `$ACEJAX_AC_ROOT/data/si_accuracy_cost/train.xyz` (2,227 configurations), `test.xyz`; split hash in `split.json` |
| Cutoff | 5.0 Å (every family) |
| Energies | `atomic_energy` = the isolated-atom DFT energy, -158.54496821 eV; the calculator adds it back |
| Model ladder | fourteen points at 5 Å: eight of the grid search of `pypolymlp/utils/grid_search`, six of model type 2 (`sweep.yaml`, family `polymlp`) |
| Ridge alpha | pypolymlp's own selection over 10^linspace(-8, 1, 10), on a validation split of *train* |
| Validation split | 10 % of each configuration type, seed 0 (`fit_polymlp.val_split`; indices in each run's `val_split.json`) |
| Weights | one dataset per configuration type, weight sigma_F,default / sigma_F from the GAP-18 force sigmas |
| Virials | fitted for the configurations that have one (`include_stress` per dataset) |
| Scoring | `evaluate.py` (test set) and `bench/si_qoi/score.py`, through `fit_polymlp.calculator` (float64) |

Notes on the choices:

- **The test set never enters pypolymlp.** pypolymlp keeps the alpha with the lowest weighted RMSE on the
  datasets passed as "test". Here those are the validation split. The coefficients are the train-only solution
  at that alpha.
- **Alpha grid.** The grid search in pypolymlp uses 10^linspace(-4, 1, 6). Validation selected its smallest
  value in a first fit, so the grid continues to 1e-8. `status.json` records `alpha_at_edge` for every size.
- **Weights.** pypolymlp has one scalar weight per dataset. It multiplies the energy, force and stress
  residuals of that dataset. pypolymlp's own row weights stay on (`mlp_dev/core/utils_weights.py`): energy rows
  by energy tier, force components above 1 eV/Å as 1/|F|, stress rows 0.1 with large values damped.
- **Stress convention.** pypolymlp's stress is the virial sum r ⊗ f in eV per cell (xx yy zz xy yz zx). The
  benchmark's calculator converts it to ASE's stress. `PolymlpASECalculator` returns it unconverted.
- **Effective size.** Each `status.json` has `design_rank`: the numerical rank of the weighted training design
  matrix with unit-norm columns (singular values above 1e-10 of the largest), next to the coefficient count.

## Timing: no 128-rank row

The benchmark's CPU protocol is 128 MPI ranks × 1 thread on a 4,096-atom cell. `pair_style polymlp` cannot run
it: the package README says "MPI is not currently available", and `pair_polymlp.cpp` indexes its per-atom arrays
by atom tag with the local atom count as their size, so a decomposed run would read and write out of bounds.
The rows recorded instead are one rank with 128 OpenMP threads (`sulis-cpu-128x1-polymlp.jsonl`; each row has
`ranks: 1, threads: 128`) and one rank with one thread (`sulis-cpu-1x1-polymlp.jsonl`). There is no GPU code.

The LAMMPS gate compares `pair_style polymlp` with pypolymlp's evaluator on three rattled 64-atom cells. They
agree to 2.7e-14 eV/atom and 8.1e-13 eV/Å or better at every size.

## Results (2026-11-12)

Full discussion: `docs/dev/accuracy-cost-results.md`, section "Polynomial MLPs". E = energy RMSE
(meV/atom), F = force RMSE (eV/Å) on the 248 test configurations; QoI = the benchmark's QoI error; times in µs
per atom-step on a 128-core EPYC 7742 node, 4,096 atoms.

| Size | Coefficients | Design rank | Alpha | E | F | QoI | 1 rank × 128 threads | 1 × 1 |
|---|---|---|---|---|---|---|---|---|
| m3g6-l44 | 141 | 135 | 1e-7 | 92.2 | 0.1347 | 2.13 | 4.45 | 71.2 |
| m3g8-l442 | 300 | 260 | 1e-5 | 40.8 | 0.1191 | 1.75 | 5.04 | 99.8 |
| m3g9-l882 | 747 | 701 | 1e-8 | 31.4 | 0.1074 | 1.21 | 14.27 | 461.7 |
| m4g6-l442 | 864 | 768 | 1e-8 | 34.4 | 0.1101 | 1.23 | 4.55 | 80.8 |
| m4g9-l442 | 1782 | 1629 | 1e-7 | 14.6 | 0.1063 | 0.96 | 5.53 | 122.6 |
| m4g8-l842 | 3536 | 3401 | 1e-6 | 13.4 | 0.1252 | 0.98 | 11.36 | 276.0 |
| m4g9-l882 | 4797 | 4569 | 1e-3 | 13.9 | 0.0956 | 1.01 | 14.96 | 532.2 |
| m4g8-l1282 | 6984 | 6776 | 1e-3 | 11.8 | 0.0922 | 0.82 | 21.51 | 750.6 |
| m2g6-l44 | 7380 | – | 1e-2 | 11.0 | 0.0966 | 1.10 | 7.94 | 360.9 |
| m2g6-l84 | 10584 | – | 1e-2 | 9.4 | 0.0953 | 0.74 | 13.02 | 554.9 |
| m2g8-l84 | 18720 | – | 1e-2 | 9.8 | 0.0994 | 0.66 | 19.46 | 921.9 |
| m2g9-l84 | 23652 | – | 1e-1 | 8.5 | 0.0965 | 0.62 | 45.51 | 1261.9 |
| m2g8-l124 | 25424 | – | 1e-1 | 8.6 | 0.1009 | 0.54 | 61.21 | 1503.4 |
| m2g9-l124 | 32130 | – | 1e-1 | 8.6 | 0.0973 | 0.54 | 122.40 | 2320.6 |

Two things to know when reading it:

- **m4g8-l842's alpha.** Validation rated 1e-6 and 1e-3 within 0.3 % of each other and kept 1e-6. On the test set
  the 1e-6 model has F = 0.125 and the 1e-3 model F = 0.101 (a diagnostic, not a selection; the row is unchanged).
  At small alpha the forces degrade and the weighted validation RMSE hardly shows it.
- **hcp.** The two models with `gtinv_maxl` [8, 8, 2] put hcp 37–42 meV/atom too low. Their neighbours are within
  10 meV/atom. This is most of the reason the QoI error does not fall between 1,782 and 4,797 coefficients.

## Commands

```bash
# environment (pypolymlp has no wheels; its sdist's build requirement `eigen` is an unrelated PyPI package)
micromamba create -p /storage/eng/essswb/venvs/ac-polymlp -f bench/accuracy_cost/envs/polymlp/environment.yml
PY=/storage/eng/essswb/venvs/ac-polymlp/bin/python

cd bench
$PY -m accuracy_cost.fit_polymlp --count                      # coefficients per sweep entry
$PY -m accuracy_cost.fit_polymlp m3g6-l44 m3g8-l442 ...      # fit: runs/accuracy_cost/polymlp/<size>/polymlp.yaml
$PY -m accuracy_cost.evaluate polymlp m3g6-l44 ...            # test errors -> results/metrics.jsonl
$PY -m accuracy_cost.qoi_models polymlp                       # register with the QoI framework; prints names and tests

# LAMMPS on Sulis: build, then gate and time
sbatch bench/scaling/envs/sulis-polymlp-build-cpu.sbatch
bash   bench/scaling/envs/sulis-polymlp.sh env                # pypolymlp for the gate (login node)
sbatch bench/accuracy_cost/envs/sulis-cpu-polymlp.sbatch                          # 1 rank x 128 threads
sbatch --export=ALL,LAYOUT=1x1 bench/accuracy_cost/envs/sulis-cpu-polymlp.sbatch  # 1 rank x 1 thread
```

A smoke run beside the real one: set `ACEJAX_AC_RUNS` and `ACEJAX_AC_RESULTS` to scratch directories.

## Seko's repository Si models (looked up 2026-11-12)

The Polynomial MLP Repository has three Si entries: `Si-2024-06-05`, `Si-2024-06-10-hybrid` and `Si-2026-03-31`
(https://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/element/index.html). The settings below are from the model
files of the convex-hull models that have files. The cheapest hull models have no files, so their settings are
not known.

| | repository | this ladder |
|---|---|---|
| Features | `gtinv` in every model; no pair-only model | `gtinv` |
| Cutoff | 6.0 or 8.0 Å (hybrid second model: 4.0 or 5.0 Å) | 5.0 Å (the benchmark's rule) |
| Model type | 2 with `gtinv_order` 3 (most models); 4 with `gtinv_order` 4 | 3 and 4 with `gtinv_order` 3 and 4; 2 with `gtinv_order` 3 (added 2026-11-13) |
| `max_p` | 2 | 2 |
| `gtinv_maxl` | type 2: [8, 4] or [12, 4]; type 4: [8, 4, 2] | [4, 4] to [12, 8, 2] |
| Gaussians | centres 0 to cutoff − 1 Å, width 1.0: 10 or 13 (2024), 7 to 14 (2026) | 6, 8 or 9 |
| Penalty grid | 1e-4 to 1e3, 15 values (`reg_alpha_params -4 3 15`) | 1e-8 to 10, 10 values |
| Coefficients | 6,512 (type 4, 6 Å) to 77,420 (type 2, 8 Å) | 141 to 32,130 |
| Hybrid | 2026 models with files are all sums of two models (second: type 4, order 3, [4, 4] or [8, 8]) | none |

Repository errors on its own dataset (13,264 structures in 2024, 74,085 in 2026): 4.1 to 1.9 meV/atom and
0.066 to 0.050 eV/Å (2024); 4.4 to 2.4 meV/atom and 0.058 to 0.046 eV/Å (2026 models with files). These are
not comparable with this benchmark's test set.

The first eight sizes of this ladder have no type 2 model, which is the form of the repository's accurate Si
models (a polynomial of all invariants at order 3, with high `l` for the order-2 invariants). The six `m2*` sizes
add it at 5.0 Å: `gtinv_maxl` [4, 4], [8, 4] and [12, 4] with 6, 8 or 9 Gaussians. They have the lowest QoI
error of the family (0.54) and its highest cost. The design rank was not computed for them. The repository's
cutoff (6 or 8 Å) and its hybrid models are not followed: the benchmark fixes the cutoff at 5.0 Å.

## What a PR could change

- The model ladder (`sweep.yaml`): other grid points, hybrid models, a cutoff other than 5 Å as a separate family.
- The weights, the alpha grid and the selection rule (`fit_polymlp.py`), for example a rule that sees forces.
- An MPI-capable pair style, which would give the 128 × 1 row.

## Licences

pypolymlp and lammps-polymlp-package: BSD-3-Clause. Nothing from either is copied into this repository.
