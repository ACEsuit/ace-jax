# YE3T in the Si accuracy-cost benchmark

This directory documents how the Young–E(3) tensor product code of Goff and Thompson (arXiv:2609.31895) enters
the silicon accuracy-cost benchmark of ace-jax (`docs/dev/accuracy-cost-spec.md`, results in
`docs/dev/accuracy-cost-results.md`). It is a first pass under a fixed protocol. **PRs that change the protocol
or the fits are welcome.**

## Code

| Repository | Commit | Licence | Used for |
|---|---|---|---|
| `ye3t-equivariance/ye3t` | f8ff96d | BSD-3-Clause | representation compiler, C++ runtime |
| `ye3t-equivariance/ye3t-methods` | 6b85c54 | BSD-3-Clause (its `native/` C++: GPL-2.0-or-later) | fits, ASE calculator, LAMMPS export |
| `ye3t-equivariance/ye3t-lammps` | 06100f9 | GPL-2.0 | `pair_style ye3t` |

Nothing from these repositories is copied into ace-jax. The LAMMPS package is built on Sulis by its own installer.

## Three entries

### 1. YE3T-symmetric as an evaluator: family `ace_cheb`, code `ye3t-sym` (timing only)

The existing linear ACE fits (`ace` o3d12, o4d12, o4d14, o4d16, o4d20) are run by `pair_style ye3t`.

`pair_style ye3t` reads ML-PACE `.yace` files, but only with `radbasename: ChebExpCos`, without core repulsion,
and with every m-tuple written together with its negation. The benchmark's exact export of these fits uses
tabulated radials, which the reader refuses. So `ace_to_ye3t.py` writes each fit once more:

- The product basis is exact.
- Each radial the products read, and the pair term, is projected by least squares on 48 ChebExpCos functions
  (decay 1.5, taper 0.001 Å). The projection error is 5e-11 of the radial's maximum.
- The file differs from the fit by at most 1.1e-10 eV/atom and 1.6e-8 eV/Å (`runs/ace_cheb/<size>/convert.json`).

Stock ML-PACE reads the same file. The timing rows are therefore three codes on one file: `ye3t-sym`
(`pair_style ye3t`, default `block_policy direct`), `mlpace` (`pair_style pace`, the recursive evaluator) and
`mlpace-product` (`pair_style pace product`). The gate compares each with ace-jax on the source fit.

`block_policy auto` needs a plan sidecar that only ye3t-methods' model workflow writes. It is not used here.
The authors report that `auto` resolved to the direct schedule on all their fitted models.

### 2. Fitted density models: families `ye3t_sym_fit` and `ye3t_mix_A`

Both use density-projected inputs (u = A), a fully symmetric scalar parent (λ = (N), L = 0, even parity), and
ChebExpCos radials. `ye3t_sym_fit` sets `young_kappa: symmetric_only`. `ye3t_mix_A` sets `young_kappa: all_valid`
(mixed-permutation intermediates).

**These two are the same model.** ye3t resolves both policies to the same space for a density source
(`ye3t/representation_config.py`, `count_fixed_content`: "symmetric_only selects the same complete space as
all_valid for those carriers"; `ye3t_methods/config.py`: "commutative density resolves all_valid Young blocks to
physically trivial block sectors"). The descriptor rows are identical bit for bit (`fit_ye3t.same_rows` returns
0.0), and the fitted coefficients are identical. `ye3t_mix_A` is fitted at three sizes to show this.
It is not drawn on the figures: its points are those of `ye3t_sym_fit`.

### 2b. The same descriptors under the benchmark's loss: family `ye3t_sym_wfit`

This family is not the authors' protocol. It has the catalogues of `ye3t_sym_fit` and ye3t-methods' own
`LinearModel.fit` (assembly and ridge solve, penalty 1e-8). The loss is the one the benchmark's other linear
families use: per-configuration-type weights (`weights_acejax.json`), energy rows × w_E/√N, force rows × w_F, and
virial rows (eV) × w_V/√N. ye3t-methods has one global weight per quantity. The weights are therefore applied to
what its fit loop receives: each structure's rows and targets are multiplied by that structure's weights, and the
global weights stay 1. That is the same least-squares problem. The stress slot carries the virial (rows −V × stress
row); a structure without a virial has zero rows and a zero target. Code: `fit_ye3t.py`, "Weighted family". The
exported model is an ordinary YE3T file: evaluation, the LAMMPS gate and the timing are unchanged.

It is shown as a second line beside `ye3t_sym_fit`, so the protocol rows stay as they were. Whether a weighted
loss of this kind is how the authors would fit silicon is a question for them.

### 3. Fitted tagged models: family `ye3t_mix_T`

The tagged basis (u = T): ye3t-methods' tagged Cauchy image with shifted-Jacobi radials, ranks 1–8.

## Protocol for the fits

| Item | Choice | In the paper (Sec. IV.3) |
|---|---|---|
| Regression | ridge, penalty 1e-8, ye3t-methods' normal-equation solve | the same |
| Ranks, caps | ranks up to 8, nmax ≤ 6, lmax ≤ 4 per rank (`sweep.yaml`) | "ranks 1–8 with nmax=6 and lmax=4" |
| Cutoff | 5.0 Å (every family) | tuned with DAKOTA (Si: 4.757 Å) |
| Radial decay | 0.7799, the value of the authors' Si models | tuned with DAKOTA |
| Reference energy | isolated-atom DFT energy, fixed (`fit_E0: False`) | residual to a fitted ZBL core |
| Weights | energy 1, forces 1, every structure | energy and force weights from a search |
| Virials | not fitted | no stress weight recorded |
| Data | the benchmark's GAP-18 split (2,227 training configurations) | the mlearn Si set |

What is deliberately not done:

- **No DAKOTA tuning** of the cutoff and decay, and no search over weights and penalty. No family in the
  benchmark is tuned per system. If a PR adds tuned YE3T fits, the `ace` ladder is tuned under the same budget.
- **No ZBL overlay.** The other families have no added core either.
- **No per-configuration-type weights.** `LinearModel.fit` has one global weight per quantity, so the GAP-18
  sigmas that the other families use are not applied. `status.json` records `per_config_type_weights: false`.
  The loss is the sum of squared total-energy errors (one row per structure) and squared force components, all
  at weight 1. ye3t-methods does contain the paper's weighted loss (per-atom energy rows, forces normalised per
  structure, target scaling, per-group weights: `portable_refit.py`, `atomistic/linear_statistics.py`), but that
  path accepts only the authors' nickel model. This weighting matters: see "Weighting diagnostic" below.
- **No virials.** The stress target needs a stress on every structure, and a third of the training set has none.

## Limits met

- **Catalogue size.** `Basis.from_config` refuses a catalogue with more than 4,096 candidate contents
  (`counts(maximum_fixed_contents=4096)`). The full nmax = 6, lmax = 4 catalogue at rank 4 has 14,286. A sweep
  entry with `max_fixed_contents` raises that default.
- **Fit speed.** The density fit computes each structure's rows on one thread. `fit_ye3t.py` runs the same row
  function in worker processes and returns the rows to the unchanged fit loop in order. The coefficients are
  identical to a serial fit.
- **Design rows.** Each run's descriptor list is `labels.json`.

- **Row memory.** A density row block needs 8.9 GB (205 descriptors), 30 GB (490) and 109 GB (1,048) for one
  217-atom training cell. The 2,080- and 4,577-descriptor sizes were dropped for that reason. `fit_ye3t.py`
  sizes its worker pools from a memory budget (`mem_budget_gb`), one size class of cells at a time.
- **Tagged solve.** At penalties 1e-8 to 1e-5 the tagged normal equations are not positive definite (column norms
  from 12 to 5e12). `ye3t_mix_T` therefore takes the smallest penalty of 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 that
  solves: 1e-4. No error, on any set, enters that choice. `status.json` records the grid and the failures.
- **Tagged gate time.** ye3t-methods' calculator takes about 30 minutes for the gate's three 64-atom cells.

## Results (2026-11-12)

Full tables and discussion: `docs/dev/accuracy-cost-results.md`, section "YE3T".
E = energy RMSE (meV/atom), F = force RMSE (eV/Å) on the 248 test configurations; QoI = the benchmark's QoI error;
times in µs per atom-step on a 128-core EPYC 7742 node, 4,096 atoms.

| Family | Size | Descriptors | E | F | QoI | 128 ranks × 1 | 1 × 1 |
|---|---|---|---|---|---|---|---|
| `ye3t_sym_fit` | r3-34 | 34 | 100.9 | 0.1852 | 4.17 | 0.12 | 9.35 |
| `ye3t_sym_fit` | r3-110 | 110 | 49.2 | 0.1338 | 2.43 | 0.16 | 12.36 |
| `ye3t_sym_fit` | r4-205 | 205 | 25.1 | 0.1182 | 1.86 | 0.33 | 25.11 |
| `ye3t_sym_fit` | r4-490 | 490 | 21.5 | 0.1024 | 1.42 | 0.70 | 46.20 |
| `ye3t_sym_wfit` | r3-34 | 34 | 56.8 | 0.2311 | 2.61 | 0.12 | 9.94 |
| `ye3t_sym_wfit` | r3-110 | 110 | 26.1 | 0.1756 | 1.69 | 0.16 | 13.13 |
| `ye3t_sym_wfit` | r4-205 | 205 | 19.5 | 0.1468 | 1.27 | 0.42 | 26.45 |
| `ye3t_sym_wfit` | r4-490 | 490 | 11.6 | 0.1278 | 1.12 | 0.70 | 48.20 |
| `ye3t_mix_T` | t8-284 | 284 | 15.4 | 0.1213 | 2.15 | 3.28 | 237.1 |

`ye3t_sym_wfit` against `ye3t_sym_fit`, same descriptors and cost: QoI error 2.61, 1.69,
1.27, 1.12 instead of 4.17, 2.43, 1.86, 1.42. Energy RMSE is about half at the largest size (11.6 vs
21.5 meV/atom) and force RMSE is higher (0.128 vs 0.102 eV/Å), the trade the ACE fits also make. Linear ACE has
QoI error 1.03 at 223 coefficients and 0.76 at 350.

`pair_style ye3t` on the linear ACE fits, 1 × 1: 28.3, 30.0, 45.1, 74.6, 326.1 (o3d12 to o4d20). It is 1.7–4.5
times faster than ML-PACE's product evaluator and 1.4–1.6 times faster than its recursive evaluator on the same
file. The 128 × 1 time of o4d20 (25.9) is anomalous and not understood; the smaller fits give 0.35 to 1.39.

### r8-1048: a failed solve, not a row

The 1,048-descriptor density fit (ranks 1 to 8) finished in 5.5 hours, and its model is wrong.

| model | test E (meV/atom) | test F (eV/Å) | training E, every 8th frame | training F, every 8th frame |
|---|---|---|---|---|
| r4-490 | 21.5 | 0.1024 | 21.4 | 0.1009 |
| r8-1048 | 267.4 | 0.4070 | 257.8 | 0.3991 |

It is as wrong on its training data as on the test set, so this is not overfitting. The r8-1048 caps contain
the r4-490 caps, so an exact least-squares solution cannot fit the training set worse. The errors are in every
configuration type (training energies: dia 30, fcc 436, sp 1,107 meV/atom).

The probable cause is conditioning, and that is not confirmed. ye3t-methods' general fit solves the normal
equations of the unscaled descriptor columns at penalty 1e-8. The tagged model failed in a related way (not
positive definite below 1e-4). The authors' weighted fit for their Ni model standardises the columns first.

The model is not in `results/metrics.jsonl`, the tables or the figures. Its numbers are in
`r8-1048-failed-solve.json`, with its timing: 2.88 µs/atom-step on 128 ranks and 89.2 on one core (r4-490: 0.70 and
46.2), which is valid as the cost of this descriptor set. Nothing was changed to make it work: a larger penalty or scaled columns is a
protocol decision for the authors.

`fit_ye3t` now keeps the density rows (`rows_dir`: `runs/accuracy_cost/ye3t_rows/<key>`, one file per structure).
The key is the catalogue and the structures, not the penalty. The r8-1048 rows of this fit were not kept, so the
first retry computes them again (5.5 hours, 3 to 4 GB on disk); later solves of that catalogue read them back.

### Weighting diagnostic

The fitted density models have higher energy and QoI errors than linear ACE of the same size, and lower force
errors. The errors are largest for small-cell bulk phases (test energy RMSE of r4-490: bcc 37, fcc 42, st12 26,
sp2 92 meV/atom; linear ACE o4d14: 3, 1, 2, 7). To test the loss, the r4-205 rows from ye3t-methods' own row
function were solved again under the benchmark's weights (`ext/logs/probe10.py`; no benchmark row changes).
Test energy RMSE (meV/atom) / force RMSE (eV/Å):

| Test set | Protocol loss | Energy 30/√N, force 1 | GAP-18 weights | Linear ACE o3d12 (223) |
|---|---|---|---|---|
| all | 25.1 / 0.118 | 13.5 / 0.135 | 18.1 / 0.144 | 15.9 / 0.158 |
| fcc | 46.8 / 0.129 | 12.4 / 0.102 | 16.4 / 0.096 | 10.1 / 0.158 |
| sp2 | 81.0 / 0.128 | 6.0 / 0.186 | 6.4 / 0.129 | 21.5 / 0.284 |
| bcc | 87.2 / 0.075 | 44.1 / 0.150 | 42.1 / 0.124 | 6.8 / 0.180 |
| st12 | 28.7 / 0.150 | 28.9 / 0.170 | 23.5 / 0.184 | 8.7 / 0.199 |
| hcp | 11.7 / 0.073 | 32.9 / 0.081 | 31.3 / 0.102 | 11.9 / 0.142 |

The protocol column reproduces the benchmark row exactly. Under ACE-like weights the 205-descriptor model has
about the same overall energy and force errors as linear ACE at 223 coefficients, so much of the gap is the loss.
It is not all of it: bcc, st12 and hcp stay 2 to 6 times worse than ACE. The QoIs were not recomputed for these
refits. A fit with a weighted loss is the first thing a PR could change.


### Loss diagnostic with QoIs (2026-11-13)

The QoI error of the fitted density models is partly the loss. The r4-205 descriptors (ye3t-methods' own energy,
force and stress rows) were solved again under four losses, with ye3t-methods' solver (normal equations, penalty
1e-8). Each solution was run through the QoI tests (`ext/logs/probe12.py`; models `YE3Tdiag_r4-205_*`; no
benchmark row changes). Virials are in eV with weight w_V/√N, as ace-jax fits them.

| Loss | E (meV/atom) | F (eV/Å) | virial (meV/atom) | QoI error |
|---|---|---|---|---|
| Protocol: E and F, weight 1 (the benchmark row) | 25.1 | 0.118 | 231 | 1.85 |
| Protocol + virials | 29.5 | 0.122 | 86 | 2.29 |
| ACE weights, no virials | 18.1 | 0.144 | 146 | 1.59 |
| ACE weights + virials (ace-jax's loss) | 19.5 | 0.147 | 97 | 1.27 |
| Linear ACE o3d12 (223 coefficients) | 15.9 | 0.158 | | 1.03 |

Signed QoI errors in units of the tolerance:

| QoI | Protocol | ACE weights + virials | ACE o3d12 |
|---|---|---|---|
| bulk modulus | 0.72 | 0.00 | −0.53 |
| c12 | 2.99 | 0.85 | 0.80 |
| c44 | −1.85 | −1.79 | 0.17 |
| ΔE fcc | −5.59 | 0.84 | −1.97 |
| ΔE bcc | −10.41 | −7.28 | −3.08 |
| ΔE bc8 | −2.23 | −0.94 | −0.13 |
| ΔE st12 | 1.68 | −0.28 | −1.70 |
| ΔE hcp | −0.10 | 2.27 | 0.14 |

- The loss accounts for 1.85 → 1.27. The energy weights and the virials are both needed: virials alone make the
  QoI error worse (2.29), because the unweighted energies then lose more.
- What remains against ACE o3d12 (1.27 vs 1.03) is mostly c44 and bcc. The loss does not move c44.
- The `ye3t_sym_fit` rows keep the protocol loss. The weighted loss is now its own family, `ye3t_sym_wfit`
  (section 2b), at all four sizes; its r4-205 row is the "ACE weights + virials" row above.

## Commands

```bash
bash bench/accuracy_cost/envs/ye3t/build-venv.sh              # torch (CPU), ye3t, ye3t-methods from clones
PY=/storage/eng/essswb/venvs/ac-ye3t/bin/python

cd bench
python -m accuracy_cost.ace_to_ye3t                           # ace ladder -> runs/accuracy_cost/ace_cheb/<size>/model.yace
$PY -m accuracy_cost.fit_ye3t --count ye3t_sym_fit            # descriptors per sweep entry
$PY -m accuracy_cost.fit_ye3t ye3t_sym_fit r3-34 r3-110 ... --procs 24
$PY -m accuracy_cost.evaluate ye3t_sym_fit r3-34 ...          # test errors -> results/metrics.jsonl
$PY -m accuracy_cost.qoi_models ye3t_sym_fit                  # register with the QoI framework

# LAMMPS on Sulis: build (pair_style ye3t and stock ML-PACE in one binary), then gate and time
sbatch bench/scaling/envs/sulis-ye3t-build-cpu.sbatch
sbatch bench/accuracy_cost/envs/sulis-cpu-ye3t.sbatch                          # 128 ranks x 1 thread
sbatch --export=ALL,LAYOUT=1x1 bench/accuracy_cost/envs/sulis-cpu-ye3t.sbatch  # 1 rank x 1 thread (the paper's setting)
```

A smoke run beside the real one: set `ACEJAX_AC_RUNS` and `ACEJAX_AC_RESULTS` to scratch directories, and
`ACEJAX_AC_YE3T_SUBSET=40` to fit every 40th training frame.

## What a PR could change

- The catalogues (`sweep.yaml`): other rank, nmax and lmax schedules, feature selection as in the paper.
- The fit settings (`fit_ye3t.py`): weights, penalty, radial decay, a ZBL reference, tuned cutoffs as a
  separate family.
- A plan sidecar for the `ace_cheb` files, to time `block_policy auto`.
- A CUDA row: ye3t-lammps has `ye3t/kk`; this pass built the CPU style only.
