# Accuracy-cost benchmark on silicon: results

The accuracy-matched cost benchmark of `accuracy-cost-spec.md` (plan: `accuracy-cost-plan.md`): every model
family fitted to the GAP-18 silicon dataset (2,227 train / 248 test configurations, `bench/accuracy_cost/data.md`),
scored on test force RMSE and 21 silicon QoIs against CASTEP, and timed in LAMMPS on Sulis. The user-facing
summary is the Performance page (`docs/user/benchmarks.md`).

## Figures

- `bench/si_qoi/results/rmse_qoi_cost.png`: force error, QoI error and the cheapest model reaching the target,
  against A100 cost, CPU-node cost and fitted parameters. The docs copy
  (`docs/user/assets/benchmarks/si_accuracy_cost.png`) is the same figure with `--no-note`.
- `rmse_qoi_cost_wl.png`: the same for the wL ladder (linear ACE at wL 1, 1.75, 2, 2.5 and order 5).
- `qoi_heatmap.png`: the per-QoI normalised errors behind the QoI score (`qoi_heatmap.py`).
- `memory.png`: peak GPU memory per model (`plot_memory.py`).

## Target and cost

Target: test force RMSE ≤ GAP-18's (0.0825 eV/Å on this split) and QoI error ≤ 0.5. The QoI error is the
geometric mean over the 21 QoIs of |error| / tolerance (`bench/si_qoi/score.py`: 10% on elastic constants,
stacking faults and decohesion stress, 5% on the (100) surface energy, 10 meV/atom on phase energies, 0.1 eV on point defects, 0.2 THz rms on phonons),
each clipped to [0.05, 10]. GAP-18 itself scores 0.38.

Cheapest model reaching the target, µs/atom-step, float64, LAMMPS:

| model | parameters | force RMSE (eV/Å) | QoI error | A100 | CPU node |
|---|---|---|---|---|---|
| linear ACE o4d20, ace-jax (lammps-jax) | 5,476 | 0.0816 | 0.41 | 5.65 | 50.0 |
| linear ACE o4d20, ML-PACE (tabulated radials) | 5,476 | 0.0816 | 0.41 | 4.12 | 5.39 |
| linear ACE o4d20, ACEpotentials.jl (trim library) | 5,476 | 0.0816 | 0.41 | – | 10.8 |
| MACE c16L0, Symmetrix | 31,152 | 0.0699 | < 0.5 | 1.93 | 2.36 |
| MACE c16L0, Symmetrix-XL | 31,152 | 0.0699 | < 0.5 | 1.23 | 2.19 |
| MACE c16L1, Symmetrix | 40,272 | 0.0579 | < 0.5 | 5.36 | 8.58 |
| MACE c16L1, Symmetrix-XL | 40,272 | 0.0579 | < 0.5 | 1.47 | 2.87 |

Not reached in their ladders (best QoI error, at that model's force RMSE):

| family | best QoI | model | force RMSE |
|---|---|---|---|
| PACE (pacemaker) | 0.60 | f2971 | 0.0707 |
| ACE, learned radials | 0.66 | o4d16 | 0.1002 |
| PACE-shaped ACE (fixed / learned radials) | 0.66 / 0.73 | md18 / md12 | 0.107 / 0.110 |
| GAP refit (glue / no glue / ACE's E:F balance) | 0.35 / 0.36 / 0.40 | s9000 | 0.088 / 0.088 / 0.085 |
| MACE, one layer | 1.38 | c32L0n1 | 0.102 |
| NEP (GPUMD) | 0.66 | n120b | 0.103 |

The GAP refits meet the QoI target but not GAP-18's force error; PACE meets the force target with fewer
parameters than linear ACE (f800: 761 parameters, 0.081 eV/Å) but not the QoI target.

## Observations

- **ML-PACE runs ace-jax's linear ACE fastest on the CPU.** Exported exactly (`write_yace`, tabulated radials;
  `ml-pace-tabulated-radials.md`), o4d20 runs at 5.39 µs/atom-step on 128 ranks against 50.0 through the
  lammps-jax CPU backend with the DAG product basis, and 3.98 for PACE f3200. On the A100 the gap is smaller
  (4.12 vs 5.65).
- **ACEpotentials.jl's compiled export** (the same fits as ACEpotentials models, `juliac --trim` libraries in
  LAMMPS, ACEpotentials PR 309) is the fastest CPU code for linear ACE up to o4d14: 0.10 µs/atom-step at o3d8
  and 0.59 at o4d14, against 0.85 for ML-PACE (o4d14) and 3.4-7.9 for lammps-jax. It ties ML-PACE at o4d16
  (1.42 vs 1.41) and is 2x slower at o4d20 (10.8 vs 5.39), still 4.6x faster than lammps-jax there. Parity
  against ace-jax on the npz: |dE|/atom <= 6.5e-12 eV, |dF| <= 1.2e-11 eV/Å over the ladder. CPU only.
- **NEP** is the cheapest family on the CPU (0.55-3.0 µs/atom-step, n10-n120c; n120b 2.0) and on the A100
  (0.04-0.22 µs/atom-step in GPUMD, mixed precision; n120b 0.16), but plateaus near 0.10 eV/Å: 0.106 (n80), 0.104 (n120) and 0.103 (n120b) after 300k generations, best QoI
  error 0.66 (n120b), with the point-defect energies 19-31% low. That is NEP's published accuracy on this
  dataset, not a fitting shortfall: Fan et al. report 0.107 eV/Å (NEP1, arXiv:2107.08119) and 0.093 (the NEP4
  retrain, arXiv:2206.07605), and GPUMD's shipped `Si_2022_NEP4_5body.txt` scores 0.0998 eV/Å on our test
  split (it was trained on all of GAP-18, test configurations included, so that is optimistic). What did not
  help:
  - more neurons (n80 -> n120: 0.106 -> 0.104) or the descriptor at GPUMD's caps with 4- and 5-body terms
    (n120b 0.103; n120c, the widest basis, 0.107 at 200k generations);
  - cutoffs 6/4 and 6/5 Å and λ1 = λ2 = 0.01 (30k-generation screens on n120b: 0.118-0.130, against n120b's
    own 0.118 at 30k);
  - the published recipe (nlit: n_max 10 8, l_max 4 2 1, 50 neurons, λ = 0.05, full batch, population 50):
    0.106 eV/Å at 100k generations, still improving slowly, but its QoI error is 1.25 (c44 17% low, vacancy
    30% low). Plotted as its own series.

  GPUMD squares the per-structure `weight`, so the n10-n120c fits weighted each config type by
  (σ_default/σ)⁴ instead of the intended (σ_default/σ)² (fixed in `fit_nep.py`). The published recipe with
  the intended weights (nlitw) scores 0.108 eV/Å and QoI error 1.22 at 100k, against 0.106 and 1.25 without
  weights, so the weighting does not explain the plateau. No paper we found distils MACE into NEP with
  reported force errors; the favourable reports are on large, narrow MD datasets.
- **Symmetrix-XL** is 1.1-3.6x faster than Symmetrix for the same MACE models, most at L = 1 on the GPU.
- **The DAG product basis** (`aa_dag="greedy"`, #90) gives ace-jax's fastest CPU rows at o4d16 and o4d20; it
  also cuts the compiled temporary memory (o4d20: 3.9 GB to 0.64 GB on one core).
- **Learned radials** do not reach the target in their ladder (to o4d16): best QoI error 0.66, against 0.41
  for linear ACE at o4d20.

## How the rows were timed

- GPU: one A100, 32,768 atoms. CPU: one Sulis node (2 x 64-core AMD EPYC 7742), 4,096 atoms, all cores, ranks
  bound to cores, each model's faster of 128 x 1 and 64 x 2 MPI ranks x threads; GAP at 32 ranks.
- Every cost is a LAMMPS row that passed its parity gate (64 atoms against the model's reference evaluation)
  in the same job; `make_cost.py` takes each model's fastest such row, float64 only. One-layer MACE and, on
  the GPU, Symmetrix MACE c64L1 have float32 timings only and are left out of the cost rows.
- ace-jax: lammps-jax bundles (`export_lammps`; CPU: `platform="cpu"`, `neighbour_capacity(procs=)`, o4d16 and
  o4d20 with `aa_dag="greedy"`). ML-PACE on the GPU: LAMMPS develop ec02ed0 (Stan Moore's 2026-08 `pace/kk`
  rewrite; about 2x faster than the 10Sep2025 patch release). Linear ACE run by ML-PACE: that tree with
  `LOCAL_ML-PACE` = v.2025.12.4.p1 + the tabulated-radial patch (`bench/scaling/envs/sulis-pacetab.sh`).
  Linear ACE run by ACEpotentials.jl: each `aj fit` model rebuilt as an ACEpotentials v0.10.2 `ace_model`
  (`bench/scaling/julia/build_trim_acejax.jl`: coefficients mapped through the AA basis, gated to 1e-10 against
  ace-jax before compiling), exported with `for_library=true` and compiled by `juliac --trim=safe` on a compute
  node (Julia 1.12.6; `accuracy_cost/build_trim.py`, `envs/sulis-cpu-trim.sbatch`), run by the PR 309
  `pair_style ace` plugin in the same LAMMPS (`bench/scaling/envs/sulis-acepot.sh`); 128 x 1 ranks only.
- NEP: LAMMPS develop ec02ed0 + NEP_CPU af615ce's USER-NEP (`pair_style nep`, float64, 128 x 1 ranks;
  `bench/scaling/envs/sulis-nep.sh`, `envs/sulis-cpu-nep.sbatch`), gated against calorine's CPUNEP (same
  code, < 2e-14 eV/Å). On the A100, GPUMD's own MD code (`envs/sulis-gpu-nep.sbatch`), which is mixed
  precision only (float descriptors and network): the one non-float64 cost in the figure, labelled there
  (`make_cost.py --allow nep-gpumd:mixed`). Test errors and QoIs use CPUNEP, in float64. Fits: GPUMD v5.9.1
  `nep`, NEP4, one A100 each (`envs/sulis-nep.sbatch`, chunks of 100k generations resumed from
  `nep.restart`); n80 and larger 300k generations (n120c and nlitw stopped at 200k and 100k).
- QoIs: compressed ("short and fat") hcp is left out of the score (outside the training data; it measures
  extrapolation); only models with every QoI computed are plotted. Single fits per point.
- GAP's parameter count is its sparse-point coefficients; the kernel also stores the sparse descriptors.

## Reproduce

Cost files from the timing rows (float64 only), then the figures:

```bash
R=bench/accuracy_cost/results/sulis; RR=bench/si_qoi/results
python bench/si_qoi/make_cost.py --dtype float64 --out $RR/cost_sulis_cpu.json \
    $(ls $R/sulis-cpu*.jsonl | grep -v -e smoke -e ace_yace -e pacetab-check -e "ace_tab\.jsonl" -e ace_tab2)
python bench/si_qoi/make_cost.py --dtype float64 --out $RR/cost_sulis_gpu.json \
    bench/accuracy_cost/results/sulis-gpu.jsonl bench/accuracy_cost/results/sulis-gpu-ace-splined.jsonl \
    $R/sulis-gpu-ace-splined-v2.jsonl $R/sulis-gpu-v2.jsonl $R/sulis-gpu-mace_xl.jsonl \
    $R/sulis-gpu-ace_tab3.jsonl $R/sulis-gpu-pace-dev.jsonl $R/sulis-gpu-nep.jsonl --allow nep-gpumd:mixed
ROWS=(--row "A100 GPU, µs/atom-step|$RR/cost_sulis_gpu.json|$RR/cost_sulis_gpu.codes.json"
      --row "CPU node (128 cores), µs/atom-step|$RR/cost_sulis_cpu.json|$RR/cost_sulis_cpu.codes.json")
python bench/si_qoi/plot_cost.py "${ROWS[@]}"
python bench/si_qoi/plot_cost.py "${ROWS[@]}" --wl --out $RR/rmse_qoi_cost_wl.png
python bench/si_qoi/plot_cost.py "${ROWS[@]}" --no-note --out docs/user/assets/benchmarks/si_accuracy_cost.png
```

Fits, QoIs and timings themselves: `bench/accuracy_cost/` (`sweep.yaml`, `timing.py`, the Sulis job scripts in
`envs/`); the fitted models are on /storage, not in the repository.

## Open

- ACE + GP was dropped from the families pending the GP LML cost study (issue #60).
- GAP has a CPU row at 32 ranks only.
