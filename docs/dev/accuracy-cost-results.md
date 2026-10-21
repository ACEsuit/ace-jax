# Accuracy-cost benchmark on silicon: results

The accuracy-matched cost benchmark of `accuracy-cost-spec.md` (plan: `accuracy-cost-plan.md`): every model
family fitted to the GAP-18 silicon dataset (2,227 train / 248 test configurations, `bench/accuracy_cost/data.md`),
scored on test force RMSE and 21 silicon QoIs against CASTEP, and timed in LAMMPS on Sulis. The user-facing
summary is the Performance page (`docs/user/benchmarks.md`).

## Figures

- `bench/si_qoi/results/rmse_qoi_cost.png`: force error, QoI error and the cheapest model reaching the target,
  against A100 cost, CPU-node cost and fitted parameters. The docs copy
  (`docs/user/assets/benchmarks/si_accuracy_cost.png`) is the same figure with `--no-note`; on the page it is the
  fallback for the interactive version (`si_accuracy_cost.json`,
  `bench/si_qoi/plot_cost_web.py`: the force and QoI panels, families toggled from the legend or by preset).
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
| linear ACE o4d20, ace-jax (lammps-jax) | 5,476 | 0.0816 | 0.41 | 3.12 | 50.5 |
| linear ACE o4d20, ML-PACE (tabulated radials) | 5,476 | 0.0816 | 0.41 | 4.13 | 5.31 |
| linear ACE o4d20, patched ML-PACE (proposed upstream) | 5,476 | 0.0816 | 0.41 | 2.42 | 2.89 |
| linear ACE o4d20, ACEpotentials.jl (trim library) | 5,476 | 0.0816 | 0.41 | – | 10.8 |
| linear ACE o4d20, ace-jax C++ kernel (lammps-jax FFI) | 5,476 | 0.0816 | 0.41 | – | 3.29 |
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

- **ML-PACE runs ace-jax's linear ACE fastest on the CPU at o4d20.** Exported exactly (`write_yace`, tabulated radials;
  `ml-pace-tabulated-radials.md`), o4d20 runs at 5.31 µs/atom-step on 128 ranks against 50.5 through the
  lammps-jax CPU backend with the DAG product basis (3.29 with ace-jax's C++ kernel, next), and 3.98 for PACE f3200. On the A100, ace-jax with the DAG
  is faster than stock ML-PACE (3.12 vs 4.13) and behind patched ML-PACE (2.42).
- **ace-jax's C++ kernel** (`ace_jax.eval.ffi`, `export_lammps(ffi=True)`: the lammps-jax CPU bundle with the XLA
  program replaced by one XLA FFI call per step, radials tabulated in r) brings ace-jax to ML-PACE's speed on the
  CPU: 10-23x faster than the XLA bundle, timed one after the other on the same node (job 2257991, node136;
  µs/atom-step, 128 ranks):

  | model | C++ kernel | XLA bundle | speed-up |
  |---|---|---|---|
  | o3d8 | 0.137 | 3.17 | 23x |
  | o3d10 | 0.160 | 1.98 | 12x |
  | o3d12 | 0.295 | 3.33 | 11x |
  | o4d12 | 0.236 | 3.47 | 15x |
  | o4d14 | 0.411 | 5.09 | 12x |
  | o4d16 | 0.738 | 7.50 | 10x |
  | o4d20 | 3.29 | 49.7 | 15x |

  The XLA column reproduces the plotted ace-jax line (`acejax-auto` rows, other nodes) to within ~5%. Against
  the other CPU codes (other nodes, so ~20% either way): at o4d20 the kernel (3.29) is between patched ML-PACE
  (2.89) and stock ML-PACE (5.31) and 3.3x faster than ACEpotentials.jl (10.8); at o4d12-o4d16 it ties or beats
  patched ML-PACE (0.236 / 0.411 / 0.738 against 0.305 / 0.578 / 0.791); below o4d14 ACEpotentials.jl stays
  fastest (o3d8 0.104 against 0.137). Parity: every gate (LAMMPS against `ACECalculator` with the same radial
  table, three rattled 64-atom cells) within 8.6e-10 eV/Å; the largest, o3d8, is roundoff amplified by that fit's
  pair and many-body forces cancelling (each ~765 eV/Å, net max |F| 2.8; separately the kernel matches the XLA
  program to 1.7e-12 relative). The kernel is single-threaded per rank, so 128 x 1 is its best layout (64 x 2
  would leave half the cores idle). It needs lammps-jax's plugin to register FFI handlers for the CPU platform:
  abhijeetgangan/lammps-jax#3 (0.2.0 registers them for CUDA only; #4 fixes that repository's unrelated `mace`
  CI failure under jax 0.12).
- **Patched ML-PACE** (ICAMS/lammps-user-pace#31: on the CPU the recursive-evaluator speed-ups, no unused
  dB_flatten clear, register-carried DAG passes, the force pass summed over n first; on the GPU the pace/kk
  prototype kernels of `pace_ref/fast`, flat-offset ComputeRho and a segmented ComputeWeights, `PACE_KK_RHOW=3`)
  against stock ML-PACE timed after each size on the same node (µs/atom-step):

  | model | CPU, 128 ranks: patched / stock | speed-up | A100: patched / stock | speed-up |
  |---|---|---|---|---|
  | o4d12 | 0.305 / 0.520 | 1.71x | 0.257 / 0.277 | 1.08x |
  | o4d14 | 0.578 / 0.848 | 1.47x | 0.385 / 0.449 | 1.17x |
  | o4d16 | 0.791 / 1.366 | 1.73x | 0.558 / 0.828 | 1.48x |
  | o4d20 | 2.89 / 5.31 | 1.84x | 2.42 / 4.13 | 1.70x |

  Sulis CPU nodes differ by up to ~20% for the same build (the patched o4d20 ran at 3.50 on node032 and 2.89 on
  node150), so only same-node pairs are compared. One core, o4d20: 2.1x. On the A100 the stock ComputeRho /
  ComputeWeights are issue-bound on integer index arithmetic; flat offsets remove it, and the segmented backward
  cuts ComputeWeights' global atomics 16x (ncu: 23.9 -> 15.1 ms at o4d20). A fused one-team-per-atom kernel
  gained nothing.
- **ACEpotentials.jl's compiled export** (the same fits as ACEpotentials models, `juliac --trim` libraries in
  LAMMPS, ACEpotentials PR 309) is the fastest CPU code for linear ACE up to o4d14: 0.10 µs/atom-step at o3d8
  and 0.59 at o4d14, against 0.85 for ML-PACE (o4d14) and 3.4-7.9 for lammps-jax. It ties ML-PACE at o4d16
  (1.42 vs 1.37) and is 2x slower at o4d20 (10.8 vs 5.31), still 4.6x faster than lammps-jax there. Parity
  against ace-jax on the npz: |dE|/atom <= 6.5e-12 eV, |dF| <= 1.2e-11 eV/Å over the ladder. CPU only.
- **NEP** is the cheapest family on the A100 (0.04-0.22 µs/atom-step in GPUMD, mixed precision; n120b 0.16)
  but not on the CPU (0.55-3.0 µs/atom-step, n10-n120c; n120b 2.0), where PACE reaches NEP n120's force error
  (0.104 eV/Å) at 0.42 µs/atom-step (f100) against 1.86. It plateaus near 0.10 eV/Å: 0.106 (n80), 0.104 (n120) and 0.103 (n120b) after 300k generations, best QoI
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
- **The DAG product basis** (`eval/aa_dag.py`) is never slower than the chain form and from ~6,000 chain-form
  multiplies per site clearly faster, so `aa_dag="auto"` (the default of `ACECalculator` and `export_lammps`)
  takes it from `AUTO_MIN_CHAIN_MUL` = 4,000. Every ace-jax model of the figure, timed in both forms one after
  the other on the same node (µs/atom-step, chain / DAG; all parity-gated):

  | model (chain multiplies) | CPU node | speed-up | A100 | speed-up |
  |---|---|---|---|---|
  | ace o3d8-o3d12 (113-1,013) | 2.09-3.41 / 2.05-3.35 | 0.99-1.02x | 0.31-0.52 / 0.31-0.51 | 0.99-1.01x |
  | ace o4d12 (2,150) | 3.46 / 3.36 | 1.03x | 0.550 / 0.532 | 1.03x |
  | PACE-shaped md10 (2,087) | 7.55 / 7.13 | 1.06x | 1.013 / 1.008 | 1.01x |
  | PACE-shaped md12 (6,207) | 12.8 / 10.3 | 1.24x | 1.666 / 1.628 | 1.02x |
  | ace o4d14 (7,095) | 8.00 / 5.19 | 1.54x | 0.664 / 0.591 | 1.12x |
  | PACE-shaped md18 (19,141) | 32.1 / 15.5 | 2.07x | 2.744 / 2.552 | 1.08x |
  | ace o4d16 (21,934) | 19.3 / 7.57 | 2.55x | 1.207 / 0.923 | 1.31x |
  | ace o4d20 (175,021) | 136 / 50.5 | 2.70x | 5.73 / 3.12 | 1.84x |

  Learned radials match linear ACE at each (order, degree), and PACE-shaped ACE with learned radials matches
  the fixed-radial rows. The figure plots each ace-jax model in the form `auto` picks (centre dot: the DAG), so
  the switch falls at o4d14 for linear ACE and learned radials and at md12 for PACE-shaped ACE. The DAG also cuts
  the compiled temporary memory (o4d20: 3.9 GB to 0.64 GB on one core).
- **Learned radials** do not reach the target in their ladder (to o4d16): best QoI error 0.66, against 0.41
  for linear ACE at o4d20.

## How the rows were timed

- GPU: one A100, 32,768 atoms. CPU: one Sulis node (2 x 64-core AMD EPYC 7742), 4,096 atoms, all cores, ranks
  bound to cores, each model's faster of 128 x 1 and 64 x 2 MPI ranks x threads; GAP at 32 ranks.
- Every cost is a LAMMPS row that passed its parity gate (64 atoms against the model's reference evaluation)
  in the same job; `make_cost.py` takes each model's fastest such row, float64 only. One-layer MACE and, on
  the GPU, Symmetrix MACE c64L1 have float32 timings only and are left out of the cost rows.
- ace-jax: lammps-jax bundles (`export_lammps`; CPU: `platform="cpu"`, `neighbour_capacity(procs=)`), each model
  timed in the chain form and with `aa_dag="greedy"` on the same node (`envs/sulis-{cpu,gpu}-dagpair.sbatch`,
  ace-jax 6debdc9); `accuracy_cost/dag_auto.py` keeps, per model, the form `aa_dag="auto"` picks
  (`results/dag_chain_multiplies.json`, from `dag_chain_multiplies.py`) in `sulis-{cpu-128x1,gpu}-acejax-auto.jsonl`,
  and `make_cost.py --source` takes the ace-jax rows from those files alone.
  ML-PACE on the GPU: LAMMPS develop ec02ed0 (Stan Moore's 2026-08 `pace/kk`
  rewrite; about 2x faster than the 10Sep2025 patch release). Linear ACE run by ML-PACE: that tree with
  `LOCAL_ML-PACE` = v.2025.12.4.p1 + the tabulated-radial patch (`bench/scaling/envs/sulis-pacetab.sh`).
  Stock rows: `sulis-cpu-128x1-ace_tab5.jsonl` / `sulis-gpu-ace_tab5.jsonl`, timed in the same jobs as the patched
  ones. Patched ML-PACE: the same LAMMPS tree with `LOCAL_ML-PACE` = ICAMS/lammps-user-pace#31
  (jameskermode/lammps-user-pace tabulated-radials 97621a3: v.2025.12.4.p1 + tabulated radials + the speed-ups +
  summed_neighbour_A), and on the GPU the pace/kk patches `pace_ref/fast/0003`, `0005`, `0006`, `0007` run with
  `PACE_KK_RHOW=3` (`sulis-pacetab.sh cpufork` / `gpufork`; `envs/sulis-{cpu,gpu}-pacefork.sbatch` with
  `REDO=1 STOCK=1`); rows `sulis-{cpu-128x1,gpu}-ace_tabfork.jsonl`, plotted as their own line
  (`plot_cost.PATCHED_MLPACE`).
  Linear ACE run by ACEpotentials.jl: each `aj fit` model rebuilt as an ACEpotentials v0.10.2 `ace_model`
  (`bench/scaling/julia/build_trim_acejax.jl`: coefficients mapped through the AA basis, gated to 1e-10 against
  ace-jax before compiling), exported with `for_library=true` and compiled by `juliac --trim=safe` on a compute
  node (Julia 1.12.6; `accuracy_cost/build_trim.py`, `envs/sulis-cpu-trim.sbatch`), run by the PR 309
  `pair_style ace` plugin in the same LAMMPS (`bench/scaling/envs/sulis-acepot.sh`); 128 x 1 ranks only.
- ace-jax C++ kernel: the same LAMMPS, lammps-jax `cpu-fixes` + the FFI-platform fix (lammps-jax#3) built in its own
  tree, the kernel compiled on a compute node with `-O3 -march=znver2` against jax 0.11.2's FFI headers
  (`envs/sulis-cpu-ffi-build.sbatch`); ace-jax branch `jax-ffi`; `envs/sulis-cpu-ffi.sbatch` (env
  `sulis-cpu-lj-ffi.json`: `ACEJAX_EXPORT_FFI`, `LAMMPS_JAX_FFI_HANDLERS`) times each size with the kernel, then the
  XLA bundle on the same node (`sulis-cpu-lj-ffi-xla.json`). Rows: `sulis-cpu-128x1-ffi.jsonl` (the plotted
  line, `plot_cost.family_of`: rows marked `ffi`) and `sulis-cpu-128x1-ffi-xla.jsonl` (the same-node control, not
  plotted). Analytic radials splined at 1e-10 as the plotted ace rows (o3d8 unsplined), then tabulated.
- The N/8 check size (512 atoms, 4 per rank) is overhead-bound for every fast CPU code: the C++ kernel,
  patched ML-PACE and ACEpotentials.jl all cost 1.4-4.2x more per atom there than at 4,096 atoms, so their rows
  carry `not_plateau`. Per-step costs that do not scale with atoms (communication, neighbour lists) are still
  a visible share of their 4,096-atom times; a larger system would favour them further. The XLA bundle,
  ~10x slower per atom, is mostly on its plateau (its o3d10 and o4d16 rows are flagged).
- NEP: LAMMPS develop ec02ed0 + NEP_CPU af615ce's USER-NEP (`pair_style nep`, float64, 128 x 1 ranks;
  `bench/scaling/envs/sulis-nep.sh`, `envs/sulis-cpu-nep.sbatch`), gated against calorine's CPUNEP (same
  code, < 2e-14 eV/Å). On the A100, GPUMD's own MD code (`envs/sulis-gpu-nep.sbatch`), which is mixed
  precision only (float descriptors and network): the one non-float64 cost in the figure, labelled there
  (`make_cost.py --allow nep-gpumd:mixed`). Test errors and QoIs use CPUNEP, in float64. Fits: GPUMD v5.9.1
  `nep`, NEP4, one A100 each (`envs/sulis-nep.sbatch`, chunks of 100k generations resumed from
  `nep.restart`); n80 and larger 300k generations (n120c and nlitw stopped at 200k and 100k).
- Elastic constants: the bulk_diamond test runs with `SI_TESTS_ELASTIC_FMAX=1e-3` (silicon-testing-framework
  branch `fp32-optimisers`, `test-runner/tight_elastic.sh`), since 2026-10-16. matscipy's `fit_elastic_constants`
  otherwise relaxes each strained cell's internal coordinates only to ASE's default fmax 0.05 eV/Å, which left the
  relaxed-ion C44 0.23–0.89 of its tolerance low for every model (DFT 72.2 GPa; e.g. linear ACE o4d20 67.8 -> 70.6,
  GAP-18 66.7 -> 70.1). C11, C12 and B moved by at most 0.02 tolerance. QoI scores changed by -13% to +9%
  (median -2.6%); which models reach the target did not change. docs/dev/fp32-results.md.
- QoIs: compressed ("short and fat") hcp is left out of the score (outside the training data; it measures
  extrapolation); only models with every QoI computed are plotted. Single fits per point.
- GAP's parameter count is its sparse-point coefficients; the kernel also stores the sparse descriptors.

## Reproduce

Cost files from the timing rows (float64, plus the reduced-precision lines `plot_cost.REDUCED`, which keep their
own names), then the figures. The light-blue mixed-precision line is patched pace/kk (`PACE_KK_RHOW=3`) built with
`KOKKOS_PREC=mixed` (`sulis-pacetab.sh gpufork_mixed`, `envs/sulis-timing-fp32.sbatch`). The ace-jax float32 line
is `envs/sulis-timing-fp32-dot.sbatch F32_F32_F32` (job 2251905, 2026-10-18, with `aa_dag="auto"`; the earlier
chain-form rows are kept as `sulis-gpu-fp32-dot-F32_F32_F32.chain.jsonl`).

```bash
R=bench/accuracy_cost/results/sulis; RR=bench/si_qoi/results
grep -h '"code": "mlpace-mixed"' $R/sulis-gpu-fp32-o4d*.jsonl > $R/sulis-gpu-fp32-mixed.jsonl   # the mixed-precision line
(cd bench && python -m accuracy_cost.dag_auto)
python bench/si_qoi/make_cost.py --dtype float64 --source acejax-ace=sulis-cpu-128x1-acejax-auto --out $RR/cost_sulis_cpu.json \
    $(ls $R/sulis-cpu*.jsonl | grep -v -e smoke -e ace_yace -e pacetab-check -e "ace_tab\.jsonl" -e ace_tab2 \
      -e ace_tab3 -e dagpair -e ffi-xla)
python bench/si_qoi/make_cost.py --dtype float64 --source acejax-ace=sulis-gpu-acejax-auto --out $RR/cost_sulis_gpu.json \
    bench/accuracy_cost/results/sulis-gpu.jsonl bench/accuracy_cost/results/sulis-gpu-ace-splined.jsonl \
    $R/sulis-gpu-ace-splined-v2.jsonl $R/sulis-gpu-v2.jsonl $R/sulis-gpu-mace_xl.jsonl \
    $R/sulis-gpu-ace_tab5.jsonl $R/sulis-gpu-ace_tabfork.jsonl $R/sulis-gpu-pace-dev.jsonl $R/sulis-gpu-nep.jsonl $R/sulis-gpu-acejax-auto.jsonl \
    $R/sulis-gpu-fp32-mixed.jsonl $R/sulis-gpu-fp32-dot-F32_F32_F32.jsonl $R/sulis-gpu-grace.jsonl --allow nep-gpumd:mixed
ROWS=(--row "A100 GPU, µs/atom-step|$RR/cost_sulis_gpu.json|$RR/cost_sulis_gpu.codes.json"
      --row "CPU node (128 cores), µs/atom-step|$RR/cost_sulis_cpu.json|$RR/cost_sulis_cpu.codes.json")
python bench/si_qoi/plot_cost.py "${ROWS[@]}"
python bench/si_qoi/plot_cost.py "${ROWS[@]}" --wl --out $RR/rmse_qoi_cost_wl.png
python bench/si_qoi/plot_cost.py "${ROWS[@]}" --no-note --out docs/user/assets/benchmarks/si_accuracy_cost.png
# the docs page's interactive version of that figure (rendered by docs/user/javascripts/plotly-figures.js)
python bench/si_qoi/plot_cost_web.py "${ROWS[@]}"     # -> docs/user/assets/benchmarks/si_accuracy_cost.json
```

Fits, QoIs and timings themselves: `bench/accuracy_cost/` (`sweep.yaml`, `timing.py`, the Sulis job scripts in
`envs/`); the fitted models are on /storage, not in the repository.

## Open

- ACE + GP was dropped from the families pending the GP LML cost study (issue #60).
- GAP has a CPU row at 32 ranks only.

## GRACE-1L and GRACE-FS (2026-10-21)

gracemaker 0.6.2.1 from scratch (`bench/accuracy_cost/fit_grace.py`, sweep families `grace`, `grace_fs`, `grace_abl`):
cutoff 5 Å, float64 parameters, the `gracemaker -t` from-scratch defaults (Adam, 50k updates = 376 epochs, Huber
E16/F32/stress 128, batch 16) and GAP-18's per-type weights. Three seeds per size, on Sulis A100s; one departure from
gracemaker's defaults: the stress term is not normalised per batch (a batch without virials gives 0/0 = NaN); the
virial weights are pre-divided by their expected batch sum instead (`bench/accuracy_cost/data.md`). Timed in LAMMPS
from the yury-lysogorskiy/lammps `grace` branch: `grace/1l/kk` (A100) and `grace/1l/cpu/kk` (CPU node, 128 x 1),
`grace/fs/kk` / `grace/fs`, float64, seed 0 of each size, each gated at 64 atoms against gracemaker's TPCalculator
(1L ≤ 2.9e-10 eV/Å; FS ≤ 6.3e-9). Every GRACE-1L size keeps max_order 4: the Kokkos 1L kernel cannot load order 3.
The A100 build was checked (Release `-O3`, CUDA 12.8, `Kokkos_ARCH_AMPERE80`, device code `-arch=sm_80`), and
`grace/1l/kk`'s atom batch was swept (`chunksize`, job 2257996, `envs/sulis-gpu-grace-chunk.sbatch`, rows
`results/sulis/sulis-gpu-grace-chunk<N>.jsonl`, each parity-gated): the default 4096 is the best or within 3%,
and larger batches are slower for c64l4 (µs/atom-step, 32,768 atoms):

| chunksize | 4096 (default) | 8192 | 16384 | 32768 |
|---|---|---|---|---|
| c16l4 | 3.81 | 3.70 | 3.79 | 3.98 |
| c64l4 | 11.76 | 12.49 | 15.61 | 17.23 |

Mean over seeds (forces: half the min-max range); µs/atom-step:

| model | parameters | force RMSE (eV/Å) | liquid force RMSE | QoI error [min, max] | A100 | CPU node |
|---|---|---|---|---|---|---|
| GRACE-1L c8l3 | 14,121 | 0.0933 ± 0.0018 | 0.162 | 0.84 [0.79, 0.90] | 2.47 | 5.87 |
| GRACE-1L c16l3 | 34,769 | 0.0870 ± 0.0009 | 0.153 | 0.75 [0.67, 0.83] | 3.25 | 9.05 |
| GRACE-1L c16l4 | 55,505 | 0.0808 ± 0.0008 | 0.140 | 0.54 [0.50, 0.57] | 3.81 | 12.5 |
| GRACE-1L c32l4 | 171,937 | 0.0779 ± 0.0005 | 0.138 | 0.50 [0.43, 0.57] | 6.95 | 27.0 |
| GRACE-1L c64l4 | 445,793 | 0.0764 ± 0.0004 | 0.138 | 0.46 [0.45, 0.47] | 11.65 | 59.1 |
| GRACE-FS small | 3,432 | 0.122 ± 0.006 | 0.246 | 0.91 [0.79, 1.11] | 1.30 | 2.51 |
| GRACE-FS medium | 4,564 | 0.121 ± 0.004 | 0.240 | 0.99 [0.97, 1.02] | 1.75 | 2.93 |

For comparison (single fits): linear ACE o4d20 0.0816 eV/Å, liquid 0.180, QoI 0.39 (A100 3.12 ace-jax / 2.42 patched
ML-PACE, CPU 2.89 patched ML-PACE); MACE c16L0 0.0699, liquid 0.123, QoI 0.41 (A100 1.93 / 1.23 Symmetrix-XL, CPU 2.36).

**Outcome (the plan's table).** GRACE-1L reaches the target at c64l4 (the smallest size whose seed mean does; c32l4
misses by 0.001), and a lower liquid force error than linear ACE o4d20 (0.138 against 0.180),
but at 3.7-4.8x linear ACE's A100 cost and 20x its CPU cost (patched ML-PACE), and 6-9.5x MACE c16L0's (GPU) / 25x
(CPU). So: GRACE reaches the target at a higher cost than linear ACE, and small MACE is cheaper still: ace-jax's
linear ACE stays competitive, and MACE c16L0 is the benchmark to beat. GRACE-FS does not reach either target.

**Ablations** (one change each to c64l4, three seeds, accuracy only):

| ablation | force RMSE | liquid | QoI [min, max] | point-defect error / tolerance |
|---|---|---|---|---|
| c64l4 (base) | 0.0764 | 0.138 | 0.46 [0.45, 0.47] | 6.9 |
| linear readout | 0.0844 (+10%) | 0.161 (+17%) | 0.55 [0.51, 0.59] | 9.7 |
| max_order 3 | 0.0846 (+11%) | 0.148 (+7%) | 0.55 [0.49, 0.61] | 8.7 |
| radial MLP frozen at its initialisation | 0.0786 (+3%) | 0.147 (+7%) | 0.53 [0.51, 0.56] | 7.9 |

Each ablation loses the QoI target. The nonlinear readout carries most of the liquid gain (its loss gives back about
half of the gap to linear ACE's 0.180) and, with body order 4, the force gain; radial learning is worth 3% on forces,
in line with ace-jax's learned radials (5-7%). A linear-readout GRACE-1L (377k parameters) is less accurate than
linear ACE o4d20 (5,476): GRACE's learned basis does not substitute for the readout.

**Training length.** One c32l4 fit at 3x the updates (150k; `grace/c32l4_long_s0`) scores 0.0680 eV/Å (50k seeds:
0.0779 ± 0.0005), energy 2.57 meV/atom (3.93), QoI 0.45 (seeds 0.43-0.57), liquid unchanged (0.139). The 50k-update
default under-trains at least c32l4, so the ladder's force errors are upper bounds; the QoI and liquid conclusions
are within seed noise of the longer fit. The ladder was not refitted.

Costs: 21.3 A100-hours of fits (ladder 10.5, FS 1.2, ablations 9.7), plus about 4 for probes, the LAMMPS build and the
two timing jobs. Rows: `results/metrics.jsonl` (`grace*`, one per seed), QoI scores `results/grace_qoi.json`, timings
`results/sulis/sulis-{gpu,cpu-128x1}-grace.jsonl`, in the main cost files and figures (the reproduce block above; the
GRACE-only `cost_sulis_*_grace.json` and `rmse_qoi_cost_grace.png` of the first write-up are folded into them).
