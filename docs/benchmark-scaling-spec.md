# Benchmark scaling plots: spec

Status: agreed 2026-09-25. Phase A is ready to implement; Phase B waits for PRs
#2–#5 to be merged and #6/#7 to be rebased onto `main`.

## Goal

Produce a final, reproducible set of scaling plots for the docs showing
ace-jax's speed and memory against ML-PACE and MACE:
- on CPU and GPU;
- used standalone (Python) and inside LAMMPS;
- for two representative systems at three model sizes.

The plots should support honest conclusions: where ace-jax is faster or
slower, by how much, and up to what system size each code fits.

## Matrix

| axis | values |
|---|---|
| codes | ace-jax `PACEModel` (`.yace`) · ace-jax `ACEModel` (`.npz`) · ML-PACE (`pair_style pace` CPU, `pace/kk` GPU) · MACE (PyTorch) |
| modes | **standalone** (one force call from Python) · **LAMMPS** (MD steps) |
| systems | **SiGe**: random 50/50 alloy on a diamond lattice (a = 5.54 Å) · **Cantor**: equiatomic random CrMnFeCoNi on an fcc lattice (a = 3.59 Å) |
| model sizes | 3 per code family: small / medium / large (below) |
| system sizes | about 256 atoms upward in factors of 2: to 32k atoms on CPU, and until out of memory on GPU (reported as the largest size that fits) |
| hardware | CPU: moriarty (32-core Xeon); the local Mac (Apple Silicon) as an optional spot check · GPU: moriarty RTX A4500 (20 GB), Modal A100-80GB |
| precision | f64 for every code; f32 as well for ace-jax and MACE (ML-PACE is f64 only) |

### Models

- **ace-jax `PACEModel` and ML-PACE share the same `.yace` files**, so that
  comparison is like-for-like. They are generated with pyace (`pace_ref/`
  tooling) at `number_of_functions_per_element` ≈ 100 / 500 / 2000 per system,
  with random coefficients, for timing only.
- **ace-jax `ACEModel` (linear ACE baseline):** exported with
  `julia/export_model.jl`, sized by `ACE_ORDER` / `ACE_TOTALDEGREE`. The target
  basis sizes are about 100 / 700 / 2800 functions per element, matching the
  existing `si_s69` / `si_m710` / `si_l2849` ladder. Fitted models are used
  where they exist (the Cantor d6 model from `bench/acegp_cantor`); otherwise
  random coefficients.
- **MACE:** MACE-MP-0b2 small / medium / large, plus MACE-MH-1, via
  `mace-torch`. All four cover the seven elements, and no training is needed.
  The b2 revision is used rather than the original MP-0 checkpoints because
  Symmetrix only exports models whose first interaction is the plain or
  density block, and the original MP-0 (like MH-1) uses the residual one.
  MP-0b2 keeps the same size ladder and r_max = 5 A.
  MH-1 extends the MACE model-size axis beyond MP-0. It is multi-head: both
  sides evaluate the `omat_pbe` head, pinned explicitly. Symmetrix can't
  export MH-1 (residual first block), so its LAMMPS rows are recorded as
  `unsupported` and only its standalone rows are timed.
- **Phase B:** the ace-jax production model (linear + species embedding +
  density embedding) at the same three sizes. See "Phase B".

Random-coefficient models are fine for timing. For MD in LAMMPS they are kept
stable by starting from the ideal lattice with zero velocity, using a
0.1 fs timestep and a short timed segment. Any run that loses atoms is
reported as a failure, never as a timing.

## Metrics (one JSON row per case)

- **Timing:**
  - `call_s`: median ASE-calculator call (energy + forces + stress), including
    the neighbour list, for every code. MACE builds its graph inside the call,
    so this is the like-for-like standalone number. ace-jax also reports
    `force_s` (the jitted model call alone) and `nlist_s`.
  - `step_s`: LAMMPS loop time per MD step, from a 200-step timed run after a
    50-step warm-up.
  - `atom_steps_per_s = n_atoms / (step_s or call_s)`.
  - `compile_s`: JIT / `jax.export` / `torch.compile` time.
- **Memory:** `peak_bytes` (device peak; peak RSS on CPU), and whether the
  case ran out of memory.
- **ace-jax's automatic choices:** `layout` and `edge_a_kind` as chosen by
  `auto`, so the plots show where dense switches to sparse.
- **Provenance:** code, mode, model, model size (number of basis functions or
  parameters), system, n_atoms, n_edges, device, dtype, versions and git SHAs
  of ace-jax / lammps-jax / LAMMPS / pace / mace, and the date.

## Parity gates (checked before timing, per environment)

- ace-jax `PACEModel` vs ML-PACE on the same `.yace` and structure:
  |ΔE|/atom ≤ 1e-6, |ΔF| ≤ 1e-5 (the spline-grid gap measured in Task 9).
- ace-jax standalone vs ace-jax in LAMMPS (lammps-jax): |ΔE|/atom ≤ 1e-10 and
  |ΔF| ≤ 1e-9 in f64.
- MACE standalone vs MACE in LAMMPS: |ΔE|/atom ≤ 1e-6.

A case that fails its gate is not timed, and the failure is recorded.

## New components

1. **`ace_jax.export.lammps`**: an ace-jax → lammps-jax exporter.
   - `export_lammps(model, meta, path, *, max_atoms, edges_per_atom, dtype, layout="auto")`
     is a thin adapter over `lammps_jax.export.export_model` (units `metal`,
     `n_species`, cutoff = the model's cutoff). Both model classes work, since
     both are `EdgeSiteModel`s.
   - The energy function maps lammps-jax's `(positions, species, graph)` onto
     `energy_from_positions`. For `layout="dense"` it regroups the packed edge
     buffer into (n, K) blocks inside the exported function. Edges arrive
     grouped by atom, so this is a scatter of the edge vectors only. Then it
     calls `site_energies_dense`.
   - `layout="auto"` decides at export time with `estimate_a_bytes` for the
     bundle's capacity (`max_atoms`, `edges_per_atom`).
   - The exported graph must contain no `custom_call` (an existing guard,
     extended to the bundle).
   - `lammps-jax` becomes an optional extra, `ace-jax[lammps]`.
   - Tests: bundle round-trip, and bundle vs `ACECalculator` parity on CPU.
2. **`bench/scaling/`:**
   - `structures.py`: deterministic supercells for both systems at every size,
     with seeded species assignment, cached as `.xyz`.
   - `models.py`: builds or collects the model files for every family and size
     into `bench/scaling/models/`. They are git-ignored, with a manifest
     recording each file's provenance.
   - `run_standalone.py`: one case per process (so peak memory is per case).
     ace-jax runs `ACECalculator` with its defaults; MACE runs `MACECalculator`
     with cuequivariance and `torch.compile` where available.
   - `run_lammps.py`: writes the input for each case and runs LAMMPS. Pair
     styles are `jax/kk` (ace-jax via lammps-jax), `pace` / `pace/kk` (ML-PACE)
     and `symmetrix/mace` (MACE through wcwitt's Symmetrix, Kokkos build).
     Neighbour skin 1.0 Å, rebuilt as needed.
   - `parity.py`: runs the gates above.
   - `sweep.py`: expands the matrix for a host, skips cases already in the
     results file, and appends JSONL to `bench/scaling/results/<host>.jsonl`.
     It stops increasing N for a code/model once a size runs out of memory.
   - `plot.py`: results → `docs/figs/scaling_*.png` and the tables in
     `docs/benchmarks.md`.
3. **Environments:**
   - **moriarty:** LAMMPS ≥ 10 Sep 2025 (required by lammps-jax) with KOKKOS
     (CUDA, `Kokkos_ARCH_AMPERE86`, plus OpenMP for CPU), ML-PACE, Python, and
     Symmetrix. Symmetrix patches the tree with `pair_symmetrix/install.sh`
     and needs C++20, CMake ≥ 3.27, GCC ≥ 11, `SYMMETRIX_KOKKOS=ON`, and
     `SYMMETRIX_SPHERICART_CUDA=ON` for GPU. The same tree also gets the
     lammps-jax plugin with CPU and GPU PJRT, plus a venv with jax[cuda12],
     torch, cuequivariance, mace-torch, symmetrix (for model extraction) and
     pyace. This lives in
     the shared home, so every node sees it.
   - **Modal A100-80GB:** the same build in a new image (`Kokkos_ARCH_AMPERE80`),
     in its own app, `bench/scaling/modal_app.py`. It stays separate from
     `bench/pace_modal/run.py`, whose stable-branch LAMMPS image is kept for
     reproducing the Task 9 results.
   - The build recipes are documented in `bench/scaling/README.md`.

CPU runs use the whole node. LAMMPS styles run with MPI ranks equal to the
core count (32 on moriarty). ace-jax standalone on CPU runs as one process
using XLA's thread pool, and MACE uses 32 torch threads. ace-jax in LAMMPS is
GPU-only: lammps-jax ships just `pair_style jax/kk`, which needs KOKKOS built
with CUDA, so there are no CPU LAMMPS rows for ace-jax (the parity gate
records them as `unsupported`).
The threading actually used is recorded in each row.

## Plots (`docs/benchmarks.md`)

1. **Throughput vs N** (log–log atom-steps/s): one panel per
   system × device, one line per code/model; solid = standalone, dashed =
   LAMMPS.
2. **Throughput vs model size** at a fixed N (8,192 atoms; 2,048 on CPU),
   per device.
3. **Memory:** peak memory vs N, and the largest N that fits per device, with
   ace-jax's dense→sparse switch marked.
4. **Precision:** f32 vs f64 for ace-jax and MACE.
5. **Table:** compile / export times.

Every figure is generated by `plot.py` from the committed JSONL files; none
are edited by hand.

## Phases

**Phase A (now, independent of the other PRs):**
- the exporter;
- structures, models (PACE `.yace`, ACE linear, MACE-MP-0b2 / MH-1), environments and
  parity;
- full sweeps on moriarty CPU, moriarty A4500 and Modal A100;
- the plots and a draft of `docs/benchmarks.md`.

**Phase B (after #2–#5 are merged and this branch is rebased on `main`):**
- **Evaluation-side density embedding.** Density embedding (`fit/density.py`:
  d_z · ssqrt(η_z · B_pair)) is fit-side only today.
  - `ACEModel`'s readout gains η (NZ × n_pair), d (NZ) and the smooth signed
    square root, stored in the `.npz`.
  - It sits in the layout-independent readout, so the sparse and dense paths
    and the exporter all carry it.
  - It's tested against fit-side predictions to 1e-10.
  - Where possible it shares the embedding-function code with `PACEModel`'s
    Finnis–Sinclair readout rather than adding a third copy.
- **Species embedding:** the factorised radial that `ACEModel` already
  evaluates (`spline_factorised`), exported via `ACE_EMBEDDING`.
- **Sweep:** export the production model (linear + species embedding +
  density embedding) at the three sizes for both systems, and run only those
  rows. The plots include it without re-running anything else. The ACE-linear
  baseline stays as a reference line.

## Non-goals

- GP-model (`GPCalculator`) timings.
- Dense-layout support inside lammps-jax's C++. The dense regrouping happens
  inside the exported JAX function.
- Multi-GPU or multi-node scaling.
- Accuracy comparisons between codes. This is about speed and memory; parity
  gates exist only to ensure like-for-like physics.
- Exporting ACEpotentials models to ML-PACE (sub-project 2 of
  `docs/pace-yace-spec.md`).

## Cost and effort

- **Engineering:** the exporter (with tests) and the LAMMPS / lammps-jax /
  Symmetrix builds on two platforms. The rest is harness code.
- **GPU time:** the moriarty A4500 is free. The Modal A100 sweep is about
  2–4 GPU-hours, so tens of dollars.
- **Phase B** is about a day of work once the merges land.

## Open risks

- **The lammps-jax plugin build on moriarty** needs LAMMPS with the Kokkos
  precision layer (≥ 10 Sep 2025) and a matching PJRT plugin. If the build
  fails, the LAMMPS-mode ace-jax rows are deferred rather than faked.
- **MACE in LAMMPS through Symmetrix** needs a C++20 / CMake ≥ 3.27 / GCC ≥ 11
  toolchain alongside CUDA (nvcc_wrapper). If moriarty's modules can't provide
  it, the Symmetrix rows are built on Modal only, and the plots say so.
  Standalone MACE stays PyTorch (`mace-torch`).
- **Random-coefficient MD stability.** This is mitigated as above, and fitted
  models are used where available.
