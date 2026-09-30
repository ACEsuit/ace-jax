# Benchmark scaling harness

Produces the timings behind `docs/benchmarks.md`: ace-jax (`PACEModel` on
`.yace`, `ACEModel` on `.npz`), ML-PACE and MACE, standalone and inside
LAMMPS, on SiGe and Cantor at three model sizes each. Design:
`docs/benchmark-scaling-spec.md`; plan: `docs/benchmark-scaling-plan.md`.

| file | role |
|---|---|
| `structures.py` | SiGe diamond / Cantor fcc supercells, the atom-count ladder |
| `models.py` | builds and lists every model (`python models.py pace\|ace\|ace-learned\|mace`) |
| `run_standalone.py` | one standalone case (ace-jax calculator or MACE PyTorch) |
| `run_lammps.py` | one LAMMPS case: writes the input, runs, parses the timed segment |
| `parity.py` | the per-host parity gates that must pass before any timing |
| `sweep.py` | expands the case matrix for a host and runs it resumably |
| `plot.py` | figures and tables for `docs/benchmarks.md` |
| `envs/moriarty.sh`, `envs/sulis.sh`, `modal_app.py` | the environments |

## Models

`python bench/scaling/models.py pace|ace|mace` writes into `models/`
(git-ignored) and records each file in `models/manifest.json`.

- **PACE:** random-coefficient `.yace` files, 100 / 500 / 2000 functions per element.
- **ACE:** built with `julia/export_model.jl`, using `ACE_NOFIT=1` (random weights,
  timing only) and `ACE_JULIA="julia +1.11"`. Cantor also needs `ACE_R0=2.54`,
  because ACEpotentials has no default bond length for Cr–Ni.
- **MACE:** MACE-MP-0b2 small / medium / large and MACE-MH-1 (the `omat_pbe`
  head), plus a Symmetrix `.json` for each system.
  - Symmetrix only exports models whose first interaction block is the plain
    or density block. The original MP-0 checkpoints and MH-1 have a residual
    first block, which is why the b2 revision is used.
  - MH-1 therefore has no LAMMPS rows; its status is `unsupported`.
- **Learned-radial proxies** (`models.py ace-learned`, after `ace`): the medium ACE models
  on the analytic radial branch, `to_analytic(n_q=12)`, with the active `rnl_Wnlq` rows
  perturbed by 10% of each row's rms (seed 0) and marked `radial_learned`, as
  `ace_{SiGe,Cantor}_medium_learned.npz`. Two lines run them: `acejax-ace-learned`
  (`spline_tol="auto"`, splined at 1e-10, as deployed) and `acejax-ace-analytic`
  (`spline_tol=None`, exact). Rows record `spline_tol` and `splined`. Medium only.

## Environments

### moriarty (RTX A4500 + 32-core CPU)

```bash
rsync this checkout to ~/bench-scaling/ace-jax, then
bash ~/bench-scaling/ace-jax/bench/scaling/envs/moriarty.sh     # all steps, idempotent
```

The script builds two LAMMPS trees, because no single version builds both add-ons:

| tree | version | contents | used for |
|---|---|---|---|
| `lammps` | `patch_10Sep2025` | Symmetrix, ML-PACE, KOKKOS (CUDA sm_86 + OpenMP); CUDA 12.4 | the ML-PACE and MACE rows |
| `lammps-dev` | `develop` | ML-PACE, PLUGIN, KOKKOS; CUDA 12.9 | the ace-jax rows, through the lammps-jax plugin |

Why each tree is pinned the way it is:
- Symmetrix doesn't compile against current `develop`.
- lammps-jax needs `Pair::eflag_only`, which is newer than 10 Sep 2025. It also
  needs `PKG_PLUGIN`, so that `LAMMPS_PLUGIN_PATH` auto-loads it, and nvcc > 12.4
  (12.4 rejects the parenthesised aggregate init in `emplace_back`).

The wrappers `lmp.sh` and `lmp-jax.sh` load the matching modules. `envs/moriarty-{gpu,cpu}.json` point at them.

### Sulis (A100, Slurm)

The moriarty recipe with Sulis modules (`GCC/12.3.0 OpenMPI/4.1.5 CUDA/12.8.0
CMake/3.26.3`), `Kokkos_ARCH_AMPERE80`, and the sources pinned to the commits
in `envs/moriarty-VERSIONS`. nvcc 12.8 compiles the lammps-jax plugin (12.4
does not). The tree is under `~/bench-scaling` (home: 2 TiB quota).

```bash
rsync src bench fixtures julia pyproject.toml README.md LICENSE (+ bench/scaling/models/) to ~/bench-scaling/ace-jax, then
bash bench/scaling/envs/sulis.sh sources venv              # login node: clone + download only
sbatch bench/scaling/envs/sulis-build.sbatch               # compile on an A100 node, then check
p=$(sbatch --parsable bench/scaling/envs/sulis-sweep.sbatch parity)     # fails unless the gates pass
sbatch --dependency=afterok:$p --kill-on-invalid-dep=yes bench/scaling/envs/sulis-sweep.sbatch acejax-ace-learned
```

A failed Symmetrix build only costs the MACE rows: `lmp.sh` then runs the dev
tree, which has ML-PACE too.

### Modal (A100-80GB)

```bash
modal run bench/scaling/modal_app.py --parity-only
modal run bench/scaling/modal_app.py [--only acejax-pace]
```

The image is the same recipe on `nvidia/cuda:12.9.1-devel` with sm_80. The
builder has no GPU, so the plugin links against the stub `libcuda`. Rows are
appended to `results/modal-a100.jsonl`, and a rerun resumes from there.

## Running

```bash
PYTHONPATH=bench:src python bench/scaling/sweep.py <host> --parity-only
PYTHONPATH=bench:src python bench/scaling/sweep.py <host> [--only CODE] [--dry-run]
```

**The parity gate runs first,** in a child process: a parent holding JAX's GPU
preallocation would starve every case after it. It compares:
- ace-jax standalone with ML-PACE;
- ace-jax standalone with ace-jax in LAMMPS;
- MACE PyTorch with Symmetrix;
- the learned-radial proxy splined against kept analytic (gate `spline`,
  standalone: |dE|/|E| <= 1e-9, max|dF| / max|F| <= 3e-8).

A failed gate blocks that code's LAMMPS rows on the host (a failed `spline`
gate blocks both modes of `acejax-ace-learned`), and a resumed sweep reuses
the recorded gate. `--only CODE` for a code the recorded gate never checked
(a line added since) re-runs the gate and appends only the new checks.

**Each case runs in a fresh process,** so peak memory is measured per case. A
line (model × mode × dtype × device) stops at its first `oom`, `error` or
`unstable` row.

## Host notes

- **ace-jax in LAMMPS is GPU-only.** lammps-jax provides only `pair_style jax/kk`,
  which needs KOKKOS built with CUDA. The ace-jax runs use `newton on neigh half`:
  the bundle's forces include ghost atoms and need reverse communication, and
  KOKKOS doesn't allow `neigh full` with `newton on`.
- **Species order is mapped at export.** A `.yace` lists its elements in fitted
  order (`[Ge, Si]`), while LAMMPS passes the data file's type order.
  `export_lammps(type_elements=...)` maps one to the other and records the
  mapping in the bundle.
- **The Symmetrix tree aborts in a static destructor after `Total wall time`.**
  It's a double free between `liblammps` and `libkokkoskernels`, and it happens
  after the results are written. `run_lammps.finished` accepts that exit.
- **Parity dumps print `%.17g`.** The default format caps the measurable force
  difference at about 5e-6.
