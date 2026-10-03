# Benchmark scaling harness

Produces the timings behind `docs/dev/benchmarks.md`: ace-jax (`PACEModel` on
`.yace`, `ACEModel` on `.npz`), ML-PACE and MACE, standalone and inside
LAMMPS, on SiGe and Cantor at three model sizes each. Design:
`docs/dev/benchmark-scaling-spec.md`; plan: `docs/dev/benchmark-scaling-plan.md`.

| file | role |
|---|---|
| `structures.py` | SiGe diamond / Cantor fcc supercells, the atom-count ladder |
| `models.py` | builds and lists every model (`python models.py pace\|ace\|ace-learned\|mace`) |
| `run_standalone.py` | one standalone case (ace-jax calculator or MACE PyTorch) |
| `run_lammps.py` | one LAMMPS case: writes the input, runs, parses the timed segment |
| `parity.py` | the per-host parity gates that must pass before any timing |
| `sweep.py` | expands the case matrix for a host and runs it resumably |
| `plot.py` | figures and tables for `docs/dev/benchmarks.md` |
| `user_page.py` | the user docs' Performance page: `docs/user/assets/benchmarks/*.png` and the `docs/snippets/benchmarks-*.md` table and notes, from the hosts in `USER_HOSTS` (`--cpu-host` / `--gpu-host` override) |
| `acepot.py`, `julia/` | the ACEpotentials.jl lines: Python side, and the pinned Julia env + drivers |
| `envs/moriarty.sh`, `envs/sulis.sh`, `envs/lestrade.sh`, `modal_app.py` | the environments |

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
- **ACEpotentials.jl** (CPU only, float64): two lines on the `ace_*.npz` models.
  - `acepotentials`: direct evaluation (`julia/run_standalone.jl`, standalone mode).
  - `acepotentials-trim`: the PR 309 `juliac --trim=safe` library, in LAMMPS
    (`pair_style ace`, MPI ranks like ML-PACE). `models.py acepotentials-trim
    [SiGe/small ...]` builds `models/trim/libace_<system>_<size>.so` per npz
    (`julia/build_trim.jl`, about 1-2 min and 1.4 GB each) and records it in
    the manifest; it is idempotent, and rebuilds when the npz, the Julia env
    or the build CPU model changes (or `FORCE=1`). An export the exporter
    refuses is recorded `unsupported`.
  - Both drivers rebuild the model with `ace1_model` + seed 11, as
    `julia/export_model.jl` did, and assert it identical to the npz (bases,
    A2B, spline tables, weights, E/F/V on the npz's test structure), failing
    loudly otherwise. A2B alone is compared to roundoff (`julia/a2b_check.jl`:
    the same pattern once |v| < 1e-12 cancellation noise is dropped, within
    4 eps max|A2B|): its coupling coefficients differ at the ULP level between
    Julia 1.11 (the npz files) and 1.12 (ace_SiGe_large: 72 of 3860 entries,
    <= 8.3e-16 relative). Rows and the trim manifest record `A2B_max_abs_diff`
    and `A2B_max_rel_diff` under `identity`.
  - The trim library compiles the *exact twin* (PR 309's
    `test/etmodels/ace1_exact_twin.jl`: the same basis and weights, unsplined
    radials). It differs from `acejax-ace` and `acepotentials` (both on the
    npz's spline tables) by the spline error, 2.8e-6 to 5.5e-4 eV/Å.

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
p=$(sbatch --parsable bench/scaling/envs/sulis-sweep.sbatch parity)     # fails only if no ace-jax gate passes
sbatch --dependency=afterok:$p --kill-on-invalid-dep=yes bench/scaling/envs/sulis-sweep.sbatch acejax-ace-learned
```

A failed Symmetrix build only costs the MACE rows: `lmp.sh` then runs the dev
tree, which has ML-PACE too.

The ACEpotentials.jl lines need three more steps (part of the default run):

```bash
bash bench/scaling/envs/moriarty.sh julia_env ace_plugin env_json
ACEPOT_JULIA="$HOME/.juliaup/bin/julia +1.12.6" ACEPOT_JULIA_DEPOT=/storage/eng/essswb/cache/julia-pr309 \
  python bench/scaling/models.py acepotentials-trim      # after `models.py ace`
```

- `julia_env` instantiates `bench/scaling/julia` (Julia 1.12.6, ACEpotentials.jl
  v0.10.2, the release that merged PR 309, from git until it reaches General)
  in its own depot, `/storage/eng/essswb/cache/julia-pr309`.
  Never the default `~/.julia`: every Julia call passes `JULIA_DEPOT_PATH` and
  `--startup-file=no`.
- `ace_plugin` builds the PR's `export/lammps/plugin` (from `pkgdir(ACEpotentials)`)
  with `mpicxx` against the `lammps-dev` headers, into `ace-plugin/aceplugin.so`.
- `env_json` adds `lmp_ace` (`lmp-ace.sh`: `lammps-dev` on the CPU), `ace_plugin`,
  `julia`, `julia_depot` and `julia_project` to the env json. `sweep.child_env`
  hands them to each case (`ACE_PLUGIN`, `ACEPOT_JULIA*`), with
  `JULIA_NUM_THREADS` as the other standalone lines' thread count (1 per rank
  in LAMMPS).
- Build the trim libraries on the host that runs them: juliac targets the build
  CPU (set `JULIA_CPU_TARGET` for another), and the `.so` links juliaup's
  `libjulia` by absolute path. A cluster needs the runtime bundled (PR 309's
  `bundle_julia_libs!`) or a per-cluster build.

### lestrade (i9-14900K, CPU only: `lestrade-cpu`)

```bash
bash bench/scaling/envs/lestrade.sh          # plugin, CPU Symmetrix tree, lmp-ace.sh, lmp-cpu.sh, envs/lestrade-cpu.json (git-ignored)
ACEPOT_JULIA="$HOME/.juliaup/bin/julia +1.12.6" ACEPOT_JULIA_DEPOT=/storage/eng/essswb/cache/julia-pr309 \
  python bench/scaling/models.py acepotentials-trim      # built here: juliac targets this CPU
taskset -c 0-15 python bench/scaling/sweep.py lestrade-cpu --only CODE
```

- Reuses moriarty's venv and lammps-dev headers read-only through the shared
  home; the plugin, `lmp-ace.sh`, `lmp-cpu.sh` and the CPU Symmetrix tree (`src/`,
  copied from moriarty's pinned checkouts) go to `/storage/eng/essswb/bench-scaling-lestrade/`.
- **P-cores only.** The 14900K's P-cores are CPUs 0-15 (8 cores x 2 HT) and its
  E-cores 16-31. `HOSTS["lestrade-cpu"]["cpus"]` pins the sweep (and so every
  case) to 0-15, giving standalone codes 16 threads; MPI codes run 8 ranks bound
  one per P-core (`mpirun_args`: `--bind-to core --map-by core`). Rows record
  `cpu_affinity` and `mpirun_args`.
- MACE in LAMMPS runs `lmp-cpu.sh`: moriarty's Symmetrix tree is AVX-512
  (`-march=native` on Cascade Lake) and gets SIGILL here, so `lammps_cpu` builds
  a CPU-only one (no Kokkos, ML-PACE, FlexiBLAS). Launch long sweeps from an
  agent in their own scope (`systemd-run --user --scope -p MemoryMax=52G`): an
  agent's memory cap otherwise throttles the large cases.

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
  standalone: |dE|/|E| <= 1e-9, max|dF| / max|F| <= 3e-8);
- ACEpotentials.jl direct against ace-jax (gate `acepot`, 1e-10 eV/atom, 1e-9
  eV/Å), the trim library in LAMMPS against Julia ETACE on its exact twin (gate
  `trim`, the same), and the trim library against ace-jax (gate `trim-ace`, 1e-3
  eV/Å: the spline error). All three evaluate the extxyz-roundtripped structure
  (extxyz rounds positions to 1e-8 Å) and are `unsupported` on GPU hosts. A
  failed `acepot` gate blocks the `acepotentials` line.

A failed gate blocks that code's LAMMPS rows on the host (a failed `spline`
gate blocks both modes of `acejax-ace-learned`), and a resumed sweep reuses
the recorded gate. `--only CODE` for a code the recorded gate never checked
(a line added since) re-runs the gate and appends only the new checks.

**Each case runs in a fresh process,** so peak memory is measured per case. A
line (model × mode × dtype × device) stops at its first `oom`, `error` or
`unstable` row.

## Host notes

- **The ACEpotentials.jl lines are CPU-only:** GPU hosts never plan them.
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
