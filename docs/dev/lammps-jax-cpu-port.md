# lammps-jax on CPU: MPI domain-decomposed ace-jax in LAMMPS

Status: investigation and plan, 2026-10-12. Nothing here is implemented or committed.

## TL;DR

- **Upstream has already done the port.** lammps-jax `main` gained a CPU path on
  2026-10-10: commit `d301b12` ("Add cpu support"), followed by `880d409`. That is one
  commit after `4a7f4fb`, the version we run on Sulis. It adds a host backend for the
  PJRT runtime, builds the pair style against Kokkos OpenMP, and documents
  `mpirun -np N lmp -k on t 1 ...` with the XLA CPU PJRT plugin. Most of what this
  document was asked to plan, the C++ port, now exists upstream. No issues or PRs
  discuss it. Its CI does not build LAMMPS: `tests/test_lammps.py` skips without
  `LAMMPS_BIN`. The wheel workflow for the CPU plugin (`pjrt-cpu.yml`) has never run,
  and `lammps-jax-pjrt-cpu` is not on PyPI. So the path is real but **brand new and
  untested outside the author's machine**.
- **Verdict: feasible, and cheap given upstream.** What remains for us is integration
  plus two small fixes, both found by probes below. Neither needs Kokkos or LAMMPS
  internals work.
  1. ace-jax `dense` and `matrix` bundles **crash the XLA CPU compiler** (a CHECK
     failure that aborts the process) because lammps-jax lowers every program for
     `platforms=("cuda",)`. A CPU bundle needs a CPU lowering: an export-platform
     option.
  2. Recent XLA CPU plugins **reject lammps-jax's GPU client options**
     (`visible_devices`, `preallocate`). The plugin upstream pins (XLA at jax 0.10.1)
     ignores them; XLA `main` and any 2026-09+ build refuse them.
- **Effort: about 6–9 working days** to Sulis CPU timing rows, plus 1–2 optional days
  for a compile-once executable broadcast. Without upstream's commit, the same result
  would have cost about 3–4 weeks: the device abstraction, the host backend, CMake, and
  an XLA plugin build pipeline.
- **Measured on moriarty** (an ace-jax Si bundle, 512 atoms, exported with upstream
  `main`, run through the exact PJRT C-API calls lammps-jax makes, against a prebuilt
  XLA CPU plugin):
  - Energy matches `ACECalculator` bit for bit. Forces agree to 2e-12 eV/Å (1e-14
    relative).
  - Compile takes 0.5–2.8 s at about 250 MB RSS.
  - Eight single-core processes deliver 3.3× the throughput of one 16-thread process,
    on half the cores.

## Goal and non-goals

Goal: run ace-jax (ACE, and PACE `.yace`) potentials in LAMMPS on CPU-only nodes with
one MPI rank per core, through lammps-jax bundles. That gives the accuracy-cost
benchmark CPU rows the same parallel model as ML-PACE, MACE and GAP (32 or 128 ranks)
instead of one 128-thread Python process. Parity must be about 1e-10 against the
ace-jax ASE calculator, on 1 rank and on N ranks.

Non-goals:
- Per-core speed-ups in ace-jax itself. Those are items 2–4 of
  `docs/dev/cpu-gap-profile.md`; this plan removes only the threading factor.
- Multi-hop or communicating (MACE-style) bundles on CPU. ACE has `n_hops=1`, so no
  model comm is needed.
- GPU changes.
- Maintaining a fork. Everything we need is small enough to upstream.

## Findings

References are to lammps-jax `880d409` (upstream `main`), checked out at
`/home/eng/essswb/projects/lammps-jax/.worktrees/upstream-main`. The stale local
checkout (`dev/julia_export`) and its `origin` (ssh) were not touched. Upstream refs
were fetched over https into `refs/remotes/upstream/*`.

### 1. The GPU path end to end (unchanged in structure by the CPU commit)

- **Bundle** (`python/lammps_jax/export.py`). A JSON file holding:
  - three programs (`force`, `energy`, `energy_and_forces`) as portable VHLO bytecode
    from `jax.export`, lowered for `platforms=("cuda",)` (`export.py:503`) under
    `default_matmul_precision("highest")`;
  - serialized `CompileOptions` with 1 replica, 1 partition and
    `compile_portable_executable=True` (`export.py:511-515`);
  - a `contract`: `input_layout` sparse-edge or neighbour-matrix, `max_atoms`,
    `max_edges` or `max_neighbors`, `max_owned`, `cutoff`, `precision`, `newton`,
    `n_hops`, `edge_pairing`, `comm_widths`, `custom_call_targets`, `uses_box`,
    `pair_sum`.

  The program inputs are positions, species, nlocal, nghost, then either
  senders/receivers/edge_mask or neighbors/num_neighbors, then an optional box
  (`pair_jax_kokkos.cpp:919-944`).
- **Plugin and client.**
  - `pjrt/plugin.cpp:45` loads the plugin with `dlopen(RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE)`
    and calls `GetPjrtApi`.
  - `ClientSession::initialize` (`client_session.cpp:26-67`) creates one client per MPI
    rank with options `visible_devices=[kokkos device]`, `preallocate=false` and an
    optional `memory_fraction` (`LAMMPS_JAX_MEM_FRACTION`), and takes device 0.
  - `Runtime::initialize` (`runtime.cpp:292-330`) prepends GPU `XLA_FLAGS` (command
    buffers) and unsets `PJRT_GPU_ENABLE_ASYNC_DISPATCH`, because the force pointer is
    read before completion. It registers FFI handlers and compiles each program on
    every rank.
- **Buffers: zero-copy both ways.**
  - Inputs are `PJRT_Client_CreateViewOfDeviceBuffer` views of Kokkos device views,
    recreated each step behind a stream-event wait (`runtime.cpp:30-60`, `:184-187`).
    They are marked non-donatable.
  - The force output's device pointer comes from `OpaqueDeviceMemoryDataPointer`
    (`runtime.cpp:63-71`, `:227-231`). A Kokkos functor then adds it into `f` directly
    (`add_model_forces_as`, `pair_jax_kokkos.cpp:948-968`), and `OutputLifetime`
    retires the buffer behind a consumer-stream event.
  - The energy is one D2H scalar copy.
- **Neighbour data.**
  - The sparse layout packs the Kokkos list into a padded edge buffer every step,
    filtered to the true cutoff (`PackNeighborFunctor`, `:54-118`).
  - The matrix layout copies the full list only on list rebuilds
    (`CopyNeighborMatrixFunctor`, `:120-155`; `rebuild_edges` at `:1001`).
  - Capacity overflow is a collective `error->all` after execution (`:1052-1070`).
- **Ghosts, forces and virial.**
  - With `newton on`, the model returns force rows for owned and ghost atoms. Ghost
    rows go back through LAMMPS's normal reverse comm, because the pair style sets
    `reverse_comm_device` and adds into `f` over `nall` (`:964`).
  - The virial is a device f·r reduction over `nall` (`VirialFDotRFunctor`, `:227-245`,
    `:1081-1089`). There is no per-atom energy or virial (`:984-985`).
- **Model comm** (`comm_config`, `service_model_comm`, `pjrt/model_comm.*`). This is
  only for communicating bundles (MACE-style per-layer feature exchange).
  - The program calls an FFI handler. On a worker thread, execution blocks while the
    LAMMPS thread runs `comm->forward_comm/reverse_comm(this)` with the pair style's
    pack/unpack (`:444-571`).
  - It is unused by ACE (`comm_widths=[]`, `n_hops=1`).
- **Multi-rank today.**
  - Each rank has its own client, compile and buffers, on its Kokkos device.
  - Owned/ghost capacities are checked collectively before launch (`:1025-1035`).
  - Our Sulis GPU runs use this, one rank per GPU.

### 2. What upstream's CPU commit (`d301b12`) did

- **Device abstraction.** `pjrt/device.h` defines a pointer, stream, event and context
  API. `device_cuda.cpp` holds the old driver-API code. `device_host.cpp` is the host
  version: `memcpy`/`memset`, no streams, no events, `malloc` for "pinned" memory, and
  platform name `"Host"`. CMake picks one from `KokkosCore_config.h`: CUDA, or host for
  OpenMP/Serial. HIP, SYCL and OpenMPTarget are rejected (`cpp/CMakeLists.txt:39-48`,
  `:86`, `:112-119`).
- **Execution.** Without streams, the session skips the stream extension. It waits for
  outputs with `PJRT_Buffer_ReadyEvent` + `PJRT_Event_Await`
  (`client_session.cpp:68`, `:103-118`).
- **Pair style.** All the `#ifdef KOKKOS_ENABLE_CUDA` guards are removed, so the same
  Kokkos functors run in the host execution space. Model comm stages through host
  memory on host Kokkos.
- **Python and comm.** `comm.py` adds `Comm.sequence`. `export.py` orders the
  energy-then-force exchange chains for communicating fused programs. This does not
  affect ACE.
- **Plugin packaging.** `pjrt-cpu/` packages XLA's `pjrt_c_api_cpu_plugin.so` as the
  `lammps-jax-pjrt-cpu` wheel (Apache-2.0 plus THIRD_PARTY_NOTICES), versioned as
  jaxlib. `.github/workflows/pjrt-cpu.yml` builds it with Bazel from the XLA commit
  pinned by `jax-v<version>` (0.10.1; a 5-hour job). It is the `[cpu]` extra.
  **jax/jaxlib ship no C-API CPU plugin:** `libjax_common.so` in jaxlib 0.11.2 exports
  31 symbols and no `GetPjrtApi`. Upstream therefore had to build XLA.
- **README CPU recipe.** Build LAMMPS with `kokkos-openmp.cmake`, `PKG_PLUGIN=on`,
  `BUILD_MPI=on` and `Kokkos_ENABLE_IMPL_VIEW_LEGACY=on`. Build the plugin with the
  host compiler (no nvcc_wrapper). Run `mpirun -np 4 lmp -k on t 1 -sf kk -pk kokkos
  newton on neigh half -var pjrt .../pjrt_c_api_cpu_plugin.so`. README: "One MPI rank
  per core runs fastest."
- **Not covered upstream:**
  - The export platform is still hard-coded to cuda.
  - Nothing controls the per-rank XLA thread count.
  - There is no compile caching.
  - The integration tests on CPU are local only.

### 3. Probes (moriarty, 2026-10-12)

Probe files:
- the probe source, `/storage/eng/essswb/cache/pjrt-probe/src/probe.c` (with the export and input scripts), about 150 lines of C
  against lammps-jax's vendored `pjrt_c_api.h`;
- bundles and inputs under `/storage/eng/essswb/cache/pjrt-probe/`.

The probe mirrors lammps-jax's runtime calls exactly: client create, compile from
bundle bytes, `CreateViewOfDeviceBuffer` views of host arrays with a no-op delete
callback, execute with every input non-donatable, `OpaqueDeviceMemoryDataPointer` on
the force output *before* awaiting completion, then await.

Plugin used: zml's prebuilt XLA CPU PJRT plugin, `zml/pjrt-artifacts` release
`manual-2026-09-08T13-23-00Z`, `libpjrt_cpu.so`. It is 275 MB unstripped, PJRT C API
0.114 (lammps-jax's header is 0.112), needs glibc ≥ 2.28 and links libc++ statically.
We use it because upstream's wheel is not published.

| probe | result |
|---|---|
| `PJRT_Client_Create` with lammps-jax's options | **fails**: `Unexpected option name passed to PJRT_Client_Create: visible_devices`. XLA `main` validates options against `{cpu_device_count, asynchronous, process_id}` (`xla/pjrt/c/pjrt_c_api_cpu_internal.cc`); the XLA pinned by jax 0.10.1 (`9b63591`) ignored unknown options. |
| … without options | OK; platform `cpu`, 4 addressable devices (default `cpu_device_count=4`; device 0 is used) |
| sparse bundle, CUDA-lowered (as exported today), compile | OK, 0.55 s (32 cores) / 2.2 s (1 core), ~250 MB RSS |
| sparse, execute, 512-atom Si cluster (ACE `si_ace_model.npz`, f64, lean) | E = −71114.0631486661 eV, **identical** to `ACECalculator`; max\|ΔF\| = 2.1e-12 eV/Å (1.2e-14 of max\|F\|) |
| dense and matrix bundles, CUDA-lowered, compile | **process abort**: `computation_partitioner.cc:269 Check failed: operand_maps.size() == 1` after `ComputeOutputToInputIndexing is not implemented for opcode call`. jaxlib 0.11.2's own in-process CPU client fails the same way, so this is the lowering, not the plugin. |
| … same programs lowered with `platforms=("cpu",)` | dense and matrix compile (2.8 s, 0.6 s); dense executes with the identical energy |
| cause | `jnp.cumsum` (ace-jax `export/lammps.py:170` and `:238`, the dense/matrix slot compaction) lowers for cuda to `chlo.scan` inside a `stablehlo.composite` (CPU lowering: `reduce_window`). XLA:CPU's fusion emitter cannot index through the resulting `call`. |
| XLA thread pool | sized from CPU affinity (`tsl::port::MaxParallelism`): about 9 threads when pinned to 1 core, 17 for 4 cores, 65 for 16 or 32 cores. There is no client option for it. |
| scaling (dense, cpu-lowered, 512 atoms, 2301-row padding) | 1 process on 1 core: 50 ms. 8 concurrent single-core processes: 78 ms each, so 52k atom-steps/s total. 1 process on 16 cores: 33 ms, so 15.7k atom-steps/s. **Ranks give about 3.3× the throughput on half the cores.** The 1.55× per-process slowdown at 8 is shared caches and DRAM, as `cpu-gap-profile.md` §3.3 found. |

The zml plugin compiled and ran bundles exported by jax 0.11.2, which is newer than
upstream's 0.10.1 wheel. The plugin's XLA must be at least as new as the exporting jax
for the StableHLO and VHLO versions; upstream's README makes the same point.

## Design

Use upstream's CPU path unchanged, plus three thin additions.

1. **Export platform** (lammps-jax, upstreamable). Add `platforms=("cuda",)` as an
   `export_model` keyword. Record it in the bundle as `export_info.platforms`, and have
   the C++ loader warn when it does not match the build's backend. ace-jax's
   `export_lammps(..., platform="cpu")` passes it through.

   A multi-platform export (`("cpu","cuda")`) is *not* proposed. It adds a
   platform-index argument that the C++ runner would have to supply.

   The alternative, rewriting ace-jax's cumsum compaction so that it lowers the same on
   every platform, works without touching lammps-jax. Keep it as a fallback (step 2b).
2. **Host client options** (lammps-jax, upstreamable, about 5 lines). In
   `client_session.cpp`, send `visible_devices`, `preallocate` and `memory_fraction`
   only when `device::has_streams()`. On host, optionally pass
   `cpu_device_count=1`, and `asynchronous` from an env var for benchmarking. Until it
   lands, carry it as a local patch in `sulis.sh`, or use upstream's 0.10.1-era plugin,
   which ignores the options.
3. **CPU plugin supply.** Either:
   - (a) build `//xla/pjrt/c:pjrt_c_api_cpu_plugin.so` from the XLA commit matching
     our jax (0.11.2) with upstream's recipe (`pjrt-cpu.yml`, Bazel + clang,
     `configure.py --backend=cpu`) on mnf148 or lestrade. This is the reproducible
     choice and gives an exact jax/XLA match.
   - (b) use zml's prebuilt release, as probed, for development.

   Do not redistribute (b): zml's repo has no licence file, although the XLA source is
   Apache-2.0. If upstream publishes `lammps-jax-pjrt-cpu`, switch to it.
4. **Threading and binding** (run-time, no code). One rank per core,
   `-k on t 1`, `OMP_NUM_THREADS=1`, and `srun --cpu-bind=cores` (or `mpirun --bind-to
   core`). Binding is required. XLA sizes its pool from the affinity mask, so 128
   unbound ranks would each start a pool of about 260 threads. With binding, each rank
   has about 9 mostly idle threads.
5. **Bundle sizing per rank** (ace-jax). Size the capacities from the *rank's
   subdomain*, not the whole cell:
   - `max_owned` ≈ N/ranks × headroom;
   - `max_atoms` = owned + the ghost shell to rcut + skin.

   Ghosts dominate small subdomains: for about 1000 Si atoms per rank the shell is
   about 2.5× owned. Owned-row blocks keep model work proportional to owned atoms.
   `layout="auto"`'s memory budget (`dense_budget_bytes`) must be *node RAM / ranks*
   for a CPU export, not device memory.
6. **Optional: compile once per node or job.** Every rank compiles the same programs.
   That costs seconds and about 250 MB each for small models. Large models (o4d20,
   Cantor) are unmeasured. If compiles get slow, rank 0 compiles, calls
   `PJRT_Executable_Serialize`, `MPI_Bcast`s the bytes, and the other ranks call
   `PJRT_Executable_DeserializeAndLoad`. That is upstreamable and about 1–2 days.

## Implementation steps

Each step ends with its acceptance test. "Parity" means against `ACECalculator` (or
`PACEModel` via ASE) on the same configuration, in float64.

1. **CPU plugin for jax 0.11.2** (0.5 d with zml; 1–1.5 d to build our own, mostly
   machine time).
   - Do: build `pjrt_c_api_cpu_plugin.so` from the XLA commit in jax-v0.11.2's
     `MODULE.bazel` on mnf148, under `/storage/eng/essswb/cache/bazel` with `-j` capped,
     using upstream's command line. Strip it, then put it under
     `/storage/eng/essswb/pjrt-cpu/<jax-version>/`. Interim: the zml `.so` already in
     `/storage/eng/essswb/cache/pjrt-probe/zml/x/`.
   - Accept: the probe (`probe.c`) runs the sparse and cpu-lowered dense Si bundles to
     energy-identical and force ≤ 1e-12 results.
   - Where: mnf148 for the build. Sulis needs glibc ≥ 2.28 (Rocky 8: yes), and the
     result is copied there by rsync.
2. **Export platform option.**
   - (a) lammps-jax: add `platforms` to `export_model`, record it, and add a warning in
     `load_bundle_file` (0.5 d, with a test in `tests/test_export.py`).
   - (b) ace-jax: add `export_lammps(platform="cuda"|"cpu")`, threaded through
     `bench/scaling/run_lammps.py` (0.5 d). Fallback if upstream declines: a cumsum
     formulation without the composite scan.
   - Accept: `tests/test_export_lammps.py` gains a check, skipped without lammps-jax,
     that a cpu-platform dense, matrix and sparse bundle compiles through jaxlib's CPU
     client and matches `make_energy_fn` run eagerly to 1e-12. The cuda-platform dense
     bundle is a known xfail on CPU.
3. **Host client options patch** in lammps-jax `client_session.cpp` (0.25 d).
   - Accept: `pair_style jax/kk` initialises against the jax-0.11.2-matched plugin. GPU
     behaviour is unchanged (re-run one Sulis GPU parity row later).
4. **Desktop CPU LAMMPS for testing** (lestrade or mnf148, not moriarty under the
   agent cap; 0.5–1 d). Build LAMMPS develop (the same `ec02ed0` as Sulis
   `lammps-dev`, or newer) with Kokkos OpenMP + Serial, `PKG_PLUGIN`, MPI and
   `Kokkos_ENABLE_IMPL_VIEW_LEGACY=on`, plus `PKG_ML-PACE` for the reference rows.
   Then build the lammps-jax plugin against it with g++.
   - Accept: upstream's `LAMMPS_JAX_TEST_PLATFORM=cpu pytest tests/test_lammps.py`
     passes. It runs its decks on 1 and 2 ranks, with Pallas and OEQ skipped.
5. **ace-jax parity in LAMMPS on CPU** (1 d). Reuse the bench parity gate (`run_lammps.py`
   already checks GPU LAMMPS against ASE) with `device=cpu`.
   - Accept, on Si diamond 4096 atoms (rattled), Cantor 5-element and one PACE `.yace`,
     all periodic and f64, for sparse, dense and matrix, on 1, 2, 8 and 32 ranks:
     - energy within 1e-10 relative of `ACECalculator`;
     - forces within 1e-10 × max|F|;
     - virial and pressure within 1e-9 relative;
     - 1-rank and N-rank results agree to the same tolerance;
     - a 1000-step NVE run conserves energy like the GPU path.

     Lean-splined or radial-table bundles use their documented approximation bounds
     instead.
6. **Threading and binding check** (0.25 d). On lestrade, run 32 bound ranks.
   - Accept: `ps -L` shows about 9–12 threads per rank, the load is about 32, and
     there is no oversubscription. Time `asynchronous=true` against `false`, if the
     option is exposed.
7. **Bench integration** (1–1.5 d).
   - `bench/scaling/envs/sulis.sh`: a new `lammps_dev_cpu` target (develop + Kokkos
     OpenMP + PLUGIN + ML-PACE, `-D Kokkos_ARCH_ZEN2=ON`) and `lammps_jax_cpu` (the
     plugin with the host backend, with the step 3 patch applied), plus a
     `sulis-build-cpu-dev.sbatch`.
   - `run_lammps.py` and the sweep: an ace-jax CPU row style that exports with
     `platform="cpu"` and per-rank capacities, then launches
     `srun --cpu-bind=cores -n R lmp -k on t 1 -sf kk -pk kokkos newton on neigh half`
     with `pair_style jax/kk <plugin>`.
   - Record the plugin's XLA commit in `VERSIONS`.
   - Accept: `tests/test_bench_scaling.py` and `tests/test_bench_accuracy_cost.py`
     cover the new row's command construction, with no LAMMPS needed.
8. **Sulis CPU timing** (1–2 d including queue). On one 128-core compute node, for the
   accuracy-cost ladder models, run 32 and 128 ranks against today's standalone
   128-thread rows and ML-PACE on the same rank counts.
   - Accept: the parity gate (step 5) passes on Sulis. Report µs/atom/step and the
     speed-up over standalone. The target is at least 2.5× from rank parallelism at 32
     cores, per `cpu-gap-profile.md` §3.3 (8–10× on 16 ranks against 2.6–3.5× on
     threads).
9. **(Optional) compile once and broadcast** (1–2 d), if step 8 shows per-rank compile
   time or memory to be significant.
   - Accept: rank 0 compiles and the others deserialize. Wall-clock to the first step
     at 128 ranks is at most 1.2× the 1-rank compile. Results are bit-identical.
10. **Upstream** (0.5 d of our time). Open the step 2a and step 3 changes as PRs, with
    the probe evidence: the dense/matrix CHECK failure and the option rejection on new
    XLA. The user decides when to open them.

## Effort

| piece | days |
|---|---|
| CPU plugin (own build; zml interim) | 0.5–1.5 |
| export platform: lammps-jax + ace-jax | 1 |
| host client-options patch | 0.25 |
| desktop CPU LAMMPS + plugin build, upstream CPU tests | 0.5–1 |
| ace-jax parity in LAMMPS on CPU | 1 |
| threading/binding check | 0.25 |
| bench integration (sulis.sh, run_lammps, sweep) | 1–1.5 |
| Sulis CPU timing campaign | 1–2 |
| **total** | **about 6–9** |
| optional executable broadcast | +1–2 |

Without upstream's `d301b12` we would also have needed about 2–3 weeks more:
- the `device.h` abstraction and host backend;
- removing the CUDA-only guards from the pair style and making the comm staging
  host-safe;
- CMake backend detection;
- the XLA plugin build pipeline.

Plus debugging those, with no CPU LAMMPS CI.

## Where things can be tested

- **moriarty** (this box, 32 cores, 16 GiB agent cap): PJRT probes, bundle exports and
  `pytest` export checks only. No LAMMPS builds.
- **lestrade / mnf148**: the Bazel XLA plugin build (mnf148: 256 GB, no priority user),
  the desktop CPU LAMMPS + plugin build, parity at 1–32 ranks, and the threading check.
  mnf148 has `Linger=no`, so launch long builds with `setsid nohup`.
- **Sulis CPU nodes** (128-core EPYC 7742; LAMMPS under `~/bench-scaling/`): the CPU
  build via a new sbatch, then the timing rows. No mount of the desktop FS, so copy the
  plugin `.so` with rsync.

## Risks

1. **Upstream CPU path is two days old and has no CI with LAMMPS.** There may be bugs
   in host-space comm staging, the half list on host Kokkos, or the f·r virial with
   OpenMP. We are its first serious user. Mitigation: steps 4–5 run upstream's own
   tests first, then our parity gate.
2. **XLA version coupling.** The plugin's XLA must accept the StableHLO and VHLO from
   our jax, and the CPU compiler is the one that crashed here. Pin the plugin to the
   exporting jax's XLA commit (step 1a) and record both in the bundle and `VERSIONS`.
   jax 0.11.2 has its own CPU issues: upstream's CI skips it for `dist`, with a
   segfault in `ghost_exchange.redistribute`, which is a multi-hop path ACE does not
   use.
3. **Platform-specific lowering** beyond cumsum, such as sort or argsort or a future
   ace-jax op. Mitigation: always export CPU bundles for `cpu`, and keep the step 2
   compile check in the test suite for all three layouts.
4. **Memory per rank.** Each rank has a client, compiled executables, padded buffers
   and XLA temporaries. That is about 0.4 GB at 512 atoms here; dense or matrix temps
   grow with owned rows × k. At 128 ranks per node, 1 GB per rank is 128 GB out of 512
   GB. It fits, but `layout="auto"` must budget per rank (design 5), and large-basis
   models (o4d20) need measuring.
5. **Ghost overhead at high rank counts.** At 128 ranks on a 10k-atom cell, a subdomain
   has about 80 owned atoms against hundreds of ghosts. Owned-row blocks keep model work
   on owned rows, but the sparse layout's edge packing and the dense padding scale with
   `max_atoms`. Benchmark cells must be large enough (≥ 500 owned atoms per rank),
   which is also what ML-PACE needs.
6. **Per-step overhead on CPU.** There is no async overlap: execute is synchronous, the
   force pointer is read and added, and input views are recreated every step. This is
   negligible against a 50 ms step, but it matters for tiny subdomains. The
   `asynchronous` client option is untested.
7. **Process-level side effects.** `Runtime::initialize` sets GPU `XLA_FLAGS` defaults
   on host too (`runtime.cpp:300-316`). With the same `XLA_FLAGS` set, the probe ran
   unchanged and gave the identical energy, so they are harmless. They are still worth
   gating on `device::has_streams()` for tidiness. The CPU
   plugin's statically linked libc++ next to LAMMPS's libstdc++ worked with
   `RTLD_LOCAL` in the probe, but has not been tested inside LAMMPS with
   ML-PACE/OpenMP loaded.

## Cheaper alternatives (brief)

- **`pair_style python`**: pairwise only, with per-pair Python callbacks. Not usable for
  a many-body ACE.
- **ML-IAP unified** (`pair_style mliap unified`, the MACE route): an embedded Python
  model per rank. LAMMPS hands it per-pair `rij`, `pair_i`, `pair_j` and elements, and
  takes back per-pair forces and energies. That matches ace-jax's edge → site-energy
  core well.
  - Cost: about 3–5 days, and it is independent of lammps-jax.
  - It needs LAMMPS with `ML-IAP` + `PYTHON` + Cython.
  - Each rank imports jax and jit-compiles: about 300–500 MB and several seconds per
    rank. Python and host-copy overhead every step.
  - It is a reasonable fallback if upstream's path proves unstable. It is not better:
    the bundle path has no Python at run time, and it is what our GPU rows already use.
- **mpi4py standalone domain decomposition**: reimplements DD, ghosts, reverse force
  comm and an integrator. That is weeks of work and duplicates LAMMPS. No.
- **Independent replicas** (N single-thread processes on 1/N cells): this is how
  `cpu-gap-profile.md` §3.3 *emulated* ranks. It is fine for throughput estimates but is
  not one MD run, so it is not a fair benchmark row.

## Upstream vs fork; licences

Contribute upstream: the export-platform option, the host client-options gating, and
optionally the compile broadcast. All three are small, general, and fix real breakage
for anyone exporting with cumsum or using a current XLA. A fork would only add
maintenance against a fast-moving repo; we already pin a commit in `sulis.sh`.
- lammps-jax is MIT (© 2026 Abhijeet Gangan). `NOTICE` lists the vendored OpenXLA
  headers (Apache-2.0).
- The CPU plugin wheel is Apache-2.0 with XLA's third-party notices.
- zml's `pjrt-artifacts` has no licence file. Use its binary only internally.

## Results on moriarty (2026-10-12)

Box: Xeon Silver 4216, 16 cores (2 threads each), 22 MB L3, 62 GB.

Stack:
- **XLA CPU plugin:** our own build, `pjrt_c_api_cpu_plugin.so`.
  - Source: XLA `91888df` (jax-v0.11.2's `MODULE.bazel`), hermetic clang, Bazel 8.7.0, `-c opt`.
  - 12,853 actions, about 55 min at `--jobs=12`. Stripped size 200 MB, needs only glibc 2.27.
  - Kept at `/storage/eng/essswb/bench-cpu/pjrt-cpu-xla-91888df/`.
- **LAMMPS:** `lammps-dev` `ec02ed0`, Kokkos OpenMP, MPI, PLUGIN; built in `/storage/eng/essswb/bench-cpu/lammps-build`.
- **lammps-jax:** upstream `880d409` plus the two fixes, on local branch `cpu-fixes`:
  - client options (`visible_devices`, `preallocate`, `memory_fraction`) are sent on GPU builds only;
  - `export_model(platform=)`.
- **ace-jax:** branch `feat/lammps-cpu`.
  - `export_lammps(platform=)`.
  - `neighbour_capacity(procs=)` for per-rank sizing.
  - A dense/matrix CPU-compile test.
  - Driver: `bench/scaling/lammps_cpu_smoke.py`.

Parity, against `ACECalculator` (o4d12, 4096 rattled Si atoms, matrix layout):
- **Unsplined:** max |dF| 1.2-1.3e-12 eV/Å and |dE| ≤ 3e-14 eV/atom, at 1, 4, 8 and 16 ranks. Both zml's plugin and ours give this.
- **Spline 1e-10 on both sides:** max |dF| 5.7e-10.
  - This is an ace-jax mismatch between the calculator's and the exporter's spline, not a CPU issue.
  - The A100 gate sees the same thing: o3d8 at 1.2e-9.

Speed: µs per atom-step, o4d12 spline 1e-10, 4096 atoms. Ranks are bound to cores with one thread each. Standalone is one `ACECalculator` process pinned with `taskset`.

| cores | standalone | LAMMPS, matrix | LAMMPS, dense | LAMMPS, sparse |
|---|---|---|---|---|
| 1 | 66.6 | 86.9 | 87.8 | 313 |
| 2 | | 44.5 | | |
| 4 | 32.1 | 25.6 | | |
| 8 | | 20.3 | | |
| 16 | 24.1 (32 threads: 24.4) | 19.8 | 19.9 | 65.8 |

At 13,824 atoms (about 950 owned per rank at 16 ranks), 16 ranks give 20.8 and 8 ranks 20.9; standalone gives 22.6.

What the numbers say:
- **Ranks and threads hit the same ceiling here (about 20 µs).** Bigger subdomains do not move it, so ghost overhead is not the limit. LAMMPS spends 98% of the step in Pair and under 2% in Comm.
- Per-rank pair time per owned atom rises 3.5× from 1 to 16 ranks. That points at shared memory bandwidth and L3, not at XLA's threading.
- On this box, MPI ranks beat the standalone process by only about 1.2×.
- One rank is about 30% slower than standalone on one core. Layout does not explain it; the gap is between the bundle's blocked energy function and the calculator's path. It is not yet investigated.
- The open question is whether Sulis's EPYC 7742 nodes behave differently. They have 16 MB of L3 per 4-core CCX and 8 DDR4-3200 channels per 64-core socket. Measure it there before drawing conclusions.

## Results on Sulis (2026-10-12)

Node: AMD EPYC 7742, 2 × 64 cores, 16 MB L3 per 4-core CCX.

Stack:
- The same XLA plugin, copied over.
- LAMMPS `ec02ed0` built for Zen 2: `build-kk-cpu`, Kokkos OpenMP.
- lammps-jax `cpu-fixes`.
- jax 0.11.2 for export.
- Scripts: `bench/scaling/envs/sulis-lammps-cpu-{build,smoke}.sbatch`, on `devel`, one exclusive node each. The build takes about 3 minutes.

Results in µs per atom-step, splined at 1e-10. Ranks are bound to cores with one thread each. Standalone is `ACECalculator` with all 128 threads on the same structure.

| model, cell | standalone | 1 | 8 | 16 | 32 | 64 | 128 ranks |
|---|---|---|---|---|---|---|---|
| o4d12, 32,768 atoms | 31.9 | | | 33.5 | 16.3 | 8.04 | 5.36 |
| o4d12, 4,096 atoms | 29.2 | 102 | 35.0 | | 10.5 | 6.35 | 3.08 |
| o4d20, 4,096 atoms | 794 | | | | 318 | 162 | 135 |

- **Parity:** at every rank count, matches `ACECalculator` to 7-9e-10 eV/Å for o4d12 and 6e-11 for o4d20. That is the known spline mismatch.
- **Scaling:** ranks scale nearly linearly to 64 on these nodes, unlike moriarty. 128 ranks are 5.9-9.5× faster than the standalone 128-thread process.
- **Small subdomains run faster per atom:** the 4,096-atom cell at 128 ranks, about 32 owned atoms each, beats the 32,768-atom cell. The likely reason is that a rank's working set fits its CCX's L3.
- **o4d20 does not scale from 64 to 128 ranks**, which is not yet understood.
- **Capacity fix:** 128 ranks on 4,096 atoms first failed with "owned-row capacity exceeded" (40 > 36). `neighbour_capacity(procs=)` now adds 3 sqrt(n) + 4 to max_owned.
- **Against the CPU cost table:** the table's ace-jax rows are standalone (o4d12 34.7, o4d20 847), and ML-PACE runs 32 ranks (f800 2.27, f3200 10.05).
  - At 128 ranks, ace-jax o4d12 is 3.1-5.4 µs and o4d20 135 µs.
  - ML-PACE was never run at 128 ranks; a fair comparison runs both codes at the same rank count.

## Open questions for the user

1. Build our own CPU plugin from jax 0.11.2's XLA on mnf148, which is a few hours of
   Bazel? Or use zml's prebuilt `.so` for the benchmark and record its release tag?
2. May we open the two small upstream PRs, export platform and host client options?
   Or ask the author first in an issue? Both are fixes the author would likely want,
   given the bundles crash.
3. Rank counts for the Sulis rows: 32 (to match today's ML-PACE, MACE and GAP rows) or
   128 (full node), or both? And the cell sizes, given risk 5.
4. Should the ace-jax CPU rows also enable `radial_table` and other CPU-only export
   options, or stay at the GPU rows' settings for a like-for-like comparison?
5. Once upstream publishes `lammps-jax-pjrt-cpu`, should we move to their pinned jax
   (currently 0.10.1)? That would mean exporting CPU bundles with an older jax than
   ace-jax's tests use.
