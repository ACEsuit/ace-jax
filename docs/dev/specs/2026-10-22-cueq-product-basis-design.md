# cuequivariance product basis (design)

Status: draft for review, 2026-10-22. Evidence: the throwaway spike on `exp/cueq-product`
(`bench/cueq_product/README.md`; Sulis jobs 2257962-2258831).

## Goal and scope

Add NVIDIA cuequivariance (cueq) as an **opt-in GPU form of the product basis** of a lean linear `ACEModel`, in
ASE/JAX evaluation and in lammps-jax GPU bundles.

- **Phase 1: single-species models.** The strong case: on one A100 in LAMMPS, float64, 32,768 Si atoms, gated,
  o4d16 0.671 against the best DAG 0.953 µs/atom-step (1.42x) and o4d20 1.611 against 3.135 (1.95x; patched
  ML-PACE 2.42).
- **Phase 2: TrACE (species-embedding) models** (`build_embedding_model`). Same machinery, a different transform;
  standalone A100 step 1.05-1.25x (float64) and up to 1.39x (float32), product stage 3-7x, cheap compiles.
- **Out of scope (user decision, 2026-10-22):** categorical multi-species models (the spike's per-atom-weight form
  gave 1.16-1.36x), PACE models, the CPU path, training/fitting (rows, LML, learned radials), and any automatic
  selection of cueq.

Success: a user of a single-species o4d16+ model on an A100 gets the gated speed-ups above from one documented
option, in ASE and in LAMMPS; results agree with the reference evaluation to the bundle parity tolerances; nothing
changes for users who do not opt in, and nothing imports cueq unless they do.
The Si linear ACE ladder is re-benchmarked with cueq and shown on the docs Performance page as an extra code
option beside the existing linear ACE lines.

## What the spike established (the design rests on these)

1. **Expression.** The folded site energy `e_i = Σ_v ctilde[v, z_i] Π_t A_i[s_v,t]` (`ACEModel._readout_folded`) is
   one cueq `SegmentedPolynomial`: one `SegmentedTensorProduct` per order k (`[A]*k -> out`), one path per
   monomial, the coefficient on the path; evaluated by `cuex.segmented_polynomial(method="uniform_1d")`. cueq's
   built-in `symmetric_contraction` does not fit categorical ACE (channel-diagonal; ACE monomials mix n).
2. **Layout.** Atoms in the u mode, U = 32 per group (A reshaped to (n/U, n_A, U)); the obvious batch-axis layout is
   6x slower than ace-jax.
3. **Compile.** NVRTC compile per kernel grows superlinearly with paths: chunk the polynomial into kernels of at
   most 1000 monomials and sum (o4d20: 64,187 monomials, 65 kernels). Cold compile tens of minutes; the persistent
   NVRTC cache (`CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR`) makes later runs load cubins, and a cache shipped next to a
   bundle suffices.
4. **Export.** cueq's CUDA lowering bakes a process-local plan handle into the custom call. lammps-jax's
   `contrib/ffi-replay` (as `examples/export_mace.py export cueq`) records the plan compiles at export
   (`<bundle>.ffi-replay`) and a shim replays them in LAMMPS; this works unchanged for ACE, single species and TrACE.
   Export is slow: o4d20 ~13 min (tracing 65 kernels and their transposes), against 3-4 s for the DAG.
5. **Row blocks.** cueq needs large dense row blocks: at `block_rows_for`'s choice (1024 at o4d20) it is no faster
   than the DAG; at 8192 it is 1.95x the best DAG. Consumer cards (RTX 4000 Ada, fp64-poor) gain nothing in float64.
6. **TrACE.** The embedding model's many-body spec is channel-diagonal and uniform across channels within an order,
   so one channel-free polynomial serves every channel (paths = monomials / d). Layout u = 32 atoms × d channels;
   coupling + per-centre readout fold into weights W[path, k, z] gathered per atom (no sorting).
7. **Parity** at roundoff in float64 for every model and both forms (site energies ≤ 5e-15, forces ≤ 1.2e-14
   relative; LAMMPS gates pass).

## Design

### Interface

- **`aa_dag="cueq"`** becomes a fourth product form beside None (chain), "prefix" and "greedy", accepted wherever
  `aa_dag` is: `with_aa_dag`, `ACECalculator(aa_dag=...)` and `export_lammps(aa_dag=...)`; `aj export` gains an `--aa-dag` flag
  (`auto` default, `chain`, `greedy`, `prefix`, `cueq`), which it lacks today.
  It applies after `lean`, as the DAG does. `"auto"` never selects it (opt-in only: optional dependency, minutes
  of compile, minutes of export).
- **Refusals** (clear `ValueError` / `CueqUnavailable`, never a silent fallback): no cueq installed; not a linear
  `ACEModel` (PACE, wrappers other than the lean base); categorical model with more than one species; platform
  not CUDA. A model the form does not support keeps working with `aa_dag="auto"`.
- **Bundle record:** `ace_jax.aa_dag = "cueq"` (as now for the DAG) plus `ace_jax.cueq` =
  `{"version", "targets": ["uniform_1d_cuda"], "replay": "<bundle>.ffi-replay", "cache": "<bundle>.cueq-cache",
  "handlers": "<LAMMPS_JAX_FFI_HANDLERS line>"}`, mirroring PR #112's `ace_jax.ffi` (`{target, library, handlers}`);
  `aj export` prints the environment lines to set, as #112 does for its kernel.

### Components

1. **`ace_jax/eval/cueq.py`** (new; imports `cuequivariance`, `cuequivariance_jax` inside functions only, as
   `tests/test_core_deps.py` requires):
   - `polynomial(model) -> CueqPlan`: from the lean model's `aa_specs` and folded readout, the chunked
     `SegmentedPolynomial`s (≤ `CHUNK` = 1000 monomials each), coefficients baked as path coefficients (single
     species). Phase 2: `trace_polynomial(model)` from the embedding spec, channel-free paths, weights W[path, k, z].
   - `with_cueq(model) -> model`: the model with the plan attached (a static field, as `aa_dag` holds its plan), its
     product stage replaced by the cueq evaluation in the atoms-in-u layout (U = 32; padding rows masked).
   - `available()` / `CueqUnavailable`, and the `LD_LIBRARY_PATH` / NVRTC-cache notes in one place.
2. **`eval/aa_dag.py`**: `with_aa_dag(model, "cueq")` dispatches to `cueq.with_cueq`; `resolve_aa_dag` unchanged
   (auto never returns "cueq").
3. **`export/lammps.py`**: for `aa_dag="cueq"`:
   - export in a fresh process state for the replay recorder (one bundle per process: plan handles are registry
     indices), installing lammps-jax's `ffi_replay_record` on `compile_uniform_1d_ctypes`, then `dump` to the sidecar;
   - pass `custom_call_targets=("uniform_1d_cuda",)` to lammps-jax's `export_model`;
   - warm the NVRTC cache into `<bundle>.cueq-cache/` during export (the export already compiles every kernel), so
     the bundle ships with it;
   - **row blocks:** `block_rows_for` gets a cueq rule: the largest of 8192, 4096 (halving) that fits the memory
     budget, instead of the DAG's per-block element target. (Independent of PR #115, which sets the standalone dense
     default; the DAG's own bundle target is a separate, measured question.)
4. **`calc/point.py`**: `ACECalculator(aa_dag="cueq")` passes through; `calc.aa_dag` reports it; a readout-only
   `calc.model` swap re-plans (single species bakes coefficients: recompile, minutes without the cache; documented).
5. **Dependencies:** a `cueq` extra (`cuequivariance`, `cuequivariance-jax`, `cuequivariance-ops-jax-cu12`,
   `cuequivariance-ops-cu12`, pinned to the tested 0.13.x), Linux x86_64 + CUDA 12 only; not in `cuda` or the dev
   group by default (CI has no GPU).

### Data flow (one step)

positions → neighbours → radials/Y → A (lean blocked, per row block) → **cueq polynomial chunks** (atoms in u;
single species: baked coefficients; TrACE: A' gathered per channel and W gathered per atom) → site energies →
energy; forces and virial by `value_and_grad` through cueq's JVP/transpose rules (as today). In LAMMPS the same
program runs from the bundle, the replay shim resolving `uniform_1d_cuda`.

### Errors and edge cases

- Atom count not a multiple of U: pad rows to the group size inside the block, masked out (padding rows already
  exist in the dense layouts).
- Missing NVRTC cache at run time: works, slowly (cold compile); log a one-line warning with the cache path.
- Replay sidecar missing or from another cueq version: the shim fails at first call; the bundle records the cueq
  version, and `aj export` documents that bundle, sidecar and cache travel together.
- float32: supported by cueq; same tolerances as ace-jax's float32 bundles.

## Testing

- **CPU CI (no cueq):** refusals and the bundle metadata schema with cueq mocked out; `test_core_deps` (no
  module-level import); `resolve_aa_dag("auto")` never returns "cueq".
- **GPU, opt-in** (`ACEJAX_REQUIRE_OPTIONAL`-style skip, run on lestrade/Sulis, not GitHub CI): parity of
  `with_cueq` against the full model (site energies ≤ 1e-12, forces ≤ 1e-10 relative, float64) on the Si fixture
  and an o4d12 model; float32 within ace-jax's float32 tolerances; a 64-atom LAMMPS bundle gate through ffi-replay;
  a readout swap; phase 2 the same on SiGe and Cantor TrACE fixtures.
- **Benchmark: re-time the whole Si linear ACE ladder with cueq** (the accuracy-cost `ace` sizes, o3d8-o4d20, the
  same fits) on one Sulis A100 at 32,768 atoms, float64, each bundle through the 64-atom parity gate first, with
  one export per model rather than per gate cell (the spike's `export_one` pattern; o4d20 exports in ~13 min). The
  rows go to their own results file and through `make_cost.py` into the GPU cost file (the CPU row has none: cueq is
  GPU-only), as a new family, "linear ACE, ace-jax + cueq", in `plot_cost.FAM` (linear ACE's blue, its own marker
  and line style, like the other same-fits-other-code lines) and in `plot_cost_web.PRESETS` ("Same model, other
  code", and "Headline" if it is the fastest linear ACE code on the A100). Regenerate every figure from the
  reproduce block in `docs/dev/accuracy-cost-results.md`; add the line, its "How the cost was measured" bullet and,
  at o4d20, its row in the "At the target" table to the user Performance page, ASD-STE style; update
  `docs/dev/benchmarks.md`. Small sizes may not gain (the spike's o4d12 standalone gain was 1.06-1.4x); the line
  shows them as measured.

## Documentation

- `docs/user/howto/lammps.md`: a "cuequivariance on the GPU" section (install the extra, build lammps-jax's
  `contrib/ffi-replay` shim, the environment lines `aj export` prints, the cache), with the cold-start and export
  times stated, ASD-STE style.
- `skills/ace-jax/SKILL.md` and `CLAUDE.md`: the new `aa_dag` value and its refusals.

## Risks and open questions (for review)

1. **Export time** (~13 min at o4d20) and **cold start** (tens of minutes without the shipped cache). Accept for
   phase 1 with the cache shipped; reducing trace time (fewer, larger chunks trade against superlinear NVRTC
   compile) is a follow-up.
2. **Dependency on lammps-jax `contrib/ffi-replay`**: a contrib shim, not a released component, and it needs libffi
   (Sulis has `libffi.so.6`, not `.so.8`). Ask the lammps-jax maintainer whether the shim can ship with the plugin.
3. **cueq version churn:** the recorder wraps a private ctypes symbol (`cuequivariance_ops_jax._common.library.
   compile_uniform_1d_ctypes`). Pin the extra; the GPU tests catch a break.
4. **Readout swaps recompile** for single species (coefficients baked). A shared (not per-atom) weight operand might
   keep most of the speed and avoid it; unmeasured. Phase 1 documents the recompile.
5. **TrACE gain is modest** because its step is dominated by radials/A/backward at ace1's cutoff. Phase 2 is worth
   doing for the shared machinery, but the bigger TrACE lever is ace-jax's own radial path.
6. **The bench's 64-atom gate overflows ghost capacity for TrACE** at ace1's rcut (spike gated on larger cells): fix
   the capacity estimate before TrACE benchmark rows.
