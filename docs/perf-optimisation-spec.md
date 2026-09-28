# ace-jax speed-ups (profiling tier 1 + chunking): spec

Status: agreed 2026-09-27. Branch `perf/ace-jax-speedups`, stacked on
`feat/bench-scaling`. Evidence: `docs/pace-performance-gap.md`, the profiling
report whose prototypes (`bench/perf/`) this spec turns into library changes.

## Goal

Close most of the GPU gap between ace-jax and ML-PACE on the same `.yace`
models, for both the ASE calculator and LAMMPS (lammps-jax), without changing
results. The gains must also reach `ACEModel` (linear ACE).

Today, on Cantor medium (float64, A100, 8192 atoms):
- the model call takes 7.0 ms, against ML-PACE's 3.65 ms;
- the calculator does about 370k atom-steps/s end to end, against ML-PACE's
  2.1–2.3M;
- LAMMPS does about 470k atom-steps/s and runs out of memory at 32k atoms.

The report traces the end-to-end gap mostly to calculator overhead: a
neighbour list built on every call, eager argument building and a Python
species map. On the model itself, the cost is SBessel math, a dense A assembly
that is 85% wasted work, product-basis scatter adjoints, and the force
scatter.

## Decisions

| question | decision |
|---|---|
| scope | the six measured changes (report §7 #1–#6), plus chunking over atoms (#9). Pallas kernels (#7, #8), the per-element product basis (#10) and spline radials (#11) are left for a later PR |
| models | the shared machinery serves both `PACEModel` and `ACEModel`. The model-level changes are made for both. `ACEModel`'s are designed from a profile of it, which is the first step for that model (none exists yet) |
| code shape | **replace in place**: one code path per model. The current forms survive only as test oracles (frozen reference outputs). There are no run-time flags |
| neighbour reuse | **on by default** in `ACECalculator`, with automatic rebuild. `skin=0` gives today's rebuild-every-call behaviour |
| base | stacked on `feat/bench-scaling`, which has the native `neighbour_matrix` path, the jitted calculator, `last_timing` and the scaling harness. It merges after #7 and the benchmark PR |

## Components

### 1. Skin neighbour state and the one-call step (`calc/skin.py`, used by `ACECalculator`)

Each `calculate` call:
1. **Checks whether the stored list is still valid.** It is valid when the atom
   count, species, cell and pbc are unchanged, and no atom has moved more than
   `skin/2` since the last build. The displacement check is a max-norm
   reduction fused into the compiled step, which returns it as a flag.
2. **Rebuilds the list when it isn't.** `neighbour_matrix` at `rcut + skin`,
   via `matscipy_neighbours` on the GPU through DLPack when JAX runs there,
   gives `idx`, the periodic shift vectors, `count`, `rev` (below) and the
   per-atom species indices. The species map is vectorised with a lookup table
   rather than a Python loop. `K_skin` is rounded up so compiled shapes stay
   stable.
3. **Evaluates in one `jit` call:**
   - form `rij = x[idx] − x + shift`;
   - compact each row to the neighbours inside `rcut`, with a cumsum plus
     search, gathers only, into a fixed width `K`;
   - evaluate E, F and V.

   Positions go to the device once and one packed result comes back.
4. **Handles a failed check.** If the flag reports an atom moved beyond the
   tolerance, the result is discarded and the call re-runs from step 2. That's
   rare in MD and costs one extra call, so results are always exact.

Also:
- **Default skin:** 1.0 Å. `skin=0` rebuilds every call, as today.
- **Layouts:** reuse applies to the dense layout. The sparse layout, kept for
  low-fill systems such as clusters in vacuum, still rebuilds every call.
  Because chunking (component 2) bounds memory, `layout="auto"` judges only
  fill, not memory.
- **Unchanged interface:** `last_timing` (where `nlist_s` now counts only
  rebuilds, and a `rebuilds` counter is added), `last_layout`, and the ASE
  interface.

### 2. Chunked evaluation (`eval/edge_model.py`, shared)

`energy_forces_virial_dense` runs the per-node work with `lax.map` over node
blocks, and `jax.checkpoint` recomputes each block in the backward pass. The
dense layout only: the sparse layout's memory is already O(edges), and blocking
an edge list would need dynamic per-block edge ranges. With chunking, dense
fits at any N, which is what makes `layout="auto"` a pure fill decision.
- **Block size:** 16,384 nodes by default. Below one block the code path is the
  same as today's.
- **Memory:** peak memory scales with the block size, not N.
- **Accumulation:** forces and virial add up across blocks, and neighbour
  indices may point into other blocks.

### 3. Reverse-edge force gather (`eval/edge_model.py`, `eval/nlist.py`, shared)

`rev[i, k]` is the slot in row `idx[i, k]` that points back to `i`. It is built
on the host once per rebuild (`nlist.reverse_slots`, matched per periodic
image), which runs every 20–50 MD steps at most. Forces become

    F_i = Σ_k g[i,k] − Σ_k g[idx[i,k], rev[i,k]]

which is a gather in place of today's `.at[idx].add` scatter.
- If `rev` is unavailable, or a reverse slot falls outside the compacted `K`
  window, the scatter is used instead. Forces are exact either way.
- The LAMMPS bundle, which regroups LAMMPS's list itself, uses the scatter
  unless its regroup can supply `rev` cheaply.

### 4. SBessel recurrence (`eval/pace_radial.py`, PACE)

`sin(kx)` and `cos(kx)` come from one `sin` and `cos` by Chebyshev recurrence,
with one reciprocal instead of a division per k. The report measured −20% on
the model call.

### 5. Pool-first A and a feature-major product basis (`eval/edge_model.py` helper, `eval/pace_model.py` hooks)

Pool-first is a shared `EdgeSiteModel` helper. Every radial here is linear in
per-pair coefficients over a fixed per-edge basis (PACE `g_k` with `crad`, ACE
polynomials with `Wnlq`, the factorised table with its embedding), so the
models supply that basis and the weights `W`, built from the trainable
coefficients inside the trace.

- **Pool first:** pool `g_k ⊗ Y_lm` per (node, neighbour-species channel)
  first, then apply `crad[zi, μ]` per node. There is no per-edge R_nl, and the
  species one-hot covers 8 columns instead of 26.
- **Feature-major:** A is produced as `(n_A, n)` by construction, so
  product-basis gathers read contiguous rows and their adjoints add whole rows.

### 6. `ACEModel` equivalents (`eval/model.py`)

1. **Profile `ACEModel` first:** use the report's harness (`bench/perf/`: Modal
   A100, `jax.profiler` plus HLO timings) on the benchmark's `ace_*` models,
   for each radial kind (`spline`, `spline_factorised`, `analytic`).
2. **Then change it where the profile shows the cost.** Expected from its
   structure:
   - its radial basis is linear in a fixed basis, so pool-first applies: pool
     `P(r) ⊗ Y`, then apply the radial weights;
   - a feature-major product basis;
   - speeding up whichever radial evaluation dominates.
3. **Keep a change only if** it is measured faster and passes parity. The plan
   has a decision point after the profile, recorded in the ledger.

### 7. LAMMPS bundle (`export/lammps.py`, `bench/scaling/run_lammps.py`)

- **Owned rows:**
  - `make_energy_fn` evaluates rows only for LAMMPS-owned atoms:
    `n_rows = ⌈1.1 · max_owned⌉`.
  - Senders are always owned, and LAMMPS numbers owned atoms first, so the
    static cap is safe. Receivers may still be ghosts, whose forces come out
    of the gradient as today.
  - Exceeding the row or slot capacity gives NaN, never a silent truncation.
  - The export records `max_owned` alongside `max_atoms`, as `ace_jax.owned_rows`
    (lammps-jax reads a `max_owned` key anywhere in the bundle as its own).
- **Slots sized for `rcut + skin`** (amended in Task 11): the plan sized
  `k_dense` and `max_edges` for `rcut` alone, on the belief that lammps-jax
  drops pairs beyond the cutoff before packing. It doesn't: its edge count is
  LAMMPS's neighbour list, skin included, and the A100 runs overflowed ("edge
  capacity exceeded", Cantor 256 atoms: 19968 = 256 × 78 edges, the
  rcut + 1 Å coordination). `max_edges` still counts owned rows only.
- **Model changes:** the bundle calls the same optimised model code
  (components 2–6).
- **Unchanged:** `pair_style jax/kk`, the bundle contract, `layout="auto"` and
  the child-process export.

### 8. Measuring the result (`bench/perf/`, `bench/scaling/`, `docs/`)

- **Micro-benchmarks:** keep a trimmed `bench/perf/` harness and time each
  component before and after in the same Modal A100 container. Record the
  results in the PR.
- **Standalone timing:** make it MD-like. Each timed call displaces the atoms
  by 1e-3 Å of noise, so skin reuse behaves as in MD, and rows record the
  rebuild count.
- **Scaling suite:** re-run the ace-jax rows on moriarty GPU and CPU and on the
  Modal A100, standalone and LAMMPS. The ML-PACE and MACE rows are unaffected
  and kept.
- **Docs:** re-render `docs/benchmarks.md` with before and after, and link the
  report.

## Parity (every change, float64 and float32)

- **Frozen reference fixtures,** recorded before any change: E, F and V from
  the current code for:
  - PACE, every radial kind in the fixtures;
  - `ACEModel`, with spline, spline_factorised and analytic radials, and with a
    pair term;
  - on open clusters and periodic cells.

  The optimised code must match them to 1e-12 relative in float64. In float32
  the gate is accuracy, not bit-level agreement with the old float32 rounding:
  each float32 result is compared with the float64 reference and must be no
  worse than the old float32 result was, i.e. at most max(1e-5,
  1.25 × old-float32 error) relative. (Two independently rounded float32
  results differ by about their own error, around 1e-4 here, so a 1e-5 gate
  on new-vs-old float32 would reject even a more accurate rewrite.) Forces on
  a configuration whose float64 reference forces are below 1e-3 eV/Å
  (max|F64| < 1e-3) are pure float32 rounding noise, so their factor is 2.0
  instead of 1.25; energy and stress keep 1.25 (Task 5 ruling).
- **Existing parity tests stay green:** ML-PACE, python-ace, Julia
  (julia-parity CI), and `tests/test_export_lammps.py` (bundle vs calculator
  at 1e-10).
- **Skin-list tests:** results must equal a fresh rebuild to 1e-12:
  - just below the rebuild threshold, and just after a rebuild;
  - after cell, species, pbc and atom-count changes;
  - with `skin=0`.
- **Benchmark parity gates:** re-run as part of the suite, with the thresholds
  in `bench/scaling/parity.py`.

## Success criteria

Cantor medium, float64, A100, 8192 atoms:

| path | now | target |
|---|---|---|
| model call alone | 7.0 ms | ≤ 4.5 ms (ML-PACE 3.65 ms) |
| calculator end to end | about 370k atom-steps/s | ≥ 1.1M |
| LAMMPS | about 470k atom-steps/s | ≥ 1.0M, and runs at 32k and 131k atoms |

- **Missed targets** don't block the PR. It reports the measured numbers and
  explains the shortfall.
- **Hard gates:** parity, and the full test suite and CI green.

## Out of scope (a later PR)

- Pallas/Triton kernels for the product basis and per-edge work (report #7,
  #8).
- The per-element product basis (#10) and spline radials (#11).
- Adopting lammps-jax's own layouts: `max_neighbors` with `max_owned`, and
  `force_output="edge-force"`.
- CPU-specific tuning. The CPU gap is unmeasured; the re-run CPU rows
  will show what these changes do there.
- ACEpotentials.jl rows and Phase B's production model (a separate
  benchmark follow-up).

## Risks

- **`jax.checkpoint` recompute** costs about one extra forward per block, but
  only above one block. The micro-benchmark checks there's no regression at
  8k atoms.
- **A validity check that is too eager** would rebuild too often. The
  benchmark's rebuild count and a test of the threshold catch this.
- **ACEModel profile:** its profile may show a different cost structure. The
  decision point (component 6) keeps only changes that pay.
- **Floating-point reordering:** pooling first and gathering instead of
  scattering change the order of summation. Parity is judged at 1e-12
  relative against frozen fixtures, not bit-for-bit. Bit-identical results are
  not a goal.
