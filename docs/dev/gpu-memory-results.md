# GPU memory of the ace-jax force call (2026-10-22)

Phase 1 of the profile-driven GPU memory spec (22 Oct 2026): where the device
memory of an energy + force call goes, and the largest system that fits on one
40 GB A100. Reference workload: the linear ACE o4d20 Si model of the
accuracy-cost benchmark (`runs/accuracy_cost/ace/o4d20/model.npz`, rcut 5 A),
float64, through `ACECalculator` as deployed (skin list 1 A, dense layout,
lean, `aa_dag="auto"` = greedy DAG product basis).

**Result: the goal is already met.** The baseline runs 4.4M atoms on a Sulis
A100-PCIE-40GB under JAX's default allocator cap (31.8 GB of the 40 GB), at
5.3 KB/atom peak. Every large buffer is bounded by the 16384-row block that
`energy_forces_virial_dense` already evaluates under `lax.map` + `jax.checkpoint`;
what grows with N is the pair list (1.6 KB/atom resident, 3.7 KB/atom of
skin-step temporaries). The spec's premise (about 100k atoms, about 400 KB/atom)
matches the LAMMPS bundles before they gained row blocks
(`perf-lammps-large-n.md`: dense ACE out of memory at 131k), not the calculator.
By the spec's stop rule (max atoms >= 1M with no speed regression) Phase 2 does
not start; the measured headroom is below. One measured option is applied: on
the GPU the dense rows now run in blocks of 4096, not 16384
(`edge_model.default_chunk`, below), which cuts the fixed per-block memory from
3.9 to 1.1 GB at the same step time (the largest reliable system, 3.65M
atoms at the default cap, is unchanged).

## Harness

`bench/gpu_memory/` (the spec's `benchmarks/gpu_memory/`, under the repo's `bench/`; one command per step, from `bench/`):

    python -m gpu_memory.run sweep  --model M.npz [--dump-dir D]   # 4 sizes x bulk/slab/crack x E+F/E
    python -m gpu_memory.run bisect --model M.npz                  # largest bulk cell whose E+F call runs
    python -m gpu_memory.run check  --model M.npz --calc-kw JSON   # parity, FD, NVE against the default
    python -m gpu_memory.report results.jsonl                      # the tables below
    python -m gpu_memory.attribute <dump dir> --n atoms            # peak memory by stage

On Sulis, `bench/gpu_memory/sulis.sbatch <cmd> ...` (staging in its header).
Each case runs in its own process with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
A row records `peak_bytes_in_use` after the neighbour-list build and after
20 MD-like calls (each displaces every atom by N(0, 1e-3 A), so the skin list
is reused); the compiled step's `memory_analysis()`; and the "program peak",
resident bytes + temp + outputs, which equalled the process peak for every
E+F case. `--variant`/`--calc-kw` run any `ACECalculator` option through the
same cases; `--chunk` and `--aa-chunk` are harness-only experiments that patch
the dense row-block size and the DAG product basis's column chunk in the child
process (no `src/` change).

**Attribution** (`gpu_memory.attribute`) reads XLA's text dump of the step
(`--xla_dump_to`): the live-range file gives the flattened schedule, while
bodies included, and the values live at its peak; the buffer assignment gives
their addresses in XLA:GPU's single preallocated-temp allocation, so aliasing
values (in-place updates) count once; each value's instruction gives its
Python stack, whose innermost ace-jax frame names the stage. Roles: "residual"
is a forward value recomputed under `jax.checkpoint` (`rematted_computation`),
"cotangent" a reverse-mode value, "forward" anything else. "Slack" is temp not
live at the peak: heap-packing fragmentation.

Workloads: diamond Si `bulk` (8000, 32768, 64000, 97336 atoms), a `slab` 43 A
thick with 20 A of vacuum, and a `crack`, bulk with a 4 A slit through half the
cell, all rattled by 0.02 A. Results: `bench/gpu_memory/results/sulis-a100*.jsonl`;
dumps in `/storage/.../runs/gpu_memory/dumps` (not committed).

## Breakdown at the peak (bulk, 32768 atoms, E+F)

Bytes per atom live at the program's peak, inside the checkpointed backward of
one 16384-row block (A100, `baseline-bulk-32768-ef`):

| at the peak | forward | residual | cotangent | total |
|---|--:|--:|--:|--:|
| per-pair radial/harmonic | 0 | 28,560 | 0 | 28,560 |
| A basis | 0 | 1,404 | 0 | 1,404 |
| product basis | 0 | 12 | 72,724 | 72,736 |
| pair list | 1,808 | 0 | 354 | 2,162 |
| other | 240 | 0 | 0 | 240 |
| argument: skin list | | | | 1,612 |
| argument: model | | | | 49 |
| outputs | | | | 24 |
| temp slack (not live at the peak) | | | | 15,761 |
| **program total** | | | | **122,547** |

- **The product basis dominates, not the per-pair tensors.** 59% is one
  buffer: the cotangent of the DAG's order-2 level L2, f64[18181, 16384]
  (2.4 GB), which the order-3 level's backward scatter-adds into
  (`aa_dag.py:_contract_cols`, in place over L2's own values). The B basis
  itself is never formed (the folded readout contracts `ctilde` from the DAG
  levels, `dag_contract`), and the energy-only call never materialises L2
  (fused into the readout): its peak is 14.7 KB/atom.
- **Per-pair tensors are 23%,** about 1.0 KB per pair at 28 pairs per atom: the
  masked R_nl f64[E, 71] (twice: the recomputed residual and the value its
  `where` reads) and the Y_lm columns, recomputed per block.
- **Against the spec's estimate** (about 4 KB per pair, 35 pairs, tripled by
  residuals: about 420 KB/atom): the per-pair cost is about 4x smaller per pair
  and is paid per block, not per atom, and the largest term is one the estimate
  did not include.
- **Neighbour padding is negligible here.** K = 28 slots; fill (edges / (n K))
  is 1.00 in bulk, 0.96 in the slab and 0.98-0.99 in the crack, so a flat pair
  list (spec 2.2) could save at most 4% of the per-pair stage.

The block size is ceil(n / ceil(n / 16384)), anywhere from 8193 to 16384 rows,
so the fixed part moves with n: the 33856-atom slab (3 blocks of 11286) peaks
at 2.86 GB against 4.06 GB for the 32768-atom bulk (2 of 16384). That, not the
vacuum, is the slab/bulk difference in the table below.

## Per case (A100, float64, baseline)

"peak": `peak_bytes_in_use`; "program": resident bytes + the step's temp + outputs;
us/atom-step: the median calculator call (host work, H2D, D2H included) over 20
MD-like calls. Energy-only ("e") process peaks at 8k include about 0.4 GB of
compile-time autotuning scratch, which the program peak excludes.

| cell | atoms | mode | K | fill | peak GB | program GB | temp GB | KB/atom (program) | us/atom-step |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| bulk | 8000 | ef | 28 | 1.00 | 1.80 | 1.80 | 1.78 | 225.1 | 4.39 |
| bulk | 8000 | e | 28 | 1.00 | 0.67 | 0.23 | 0.20 | 28.3 | 1.26 |
| bulk | 32768 | ef | 28 | 1.00 | 4.06 | 4.06 | 3.96 | 123.8 | 4.83 |
| bulk | 32768 | e | 28 | 1.00 | 0.63 | 0.53 | 0.43 | 16.1 | 1.04 |
| bulk | 64000 | ef | 28 | 1.00 | 4.11 | 4.11 | 3.91 | 64.2 | 4.80 |
| bulk | 64000 | e | 28 | 1.00 | 0.73 | 0.66 | 0.47 | 10.3 | 0.71 |
| bulk | 97336 | ef | 28 | 1.00 | 4.28 | 4.28 | 4.07 | 43.9 | 4.18 |
| bulk | 97336 | e | 28 | 1.00 | 0.75 | 0.69 | 0.49 | 7.1 | 0.81 |
| slab | 7744 | ef | 28 | 0.96 | 1.84 | 1.84 | 1.81 | 237.9 | 4.45 |
| slab | 7744 | e | 28 | 0.96 | 0.64 | 0.22 | 0.19 | 28.4 | 1.29 |
| slab | 33856 | ef | 28 | 0.96 | 2.86 | 2.86 | 2.76 | 84.3 | 4.14 |
| slab | 33856 | e | 28 | 0.96 | 0.39 | 0.39 | 0.30 | 11.6 | 0.73 |
| slab | 65536 | ef | 28 | 0.96 | 4.24 | 4.24 | 4.05 | 64.7 | 4.80 |
| slab | 65536 | e | 28 | 0.96 | 0.73 | 0.66 | 0.47 | 10.1 | 1.03 |
| slab | 97344 | ef | 28 | 0.96 | 4.32 | 4.32 | 4.11 | 44.4 | 4.80 |
| slab | 97344 | e | 28 | 0.96 | 0.75 | 0.70 | 0.50 | 7.2 | 0.73 |
| crack | 7700 | ef | 28 | 0.98 | 1.80 | 1.80 | 1.77 | 234.3 | 4.36 |
| crack | 7700 | e | 28 | 0.98 | 0.64 | 0.22 | 0.19 | 28.5 | 1.30 |
| crack | 32000 | ef | 28 | 0.99 | 3.92 | 3.92 | 3.83 | 122.6 | 4.84 |
| crack | 32000 | e | 28 | 0.99 | 0.63 | 0.53 | 0.43 | 16.4 | 0.74 |
| crack | 62800 | ef | 28 | 0.99 | 4.07 | 4.07 | 3.88 | 64.8 | 4.70 |
| crack | 62800 | e | 28 | 0.99 | 0.73 | 0.61 | 0.42 | 9.8 | 0.74 |
| crack | 95749 | ef | 28 | 0.99 | 4.24 | 4.24 | 4.03 | 44.3 | 4.12 |
| crack | 95749 | e | 28 | 0.99 | 0.75 | 0.66 | 0.45 | 6.9 | 0.84 |

- E+F memory is flat in n above one block: 4.06 GB at 32768 atoms, 4.28 GB at
  97336. Per atom it falls from 225 KB at 8000 atoms to 44 KB at 97336.
- The slab and the crack behave like bulk at the same block size.
- Energy alone needs 0.5-0.7 GB at these sizes: the forward pass fuses the
  product basis into the readout, so only the per-pair stage is stored.

## Largest system that fits

Bulk cells by bisection on the supercell repeat (`bisect`; E+F, 2 calls each),
under JAX's default allocator cap: with preallocation off the pool is still
limited by `XLA_PYTHON_CLIENT_MEM_FRACTION` (0.75), so `bytes_limit` is
31.8 GB of the card's 40 GB.

| atoms | program peak GB | resident GB | temp GB | KB/atom | first list build s | step s |
|--:|--:|--:|--:|--:|--:|--:|
| 778,688 | 6.99 | 1.39 | 5.59 | 9.0 | 89 | 3.19 |
| 1,560,896 | 9.97 | 2.53 | 7.40 | 6.4 | 186 | 7.85 |
| 3,112,136 | 16.57 | 5.03 | 11.47 | 5.3 | 393 | 14.94 |
| 3,652,264 | 19.55 | 6.01 | 13.46 | 5.4 | 476 | 16.20 |
| 4,410,944 | 23.48 | 7.12 | 16.25 | 5.3 | 608 | 18.80 |
| 4,574,296 | out of memory (15.7 GiB temp request) | | | | | |
| 4,741,632 | out of memory (15.4 GiB temp request) | | | | | |
| 5,268,024 | out of memory (17.0 GiB temp request) | | | | | |
| 6,229,504 | out of memory (21.1 GiB temp request) | | | | | |

- **4.41M atoms fit** (5.3 KB/atom); 4.57M does not. 4,410,944 also failed
  once, in another job, on a 15.1 GiB contiguous request with 23.5 GB needed of
  31.8 GB (allocator fragmentation near the cap, below), so 3.65M is the largest
  size that has never failed.
- At this scale the marginal cost is 5.3 KB/atom: 1.62 KB/atom resident (the
  skin list: `rij0_s` (n, 48, 3) float64, `idx_s`, `rev_s`, `live_s`) and
  3.7 KB/atom of step temporaries (the skin step's (n, 48) and (n, 28)
  compaction arrays, the compacted rij and its block-padded copy, g_r, and the
  reverse-edge gather of the force assembly) that are live around the blocked
  model call. The 3.9 GB block is then a minor term.
- **Time, not memory, is the practical limit at this size**: the first skin
  list is built on the host (a sparse list, then `reverse_slots` in numpy),
  608 s at 4.4M atoms; only rebuilds use the GPU neighbour matrix. The step
  itself stays at 4.3 us/atom.

## Headroom (measured, not applied)

The spec's table, one row per variant (A100, float64, E+F; "us/atom-step" is the
compiled step, `last_timing["model_s"]` / n, median of 20 MD-like calls;
parity: the variant against the default calculator on three rattled 1728-atom
bulk cells, the crack, the slab and an 8000-atom bulk cell, max relative error
over E, F and stress):

| variant | program bytes/atom at 32k / 97k | max atoms (31.8 GB cap) | us/atom-step at 32k / 97k | parity max rel |
|---|--:|--:|--:|--:|
| baseline (rows in blocks of 16384, DAG product basis, B never formed) | 124 KB / 44 KB | 4.41M | 4.69 / 4.04 | 5.4e-14 (GPU noise: default vs default) |
| + chunking: rows in blocks of 4096 (`--chunk 4096`) | 34 KB / 15 KB | not bisected (fixed part 3.9 -> 1.1 GB: about +0.5M) | 4.43 / 4.44 | 4.6e-14 |
| + flat pair list | not attempted: fill 0.96-1.00, at most 4% of the per-pair stage | | | |
| + custom adjoint | not attempted (stop rule) | | | |
| + no materialised B: L2 in checkpointed column chunks of 1024 (`--aa-chunk 1024`) | 44 KB / 17 KB | not bisected (about +0.5M) | 5.16 / 5.38 | 4.8e-14 |
| both (`--chunk 4096 --aa-chunk 1024`) | 19 KB / 9.7 KB | not bisected (about +0.6M) | 5.80 / 5.09 | (each validated alone) |

Row-block size alone (`--chunk`, step ms at 32768 / 97336 atoms; program GB):

| block rows | 1024 | 2048 | 4096 | 8192 | 16384 (default) | unblocked |
|---|--:|--:|--:|--:|--:|--:|
| step ms | 158 / 423 | 147 / 437 | 145 / 432 | 144 / 432 | 154 / 393 | 170 / 511 |
| program GB | 0.44 / 0.73 | 0.69 / 0.99 | 1.12 / 1.44 | 2.10 / 2.39 | 4.06 / 4.28 | 7.75 / 8.08 |

DAG column chunk alone (`--aa-chunk`, rows at the default 16384):

| columns | 512 | 1024 | 2048 | 4096 | none (default) |
|---|--:|--:|--:|--:|--:|
| step ms | 182 / 560 | 169 / 524 | 158 / 484 | 157 / 481 | 154 / 393 |
| program GB | 1.43 / 1.70 | 1.43 / 1.70 | 1.64 / 1.88 | 1.95 / 2.20 | 4.06 / 4.28 |

What is left, largest first:

1. **The O(N) step temporaries (3.7 of 5.3 KB/atom at 4.4M atoms).** The only
   lever that moves the largest system much. Running the whole skin step, not
   just the model, over row blocks (compaction, model, reverse-edge force
   gather per block, F accumulated) would leave the skin list and F resident:
   about 1.7 KB/atom, about 3x more atoms at the same cap. None of the spec's
   Phase 2 items address it.
2. **The fixed block (3.9 GB at the default 16384 rows).** Rows in blocks of
   4096-8192 cut it to 1.1-2.1 GB at the same speed within the timing spread
   (6% faster at 32k, 10% slower at 97k; the default's block size itself
   varies with n between 8193 and 16384 rows). The DAG column chunk cuts it to
   1.4 GB but costs 2-40% on the A100 (on an RTX 4000 Ada it was free at 8000
   atoms: 0.86 vs 1.88 GB temp, 13.2 vs 13.5 us/atom-step). Neither changes the
   largest system by more than about 0.5M atoms, and both stay exact (parity at
   GPU noise).
3. **The allocator cap.** `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` raises the pool
   from 31.8 GB to about 38 GB: 5,268,024 atoms, out of memory at the default,
   then runs at 27.2 GB (5.2 KB/atom; `baseline-memfrac0.95`, not bisected
   further). The failures from 4.57M atoms at the default cap asked for one
   contiguous 15-21 GiB temp while the program needed 23.5-27 GB of the 31.8 GB
   pool: fragmentation of XLA's BFC allocator, not a shortage. The step's temp
   is one allocation, so the largest system is about 0.75 of the pool divided
   by 5.3 KB/atom.
4. **Host time.** The first list build (608 s at 4.4M atoms) would need the GPU
   neighbour matrix on the first build too, not only on rebuilds.

Against the spec's floor estimate (A basis plus pair list): the A basis is
never held for all atoms (only per block), so the floor at large N is the pair
list, 1.6 KB/atom; the baseline is at 3.3x that, the spec's 2x not met, but its
other stop condition is: 4.4M atoms >= 1M with no change to speed.

## Applied: GPU row blocks of 4096

`energy_forces_virial_dense` with no `chunk` now takes `default_chunk()`:
`GPU_CHUNK_NODES` = 4096 on the GPU backend, `CHUNK_NODES` = 16384 elsewhere
(the CPU was not measured, so it is unchanged). The LAMMPS bundles and
`ace_jax.jaxmd` size their own blocks (`block_rows_for`) and are unaffected.
A100, float64, E+F (`default-4096` rows; step = `model_s`):

| cell | atoms | program GB (was) | step ms (was) | us/atom-step (was) |
|---|--:|--:|--:|--:|
| bulk | 32768 | 1.12 (4.06) | 145.5 (155.1) | 4.44 (4.73) |
| bulk | 97336 | 1.44 (4.28) | 434.3 (398.4) | 4.46 (4.09) |
| crack | 32000 | 1.13 (3.92) | 141.9 | 4.43 |
| crack | 95749 | 1.41 (4.24) | 385.7 | 4.03 |

The step time moves within the spread over block sizes (6% faster at 32k, 9%
slower at 97336 bulk, where the old default's 6 blocks of 16223 rows happened
to be fast; the 95749-atom crack is at the old speed). Check against the old
block size (`--chunk 16384`): parity 4.0e-14, FD 5.6e-9, NVE drift 2.29e-5
eV/atom, as the baseline.

It does not raise the largest system. At the default cap, 3,652,264 atoms runs
at 18.82 GB (19.55 GB with 16384-row blocks), but 3,944,312, 4,410,944,
4,574,296 and 4,741,632 atoms are out of memory on one contiguous 14.2-14.7 GiB
temp request (the old default ran 4,410,944 once and failed it once). At that
scale the per-atom terms dominate and the failure point is set by allocator
fragmentation, so 3.65M atoms is the reliable size for both block sizes;
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` is the lever there.

## Validation

`check`, every variant against the default calculator in one process (the default
evaluated first, then the variant's patch applied, so each compiles its own
step), A100:

| variant | parity max rel (E, F, stress) | FD force rel err (64 atoms, h = 1e-4 A) | NVE drift, 1000 x 1 fs, 4096 atoms, 300 K (eV/atom) | max deviation (eV/atom) |
|---|--:|--:|--:|--:|
| baseline | 5.4e-14 | 5.6e-9 | 2.29e-5 | 3.93e-5 |
| chunk-4096 | 4.6e-14 | 5.6e-9 | 2.29e-5 | 3.93e-5 |
| aa-chunk-1024 | 4.8e-14 | 5.6e-9 | 2.29e-5 | 3.93e-5 |

The baseline's parity is the default against a second default calculator: GPU
scatter-add order, the noise floor for every variant. The FD error is the
O(h^2) truncation of the central difference.
