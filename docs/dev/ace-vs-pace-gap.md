# ACE vs PACE on size-matched models: why the ranking flips

**Question.** The benchmark models are matched on basis functions per central
element (`bench/scaling/model_sizes.json`), yet on the A100 in float64 the
cheaper model depends on the system:

| medium, A100 float64 | ACE | PACE | faster |
|---|--:|--:|---|
| SiGe, standalone, large N | 1.9M atom-steps/s | 0.93M | ACE, 2.0× |
| SiGe, LAMMPS | 1.5M | 1.0M | ACE, 1.5× |
| Cantor, standalone, large N | 1.05M | 1.9M | PACE, 1.8× |
| Cantor, LAMMPS | 0.6M | 1.1M | PACE, 1.8× |

The numbers come from `bench/scaling/results/modal-a100.jsonl`.

**Answer, in short.** Equal n_B does not give equal cost. The cost of a force
call is roughly

  K × (per-edge width) + (per-node product-basis work)

where K is the neighbour-row width: 42 for fcc Cantor and 28 for diamond SiGe
at r_cut = 5 Å. The two model families spend their n_B budget on different
terms.

- **Cantor ACE: the per-edge term, much of it dead.** The ACE ladder uses
  order 2 on Cantor. That puts the whole budget into a wide A, and on the
  current code most of the per-edge and A-assembly work is wasted.
  - The exporter's `lmax = 4` is never reached: products stop at l = 2.
  - 41 of the 99 R_nl columns are never read by A.
  - Of the columns that are read, only 12 are nonzero on any one edge, because
    ACE1's R_nl is block-sparse in the neighbour species.
  - The dense pool forms an n_rnl × n_Y outer product, of which 4.5% is used.
  - That per-edge stage (radial, Y_lm and pool) is **84%** of the Cantor ACE
    call.
- **Cantor PACE: the per-edge term is tiny.** PACE spends the same n_B on
  neighbour-species combinations at lmax = 2 and 8 radial functions.
- **SiGe: the roles reverse.** PACE's SiGe `.yace` goes to correlation order 5,
  with 2,941 products against ACE's 991. It also has 13 SBessel radial
  functions, whose fused kernel costs more than linearly in their number. So
  PACE is heavy on SiGe both per node and per edge.

**Inherent or fixable?**

- **The Cantor ACE gap is an implementation inefficiency.** It is fixable
  exactly, to roundoff, with no change of model. The prototype in
  `bench/perf/ace_fast.py` takes Cantor_medium from 5.47 to 2.53 ms at 8192
  atoms, against PACE's 3.68. At 131k atoms it goes from 112.7 to 34.4 ms,
  against 59.6. It also speeds up every other ACE model by 1.3–1.9×.
- **The SiGe PACE gap is mostly inherent to that `.yace`** (order 5, n_AA =
  2,941). A smaller part is fixable:
  - the SBessel kernel (§4.3);
  - the product basis, which evaluates both central species' products for
    every atom (§3.3).

**Numbers in this document:**

- **Hardware:** Modal A100-80GB, JAX 0.11.2, float64. Models are
  `bench/scaling/models/*` as rebuilt in 2246d39 and b56cc4a.
- **Calls:** "efv" is the dense `energy_forces_virial_dense` call, jitted, with
  inputs on the device. Dense rows run in `CHUNK_NODES` = 16384 blocks.
- **Same card:** each A/B was run interleaved in one process on one GPU.
  - The stage profiles ran on an A100 80GB PCIe.
  - All the A/B tables ran on an A100-SXM4-80GB.
  - `device_name` is recorded in every result.

---

## 1. Structure of the benchmark models

Computed by `bench/perf/ace_vs_pace_structure.py`
(`bench/perf/results/ace_vs_pace_structure.json`).

**What the columns mean:**

- **Radial columns per edge:**
  - ACE: n_rnl as exported, with the columns an edge actually carries in
    brackets (used by A, and nonzero for the edge's z_j).
  - PACE: C × nradbase, the one-hot channel times g_k, with the live count in
    brackets.
- **lmax and n_Y:** ACE shows the exported value, with the largest value any
  A entry uses in brackets.
- **Pooled per node:**
  - ACE: `A_full` = n_rnl · n_Y.
  - PACE: `Ag` = C · n_b · n_Y.
- **Pool MACs per node:**
  - ACE: K · n_rnl · n_Y.
  - PACE: K · C · n_b · n_Y, plus the `sel_y` and W contractions.
- **Readout MACs per node:**
  - ACE: n_AA, against the gathered `ctilde[:, z]`.
  - PACE: NZ · P · n_AA. It computes ρ for every species, then selects.

| model | code | NZ | K | n_B | radial cols/edge (live) | lmax (used) | n_Y (used) | C | n_A | n_AA (orders 1, 2, …) | max ord | product gathers/node | pooled/node | pool MACs/node | readout MACs/node |
|---|---|--:|--:|--:|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|
| SiGe/small | ACE | 2 | 28 | 101 | 22 (7) | 2 (1) | 9 (4) | 1 | 19 | 117 (10, 37, 40, 30) | 4 | 324 | 198 | 5.5k | 0.1k |
| SiGe/small | PACE | 2 | 28 | 100 | 2×9 (9) | 3 | 16 | 2 | 104 | 236 (18, 176, 38, 4) | 4 | 500 | 288 | 24.0k | 0.9k |
| SiGe/medium | ACE | 2 | 28 | 469 | 63 (18) | 4 (2) | 25 (9) | 1 | 76 | 991 (18, 234, 739) | 3 | 2703 | 1575 | 44.1k | 1.0k |
| SiGe/medium | PACE | 2 | 28 | 499 | 2×13 (13) | 4 | 25 | 2 | 258 | 2941 (26, 668, 1706, 511, 30) | 5 | 8674 | 650 | 105.4k | 11.8k |
| SiGe/large | ACE | 2 | 28 | 1482 | 77 (22) | 6 (3) | 49 (16) | 1 | 101 | 3791 (20, 334, 1354, 2083) | 4 | 13082 | 3773 | 105.6k | 3.8k |
| SiGe/large | PACE | 2 | 28 | 1684 | 2×15 (15) | 4 | 25 | 2 | 330 | 19873 (30, 1377, 10332, 7830, 304) | 5 | 66620 | 750 | 149.7k | 79.5k |
| Cantor/small | ACE | 5 | 42 | 126 | 37 (5) | 2 (1) | 9 (4) | 1 | 32 | 138 (20, 118) | 2 | 256 | 333 | 14.0k | 0.1k |
| Cantor/small | PACE | 5 | 42 | 96 | 5×4 (4) | 1 | 4 | 5 | 50 | 250 (20, 90, 100, 40) | 4 | 660 | 80 | 4.4k | 2.5k |
| Cantor/medium | ACE | 5 | 42 | 447 | 99 (12) | 4 (2) | 25 (9) | 1 | 112 | 671 (35, 636) | 2 | 1307 | 2475 | 104.0k | 0.7k |
| Cantor/medium | PACE | 5 | 42 | 496 | 5×8 (8) | 2 | 9 | 5 | 180 | 1486 (40, 545, 605, 275, 21) | 5 | 4150 | 360 | 29.5k | 14.9k |
| Cantor/large | ACE | 5 | 42 | 1651 | 54 (7) | 2 (1) | 9 (4) | 1 | 52 | 2001 (25, 231, 668, 1077) | 4 | 6799 | 486 | 20.4k | 2.0k |
| Cantor/large | PACE | 5 | 42 | 1998 | 5×11 (11) | 4 | 25 | 5 | 450 | 7688 (55, 1460, 3440, 2617, 116) | 5 | 24343 | 1375 | 186.4k | 76.9k |

### What the table shows

- **Every ACE model evaluates about twice the Y_lm it uses.**
  - ACEpotentials derives `lmax` from the total degree. But a product of
    correlation order ν needs ν factors whose l values couple to 0 within the
    degree budget, so no A entry ever reaches the exported lmax.
  - The used lmax is 1–3, against 2–6 exported.
  - `real_spherical_harmonics` evaluates all (lmax+1)² columns, and the dense
    pool contracts them all.
- **About 40% of the R_nl columns are never read by A** (58 of 99 on
  Cantor_medium). The radial spline still evaluates them per edge.
- **ACE1's R_nl is block-sparse in the neighbour species.**
  - Every column is nonzero for exactly one z_j, independent of z_i. This was
    checked on every model (`ace_fast._owner`).
  - So on one edge only about n_rnl / NZ columns are nonzero: 12 of the 58
    used columns on Cantor_medium.
  - The spline gathers `coefs[z_i, z_j]` for all n_rnl columns anyway.
  - In effect ACE1 is a species-channel model like PACE, with the channel
    folded into the radial index, but the evaluator does not exploit that.
- **The dense pool wastes most of its outer product.** `pool_a_dense` forms the
  full n_rnl × n_Y outer product per node (`A_full`, 2475 per node on
  Cantor_medium) and then selects the n_A = 112 entries A uses: 4.5%. Only the
  l(r) = l(y) diagonal blocks are ever read.
- **The pair channel is wider than it needs to be.** It evaluates n_pair spline
  columns per edge (35 on Cantor_medium), only to contract them with
  `Wpair[:, z_i]` after pooling.
- **The PACE `.yace` spends n_B differently by system.** pyace's
  `number_of_functions_per_element` fills the budget with whatever is cheapest
  in its degree ordering.
  - **With 5 species** those are neighbour-species combinations at low l and n.
    Cantor_medium has lmax 2, n_Y = 9 and nradbase 8.
  - **With 2 species** it goes to high l, n and order. SiGe_medium has lmax 4,
    nradbase 13, order 5 and n_AA = 2,941, three times ACE's 991.
  - So PACE Cantor is light per edge, and PACE SiGe is heavy per edge and per
    node.
- **PACE evaluates the union of all central species' products for every atom.**
  - Each atom needs only its own species' products: 1834 of 2941 on
    SiGe_medium (62%), and 770 of 1486 on Cantor_medium (52%).
  - The ρ contraction also runs for all NZ · P outputs before selecting.
  - ACE has no such waste. Its n_AA is shared by every species, and
    `ctilde[:, z]` is gathered.

## 2. Where the time goes (8192 atoms, float64)

These are GPU kernel µs per call, from the trace (`bench/perf/profile_gap.py`,
via `modal_gap.py::profile`; `bench/perf/results/gap_profile_8192.json`).

- **Card:** one container, A100 80GB PCIe.
- **Attribution:** each kernel is attributed to stages by the call chains of its
  fused HLO instructions (`profile_ace.attribute`). A kernel spanning stages is
  split by instruction count.
- **Same attribution for PACE:** the stage map is swapped for PACE, and
  `_node_energies_t` is replaced by an op-for-op identical copy, so the product
  basis and the ρ contraction attribute separately.

| stage | ACE Cantor_med | PACE Cantor_med | ACE SiGe_med | PACE SiGe_med |
|---|--:|--:|--:|--:|
| radial (per edge) | **2058** (41%) | 503 (16%) | 907 (31%) | **2219** (34%) |
| angular Y_lm | 789 (16%) | 142 | 537 (18%) | 542 (8%) |
| core repulsion | – | 73 | – | 54 |
| pool / A assembly | **1365** (27%) | 777 (25%) | 634 (22%) | 1213 (18%) |
| product basis `aa` | 166 (3%) | **838** (27%) | 472 (16%) | **1661** (25%) |
| readout | 143 | 150 | 110 | 243 |
| embedding / tail | – | 151 | – | 381 |
| force scatter | 363 | 328 | 181 | 187 |
| strain + other | 105 | 124 | 74 | 103 |
| **kernel sum** | **4989** | **3085** | **2915** | **6602** |
| efv (ms), energy-only (ms) | 5.78, 2.45 | 3.76, 1.34 | 3.63, 1.54 | 7.40, 3.19 |
| per-edge stages / per-node stages | 84% / 6% | 48% / 37% | 71% / 20% | 61% / 35% |

"Per-edge stages" are radial, angular, core and pool. "Per-node stages" are
`aa`, readout and tail.

### The top kernels

**ACE Cantor_medium:**

| µs | kernel |
|--:|---|
| 589 | Y_lm forward (`loop_concatenate`, `f64[344064,25]`, 345 GB/s) |
| 506 | spline forward (`f64[344064,99]` ×2) |
| 466 | radial adjoint |
| 364 | pool GEMM adjoint (`f64[8192,42,25]`) |
| 348 | force scatter |
| 332 | pool GEMM forward (`f64[8192,99,25]`) |

**PACE Cantor_medium:**

| µs | kernel |
|--:|---|
| 343 | force scatter |
| 296 + 204 + 179 | three `aa` adjoint scatters (`f64[8192,180]`) |
| 229 | radial, core and angular fused |

**PACE SiGe_medium:**

| µs | kernel | note |
|--:|---|---|
| **1755** | `loop_concatenate_fusion_2`, `f64[229376,13]` | the SBessel g_k forward; 132 GB/s, so compute-bound |
| 829 | `aa` adjoint scatter (`f64[8192,258]`) | |
| 571 | pool/tail transpose | |
| 400 | Y_lm | |

The 13-column g_k kernel alone is 27% of the call. On Cantor, with 8 columns,
the same kernel is not in the top 10 (less than 110 µs). §4.3 explains why.

**ACE SiGe_medium:**

| µs | kernel |
|--:|---|
| 403 | order-3 `aa` adjoint scatter (`f64[76,8192]`, feature-major) |
| 394 | Y_lm |
| 252 | radial adjoint |

### Reading the profiles

- **The Cantor gap is entirely per edge.** The per-edge stages cost ACE
  4.2 ms, against PACE's 1.5 ms. On the per-node side ACE is cheaper: 0.31 ms
  against 1.14 ms.
- **The Y_lm kernel shows the same superlinear pattern as PACE's SBessel.**
  - The harmonics recursion is stacked into one `loop_concatenate` fusion, in
    which each output column re-evaluates its recursion chain.
  - So 25 columns (lmax 4) cost about 4× what 9 columns cost: 0.59 ms against
    0.15 ms after pruning.
  - A Y_lm evaluated beyond the used lmax is therefore expensive, not merely
    wasted.
- **The SiGe gap is PACE's radial and per-node work together.** PACE's radial
  kernel is 1.76 ms, and its `aa` + readout + tail are 2.29 ms against ACE's
  0.58 ms.

## 3. The explanation, with evidence

### 3.1 Why equal n_B is not equal cost

n_B counts functions, not work. Per atom, the work is:

- **per edge:** K × (radial columns + Y_lm columns), plus pool MACs of
  K · (pooled width);
- **per node:** the product basis (Σ_ν ν · n_ν gathers, with ν-fold products
  and their adjoint scatters) and the readout.

The same n_B can sit almost entirely in one of these two terms.

- **ACE Cantor, order 2:** 447 functions from n_A = 112 A functions, with only
  671 products. All the cost is per edge, and K = 42 multiplies it.
- **PACE Cantor:** 496 functions from species combinations of a small A (lmax
  2, 8 g_k). The per-edge work is small, the product work moderate.
- **PACE SiGe:** 499 functions from high order (up to 5) and high l and n. The
  per-edge work is large (13 g_k, lmax 4), and so is the product work (2,941
  products, 8,674 gathers per node).
- **ACE SiGe, order 3:** moderate on both, with 991 products.

The cost model reproduces the ranking.

- **Cantor, per-edge pool MACs per node:** ACE 104k against PACE 30k. ACE's
  gather work per node is smaller (1.3k against 4.2k), but per-edge work
  dominates at K = 42.
- **SiGe:** PACE is 105k against 44k in pool MACs, 8.7k against 2.7k in
  gathers, and 12k against 1k in readout MACs.

### 3.2 Why ACE carries so much dead per-edge work on Cantor

The four sources of waste are listed in §1: the unused lmax, the unread R_nl
columns, the species-zero columns and the n_rnl × n_Y pool. They multiply
together.

The prototype removes them one by one. Every variant matches the stock model to
max |ΔF| ≤ 2e-14 eV/Å (see the `parity` field in `gap_bench*.json`). All times are
ms per efv call.

| Cantor_medium, A100-SXM4 | 8192 atoms | vs stock | 131072 atoms | vs stock |
|---|--:|--:|--:|--:|
| ACE stock | 5.47 | 1.00 | 112.7 | 1.00 |
| + prune (lmax 4 → 2, n_rnl 99 → 58) | 3.69¹ | 1.48 | 69.0¹ | 1.63 |
| + prune + pairfold | 3.15 | 1.74 | 50.4 | 2.23 |
| + compact + pairfold | 2.86 | 1.91 | 40.0 | 2.82 |
| **+ compact + pairfold + fm** | **2.53** | **2.16** | **34.4** | **3.27** |
| PACE stock | 3.68 | 1.49 | 59.6 | 1.89 |

¹ From round 1 (`gap_bench.json`), in which stock ran at 5.39 and 112.6 ms; the
other rows are round 2 (`gap_bench_r2.json`).

What the fix leaves per call, from the trace of `compact+pairfold`:

- The per-edge stages drop from 4.2 ms to 1.44 ms (radial 0.51, radial and pool
  0.77, Y_lm 0.15).
- The kernel sum drops from 4989 to 2327 µs (§2 trace,
  `gap_profile_8192.json`, rows `compact+pairfold`).

### 3.3 Why PACE is slow on SiGe

- **Mostly inherent.** The product basis, readout and tail take 2.29 ms,
  because the `.yace` is order 5 with n_AA = 2,941.
- **Two parts are implementation.**
  1. **The SBessel g_k kernel** grows superlinearly with nradbase (§4.3). It is
     1.76 ms at nradbase 13.
  2. **Every atom evaluates both species' products.** Each species needs 62%
     of them. A species-sorted product basis could save up to about 38% of the
     0.84 ms `aa` adjoint plus forward. This was not prototyped: it needs
     species-grouped node blocks, a layout change.
- **The readout's NZ · P full matmul costs little** (0.24 ms).

### 3.4 Large N, small N, and chunking

efv throughput (atom-steps/s) as N grows. Rows above 16384 atoms run in
`CHUNK_NODES` blocks (`lax.map` + `checkpoint`). Sources: `gap_bench.json` and
`gap_bench_r2.json`.

| | 1024 | 8192 | 32768 | 131072 | 8192 → 131072 |
|---|--:|--:|--:|--:|--:|
| ACE Cantor_med stock | 0.79M | 1.52M | 1.11M | 1.16M | **−24%** |
| ACE Cantor_med compact+pairfold | 1.13M | 2.94M | 3.01M | 3.28M | +12% |
| ACE Cantor_med compact+pairfold+fm | – | 3.24M | – | 3.81M | +18% |
| PACE Cantor_med | 0.98M | 2.27M | 2.10M | 2.20M | −3% |
| ACE SiGe_med stock | 1.00M | 2.35M | 1.99M | 2.13M | −9% |
| ACE SiGe_med prune+lblock+fm+pairfold | – | 3.29M | – | 4.05M | +23% |
| PACE SiGe_med | 0.59M | 1.33M | 0.96M | 0.99M | **−26%** |

- **Small N is overhead-bound.** At 1024 atoms every model sits within about
  30% of a ~1 ms floor, so the ratios compress. That is why ACE looks
  competitive on Cantor at small N.
- **The large-N gap is the throughput gap plus a chunking penalty on the model
  with the larger per-row intermediates.**
  - Stock ACE Cantor holds `A_full` (n × 2475) and R_nl (n·K × 99) per block.
    It loses 24% once blocks are recomputed under `checkpoint`.
  - PACE SiGe loses 26% the same way.
  - The pruned ACE variants have small intermediates. They do not lose at all:
    they get faster per atom at large N.
- **So the chunking penalty is not a separate defect.** It is the dead-work
  inefficiency paid twice, forward and recomputed forward.
- **CHUNK_NODES itself is not the problem.** The fixed variants chunk at the
  same 16384 without loss.

### 3.5 LAMMPS against standalone

The LAMMPS bundle evaluates the same `site_energies_dense_blocked`
(`export/lammps.py:82`), in `BUNDLE_BLOCK_ROWS` = 32768 blocks. So the model
work is the same as standalone large N.

- **Cantor:** the LAMMPS ratio (ACE/PACE 0.55) matches standalone large N
  (0.55).
- **SiGe:** the LAMMPS ratio is 1.5×, against 2.0× standalone. This is
  consistent with a common per-step LAMMPS cost (neighbour list, ghosts, the
  host–device round trip) added to both models, which compresses the ratio.
  - In LAMMPS, ACE SiGe runs at 1.5M against 1.9M standalone. PACE runs at
    1.0M in both.
  - So LAMMPS takes about 0.14 µs per atom-step more for ACE. For PACE the
    difference is within noise.
- **Not re-measured in LAMMPS here.** `ACEBlocked` overrides
  `site_energies_dense`, so a bundle exported from it would pick up the same
  gain. That is untested: it needs a lammps-jax run.

## 4. Fixes: prototypes and measured gains

All are in `bench/perf/`, and none are in `src/`.

- `ace_fast.py`: the ACE variants.
- `pace_fast.py`: the PACE SBessel variant.
- `ace_gap_bench.py`: the interleaved A/B (rounds alternate order).
- `modal_gap.py`: the Modal driver.

**Parity:** every variant matches the stock model on E, F and virial to
|ΔE|/atom ≤ 2e-16 eV and |ΔF| ≤ 2e-13 eV/Å; the largest virial difference, 1.5e-10 eV, is on a virial summed over 131072 atoms (SiGe_medium), so it is relative roundoff (`parity` in the result
JSONs).

### 4.1 ACE variants (exact, static)

| variant | what it does | code in the hot path |
|---|---|---|
| `prune` | Drop R_nl columns no A entry reads, and set `lmax` to the largest used l. Columns are re-sorted by (l, z_j). | none: a data transform run by stock `ACEModel` |
| `pairfold` | Fold `Wpair[:, z_i]` into the pair spline table, giving 1 column per edge instead of n_pair. | none: a data transform, the same idea as `fold_readout` |
| `lblock` | Pool A per l-block (R_{·,l} ⊗ Y_l), with `aa_specs` remapped onto the block layout. No `A_full`, no select. | overrides `site_energies_dense` |
| `compact` | Per edge, evaluate only the M_l R_nl columns of its own z_j (a compacted `(NZ, NZ, ncoef, ΣM_l)` table), then expand by a one-hot over z_j. PACE's channel trick. Implies `lblock`. | overrides `site_energies_dense` |
| `fm` | Pool the l-blocks feature-major and hand `_aa` A = Aᵗ. | same |

The `lblock`/`compact` row order breaks XLA's choice of a feature-major A for the
order-3/4 `aa` adjoint: SiGe `aa` went from 472 to 771 µs in the trace. `fm`
restores it, so without `fm`, `lblock` is slower than plain `prune` on SiGe.

**Best variant per model**, 8192 atoms, ms per efv call, on one A100-SXM4
per row (`gap_bench_r2.json`, `gap_bench_r3.json`):

| model | ACE stock | prune+pairfold | compact+pairfold+fm | best vs stock | PACE | best ACE vs PACE |
|---|--:|--:|--:|--:|--:|--:|
| Cantor_small | 3.20 | 2.54 | **1.98** | 1.61× | 2.12 | 1.07× faster |
| Cantor_medium | 5.47 | 3.15 | **2.53** | 2.16× | 3.68 | 1.45× faster |
| Cantor_large | 4.30 | 3.59 | **3.04** | 1.42× | 11.32 | 3.7× faster |
| SiGe_small | 1.80 | 1.45 | **1.39** | 1.29× | 2.37 | 1.7× faster |
| SiGe_medium | 3.46 | 2.46 | **2.44** | 1.42× | 6.05 | 2.5× faster |
| SiGe_large | 6.95 | **4.48** | 4.62 | 1.55× | 21.16 | 4.7× faster |

At 131k atoms, Cantor_medium goes from 112.7 to 34.4 ms (3.3×) and SiGe_medium
from 60.7 to 32.4 ms (1.9×).

**Cost of each part:**

- **`prune` + `pairfold`** are free: pure data transforms at load time. They
  give 1.2–1.7× on every model.
- **`compact` + `fm`** add a further 1.1–1.3× on the multi-species models,
  where the one-hot saves the most.
  - On SiGe (NZ = 2) the gain is within noise.
  - For each model, the column with the best time is in bold.

### 4.2 End to end (ACECalculator, MD-like, skin list), 8192 atoms

These are single measurements from round 1 (`gap_bench.json`, `calc_s`). At this
size, calculator overhead adds about 3–5 ms to every model, so these ratios are
compressed.

| | Cantor_medium ms | SiGe_medium ms |
|---|--:|--:|
| ACE stock | 9.33 | 7.32 |
| ACE prune | 7.67 | 6.32 |
| ACE compact+pairfold | 6.02 | 7.70² |
| PACE | 8.60 | 12.44 |

² One sample. The efv time for the same model was 2.94 ms against stock's
3.48. The difference is calculator-side noise and overhead, which this
harness does not separate.

### 4.3 PACE SBessel as one sin plus a constant matrix (`pace_fast.sbessel_mm`)

**Why the stock kernel grows superlinearly.**

- `pace_radial._sbessel` builds the g_k by two chained recurrences:
  - the sin/cos rotation for sin(kx);
  - PACE's orthogonalisation, g_n = (f_n + a_n g_{n−1}) / b_n.
- The g_k are then stacked. XLA fuses the stack into one `loop_concatenate`
  kernel, in which each output column re-evaluates its chain.

**The alternative.** The orthogonalisation is linear with constant
coefficients, so

  g = (S · M) / r_c^1.5

where S[:, k] = sinc((k+1)x) is one elementwise sin per entry, and M is a
constant (K+1) × K matrix. It is exact to 6e-15.

| efv ms (A100-SXM4) | stock | sbessel_mm | gain |
|---|--:|--:|--:|
| SiGe_medium 8192 (nradbase 13) | 6.05 | 5.70 | −6% |
| SiGe_medium 131072 | 132.2 | 104.4 | **−21%** |
| SiGe_large 8192 (nradbase 15) | 21.16 | 19.95 | −6% |
| Cantor_large 8192 (11) | 11.32 | 11.19 | −1% |
| Cantor_medium 8192 (8) | 3.68 | 4.16 | **+13%** |
| Cantor_small 8192 (4) | 2.12 | 2.52 | +19% |
| SiGe_small 8192 (9) | 2.37 | 2.63 | +11% |

- **It helps only at large nradbase.** At K ≤ 9 the recurrence kernel is cheap,
  and the extra (E, K+1) sin array and matmul cost more.
- **So it is not a drop-in.** A switch at nradbase ≳ 12 would be the cheap
  form.
- **Even at its best it moves SiGe PACE from 0.99M to 1.26M atom-steps/s at
  131k atoms.** The rest of the SiGe gap is the order-5 product basis.
- **The same stacked-recurrence pattern costs ACE's Y_lm** (§2). Whether
  `harmonics.py` benefits from a similar reformulation was not tested here.
  With `prune`, ACE avoids most of that cost anyway.

## 5. Inherent against fixable

| gap | inherent to the model structure | fixable (implementation) |
|---|---|---|
| **Cantor: ACE slower than PACE** | Little. Order 2 at high degree does put the whole budget per edge, and per-edge cost scales with K = 42. But the fixed ACE is 1.45× faster than PACE at 8192 atoms and 1.7× at 131k. | **Almost all of it.** The dead lmax, unread R_nl columns, species-zero R_nl columns, the n_rnl × n_Y pool, the n_pair-wide pair channel, and the chunking penalty that these large intermediates cause. |
| **SiGe: PACE slower than ACE** | **Mostly.** The `.yace` goes to order 5 with n_AA = 2,941 (3× ACE), 13–15 radial functions and lmax 4. Fixing ACE widens the gap further. | Partly. SBessel at large nradbase (6–21%, §4.3). Per-species product lists (up to about 38% of `aa`, not prototyped). The NZ · P ρ matmul is minor. |
| **ACE per function, in general** | ACE's shared n_AA and gathered `ctilde` are cheaper per node than PACE's per-species union. | ACE's per-edge path evaluated about 2× the Y_lm and about 8× the live radial columns it needs, on every model. |

The brief's hypothesis was that "order-2 wide-A models cost more per function".
That is true of the stock code only because of the dead work. After the fix,
Cantor_medium ACE at order 2 costs 2.53 ms for 447 functions against PACE's
3.68 ms for 496.

## 6. Recommendations

1. **Land `prune` and `pairfold` in `ace_jax.eval.io.load`.** (Landed, but
   applied by the calculator and exporter, not `load`; see §8.) Make them a
   normalisation step next to `fold_readout`.
   - They are exact, static data transforms that need no new code in the hot
     path, and give 1.2–1.7× on every ACE model.
   - Also consider exporting the pruned tables from `julia/export_model.jl`.
     The `lmax` it writes is ACEpotentials' degree-derived bound, not what the
     basis uses.
   - Keep `site_basis` and descriptors unaffected. Pruned columns carry no A
     entry, and the pair fold is readout-only. So apply `pairfold` only on the
     folded energy path, as `fold_readout` does.
2. **Make `compact` + `lblock` + `fm` the `ACEModel` dense energy path** (landed
   in the lean form, dense only; see §8) when
   R_nl is species-block-sparse, which it is for every ACE1-compat model.
   - This adds 1.1–1.3× on the multi-species models.
   - Carry it to `site_energies` (sparse) and to the LAMMPS bundle, and re-run
     `bench/scaling`, both standalone and LAMMPS, for ACE.
   - Without `fm` the l-blocked layout regresses the order-3/4 `aa` adjoint, so
     keep `fm`.
3. **PACE: gate the SBessel matrix form on nradbase ≥ 12.** Or find a
   formulation that avoids the stacked recurrence at every K.
   - A species-grouped product basis is the larger remaining PACE lever on SiGe
     (§3.3). It needs node blocks per central species.
4. **Revisit `harmonics.py`'s stacked recursion.** The same superlinear
   `loop_concatenate` shows up there: 0.59 ms for lmax 4 at 8192 atoms, Cantor.
5. **The ACE size ladder should keep matching function count, not cost.**
   - n_B per central element is the capacity axis the benchmark compares on.
     Matching cost instead would make the comparison circular, since cost is
     what is being measured.
   - Cost also depends on K and on the evaluator, so a cost-matched ladder
     would move with every optimisation, as the fixes here show.
   - What should change is the reporting. Publish this structure table (§1)
     next to the results, so the per-edge and per-node split is visible. Note
     that Cantor small/medium are order 2 and SiGe order 3/4.
   - With the fixes landed, the ACE ranking against PACE no longer flips
     between systems: ACE is faster on all six sizes.
   - The table in §1 predicts where each model's time goes. Where n_B sits
     (orders, l, species) explains the cost, and n_B does not.

## 7. Reproduce

```bash
PYTHONPATH=bench:src uv run python bench/perf/ace_vs_pace_structure.py          # §1, no GPU
uv run --with modal modal run bench/perf/modal_gap.py::profile                  # §2
uv run --with modal modal run bench/perf/modal_gap.py::bench                    # §3.4, §4.1 round 1
uv run --with modal modal run bench/perf/modal_gap.py::bench --ns 8192,131072 \
  --variants prune+pairfold,compact+pairfold,compact+pairfold+fm,prune+lblock+fm+pairfold \
  --pace-variants sbessel_mm --extra "" --tag _r2                               # round 2
```

**Modal time used for this document:** about 26 min of A100.

| run | time |
|---|--:|
| profile | 2 min |
| round 1 | 12 min |
| round 2 | 8 min |
| round 3 | 4 min |

Plus a few minutes of containers that crash-looped at import before any GPU
work.

**Round 2's extras rows failed** (n = 0, a driver bug since fixed:
`--extra-n`). They are kept in `gap_bench_r2.json`, and round 3 re-ran them.

## 8. Landed

Recommendations 1–3 are in `src/`, on branch `perf/ace-fast-path`, with these deviations from §6:

- **Not in `load`.** The transforms are applied by the calculator and the exporter (`ACECalculator(lean=True)`, `export_lammps(lean=True)`). The pair fold would break the fit rows, which need Apair, and the fit pipeline loads with `load`.
- **The l-blocked pool is dense-only.** The sparse layout runs the pruned, pair-folded data with its unchanged code.

- **ACE: `ace_jax.eval.model.lean(model)`.** It composes three exact, load-time transforms of a folded `ACEModel`:
  - `prune_columns`: the doc's `prune`.
  - `fold_pair`: `pairfold`.
  - `block_dense`: `lblock` + `compact` + `fm`. It is species-compact only where R_nl is block-sparse in z_j and NZ > 1.

  `ACECalculator(lean=True)` and `export_lammps(lean=True)` apply it. Both default to on.
- **`load` never applies it.** Fitting, descriptors and the learned radials keep the full model. A lean model is `energy_only`: its basis methods raise.
- **The sparse layout.** Its code is unchanged, but it runs the pruned, pair-folded data. On CPU at 1024 atoms that gave 8–20% faster forces: Cantor_medium went from 46.9 to 37.6 ms, SiGe_medium from 22.6 to 18.5 ms, and SiGe_large from 92.4 to 84.4 ms. The l-blocked pool is dense-only. The sparse pool is already n_A wide per edge, so a per-l outer product would pool more, not less.
- **PACE: `PACEModel.sbessel_form`.** It is fixed by `load_yace` as `"matmul"` (`pace_radial._sbessel_mm`) when `nradbase >= 12` (`SBESSEL_MATMUL_MIN_K`), and `"rotation"` otherwise.
- **Parity.** `tests/test_lean.py` holds each transform to 1e-12 on E, F and virial. It covers five fixtures: 1, 2 and 5 species, with spline, analytic and factorised radials and a pair term in each. The largest differences were |ΔF| 1.4e-13 and |ΔV| 4.6e-13. `tests/test_perf_parity.py` (288 frozen cases) is unchanged. On the A100, the before and after energies agree to at most 1.1e-16 eV/atom.

**Before/after measurement.**

- **Setup:** Modal A100-SXM4-80GB, float64, `main` f69dfdb against this branch.
- **Timed call:** `ACECalculator`, MD-like, with the skin list.
- **Protocol:** each case ran in its own process, all in one container, with before and after alternating per case and per round (`modal_profile.py::microbench_ab`, 2 rounds × 15 reps).
- **Quantity:** `step` is the compiled skin step with inputs on the device, and `call` is the whole calculator call. Each value is the median.
- **Files:** the results are in `bench/perf/results/ace_fast_ab.json` (raw: `microbench_{before,after}_ace_fast_r2.json`). A first single-round pass (`ace_fast_ab_r1.json`) agrees within the 8192-atom noise, which is about ±10% on unchanged code.

| model | N | step before (ms) | step after (ms) | step speed-up | call speed-up |
|---|--:|--:|--:|--:|--:|
| ace_SiGe_small | 8192 | 5.78 | 2.50 | 2.31× | 1.87× |
| ace_SiGe_small | 131072 | 22.07 | 15.17 | 1.46× | 1.30× |
| ace_SiGe_medium | 8192 | 4.36 | 3.86 | 1.13× | 1.12× |
| ace_SiGe_medium | 131072 | 59.91 | 30.66 | 1.95× | 1.79× |
| ace_SiGe_large | 8192 | 7.40 | 5.98 | 1.24× | 1.38× |
| ace_SiGe_large | 131072 | 134.94 | 70.11 | 1.92× | 1.86× |
| ace_Cantor_small | 8192 | 3.88 | 2.89 | 1.34× | 1.23× |
| ace_Cantor_small | 131072 | 46.24 | 21.46 | 2.15× | 1.86× |
| ace_Cantor_medium | 8192 | 7.13 | 3.52 | 2.03× | 1.70× |
| ace_Cantor_medium | 131072 | 116.15 | 35.29 | **3.29×** | 2.92× |
| ace_Cantor_large | 8192 | 5.64 | 4.08 | 1.38× | 1.29× |
| ace_Cantor_large | 131072 | 74.04 | 43.25 | 1.71× | 1.62× |
| pace_SiGe_small (nradbase 9, rotation) | 8192 | 5.07 | 4.68 | 1.08× | 0.99× |
| pace_SiGe_small | 131072 | 50.00 | 49.97 | 1.00× | 1.01× |
| pace_SiGe_medium (13, matmul) | 8192 | 8.54 | 8.35 | 1.02× | 1.10× |
| pace_SiGe_medium | 131072 | 132.50 | 104.29 | **1.27×** | 1.26× |
| pace_SiGe_large (15, matmul) | 8192 | 22.79 | 21.50 | 1.06× | 1.12× |
| pace_SiGe_large | 131072 | 462.07 | 388.63 | **1.19×** | 1.18× |
| pace_Cantor_small (4, rotation) | 8192 | 4.20 | 4.20 | 1.00× | 0.99× |
| pace_Cantor_small | 131072 | 24.67 | 24.63 | 1.00× | 1.00× |
| pace_Cantor_medium (8, rotation) | 8192 | 5.86 | 6.26 | 0.94×¹ | 0.98× |
| pace_Cantor_medium | 131072 | 61.00 | 61.11 | 1.00× | 1.00× |
| pace_Cantor_large (11, rotation) | 8192 | 13.50 | 13.52 | 1.00× | 1.02× |
| pace_Cantor_large | 131072 | 238.03 | 237.98 | 1.00× | 1.00× |

¹ The code is unchanged here. The single-round pass measured 1.12× the other way, so this is noise.

**What the table shows:**

- **Every ACE model is faster.** The step gain is 1.13–2.31× at 8192 atoms and 1.46–3.29× at 131k.
- **The gain grows with N**, because the smaller per-row intermediates remove the chunking penalty (§3.4).
- **Cantor_medium ACE against PACE.** At 131k atoms it goes from 116 ms, slower than PACE's 61 ms, to 35 ms, 1.7× faster than PACE. At 8192 it goes from 7.1 to 3.5 ms, against PACE's 5.9.
- **The PACE switch.**
  - SiGe_medium and SiGe_large (nradbase 13 and 15) gain 1.27× and 1.19× at 131k. At 8192 the change is within noise: there are only 2 samples per side, and SiGe_medium's overlap.
  - All Cantor models and SiGe_small stay on the rotation recurrence, and are unchanged within noise.
- **Modal time:** about 38 min of A100: 16 min for the single-round pass and 22 min for the two-round pass.

**Not re-measured here:** LAMMPS itself. The bundle is built from `lean(model)` (`ace_jax.lean` in the bundle), so it runs the same `site_energies_dense_blocked`. The controller's `bench/scaling` re-run measures it.
