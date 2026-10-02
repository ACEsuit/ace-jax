# `ACEModel` GPU profile, and the pool-first / feature-major decision

Task 8 of the perf plan. This profiles the linear ACE model's force call and
decides which of PACE's model-level changes `ACEModel` should get in Task 9.

- **Hardware:** Modal A100-SXM4-80GB, one container for all six models, so the
  models are compared on one card.
- **Software:** JAX 0.11.2 (CUDA 12). The image is
  `bench/scaling/modal_app.py::base_image`.
- **What was profiled:** `energy_forces_virial_dense` in float64 at 8192 atoms.
  - Each model is loaded as the benchmark loads it: `io.load`, with the readout
    folded into `ctilde`.
  - At 8192 atoms the call runs as a single chunk.
- **Raw results:** `bench/perf/results/ace_{SiGe,Cantor}_{small,medium,large}_8192_float64.json`.

## Decision

All six benchmark `ace_*` models use the **`spline`** radial (§1). No
benchmark model uses `analytic` or `spline_factorised`.

| candidate | rule | measured | verdict |
|---|---|---|---|
| **Pool-first A, `spline` radial** | C·n_b ≤ n_rnl, **and** radial + A assembly ≥ 15% | radial + A assembly is **21.6–56.4%** of GPU time, so the profile test passes on every model. The width test fails on every model: NZ·ncoef = 204 (SiGe) or 510 (Cantor), against n_rnl = 51–228 | **skip** |
| Pool-first A, `analytic` radial | same rule | no benchmark model uses it, so there is nothing to measure | **skip** (not measured) |
| Pool-first A, `spline_factorised` radial | same rule | no benchmark model uses it | **skip** (not measured) |
| **Feature-major `_aa`** | `_aa` forward + adjoint ≥ 15% | **22–59%** on 5 of 6 models. SiGe_small is at 10.6% (upper bound 13.9%) | **apply** by the rule; **measured slower on every model in Task 9 and reverted** (§5) |

Two notes qualify these verdicts.

- **Channels for pool-first are NZ, not 1** (§3.1). The brief writes
  "channels (1 for ACE)". That is true of `ACEModel`'s current `edge_a` path,
  where species enter through the radial. Pool-first undoes exactly that: the
  radial weights depend on the neighbour species z_j, so the pooled b ⊗ Y must be
  kept per neighbour species, as PACE's `crad[z_i, z_j]` is.
  - With C = 1 the width test would pass for four models (ncoef = 102 against
    n_rnl = 108, 108, 126 and 228). But C = 1 cannot be implemented.
  - Even taking the whole radial stage as saved, pool-first with C = NZ costs
    more than it saves on every model: **−0.17 to −2.6 ms per call** (§3.2).
- **Feature-major `_aa` is applied by the rule, but its gain may be modest.**
  XLA's layout assignment **already** makes A feature-major for the dominant
  order-3 adjoint scatter on five of six models: the scatter outputs are
  `f64[n_A, 8192]` (§2.3).
  - The exception is SiGe_large. Its order-4 adjoint scatters are node-major
    (`f64[8192,165]`), taking 8.6 ms of 15.0 ms.
  - In PACE, feature-major moved the same scatter only from 1.26 to 0.99 ms
    (`docs/dev/pace-performance-gap.md` §8.3).
  - **Task 9 should measure it per model,** and not assume PACE's gain carries
    over.
  - The real remedy for the adjoint scatter is PACE's candidate #7: a custom
    kernel that keeps A and dA on chip.

## 1. The models

| model | radial kind | NZ | n_rnl | ncoef (n_b) | n_Y | n_A | n_AA (max order) | K | efv ms | energy-only ms | kernel sum ms | peak GB |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|
| SiGe_small | spline | 2 | 51 | 102 | 25 | 58 | 550 (3) | 28 | 3.49 | 1.50 | 2.92 | 0.48 |
| SiGe_medium | spline | 2 | 108 | 102 | 49 | 165 | 4918 (3) | 28 | 9.02 | 2.81 | 8.35 | 2.76 |
| SiGe_large | spline | 2 | 108 | 102 | 49 | 165 | 13919 (4) | 28 | 15.95 | 3.25 | 14.95 | 7.42 |
| Cantor_small | spline | 5 | 74 | 102 | 9 | 72 | 1964 (3) | 42 | 5.54 | 1.45 | 4.85 | 1.23 |
| Cantor_medium | spline | 5 | 126 | 102 | 25 | 157 | 7600 (3) | 42 | 10.48 | 2.77 | 11.19 | 4.49 |
| Cantor_large | spline | 5 | 228 | 102 | 49 | 355 | 45800 (3) | 42 | 34.35 | 6.28 | 39.58 | 20.46 |

- **Radial kinds.** Every model has `radial_kind = "spline"` and
  `pair_radial_kind = "spline"`. None carries `rnl_spline_coefs_single`, so none
  is `spline_factorised`. The spline is uniform cubic with n = 100 intervals,
  so ncoef = 102 for every model.
- **The columns.**
  - `efv` is the median of 20 timed calls.
  - `kernel sum` is the GPU kernel time per call in the profiler trace. Under
    CUPTI it runs up to 15% above `efv`.
  - `K` is the dense row width.
- **Agreement with the benchmark.** The `efv` times match the benchmark's A100
  `force_s` for these models; for example, Cantor_large is 34.7 ms there.
- **Forces are expensive.** Forces cost 2.3–5.5× the energy-only call. The
  reverse pass dominates every stage except the angular one (§2.2).

## 2. Where the time goes

### 2.1 Per-stage GPU time per call

The numbers are ms, with the share of the kernel sum in brackets, from the
trace.

| model | radial | angular | pool (A assembly) | `_aa` | readout | force scatter | strain | other |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SiGe_small | 1.01 (34.6%) | 0.64 (22.1%) | 0.55 (18.8%) | 0.31 (10.6%) | 0.10 (3.4%) | 0.22 (7.7%) | 0.08 (2.6%) | 0.01 |
| SiGe_medium | 1.46 (17.4%) | 1.71 (20.5%) | 1.83 (21.9%) | 2.51 (30.1%) | 0.52 (6.2%) | 0.22 (2.6%) | 0.08 (0.9%) | 0.03 |
| SiGe_large | 1.42 (9.5%) | 1.70 (11.4%) | 1.80 (12.0%) | 8.12 (54.3%) | 1.55 (10.4%) | 0.24 (1.6%) | 0.08 (0.5%) | 0.04 |
| Cantor_small | 2.01 (41.3%) | 0.19 (3.9%) | 0.73 (15.0%) | 1.07 (22.1%) | 0.27 (5.6%) | 0.46 (9.4%) | 0.11 (2.3%) | 0.01 |
| Cantor_medium | 2.79 (25.0%) | 0.94 (8.4%) | 1.78 (15.9%) | 4.19 (37.4%) | 0.90 (8.1%) | 0.44 (4.0%) | 0.11 (1.0%) | 0.02 |
| Cantor_large | 3.99 (10.1%) | 2.33 (5.9%) | 4.96 (12.5%) | 23.38 (59.1%) | 4.39 (11.1%) | 0.39 (1.0%) | 0.11 (0.3%) | 0.04 |

What each stage covers:

| stage | code |
|---|---|
| radial | `ACEModel.radial`: the Agnesi transform, envelope, spline `Rnl` and the pair radial, plus the mask on `Rnl` |
| angular | the spherical harmonics |
| pool | `pool_a_dense` (batched outer product and column select) and the pair channel's `pool_dense` |
| `_aa` | the product basis |
| readout | `AA · ctilde[:, z] + Apair · Wpair[:, z] + E0` |
| force scatter | the scatter-add after the gradient in `energy_forces_virial_dense` |
| strain | the `r + r @ sym` of the virial trick |

### 2.2 Forward and adjoint split, in ms (forward / adjoint)

| model | radial | pool | `_aa` | readout |
|---|---|---|---|---|
| SiGe_small | 0.35 / 0.66 | 0.17 / 0.38 | 0.03 / 0.28 | 0.03 / 0.07 |
| SiGe_medium | 0.56 / 0.89 | 0.54 / 1.29 | 0.23 / 2.28 | 0.10 / 0.42 |
| SiGe_large | 0.56 / 0.87 | 0.54 / 1.26 | 0.65 / 7.47 | 0.17 / 1.38 |
| Cantor_small | 0.74 / 1.27 | 0.23 / 0.50 | 0.10 / 0.97 | 0.07 / 0.20 |
| Cantor_medium | 1.14 / 1.66 | 0.49 / 1.29 | 0.34 / 3.84 | 0.11 / 0.80 |
| Cantor_large | 1.75 / 2.24 | 1.41 / 3.55 | 1.64 / 21.74 | 0.47 / 3.92 |

The `_aa` stage is 90–93% adjoint.

### 2.3 The quantities the rules test, with bounds

A kernel that XLA fused across stages is attributed by the share of its fused
instructions in each stage. That share is the estimate. It is bounded:

- below, by counting only kernels wholly inside the stage ("pure");
- above, by counting every kernel that touches the stage ("touch").

| model | radial + pool | [pure, touch] | `_aa` | `_aa` [pure, touch] | `_aa` + readout | top kernel |
|---|--:|---|--:|---|--:|---|
| SiGe_small | 53.3% | [42.8, ≤86] | 10.6% | [0.3, 13.9] | 13.9% | `loop_concatenate_fusion` (Y_lm, forward), 0.49 ms |
| SiGe_medium | 39.3% | [35.2, ≤66] | 30.1% | [0.2, 37.4] | 36.3% | `input_scatter_fusion.1`, 2.40 ms, out `f64[165,8192]` |
| SiGe_large | 21.6% | [20.2, ≤36] | 54.3% | [0.3, 64.4] | 64.7% | `input_scatter_fusion.5`, 6.49 ms, out `f64[8192,165]` |
| Cantor_small | 56.4% | [41.4, ≤78] | 22.1% | [0.2, 27.1] | 27.7% | `input_scatter_fusion.1`, 0.96 ms, out `f64[72,8192]` |
| Cantor_medium | 40.9% | [32.7, ≤58] | 37.4% | [0.2, 45.6] | 45.5% | `input_scatter_fusion.1`, 4.09 ms, out `f64[157,8192]` |
| Cantor_large | 22.6% | [19.9, ≤31] | 59.1% | [0.1, 70.6] | 70.2% | `input_scatter_fusion.1`, 24.37 ms, out `f64[355,8192]` |

- **The `_aa` "pure" share is near zero by construction.** XLA fuses the
  product-basis adjoint with the readout's cotangent, `ctilde[:, z]` gathered
  inline. The kernel is still the product-basis adjoint, so the estimate
  (85% `_aa`, 15% readout by instruction count) is the fair figure.
  - For Cantor_large I read the fusion body (`bench/perf/hlo_fusion.py`). It is
    the order-3 `_aa` adjoint: it gathers A, recomputes the partial products,
    multiplies them by the gathered `ctilde[:, z]`, and scatter-adds 128,502
    row updates into A (355 × 8192).
  - That one kernel is **62% of the call**.
- **Radial + A assembly clears 15% even at its lower bound** on every model,
  at 19.9–42.8%.
- **`_aa` clears 15% on its estimate for five models,** at 22.1–59.1%. The
  single fused kernel it lives in confirms this: the "touch" bound is
  27–71%.
- **SiGe_small is the one model below the threshold.** Its `_aa` estimate is
  10.6%, with an upper bound of 13.9%.

### 2.4 Isolated stage timings (secondary)

`profile_ace.py` also times each stage as its own `jit`, forward and
forward + VJP. These times include host dispatch of the model pytree (roughly
0.3–0.8 ms per stage), and they sum to 1.5–1.9× `efv`. So they are recorded in
the JSON (`stages`) but not used for the decision. They agree with the trace on
the ordering: the `_aa` VJP is the largest isolated stage on every model except
SiGe_small, and it reaches 41.5 ms on Cantor_large.

## 3. Pool-first for the `spline` radial

### 3.1 Why C = NZ

The spline radial is

  R_nl(e) = env(e) · Σ_c coef[z_i, z_j, c, n] · b_c(x_e)

- b_c is the cubic B-spline basis: 4 nonzeros out of ncoef = 102.
- x_e = T_{z_i z_j}(r_e) is the transformed distance.

Pool-first moves the coefficients outside the sum over neighbours:

  A[i, (n, y)] = Σ_{z} Σ_c coef[z_i, z, c, n] · ( Σ_{j ∈ N(i), z_j = z} env · b_c(x) · Y_y )

The bracket is the pooled tensor. The coefficient depends on the neighbour
species, so the bracket must be kept per z: that is C = NZ blocks of ncoef × n_Y
per node.

The same holds for `analytic`, where `Wnlq[z_i, z_j]` also depends on the
neighbour species. This is exactly PACE's `a_channels = NZ` for
`crad[z_i, z_j]`. `ACEModel.a_channels = 1` describes today's `edge_a` path,
which applies the z_j-dependent weights per edge before pooling.

### 3.2 Pooled width and estimated cost

The pool stage scales with the pooled width, because the `nkr,nky->nry`
contraction costs n·K·r·y. The estimate below assumes, generously, that
pool-first removes the whole radial stage. In fact b(x), the transform, the
envelope and the pair radial all remain.

| model | n_rnl·n_Y (now) | ncoef·n_Y (C=1) | NZ·ncoef·n_Y (C=NZ) | ratio (C=NZ) | pool now | pool, pool-first est. | radial now (max saving) | net, best case |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SiGe_small | 1275 | 2550 | 5100 | 4.00× | 0.55 | 2.19 | 1.01 | −0.63 ms |
| SiGe_medium | 5292 | 4998 | 9996 | 1.89× | 1.83 | 3.45 | 1.46 | −0.17 ms |
| SiGe_large | 5292 | 4998 | 9996 | 1.89× | 1.80 | 3.40 | 1.42 | −0.18 ms |
| Cantor_small | 666 | 918 | 4590 | 6.89× | 0.73 | 5.03 | 2.01 | −2.29 ms |
| Cantor_medium | 3150 | 2550 | 12750 | 4.05× | 1.78 | 7.21 | 2.79 | −2.64 ms |
| Cantor_large | 11172 | 4998 | 24990 | 2.24× | 4.96 | 11.08 | 3.99 | −2.14 ms |

- **Pool-first is a net loss on every model, even in the best case.** Peak A
  memory also grows by the same ratio.
- **The 4-sparsity of b could avoid the dense ncoef-wide pool,** by a
  segment-sum of 4·n_Y per edge into the pooled tensor. But that
  reintroduces the scatter the dense layout exists to avoid, and it doesn't
  change the per-node W contraction.

### 3.3 `analytic` and `spline_factorised` (not measured)

No benchmark model uses these kinds, so the rule's profile test cannot be
evaluated, and neither is applied in Task 9. For reference, here is what the
width test would see.

- **`analytic`:** n_b = n_q, the polynomial count, and C = NZ, because `Wnlq`
  depends on z_j.
  - The committed single-species fixtures have n_q well below n_rnl:
    `si_ace_model` 15 against 37, `si_l2849` 27 against 84, `si_m710` 21
    against 60, and `si_s69` 12 against 24.
  - So a single-species analytic model would pass the width test.
  - A multi-species one passes only if NZ·n_q ≤ n_rnl.
- **`spline_factorised`:** R = P(x)[nidx] · emb[z_j][kidx].
  - The z_j dependence sits in the embedding. So the pooled tensor can be kept
    per embedding channel (P ⊗ emb_k ⊗ Y) rather than per species.
  - That gives C = min(NZ, d), with n_b = n1, the number of columns of the
    species-independent table.

## 4. Other observations (not decided here)

- **Angular.** The Y_lm forward (`loop_concatenate_fusion`, spherical
  harmonics up to lmax = 6) costs 0.5–2.0 ms, 4–22% of the call. This is the
  single largest kernel on SiGe_small. None of the candidates addresses it.
- **Force scatter.** The force scatter is only 1–9% of the call, so
  reverse-edge gathers (PACE's candidate #6) have little to win for ACE at
  this size.
- **The A-select adjoint.** The adjoint of `A_full[:, sel]` in `pool_a_dense`
  is a zero fill of `f64[8192, n_rnl·n_Y]` plus a scatter into it. Together
  they cost 0.07 ms (Cantor_small) to 0.83 ms (Cantor_large), and are
  attributed almost entirely to pool.
  - A one-hot matmul form, or feature-major A, would remove this scatter.
  - It is part of the pool stage's adjoint in §2.2.

## 5. Task 9 outcome: feature-major product basis measured, not adopted

Task 9 implemented the feature-major form the decision above marked **apply**,
measured it against the committed node-major form, and **reverted it**: it is
slower on all six models. `ACEModel` is unchanged. No pool-first was attempted
for any radial kind (§3).

### 5.1 What was measured

The candidate (`fm`), as the controller's ruling specified, in the folded
dense energy path:

- A assembled feature-major, At (n_A, n): the batched outer product with the
  node axis last, `einsum("nkr,nky->ryn")`, then the used rows selected;
- `_aa` on rows, `jnp.prod(At[g.T], axis=0)`, the form of PACE's
  `_node_energies_t`;
- the readout `ctilde.T @ AA`, (NZ, n), then each node's centre-species entry.

The unfolded path (`_from_pooled`) was made consistent (AA_t gathered by row for
the sparse A2B, `A2B @ AA_t` for the dense). All tests passed with it, including
the 288 frozen references, and E, F and V agreed with the node-major form to
≤ 2e-16 relative in E and ≤ 2.4e-13 eV/Å in F on the A100.

Three further variants attribute the difference:

- `fm_T`: `fm` with At = `pool_a_dense(...).T`, the node-major assembly
  transposed;
- `fm_T_ein`: `fm_T` with the node-major readout's form, `ctilde[:, z]`
  gathered and multiply-reduced, instead of the GEMM;
- `nm_gemm`: the committed node-major form with only the readout replaced by
  the GEMM, `(AA @ ctilde)[i, z_i]`.

All variants are `ACEModel` subclasses in `bench/perf/ab_ace_fm.py`, so the
candidate survives there and nowhere in `src/`.

### 5.2 Results

`energy_forces_virial_dense`, float64, 8192 atoms, readout folded, Modal
A100-SXM4-80GB. Every variant ran in one process on one card, interleaved over
rounds; each entry is the median over rounds of the median of 20 calls.
Round-to-round spread after the first round was under 1%.

Run 1 (5 rounds), `bench/perf/results/ab_fm_8192_float64.json`, ms (change
against node-major):

| model | node-major (committed) | `fm` (candidate) | `fm_T` |
|---|--:|--:|--:|
| SiGe_small | 3.16 | 3.73 (+17.9%) | 3.29 (+4.1%) |
| SiGe_medium | 7.54 | 10.23 (+35.6%) | 8.44 (+11.8%) |
| SiGe_large | 13.14 | 19.40 (+47.6%) | 17.73 (+35.0%) |
| Cantor_small | 4.56 | 5.23 (+14.6%) | 4.92 (+7.9%) |
| Cantor_medium | 10.28 | 12.38 (+20.5%) | 11.35 (+10.4%) |
| Cantor_large | 34.48 | 47.89 (+38.9%) | 44.75 (+29.8%) |

Run 2 (3 rounds, attribution), `bench/perf/results/ab_fm_8192_float64_attr.json`:

| model | node-major | `fm_T` | `fm_T_ein` | `nm_gemm` |
|---|--:|--:|--:|--:|
| SiGe_small | 2.96 | 3.05 (+2.8%) | 2.97 (+0.3%) | 3.03 (+2.2%) |
| SiGe_medium | 7.61 | 8.41 (+10.6%) | 7.81 (+2.7%) | 8.04 (+5.7%) |
| SiGe_large | 13.14 | 17.69 (+34.7%) | 14.47 (+10.1%) | 16.10 (+22.5%) |
| Cantor_small | 4.55 | 4.95 (+8.9%) | 4.64 (+1.9%) | 4.90 (+7.9%) |
| Cantor_medium | 10.26 | 11.36 (+10.7%) | 10.51 (+2.5%) | 11.12 (+8.4%) |
| Cantor_large | 34.34 | 44.45 (+29.4%) | 35.41 (+3.1%) | 41.75 (+21.6%) |

### 5.3 Reading

- **No variant is faster than node-major on any model.** The best
  feature-major form, `fm_T_ein`, is neutral only on SiGe_small (+0.3%) and
  slower by 1.9–10.1% elsewhere.
- **The GEMM readout is the largest single loss** (`nm_gemm`: +2.2 to
  +22.5%, growing with n_AA). A plausible reading, not checked in the HLO: the
  node-major einsum lets XLA fuse the `ctilde[:, z]` gather into the `_aa`
  adjoint kernel (§2.3), while the GEMM's adjoint materialises dAA =
  ctilde · dE as a separate (n_AA, n) product first.
- **Feature-major `_aa` alone does not pay either** (`fm_T_ein`). This is
  consistent with §4 and with `bench/perf/variants.py`: XLA's layout
  assignment already lays A out feature-major where it matters, so writing it
  that way adds transposes rather than removing scatters.
- **Assembling A with the node axis last costs more still** (`fm` against
  `fm_T`: a further 3–14%).

### 5.4 Decision

The ruling's aggregate rule: keep the change only if it is faster or neutral
(within ±3%) on every model and faster on some. The candidate is 14.6–47.6%
slower on every model, so it is **reverted**, and no run-time flag was added.

What Task 9 keeps:

- a gradient guard, `tests/test_pool_first.py::test_ace_energy_gradient_wrt_linear_weights`:
  dE/d(ctilde) (folded) and dE/d(WB) (unfolded) match a central finite
  difference, in the sparse and dense layouts, on `si_ace_model` (analytic,
  one species) and `sige_nofit` (spline, two species). A `stop_gradient` on
  either weight fails all 8 cases;
- the A/B harness `bench/perf/ab_ace_fm.py` and `modal_profile.py::ab_fm`.

The remedy for the `_aa` adjoint scatter is still PACE's candidate #7 (§ Decision
above): a custom kernel that keeps A and dA on chip.

Cost: two single-container runs, about 6 A100-minutes in all.

## 6. Method and reproduction

```bash
uv run --with modal modal run bench/perf/modal_profile.py::ace
# one model:   ... ::ace --models Cantor_medium
# Task 9 A/B: uv run --with modal modal run bench/perf/modal_profile.py::ab_fm
#   attribution: ... ::ab_fm --variants node_major,fm_T,fm_T_ein,nm_gemm --rounds 3 --tag _attr
# locally:     PYTHONPATH=bench:src python bench/perf/profile_ace.py \
#                  bench/scaling/models/ace_SiGe_small.npz SiGe 512 --reps 2
```

### Inputs and timing

- **The structure** is `scaling.structures.supercell` at 8192 atoms.
- **The graph** is `nlist.dense_graph`, built on the host, with K equal to the
  largest neighbour count.
- **Timing** is the median of 20 calls after 3 warm-up calls, under
  `highest_precision()`.

### How kernels are attributed to stages

- **Two runs.** A second process traces 5 calls with `jax.profiler` and dumps
  the optimised HLO (`--xla_dump_to`). Each GPU kernel's time comes from the
  trace, and its stage comes from the HLO.
- **Call chains, not source lines.** JAX 0.11 writes no `source_line` into the
  HLO. Instead, each instruction carries a `stack_frame_id` into the module's
  `FileNames`, `FunctionNames`, `FileLocations` and `StackFrames` tables.
  - `profile_ace.kernel_stages` walks each fused instruction's call chain,
    innermost first, to the first function that names a stage: `ACEModel._aa`,
    `EdgeSiteModel.pool_a_dense`, `ACEModel.radial`, and so on.
  - A kernel is weighted by its fused instructions' stages.
  - `strain` is elementwise glue, so it counts only when a kernel has nothing
    else.
  - The earlier PACE study's `hlo_trace.kernel_table` looked for
    `source_line`, and so found no sources under this JAX.
- **Unnamed library GEMMs.** A library GEMM inside a CUDA command buffer
  (Cantor_large's cuBLAS-Lt `custom-call.1`, the pool einsum's adjoint for Y,
  1.26 ms) is named only by its buffer. It is mapped to the module's single
  `__cublas` custom-call.
  - I added this fallback after the run.
  - Cantor_large's `kernels` block was re-attributed from the saved trace and
    HLO with `--reattribute`. The kernel times are unchanged, and the JSON
    records this in `kernels.note`.
- **Where the raw data is.** The traces and optimised HLO are on the Modal
  volume `ace-jax-perf-profile`, under `/ace_<model>_8192_float64/`.

### Cost

About 6 A100-minutes: a 2-minute smoke run, then 4 minutes for all six models.
