# Coupled product basis for the C++ and CUDA kernels

Status: in progress (see "Status" below), 2026-11-19. Branch `perf/symmetrix-ace`.

## Why

The kernels evaluate the product basis as a DAG of monomials of A (`eval/aa_dag.py`, `ffi.plan`). The readout
`ctilde = A2B^T WB` has the rotational coupling multiplied out, so the DAG does not see that the monomials of one
(n, l)-multiset are a few invariants. A linear ACE of correlation order <= 4 is exactly

    z = [A; X],   X^L_{P,M} = sum_{m1 m2} T^{l1 l2 L}_{m1 m2 M} A_{b1 m1} A_{b2 m2},   P = (b1, b2)
    E = E0 + lin . A + 1/2 z^T W z

with b = (radial column, l) a block of A, T the real Clebsch-Gordan tensor and W sparse and the same for every M of
a pair of blocks of one L. Counts per site, Si models of the accuracy-cost benchmark:

| model | DAG nodes | X | CG terms | weights |
|---|---|---|---|---|
| o4d12 | 1,024 | 93 | 248 | 524 |
| o4d16 | 9,280 | 439 | 1,523 | 3,251 |
| o4d20 | 68,335 | 1,724 | 7,732 | 16,949 |

Measured before this work (product-basis pass alone, four sites a SIMD lane, one thread, plain CSR gathers): 3.2x
at o4d20, 2.3x at o4d16, 1.5x at o4d12, none at o3d8. The DAG is ~80 % of the CPU kernel at o4d20. Two elements
gain more: Si-Ge o4d12 has 14,460 DAG nodes against 541 X, 1,457 CG terms and 6,065 weights per centre species.

The coupled form, and most of the GPU ideas tried here, are Chuck Witt's: his linear ACE pair style in Symmetrix
(`symmetrix/ace`, single element, in development) evaluates ace-jax's Si models about six times faster than
ace-jax 0.5 on an A100. Goal: as much as possible of that speed, with no
code that is specific to one element, one radial form or one model.

## Design

### 1. The coupled plan (`eval/coupled.py`, framework layer, NumPy only)

Input: the A spec (radial column, harmonic) of a lean model, its `aa_specs` and `ctilde` (n_AA, NZ).

- Blocks: A entries of one (radial column, l), all 2l + 1 of them. A model with an incomplete block, or with
  correlation order > 4, has no coupled plan and keeps the DAG (order 5+ by nested coupling is a later step).
- Monomials are grouped by their multiset of blocks. Candidates per multiset: order 1 (an l = 0 block) linear;
  order 2 (equal l) the dot product, a weight between two A blocks; order 3 X^{l3}_{b1 b2} . A_{b3}; order 4
  X^L_{b1 b2} . X^L_{b3 b4} for every allowed L.
- Pairing: which two blocks of an order-3 or order-4 multiset are coupled is a free choice (3 ways), made to
  share X blocks between multisets: greedy, then `scipy.optimize.milp` under a time limit. Cost of an X block:
  its 2L + 1 values plus its CG terms, weights tunable. It depends on the spec only, not on the readout.
- Weights: per multiset, least squares of the candidates' monomial expansions against `ctilde`'s rows, kept as
  the linear map `w = P ct` (sparse, block per multiset), so a readout swap is one sparse product and all centre
  species share one structure (weights (NZ, n_w)). The fit must be exact: relative residual <= 1e-10 or no plan.
- Real CG tensors: `basis/cg.py::real_cg` moves to `eval/cg.py` (`eval` imports no other package); `basis.cg`
  re-exports it.
- Cached on disk by content (spec hash, pairing options) beside the FFI cache: the pairing search is seconds to
  a minute at o4d20.
- A NumPy evaluator of the plan (E and dE/dA from A) is the reference for the tests and for the kernels' emulation.

### 2. Evaluation, both kernels

Three passes over z of a batch of sites of one centre species, every sum by the owner of its target:

1. X: per pair block (b1, b2, L), the CG table of (l1, l2, L) applied to two runs of A. Tables are per type,
   shared by all pair blocks, so the stream is (a1, a2, x, type).
2. g = W z by block rows: g_U[M] = sum_V w_UV z_V[M] for M = 0..2L, one weight stream for 2L + 1 rows.
   E = lin . A + 1/2 z . g.
3. dE/dA_a = lin_a + g_a + sum over CG terms c g_x A_partner, gathered per A entry.

Kernel interface: new operands for the plan (int32) and the weights (the DAG's type, per centre species); an empty
plan means the DAG. `ffi.kernel_inputs` picks the form (`ffi_form="auto" | "dag" | "coupled"`, auto = coupled
when a plan exists and it has fewer operations than the DAG). Mixed precision as now: z, g in float32.

### 3. Order of work

1. `eval/coupled.py` + tests (exact against the DAG on the fixtures: one element, several elements, species
   embedding, PACE-derived; readout swap; cache). Pairing: measure greedy against the integer program.
2. C++ kernel (`ace_ffi.cc`): the three passes, four sites a lane; `ACECalculator(ffi=...)`, `BatchCalculator`,
   CPU bundles. Develop on moriarty; time on a Sulis compute node (128 x 1, 4,096 atoms) against the table in
   `docs/dev/accuracy-cost-results.md`.
3. CUDA kernel (`ace_ffi_gpu.cu`): z of a batch in shared memory when it fits, else scratch; no atomics.
   Develop on lestrade / moriarty (float32 on the workstation GPUs), time on a Sulis A100 against the same-GPU
   run of Symmetrix's `symmetrix/ace` and of main. Then decide, by measurement, on the float64 tensor-core
   form of pass 2 (sm_80+ data-centre GPUs only) and on the A and force passes (analytic radial, recurrence
   harmonics), each only in a form that keeps several elements and learned radials.
4. Docs (`docs/user` performance page, `skills/ace-jax/SKILL.md`, CLAUDE.md), benchmark rows.

Not in this work: a coupled form in the JAX (XLA) model path; changes to the coupling library. The plan is
built from `ctilde` and our own CG tables, so `ace-jax-coupling` and the EquivariantTensors fork are untouched.
Emitting the pair-coupled form from EquivariantTensors directly would remove the fit and is the form to propose
upstream if this proves out.

## Tests and gates

- Plan against DAG: E and dE/dA at random A to 1e-12 relative, every fixture.
- Kernels: the existing `tests/test_ffi.py` cases for both forms; forces against the JAX model at the present
  tolerances; `tests/test_ffi_plan.py`'s emulation extended to the coupled passes.
- Speed: no model slower than with the DAG in `auto`.

## Status, 2026-11-19 (evening)

Done, in the working tree of `perf/symmetrix-ace` (not committed):

- `eval/cg.py` (moved from `basis/cg.py`, which re-exports it), `eval/coupled.py` (`build`, `Plan`, `evaluate`,
  `check`), `ffi.coupled_of` / `coupled_plan` / `coupled_gpu_plan` / `choose_form`, `tests/test_coupled.py`.
  The plan builds in 1.8 s at o4d20 (greedy pairing; a 30 s integer program lowers X by 4 %: off by default,
  `ffi.ILP_SECONDS`) and equals the monomials to 3e-16.
- C++ kernel: the coupled passes (attribute `form`; the plan and its reals take the DAG's operands), `auto` by
  operation count. Also, for both forms: harmonic gradients from the degree below (no stored dY), the radial
  table by four columns and four neighbours with no division, harmonics by m with the l recurrence in
  registers, runs sorted by length and unrolled. `-DACEJAX_FFI_PROF` prints the three stages' times.
- CUDA kernel: `ace_coupled` (opt-in: `ffi.FORM = "coupled"` or `ACEJAX_FFI_FORM=coupled`; the default on a GPU
  is still the DAG). Per-thread task streams, long tasks in pieces, no atomics; g of the X part in scratch.

One thread, moriarty (i9), 512 Si atoms through `ACECalculator`, µs/atom, float64 (forces equal to 1e-12):

| model | main | DAG, new A and force stages | coupled |
|---|---|---|---|
| o3d8 | 9.4 | 8.9 | 8.0 |
| o4d12 | 13.7 | 11.4 | 10.5 |
| o4d16 | 23.7 | 21.7 | 16.9 |
| o4d20 | 88–94 | 86.6 | 33.7 |

(~5 µs/atom of each figure is the calculator outside the kernel.) Stages of the coupled kernel at o4d20: A 6.5,
product basis 14.8, forces 5.4 µs/site; mixed precision: product basis 7.9.

An A100 (Modal, A100-SXM4-40GB), 32,768 Si atoms, each kernel's duration by Nsight Compute, ms:

| model, type | product basis: DAG | coupled | forces | A |
|---|---|---|---|---|
| o4d20 float64 | ~18 | 4.7 | 2.6 | 0.98 |
| o4d20 float32 | 10.7 | 2.9–3.4 | 1.6 | 0.78 |
| o4d16 float64 | 3.3 | 1.4 | 1.4 | 0.77 |
| o4d16 float32 | 2.0 | 0.8 | 1.1 | 0.61 |
| o3d8 float64 | 0.08 | 0.08 | 0.51 | 0.35 |

Symmetrix's passes at o4d20 float64 (its notes, A100-SXM4): product basis 2.7 (0.85 of it
on float64 tensor cores, 0.27 generated CG code), forces 0.78, A 0.71. So after the coupled pass the force and A
passes are the larger gap, and they are all of the gap for the small models.

What Nsight shows for `ace_coupled` (o4d20 float64, 2 rows a block, 3 blocks an SM, 256 threads, 80 registers):
issue slots 34 % busy, no eligible warp 66 % of cycles; the lists are read from L2 (L1 hit 42 %: shared memory
leaves L1 ~24 KB); shared loads take 4.6 wavefronts where 4 is the least for 32 lanes of 16 bytes, so ordering
items by bank cannot help. Tried: 4 rows a block (1 block an SM) 6.5 ms, 1 row 6.0, 128 threads 5.5 (float32:
128 threads 2.9 against 3.4 at 256).

Next, in order of expected gain:

1. Forces and A passes, a lane per neighbour (now a team of 8 lanes per pair: 13 instructions per flop,
   25 % occupancy at o4d20).
2. `ace_coupled`: threads per block by type, fewer registers (4 blocks an SM), piece slots.
3. To decide with the user: pass 2 on float64 tensor cores (sm_80 / sm_90 data-centre GPUs only), generated
   straight-line CG code per (l1, l2, L) (any element, but one source per lmax), an analytic-radial GPU path
   (per radial kind). Each is generic in elements; none is generic in hardware or model kind.
4. LAMMPS timings on Sulis (jobs 2295320 CPU, 2295321 A100 with `symmetrix/ace` on the same GPU).

Tried and removed (2026-11-19): a force pass with a lane per neighbour and a warp per block (the lane keeps its
table rows, per degree takes that degree's radial columns into its column of shared memory, then dE/dY over
them). Correct on every GPU test, 3x fewer instructions, and slower on the A100: o4d20 float64 6.9 ms against
2.6, o3d8 1.16 against 0.51. With a radial table each lane reads its own four rows, so a load instruction touches
32 cache lines where a team of 8 lanes on one pair touches one, and only 5 warps fit an SM. A lane per neighbour
pays only with a radial that needs no table per neighbour (Symmetrix's analytic radial), or
with the table stage kept in teams and only the contraction per lane.

LAMMPS on Sulis, 128 x 1, Si 4,096 atoms, EPYC 7742, one node (job 2295320, node022; it reached its hour during
o4d12's main run and did not start o3d8), µs/atom-step:

| model | main (7ad4803) | this tree |
|---|---|---|
| o4d20 | 1.571 | 0.436, 0.450 |
| o4d16 | 0.329 | 0.187 |
| o4d12 | (0.151 in the published table, another node) | 0.121 |

## Analytic radial on the GPU (2026-11-19, late)

Agreed with the user: a path that needs the polynomial tensor radial (`radial_kind` "analytic", `rnl_basis`
"poly") is acceptable. Opt-in for now: `ffi.RADIAL = "analytic"` / `ACEJAX_FFI_RADIAL=analytic`; models without
such a radial (splines, learned radials after `to_spline`, PACE) keep the table. `ffi.analytic_plan`: per species
pair and degree, the W entries by q; `ace_forces_lane` / `ace_density_lane` in `ace_ffi_gpu.cu` (a warp per
block, a lane per neighbour, one instantiation per lmax up to 8).

A100 (Modal), 32,768 Si atoms, ms, table kernel then analytic:

| model, type | forces | A |
|---|---|---|
| o4d20 float64 | 2.60, 1.83 | 0.98, 2.54 (not taken) |
| o4d16 float64 | 1.41, 0.79 | 0.77, 1.07 (not taken) |
| o3d8 float64 | 0.51, 0.26 | 0.35, 0.23 |
| o4d20 float32 | 1.60, 1.03 | 0.78, 1.22 (not taken) |
| o4d16 float32 | 1.13, 0.51 | 0.61, 0.62 (not taken) |
| o3d8 float32 | 0.41, 0.16 | 0.29, 0.17 (not taken: 2e-3 eV/atom) |

What it took, in order:

- The analytic radial as a drop-in for the table inside the team kernels: slower (forces 5.6 ms, A 3.2 at o4d20):
  every lane of a team repeats the recursion and scans all entries. Removed.
- A lane per neighbour with the degree's radial columns in the lane's column of shared memory: o3d8 0.39, but
  o4d16 2.2 and o4d20 6.1 (5 warps an SM, index work per (harmonic, column)).
- The degree's 2l + 1 sums dE/dY, dE/dY' in registers, a function per degree, the plan's entries naming the
  first A of their column: o4d20 1.86. One kernel per lmax (registers follow the largest degree): 1.83, o4d16
  0.86 -> 0.79, o3d8 0.33 -> 0.26.
- The A pass the same way sums each A entry over the warp by shuffles: it wins only for few A, and float32
  loses the pair term's digits in the transform, so it is taken for n_a <= 64 in float64 only. Where it is not
  taken the energy is the table's and the forces the analytic radial's: they differ by the table's error.

Nsight on `ace_forces_lane` (o4d20 float64): 168 registers, 17.5 KB shared a warp (12.5 of it the lane's
harmonics), 8 warps an SM, no eligible warp 65 % of cycles: occupancy-bound. Symmetrix's 0.78 ms comes with
generated contracted harmonics (no Y array per lane).

Kernel sums now, o4d20 float64: 4.7 + 1.83 + 0.98 = 7.5 ms (main 21.6, Symmetrix ~4.2); o4d16 1.3 + 0.79 + 0.77
= 2.9 (main 5.5); o3d8 0.08 + 0.26 + 0.23 = 0.57 (main 0.94; Symmetrix's whole step 0.59).

Symmetrix's `symmetrix/ace/kk` on a Sulis A100-PCIE-40GB (job 2295321, gpu011; LAMMPS 0165101, its `in.acejax`,
32,768 Si atoms, three runs each), µs/atom-step, float64 then its float32 mode: o4d20 0.121-0.124 and
0.092-0.095; o4d16 0.0556 and 0.047; o3d8 0.0165 and 0.0166. These reproduce its own Perlmutter figures (0.116,
0.0555, 0.0180). That job was cancelled by mistake before the ace-jax legs; 2295397 repeats all three.

## LAMMPS on one Sulis A100 (job 2295397, gpu-devel, A100-PCIE-40GB, 32,768 Si atoms), µs/atom-step

| model | Symmetrix float64 / its float32 mode | main float64 / float32 | this tree float64 / float32 |
|---|---|---|---|
| o4d20 | 0.120-0.123 / 0.092-0.094 | 0.745 / 0.447 | 0.283 / 0.180 |
| o4d16 | 0.056 / 0.045-0.047 | 0.190 / 0.133 | 0.124 / 0.096 |
| o3d8 | 0.0165 / 0.0166 | 0.0425 / 0.0448 | (parity gate failed: below) |

This tree: `ACEJAX_FFI_FORM=coupled ACEJAX_FFI_RADIAL=analytic` at e17b7ef. So 2.6x on main at o4d20 and 1.5x at
o4d16, and Symmetrix is 2.3x and 2.2x ahead where it was 6.2x and 3.4x.

The o3d8 gate (64 atoms against the JAX model, 1e-8) failed with max |dF| 3.6e-7: the tensor radial was analytic
and the pair radial still the table's. At o3d8 the two give forces of ~600 eV/A each that cancel to ~2, and in
one table their errors cancel too (the all-table kernel: 1e-8); an exact tensor radial beside the table's pair
radial does not (1e-7). Fixed after e17b7ef: the pair radial is analytic in the lane kernels too (`an_pair`:
Agnesi transform, recursion, `env_ace1_poly1sr` / `env_poly1sr`), and `analytic_plan` is None unless both
radials are analytic. Against the JAX model without a table, 64 rattled Si atoms, max |dF| in eV/A, table kernel
then analytic: o3d8 9.9e-9, 1.6e-11; o4d12 2.4e-9, 2.0e-12; o4d16 3.3e-9, 1.0e-12. Job 2295415 repeats the
ace-jax legs with this.

## Recoupling over all pairings, and the second Sulis run (2026-11-19, night)

Symmetrix's update of the day (ef51f70: o4d20 0.116 -> 0.102 µs/atom-step) re-expresses each multiset over the
candidates of all three pairings and every L, choosing those that share X blocks. `coupled.build(recouple=True)`
(now `ffi.RECOUPLE`, the default) does the same in the plan: per multiset the candidates of every pairing, ordered
by the cost of the X blocks no other multiset reads yet, the independent ones kept, three sweeps. Exact as before.

| model | X before, after | CG terms before, after |
|---|---|---|
| o4d12 | 103, 78 | 230, 220 |
| o4d16 | 485, 378 | 1,419, 1,300 |
| o4d20 | 1,879, 1,274 | 7,166, 5,832 |

(Symmetrix's two o4d20 candidates: X 1,530 and 1,468.) C++ kernel, one thread, o4d20: product basis 14.8 ->
12.4 µs/site, the kernel 33.7 -> 30.1 µs/atom (2.84x its DAG). A100 `ace_coupled`: o4d20 float64 4.74 -> 4.15 ms,
float32 2.93 -> 2.42; o4d16 float64 1.32 -> 1.23.

Sulis job 2295415 (one A100-PCIE-40GB, LAMMPS, 32,768 Si atoms; this tree before recoupling, with the analytic
pair radial), µs/atom-step, float64 / float32, this tree then main:

| model | this tree | main |
|---|---|---|
| o4d20 | 0.283 / 0.181 | 0.742 / 0.448 |
| o4d16 | 0.1245 / 0.095 | 0.190 / 0.1325 |
| o4d12 | 0.063 / 0.046 | 0.0665 / 0.0515 |
| o3d8 | (gate) | 0.0423 / 0.0429 |

- o3d8's gate still fails, now at max |dF| 1.28e-8 against its 1e-8: the gate's reference is the JAX model with
  the radial table, which is itself 1e-8 from the model (main's bundle, on the same table, passes at 1.3e-9); the
  analytic kernel is 1.6e-11 from the model. The gate needs the reference without a table for an analytic bundle.
- LAMMPS steps cost more than the kernels: o4d16 is 4.1 ms a step where the three kernels are 2.9 ms, o4d12 2.1 ms,
  and main's o3d8 1.39 ms where its kernels are 0.94. Symmetrix's whole o3d8 step is 0.54 ms. For the small models
  the next gain is in the path around the kernel (lammps-jax's neighbour matrix and call, `ace_count` /
  `ace_fill`), not in it.

Sulis job 2295450 (the same A100, gpu011; this tree at 9025f71, recoupled), µs/atom-step, float64 / float32:
o4d20 0.262 / 0.163, o4d16 0.121 / 0.091, o4d12 0.0615 / 0.048; o3d8 stopped at the same gate (1.28e-8). Against
main on that GPU (job 2295415): 2.8x / 2.8x at o4d20, 1.6x / 1.5x at o4d16, 1.08x / 1.07x at o4d12. Symmetrix on
it (job 2295397, before its recoupling update): 0.121 at o4d20, 0.056 at o4d16.
