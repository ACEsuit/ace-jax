# fp32 GPUs: step 1 measurements (2026-10-15)

Plan: `docs/dev/plans/2026-10-15-fp32-gpus.md`. Scripts: `bench/fp32/`. Raw
outputs (row dumps, JSON): `/storage/eng/essswb/projects/ace-jax/runs/fp32_step1/`.

**Setup.**
- Models: the Si accuracy-cost fits `runs/accuracy_cost/ace_per_quantity_noise/`
  o3d8, o3d12, o4d12 and o4d16 (L = 62, 223, 350 and 1461 columns), with their
  MAP θ (`theta_map.json`) and prior γ.
- Data: every 8th configuration of the GAP-18 training set with at most 256
  atoms. That gives 279 configurations and 59,928 rows (E, F, V), weighted with
  `weights_acejax.json`.
- Hardware: moriarty, RTX A4500 (fp64 at 1/64 of the fp32 rate), jax 0.11.2
  with CUDA 12.
- fp32 means a separate process with x64 off. Every array the library builds
  is then float32. The Dataset is built in numpy float64 and cast on transfer,
  so `rij` is formed in fp64 and then rounded.
- "TF32" is the same run with JAX's default GPU matmul precision. All other
  runs set `jax_default_matmul_precision="highest"`.
- Check: refitting the full fp64 model's readout on these rows reproduces the
  run's own `metrics.csv`, for example o3d8 at 86.7 meV/atom and 206 meV/Å
  against 87.5 and 207.

**Caveat.** The subsample has 1/8 of the data, so κ is higher and the
posterior wider than in the real fits. θ is the full-data MAP, so the LML
gradient at θ is not zero here.

## Results

### 1. Row error (fp32 vs fp64, relative)

Frobenius error over each block, with the worst column (column-norm relative)
in brackets.

| L | E | F | V | site X |
|---|---|---|---|---|
| 62 | 6e-7 (2e-6) | 2e-6 (4e-6) | 1e-6 (3e-6) | 6e-7 |
| 223 | 1e-6 (4e-6) | 2e-6 (5e-6) | 3e-6 (4e-6) | 1e-6 |
| 350 | 1e-6 (4e-6) | 2e-6 (5e-6) | 3e-6 (4e-6) | 1e-6 |
| 1461 | 7e-7 (6e-6) | 3e-6 (5e-6) | 2e-6 (6e-6) | 9e-7 |
| **TF32**, 1461 | 7e-7 | **5e-4 (2e-3)** | **4e-5 (3e-4)** | 9e-7 |

- fp32 rows are good to a few ε₃₂, uniformly across columns. No column type or
  degree loses more.
- **TF32 silently costs 2–3 orders of magnitude on the force and virial rows.**
  The edge-Jacobian contraction runs as a dot product. The energy rows are
  unaffected.

### 2. Conditioning

κ of the weighted design, with "+prior" meaning augmented with diag(γ/σ_c),
which is the system the QR factors:

| L | raw | Jacobi | +prior | +prior, Jacobi |
|---|---|---|---|---|
| 62 | 5.5e10 | 1.2e10 | 2.4e7 | 5.7e6 |
| 223 | 3.4e17 | 7.4e15 | 1.1e9 | 2.2e7 |
| 350 | 3.5e17 | 8.2e15 | 1.4e9 | 3.8e7 |
| 1461 | 3.5e19 | 1.3e16 | 1.4e10 | 8.0e6 |

The prior-augmented system is at or beyond 1/ε₃₂ ≈ 1.7e7. Jacobi column
scaling brings it back to 6e6–4e7.

### 3. Effect on the fit

Readout solved on the prior-augmented system, with predictions compared on the
fp64 rows. The table gives the RMS change in predictions against the fp64-rows,
fp64-QR solution, as E meV/atom / F meV/Å. For scale, "post. std" is the BLR
posterior std of the mean prediction, and the training RMSE is also shown.

| L | train RMSE | post. std | fp32 rows, fp64 QR | fp32 rows, fp32 QR | fp32 rows, Jacobi fp32 QR | TF32 rows, fp64 QR | fp32 normal eq. |
|---|---|---|---|---|---|---|---|
| 62 | 69 / 206 | 6.7 / 5.8 | 1.3 / 1.8 | 1.3 / 1.7 | 1.3 / 1.8 | 2.3 / 3.5 | Cholesky fails |
| 223 | 62 / 131 | 5.9 / 7.6 | 0.09 / 0.19 | 0.12 / 0.19 | 0.08 / 0.18 | 1.7 / 6.2 | fails |
| 350 | 19 / 123 | 9.6 / 8.9 | 1.6 / 1.1 | 1.7 / 1.1 | 1.6 / 1.1 | 5.3 / 14 | fails |
| 1461 | 14 / 89 | 8.8 / 13 | 0.04 / 0.14 | 0.13 / 0.15 | 0.08 / 0.16 | **10 / 7.2** | fails |

- **fp32 rows move the predictions by 1–20% of the posterior std.** The fit
  barely notices.
- **The coefficients change by O(1) relative** (`coef_rel` 0.4–6), as κ
  predicts. Those directions are what the data leaves undetermined.
- **A fp32 Householder QR adds almost nothing to the row-rounding error, even
  at κ = 1.4e10.** It is backward stable, and the prediction error follows the
  backward error, not κ.
- **The fp32 normal equations fail outright at every size.**
- **TF32 rows are not acceptable.** On o4d16 they raise the energy RMSE from
  13.7 to 20.6 meV/atom.

**LML and θ.**
- The LML is about 1e4–5e4 nats, and its gradient in log σ_{c,E,F,V} was taken
  by central differences.
- fp32 rows change the LML by 0.2–8 nats.
- The gradient errors, divided by the LML's diagonal curvature, shift log σ_E,
  log σ_F and log σ_V by under 0.006 (o4d16: 0.001, 5e-5 and 0.004).
- log σ_c is weakly identified (a flat or non-concave direction on this
  subsample), so its shift estimate is unreliable. On o4d16 it is 0.01.
- **fp32 QR with y as an extra column (`stats._qr_merge`'s form) gives the same
  LML and gradient as fp64 QR on the same rows, to within 0.1 nat.**
- A first version computed the residual as y·y − qᵀq. In fp32 that form
  cancels, and it gave gradient errors of 1e2–2e3 (wrong σ_E by e^2.6). The
  algorithm matters, not the precision.

### 4. Readout cancellation and fp32 inference

Per site, Σ_k |c_k B_k| / |Σ_k c_k B_k| (fitted readouts):

| L | median | p99 | max \|c\| | site error, all-fp32 readout (meV) | fp32 B, fp64 accumulation |
|---|---|---|---|---|---|
| 62 | 3000 | 5700 | 2950 | 1.0 rms, 6.7 max | 0.87 |
| 223 | 2100 | 5100 | 1600 | 0.73, 3.9 | 0.52 |
| 350 | 4200 | 21000 | 1600 | 1.1, 9.8 | 0.56 |
| 1461 | 450 | 2100 | 46 | 0.23, 2.6 | 0.03 |

The many-body block dominates Σ|c B|. The small bases have large coefficients
(up to about 3000) that cancel 10³-fold. o4d16's readout is better
conditioned. With fp64 accumulation it is good to 0.03 meV/atom. For the small
bases, rounding B itself dominates, so fp64 accumulation only halves the error.

`ACECalculator(dtype=float32)` (x64 on, lean, dense skin path) against
`dtype=float64` on the 248 test configurations:

| L | E rms / max (meV/atom) | F rms / max (meV/Å) | \|ΔF\|/\|F\| | stress rel. | 4096-atom step f64 → f32 (ms) |
|---|---|---|---|---|---|
| 62 | 0.63 / 3.7 | 0.86 / 3.8 | 1.1e-3 | 7e-4 | 5.0 → 2.4 |
| 223 | 0.41 / 2.0 | 0.48 / 2.7 | 6e-4 | 6e-4 | 6.9 → 3.7 |
| 350 | 0.60 / 3.3 | 0.51 / 2.7 | 6e-4 | 1e-3 | 7.3 → 3.8 |
| 1461 | 0.07 / 0.49 | 0.05 / 0.59 | 7e-5 | 3e-4 | 20.6 → 8.5 |

- **fp32 inference costs about 1e-4–1e-3 in relative force error.** That is
  1000× ε₃₂, and it tracks the readout's coefficient size. It is negligible
  against these models' errors (89–206 meV/Å).
- **Energies are noisy at 0.1–0.6 meV/atom rms.** That matters for small
  energy differences (defects, phase ordering), and it sets an MD energy-drift
  floor. The total is already summed compensated in fp64
  (`edge_model.total_energy`, `docs/dev/energy-sum-results.md`).
- **TF32 made no difference on the calculator path,** which uses no dot
  product affected by the default precision. Step 2 must still pin the
  precision on the `matmul` and `block_dense` paths.
- **2.1–2.4× per step on the A4500,** in line with the 2.7–3.8× in
  `docs/dev/benchmarks.md`.

### 5. Throughput of the mixed fit (rows on the GPU, merge on the host)

GPU row assembly per batch (2 configurations, about 430 rows) against the host
fp64 dgeqrf merge, amortised per batch over ~L buffered rows:

| L | GPU rows f64 | GPU rows f32 | host merge f64 |
|---|---|---|---|
| 62 | 4.5 ms | 1.8 ms | 1.1 ms |
| 223 | 17 ms | 3.6 ms | 7 ms |
| 350 | 22 ms | 4.7 ms | 13 ms |
| 1461 | 125 ms | 14 ms | 35 ms |

- fp32 cuts row assembly 2.5–9× (o4d16: 125 → 14 ms).
- **At L ≳ 200 the host fp64 merge, not the GPU, becomes the bottleneck.** A
  GPU fp32 → host fp64 pipeline is merge-bound at production basis sizes.
- **The merge depends on the thread count.** For o4d16, one merge of L rows
  (2922 × 1461 dgeqrf) takes 574, 212, 152, 121 and 154 ms with 1, 4, 8, 16
  and 32 OpenBLAS threads. The best, 16 threads, is about 86 GFLOP/s. The
  table uses it.
- **Set the thread count explicitly.** The session's default ran one thread.
  `nproc` reports 1 in a login shell, so OpenBLAS sized itself to one core.

### 6. ML-PACE (KOKKOS) in reduced precision, against ace-jax on the same `.yace`

**Setup.**
- LAMMPS develop (lestrade's `lammps-dev` source, 2 Oct 2026) with ML-PACE
  v.2025.12.4.p1, built for the A4500 at `KOKKOS_PREC=double|mixed|single`
  (`bench/fp32/build_mlpace.sh`).
- `pair_style pace product`, one GPU, `newton on neigh half`.
- ace-jax: `ACECalculator(dtype=...)` standalone on the same file
  (`bench/fp32/mlpace_prec.py`).
- Cells: 4096-atom diamond Si, rattled 0.05 Å (timed: NVE steps after a
  warm-up) and rattled 0.25 Å (strongly distorted, minimum bond 0.81 Å; step 0
  only, because NVE at 1 fs loses atoms from there).
- Errors are against ML-PACE double. ace-jax fp64 matches it to 2e-8–6e-8
  relative in force, which shows both codes evaluate the same model.

Step time (ms), cell rattled 0.05 Å:

| model | ML-PACE double | mixed | single | ace-jax fp64 | ace-jax fp32 |
|---|---|---|---|---|---|
| pacemaker f400 | 5.45 | 1.92 | 1.71 | 12.9 | 4.03 |
| pacemaker f1352 | 20.0 | 6.72 | 6.43 | 36.0 | 8.69 |
| `ace_yace/o4d16` | 33.2 | 13.9 | 13.3 | 193 | 15.9 |

The ML-PACE times are MD steps inside LAMMPS. The ace-jax times are standalone
calculator calls, so the comparison between codes is indicative only. The fair
comparison is the fp32 lammps-jax bundle (step 2).

Error against ML-PACE double, as ΔE (meV/atom) / ΔF rms / ΔF max (meV/Å):

| model, cell | ML-PACE mixed | ML-PACE single | ace-jax fp32 |
|---|---|---|---|
| f400, r0.05 | 0.001 / 0.019 / 0.08 | 0.006 / 0.019 / 0.08 | 0.006 / 0.016 / 0.06 |
| f400, r0.25 | 0.003 / 0.67 / **35** | 0.004 / 0.66 / 34 | 0.004 / 0.027 / **1.2** |
| f1352, r0.05 | 0.009 / 0.052 / 0.21 | 0.006 / 0.053 / 0.20 | 0.006 / 0.049 / 0.21 |
| f1352, r0.25 | 0.006 / 0.73 / **24** | 0.008 / 0.70 / 24 | 0.008 / 0.13 / **4.5** |

- **On these models, KOKKOS ML-PACE gains 2.7–3.2× from reduced precision on
  the A4500.** mixed and single give the same accuracy. single is about 5%
  faster.
- **Near equilibrium, ML-PACE in float and ace-jax in fp32 are equally
  accurate:** ΔF about 2–6e-5 relative and ΔE under 0.01 meV/atom. These
  pacemaker models have small coefficients (|c̃| ≤ 4.8, |crad| ≤ 5.3), so
  §4's cancellation does not arise.
- **On the distorted cell, ace-jax fp32 is 5–25× more accurate than ML-PACE in
  float** (max ΔF 1.2–4.5 against 24–35 meV/Å). ML-PACE's single and mixed
  errors are the same, so the loss is in its float evaluation (spline tables,
  radial functions), not in its accumulation.
- **The fp64 gap is wide.** ML-PACE double is 1.8–6× faster than ace-jax fp64
  standalone, but in fp32 the gap closes to 1.3–2.1× (and 1.1× on
  `ace_yace/o4d16`).
- **`ace_yace/o4d16` is unusable in reduced precision in both codes.**
  - Errors: ML-PACE mixed ΔF/F = 3.8, ace-jax fp32 2e4.
  - Cause: `ace_to_yace.py` refits the ACE radials onto 80 SBessel functions,
    giving |crad| up to 3.7e8 (median 8e5, against at most 5.3 for pacemaker's
    fit) and |c̃| up to 6e5. Those coefficients cancel far beyond fp32.
  - On the distorted cell it diverges even in double (ΔE about 1e17–1e20
    meV/atom): the refit radial is wrong outside its fitted range.
  - So for ACE fits in fp32, use the native `.npz` or the `Tabulated` `.yace`
    (`write_yace`), never the SBessel refit.

## Step 2: fp32 inference in LAMMPS (2026-10-15)

### A bug: `aj export --dtype float32` bundles were float64 programs

Under x64, which `aj` always enables, `export_lammps(dtype="float32")` traced the float64
model with float32 positions. The bundle lowered to a float64 program (about 400–490 f64
ops) with only its I/O in float32. It got no float32 speed, and its inputs were rounded.
`ACECalculator(model, meta, dtype=float32)` with a pre-loaded model did the same. The
scaling suite never showed it, because its export child turns x64 off.

Fixed in `dbbba90`:
- `eval.with_dtype` casts a model's floating leaves.
- `export_lammps` casts the traced model after lean, splines and tables, which are
  computed in float64.
- The calculator reads the file in float64 and casts only `eval_model`.
- `tests/test_fp32.py` checks that a float32 bundle lowers with no f64 tensor. Without
  the cast, 4 of its tests fail.

Reading the file in float64 and casting afterwards does not change §4's numbers: o3d8
through o4d16 give the same float32 error either way. The error is the model's.

`calc.precision.precision_check(model, meta, atoms)` measures it: float32 against float64
energy, force and stress on one structure.
- `export_lammps(precision_structure=)` records the result as `ace_jax.precision_check`.
- It warns above `PRECISION_WARN`: 0.1 meV/atom of energy, 1e-4 of relative force.
- `aj export --dtype float32` checks on `--structure`.
- On `si_fitted` (64 atoms): 0.002 meV/atom and 6e-6 force. NVE drift over 500 fs at 600 K
  is the same in float32 as in float64 (about 0.02 meV/atom).

### Bundles against ML-PACE, in LAMMPS on the A4500

Setup (`bench/fp32/bundle_prec.py`):
- lammps-jax `pair_style jax/kk`, in the scaling suite's moriarty LAMMPS (develop, 25 Sep
  2026, KOKKOS double) with its plugin. Bundles exported under x64 with the fix.
- ML-PACE `pace/kk product` at each `KOKKOS_PREC` (§6 builds).
- Diamond Si rattled 0.05 Å: 4096, 32768 and 262144 atoms (boxes 43, 87 and 174 Å).
- Reference: the ace-jax float64 calculator. The bundle and ML-PACE in float64 match it
  to 5.9e-8.
- Times are per NVE step (0.5 fs, 100 steps after a 50-step warm-up), matrix layout.

Step time (ms):

| model, atoms | bundle f64 | bundle f32 | ML-PACE double | mixed | single |
|---|---|---|---|---|---|
| f400, 4096 | 16.2 | 3.40 | 5.42 | 2.00 | 1.79 |
| f400, 32768 | 131 | 26.1 | 43.9 | 16.0 | 14.4 |
| f400, 262144 | 1156 | 216 | 357 | 131 | 117 |
| f1352, 4096 | 40.3 | 8.50 | 20.3 | 6.74 | 6.46 |
| f1352, 32768 | OOM | 78.6 | 164 | 55.0 | 52.5 |
| ACE o4d16, 4096 | 30.8 | 8.53 | — | — | — |
| ACE o4d16, 262144 | 2503 | 1385 | — | — | — |
| ACE o3d8, 4096 | 4.56 | 1.18 | — | — | — |
| ACE o3d8, 262144 | 378 | 86.3 | — | — | — |

Force error, relative to the float64 reference (ΔF max in meV/Å in brackets):

| model, atoms (box) | calc f32 | bundle f32 | ML-PACE mixed | ML-PACE single |
|---|---|---|---|---|
| f400, 4096 (43 Å) | 1.9e-5 | 2.3e-5 (0.08) | 2.3e-5 (0.08) | 2.3e-5 |
| f400, 32768 (87 Å) | 1.9e-5 | 3.3e-5 (0.13) | 3.3e-5 (0.13) | 3.3e-5 |
| f400, 262144 (174 Å) | 1.9e-5 | 5.7e-5 (0.29) | 5.7e-5 (0.31) | 5.7e-5 |
| f1352, 4096 | 5.9e-5 | 6.5e-5 | 6.5e-5 | 6.4e-5 |
| ACE o4d16, 4096 | 3.0e-5 | 3.4e-5 | — | — |
| ACE o4d16, 262144 | 3.2e-5 | 6.3e-5 (0.30) | — | — |
| ACE o3d8, any | 1.1e-3 | 1.1e-3 (4) | — | — |

- **On these GPUs, the float32 bundle is 3.6–5.3× faster than the float64 bundle.** The
  exception is o4d16 at 262k atoms, at 1.8×. ML-PACE gains 2.7–3.1× from `mixed` and
  `single`.
- **The float32 bundle beats ML-PACE double by 1.6–2.4×.** ML-PACE mixed/single stays
  1.3–1.9× ahead of the float32 bundle on the pacemaker models. That ratio is close to
  the float64 one (`docs/dev/benchmarks.md`).
- **The float32 bundle and ML-PACE mixed/single are equally accurate** at every size
  (2e-5 to 7e-5 relative force).
- **The error grows with the box in both.** The calculator forms displacements in float64
  and does not grow (1.9e-5 at all sizes). The bundle and ML-PACE in float both rise from
  2.3e-5 to 5.7e-5, 43 → 174 Å. Both form rij from float32 absolute positions (ulp about
  |x| × 6e-8). At a million atoms (about 280 Å) expect about 1e-4. That is still well
  below these models' force errors (89–206 meV/Å), but it is the dominant float32 error
  on a well-conditioned model at scale.
- **Energy is returned in float32 and quantised at ulp(|E_total|).** Because E0 is
  included, |E| is about 163 eV × N, which gives 0.015 meV/atom against the calculator,
  whatever N is. lammps-jax casts the bundle's energy to the contract dtype.
- **A model with large coefficients stays at its own float32 error.** o3d8 is at 1.1e-3
  (4 meV/Å max) everywhere, as `precision_check` predicts.
- **Two issues, resolved below:** o4d16 scaled superlinearly at 262144 atoms, and the
  float64 bundle ran out of memory on f1352 at 32768 atoms.

### Resolved: block size, and the out-of-memory error

**Cause of the o4d16 superlinearity.** The bundle runs dense and matrix rows in blocks
(`lax.map` + `jax.checkpoint`). `block_rows_for` took the largest halving of 32768 rows
that fits the memory budget, a policy tuned on an A100 with Cantor (fp64, dense layout,
`perf-lammps-large-n.md`). On the A4500 bigger blocks are slower (`bench/fp32/diag_blocks.py`,
LAMMPS step ms, 32768 atoms, matrix layout):

| model, dtype | 1024 | 2048 | 4096 | 8192 | largest that fits (rows) |
|---|---|---|---|---|---|
| Si ACE o4d16, f32 | **58.2** | 74.8 | 75.0 | — | 129 (16384) |
| Si ACE o4d16, f64 | **183** | 284 | 288 | — | 324 (8192) |
| Si ACE o3d8, f32 | 11.6 | 10.1 | 9.4 | — | 9.5 (32768) |
| Si PACE f1352, f32 | 67.7 | 70.6 | 69.4 | — | 78.9 (32768) |
| Si PACE f1352, f64 | 325 | 313 | 320 | — | 300 (32768) |
| Si PACE f400, f32 | 27.1 | 27.8 | 26.7 | — | 26.2 (32768) |
| Cantor ACE medium, f64 | 119 | 115 | 110 | 107 | 108 (16384) |
| Cantor ACE medium, f32 | 23.5 | 21.6 | 21.9 | 21.5 | 25.4 (32768) |
| Cantor PACE medium, f64 | 131 | 125 | 121 | 121 | 128 (32768) |
| Cantor ACE large, f32 | 21.3 | 21.3 | 20.0 | 19.4 | 23.6 (32768) |

- The best block size tracks the per-row size of the blocked program, measured in
  *elements*: the same rows are best in float32 and float64 (o4d16: 1024 in both).
- The rule: target about 2^26 elements per block (`estimate_a_bytes` / itemsize). Take
  the largest halving at or under it, at least 1024 rows. That picks 1024 rows for o4d16
  (65k elements/row), 2048 for f1352 and Cantor ACE medium, and 4096 for o3d8, f400,
  Cantor PACE medium and Cantor ACE large.
- Against the best measured size, it is within 7% everywhere. Against the old policy it
  is 1.15–2.65× faster on 7 of 10 cases, and up to 7% slower on 3: Cantor ACE medium
  fp64 (115 against 108 ms), f1352 fp64 (313 against 300) and o3d8 fp32 (9.6 against 9.5).
- The memory budget still caps the block.

With the change (`export.lammps.BUNDLE_BLOCK_TARGET_ELEMS`), default exports:

| case | before | after |
|---|---|---|
| o4d16 f32, 4096 atoms | 8.53 ms (one block of 4506) | 7.5 ms (1024) |
| o4d16 f32, 32768 | 129 ms (16384) | 57.7 ms (1024) |
| o4d16 f64, 32768 | 324 ms (8192) | 181 ms (1024) |
| o4d16 f32, 262144 | 1385 ms (16384) | 520 ms (1024) |

o4d16 now scales linearly: 7.5 → 57.7 → 520 ms for 4k → 32k → 262k atoms.

**Caveat.** This was measured on the A4500 only. The earlier A100 finding was that
splitting a 16k-atom system into two blocks cost 1.5× through the checkpoint recompute.
That contradicts this result at 4096 atoms (5 blocks of 1024 beat one block). It predates
the lean and matrix code, and should be rechecked on an A100 (Sulis).

**The out-of-memory error did not reproduce.** The same f1352 float64 bundle (one
32768-row block, sized against 15.8 GB) ran at 300 ms when repeated. The o4d16 reference
calculator's out-of-memory error at 32768 atoms did not repeat either, and 262144 atoms
ran. Both look like transient GPU memory pressure. The new rule cuts that bundle's block
from about 7 GB to 0.44 GB (2048 rows), so the export's budget, which ignores what
LAMMPS holds on the GPU, matters much less now.

**For lammps-jax (upstream):**
- Pass a float32 bundle positions relative to a nearby origin, formed in double before
  the cast (for example per subdomain), or float64 positions with float32 compute. That
  makes rij independent of the box, as in the calculator.
- Optionally return the energy in float64. These are small effects, and they also apply
  to ML-PACE's own float builds.

## Conclusions so far

1. **Fitting in fp32 is feasible at these basis sizes.** Use fp32 rows with the
   y-as-a-column QR. The precision of the QR itself is secondary: fp32 Householder
   matched fp64 for both the mean and the LML. This changes the plan. **Step 3
   should try the merges in fp32 on the GPU** (`device_qr_stream`) before the
   host split, because the host merge would be the bottleneck.
2. **Open for step 3:**
   - error growth over many streamed merges (here, one QR of all rows);
   - the full-size datasets;
   - MACE-label fits, where σ_E sits at its floor;
   - the GP arm.
3. **Never use the fp32 normal equations or TF32 rows.** Both fail here.
4. **Inference in fp32 is fine for forces.** Energies carry 0.1–0.6 meV/atom
   noise, driven by readout cancellation. To reduce it:
   - accumulate the readout in fp64 (this helps the well-conditioned model);
   - use better-regularised fits (smaller |c|);
   - record the cancellation ratio at export.
5. **Step 2 delivered:**
   - float32 bundles and calculators are now float32 programs;
   - a per-model precision check is recorded at export;
   - the float32 bundle is 3.6–5.3× faster than float64 and beats ML-PACE double;
   - a block-size rule fixes the o4d16 slowdown (up to 2.65× faster in both dtypes);
   - ML-PACE mixed/single stays 1.3–1.9× ahead, about its float64 margin.
   Closing that gap is the existing performance work (`docs/dev/pace-performance-gap.md`),
   not a precision question.
6. **KOKKOS ML-PACE's reduced precision is the GPU baseline to beat.** It is
   2.7–3.2× faster than its double build at the same near-equilibrium
   accuracy as ace-jax fp32. ace-jax fp32 is clearly more accurate on strongly
   distorted structures. Step 2 must benchmark the fp32 lammps-jax bundle
   against ML-PACE `KOKKOS_PREC=single`/`mixed`.
7. **Model conditioning decides fp32 usability more than the evaluator does.**
   It shows in the readout cancellation (§4) and in `ace_yace` (§6). Record a
   cancellation measure at export, and keep coefficients small in fitting.
