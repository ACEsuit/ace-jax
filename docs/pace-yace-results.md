# PACE `.yace` import: deployment results (Task 9)

Date 2026-09-24. Branch `feat/pace-yace`. Raw numbers are in
`bench/pace_modal/last_results.json`, produced by `modal run bench/pace_modal/run.py`
on one Modal **L4** GPU.

- **LAMMPS:** `stable` (git), built with ML-PACE, KOKKOS/CUDA (`Kokkos_ARCH_ADA89`) and the Python module.
- **PACE library:** the version LAMMPS pins, `v.2023.11.25.fix2`.

## Parity with LAMMPS

E, F and ASE-convention stress, ace-jax (f64) vs LAMMPS `run 0`:

| model | atoms | `pace` (recursive) | `pace product` | `pace/kk` (GPU) |
|---|---|---|---|---|
| `c_ace.yace` (published carbon, LAMMPS examples) | 216 | dE 2.1e-12 eV/atom, dF 9.7e-9, dS 2.1e-9 | same | same |
| `Cu-1.yace` (published copper, LAMMPS examples) | 108 | dE 3.4e-13, dF 1.6e-9, dS 1.3e-9 | same | same |
| `sige_zbl` fixture, bulk | 64 | dE 1.3e-15, dF 4.0e-10, dS 1.9e-8 | same | same |
| `sige_zbl` fixture, close contact (ZBL active) | 64 | dE 4.9e-15, dF 9.1e-10, dS 2.6e-8 | same | same |

Units: dE in eV/atom, dF in eV/Å, dS in eV/Å³.

**Export:** LAMMPS reads a file written by `write_yace` (`c_ace`, unchanged model) with ΔE = 0.

These differences sit at the level of the C++ spline tables (`deltaSplineBins`), with ace-jax's radials being analytic. This also confirms the python-ace parity used in the test suite.

## Performance

4096-atom diamond carbon cell, `c_ace.yace`, rcut 5.5 Å, 499,712 edges, one L4:

| | energy only | energy + forces + virial | peak GPU memory |
|---|---|---|---|
| LAMMPS `pace/kk`, `product` (double) | n/a | **0.040 s**/step (103k atom-steps/s) | n/a |
| ace-jax, f64, jitted | **0.039 s** | 0.275 s (15k atom-steps/s) | 3.1 GB |
| ace-jax, f32, jitted | 0.165 s | 0.364 s (11k atom-steps/s) | 1.5 GB |
| LAMMPS `pace`, 1 CPU core | n/a | 1.82 s/step | n/a |

The LAMMPS step time includes neighbour-list upkeep and integration. The ace-jax time is one `energy_forces_virial` call on a fixed edge list.

**Reading:**
- The ace-jax **forward pass already matches `pace/kk`'s whole step in f64**. The 7× gap is the reverse (force) pass.
- The spec's candidate levers all target that pass:
  - the one-hot "matmul" A-form (its adjoint is a matmul, not a scatter);
  - a per-bond-type `crad` contraction instead of the per-edge `crad[zi, zj]` gather;
  - computing ρ only for each node's own element instead of for all NZ.
- **f32 is slower than f64 here** (4× in the forward pass). This is after `9b14096` removed every float64 array from the f32 graph, which a test now checks. The cause isn't established. XLA reports "very slow compile" for the f32 reduction fusions, which points at fusion and codegen choices rather than arithmetic. It's an open question.
- The L4 has 1/64-rate FP64. On data-centre GPUs (A100/H100) both f64 columns would be several times faster, and the ranking may change.
- XLA compile time on the GPU was about 5 minutes per large kernel. lammps-jax compiles ahead of time, so this is paid once per model and shape bucket, not per run.

## Size of the real-product basis (the spec's AA-union assumption)

| model | functions | PACE ms-combs | ace-jax real AA | local A entries | T non-zeros |
|---|---|---|---|---|---|
| `c_ace` | 475 | 3156 | 7406 | 314 | 7895 |
| `Cu-1` | 24 | 33 | 42 | 33 | 42 |
| `sige_zbl` (2 elements) | 120 | 152 | 125 | 34 | 184 |

The complex→real expansion costs at most about 2.3× on a production-size single-element model. Across centres, the two-element fixture merges to fewer real products than PACE ms-combs, which is consistent with the spec's assumption that the union is close to one element's set. A large multi-element production model is still unmeasured.
