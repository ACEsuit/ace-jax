# pace/kk prototype kernels

These patches modify LAMMPS (GPL-2.0) and are distributed under GPL-2.0, not the repository's MIT
licence (see `pace_ref/README.md`, Licences).

Patches to LAMMPS develop ec02ed0's `src/KOKKOS/pair_pace_kokkos.{cpp,h}`, applied in order by
`bench/scaling/envs/sulis-pacetab.sh gpufork` against the ML-PACE of ICAMS/lammps-user-pace#31 (which carries the
CPU speed-ups, tabulated radials and `summed_neighbour_A`; the ML-PACE side is no longer kept here). They are a
prototype for a LAMMPS PR, not yet proposed: the device path is chosen at run time by `PACE_KK_RHOW`.

| patch | what | `PACE_KK_RHOW` |
|---|---|---|
| 0003 | fused ComputeRho + ComputeFS + ComputeWeights, one team per atom, A and weights in shared memory; also the term tables 0006/0007 build on | 1 |
| 0005 | `summed_neighbour_A` in pace/kk (needs `ACE_HAS_SUMMED_NEIGHBOUR_A` from ML-PACE) | any |
| 0006 | ComputeRho / ComputeWeights in the stock thread layout with precomputed flat offsets | 2 |
| 0007 | 0006's ComputeRho plus a segmented ComputeWeights: target-sorted records, one atomic per run of 16 | 3 |
| 0008 | the densities `rhos` (and the team kernel's partial sums) in `KK_ACC_FLOAT`: double under `KOKKOS_PREC=mixed`, unchanged otherwise; the terms stay `KK_FLOAT` | any |

`PACE_KK_RHOW=0` (the default) runs the stock kernels. On an A100 (Si linear ACE, own input, µs/atom-step) mode 3
is 1.21x faster than stock at o4d12/o4d14, 1.51x at o4d16 and 1.71x at o4d20 (6.55 -> 3.82); mode 1 gained nothing.
On fp64-poor cards (RTX A4500) mode 3 is still faster than stock but slower than mode 2. Forces agree with stock to
2e-13 eV/Å. `accuracy_cost/envs/sulis-gpu-pacekk-profile.sbatch` times and profiles (nsys, ncu) the modes.
