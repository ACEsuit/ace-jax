# ML-PACE tabulated radials (draft patch)

A minimal addition to ML-PACE (ICAMS/lammps-user-pace) so a C-tilde `.yace` can carry its radial
functions as tables instead of coefficients over a fixed radial basis. Its purpose: ace-jax linear ACE
models (ACEpotentials-style radials: transform, polynomials, envelope; or learned radials) export to ML-PACE
**exactly**, and then run in `pair_style pace` and `pace/kk` (CPU and GPU) unchanged.

Patch: `pace_ref/tabulated/0001-tabulated-radials.patch`, on `v.2025.12.4.p1` (the release LAMMPS
develop pins). It also applies cleanly to master `99aa6e6`. The patch is local only and has not been
proposed upstream.

## Why the patch is needed

Upstream ML-PACE reads only coefficients `crad[n][l][k]` over its own `g_k` (ChebExpCos, ChebPow,
ChebLinear, SBessel). ACE1's old `radbasename: ACE.jl` spline input lived in a fork that was never ported to
Kokkos/GPU.

The alternative is to refit the radials to SBessel (tried in the accuracy-cost benchmark's converter). On o4d20 at 80
`g_k`, that leaves a 2.3 meV/Å force rms difference. It also fails the timing parity gate: ML-PACE's own
look-up table cannot follow the high-frequency `g_k` (2-9e-5 eV/Å against ace-jax on the same file).

## The change

ML-PACE never evaluates a radial basis during MD. `ACERadialFunctions::setuplookupRadspline` samples `g_k`
and `R_nl = crad . g` (values and r-derivatives) at `r_n = n * rcut / ntot`, `ntot = rcut / deltaSplineBins`,
into cubic-Hermite tables (`SplineInterpolator`). `pair_pace_kokkos` copies those host tables to the device.

The patch therefore only changes what the table builder samples:

- `TabulatedRadial` (in `ace_radial.h`) holds per bond the values and derivatives of the `g_k` (read by rank-1
  functions) and of each `R_nl`, on a uniform grid `r_i = i * rcut / nbins`. It interpolates them by the same
  cubic Hermite scheme.
- `ACERadialFunctions::tabulated_gk` / `tabulated_rnl` fill `gr, dgr` / `fr, dfr` from the tables. They are
  zero outside `(cut_in - dcut_in, cut)` and apply the inner cutoff exactly as `radbase` does.
- `setuplookupRadspline` binds them for a `Tabulated` bond. `all_radfunc` takes `R_nl` from the table instead
  of `crad . g`.
- `ACEBondSpecification` reads and writes `radtable`; `radparameters` and `radcoefficients` are optional for
  `Tabulated`.
- `ACECTildeBasisSet::load_yaml` hands the tables to `ACERadialFunctions` (in the global `nradmax x (lmax+1)`
  layout).

There are no Kokkos, evaluator or LAMMPS changes. With `nbins = rcut / deltaSplineBins`, the table grid
coincides with the look-up grid, so the look-up table reproduces the file's values and derivatives at every
node.

## The bond format

```yaml
bonds:
  [0, 0]: {radbasename: Tabulated, nradmax: 20, lmax: 14, nradbasemax: 1, rcut: 5.0, dcut: 0.01,
           prehc: 0, lambdahc: 0, rcut_in: 0, dcut_in: 0, inner_cutoff_type: distance,
           radtable: {nbins: 5000,
                      gk:  [{k: 1, f: [...nbins+1], df: [...]}],
                      rnl: [{n: 1, l: 0, f: [...], df: [...]}, ...]}}   # sparse; absent = zero
deltaSplineBins: 0.001        # = rcut / nbins: the look-up grid is the table grid
```

`k` and `n` are 1-based, as in `functions`. `df` is dR/dr.

## The ace-jax side

`write_yace(ace_model, path)` (`src/ace_jax/export/yace.py`) converts the model as follows:

- **Product basis:** exact. Real-Y products are expanded into PACE's complex Y and merged by sorted `(n, l, m)`.
  The coefficients come out exactly real.
- **Radials:** each radial column `R(r) r^l` (solid harmonics) becomes one tabulated `R_nl`. The values come
  from the model and the derivatives from forward-mode autodiff.
- **Rank-1 terms:** the pair radial plus the order-1 terms are one tabulated `g_1`, read by one rank-1
  function with `ctilde = 1`.
- **Short range:** below 0.5 Å the tables continue linearly, because the ACE pair envelope is singular at r -> 0.

Sizes: o4d12 2.7 MB, o4d20 17 MB (97 radials x 5001 nodes x 2). ML-PACE loads o4d20 in about 6 s.

## Validation

`pace_ref/tabulated/` holds the local check (build against a patched lammps-user-pace with its CMakeLists.txt, `-DPACE_SRC=...`):

- `tab_eval.cpp` is a short driver over `ACECTildeEvaluator`, with a brute-force periodic neighbour list
  and forces accumulated as `pair_pace` accumulates them. Each atom line carries its element index.
- `compare.py <tab_eval> <model.npz>` exports the model with `write_yace`, evaluates rattled, strained,
  mixed-species diamond cells (64 atoms) with ML-PACE and compares with `ACECalculator` on the `.npz` (the
  whole export) and on the `.yace` (the two evaluators on the same tables).
- `tab_time.cpp` times ML-PACE's recursive or product evaluator per atom on a rattled diamond cell.

| model | test configurations | max dE/atom | max dF | rms dF |
|---|---|---|---|---|
| pacemaker f800 (stock radials; checks the driver) | 10 | 4.6e-13 eV | 3.8e-9 eV/Å | 7.1e-10 |
| ACE o4d12, tabulated | 10 | 2.1e-12 eV | 3.7e-9 eV/Å | 9.6e-10 |
| ACE o4d20, tabulated | 10 | 6.5e-13 eV | 4.3e-9 eV/Å | 1.2e-9 |

The residual is ML-PACE's own look-up-table interpolation, at the level stock files already show.

Multi-species (`compare.py`, `delta` 0.001 Å, the patch on master `99aa6e6`):

| fixture | elements | file | vs `.npz`: max dE/atom, max dF | vs `.yace`: max dE/atom, max dF |
|---|---|---|---|---|
| `si_ace_model` | Si | 4.4 MB | 2.8e-11 eV, 7.4e-6 eV/Å (max \|F\| 166) | 8.5e-14 eV, 6.1e-11 eV/Å |
| `sige_nofit` | Si, Ge | 9.5 MB | 7.1e-14 eV, 6.6e-9 eV/Å | 1.8e-15 eV, 2.3e-12 eV/Å |
| `ace_cantor5_small` | Cr, Mn, Fe, Co, Ni | 30 MB | 1.1e-14 eV, 1.1e-10 eV/Å | 1.2e-17 eV, 1.4e-13 eV/Å |

Against the `.yace`, ML-PACE and ace-jax agree to roundoff, species pair by species pair. Against the
`.npz`, the difference is the cubic Hermite tables' O(h^4) error (4e-8 of max |F| on the Si fixture's
strongly compressed cells). Kokkos (`pace/kk`) was run on single-species files only (A100, the benchmark).

The accuracy-cost benchmark (branch `feat/accuracy-cost-bench`) times these exports in LAMMPS against a build of the patched library (`LOCAL_ML-PACE`).

## Performance: what an ACE export must avoid

The first exports ran o4d20 3.8x slower than PACE f3200 on the CPU node, and 1.5x slower than ace-jax on
the A100. Both causes were in the converter, and both fixes are exact (same E/F to 1e-12):

1. **Write lmax and nradmax as the products use them, not as the model has them.** ML-PACE does its
   per-neighbour work over the whole `nradmax x (lmax+1)` rectangle: splines for every `R_nl`, and A for every
   `(n, l, m >= 0)`. On the GPU this is one thread per (atom, neighbour) doing two global atomics per entry.
   o4d20 has lmax 14 (its order-1 Y) and 20 radial columns. Its rank > 1 products read only l <= 6 and
   19 / 16 / 13 / 10 / 7 / 4 / 1 radials at l = 0..6: 196 of 2400 A entries per neighbour. The order-1 terms
   are folded into the rank-1 `h(r)`, so their radials need no `R_nl`. Trimmed: lmax 6, nradmax 19, 532
   entries.
2. **Write a half basis, as pacemaker does.** An invariant has `sum(m) = 0`, so `P(-m) = conj(P(m))`, and an
   m-tuple and its negation give the same `Re(c P)`. Keep one with the summed coefficient: o4d20 has 45k ->
   26k product terms. The GPU's `ComputeWeights` (the backward pass through the products; 70% of pace/kk's
   time) and `ComputeRho` scale with the term count.

Serial per-atom time, ML-PACE's recursive evaluator (`pair_style pace`, one core, 512 atoms):

| model | as first exported | + trimmed lmax/nradmax | + half basis |
|---|---|---|---|
| PACE f3200 (pacemaker) | 355-406 us | | |
| ACE o4d16 | 331-340 us | 133 us | 104-124 us |
| ACE o4d20 | 900-960 us | 541-580 us | 439-481 us |

A100, pace/kk product, 4096 atoms, per-kernel time over 120 steps (Nsight Systems):

| kernel (ms) | f800 | f3200 | o4d16 trimmed | o4d20 as first exported | o4d20 trimmed |
|---|---|---|---|---|---|
| ComputeWeights | 183 | 1201 | 374 | 2468 | 2561 |
| ComputeAi (radial splines + A atomics) | 54 | 84 | 100 | 922 | 183 |
| ComputeRho | 90 | 361 | 112 | 660 | 676 |
| ComputeDerivative | 30 | 53 | 47 | 316 | 95 |
| total | 382 | 1724 | 661 | 4403 | 3541 |

The half basis is not in this profile: it cuts the terms that `ComputeWeights` and `ComputeRho` loop over.

**Skipping unused (n, l) inside the rectangle** (a per-n lmax in the recursive evaluator's three
per-neighbour loops, branch `skip-radials`, not proposed) gave identical E/F and no speed-up once the
rectangle was trimmed: 439 -> 483 us/atom on o4d20 (noise), f3200 unchanged. The remaining cost is the
product basis, as for pacemaker models, so the patch stays radial-only.

## Open points

- python-ace (pyace) and `ace_b_basis` (B-basis `.yaml`) do not know `Tabulated`. Only C-tilde `.yace` is
  covered.
- File size: a coarser `nbins` than the look-up grid is allowed (the look-up table then re-samples the
  Hermite interpolant). A binary sidecar would be smaller than YAML.
