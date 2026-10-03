# Total-energy summation and ASE Armijo line searches (results, 2026-10-03)

Question: does the plain (non-compensated) sum of per-site energies make the
total energy noisy enough to break ASE `PreconLBFGS` Armijo line searches on large
systems, and does a `math.fsum`-like compensated sum fix it?

Setup: `fixtures/si_fitted.npz` (E0 = -158.5 eV/atom is part of every site
energy, so |E| is about 163 eV x N), (110)[001] Si crack-tip clusters built with
matscipy (CLE K field at K_G from the model's own C_ij and relaxed (110) surface
energy, outer 6 A shell fixed) of 2k, 10k, 30k and 60k atoms. lestrade CPU
(float64 and float32) and RTX 4000 Ada. The scripts, logs, plots and the full
report are in `/storage/eng/essswb/projects/ace-jax/spikes/energy-sum/`
(`REPORT.md`).

## Findings

- **float64, CPU:** the XLA sum of the site energies inside the compiled
  energy/force step was within 1 ulp of `math.fsum` at every size (0 ulp at 2k,
  10k and 60k; 1 ulp, 9.3e-10 eV, at 30k). Permuting the atoms, and the sparse,
  dense and skin-list layouts, changed E by at most 1 ulp.
- **float64, GPU:** up to 3 ulp from `fsum`, and up to 12 ulp (2.8e-9 eV at 10k)
  between atom orders.
- **Per-site rounding** (float64) is about 9e-15 eV rms per site, about 2e-12 eV
  summed at 60k. That is three orders of magnitude below the floor.
- **The floor is ulp(|E|)** of the float64 that ASE receives: 5.8e-11 eV at 2k
  and 1.9e-9 eV at 60k, whatever the summation. Near convergence, the Armijo
  decrease c1 a g.p falls below it around fmax ~ 1e-5 (10k) to 1e-4 (60k). After
  that, line searches accept steps only where E(x + a p) == E(x) exactly (a ~
  1e-2 ... 1e-8). The steps collapse and progress stalls.
- **Relaxations (float64,** fmax ladder 1e-3 down to 1e-7, ensembles of rattled
  starts and a same-start A/B at 60k)**:**
  - Every variant reached 1e-6, and none reliably reached 1e-7.
  - Compensated against plain summation showed no systematic difference. The
    medians of energy/force calls in the 1e-6 stage were 41 against 52 (2k, 7
    runs) and 36 against 151 (10k, 3 runs). The outliers were 898 and 513.
  - From the same 60k start: plain reached 1e-6 in 158 calls, compensated in
    253, and the reference-subtracted sum in 16.
  - Removing the per-site reference (summing E_i - E_ref, so |E| is 10^3-10^4x
    smaller) did help: 8-15 calls for the 1e-6 stage at every size.
- **float32:** the total was a float32, quantised at its own ulp. That is
  0.03 eV at 2k and 1 eV at 60k: 10^2-10^3 times the sqrt(N) per-site float32
  rounding (5e-6 eV rms per site). The Armijo line search then stalled at fmax
  0.17 eV/A (2k) and 0.29 eV/A (10k). Accumulating the float32 site energies in
  float64 moved the stall to 0.0073 and 0.0086 eV/A, where the per-site float32
  rounding takes over.
- **Cost:** for 1e4-1e6 sites, `compensated_sum` takes 15-113 us on the RTX 4000
  Ada (f64), against 18-24 us for `jnp.sum`. On the CPU it takes 17-557 us,
  against 4-37 us. Either way that is under 0.1% of an energy/force call (2 s at
  60k atoms on CPU). The calculator call time was unchanged within noise at
  2k-60k atoms.

## What changed

`edge_model.total_energy` replaces `jnp.sum` in `energy_forces_virial` and
`energy_forces_virial_dense`. It forms the compensated (hi, lo) pair in the site
dtype and returns hi + lo in float64 when x64 is on. A float32 graph therefore
gains no float64 arrays (`test_pace_model`), only two widened scalars, and a
float32 model's total is exact to float64 rounding on the tests' data. Its
`custom_jvp` keeps the derivative of a plain sum, so forces and virials are
bitwise unchanged (`tests/test_energy_sum.py`). The skin step packs a float64
total as hi + lo in the model dtype (`calc.skin.unpack`).

- **What it buys in float64:** a correctly rounded total that does not depend on
  atom order, layout or device. It does not buy line-search robustness: that is
  limited by ulp(|E|).
- **What it buys in float32 (x64 on):** a 20-40x lower stalling fmax.
- **What it cannot fix:** float32 with x64 off. Its total is still a float32.

A reduced |E| lowers the float64 floor. `ACECalculator(energy_reference="E0")`
does that as an opt-in: the evaluation model's E0 is zeroed, so the per-atom
constant never enters the summed values, and `results["e0_offset"]` carries it.
The default energy convention is unchanged.

Measured on the (110) crack clusters, same seed-0 start and protocol as above
(PreconLBFGS, Armijo, float64). Energy/force calls in the 1e-5 -> 1e-6 stage:

| Atoms | naive | comp | shift (E - E_bulk) | E0-relative, naive | E0-relative, comp |
|---|---|---|---|---|---|
| 1952 | 201 | 41 | 10 | 20 | 10 |
| 10024 | 276 | 36 | 10 | 7 | 9 |

E0-relative energies (|E| about 4.6 eV x N instead of 163 eV x N) recover the
shift control's behaviour. These are single runs; trajectories are chaotic at
the 1-ulp level.
