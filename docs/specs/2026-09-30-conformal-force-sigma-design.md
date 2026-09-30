# Conformal calibration of the ARD force σ (Mondrian groups, target-regime calibration, support flag)

Status: design, for review · Base: `main` (after #18) · Branch: `feat/conformal-uq`
Evidence: `bench/defect_uq` (#21): `scoring/conformal_shift.py`, `scoring/conformal_cv.py`, and
`acegp-data/results/2026-09-28/bench365_ard/ACCEPTANCE.md`.

## Goal

The served per-atom force uncertainty (`forces_std`, and a new guaranteed radius) should have a
**stated, verifiable coverage** instead of an rms-matched scale. It should also stay conservative where
the calibration data says nothing. Three pieces:

1. **Mondrian (group-conditional) split conformal calibration** replaces the scalar λ (or κ). There is
   one quantile per group of a geometric feature computed at prediction time.
2. **Target-regime calibration:** `aj calibrate` adds a user-supplied labelled set (e.g. a few MACE- or
   DFT-labelled crack cells) to the calibration scores of a fitted posterior.
3. **A covariate-shift support flag** (Tibshirani, Barber, Candès & Ramdas 2019), used as a
   **diagnostic**. It gives the per-atom weighted-conformal quantile against the calibration pool, from
   the unlabelled target structure, and flags atoms with no calibration support (an infinite quantile).
   It does not change the served σ.

## Why (benchmark facts this design rests on)

- **The in-distribution scale is already right.** On bench365 the conformal 90 % quantile of the
  normalised score is 2.48, against 2.50 for a calibrated Gaussian, with 0.902 held-out coverage.
  Conformal calibration changes the number little there, but it turns rms-z ≈ 1 into a coverage
  statement.
- **Out of distribution, marginal is not conditional.** Calibrating with labelled crack cells gives
  whole-cell coverage 0.893–0.899 (leave-one-realisation-out, 10 realisations, 30 cells). Crack tips
  (≤ 10 Å) still get 0.866 from plain split conformal. Mondrian grouping by local distortion ×
  coordination raises the tip to **0.885** (folds 0.864–0.900) and leaves the whole cell at 0.899.
- **Covariate-shift weighting has two regimes.**
  - With no target labels, weighting from the unlabelled crack structures lifts whole-crack
    coverage from 0.871 to 0.897. At the tip it mostly flags: 10.7 % infinite quantiles, and 0.858
    coverage on the finite ones.
  - With target labels in the pool it adds nothing over split conformal (0.898 against 0.893).
  - Its useful product is therefore the flag, not the scale.
- **The dislocations stay conservative** (0.92–0.94) under every method. That matches the stated
  preference: conservative but reliable beats overconfident.

## Mathematics

**Score.** Let v_i be the served, *unscaled* per-atom force variance: v_i = Σ_c ‖Qᵀφ̃_{i,c}‖² for
`--ard-variance sandwich`, or Σ_c φ_{i,c} A⁻¹ φ_{i,c}ᵀ for `kappa`. For a labelled atom,

  s_i = |ΔF_i| / √(v_i / 3),

where ΔF_i is the error of the *served* model. If √(v/3) were an exact isotropic Gaussian scale, s
would follow the χ₃ distribution.

**Mondrian split conformal.** Let g(x) ∈ {1..G} be a group label computable from geometry alone, and
let (s_i, g_i), i ∈ cal, be the calibration scores. For target coverage 1 − α, the group quantile is

  q_g = the ⌈(n_g + 1)(1 − α)⌉ / n_g empirical quantile of { s_i : g_i = g }.

The served quantities for an atom x in group g = g(x) are:
- `forces_q(x) = q_g · √(v(x)/3)`, the **guaranteed radius**: P(|ΔF| ≤ forces_q) ≥ 1 − α for an atom
  exchangeable with group g's calibration atoms;
- `forces_std(x) = (q_g / χ₃⁻¹(1 − α)) · √v(x)`, the σ whose Gaussian (1 − α) ball is exactly the
  conformal radius. It generalises the present λ·√v: the ratio q_g / χ₃⁻¹(1 − α) takes λ's place,
  one value per group. On in-distribution data it equals λ to within about 1 % (bench365: 2.48 / 2.50).

**Exchangeability unit.** Atoms within a configuration are correlated, so the exact guarantee holds
for configurations, not atoms. The atom-level quantile is the standard practical approximation.
`ard.json` and `aj calibrate` report both the number of atoms and the number of configurations per
group. Groups with fewer than `n_min_cfg` calibration configurations (default 5), or fewer than
`n_min_atoms` atoms (default 200), are merged into the nearest band. If that still fails, they fall
back to the global quantile, and the merge is recorded.

**Groups (the Mondrian feature).** The feature must be reference-free, because the benchmark's
version used the ideal fcc nearest-neighbour distance a0/√2. Let r₁ be the first minimum of the
training radial distribution function, computed at fit time and stored in the posterior. For atom i:
- **coordination** z_i = #{j : r_ij < r₁};
- **distortion** d_i = std/mean of { r_ij : r_ij < r₁ }, the dimensionless spread of the first shell;
  NaN when z_i < 2.

The group is the band of d_i, using band edges from the calibration pool's d quantiles
(default p50 / p90 / p99, so 4 bands), crossed with whether z_i equals the modal training
coordination z* (bulk-like or not). That gives G = 8 by default. Band edges, r₁ and z* are stored.
*This reference-free d must be re-validated* (see Acceptance). The benchmark used the rms deviation
of the 12 nearest bonds from a0/√2.

**Calibration sets.**
- **Default: the existing ARD train hold-out.** The errors come from the subset-fitted model, and v
  from the served full posterior with each atom's own cluster left out (the λ rule, unchanged). This
  replaces `lam = kappa_closed_form(e2, m2_full)` with per-group quantiles of
  s = √(e2 / (m2_full / 3)). λ is still reported.
- **`aj calibrate` (target regime).** A labelled xyz scored with the *served* model and posterior. It
  is **appended** to the stored hold-out scores by default (`--replace` uses it alone), and the group
  quantiles are recomputed.

**Support flag (diagnostic).**
- Calibration descriptors are stored at fit time: the whitened 2-body columns of the calibration
  atoms, float32, about 40 MB for the bench365 pool. A pool subsample cap (`support_max_atoms`,
  default 50k) bounds the size.
- For a target structure: a per-species L2 logistic classifier (calibration against target atoms)
  gives w(x) = P(t|x)/P(c|x), and the weighted quantile of the calibration scores has mass
  w(x)/(Σw + w(x)) at +∞.
- Returned per atom: `support_ok` (finite weighted quantile), `support_q` (the weighted quantile, or
  +∞), and per species `n_eff = (Σw)² / Σw²`.
- It never enters `forces_std` or `forces_q`.

## Interface

- **`FitConfig`:**
  - `ard_calibration` ∈ {"conformal", "gaussian"}, default **"conformal"** (see decision 1);
  - `ard_coverage` = 0.9;
  - `ard_groups` ∈ {"distortion", "none"}, default "distortion";
  - `ard_group_bands` = (0.5, 0.9, 0.99);
  - `n_min_cfg` = 5, `n_min_atoms` = 200.
- **CLI:** `--ard-calibration`, `--ard-coverage`, `--ard-groups`.
- **`ARDPosterior`, schema 3.** New fields, all None/empty for "gaussian" and for schema-2 files:
  - `conformal`: {coverage, r1, z_star, band_edges, q_g (G,), n_atoms_g, n_cfg_g, merged, source};
  - `cal_scores` (n,), `cal_groups` (n,) and `cal_cfg` (n,), float32 and int, so that `aj calibrate`
    can append and recompute;
  - `support`: whitened-pair calibration descriptors, the whitener per species, and the calibration
    scores, or absent when disabled.

  Schema 1 and 2 files still load and serve the λ/κ scale, with `forces_q` not available.
- **`ACECalculator(model, posterior=...)`** serves `forces_std` (as above) and a new
  `forces_q` (N,) via `get_property`. `get_property("forces_support", atoms)` returns a
  `{support_ok, support_q, n_eff}` dict. The group features come from the calculator's neighbour list
  (r₁ < r_cut), and per-atom `group` is exposed for inspection.
- **`aj calibrate --model model.npz --posterior posterior.npz --data cal.xyz [--energy-key ...]
  [--force-key ...] [--coverage 0.9] [--replace] [--out posterior_cal.npz]`.** It writes a new
  posterior file (never in place, by default) and prints the per-group before/after table and the
  provenance (sha256 of cal.xyz, n_cfg, n_atoms).
- **`aj eval --posterior ... --per-atom out.xyz`** adds `forces_q`, `group`, and, with `--support`,
  `support_ok`/`support_q`.

## Decisions (review these)

1. **Default "conformal".** It changes served σ values slightly, by up to about ±10 % per group, and
   adds a guarantee. The alternative is opt-in until one more fit has been validated.
2. **Atom-level quantile with configuration counts reported**, not a configuration-level score such as
   the maximum over atoms. A configuration-level score is exact but far too conservative for big cells.
3. **A reference-free distortion feature** (first-shell spread below the RDF minimum), not the
   benchmark's lattice-referenced one. Portable to non-fcc systems; must be re-validated.
4. **Covariate-shift weighting is a diagnostic only.** With no target labels it mostly produces
   infinite quantiles at the tip. With target labels it is redundant. So it flags, it does not scale.
5. **E and V are unchanged** (untempered, as now).

## Non-goals

- Closing the residual tip gap (about 1.5 points under target at crack tips). No feature tried —
  distortion, coordination, 2-body density ratio, 2-body kNN — separates tip atoms fully.
- Conformal calibration of energies or virials.
- Localized (kNN) conformal: it gave no gain over split conformal with 2-body features.
- Buffered block-wise MACE labelling for large calibration cells: a 3.3k-atom cell in float64 needs
  about 27 GB, so it labels on Modal.

## Testing

- **Finite-sample coverage.** Synthetic exchangeable scores give P(s ≤ q) ≥ 1 − α over repeats, and
  ≤ 1 − α + 1/(n + 1).
- **Groups.** A band/coordination assignment on an fcc lattice, a strained lattice and a slab; the
  merge/fallback when a group is below `n_min_*`; band edges from calibration quantiles.
- **The r₁ / z\* detector** on fcc, bcc and a small amorphous sample (first RDF minimum).
- **Schema 3 round trip**, and schema 1 and 2 files still load with the old scale.
- **Calculator matches pipeline.** `forces_std` and `forces_q` from `ACECalculator` equal the
  pipeline's predictions on the same configs, including with lean on and off.
- **`aj calibrate`.** Appending a set changes only the groups it populates. `--replace` equals a
  calibration computed from scratch on that set. Provenance is recorded.
- **Support flag.** On synthetic data with a disjoint target cluster, `support_ok` is False there and
  True in distribution, and n_eff drops accordingly.

## Acceptance

- Rerun `bench/defect_uq` (fit `ard`, bench365) with the library implementation, and port
  `scoring/conformal_cv.py` to call it.
- **In-distribution** held-out coverage 0.90 ± 0.01.
- **With `aj calibrate`** on the other 9 crack realisations (leave-one-realisation-out, v3 cells):
  - crack whole cell ≥ 0.89;
  - crack tip ≤ 10 Å ≥ 0.88;
  - edge and screw ≥ 0.90.

  These are the benchmark's numbers, within noise. The reference-free distortion feature must meet
  them. If it misses, fall back to the lattice-referenced feature behind a `--ard-groups lattice`
  option and record why.
- **Without target labels:** crack coverage ≥ 0.87, and `support_ok` False on at least 5 % of tip
  atoms, as a flag.

## Risks

- **The groups are benchmark-derived.** Distortion and coordination were chosen because they tracked
  the crack miscoverage. Another failure mode, such as chemical short-range order, would need its
  own group. The support flag is the guard against unknown shifts.
- **Small calibration sets:** 3–9 cells give an effective sample size near the number of cells. The
  per-group counts are reported for exactly this reason.
- **Storage:** the calibration scores and support descriptors add about 50 MB to `posterior.npz` at
  bench365 scale. A cap and an opt-out (`--no-support`) are provided.
