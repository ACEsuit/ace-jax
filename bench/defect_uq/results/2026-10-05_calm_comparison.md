# Force UQ vs CALM (arXiv:2609.40060): results of Steps 1–5 and 7

Plan and hypotheses H1–H5: [`docs/dev/force-uq-vs-calm.md`](../../../docs/dev/force-uq-vs-calm.md).
Branch `feat/force-uq-vs-calm`, on `main` after #57 (rotation-invariant built bases). Run outputs:
`/storage/eng/essswb/projects/ace-jax/runs/bond_scan/` (CPU, mnf148) and the Modal volume `acegp-prod-out`
(GPU timings).

**Models.**
- **Cantor:** the bench365 `ard_default` run (library defaults: aniso, transfer exponent, joint E0; acceptance
  report §7), basis `cantor_embed_d16_deg10` (L = 15 035, exported from ACEpotentials, so #57 does not apply).
- **Si GAP-18:** `aj fit --order 4 --max-degree 12 --rcut 5.5 --uq ard` on the GAP-18 Si database (2 227
  train / 248 test configurations, every 10th to test). Test errors: F 0.143 eV/Å, E 22 meV/atom. Fitted
  twice, with `--ard-support-features raw` and `normalised`; the two fits are identical except for the
  support reference.
- **Si tutorial:** the how-to's quick-start fit on the 40-configuration tutorial set (9 hold-out
  configurations, so all 8 groups merge into one).

## A bug found on the way: `forces_support` used the wrong quantile

Since 0.2.0, `ACECalculator` passed 1 − α to `support_check`, whose argument is α, so `support_q` was the
weighted α quantile (10 %) instead of the 1 − α quantile (90 %) the maths page defines, and `support_ok` failed
only when an atom's own weight exceeded 90 % of the pool. Fixed on this branch (`9c492d0`, regression test
pinned against `support_check(alpha)`). No results in earlier reports quoted the flag. Everything below uses
the fixed level.

## Step 1: bond scan (H1, H2)

`scoring/bond_scan.py`. Each prototype is rattled once (1 % of d, a fixed pattern scaled with d) and scaled
hydrostatically from 0.5 d_eq to r_cut. Without the rattle the shape is identically zero: every per-atom
vector, the force design rows included, vanishes by symmetry in a perfect fcc, bcc or diamond lattice
(checked: max v ≤ 2e-28). Values are the maximum over atoms; the support flag is the fraction of atoms flagged.

Fraction of stretched points (1.4 d_eq < d < r_cut) where the metric exceeds f × its d_eq value; compression
(d < 0.8 d_eq) is detected by every metric at every f in every case.

| model, prototype | shape v, f = 3 / 10 | `forces_q`, f = 3 / 10 | support flag (any atom) |
|---|---|---|---|
| Si tutorial, diamond | 0.69 / 0.07 | 0.08 / 0.00 | all points, including d_eq (see below) |
| Si GAP-18, diamond | 1.00 / 1.00 | 1.00 / 0.07 | every point outside 0.93–1.05 d_eq |
| Cantor, random alloy fcc | 0.90 / 0.19 | 0.33 / 0.00 | most atoms at every point outside 0.98–1.03 d_eq |

Against the random alloy's own equilibrium (Cantor):

| prototype | v at d_eq | v at 1.4 d_eq | v near r_cut | `forces_q` near r_cut | support at d_eq |
|---|---|---|---|---|---|
| CrMnFeCoNi fcc | 1 | 16.5 | 1.9 | 2.0 | not flagged |
| Ni fcc | 112 | 7.6 | 3.1 | 2.7 | flagged |
| Fe bcc | 60 | 4.5 | 0.6–1.6 | 1.2–2.0 | flagged |

- **H1 (stretching blindness of the shape) holds partially, depending on the training data.** With
  decohesion and dimers in training (GAP-18), the shape stays ≥ 48× its equilibrium value up to r_cut. With
  bulk-only training (tutorial Si, Cantor), it peaks around 1.4 d_eq and decays: near r_cut a dissociating
  alloy atom is only ~2× as uncertain as at equilibrium, and a dissociating pure-Fe atom can fall below the
  alloy's equilibrium shape. The group scales barely help: the top band's λ is 3 % above the modal band's on
  Cantor, and 40 % below it on Si GAP-18.
- **The descriptor norm does not simply go to zero.** ‖φ‖ rises before collapsing near r_cut (Si GAP-18: to
  ~1.4k at 1.6 d_eq from 1.3k at d_eq; Cantor falls from 345 to ~20 by 1.4 d_eq), so CALM's mechanism
  ("φ → 0, inside the training cloud") is not what limits the shape here.
- **H2 (raw support features miss stretching) is not supported.** On both systems the raw support reference
  flags every compressed and every stretched point, as do the normalised one and the stored one. The support
  flag catches exactly the cases the shape under-rates.
- **Chemical novelty (H3, extreme case).** Pure Ni and Fe in an alloy-trained model are 60–112× the alloy's
  shape at equilibrium and flagged by the support reference: novelty that strong is visible. Step 3 tests the
  realistic, subtler case.
- **The tutorial fit is degenerate for support.** With 9 calibration configurations the weighted quantile
  cannot reach 0.9 for a cell of near-identical atoms, so every atom is flagged even at d_eq (n_eff ≈ 5).

## Step 2: normalised support features (`--ard-support-features normalised`)

`scoring/support_rebuild.py` rebuilds a fitted posterior's support reference from its stored calibration
atoms (scores, groups, configurations), with a PCA pool of 10 000 training atoms per species. The rebuilt raw
reference reproduces the stored one (below), so raw vs normalised is compared on equal terms.

Fraction of atoms flagged (`support_ok = False`); flags on the held-out test split are false alarms:

| system, reference | test (in distribution) | big cells: all | crack | edge / screw |
|---|---|---|---|---|
| Si GAP-18, raw (stored) | 5.1 % | – | – | – |
| Si GAP-18, normalised | 3.3 % | – | – | – |
| Cantor, raw (stored) | 12.2 % | 7.6 % | 10.8 % | 3.0 / 3.7 % |
| Cantor, raw (rebuilt) | 11.8 % | 8.8 % | 11.9 % | 4.3 / 5.3 % |
| Cantor, normalised | **8.1 %** | **14.7 %** | **20.4 %** | 6.4 / 8.5 % |

On the Cantor v3 big cells (big3_mh1, 67 092 free atoms):

| reference | flagged | share of the worst 1 % of errors flagged | crack-tip atoms flagged | rms \|ΔF\| flagged / not | P(\|ΔF\| > calibrated p90) flagged / not |
|---|---|---|---|---|---|
| raw (rebuilt) | 8.8 % | 17 % | 13 % | 0.203 / 0.168 | 7.4 / 8.7 % |
| normalised | 14.7 % | **40 %** | **26 %** | 0.214 / 0.163 | 7.6 / 8.8 % |

- **On Cantor, raw support does not discriminate:** it flags more in-distribution test atoms (12 %) than big-cell
  atoms (8 %). Normalised features reverse that (8 % vs 15 %) and aim better at large errors (2.7× enrichment of
  the worst 1 % against 2.0×).
- **Flagged atoms are not under-covered.** Under either reference they exceed their calibrated 90th percentile
  no more often than unflagged atoms. The flag identifies novel, larger-error atoms that `forces_std` already
  scales for: a labelling signal, not a coverage correction.
- **Costs.** The normalised reference's weights concentrate on fewer calibration atoms (Si median n_eff 12 → 3;
  Cantor test 3.8k → 1.2k), and on Si it flags the (111) surface more (21 % vs 4 %). In-distribution false
  flags at α = 0.1 are high under every reference (Cantor 8–12 %).
- **The PCA cap binds on Cantor.** With 64 components the support space keeps 76 % (raw) / 79 % (normalised)
  of the descriptor variance, far from the 99 % target; on Si 29–30 components reach 99 %. A separate,
  untested lever (raising the cap, or CALM's random projection).
- **Default.** Stretching detection does not need normalised features (Step 1), but normalised beats raw on
  every big-cell support measure on Cantor and on overall false flags on both systems. Not changed on this
  branch: a default change is a maintainer decision, and the fit-time path has been tested only on GAP-18 Si.

## Step 4: numerical precision (H4)

`scoring/precision_audit.py`. R (the shape factor, computed in float64 at fit time) and the force rows are
perturbed at float32 rounding size; separately, R's components along the eigen-directions of S below
τ λ_max are removed.

| system (atoms) | cond(S) | float32 rows: max \|Δv/v\| | float32 R: max \|Δv/v\| | ρ under float32 | v with S-directions < 1e-12 λ_max removed: atoms moving > 1 % / ρ |
|---|---|---|---|---|---|
| Si tutorial (26) | 2.5e12 | 6e-7 | 2e-6 | 1.000 | 69 % / 0.999 |
| Si GAP-18 (6 827) | 1.06e14 | 1.5e-4 | 2.7e-4 | 1.000 | 97 % / 0.985 |
| Cantor (7 907: 30 test cells + 2 big cells) | 1.02e14 | 8e-7 | 9e-7 | 1.000 | 100 % / 0.998 |

- **Float32 storage is harmless.** The ranking is unchanged to 1e-8 and no atom moves by 1 %. CALM's finding (e)
  (rankings set below single-precision resolution) does not occur, because R is formed in float64 and only
  stored in float32: elementwise rounding is benign at any condition number.
- **The shape leans heavily on weakly determined directions.** On GAP-18, 126 of S's 350 eigenvalues lie below
  1e-12 λ_max, and removing those components of R changes v by up to 118 % (ρ 0.985). These are directions the
  data barely constrain, whose size the prior floor `ard_cond_max` sets. That is modelling, not roundoff: the
  acceptance runs found `ard_cond_max` 1e14 → 1e16 changed neither calibration nor ranking on Cantor.
  On Cantor, 5 170 of S's 15 035 eigenvalues lie below 1e-12 λ_max; removing those components moves every atom's
  v (by up to 59 %) yet barely reorders atoms (ρ 0.998).

## Step 5: the shape as an r-output linear ACE

`ACECalculator(shape_path="committee")`, `aj eval/calibrate --shape-path committee` (equal to the design-row
path to ≤ 5e-14, tests). Timed on Modal (`modal/modal_shape_eval.py`) on three v3 cells of 3 276–3 918 atoms:

| path (A100-40GB) | rank | time per cell | GPU peak | coverage | ρ vs rows |
|---|---|---|---|---|---|
| rows (default) | 3 680 | 22–23 s | 6.4 GB | 0.962 | 1 |
| committee | 3 680 | 20 s | 12.6 GB | 0.962 | 1 |
| committee | 800 | 16 s | 8.8 GB | 0.843 | 0.997 |
| committee | 200 | 15 s | 8.8 GB | 0.539 | 0.987 |
| committee | 50 | 15 s | 8.7 GB | 0.263 | 0.956 |

A100-80GB gives the same numbers.

- **The default path already fits a 40 GB GPU at 3–4k atoms.** The how-to's "A100-80GB-class GPU needed"
  predated the node-chunked rows; corrected on this branch.
- **No gain from the committee path at this size.** It is exact but not faster, and peaks higher. A
  species-wise contraction instead of the per-node coefficient gather was tried: same peak, 2× slower,
  reverted. Its O(N r) arrays matter only for much larger cells (at 100k atoms the rows alone are ~36 GB); not
  measured.
- **Truncation keeps the ranking but not the coverage.** The conformal scales are fitted at full rank, so a
  truncated shape is uniformly smaller and under-covers. A truncated rank would need its own calibration.

## Step 7: CALM-style metrics on the existing big-cell runs

`scoring/calm_metrics.py` on the `ard_default` run, all 34 v3 cells (217 860 free atoms):

| subset | Spearman ρ(σ, \|ΔF\|) | AUC, \|ΔF\| > 0.5 eV/Å | worst 1 % of errors in the top 10 % of σ |
|---|---|---|---|
| all big cells | 0.38 | 0.91 (932 positives) | 59 % |
| crack | 0.37 | 0.90 | 57 % |
| crack tip (≤ 10 Å) | 0.32 | 0.88 | 53 % |
| edge / screw | 0.30 / 0.32 | 0.98 / 1.00 (9 / 10 positives) | 52 / 63 % |

- A calibrated Gaussian with the served σ would reach ρ = 0.48 on these atoms (simulated), so the observed 0.38
  is ~80 % of what this σ could give; ρ is capped by how little σ varies between atoms (2.4× from its 5th to
  95th percentile).
- Not comparable to CALM's ρ 0.38–0.62 (grade) or 0.69–0.85 (ensembles): different models, data and error scales.

## Step 3: feature-space novelty as a Mondrian variable (H3), offline

`scoring/novelty_groups.py`: a CALM-like grade γ per species, fitted on 10 000 training atoms per species only
(label-free, so the exchangeability argument for the groups is unchanged): normalised support features in the
rebuilt reference's whitened space (D = 68), k-means with an elbow over K ∈ {1, 2, 4, 8}, nearest-cluster
Mahalanobis distance, θ = median + 3 × 1.4826 MAD. The elbow chose K = 1 for every species, so γ is a
per-species Mahalanobis distance; 6 % of training atoms have γ > 1 (CALM: 2–5 %). Coverage of the aniso score
s ≤ q at nominal 0.90, per γ bin, with q from the served groups, from γ bins alone, or from distortion group ×
γ bin (falling back to the distortion group's q below effective_n_min configurations; 13 of 24 crossed groups
qualify). Each crack file holds 6 cells × 3 rattles; ranges are over the five crack files.

| crack cells | share of atoms | served groups | novelty groups | distortion + novelty |
|---|---|---|---|---|
| γ ≤ 1 | 70 % | 0.882–0.897 | 0.872–0.891 | 0.890–0.903 |
| 1 < γ ≤ 2 | 23 % | 0.933–0.939 | 0.897–0.905 | 0.900–0.908 |
| γ > 2 | 8 % | 0.966–0.977 | 0.920–0.931 | 0.949–0.961 |
| all | | 0.900–0.912 | 0.881–0.897 | 0.898–0.908 |

Edge and screw: γ ≤ 1 for 94 % of atoms; served coverage 0.935–0.940 there and 0.967–0.980 above.

- **H3 is not borne out on these cells.** The atoms that are novel in feature space are over-covered by the
  geometric groups (0.97 for γ > 2): the served σ already grows with novelty, slightly more than needed. The
  least novel crack atoms are the ones just under nominal.
- **A novelty axis does not help.** Novelty-only groups drop crack coverage to 0.88–0.89 (borderline against
  the 0.89 target); crossed groups shift coverage from the novel atoms to the typical ones, with no net gain.
- **Not tested:** chemical shift at fixed geometry (CALM's Mo–W experiment), which needs new Cantor fits on a
  restricted composition window. The extreme case (pure elements, Step 1) is visible to both the shape and the
  support flag.

## Not done

- The synthetic chemical-shift test of Step 3 (new Cantor fits, B200).
- **Step 6** (differentiable uncertainty, HAL) and **Step 8** (conformal layer on GRACE's grade): not started.
