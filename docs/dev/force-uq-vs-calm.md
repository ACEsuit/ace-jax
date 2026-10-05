# ace-jax force UQ vs CALM (arXiv:2609.40060): notes and development plan

**Purpose.** Hand-off notes for Claude Code working on the ace-jax `--uq ard` force-uncertainty pipeline (`feat/conformal-uq` branch). They summarise a comparison with the CALM extrapolation grade of Lysogorskiy, Bochkarev and Drautz, *A latent-space extrapolation grade built into graph atomic cluster expansion foundation potentials* (arXiv:2609.40060, 30 Sep 2026). They then list concrete development steps.

**References.**
- Paper: https://arxiv.org/pdf/2609.40060 (implementation in GRACEmaker 0.6.0, https://github.com/ICAMS/grace-tensorpotential/releases/tag/0.6.0)
- Our docs: `docs/howto/force-uncertainty` and `docs/concepts/force-uncertainty-maths` (published at https://acesuit.github.io/ace-jax/)
- Current acceptance report: `bench/defect_uq/results/2026-10-03_rev2_acceptance.md`

**Results (2026-10-05).** Steps 1–5 and 7 have been run:
[`bench/defect_uq/results/2026-10-05_calm_comparison.md`](../../bench/defect_uq/results/2026-10-05_calm_comparison.md).
In short: H1 holds partially (bulk-only training); H2, H3 (on the held-out cells) and H4 are not borne out;
H5 was not tested. The shape-as-committee path is exact but gives no gain at 3–4k atoms.

**Status of claims below.** Statements about CALM come from the paper. Statements about how CALM's findings *transfer to ace-jax* are hypotheses until the tests in each step have been run. Do not write them into docs as findings before then.

---

## 1. What CALM is

1. **Feature.** Take the concatenated l=0 invariant product basis B (all layers; 7194-dim for GRACE-2L).
   - L2-normalise it: B̂ = B/(‖B‖+1e-12).
   - Apply a fixed Gaussian Johnson–Lindenstrauss projection to D_rp = 128.
   - Append log-norm channels: log‖B‖ and log‖B_m‖ per layer block. This gives z ∈ R^{131}.
2. **Clusters.** Per element, run MiniBatchKMeans followed by a global KMeans. K per element comes from an elbow criterion over {1,2,4,8,16}. Clusters with fewer than D+1 atoms are merged into their nearest neighbour.
3. **Score.** σ = Mahalanobis distance of z to its nearest centroid, using that cluster's covariance (+1e-6 I, pseudoinverse). The grade is γ = σ/θ_{e,k}.
4. **"Calibration".** θ = median(σ_train) + 3·1.4826·MAD(σ_train) per cluster. This is a threshold on the training distance distribution; it has no link to errors. 2–5% of training atoms lie above γ=1.
5. **Neighbourhood average.** Optional: γ^nbr is a cosine-cutoff-weighted average over the neighbour list.
6. **Cost and use.** Overhead is 1–6% in LAMMPS (Kokkos/TF). The gradient of σ with fixed cluster assignment costs about +30% and is used for HAL-biased MD with on-the-fly covariance updates.

**Key results.**

| Setting | Atomic γ | Neighbourhood γ | Deep ensemble ×3 |
|---|---|---|---|
| ρ(γ, \|ΔF\|), OMat24-val | 0.38 | 0.50 | 0.69–0.72 |
| ρ(γ, \|ΔF\|), SMAX | 0.54 | 0.62 | 0.80–0.85 |
| ROC-AUC @ 1 eV/Å | 0.76–0.84 | | 0.90–0.94 |

**Findings most relevant to us.**

- **(a) Stretching blindness.** The *unnormalised* projected basis (the analogue of our φ) gives the best error correlations but fails the stretching test on all six elemental prototypes (Al, Cu, Fe, Si, Ti, W). As neighbours leave the cutoff, the features go to 0, which lies inside the broad training distribution. Unnormalised norms also span many orders of magnitude (scores up to 1e18) and condition the covariances badly. L2 normalisation plus log-norm channels fixes both.
- **(b) Chemical extrapolation with fixed geometry.** In Mo–W solid solutions (0.75% lattice mismatch), γ and the error rise with W content.
- **(c) Strong γ-dependence of conditional errors.** In Table S1, p99 of |ΔF| changes by about two orders of magnitude across γ bins, while the median changes modestly.
- **(d) Centring and clustering.** Centred per-cluster scores beat uncentred per-element scores (D-optimality on the same features).
- **(e) Uncentred leverage is fragile.** Global LLPR (an uncentred Gram matrix on the 65-dim hidden layer) is nearly useless: ρ≈0.1, AUC≈0.5–0.6. Its Gram cond ≈ 4.5e13, and the authors argue the ranking is driven by directions below single-precision resolution.
- **(f) Neighbourhood averaging** improves every site-energy-feature score by +0.05–0.16 ρ. It slightly *hurts* the ensemble, whose signal is already force-localised.

## 2. How ace-jax `--uq ard` differs

| | CALM | ace-jax |
|---|---|---|
| Question answered | Is this environment novel relative to training? | How large is the force error, with stated coverage? |
| Labels used | none | hold-out errors; target-regime sets via `aj calibrate` |
| Shape | scalar Mahalanobis distance in site-descriptor space | 3×3 CR3 delete-one-cluster jackknife covariance of force predictions |
| Scale | MAD threshold | per-Mondrian-group rms λ_g and conformal q_g, plus transfer exponent β |
| Groups | element × k-means cluster (feature space) | distortion band × [z≠z\*] (geometry only) |
| OOD guard | γ>1 itself | support diagnostic: whitened PCA(φ) → logistic density ratio → weighted conformal quantile |
| Per-step MD cost | ~1–6% | not per-step; needs whole-cell force design rows (A100-80GB for 3–4k atoms) |
| Differentiable | yes (used for HAL) | not exposed |
| ρ with \|ΔF\| | 0.38–0.62 | 0.29–0.38 on large Cantor defect cells |

The correlations are **not directly comparable**: the models, datasets and error scales all differ.

**Mapping.** CALM is closest to our *support diagnostic* (covariate novelty), not to `forces_std` or `forces_q`. Our calibrated scale layer has no counterpart in CALM. The paper explicitly defers error quantiles to "independent reference data", which is what `aj calibrate` provides.

**Our strengths to preserve.** Coverage in eV/Å, an anisotropic 3×3 region, a shape robust to misspecification, the transfer exponent for approximation-dominated error, recalibration on target data, and configuration-level exchangeability.

## 3. Hypothesised weaknesses of ace-jax suggested by the paper

- **H1 (stretching).** As an atom loses neighbours, φ_α → 0, so v(x) → 0. Within a group, partially dissociated or stretched atoms may therefore rank as *confident*. Groups partly compensate, since z≠z\* and undefined d go to the top band.
- **H2 (support features).** The support diagnostic uses unnormalised φ in a whitened PCA (99% variance, ≤64 components). The PCA directions are likely dominated by large-norm (compressed) environments, and the classifier is insensitive to stretching. This is the same failure mode as (a).
- **H3 (chemical shift).** The geometric Mondrian groups cannot resolve chemical novelty such as SRO or composition outside training (our limitation 6). In the Cantor alloy this may matter.
- **H4 (precision).** cond(S) is floored at 1e14 and the posterior factor is stored as float32. Rankings from the `kappa` variant, and possibly from the sandwich through S⁻¹ in Q̃, may depend on poorly resolved directions (cf. (e)).
- **H5 (neighbourhood averaging).** Not expected to help, because our force rows already aggregate neighbour site descriptors. Test cheaply anyway.

## 4. Development steps

Do these roughly in order. Each step lists what to build, how to test it, and when it counts as done. Keep the existing defaults unchanged unless a step's acceptance criterion is met and the full acceptance benchmark still passes:
- in-distribution coverage 0.90 ± 0.01;
- crack ≥ 0.89;
- crack tip ≥ 0.88;
- edge and screw dislocations ≥ 0.90.

### Step 1: Bond-scan diagnostic (CALM Test 1)

- **Build.** A benchmark script, e.g. `bench/defect_uq/bond_scan.py`. For a fitted model and posterior:
  - take elemental ground-state prototypes (Si diamond for the tutorial fits; Cr/Mn/Fe/Co/Ni in fcc/bcc for the Cantor fit, or pure-element cells of the alloy lattice);
  - scale the nearest-neighbour distance hydrostatically from about 0.5·d_eq to r_cut (80–160 points), using a fixed supercell larger than the cutoff;
  - record the max over atoms of v(x), `forces_std`, `forces_q`, `forces_group`, `support_q` and `support_ok`.
- **Detection rule, following the paper.**
  - Compression: d < 0.8 d_eq.
  - Stretching: 1.4 d_eq < d < r_cut.
  - "Detected" means `forces_q` (or `support_ok=False`) exceeds its value at equilibrium by a stated factor. Report the factor sweep, as the paper does for its ensemble rule (SI2).
- **Done when** a plot and table are added to the acceptance report showing whether stretching is detected by (i) the shape alone, (ii) `forces_q` including the group scale, and (iii) the support flag. This confirms or refutes H1 and H2.

### Step 2: Normalised features for the support diagnostic

- **Build.** An option `--ard-support-features {raw,normalised}`. The `normalised` variant uses, per species:
  - φ̂ = φ/(‖φ‖+ε);
  - log‖φ‖;
  - log‖φ_b‖ for each body-order block b (2, 3 and 4-body; pair columns count as 2-body, consistent with the ARD groups).

  Feed these to the existing whitened-PCA + logistic pipeline. Keep the PCA cap as is, but report the explained variance.
- **Optional.** Replace PCA with a fixed Gaussian JL projection (seeded and stored in `posterior.npz`) to match CALM and avoid fitting PCA to compression-dominated variance.
- **Test.** Repeat the Step 1 bond scan, then rerun the support diagnostic on the crack and dislocation cells. Compare n_eff and the fraction flagged.
- **Done when** stretching is detected by `support_ok` and in-distribution false-flag rates are no worse than raw. Make it the default only if that holds.

### Step 3: Feature-space novelty as a Mondrian variable (addresses H3)

- **Build.** A CALM-like per-species novelty score γ_nov on the Step 2 features:
  - per-species k-means (elbow over K ∈ {1,2,4,8}, merge clusters with fewer than D+1 members);
  - nearest-centroid Mahalanobis distance;
  - median + 3·MAD threshold.

  Everything is computed from T (training) only and frozen at fit time, like the band edges. Add `--ard-groups {distortion,novelty,distortion+novelty}`, where `novelty` bins γ_nov at, e.g., {≤1, 1–2, >2} and the combined option crosses these with the existing groups. The existing n_min merging rules apply; check that the merge order generalises to the extra axis.
- **Test.**
  - Conditional coverage per γ_nov bin on the Cantor test set and on the large cells.
  - A synthetic chemical-shift test modelled on CALM's Mo–W experiment: train on a restricted composition window (or random-solution cells only), then evaluate on cells outside the window or with imposed SRO.
  - Report whether per-group coverage stays near nominal where the geometric-only groups under-cover.
- **Done when** an ablation table is in the acceptance report.
- **Note.** The group edges must remain label-free and frozen at fit time, to preserve the exchangeability argument in section 9 of the maths page.

### Step 4: Numerical-precision audit (H4)

- **Test.** For the Cantor posterior, compute `forces_std` rankings with R and B in float64 versus float32. Report the Spearman correlation between the two rankings and the maximum relative change. Also report the singular-value spectrum of Q̃ and how much of v(x) comes from directions near the cond floor.
- **If** the rankings move materially, store R in float64, or truncate R at a tolerance relative to the float32 epsilon. Document the result either way.
- **Done when** there is a short section in the acceptance report.

### Step 5: Cheap shape evaluation as an r-output linear ACE (MD-step cost)

- **Idea.** v(x) = Σ_α ‖Rᵀ u_α(x)‖² with u_α = D⁻¹ φ_α. Since φ_α = −∂(Σ_j φ(x_j))/∂r_{iα}, the vector Rᵀu_α is the force on atom i of an **r-output linear ACE model** with coefficient matrix C = D⁻¹R ∈ R^{L×r}. So V(x) can be computed by evaluating a multi-output linear ACE (a "committee" of r coefficient vectors sharing one basis evaluation), with no design rows stored.
- **Build.**
  - A JAX path that evaluates site energies for an L×r coefficient matrix and the per-atom force Jacobians for all r outputs. This is the adjoint through the A-basis with r right-hand sides, implemented with vmap or batched VJPs.
  - SVD truncation of R to rank r with the existing τ option. Report the coverage and ranking loss against r.
- **Test.**
  - Exact agreement with the current shape at full rank.
  - Wall-clock time and memory against the current design-row path on the 3–4k-atom cells.
  - Target: runs on 40 GB GPUs, at a few times a force call for r ≈ 50–200.
- **Done when** an `ACECalculator` option serves the shape via this path; ideally it becomes the default if it agrees with the current path.
- **Follow-up.** Export to LAMMPS as a multi-output PACE/.yace model. Check what ML-PACE's `pace/extrapolation` infrastructure already supports.

### Step 6: Differentiable calibrated uncertainty and HAL

- **Build.** Given Step 5, expose ∂ forces_std/∂r (or ∂v/∂r, with the group held fixed, analogous to CALM fixing the cluster assignment) and a biased-MD driver following van der Oord et al. 2023 and CALM eq. 13. The **collection rule** is `forces_q` > a user tolerance in eV/Å, rather than an ad hoc dimensionless threshold.
- **Test.** Reproduce CALM's aluminium protocol at a fixed labelling budget (50 seed frames + 200 collected; unbiased MD vs HAL; evaluate on 300–1100 K MD, vacancy, (111) surface and liquid). Use a foundation model as the label source, as they do.
- **Done when** there is a results table comparable to CALM Fig. 7.

### Step 7: Head-to-head metrics on shared benchmarks

- **Build.** Add CALM's metrics to our evaluation reports:
  - per-atom Spearman ρ;
  - ROC-AUC for |ΔF| > 1 eV/Å, swept over thresholds 0.05–5 eV/Å;
  - contamination P(|ΔF| > 0.5 eV/Å | confident) against the kept fraction;
  - the bond-scan pass/fail from Step 1.
- **Optional.** A reduced-chemistry slice of OMat24/SMAX that linear ACE can fit, to report numbers on the same evaluation sets as the paper.
- **Optional.** Test a cosine-weighted neighbourhood average of `forces_std` (H5) for completeness.

### Step 8 (separate project): conformal layer on top of CALM

- **Idea.** Our scale stage is agnostic to the source of the score. Apply our grouped, configuration-weighted conformal calibration and transfer machinery to GRACE's γ, for example with score s = |ΔF| / g(γ) and Mondrian bins on γ. This would turn an uncalibrated grade into eV/Å error bars with coverage on foundation models.
- **Scope.** This needs GRACEmaker 0.6.0 and labelled target data; scope it as a short standalone study rather than part of the ace-jax core.

## 5. Docs to update once results exist

- Maths page, section 11 (support diagnostic): feature normalisation and its rationale.
- Maths page, section 9 (groups): the novelty axis, if adopted.
- Maths page, section 13 (limitations): stretching behaviour, from the Step 1 results.
- How-to page, *Validation*: bond-scan results and the precision audit.
- How-to page, *Cost and memory*: the r-output evaluation path, once Step 5 lands.
- Add CALM (arXiv:2609.40060) to the references where its findings motivated a change.
