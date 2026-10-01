# Calibrated per-atom force uncertainty: jackknife shape and Mondrian conformal scale

Status: design, revision 2 (1 October 2026), for review · Base: `main` · Branch: `feat/conformal-uq`

**Mathematical reference (binding):** `docs/specs/tex/force-uq-math-pipeline.tex` (revision 2;
compile with `latexmk -pdf`). Section numbers below (§) refer to that document. This spec fixes the
interfaces, defaults and validation; where the two disagree, the maths document wins.

**Evidence:** `bench/defect_uq` (#21): `scoring/conformal_shift.py`, `scoring/conformal_cv.py` and
`acegp-data/results/2026-09-28/bench365_ard/ACCEPTANCE.md`.

## Goal

The per-atom force uncertainty served by `--uq ard` should have a **stated, verifiable coverage**.
It should stay conservative where calibration data say nothing, and be usable on large cells for
fracture and plasticity. The served quantity is a **shape** times a **per-group scale** (§ Summary):

- **Shape: the per-atom 3×3 block V(x) of an exact, centred, delete-one-cluster jackknife covariance**
  (CR3), with v(x) = tr V(x). It replaces the uncentred, uncorrected configuration-clustered sandwich
  m²(x) of #18.
- **Scale, two of them, both per Mondrian group g and both configuration-weighted:**
  - an rms factor λ_g^rms, which defines `forces_std` and `forces_cov`;
  - a conformal quantile q_g, which defines `forces_q`.
- **Calibration:** errors and shape both come from the hold-out posterior P_fit (§7). The split is
  stratified by group (§4). `aj calibrate` replaces a group's pool with target-regime data by default
  (§10).
- **A covariate-shift support flag** (§11), a diagnostic that never scales σ.

## What changes and what does not

| | change |
|---|---|
| mean c̄, E/V predictions and variances, the evidence fit (§3) | unchanged |
| `--ard-variance kappa` (epistemic shape A⁻¹) | unchanged as a shape. It gets the new scales and the anisotropic block V_κ. |
| `--ard-variance sandwich` shape | **changed**: PRESS-corrected, centred, spatially sub-clustered jackknife (§5–6) |
| scalar λ (#18) | replaced by per-group λ_g^rms and q_g |
| hold-out | stratified by configuration stratum; scores from P_fit only (`m²_−own` removed) |
| `posterior.npz` | schema 3; schema 1 and 2 files still load and serve their old scalar scale |

Served σ values change from #18. There is no flag that reproduces the #18 sandwich σ for *new* fits,
other than the bench-only ablation switches (Decision 7).

## Components and interfaces

### 1. Sandwich clusters — `fit/clusters.py` (new), §5

- `assign_clusters(batch_or_config, ell) -> cluster ids for every E, F and V row of each config`.
- **A small configuration** is a single cluster containing all its rows. A configuration counts as
  small unless it holds at least two blocks of side ℓ along some lattice direction.
- **A large configuration** has:
  - one cluster per spatial block (a grid with n_a = max(1, ⌊w_a/ℓ⌋) divisions, w_a the width
    perpendicular to the other two lattice vectors, or the bounding-box extent along a non-periodic
    direction, with atoms assigned by fractional coordinate, respecting periodicity), holding the
    force rows of that block's atoms;
  - one cluster holding the energy row and the six virial rows.
- `ell = ard_cluster_size × r_cut`: default **3**; `inf` means whole configurations.

### 2. Jackknife shape — `fit/jackknife.py` (new), §6

- **`press_scores(post, prob, ds, clusters) -> G̃` (L, K).** For cluster k:
  g̃_k = Ψ_kᵀ (I − H_kk)⁻¹ ρ_k, where H_kk = Ψ_k A⁻¹ Ψ_kᵀ and ρ_k are the residuals at the posterior
  mean.
  - **Exact** when n_k ≤ L: solve with W_k = L⁻¹D⁻¹Ψ_kᵀ.
  - **For n_k > L**, use the push-through form A⁻¹g̃_k = A_(−k)⁻¹ Ψ_kᵀ ρ_k, with a rank-n_k Cholesky
    downdate.
  - Clusters are batched with `vmap` over padded n_k buckets.
- **`ard_press = "exact" | "block"`**, default `"exact"`. `"block"` is the HC3-like per-row-block
  correction (energy row, 3×3 force block per atom, 6×6 virial block). It is never the default.
- **`shape_factor(post, G̃, tau=1.0) -> R`** with RRᵀ = Q̃Q̃ᵀ, where Q̃ = S⁻¹D⁻¹[g̃_k − ḡ] (centred).
  - R = Q̃ if K ≤ L; otherwise U_rΣ_r from the thin SVD of Q̃.
  - The optional truncation keeps a fraction τ of Σσ² (default τ = 1, exact).
- **`atom_shape(post, Frows) -> V` (N, 3, 3).** V_αβ = (Rᵀu_α)·(Rᵀu_β), with u_α = D⁻¹φ_αᵀ.
  For the κ variant, V_αβ = φ_α A⁻¹ φ_βᵀ.
- **`leverage_report(...)`:** λ_max(H_kk) per cluster, for validation item 2.

### 3. Groups, split and scales — `fit/conformal.py` (new), §4, §8, §9

- **Features:** r₁ (the first minimum of the training RDF; fall back to 1.25 × the first peak if none),
  z* (the modal training coordination), and per atom z and d. They are computed from a Dataset
  batch's own neighbour list.
- **Groups:** g = band(d) × [z = z*], so G = 8. The band edges are the p50 / p90 / p99 quantiles of
  d over **all atoms of T**; NaN d goes in the top band. The edges are frozen at fit time and stored.
- **`stratified_split(cfg_strata, f, seed) -> (fit_idx, val_idx)`.**
  - Each configuration's stratum is its most extreme populated group: highest band, ties to z ≠ z*.
  - Within each stratum, a fraction f goes to T_val; every stratum with ≥ 2 configurations has at
    least one on each side.
  - Strata with fewer than 2 configurations are merged into the next less extreme stratum.
- **Scores** (§7): isotropic s = |e| / √(v_fit/3); anisotropic s = √(eᵀ(V_fit + ε_i I)⁻¹e), with
  ε_i = ε·tr V_fit/3 and ε = 1e-3.
- **Per group, configuration-weighted** (weight 1/n_{c,g} per atom, so each configuration counts 1
  in each group):
  - (λ_g^rms)² = (1/(3 n_cfg,g)) Σ_c (1/n_{c,g}) Σ_i s_i²;
  - q_g = inf{t : F̂_g(t) ≥ 1 − α}, with F̂_g the pooled CDF including a +∞ test point (Dunn,
    Wasserman & Ramdas).
  - **All-groups fallback:** when no neighbour qualifies, the pool is all atoms with the same weights
    1/n_{c,g} (a configuration spanning k groups has total weight k), and the +∞ test point weighs
    Σw/n_cfg (one configuration's mean total weight): F̂(t) = [Σ_i w_i 1{s_i ≤ t} + (Σw/n_cfg)·1{∞ ≤ t}] /
    (Σw (1 + 1/n_cfg)), so q is finite iff n_cfg ≥ ⌈(1−α)/α⌉ in every pool.
- **Merging:** groups with n_cfg,g < n_min (default **20**) merge into the neighbouring band with the
  same [z = z*] flag, then across the flag. Merges are recorded.
- **The per-group table** (stored): n_cfg,g from T_val and from U, total atoms, λ_g^rms, q_g,
  r_g = q_g / (λ_g^rms · χ₃⁻¹(1−α)), merges, and pool composition.

### 4. The ARD stage — `fit/ard.py` (`run_ard_stage`), §4, §7

1. **Features and stratified split.** Compute r₁, z*, the band edges (all of T) and the strata;
   split with `stratified_split`. This replaces the uniform permutation split.
2. **Hold-out posterior.** Fit h on T_fit to get P_fit. Build P_fit's own clusters, PRESS scores and
   centred Q̃_fit over T_fit only (not stored).
3. **Scores.** For T_val: e_i = F_i − F̂_i(c̄_fit), the shape V_fit(x_i), and the scores in the
   chosen `force_shape` mode.
4. **Served posterior.** Refit on T to get P. Build its clusters, PRESS scores, centred Q̃ and stored
   factor R.
5. **Scales.** Per-group λ_g^rms and q_g from the T_val scores. κ is still reported, and so is the
   old scalar λ, computed from the new scores for comparison only.
6. **Report:** `ard.json` gains `shape` (mode, ℓ, K, the rank of R, the leverage summary),
   `groups` (the table), `split` (strata counts), and `transfer` (f, N_fit).

### 5. `ARDPosterior` schema 3 — `fit/ard.py`

New trailing fields, all None for schema 1 and 2 files:
- `R` (L, r), float32 on disk;
- `force_shape` ("iso" | "aniso");
- `eps`;
- `group_consts` (r₁, z*, edges);
- `group_table` (the per-group arrays above, plus α);
- `cal` (per calibration atom: score f32, group i8, configuration id i64, pool source i8 — 0 for
  T_val, 1.. for `calibrate` sets — kept so `calibrate` can recompute);
- `support` (§6 below).

Methods:
- `atom_shape(Frows) -> V`;
- `forces_std(Frows, groups)`, `forces_cov(Frows, groups)`, `forces_q(Frows, groups)`, and
  `forces_q_mahal(groups)` (anisotropic mode);
- `groups_of(batch)` (features → groups with the stored constants).

`predict_ard`'s `F_var` per component is diag(`forces_cov`), so Σ_α F_var = `forces_std`².

### 6. Support diagnostic — `fit/support.py` (new), §11

- **Features:** for each species, a whitened PCA of that species' site-descriptor block on the
  training atoms, keeping 99 % variance, capped at 64 components.
- **Density ratio:** an L2 logistic classifier (target 1, calibration 0, class-balanced). Its L2
  strength is chosen by 5-fold cross-validation grouped by configuration.
- **Quantile:** calibration masses w(x_i)/n_c(i); test mass w(x); the weighted quantile has a point
  mass at +∞.
- **Outputs:** `support_ok`, `support_q`, and per-species n_eff over the calibration masses.
- **Stored:** the PCA per species, plus the calibration pool's PCA features, scores and
  configuration ids. The pool is capped at `ard_support_max_atoms` (default 50 000), with whole
  configurations sampled.

### 7. Serving — `calc/point.py`, `cli.py`

- **`ACECalculator(model, posterior=…)`** properties, all on request:
  - `forces_std` (N,);
  - `forces_cov` (N, 3, 3);
  - `forces_q` (N,): the radius in isotropic mode, the largest semi-axis q_g√λ_max(V+εI) in
    anisotropic mode;
  - `forces_q_mahal` (N,): q_g, anisotropic mode only;
  - `forces_group` (N,);
  - `forces_support` (dict).

  The shape uses the **full** `calc.model`, never `eval_model`, as now.
- **`aj fit --uq ard`** flags:
  - `--force-shape iso|aniso` (default iso);
  - `--ard-coverage 0.9`;
  - `--ard-groups distortion|none` (none means G = 1);
  - `--ard-cluster-size 3` (× r_cut; `inf` = whole configurations);
  - `--ard-press exact|block`;
  - `--ard-n-min 20`;
  - `--ard-val-frac 0.2`;
  - `--no-ard-support`.
- **`aj calibrate --model M --posterior P --data U.xyz [--energy-key …] [--force-key …]
  [--append | --replace] [--coverage c] --out P2`:**
  - scores come from the served model and served V (§10);
  - by default, a group's pool is U alone if U has ≥ n_min configurations in it, otherwise
    T_val ∪ U;
  - `--append` forces T_val ∪ U everywhere; `--replace` forces U everywhere, merging as needed;
  - it prints the per-group table (composition, λ_g^rms, q_g, r_g, merges) and the provenance
    (sha256, n_cfg, n_atoms);
  - it never modifies the input file.
- **`aj eval --posterior … --per-atom out.xyz [--support]`** writes `forces_std`, `forces_q`,
  `forces_group` and (with `--support`) `support_ok` / `support_q`. With `--force-shape aniso` it also
  writes `forces_cov` (as 9 columns) and `forces_q_mahal`.

## Decisions (review these)

1. **Default `--force-shape iso`** until the ablation (§13, item 5) shows the anisotropic mode helps.
2. **ℓ = 3 r_cut by default**, to be revised to the smallest ℓ on the plateau of validation item 3.
3. **n_min = 20 configurations per group** (not an atom threshold).
4. **Band edges from all of T**, used for both stratification and storage, and frozen before
   `calibrate`.
5. **`calibrate` per-group replace by default**, so a target-regime group is calibrated against
   target-regime data only.
6. **`forces_group` is served per atom; r_g is not.** r_g lives in the stored per-group table, joined
   through the group id.
7. **The ablation baselines are bench-only switches, not user flags.** They are needed for item 5: the
   30 September sandwich (uncentred, no PRESS, m²_−own), and "+A" (P_fit shape, old sandwich).
   Proposed: private `FitConfig` fields `_shape_variant ∈ {"press", "legacy"}` and
   `_score_source ∈ {"fit", "mixed"}`, undocumented in the CLI and used only by `bench/defect_uq`.
8. **The exact finite-sample conformal constructions** (Dunn et al. double or subsampled) are not in
   this PR. The pooled CDF is approximate, with error controlled by n_cfg,g.

## Validation (this PR) — §13

- **Unit: exact deletion.** On a small system with h fixed, refit without cluster k and check
  c̄ − c̄_(−k) = A⁻¹g̃_k to machine precision. Do this for a whole-configuration cluster, a spatial
  block and an E/V cluster, plus the push-through (n_k > L) path.
- **Unit:** centring (Σ_k (g̃_k − ḡ) = 0); R reproduces Q̃Q̃ᵀ (exact, and truncated to τ);
  V = λ²-scaled `forces_cov`, with trace equal to `forces_std`²; the configuration-weighted λ and
  pooled-CDF q against brute force; merging; the stratified-split guarantees; schema-3 round trip;
  schema-2 files still serve the scalar scale; lean/full parity of all served quantities; and the
  calibrate per-group-replace, append and replace semantics.
- **Benchmark** (Modal; bench365, with the 10-realisation crack set):
  1. the leverage report (λ_max(H_kk), and whether the high-leverage clusters are crack-like);
  2. the block-size sweep with crack cells in training, ℓ/r_cut ∈ {2, 3, 4, 6, ∞}, reporting tip
     coverage with tip data in training against calibration only;
  3. transfer at f ∈ {0.1, 0.2, 0.3}, extrapolating log λ_g in log N_fit if it trends;
  4. the ablation (30 Sep baseline, +A, +A+B, +A+B+blocks, +anisotropic), each measured by Spearman
     ρ, global λ_rms, per-group coverage weighted by configuration and by atom (configuration
     bootstrap 90 % CI, B = 1000), crack-tip coverage, and mean region volume at equal coverage.
- **Acceptance:**
  - in-distribution held-out coverage 0.90 ± 0.01;
  - with `calibrate` leave-one-realisation-out: crack whole cell ≥ 0.89, tip ≥ 0.88, edge/screw ≥ 0.90.

  These are the 30 September numbers; the revision must not be worse.

## Cost

- **PRESS needs one pass over the training rows per posterior**, with a triangular solve per cluster.
  That is about L²·Σn_k flops, the same as the leverage pass of the sandwich spike (minutes on a B200
  at bench365 size). It runs twice: once for P_fit and once for P.
- **R is stored** as L × min(K, L) float32, i.e. 0.22 GB at L = 15k with K = 3.7k.
- **The support reference** is capped at 50k atoms × ≤ 64 PCA dimensions.

## Non-goals

- Energy and virial calibration.
- Exact finite-sample hierarchical conformal (Decision 8).
- The served σ derived from the support weights.
- A larger-r_cut test of the locality-noise explanation of any remaining tip gap (§14).
- Buffered block-wise MACE labelling.

## Risks

- **Transfer is empirical** (§7). If the error is bias-dominated, the hold-out scale is low by up to
  (1−f)^−½. Validation item 3 measures this.
- **Small n_cfg** in the extreme groups merges them away from the tip, so the tip quantile is shared
  with less extreme atoms. The per-group table makes this visible.
- **The shape is a parameter-variance proxy** for approximation error (§14); ρ measures how well it
  ranks.
- **PRESS cost on large training cells** if sub-clustering is disabled (`--ard-cluster-size inf`): the
  push-through downdates are O(L³) per cluster.
