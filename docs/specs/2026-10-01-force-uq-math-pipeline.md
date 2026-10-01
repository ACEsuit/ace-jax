# The per-atom force-uncertainty pipeline: full mathematical description (current + proposed)

Status: for review · Companion to `2026-09-30-conformal-force-sigma-design.md` (interfaces) and
`2026-09-28-tempered-ard-uq-design.md` (ARD and sandwich, implemented in #18).
Notation: per atom i, per Cartesian component α ∈ {x, y, z}; training configurations c ∈ T.

**Short answer to "does conformal replace the sandwich?"** No. The pipeline has two separate
factors: a **shape** v(x) and a **scale** λ.

- The **shape** v(x) is the unscaled per-atom variance, which is the sandwich by default. It decides
  how σ varies from atom to atom, and so carries all the ranking. It is **unchanged** by this proposal.
- The **scale** is currently one scalar λ (or κ), fitted so that rms-z = 1.
- The proposal replaces only the scale. It becomes one conformal factor λ_g per group g of a geometric
  feature, taken from score *quantiles*. The score itself is normalised by the sandwich variance.

Within a group the ordering of atoms by σ is exactly the sandwich's. Across groups it changes only by
the ratios of the λ_g.

---

## 1. Model and data

**Linear model.** For each atom i with environment x_i, the site descriptor φ(x_i) ∈ ℝ^L is the ACE
product basis in species-major blocks, plus pair blocks. The model is

  E(R) = Σ_i φ(x_i) · c + Σ_i E0_{Z_i}.

The training observations of a configuration are the energy, the forces and the virial. Each is
linear in c:

  E-row  Φ_E = Σ_i φ(x_i),   F-rows  Φ_{F,iα} = −∂Φ_E/∂r_{iα},   V-rows  Φ_{V,αβ} = −Σ_i r_{iα} ∂Φ_E/∂r_{iβ}.

**Weighting.** Each row has a structural weight w (per-atom energy normalisation and per-type
weights) and a noise scale σ_q for its quantity q ∈ {E, F, V}. The whitened rows and targets are

  ψ = Φ w / σ_q,   ỹ = y w / σ_q,

with y the target minus the E0 baseline.

## 2. Prior and posterior

**Prior.** c ~ N(0, Λ⁻¹), with Λ = diag(Γ_j² · exp(a_{k(j)})).
- Γ_j is the smoothness prior of column j.
- k(j) ∈ {2, 3, 4} is the column's body order. Pair columns and correlation order 1 both count as
  2-body.
- a_k is the ARD log-precision multiplier of body order k.

**Posterior.**

  A = Σ_rows ψψᵀ + Λ,   c̄ = A⁻¹ Σ_rows ψ ỹ,   Cov(c) = A⁻¹.

**Numerics.** Work in the prior-scaled system S = D⁻¹AD⁻¹ with D = diag Γ, factorised as S = LLᵀ.
Each a_k is floored, a_k ≥ a_floor, so that cond(S) ≤ 10¹⁴.

## 3. Hyperparameters: type-II maximum likelihood

Let h = (log σ_E, log σ_F, log σ_V, a₂, a₃, a₄). Using the per-quantity statistics G_q = Σ ψψᵀ, b_q,
yᵀy_q and n_q (up to constants):

  log p(D|h) = −½ Σ_q yᵀy_q/σ_q² + ½ bᵀA⁻¹b − ½ log|A| + ½ log|Λ| − Σ_q n_q log σ_q.

- **Joint mode** maximises this over all six components of h, with L-BFGS-B on the
  gradient-normalised objective.
- **Sequential mode** fixes σ_q at the linear MAP and fits only the a_k.

*(Unchanged.)*

## 4. The hold-out protocol (unchanged; feeds both scalings)

1. Split the training configurations, by configuration, into T_fit (1 − f) and T_val (f = 0.2).
2. Fit h on T_fit, form the posterior P_fit, and record each held-out atom's error
   e_i = F_i − F̂_i(c̄_fit) for i ∈ T_val. **These errors are honest:** P_fit never saw T_val.
3. Refit on all of T, starting from h_fit. This gives the **served** posterior P with mean c̄ and
   factor L.

## 5. The shape: unscaled per-atom variance v(x) (unchanged)

For a target atom with force rows φ_α(x), α = x, y, z (raw, unweighted), there are two variants
(`--ard-variance`).

**Epistemic (`kappa`):**

  s²(x) = Σ_α φ_α A⁻¹ φ_αᵀ = Σ_α ‖L⁻¹ D⁻¹ φ_αᵀ‖².

**Configuration-clustered sandwich (`sandwich`, default).** The residuals at the served mean are
ρ_r = ỹ_r − ψ_r · c̄. Each training configuration's score sums its rows:

  g_c = Σ_{r∈c} ψ_r ρ_r     (all of c's E, F and V rows).

The meat is M = Σ_c g_c g_cᵀ, of rank ≤ |T|, and

  m²(x) = Σ_α φ_α A⁻¹ M A⁻¹ φ_αᵀ = Σ_α ‖Qᵀ D⁻¹ φ_αᵀ‖²,   Q = S⁻¹ D⁻¹ [g_1 … g_|T|]   (L × |T|).

This is the Huber–White misspecification-robust covariance of ĉ, also Müller (2013)'s sandwich
posterior. κ²A⁻¹ is its homoscedastic special case (M ∝ A). The two give the following picture:
- φA⁻¹g_c is the one-step change in the prediction at x from deleting configuration c, so m²(x) is
  the infinitesimal-jackknife variance of the prediction over training configurations;
- this is what ranks local errors (ρ 0.30–0.37, against 0.19–0.26 for s²).

**Leave-own-cluster-out (calibration only).** For a held-out atom i in configuration c(i), take

  m²_{−own}(x_i) = Σ_α Σ_{c ≠ c(i)} (φ_α A⁻¹ g_c)²,

because a genuinely new configuration has no own term. **Served** variances use all c.

Write v(x) for the chosen variant (m² or s²), and v_i^cal for its calibration version (m²_{−own} or
s² at the held-out atoms).

## 6. The scale: current (`gaussian`) and proposed (`conformal`)

**Normalised score** for a labelled calibration atom:

  s_i = |e_i| / √(v_i^cal / 3).

If √(v/3) were an exact isotropic Gaussian per-component scale, s would follow χ₃ (the chi
distribution with 3 dof).

**6a. Current: Gaussian rms matching (one scalar).**

  λ² = (1/3) · mean_{i∈T_val} s_i²     (Gaussian NLL-optimal; κ is the same formula with s²).

The served values are σ(x) = λ √v(x), interpreted as ΔF ~ N(0, σ²/3 · I₃). The only claim is
rms-z = 1 on T_val. Any coverage statement depends on the Gaussian assumption.

**6b. Proposed: Mondrian split conformal (one factor per group).**
- **Groups** g(x) ∈ {1..G} come from geometry alone (§7).
- **For target coverage 1 − α**, and each g with n_g calibration atoms:

    q_g = the ⌈(n_g + 1)(1 − α)⌉ / n_g empirical quantile of { s_i : g(x_i) = g }.

- **Served values:**

    `forces_q(x)` = q_{g(x)} · √(v(x)/3)                           (guaranteed radius),
    `forces_std(x)` = λ_{g(x)} · √v(x),   λ_g := q_g / χ₃⁻¹(1 − α)   (the σ whose Gaussian (1−α) ball is that radius).

- **Guarantee (split conformal, per group).** If a target atom is exchangeable with group g's
  calibration atoms, then

    1 − α ≤ P(|ΔF| ≤ `forces_q`) ≤ 1 − α + 1/(n_g + 1).

  No Gaussianity is assumed; the distribution of s within the group can be anything. Atoms within a
  configuration are correlated, so the exact statement holds for configurations; the atom-level
  version is the standard approximation. n_cfg per group is reported, and groups below n_min are
  merged.
- **Relation to 6a.** 6a is G = 1 with the rms in place of the (1 − α) quantile. On in-distribution
  data the two scales agree to about 1 % (bench365: q = 2.48 against λ·χ₃⁻¹(0.9) = 2.50). Where they
  differ, 6b is the one with the stated coverage.

**What is invariant.**
- The mean c̄ and the E/V predictions.
- v(x).
- The ordering of atoms by σ *within a group*, since the transform there is one positive factor.

What changes is the relative σ *between* groups, by λ_g / λ_{g'}. That is how the tip groups get
more σ without inflating the bulk.

## 7. The group feature (Mondrian)

**Fit-time constants:**
- r₁, the first minimum of the training radial distribution function;
- z*, the modal training coordination.

**Per atom:**
- coordination z_i = #{j : r_ij < r₁};
- distortion d_i = std / mean of { r_ij : r_ij < r₁ } (dimensionless first-shell spread; NaN if z_i < 2).

**Band edges** are calibration-pool quantiles of d: p50, p90, p99 by default, giving 4 bands. The
group is g = band(d) × [z = z*], so G = 8.

**Why these.** On the big cells, miscoverage tracked distortion (σ grows ×1.7 from low to high
distortion while the error grows ×2.0) and coordination: bulk-like atoms are calibrated, surface-like
ones conservative, and strained-shell ones overconfident. In distribution, coverage is flat across
these groups, so the groups only cost samples where nothing is wrong.

## 8. Calibration sets

- **Default (in the fit):** T_val, with honest errors (§4) and v^cal (§5).
- **`aj calibrate` (target regime).** Labelled configurations U, e.g. a few cells from the production
  regime labelled with MACE or DFT.
  - The errors are those of the *served* model: e_u = F_u − F̂_u(c̄). U is disjoint from training,
    so these are honest too.
  - v_u is the served v, with no own term, because U ∉ T.
  - The scores s_u are appended to T_val's per group (or replace them with `--replace`), and the q_g
    are recomputed.
  - **Exchangeability is now with "T_val ∪ U" in each group.** A target atom like U's is covered
    because U is in the pool.
- **Bench365 evidence** (leave-one-realisation-out over 10 crack realisations): whole crack cell
  0.899, crack tip 0.885 (folds 0.864–0.900), dislocations 0.92–0.93.

## 9. The support diagnostic (covariate-shift weighted conformal; not served)

Our labels are a deterministic function of the structure, so P(y|x) is fixed: train → target is a
pure covariate shift. For a target structure, with only unlabelled atoms:

1. **Density ratio.** For each species, an L2-regularised logistic classifier on whitened 2-body
   descriptors (target = 1, calibration = 0, class-balanced) gives w(x) ∝ P(t|x) / P(c|x).
2. **Weighted quantile** (Tibshirani, Barber, Candès & Ramdas 2019). For each target atom x:

     q_w(x) = the (1 − α) quantile of  Σ_{i∈cal} p_i δ(s_i) + p_x δ(+∞),
     p_i = w(x_i) / (Σ_j w(x_j) + w(x)),   p_x = w(x) / (Σ_j w(x_j) + w(x)).

3. **Report** per atom `support_q` = q_w(x) and `support_ok` = (q_w < ∞), and per species
   `n_eff = (Σ w)² / Σ w²`.

`support_ok` = False means that, under the estimated shift, the calibration data cannot certify
coverage for that atom. These are the atoms to label next (active learning). On bench365, with no
target labels, 10.7 % of crack-tip atoms are flagged, against 0.4–0.5 % for the dislocations. The
diagnostic is not used to scale σ: with target labels in the pool it adds nothing over 6b, and
without them its finite quantiles rest on n_eff of order 10–10³.

## 10. End-to-end summary

```
fit:       data -> whitened rows psi, targets y~ -> type-II ML h (sec 3, on T_fit; refit on T)
           -> served posterior (c-bar, L)                                   [mean / epistemic]
           -> config scores g_c at c-bar -> Q = S^-1 D^-1 G                 [sandwich shape]
           -> T_val scores s_i = |e_i| / sqrt(v_cal_i/3), groups g(x_i)     [calibration]
           -> per-group q_g (conformal)  or  scalar lam (gaussian)          [scale]
calibrate: + labelled target set U -> append s_u per group -> recompute q_g
serve:     v(x) (shape) x lam_{g(x)} = q_{g(x)} / chi3^{-1}(1-alpha)  ->  forces_std, forces_q
diagnose:  unlabelled target -> w(x) -> weighted quantile -> support_ok, support_q, n_eff
```

## 11. Assumptions and known limitations

1. **Exchangeability within a group, at configuration level.** The per-atom guarantee is approximate.
   Groups and n_cfg are reported.
2. **The groups are chosen from benchmark evidence.** A shift the groups don't resolve, such as
   chemical short-range order or a new phase, is covered only marginally. §9 is the guard: it flags
   unsupported atoms without labels.
3. **There is a residual tip gap.** Even with labelled crack cells, coverage at the tip itself is
   about 1.5 points under target. No feature tried resolves the tip completely: distortion,
   coordination, 2-body density ratio, 2-body kNN.
4. **Forces only.** E and V variances remain the unscaled posterior variances (unchanged).
5. **No aleatoric term.** Labels are deterministic, so all error is approximation error. The scale
   (λ or q_g) absorbs it; the shape v(x) decides where it is larger.
