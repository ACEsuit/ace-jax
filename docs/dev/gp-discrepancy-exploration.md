# GP discrepancy — Phase 0 exploration report

Status: Phase 0 of `docs/dev/specs/2026-10-05-gp-discrepancy-design.md`, for review. Branch
`explore/gp-discrepancy`. No `src/` changes. Scripts: `bench/defect_uq/scoring/d{1,2,3}_*.py`; D4 arms in
`bench/defect_uq/modal/ard_arms.py`. Tables: `bench/defect_uq/results/2026-10-05_gp_discrepancy/`; caches and logs:
`/storage/eng/essswb/projects/ace-jax/runs/gp_discrepancy/d{1,2,3}/`.

**Caveat carried from #60 / `feat/gp-lml-cost`:** the `rms_z`/coverage columns of `metrics.csv` for the BLR
and GP rungs use the latent σ without the observation noise (force rms_z ~1e4). Nothing below reads UQ
calibration from those columns. Every coverage number here is the served `--uq ard` region (`forces_q`, aniso
Mahalanobis), which that bug does not touch.

## Summary

- **Code map (0.1).** The spec's Background is correct, with these corrections and additions:
  1. The default MAP maximises LML + log_prior, but the served `--uq ard` posterior re-maximises the *pure*
     evidence over (σ_E, σ_F, σ_V, a_k) in joint mode. The noise model of Phase 1 must therefore also be
     built into `ARDEvidence`.
  2. `solve._prior_block`'s R0 can replace the diagonal D in the PRESS/shape algebra if the GP block gets at
     most one ARD scale and θ is fixed.
  3. `objective="loo"` is leave-one-configuration-out with exact block leverages. Combined with
     `lml="host-cache"` it is silently ignored (a bug).
  4. `feat/shared-noise` is PR #67.
  5. MACE-MH-1's receptive field is 2 × 6.0 = 12 Å.
  6. The `predict.py` docstrings still describe SoR-only F/V variances.
- **D1 / G1:**
  - The force-UQ score shows no robust within-group dependence on the environment at 8–12 Å beyond what the
    r_cut feature explains. Pooled over groups, |split| ≤ 0.027 except dislocations; 0 of 333 cells clear a
    threshold with their CI.
  - **Decided at review: G1 does not open; Phase 5 closed, Phase 2 not opened** (the literal point-estimate
    rule would have opened it: 23 cells cross).
  - The current default already gives tip coverage 0.893 (≥ 0.89).
- **D2:**
  - Force errors are anti-correlated at the first shell (c_L ≈ −0.27, a bond-force signature) and
    uncorrelated beyond ~5.5 Å.
  - **Force rows are ≈ 96 % independent**, so #64's over-counting argument does not hold. Weight cancellation
    stands alone as Phase 1's motivation.
- **D3:**
  - On the first, unconverged bench365 GP, the kernel had collapsed to an environment-blind δ(s)·species
    form (ρ on its bound, ℓ ≈ 20).
  - **A converged refit (PCA-16) undoes the collapse:** ℓ = 1.96, +432 nats, two starts agreeing.
  - On that fit, k − q is ≤ 0.8 % of the force prior but 14–34 % of the *predictive* force variance. It is
    flat across categories and its share is lowest at the tip (0.14 against 0.34 ID).
  - **Phase 4 is not motivated:** PITC would add a near-constant noise, not tip structure.
  - Phase 3 remains a live experiment.
- **D4:**
  - Si, shared noise against per-quantity: E 10.6 vs 28.2 meV/atom (Phase 1 acceptance met); F 0.144 vs
    0.125 eV/Å (**−15 %, acceptance "no worse" missed**); V 95.8 vs 110.3.
  - Cantor ARD under tied noise: every coverage cell within +0.003 of per-quantity (Phase 1 Cantor
    acceptance met). Test energy RMSE +79 % without explicit weights (3.03 → 5.42 meV/atom).
- **Follow-up runs (approved 2026-10-06, done):** D4 Cantor (below) and a converged bench365 GP refit (D3
  addendum).

## Artefacts: what exists and what is missing

| needed by | artefact | status |
|---|---|---|
| D1, D2 | rev2 default ARD fit (`ard_default`, polish + joint E0, rev2 §7): `model.npz`, `posterior.npz` (cal scores, group table), `ard.json` | **present**: `acegp-data/results/2026-10-04-groups/default_pol/` (same `ard.json` md5 as `2026-10-01-rev2/bench365_ard_default_pol`, whose `posterior.npz` is a 542-byte stub) |
| D1, D2 | per-atom served arrays on the v3 big cells (`dF`, `forces_cov`, `forces_group`, `forces_q`, `fixed`, `r_core`) | **present**: `2026-10-01-rev2/bench365_ard_default_pol/big3*_err.npz` (5 files, 34 cells) |
| D1, D2 | bench365 `train.xyz` / `test.xyz`, v3 `big3_mh1.xyz`, `big3_cracks_r*.xyz` | **present**: `acegp-data/cantor/{bench365,big3}/` |
| D2 | per-atom test errors of a linear and a GP fit on bench365 test | **present**: `2026-09-28/bench365_{pops,gp}/pred_test_map.npz` (F, F_mean per atom) |
| D3 | Cantor GP fit | **partly missing.** No `gp_model.npz` was ever written for any Cantor GP run (`2026-09-26/cantor_gp_pca128_ms4`, `2026-09-27/..._s100`, `2026-09-28/bench365_gp` hold `theta_map.json`, `map_restarts.json`, predictions). θ_MAP is present; the inducing set and feature map were **rebuilt** from the `fit_bench.py` GP arm (FPS, deterministic; PCA up to column signs, which the cosine kernel does not see). The posterior (training statistics) was not rebuilt: that is a GPU pass. |
| D3 | MACE-MH-1 teacher | **present**: `~/.cache/mace/mace-mh-1.model` (torch), `/storage/.../macejax-gpu/bundles/mace-mh-1.json` |
| D4 Si | GAP-18 Si o4d12 linear fits, shared vs per-quantity noise | **present**: `/storage/eng/essswb/projects/ace-jax/runs/shared_noise_o4d12/{shared,per-quantity}` (run on `feat/shared-noise`, PR #67) |
| D4 Cantor | bench365 ARD refits under tied noise | **run 2026-10-06** (Modal, 4.0 B200-h + `big_errors`): `acegp-data/results/2026-10-06-gp-discrepancy-d4/bench365_ard_seq_{pq,shared}` |

## 0.1 Code map

Line numbers are at 6270dfe (= 07e94c6 + the spec).

### Background bullets

| spec bullet | verdict | where |
|---|---|---|
| At fixed θ the GP arm is BLR over Φ = [B \| k_θ(B, B_M)] with prior precision blkdiag(Γ²/σ_c², K_MM) | **confirmed** | `fit/objective.py:1-5, 28-44` (`linear_prior_diag`, `prior_precision`); `K_MM` adds a jitter of 1e-8·mean diag (`fit/kernels.py:350-353`) |
| GP F/V rows are the exact position/strain derivatives of the GP energy rows | **confirmed** | `fit/rows.py:338-355` (`_residual_step`: ∂k/∂U·JU + ∂k/∂s·Js, scattered to i and j, virial by `_voigt`), `fit/rows.py:392-418`; `kernels.grad_k_rows` `fit/kernels.py:343-347` |
| `kernel = δ(s)δ(s')·κ·ψ·(e_z·e_z')` | **confirmed**, with one addition: `KernelSpec.s_floor` clamps s from below before δ (`fit/kernels.py:330-331`); off unless `delta_s_floor_q` is set. s is the power-mean nearest-neighbour distance (`fit/summary.py:359-400`, p = 10, cutoff r_cut) | `fit/kernels.py:302-335` |
| Training LML is SoR; DTC k − q only at prediction (`deriv_dtc=True`, cosine only) | **confirmed** | LML: `fit/objective.py:91-96` (no k − q term anywhere in `stats`/`objective`); DTC energy `fit/predict.py:43-65`, force/virial `fit/predict.py:136-228`, used in `_predict_batch` `fit/predict.py:231-246`; cosine-only raise `fit/predict.py:154-158`. **Stale docstrings:** the module and `Prediction` docstrings (`fit/predict.py:1-7, 28-31`) still say F/V variances are SoR (Ruling R29); the code carries the derivative DTC (R30). |
| `combine()` scales each quantity's block by 1/σ_q² with independent σ_E, σ_F, σ_V | **confirmed**; per-config-type ratios multiply it (`log_ratios`) | `fit/objective.py:47-81` |
| `hypers.py` log-normal hyperpriors, μ σ_E = 1e-3, σ_F = 5e-2 | **confirmed**; also σ_V = 5e-2; widths 2.0 in log for all three, 2.0 for σ_c | `fit/hypers.py:180-197` |
| `make_log_density` adds `log_prior`; does the default MAP use it? | **yes, the MAP maximises LML + log_prior on every path**, but `make_log_density` is not the path: `make_objective` builds `vg` on `lik + log_prior` itself | host-cache `fit/pipeline/objective.py:41-45`; device `:55-62`; the polish's `logpost` `fit/pipeline/mapfit.py:209`. `make_log_density` (`fit/objective.py:228-247`) is used only for the sharded / LOO likelihood (its `.likelihood`). |
| `FitConfig` has `objective = "lml" \| "loo"` and `sigma_type` | **confirmed** | `fit/pipeline/config.py:18, 35` |
| `config.py` rejects `uq='ard'` unless `arm == "linear"` | **confirmed** | `fit/pipeline/config.py:118-120` |

### Additional questions

**Where the jackknife's prior lives, and what it assumes about D = diag Γ.**
`--uq ard` is `fit/ard.py::run_ard_stage` (`:792-985`), with the PRESS shape in `fit/jackknife.py`. Everything
works in the prior-scaled system S = D⁻¹AD⁻¹, D = diag Γ (`ard_gamma`, `fit/ard.py:42-49`; joint-E0 columns
append √e0_prec). D is diagonal by construction, and the code uses that in four ways:

1. **Elementwise column scaling by `dinv`**: `_ev_parts` (`fit/ard.py:199-211`: `dinv[:,None]*M*dinv[None,:]`,
   `dinv*b`), the stored mean `ev.dinv * x` (`:518`), `var_rows`/`misspec_var_rows` (`:420-453`),
   `_val_atoms` (`:654-668`), `press_scores` (`fit/jackknife.py:70`: W = L⁻¹(Ψ_k D⁻¹)ᵀ), `shape_factor`
   (`:94`), `atom_shape` (`:124`: u = D⁻¹φᵀ).
2. **Division by `dinv`** to map back: `press_scores` push-through branch (`fit/jackknife.py:73`,
   g̃ = L z / dinv, i.e. D·(Lz)).
3. **The ARD scales are per-column diagonal precisions** Λ̃ = diag(exp a_k(j)), so Λ = DΛ̃D
   (`fit/ard.py:209-211`); the Γ terms of log|A| − log|Λ| cancel because D is a square root of the prior.
4. **Only the linear columns exist**: `ard_statistics` streams the L-column statistics, and its docstring
   says the inducing columns never enter (`fit/ard.py:60-66`). `ARDPosterior` stores `dinv` (L,) and `chol`
   (L, L) and has no notion of a GP block.

No other module reads `dinv` (`calc/`, `eval/`, `conformal.py`, `support.py` only call `ARDPosterior` methods).

**Can `solve._prior_block`'s R0 stand in for D?** (`fit/solve.py:63-78`: R0 = blockdiag(diag(Γ/σ_c),
chol(K_MM)ᵀ), upper triangular.) **Yes, algebraically, under three conditions.**
- Replace "× dinv on columns" by a right triangular solve Φ R0⁻¹ (diagonal on the linear block, an M×M
  triangular solve on the GP block, so it stays cheap), and "/ dinv" by R0ᵀ·. Then S = R0⁻ᵀAR0⁻¹ =
  R0⁻ᵀΦᵀΦR0⁻¹ + Λ̃, and log|A| − log|Λ| = log|S| − log|Λ̃| holds for any invertible R0. The PRESS identity
  A⁻¹g̃_k = c − c₍₋ₖ₎ and the shape Q̃ = S⁻¹R0⁻ᵀ(G − ḡ) are basis changes, so they carry over unchanged.
- The ARD scale must be **constant within the GP block** (one a_GP, or none): R0ᵀdiag(λ)R0 = blkdiag(Γ²λ_lin,
  λ_GP K_MM) only if λ is constant on the dense block. A per-inducing-point scale would no longer be a
  scaling of K_MM.
- θ is **fixed** at θ_MAP: K_MM and the residual rows depend on θ, so the joint Gram [B | K_NM] must be
  streamed once at θ_MAP (the `residual_statistics` path exists, `fit/stats.py`). The ARD evidence then
  optimises over (σ_q, a_k, a_GP) at that fixed kernel. At serve time the GP force rows `res.F` are needed
  alongside the linear rows (`GPCalculator` already builds them).

**Default MAP: LML alone or LML + log_prior?** LML + log_prior (above). **But the served `--uq ard` posterior
does not see the hyperprior.** In joint mode (the default) `fit_ard` re-maximises the pure evidence over
(log σ_E, log σ_F, log σ_V, a_k) (`fit/ard.py:214-218`, no `log_prior` term), so the ARD posterior's noise
scales are type-II ML regardless of the MAP. Phase 1's `hier` mode and any `shared` mode must therefore also
be built into `ARDEvidence`, or run with `ard_mode="sequential"` (`feat/shared-noise` takes the latter route:
`noise="shared"` refuses joint ARD).

**How `objective="loo"` is computed: cluster or row LOO?** **Leave-one-configuration-out** (cluster LOO), with
the exact block leverage: per configuration, the block of its energy row, all 3N force rows and its 6 virial
rows (`fit/loo.py:94-107`); the pseudo-likelihood is Σ_c log N(g_c; 0, (I − H_cc)) with a per-block
Cholesky (`fit/loo.py:123-143`). It needs the full (non-cached) statistics (`fit/objective.py:232-240`) and
turns off batch packing (`fit/pipeline/config.py:101-105`). **Bug found:** `objective="loo"` with
`lml="host-cache"` passes `validate()` (`fit/pipeline/config.py:149-155` does not check `objective`), and
`make_objective` takes the host-cache branch (`fit/pipeline/objective.py:39-47`), so the fit silently
maximises the LML instead.

**Status of `feat/shared-noise` (#64).** It is now on the remote: **PR #67 (open, 2026-10-05)**, 7 commits on
07e94c6. It implements `--noise {per-quantity,shared}` (no `hier`) by tying σ_E, σ_V to σ_F in `paramset`;
default stays `per-quantity`; refuses `sigma_type`, joint ARD and `--solver lstsq`; records `log_evidence` in
`map_convergence.json`; covers the learned-radial stage; goldens re-recorded on the CPU. GAP-18 Si runs on
/storage (D4 below).

**Receptive field of MACE-MH-1.** Read from the torch model (`ScaleShiftMACE`): `r_max` = 6.0 Å,
`num_interactions` = 2 (two `RealAgnosticResidualNonLinearInteractionBlock`s), plus a short-range
`pair_repulsion_fn`. **Receptive field = 2 × 6.0 = 12.0 Å** (the JAX bundle's JSON agrees). The ACE fit's
r_cut is 6.25 Å, so the teacher sees up to 5.75 Å beyond it.

## 0.2 Diagnostics

### D1 — non-local dependence of the score

Script `bench/defect_uq/scoring/d1_nonlocal.py`; full tables (every group × feature × R, with 90 % cell-bootstrap
intervals, B = 100) in `runs/gp_discrepancy/d1/d1.md`, machine-readable `d1_gate.json`. 32 min on 4 CPU threads,
peak RSS 2.9 GB.

**Set-up.**
- Posterior: the library-default ARD fit (`ard_default`: aniso + transfer exponent, polish, joint E0; rev2 §7).
- Score: the conformal ratio t = s/q_g, covered iff t ≤ 1.
  - T_val: the posterior's own calibration scores (`cal_scores`, P_fit errors, transfer factor applied),
    736 configurations and 30 240 atoms.
  - Big cells: the served aniso region from `big3*_err.npz`, all 34 v3 cells, free atoms only.
- **Alignment check.** T_val: atom counts and the recomputed hard groups match `cal_groups` in 736/736
  configurations. Big cells: the recomputed groups match `forces_group`. The coverages reproduce rev2 §7:
  tip 0.893, all free 0.912, T_val 0.907.
- Features u_j:
  - z̃ − z*: soft coordination, sigmoid at r1 = 3.10 Å, width 0.05 Å; z* = 12.
  - d̃: soft first-shell distortion.
  - s: the GP summary (r0 = 2.5, p = 10, r_cut 6.25).
- g_i = Σ_j f_c(r_ij; R) u_j / Σ_j f_c with j = i included, for R ∈ {0, 6.25, 8, 10, 12}. 12 Å is the
  teacher's receptive field.
- Big cells exclude atoms within 12 Å of the cylinder's outer vacuum, so g at R = 12 does not see the cut.
- **"Not explained by the r_cut feature"** is measured two ways:
  - the partial Spearman ρ(t, g_R | g_{r_cut}), from rank residuals;
  - the *conditional split*: within quintiles of g_{r_cut}, the coverage of the upper half of g_R minus the
    lower half, pooled.

**Pooled over groups** (t is already group-normalised): the non-local part, R > r_cut, given g(r_cut).

| population | atoms | worst \|partial ρ\| (feature, R) | worst \|cond. split\| (feature, R) | raw (unconditioned) coverage spread over z̃ quintiles, R = 8 |
|---|---|---|---|---|
| T_val (ID) | 30 240 | 0.013 (z, 8) | 0.011 (z, 8–10) | 0.066 |
| big cells, all free | 203 762 | 0.062 (s, 8) | 0.020 (z, 10) | 0.067 |
| crack | 177 691 | 0.063 (s, 8) | 0.027 (z, 10) | 0.075 |
| crack tip (r_core ≤ 10 Å) | 28 080 | 0.048 (s, 8) | 0.019 (z, 8) | 0.055 |
| edge | 12 039 | **0.149** (s, 12) [0.12, 0.17] | **0.032** (s, 12) [0.020, 0.042] | 0.040 |
| screw | 14 032 | 0.129 (s, 12) | 0.023 (s, 12) | 0.052 |

**Within single Mondrian groups** (≥ 300 atoms), 333 (population, group, feature, R > r_cut) cells:
- 23 cross a threshold on the point estimate.
- **None does so robustly**: no 90 % interval lies wholly beyond 0.15 (partial ρ) or 0.03 (split).
- The largest at the tip is group 4 (d band 2, z = z*), z̃ at R = 12: split −0.034 [−0.044, −0.015].
- Others:
  - crack group 6, z̃ at R = 8: −0.034 [−0.043, −0.025];
  - crack and big group 1, d̃ at R = 10: +0.033 [0.006, 0.056];
  - T_val groups 1 and 3, s at R = 12: partial ρ +0.17/+0.18 [0.02, 0.28], on 360–398 atoms (CIs of ±0.1).

**Reading.**
1. The raw dependence of coverage on these features is 0.04–0.08 across bins. Almost all of it is
   in-locality: it does not grow from R = r_cut to 12 Å, and conditioning on g(r_cut) leaves |split| ≤ 0.03.
2. The one coherent non-local signal is in the dislocations, through s:
   - partial ρ −0.13 to −0.15: atoms whose 8–12 Å surroundings are more expanded than their r_cut shell
     predicts get lower scores;
   - split +0.02 to +0.03: those atoms are *over*-covered.
   - It is a ranking effect on families that are already conservative (edge 0.937, screw 0.942). It is not a
     coverage gap.
3. At the tip the non-local effects are ≤ 0.034 in one group and ≤ 0.02 pooled. Their sign depends on
   the feature, and the CIs reach inside the thresholds.
4. **The tip gap the spec targets is already closed by the current default.** Tip coverage is 0.893 for
   `ard_default` (rev2 §7), against 0.885 for the pre-polish `aniso_tx` run that the spec's Goal quotes.
   Phase 2's acceptance (tip ≥ 0.89) is therefore met before Phase 2 starts.

### D2 — within-configuration force-error correlation

Script `bench/defect_uq/scoring/d2_error_corr.py`; tables in `runs/gp_discrepancy/d2/d2.md`.
- c(r) = Σ e_i·e_j / (n_pairs ⟨|e|²⟩) over within-configuration pairs, with its longitudinal and transverse
  parts.
- Periodic cells only to half the shortest cell width: about 6 Å for bench365, 7.75 Å along the line for the
  v3 cells.
- **Effective rows.** C = [c(r_ij) I₃] from each set's own curve gives:
  - n_mean = (3N)²/1ᵀC1;
  - the participation ratio n_pr = (tr C)²/tr C², the number of effectively independent rows.

| set | atoms | rms\|e\| (eV/Å) | c, 2.0–2.5 Å (c_L / c_T) | c, 2.5–3.0 Å | \|c\| beyond 5.5 Å | n_pr / 3N (median [p10, p90]) | n_mean / 3N |
|---|---|---|---|---|---|---|---|
| bench365 test, linear (POPS-fit mean) | 38 516 | 0.120 | −0.105 (−0.275 / −0.020) | −0.043 | ≤ 0.016 | 0.960 [0.937, 0.976] | 3.1 |
| bench365 test, GP-arm MAP mean | 38 516 | 0.119 | −0.106 (−0.272 / −0.022) | −0.044 | ≤ 0.035 (104 pairs) | 0.959 [0.936, 0.975] | 3.2 |
| v3 crack, ARD default | 37 692 | 0.189 | −0.104 (−0.278 / −0.017) | −0.037 | ≤ 0.003 | 0.961 [0.955, 0.965] | 7.9 |
| v3 edge | 13 500 | 0.146 | −0.096 (−0.269 / −0.010) | −0.044 | ≤ 0.004 | 0.958 | 8.3 |
| v3 screw | 15 900 | 0.146 | −0.096 (−0.283 / −0.003) | −0.042 | ≤ 0.002 | 0.960 | 6.2 |

**Reading.**
1. Force errors within a configuration are **anti-correlated** at the first shell, and almost entirely
   longitudinally (c_L ≈ −0.27, c_T ≈ 0). That is the signature of an error in a pair-like bond force: the
   two atoms receive equal and opposite errors along the bond. The correlation is ≈ 0 beyond ~5.5 Å, inside
   r_cut, the same in every family and for the linear and GP arms.
2. **The participation ratio is ≈ 0.96 × 3N everywhere: force rows are close to independent.**
   - The over-counting argument of #64 (correlated force rows overstate the information, so the evidence
     overweights forces) is **not supported** at the level of second moments.
   - The cancellation of the E:F:V weights at a converged MAP (#64's main point) is algebraic and does not
     depend on this.
   - The reason the evidence prefers σ_E ≫ σ_F is not row correlation. It is that the misspecification
     differs between quantities: the energy error of a configuration is a sum of correlated site errors.
3. n_mean > 3N: sums of force errors (energy-like functionals) are *better* determined than iid rows would
   suggest. The anti-correlation (and Σ_i e_i = 0 in periodic cells) cancels them.
4. Caveat: C uses the scalar c(r) I₃. The tensor structure (c_L ≠ c_T) would lower n_pr by a few per cent at
   most.

### D3 — GP span at the tip

Script `bench/defect_uq/scoring/d3_gp_span.py`; table `runs/gp_discrepancy/d3/d3.md`, per atom `d3.json`.
Ran on mnf148 (CPU): rebuilding the inducing set takes 9.5 min and **17.8 GB peak RSS**, more than a 16 GB agent
scope holds. Each target took ~4.5 min, 48 targets in all.

**Set-up.** The bench365 GP fit (`2026-09-28/bench365_gp`: PCA-128, M = 100 per species = 500, cosine-SE)
kept only θ_MAP. Its prior is rebuilt exactly:
- the inducing set by FPS, which is deterministic;
- the PCA map up to column signs, which the cosine kernel does not see.

q and k − q are prior quantities, so no training statistics are needed. Force level: k_F, q_F = K_oM K_MM⁻¹
K_Mo and k_F − q_F from `predict._dtc_deriv_residual`, summed over the target's 3 components. Each target
sits in a non-periodic cluster of all atoms (images included) within 2 r_cut + 0.5 Å, which is exact for the
target's rows. The SoR *posterior* variance is not computed: it needs the training statistics, a GPU pass.

| atoms (v3 relaxed cells; bench365 test) | n | median k (eV²) | median (k−q)/k | median k_F (eV²/Å²) | median (k_F − q_F)/k_F [p10, p90] | max (k_F − q_F)/k_F |
|---|---|---|---|---|---|---|
| crack tip, r_core < 5 Å | 6 | 1.9e-2 | 0.001 | 0.18 | 0.002 [0.001, 0.003] | 0.003 |
| crack, 10–20 Å | 6 | 4.0e-2 | 0.001 | 0.29 | 0.002 [0.001, 0.003] | 0.004 |
| edge core, r < 5 Å | 10 | 3.8e-2 | 0.000 | 0.17 | 0.003 [0.002, 0.005] | 0.005 |
| screw core, r < 5 Å | 10 | 5.5e-2 | 0.000 | 0.31 | 0.002 [0.002, 0.004] | 0.007 |
| big-cell bulk, r 18–26 Å | 6 | 3.4e-2 | 0.001 | 0.17 | 0.002 [0.001, 0.003] | 0.004 |
| bench365 test (in distribution) | 10 | 3.2e-2 | 0.001 | 0.21 | 0.002 [0.002, 0.003] | 0.003 |

**Reading (first pass; reasons 2–4 are superseded by the addendum below).**
1. **There is no k − q excess at the tip.** At most 0.1 % of the site prior and 0.7 % of the force prior lies
   outside the inducing span, in every category, tip and in-distribution alike.
2. **Why: the fitted kernel is environment-blind.** θ_MAP has:
   - ℓ = e^2.98 = 19.7 on the unit sphere, so κ ≥ e^(−4/2ℓ²) = 0.995 for every pair of sites;
   - ρ = 100.0, **on its L-BFGS upper bound** (`fit/pipeline/mapfit.py:17`), so ψ ≈ 1.

   The kernel is then δ(s)δ(s′)(e_z·e_z′) to within 0.5 %: a function of the nearest-neighbour summary s and
   the species only. Its prior is nearly rank-NZ in descriptor space, which 500 inducing points span
   trivially.
3. This explains the record the spec cites: GP σ ranks poorly out of distribution (AUROC 0.27–0.70). The
   fitted discrepancy cannot tell a crack tip from bulk at the same s.
4. **Caveat: that MAP is not converged.** All 4 L-BFGS restarts stopped at the 40-iteration limit
   (`map_restarts.json`: "TOTAL NO. OF ITERATIONS REACHED LIMIT"; the run predates the converged MAP of #63).
   Restart 1 also ran to ρ ≈ 99; restart 0 to ρ = 47, ℓ = 27. So the collapse is the direction the evidence
   pulls, not an artefact of one start. Whether a *converged* MAP stays there is open.

#### D3 addendum: the converged GP refit (2026-10-06)

**Refit.** `fit_bench.py` arm `gp_conv`: the GP arm with PCA-16 + host-cache (the #60 fast path), 4 L-BFGS
starts of up to 300 iterations; Modal B200, 3.5 h. It saves `gp_model.npz`
(`acegp-data/results/2026-10-06-gp-discrepancy-d4/bench365_gp_conv/`).

| fit | starts: logpost (evaluations) | best ℓ | best ρ | min κ over pairs | test E (meV/atom) / F (eV/Å) |
|---|---|---|---|---|---|
| `bench365_gp` (PCA-128, `map_steps` 40) | 540 424 / 539 246 / 536 478 / – (all at the iteration limit) | 19.7 | 100 (bound) | 0.995 | 3.06 / 0.0687 |
| `gp_conv` (PCA-16, converged L-BFGS) | **540 856** (111) / 540 855 (63) / 537 823 (92) / 540 412 (178) | **1.96** | 76 | **0.59** | 2.98 / 0.0687 |

- **The environment-blind kernel was an artefact of the unconverged MAP.** Two independent starts reach the
  same optimum (ℓ ≈ 1.9–2.0, within 1 nat), 432 nats above the old fit.
  - There κ spans 0.59–1 over site pairs: the kernel resolves the environment.
  - ρ = 76 is off its bound, but ψ is still ≈ 1, so the bump factor is inactive.
  - Start 3 settled in the old basin (ℓ = 9.5, 540 412). That is the local mode the 40-iteration fit was
    heading for.
- The convergence check still reports `converged: False` (predicted Newton gain 46.6 nats, in `log_sigma_F`).
  Host-cache has no polish, so this is as converged as this path gets.
- Test errors are unchanged. Two things differ from the old fit (PCA-16 against PCA-128, converged against
  not), so the change in θ is not attributed to convergence alone.

**D3 on the converged fit** (`d3_gp_span.py --gp-model`, same 48 targets; tables
`bench/defect_uq/results/2026-10-05_gp_discrepancy/d3_converged.md`). The posterior is now available, so the
predictive force variance splits into the SoR posterior part and the derivative-DTC part:

| atoms | n | median (k_F − q_F)/k_F (prior) | median SoR posterior var (eV²/Å²) | median DTC var | median DTC / predictive [p10, p90] |
|---|---|---|---|---|---|
| crack tip, r < 5 Å | 6 | 0.005 | 6.5e-4 | 1.2e-4 | **0.14** [0.11, 0.24] |
| crack, 10–20 Å | 6 | 0.004 | 4.8e-4 | 1.5e-4 | 0.21 [0.16, 0.30] |
| edge core | 10 | 0.005 | 5.2e-4 | 1.5e-4 | 0.21 [0.16, 0.27] |
| screw core | 10 | 0.004 | 5.0e-4 | 1.7e-4 | 0.24 [0.16, 0.37] |
| big-cell bulk | 6 | 0.004 | 5.6e-4 | 1.5e-4 | 0.18 [0.12, 0.24] |
| bench365 test (ID) | 10 | 0.005 | 2.3e-4 | 1.3e-4 | 0.34 [0.25, 0.52] |

**Reading (supersedes D3's reasons 2–3; reason 1 stands).**
1. The prior is still ≥ 99.2 % inside the inducing span everywhere. The posterior then shrinks the SoR part
   to ~2 % of the prior, so the near-constant k − q becomes 14–34 % of the *predictive* force variance.
   **It is not negligible in the served variance.**
2. **The DTC term carries no tip signal.** It is flat across categories (1.2–1.7e-4 eV²/Å²) and its share is
   *lowest* at the tip. What grows away from the training data is the SoR posterior variance (tip 2.8× ID).
3. PITC/FITC would put that flat k − q into the training likelihood as an almost homoscedastic extra force
   noise. It cannot carry the structured tip error Phase 4 was meant to capture. **Phase 4 is still not
   motivated, now on a converged fit.**
4. **Phase 3 (GP-arm sandwich) is back to plausible.** The converged GP's kernel does resolve local
   structure, so the GP arm is not a priori blind. `--ard-variance dtc` would add an almost constant term; the
   SoR/sandwich part is where any gain must come from. Its acceptance test (match or beat linear-arm rev2 on
   tip coverage and ρ) is the experiment.
5. Peak RSS for D3 on a saved model: 10.9 GB (the 15 540² posterior factor), against 17.8 GB for the rebuild.

### D4 — noise-model sensitivity

**GAP-18 Si, linear o4d12, ACEpotentials weights (E 30 / F 1 / V 1), converged MAP + polish.** Run on
`feat/shared-noise` (PR #67), `runs/shared_noise_o4d12/` (UQ columns not read, see the caveat at the top).

| noise | σ (learned) | E (meV/atom) | F (eV/Å) | V (meV/atom) | logpost |
|---|---|---|---|---|---|
| per-quantity | σ_E 2.71, σ_F 0.102, σ_V 0.348 | 28.2 | **0.125** | 110.3 | 315 170 |
| shared (tied) | σ = 0.137 | **10.6** | 0.144 | **95.8** | 193 276 (not comparable: tied scales drop two hyperpriors) |
| ACEpotentials BLR, like-for-like basis (from the PR #67 CHANGELOG) | one σ | 10.2 | 0.135 | 70 | – |

- Shared noise moves the energy within 4 % of ACEpotentials (Phase 1 acceptance: within 10 % of 9.9 → ≤ 10.9,
  **met**).
- **Forces get 15 % worse** than per-quantity (0.144 against 0.125). Phase 1's acceptance "force RMSE no
  worse than per-quantity" is **not met** with these weights.
- It is a trade the weights set, as in ACEfit: ACEfit itself sits at 0.135. The acceptance criterion should
  say so (see open question 1).
- Per config type, shared improves energies almost everywhere and costs forces most on liquid, amorphous and
  surfaces.

**Cantor (bench365, `--uq ard`, run 2026-10-06 on Modal).**
- Arms `ard_seq_pq` and `ard_seq_shared` (`D4_ARMS`), source `feat/shared-noise` @ 2936506. Both are
  sequential: shared noise refuses joint ARD.
- Fits took 1.9 and 2.1 B200-h; `big_errors` on all 34 v3 cells.
- Scored by `bench/defect_uq/scoring/d4_score.py`:
  - the 300-config ID sample of the rev2 tables, through `ACECalculator(posterior=)`;
  - `validate_shape` coverage on the big cells, cell-weighted, with 90 % cell-bootstrap CIs.
- Table: `bench/defect_uq/results/2026-10-05_gp_discrepancy/d4_cantor.md`; run files:
  `acegp-data/results/2026-10-06-gp-discrepancy-d4/`.

| run | MAP σ_E / σ_F / σ_V | ID cfg (atom) | all | crack | tip | edge | screw | ρ crack / edge / screw | test E (meV/atom) / F (eV/Å) / V |
|---|---|---|---|---|---|---|---|---|---|
| `ard_default` (joint, per-quantity; rev2 §7) | – | 0.898 (0.903) | 0.911 | 0.908 | 0.893 | 0.937 | 0.942 | 0.37 / 0.30 / 0.32 | 3.03 / 0.0687 / 0.734 |
| `ard_seq_pq` (sequential, per-quantity) | 0.0171 / 0.0661 / 0.102 | 0.898 (0.903) | 0.911 | 0.908 | 0.893 | 0.937 | 0.942 | 0.37 / 0.30 / 0.32 | 3.03 / 0.0687 / 0.734 |
| `ard_seq_shared` (sequential, shared) | 0.0675 (tied) | 0.898 (0.904) | 0.913 | 0.909 | 0.894 | 0.940 | 0.944 | 0.37 / 0.28 / 0.30 | **5.42** / 0.0689 / 0.691 |

- **The baseline reproduces rev2 §7 exactly.** Sequential and joint ARD give identical tables at three
  decimals, so the switch to sequential that shared noise forces costs nothing.
- **Calibrated force UQ is insensitive to the noise model.** Every coverage cell under shared noise is
  within +0.003 of per-quantity (90 % CIs about ±0.004). This is Phase 1's Cantor acceptance (within 0.01),
  **met**.
  - The conformal per-group scales absorb the change, as expected.
  - Rank correlation drops by 0.02 on edge and screw.
- **The mean fit is not insensitive.** With `fit_bench`'s default weights (E:F:V = 1:1:1, no `--weights`),
  shared noise:
  - raises the test energy RMSE by **79 %** (3.03 → 5.42 meV/atom);
  - leaves forces unchanged (0.0687 → 0.0689 eV/Å) and improves the virial by 6 %.

  With no explicit weights, a tied σ hands the E:F:V balance to an arbitrary default. That is the case for
  open question 1's answer: `shared` only when `--weights` is given.

## Gate decisions

**G1 (D1): not passed — decided at review (2026-10-06). Phase 5 is closed and Phase 2 is not opened; the tip gap
is in-locality.** The literal point-estimate rule would have opened it; the reasons for overriding it:
- **Literal reading.** "coverage spread across g bins > 0.03, or |ρ| > 0.15" on point estimates opens the gate:
  - 23 of 333 within-group cells cross a threshold;
  - at the tip, group 4, z̃ at R = 12: conditional split −0.034.
- **Why I propose not to treat that as passing:**
  1. *Multiplicity and noise.* The thresholds were set for one comparison. Across 333 cells of 360–24 000
     atoms the point estimates scatter by ±0.03 (split) and ±0.1 (partial ρ) from sampling alone. No cell's
     90 % cell-bootstrap interval lies wholly beyond a threshold.
  2. *Pooled over groups it is in-locality.* |conditional split| ≤ 0.027 and |partial ρ| ≤ 0.063 in every
     population except the dislocations. Raw coverage dependence (0.04–0.08) is the same at R = r_cut as at
     12 Å.
  3. *Inconsistent sign.* The tip's largest effects have opposite signs in different groups and features
     (−0.034 for z̃ in group 4, +0.033 for d̃ in group 1). A single non-local band could not fix both.
  4. *No gap left to close.* Tip coverage is 0.893 for the current default (`ard_default`), above Phase 2's
     acceptance of 0.89.
- **The one robust non-local signal is the dislocations' s at 8–12 Å** (partial ρ −0.13 to −0.15, pooled). It
  points to over-coverage of atoms in expanded far fields, in families already at 0.94. It is a *ranking*
  lead (Spearman ρ on large cells, the spec's other target), not a coverage fix. If Phase 2 is wanted for
  ranking, `s` averaged to 8–12 Å is the D1 winner, but nothing here says it would lift ρ by more than a few
  hundredths.
- If review prefers the literal rule, the cheapest test is Phase 2 itself: one fit plus `big_errors`, ~1.5
  GPU-h. Its outcome is predictable from D1: tip coverage moves by ≤ 0.01.

**D3 → Phase 4 motivation (no formal gate): not motivated.** On the converged GP refit the derivative-DTC
term is 14–34 % of the predictive force variance but flat across categories, with its smallest share at the
tip (D3 addendum). PITC would add a near-homoscedastic noise, not the structured tip error. The first D3
pass's reason (an environment-blind kernel) was an artefact of an unconverged MAP and is withdrawn. **Phase 3
is the GP-arm experiment worth running.** Use the converged `gp_conv` fit as its base, and note that it still
reports a 46.6-nat predicted gain.

**D2 → Phase 1 rationale:** the force-row over-counting argument is not supported (n_pr ≈ 0.96 × 3N). Phase 1
stands on the weight-cancellation argument alone. D4-Si shows its price: −62 % energy RMSE, +15 % force RMSE
at ACEpotentials' weights.

**D4 (Cantor):** calibrated force UQ unchanged under shared noise (all cells within +0.003); the energy
RMSE is not (+79 % at default weights). Phase 1 should ship `shared` only with explicit weights.

## Open questions: proposed answers

1. **Default for `--noise`.** Proposed: **`shared` only when `--weights` is given explicitly; keep
   `per-quantity` otherwise.**
   - With explicit weights the user has stated the E:F:V balance, and per-quantity noise silently undoes it
     (#64).
   - Without weights, the default 1/1/1 balance has no intent behind it, so the evidence-learned balance is the
     better guess.
   - D4-Si shows `shared` is a trade, not a free win: −62 % E, +15 % F.
   - D2 removes the over-counting argument for making `shared` universal.
   - Also amend Phase 1's acceptance: "force RMSE no worse than per-quantity" cannot hold under `shared` at
     energy-heavy weights. Replace it with "within 10 % of ACEpotentials BLR on E, F and V at the same
     weights" (Si: 10.6 / 0.144 / 95.8 against 10.2 / 0.135 / 70; the virial misses that by 37 %, worth a
     look first).
2. **`hier`.** Research option only, behind `--noise hier`, and only if the Cantor D4 run shows that shared
   noise moves the ARD posterior. It must be built into `ARDEvidence` as well as the MAP: joint ARD refits σ_q
   by pure evidence (0.1).
3. **G1 thresholds.** Keep the values, but require the 90 % cell-bootstrap interval, not the point estimate,
   to clear them, and apply the test pooled over groups as well as per group (≥ 1000 atoms per group). As
   written, the rule fires on noise across ~300 cells (G1 above).
4. **Phase 3 naming.** Introduce `--uq sandwich` (the jackknife-sandwich posterior on either arm) and keep
   `--uq ard` as its linear-arm alias, with the ARD evidence as the prior.
   - On the GP arm the prior is not ARD (one scale on the K_MM block at most; 0.1).
   - The served quantities, groups and conformal scales are the same, so one name for the UQ family is
     clearer.
   - D3 (converged) suggests `--ard-variance dtc` adds a near-constant term; the sandwich over
     [B | k(B, B_M)] is the part to test.
5. **PITC blocks = jackknife clusters.** Acceptable in principle: both are "rows whose misspecification is
   correlated". D2 measures that correlation as confined to < 5.5 Å, so any block of ≥ 3 r_cut holds it.
   **But D3 argues against Phase 4 as specified**: on the converged GP the k − q term is flat across
   categories and smallest, as a share, at the tip, so PITC would add a near-constant noise. Defer Phase 4.
6. **Non-local R.** Always explicit (`--nl-radius`), with the teacher's receptive field *reported* in the fit
   log when the labels' provenance is known.
   - ace-jax cannot see a foundation model's identity in an extxyz.
   - D1 found no R in 8–12 Å better than another; the effects saturate by 8 Å.
   - Moot while Phase 5 is closed.

## Phase 3 result: `--uq ard-gp` (2026-10-07)

Implemented on `feat/gp-sandwich` (plan `docs/dev/plans/2026-10-06-gp-sandwich.md`). Acceptance on bench365 and
the 34 v3 big cells: `bench/defect_uq/results/2026-10-07_ard_gp_acceptance.md`.

| arm | ID | tip | crack | ρ crack / edge / screw |
|---|---|---|---|---|
| linear `ard_default` | 0.898 | 0.893 | 0.908 | 0.370 / 0.296 / 0.318 |
| GP sandwich (`ard_gp_sw`) | 0.900 | **0.902** | 0.912 | 0.362 / 0.286 / 0.312 |
| GP `dtc` (`ard_gp_dtc`) | n/a | 0.827 | 0.838 | 0.282 / 0.136 / 0.163 |

- **Sandwich: better tip coverage, worse ranking.** Tip coverage improves (+0.009, paired interval excludes 0)
  and in-distribution coverage is on target. But ρ is lower by 0.006–0.009 in every family (intervals exclude
  0). It does not "match or beat" on both, so **`ard-gp` stays experimental**, as the spec rules.
- **`dtc` fails clearly:** under-coverage off distribution, and ρ lower by 0.09–0.16. This is what D3
  predicted: the DTC term is near-constant across environments. `--ard-variance dtc` was removed at review.
- **Phase 4 (PITC)** was already unmotivated (D3). This result does not change that: the GP block's fitted
  ARD scale is a_GP = −0.04, so the evidence keeps the MAP prior, and the structured tip error is not in the
  span the GP misses.
