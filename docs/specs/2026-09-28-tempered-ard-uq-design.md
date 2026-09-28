# Tempered ARD posterior: calibrated per-atom force uncertainty for linear ACE

Status: design, for review · Base: `main` (after #8 and #10) · Branch: `feat/ard-uq`
Evidence: `bench/defect_uq/README.md` (PR #12); the numbers below are from there.

## Goal

A fitted linear ACE model should report, for any structure, a **calibrated per-atom force
uncertainty** `forces_std`. It must:
- come from a Bayesian posterior;
- rank where the model is extrapolating;
- stay calibrated on defect *combinations* absent from training;
- work on big cells (≥ 10⁴ atoms), for fracture and plasticity runs.

The construction is the **tempered ARD posterior**. It is the Bayesian linear regression (BLR)
posterior of the linear model, with three changes:
- (i) prior scales per body order, fitted by evidence **jointly with the noise scales**;
- (ii) a single temperature κ, fitted by held-out NLL, that inflates the posterior to absorb model
  misspecification (a generalised or tempered posterior);
- (iii) it is evaluated on per-atom force rows. Site energies are not identifiable from energy and
  force data, and their posterior σ is anti-correlated with error.

On the defect benchmark it is the best out-of-distribution ranker:
- AUROC 0.75–0.90 on held-out combinations, against 0.27–0.70 for POPS and GP σ;
- a single κ = 6.85, fitted in distribution, gives rms-z 0.91–1.02 and 90 % coverage 0.87–0.92 on
  every held-out family.

## Non-goals

- The GP arm. Its σ ranked poorly OOD, and big-cell GP rows are out of scope.
- Changing POPS or the `blr` default.
- Energy or virial uncertainty. Forces first; energies follow the same algebra later.

## Mathematics

The linear model is E = Σᵢ φ(xᵢ)·c. The fit rows are energy, force and virial rows, weighted by
w/σ_q. The pieces are:
- the data Gram M = Σ_q G_q/σ_q² and the moment b = Σ_q b_q/σ_q², from one sufficient-statistics
  pass;
- the prior precision Λ = Γ² ⊙ exp(a_{k(j)}), where Γ is the smoothness prior and k(j) ∈ {2, 3, 4}
  is the body order of column j (pair and correlation-order-1 columns are 2-body);
- A = M + Λ, the posterior N(A⁻¹b, A⁻¹), and the Cholesky factor A = LLᵀ.

**Evidence.** The hyperparameters are h = (log σ_E, log σ_F, log σ_V, a₂, a₃, a₄), with the
per-quantity statistics G_q, b_q, yᵀy_q and n_q taken from one statistics pass. Up to constants,

log p(D|h) = −½ Σ_q yᵀy_q/σ_q² + ½ bᵀA⁻¹b − ½ log|A| + ½ log|Λ| − Σ_q n_q log σ_q.

- It is maximised **jointly over all six** (type-II ML) by L-BFGS with a JAX gradient, initialised
  from the linear MAP (σ_q) and Γ-only scales. Each evaluation costs one Cholesky.
- **Numerics (measured on bench365):**
  - Work in the prior-scaled system S = D⁻¹AD⁻¹, D = diag(Γ). cond(A) reached 1.3·10¹⁷ at the
    fitted point, against 2.3·10¹³ for S.
  - Divide the objective by its initial gradient norm. L-BFGS-B's first bounded step is the full
    gradient (~3·10³), which otherwise lands on the box corner.
  - Floor the 2-body prior precision so cond(S) ≲ 10¹⁴. At the unfloored ARD optimum cond(S) ≈
    4·10¹⁷ and repeated evaluations differ by up to 0.3 nats, which breaks the line search.
- **Measured effect of joint vs ARD-after-MAP:**
  - +4865 nats for ARD over Γ-only, and a further ≈ +2.4 nats for the joint fit;
  - log σ_F moves by 0.0023 (0.2 %): the noise scales are pinned by ~10⁶ force rows and nearly
    decoupled from the prior scales;
  - the Laplace std of h is 0.001–0.11 (log units), so marginalising adds a per-cent-level effect
    on σ_F, negligible next to κ.
- This is a dedicated linear-evidence optimiser. `Hypers` is untouched.
- **Diagnostic, not a feature:** a Laplace approximation over h (6×6 Hessian by finite differences
  of the gradient). It reports each scale's posterior std and the hyperparameter contribution to
  σ_F. With thousands of configurations, the hyperparameter posterior is expected to be nearly a
  point.

**Temperature.** κ is deliberately *not* an evidence hyperparameter. It absorbs model
misspecification: systematic, correlated residuals, with an effective sample size ~1/κ² of the
nominal. Evidence maximisation cannot see this. It treats σ_F as independent noise, the
Swinburne–Perez argument. κ is therefore a generalised-Bayes learning rate, chosen by held-out
predictive performance.

Hold out a fraction of training configurations (default 0.2, as POPS does). Fit the ARD posterior
on the rest. Choose κ to minimise the Gaussian NLL of held-out per-atom force
errors, with the error ~ N(0, κ² s²/3 · I₃) and s² = Σ_c φ_{F,i,c} A⁻¹ φ_{F,i,c}ᵀ. The closed form
is κ² = mean(|ΔF|² / (s²/3)) / 3. Then refit the posterior on all the training data and keep κ.

**Prediction** for atom i: σ_F,i = κ · ‖L⁻¹ Φ_F,iᵀ‖_F. Φ_F,i is the atom's (3, L) block of force
design rows. The cost is a triangular solve per atom: 3N·L² flops per structure.

## Components

### 1. Node-chunked linear design rows (`fit/rows.py`)

`linear_rows` materialises the full edge Jacobian J (edges × D × 3). For one cell of more than
~2.8k atoms, the J tensor and its GEMM exceed 2³¹ elements, so int32-indexed GEMM autotuning fails.
A new `linear_rows_chunked(model, cfg, batch, node_chunk)` will:
- return the same `Rows` (E, F, V), but never J or X;
- loop over chunks of centre nodes, computing each chunk's edge Jacobian and scatter-adding it into
  F (send and receive atoms) and V;
- stay exactly equal to `linear_rows` (tested to 1e-12);
- be used by every linear-arm path that does not need J: sufficient statistics for the linear arm,
  BLR/POPS prediction, and the UQ calculator.

The GP path keeps `linear_rows`.

### 2. ARD evidence fit (`fit/ard.py`, new)

- `body_order_columns(meta, cfg) -> (L,) int` gives each column's body order in the `_place`
  layout.
- `fit_ard(stats, gamma, groups, h0, mode="joint") -> (h, logev)` does the joint type-II ML over
  (σ_E, σ_F, σ_V, a₂, a₃, a₄).
  - `mode="sequential"` fixes σ_q at the linear MAP and fits only the ARD scales. It needs just the
    combined Gram, one L² matrix, where the joint mode keeps G_E, G_F and G_V separate (three).
  - Sequential is the low-memory fallback for large bases: 29 GB against 86 GB in float64 at
    L = 60k; 1.8 against 5.4 GB at L = 15k.
  - On bench365 the two agree to 2.4 nats and 0.2 % in σ_F.
  - `FitConfig.ard_mode` ∈ {"joint", "sequential"}, default "joint"; CLI flag `--ard-mode`.
- `laplace_hypers(stats, gamma, groups, h) -> (cov_h, report)` is the diagnostic.
- `ArdPosterior(mean, chol, kappa, a, groups)` has two methods:
  - `.forces_std(Phi_F)`, with Φ_F as (N, 3, L), returns (N,) values of κ‖L⁻¹Φᵀ‖;
  - `.save(path)` / `load(path)`.

### 3. Pipeline (`fit/pipeline/`)

`FitConfig.uq` gains `"ard"`, valid for the linear arm only.
- After the linear MAP, `fit()` runs the ARD stage:
  - statistics (the cached M, b where available);
  - `fit_ard`;
  - the held-out κ fit;
  - the full refit.
- Stage timings go into the log, and there is a checkpoint hook.
- Predictions use the tempered σ for F_var and the ARD mean for F_mean.
- Energy and virial variances use the same posterior, untempered and flagged as such.
- `write_outputs` writes `model.npz` (the ARD mean, loadable by `ACECalculator`), `posterior.npz`
  (Cholesky factor, float32 by default, plus κ, scales, groups and a schema version), and
  `ard.json` (evidence, scales, κ, held-out NLL and rms-z).

### 4. Serving (`calc/point.py`, `cli.py`)

- `ACECalculator(model, posterior=None)`. With a posterior, it adds `forces_std` per atom, computed
  from chunked design rows. It runs only when `forces_std` is requested, or when it is set to
  compute every call.
- `aj fit ... --uq ard [--ard-val-frac 0.2]`.
- `aj eval --model model.npz --posterior posterior.npz` adds a `fmax_std` column, and
  `--per-atom out.xyz` writes per-atom `forces_std` arrays for visualisation (e.g. OVITO colouring
  of a crack tip).

## Decisions taken (review these)

1. **Joint type-II ML over the noise scales and the ARD scales** (default; `ard_mode="sequential"`
   is the low-memory fallback), in a dedicated linear-evidence
   optimiser initialised from the linear MAP. It supersedes an earlier draft that fitted ARD after
   the MAP with σ_q fixed: that was one step of coordinate ascent on the same objective. The
   hyperparameters are *maximised*, not marginalised. The Laplace diagnostic measures what
   marginalising would add, and is to be revisited only if that is material.
2. **κ comes from an internal train hold-out, never the test set.** The final posterior is refit on
   all the training data with the same κ. In the benchmark, κ was constant across families (rms-z
   6.2–7.0), so this should be stable.
3. **The posterior factor is stored in float32 in a separate file**: 0.9 GB at L = 15k, against
   1.8 GB in float64. `model.npz` stays a plain, small ACE file. The σ values are insensitive to
   float32 (to be tested).
4. **Groups are body orders only.** Per-degree ARD is a later option.
5. **σ is served on forces only**, as above.

## Testing

- The chunked rows equal the unchunked rows (E/F/V to 1e-12), including a chunk count that is not
  a divisor of N and a >2.8k-atom synthetic cell on CPU (slow marker).
- `fit_ard` evidence gradient: forward-mode vs reverse-mode.
- The joint maximum's evidence is ≥ both the Γ-only value and the ARD-after-MAP value on the
  fixture.
- The evidence matches a dense reference (explicit marginal Gaussian likelihood on a tiny problem).
- The κ closed form matches brute-force NLL minimisation.
- A `posterior.npz` round trip reproduces `forces_std` (float32 within 1e-4 relative).
- The calculator's `forces_std` equals the pipeline's predicted F σ for the same configurations.
- CLI: `aj fit --uq ard`, then `aj eval --posterior` on the Si fixture.
- **Acceptance:** rerun `bench/defect_uq` with `--uq ard` and match the benchmark's numbers
  (AUROC and calibration within noise). Add the first **big-cell** result (crack and dislocation
  σ_F calibration), which is currently missing.

## Risks

- **The evidence fit can be ill-conditioned for groups with little data.** The 2-body scale went to
  e⁻¹³. Mitigations are bounds on a and float64 throughout.
- **Memory.** The factor needs L² on the GPU at prediction time. That is 1.8 GB at L = 15k, fine,
  but a basis with L = 60k would need a low-rank or block approximation (a follow-up).
- **κ fitted on bulk and simple defects may not transfer to regimes far from training** (big-cell
  cracks are the test). The acceptance run measures it.
