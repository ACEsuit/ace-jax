# Noise, discrepancy and non-local GP terms for calibrated force UQ — design

Status: design, for review · Base: `main` at 07e94c6 · Related: #64 (per-quantity noise),
`docs/dev/specs/2026-10-01-force-uq-math-pipeline.md`, `2026-09-28-tempered-ard-uq-design.md`
Evidence: `bench/defect_uq/` (Cantor, MACE-MH-1 labels), rev2 acceptance
`bench/defect_uq/results/2026-10-03_rev2_acceptance.md`.

This spec is written for Claude Code. **Phase 0 is exploration only**: it reads code, runs
diagnostics and writes a findings report. It changes nothing under `src/`. Every later phase
has a gate. A gate decides whether the phase is built, and the gate result is written to the
findings report before any code for that phase starts.

## Goal

Reduce the remaining force-UQ failure modes of `aj fit --uq ard`, chiefly the crack-tip gap
(coverage 0.885 at nominal 0.90; rank correlation 0.29–0.38 on large cells), by treating
model error as a **structured discrepancy** instead of iid per-quantity noise. Concretely:

1. make the E:F:V noise model coherent (#64);
2. let the existing residual GP feed the calibrated UQ pipeline;
3. keep the derivative coupling of the GP discrepancy in the training likelihood (PITC);
4. add an affordable **non-local** discrepancy term for error beyond `r_cut`.

## Non-goals

- Energy or virial calibration (forces only, as now).
- LAMMPS export of any new GP term. ASE `GPCalculator` and `aj eval` only.
- Replacing the hold-out + transfer exponent with jackknife+. It is related, but a separate
  spec.
- Changing the linear arm's mean model, except through the noise model of Phase 1.

## Background (what the code does now — verify in Phase 0)

- At fixed θ the GP arm is BLR over `Φ = [B | k_θ(B, B_M)]` with prior precision
  `blkdiag(Γ²/σ_c², K_MM)` (`fit/objective.py`). So `δ(x) = k(x, X_M) w`, `w ~ N(0, K_MM⁻¹)`:
  a site-additive GP discrepancy with the Nyström kernel (subset of regressors, SoR).
- GP force and virial rows are the exact position and strain derivatives of the GP energy rows
  (`fit/rows.py::_residual_step`, `kernels.grad_k_rows`). E–F–V coupling of the discrepancy is
  therefore built in, but only for the part inside the inducing span.
- `kernel = δ(s)δ(s')·κ·ψ·(e_z·e_z')`: a learned one-point amplitude on the summary `s`
  (`fit/summary.py`), i.e. a heteroscedastic discrepancy.
- The training LML is SoR. The DTC residual `k − q` enters only at prediction
  (`fit/predict.py`, `deriv_dtc=True`, cosine kernel only).
- `combine()` scales each quantity's block by `1/σ_q²` with independent `σ_E, σ_F, σ_V`.
  At a converged MAP these cancel the user's E:F:V weights (#64).
- `hypers.py` carries log-normal hyperpriors on the σ's (`μ: σ_E = 1e-3, σ_F = 5e-2`).
  `make_log_density` adds `log_prior`. Check whether the default MAP path uses it.
- `FitConfig` already has `objective = "lml" | "loo"` and `sigma_type` (per-config-type noise).
- `config.py` rejects `uq='ard'` unless `arm == "linear"`.
- The tempered-ARD spec recorded that GP σ ranked poorly out of distribution
  (AUROC 0.27–0.70 against 0.75–0.90 for ARD). Any GP-arm UQ must beat that record, not assume
  otherwise.

## Phase 0 — explore and diagnose (no `src/` changes)

Branch `explore/gp-discrepancy`. Scripts go in `bench/defect_uq/scoring/`. The report goes in
`docs/dev/gp-discrepancy-exploration.md`.

### 0.1 Code map

Confirm or correct each "Background" bullet with file:line references. In addition, answer:

- Which code path builds the prior for the jackknife (`fit/ard*` or wherever `--uq ard` lives),
  and what it assumes about a *diagonal* prior `D = diag Γ`. The GP block `K_MM` is dense.
- Whether `solve._prior_block` (upper-triangular `R0 = blockdiag(diag(Γ/σ_c), chol(K_MM)ᵀ)`)
  can stand in for `D` in the PRESS and shape algebra.
- Whether the default MAP maximises the LML alone or LML + `log_prior`.
- How `objective="loo"` is computed, and whether it is cluster or row LOO.
- The current status of the `feat/shared-noise` work referenced in #64 (it is not on the remote
  at 07e94c6).
- The receptive field of the MACE-MH-1 teacher (number of interactions × `r_max`, read from the
  model). Do not assume a value.

### 0.2 Diagnostics

Each diagnostic is a script plus a table in the report. Use the existing rev2 fit (`--uq ard`,
linear arm) unless stated otherwise.

- **D1 — non-local dependence of scores.** For every `T_val` atom and every big-cell atom,
  compute smoothed non-local features
  `g_i = Σ_j f_c(r_ij; R) u_j / Σ_j f_c(r_ij; R)` with `u_j ∈ {z̃_j − z*, d̃_j, s_j}` (soft
  coordination, soft first-shell distortion, the GP summary) for `R ∈ {r_cut, 8, 10, 12} Å` and
  the teacher receptive field. Within each Mondrian group report:
  Spearman ρ(score `s_i`, `g_i`); coverage of `forces_q` in `g` quantile bins; the same for
  crack-tip atoms alone.
  **Gate G1:** a within-group dependence of coverage on `g` at `R > r_cut` that the `R = r_cut`
  feature does not explain (coverage spread across `g` bins > 0.03, or |ρ| > 0.15) opens
  Phases 2 and 5. Otherwise Phase 5 is closed and the report says the tip gap is in-locality.
- **D2 — within-configuration error correlation.** Estimate the correlation of force-error
  components within a configuration as a function of distance, and the effective number of
  independent force rows per configuration. This is the evidence for the force-row
  over-counting argument in #64.
- **D3 — GP span at the tip.** On the existing Cantor GP fit (`bench/acegp_cantor`), compare
  the SoR variance `q` and the DTC residual `k − q` (with `deriv_dtc`) at crack-tip, dislocation
  and bulk atoms. A large `k − q` share at the tip motivates Phase 4.
- **D4 — noise-model sensitivity.** Linear arm, GAP-18 Si and Cantor: refit with
  `σ_E = σ_F = σ_V` tied (a local hack in the bench script is fine). Report E/F/V RMSE and the
  rev2 coverage table under both noise models.

### 0.3 Deliverable

`docs/dev/gp-discrepancy-exploration.md`: the code map, D1–D4 tables, the gate decisions, and
the list of open questions below with proposed answers. Stop and ask for review.

## Phase 1 — noise model (#64)

Branch `feat/shared-noise`. Both arms.

- `aj fit --noise {per-quantity,shared,hier}` (`FitConfig.noise`, `fit.yaml` key `noise`).
  - `shared`: one learned `σ`. Every row is scaled by `w/σ`, so the weights set the E:F:V
    balance, as in ACEfit.
  - `per-quantity`: today's behaviour.
  - `hier`: `log σ_q = log σ + η_q`, `η_q ~ N(0, τ²)`, `τ` from `--noise-tau` (default 0.5).
    Implement through the existing hyperprior machinery. The MAP objective must include the
    prior for this mode (see 0.1).
- `sigma_type` multiplies whichever mode is chosen. Keep its semantics.
- Default: **`shared`** (proposed — confirm at review). CHANGELOG entry saying that the mean fit
  changes and that calibrated force UQ should change little.
- Re-record the pipeline goldens on lestrade (James runs `make_golden.py`; Claude Code cannot).
  Say so in the commit.
- Update `skills/ace-jax/SKILL.md`, the CLI reference and the fit how-to.

Tests (`tests/test_noise_model.py`):
- `per-quantity`: scaling all E weights by a constant leaves the converged MAP coefficients
  unchanged (to 1e-8). This documents the cancellation.
- `shared`: the same scaling changes them.
- `hier`: `τ → 0` reproduces `shared`, and `τ → ∞` reproduces `per-quantity` (to 1e-6).
- `shared` on the GAP-18 Si fixture reproduces ACEfit BLR within the existing julia-parity
  tolerance, if a fixture exists. Otherwise record the numbers in the report.

Acceptance:
- GAP-18 Si, order 4 / degree 12: energy RMSE within 10 % of ACEpotentials BLR (9.9
  meV/atom), force RMSE no worse than per-quantity.
- Cantor rev2 table with `--noise shared --uq ard`: every coverage cell within 0.01 of rev2, or
  better.

## Phase 2 — non-local calibration groups (cheap fix; gated by G1)

Branch `feat/nonlocal-groups`. Linear arm only.

- `fit/nonlocal.py`: smooth, differentiable `g_i` features from D1 (soft `z̃`, soft `d̃`, `s`,
  averaged over `R`). Phase 5 reuses them. Rotation-, translation- and permutation-invariant.
- `--ard-groups distortion+nl`: add one band on a scalar `g` (the D1 winner) to the Mondrian
  groups, with band edges frozen at fit time as for `d`. Recalibration never moves an atom
  across groups (unchanged rule).
- Stored with the posterior. Old posteriors still load and serve.

Tests: invariances of `g` (random rotations, reflections, permutations); group assignment is
identical between fit and serve; `--ard-groups distortion` is bit-identical to today.

Acceptance (Cantor, no target calibration): crack-tip coverage ≥ 0.89; in-distribution 0.90
± 0.01; no other cell of the rev2 table worse by > 0.01.

## Phase 3 — calibrated UQ on the GP arm

Branch `feat/gp-sandwich`. Lift the `uq='ard' ⇒ arm linear` restriction.

- Jackknife sandwich over the joint design `[B | k_θ(B, B_M)]` at fixed θ. Replace the
  diagonal `D` in the PRESS and shape algebra by a general prior square root (`R0` from
  `solve._prior_block` or equivalent). Keep the linear-arm path bit-identical
  (`tests/test_ard*`).
- New shape option `--ard-variance dtc`: the GP predictive covariance per atom, including the
  derivative DTC block (cosine kernel only; raise otherwise). This is the analogue of `kappa`
  that grows away from the inducing set.
- Groups, transfer exponent, conformal quantiles and support flag: unchanged.
- Memory: report the derivative-DTC footprint on 3–4k-atom cells. Respect the existing
  `_warn_deriv_dtc_size` path.

Tests: the PRESS/DFBETA identity against an explicit refit on a tiny GP fit (clusters of every
kind), to machine precision; rotation equivariance of `V`; `M = 0` reduces exactly to the
linear-arm result.

Acceptance: on Cantor, GP arm + sandwich (and + `dtc`) must **match or beat the linear-arm rev2
result** on crack-tip coverage and on Spearman ρ on large cells. If neither shape does,
document the result and keep the option experimental (no default change).

## Phase 4 — PITC training likelihood for the GP arm

Branch `feat/gp-pitc`. `--gp-likelihood {sor,fitc,pitc}`, default `sor` (unchanged).

- `fitc`: add the diagonal of `k − q` per row to the noise variance. For force rows this is
  the diagonal of the derivative DTC block, already computed in `predict.py`.
- `pitc`: block-diagonal `k − q` per block, with the **jackknife clusters** as blocks (spatial
  blocks for big cells, whole small configurations), so block size stays ≲ 10³ rows. Per block
  `Σ_b = diag(σ²/w²) + (K − Q)_b`. The LML uses `Σ_b Φ_bᵀ Σ_b⁻¹ Φ_b` and `Σ_b log|Σ_b|`.
- Consequence to document: `G_BB` becomes θ-dependent, so the cached linear Gram
  (`assemble_statistics`) and host-cache LML no longer apply under `fitc`/`pitc`. Measure the
  cost per LML evaluation on Cantor and report it.

Tests:
- On a tiny Si fixture, with inducing points at every training site, `fitc` and `pitc` equal
  the exact derivative GP LML (a dense reference built in the test) to 1e-8.
- `pitc` with one block per row equals `fitc`.
- LML gradients against finite differences.

Acceptance: on Cantor, learned `σ_F` falls under `pitc` against `sor` (the discrepancy, not
the noise, carries the structured error), test RMSE no worse, and the Phase 3 crack-tip
coverage and ρ no worse. Report fit time; ≤ 2× the `sor` GP fit is the budget.

## Phase 5 — non-local discrepancy kernel (gated by G1, and Phase 3 or 4)

Branch `feat/gp-nonlocal`.

- An additive term `δ_nl(i) = k_nl(g_i, G_M') w'`, `w' ~ N(0, K_M'M'⁻¹)`, with its own
  amplitude `A_nl`, length `ℓ_g`, and species embedding. `g_i` from `fit/nonlocal.py`
  (Phase 2), low-dimensional (2–4 scalars).
- Inducing set: farthest-point sampling on `g` per species, `M'` small (default 32).
- Rows: a second residual block. Force rows by the chain rule through `g`, with edges out to
  `R + r_cut`. Virial rows likewise.
- `R` default: (teacher receptive field − `r_cut`), from 0.1, configurable via `--nl-radius`.
- Variant B, behind `--nl-features hop`, only if variant A passes G1 but leaves a gap:
  `g_i = Σ_j f_c(r_ij; R) P x_j` with `P` the leading `d' ≤ 8` columns of `Pmap`, under the
  existing cosine-SE kernel.
- `GPCalculator` and `aj eval` serve it. Neighbour lists to `R + r_cut`.
- Composes with Phase 3 (sandwich/`dtc` shape) and Phase 4 (`pitc`).

Tests:
- Finite-difference forces and virials of the non-local rows (relative 1e-6).
- Invariances (rotation, reflection, translation, permutation).
- `A_nl → 0` reproduces the local GP fit bit-for-bit.
- `R = r_cut` gives a well-defined fit (a local special case).

Acceptance on Cantor (no target calibration):
- crack-tip coverage ≥ 0.89, in-distribution 0.90 ± 0.01;
- Spearman ρ on large cells above the Phase 3 result;
- tip force RMSE below the local GP's;
- LML improvement over the local GP on the same data;
- fit time ≤ 1.5× the local GP; serve-time memory on 4k-atom cells within an A100-80GB.

## Working rules for Claude Code

- One branch and one PR per phase. Conventional-commit prefixes with a scope (`feat(gp):`,
  `test(noise):`, `docs(dev):`). Squash-merged.
- Follow `CLAUDE.md`: float64 at the top of tests and scripts, plain pytest, `slow` marks for
  heavy tests, no reformatting of unrelated code, Python 3.11 syntax, optional imports inside
  functions.
- Never edit `fixtures/`. Golden re-recording is James's step on lestrade.
- Heavy Cantor runs go through the existing Modal drivers in `bench/defect_uq/modal/`. Ask
  before launching anything that costs more than a few GPU-hours.
- Update user docs (`docs/user/howto/force-uncertainty.md`,
  `docs/user/concepts/force-uncertainty-maths.md`), `skills/ace-jax/SKILL.md` and the CHANGELOG
  in the PR that changes user-facing behaviour.
- At each gate, write the numbers and the decision into the exploration report and stop for
  review.

## Open questions for review

1. Default for `--noise`: `shared` always, or `shared` only when `--weights` is given?
2. Is `hier` worth shipping, or only as a research option?
3. G1 thresholds (coverage spread 0.03, |ρ| 0.15): right sensitivity?
4. Phase 3 naming: keep `--uq ard` for the GP arm (the prior is not ARD there), or introduce
   `--uq sandwich` and alias `ard` to it on the linear arm?
5. PITC blocks = jackknife clusters: acceptable coupling of two otherwise separate concepts?
6. Non-local `R`: tie to the teacher's receptive field automatically when labels come from a
   known foundation model, or always explicit?
