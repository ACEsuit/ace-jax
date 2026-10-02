# Learned radial basis by VarPro (+ MACE-initialised radials) — design

> **Update 2026-09-28.** Phase 1 (learned radials, held-out gate) merged in #9;
> results in [learn-radial-results.md](../learn-radial-results.md). Phase 2
> (MACE-initialised radials) is not implemented, and `tests/test_mace_radial.py`
> does not exist. The body is the design record.

**Date:** 2026-09-26. **Code:** `ace_jax.fit` / `ace_jax.construct`. **Branch:**
`feat/learn-radial` (off `main` at f33ebbf). **Status:** design, awaiting review.

## Goal

Better accuracy **per basis function** for the linear ACE model by learning the
tensor-basis radial mixing weights `rnl_Wnlq` (analytic branch), with the
linear readout projected out (variable projection, as for the FS density in
`fit/density.py` + `fit/varpro.py`). A MACE foundation model's first-layer
radials are an optional **initial guess** that may make the non-convex outer
problem easier.

Not goals: scaling to many elements (the frozen species embedding already
covers that), transfer/distillation from MACE as an end in itself.

### Success criterion

Per benchmark system (Si, SiGe, CrMnFe), at equal basis size, the held-out
gate (below) is run on `{identity, learned←identity}` and, in phase 2,
`{mace, learned←mace}`. The feature is worth keeping if `learned←*` wins the
gate on held-out E+F RMSE on at least two of the three systems. *(Threshold is
an assumption — confirm at review.)* The removal of the learned species
embedding before #5 merged ("the ablations did not justify it") is the
precedent: the ablation table is part of the deliverable, not an afterthought.

## Decisions (from brainstorming, 2026-09-24/26)

| Question | Decision |
|---|---|
| What is learned | `rnl_Wnlq` only, shape `(NZ, NZ, n_rnl, n_q)`. Agnesi transform `(r0, p, q)` is a possible later phase. Pair basis, envelopes fixed. |
| Radial span | Full (not triangular) `Wnlq` over a richer `n_q ≈ 2–3 × n_max`, so learning can reach functions outside `span{P_0..P_nmax}`. |
| Smoothness prior Γ (#3) | Unchanged, indexed by nominal `(n, l)`. Kept meaningful by a gauge normalisation and a degree-weighted roughness penalty on the radials. |
| Residual GP | Not involved. Radials are learned on the linear model (M = 0); the hybrid fit (GP primarily for UQ) runs afterwards on the frozen learned model. |
| Objective | VarPro projected ridge residual (not the profiled LML). |
| Optimiser | L-BFGS (`optax.lbfgs`, zoom line search). Kaufman Gauss–Newton/LM is a follow-up behind the same interface. |
| Initialisation | `identity` (default) / `glorot` / `from_table` (sampled radials → `Wnlq`); MACE via `from_table` in phase 2. |
| MACE source | mace-torch (not mace-jax), in a separate environment; ace-jax never depends on torch or mace. |

## Relation to existing code

- `ACEModel` already carries a live `rnl_Wnlq` leaf on the analytic branch
  (`radial_kind == "analytic"`, `eval/model.py`).
- `construct.build_model` (#4) authors analytic-branch models; its
  `radial_mode="onehot"` is the identity init. `tensor_radial_init` hard-codes
  `n_q = ceil(1.5 · maxn)` (`construct/radial_init.py:267`).
- `fit.stats.linear_statistics` streams `G_BB, b_B, yy` per observation type
  with a `jax.checkpoint`-ed scan body whose carry is only accumulated.
- `fit.objective.combine` forms the σ-weighted `G, b, yy` (with per-type
  `log_ratios`); `prior_precision` forms `Λ` from Γ and σ_c.
- `fit.ladder.run_map` gives the θ-MAP for an LML closure.
- `fit.varpro` is the precedent: projected residual, autodiff through the
  inner solve, BFGS. It works on an in-memory Φ; here Φ is streamed.
- The #5 `varopt` route and `varopt.py` (`learn`, `select_by_holdout`) were
  **removed before merge** (`ROUTES = ("fixed", "lml")`). This design does not
  depend on them: the radial learner is a standalone pre-training step like the
  density VarPro, and carries its own small held-out gate.

## Phase 1 — learning machinery

### Components

**`construct/radial_init.py`** (extend)
- `n_q_factor` kwarg (default `1.5`, preserving current behaviour) threaded
  through `tensor_radial_init` and `build_model`.
- `from_table(x, R, envelope, polys, weights=None)` → `(Wnlq, relres)`: the
  projector works in the transformed coordinate `x` (a grid on `[-1, 1]`, not
  `r`), with the envelope folded into the basis (`env(x)·P_q(x)`) rather than
  divided out of `R` — density-weighted least squares `min_W Σ_x w (env·P·Wᵀ −
  R)²` per `(zi, zj)`. Returns `Wnlq (NZ, NZ, n_rnl, n_q)` and the per-radial
  relative residual `relres (NZ, NZ, n_rnl)`.
- Used for (a) converting a Julia-exported **splined** model to the analytic
  branch (sample its splines, project) so an existing fitted ACE can be the
  starting point, and (b) the MACE init (phase 2).

**`fit/radial_learn.py`** (new)
- `with_radial(model, W)` — `eqx.tree_at` replacement of `rnl_Wnlq`; raises
  unless `radial_kind == "analytic"`.
- `pair_density_weights(model, ds, x_grid)` — per-`(zi, zj)` quadrature
  weights on an `x` grid from the dataset's edge-distance histogram (once).
- `normalise(V, Q, active)` — `W = V / ‖R_n‖_Q` per `(zi, zj, n)`: unit norm of
  each radial under the empirical pair density (the Gram `Q`, `radial_gram`).
  Fixes the scale gauge that would otherwise trade against Γ and the readout.
  Rows that are zero in `W0` (`row_active`; the onehot init's structural zeros
  for `NZ > 1`) are frozen at zero rather than normalised (0/0 avoided by
  `where`). The optimiser works on `V`.
- `roughness(W, D2, wn)` — `Σ_{zi,zj,n} wn[n] · W[zi,zj,n]ᵀ D2 W[zi,zj,n]`,
  `D2[q,q'] = ∫ P_q'' P_q'' dx` (precomputed, `x`-space), `wn[n] = 1/(1+n)²`.
- `projected_residual(W, theta, prob, ds, log_ratios=None)` —
  `yy − bᵀ(G+Λ)⁻¹b` from `combine(theta, linear_statistics(with_radial(...)))`
  and `prior_precision` with M = 0. This equals
  `min_c ‖Φc − y‖²_w + cᵀΛc` exactly, so no Φ is ever materialised.
  Float64; plain Cholesky of `G+Λ` — **no jitter-retry exists in `solve.py`**;
  a failed Cholesky yields NaN, and `lbfgs_loop` treats that as a stop signal,
  returning the best finite iterate seen so far rather than retrying.
- `learn_radial(prob, ds, W0, *, lam_rough, reprofile_every=10, steps=200,
  tol=1e-6, val=None, seed=0)` → `(W_sel, info)`.

### Data flow

1. Build the model with `build_model(..., radial_mode="onehot", n_q_factor=…)`
   or convert an exported model with `from_table`; take `W0`, `V0 = W0`.
2. θ₀ = `run_map` on the M = 0 LML at `normalise(V0)`.
3. L-BFGS on `f(V) = projected_residual(normalise(V); θ) + λ_r · roughness(normalise(V))`.
4. Every `reprofile_every` steps: re-run `run_map` on the current Gram (no
   restream — the M = 0 LML depends on θ only through `combine`/`Λ`), then
   **reset the L-BFGS state** (the objective changed).
5. Stop at `steps`, or when the relative change in `f` is < `tol` for 3
   consecutive steps, or on line-search failure (logged; best iterate kept).
6. Held-out gate (below); freeze; save via `construct.export.save_npz`.
7. The ordinary hybrid fit (linear + residual GP) runs on the saved model.

`steps=0` returns `normalise(V0)` unchanged with `info["selected"] = "init"`.

### Gradient and memory

`jax.grad` of `f` through `linear_statistics`. Because the scan carry is only
added to and the body is checkpointed, reverse mode should keep only the
per-batch inputs, not `n_batches` copies of `L × L`. **This is an assumption
to verify first** (plan task 1): measure peak memory of value-and-grad vs value
at realistic `L` on moriarty.

Fallback if it fails: a manual two-pass streamed adjoint — pass 1 streams
`G, b, yy`; `Ḡ = ∂f/∂G`, `b̄`, `ȳy` in closed form; pass 2 scans
`jax.vjp(batch_linear_stats)` with that cotangent, accumulating `∂f/∂W`.

### λ_r selection

`lam_rough` is RELATIVE, not an absolute penalty weight: `λ = lam_rough ·
r(W0) / roughness(W0)`, where `r(W0)` is the projected residual and
`roughness(W0)` the roughness of the (normalised) init, so a given
`lam_rough` means roughly the same trade-off regardless of the model's
absolute scale. Default grid `(0, 1e-3, 1e-2, 1e-1)`, chosen on the held-out
split, not optimised inside the loop.

### Held-out gate

A private helper in `radial_learn.py` (the removed `select_by_holdout` is not
revived as a shared API). Candidates `{init, learned}` (phase 2 adds `mace`,
`learned_from_mace`). Each candidate is refitted (readout only, M = 0) at its
OWN θ-MAP on the fit split — warm-started from the init's θ-MAP, not
recomputed from scratch — and scored by weighted E+F SSE on a **disjoint**
validation split, using the same split construction as the learned-embedding
gate (commit 516e0ba). Scores are normalised by the INIT's σ (`a_norm = a0`
for every candidate, fixed across the grid), not each candidate's own
θ-MAP σ, so scores stay comparable across candidates whose θ-MAP can drift.
Ties go to `init`.

### Outputs

`info`: objective trace, θ at each re-profile, λ_r grid scores, gate scores,
selected label, projection residuals (when `from_table` was used). The learned
model is saved as an npz (analytic branch) plus a JSON summary of `info`.

### Implementation notes (added during execution)

- `fit/radial_model.py:to_analytic(model, n_q, n_x=2001) -> (model, relres)`
  converts a splined model to the analytic branch (`from_table` sampled on a
  uniform `x`-grid) or, if already analytic, just widens it (`widen_radial`).
  It is EXACT only for `ace_model`-family splines (Legendre-recursion
  polynomials in the agnesi `x` of `ace_model`): an ACE1 spline is not
  polynomial in that `x`, so its projection has residual, not roundoff, error
  — measured max relative radial error 9.2e-5 at `n_q=30` on the committed
  ACE1 fixture, converging as `n_q` grows. `bench/learn_radial/run.py` records
  the run's `to_analytic_relres_max` in `radial_info.json` so this is visible
  per benchmark, not just asserted here.
- The test suite's in-memory VarPro reference (`_in_memory_residual` in
  `tests/test_gp_learn_radial.py`) uses reduced QR (`jnp.linalg.qr`), not
  `jnp.linalg.lstsq`: `lstsq` has no custom VJP, so its gradient is plain
  autodiff through SVD, whose singular-VECTOR gradient is NaN on the
  degenerate ACE design (repeated singular values). Reduced QR is
  differentiable for a full-column-rank design, and the prior-augmented rows
  make that design full rank.
- The L-BFGS step (`radial_learn._lbfgs_step`) is one module-level
  `jax.jit`-ed function with the objective `f` and `cfg` (e.g. `prob.cfg`)
  STATIC and `x`/`state`/`args` (θ, model, data, ...) traced; see
  `lbfgs_loop`'s docstring. One compile serves every θ-reprofiling round
  within a `learn_radial` call, since only the VALUE of `args` changes
  between rounds, not its shape/dtype. `info["round_lengths"]` records the
  trace length of each round, so the flat `info["trace"]` can be split back
  into per-round segments.
- The Task 6 gradient-memory gate (`test_gradient_memory_does_not_scale_with_batches`)
  passed on CPU: temp bytes `grad(2 batches) = grad(6 batches) = 59724808`,
  `value(6) = 13905096` — gradient memory does not grow with the number of
  streamed batches, and is in fact identical at 2 and 6 batches on this
  fixture, well inside the manual two-pass-adjoint fallback's motivating
  concern. No two-pass adjoint was needed. This was measured on CPU at fixture
  scale; the production-size measurement (moriarty, Cantor, `L=1950`, 38
  batches of 4, `n_q=30`) confirms it at scale: `value(all batches) =
  1147.0 MB`, `grad(2 batches) = 1333.1 MB`, `grad(all batches) = 1333.6 MB`
  — a grad/value ratio of **1.16×**, within the spec's <3× bar and, as on
  CPU, independent of batch count. See `docs/dev/learn-radial-results.md`.

### Entry points

- Library: `learn_radial`/`fit_radial`/`save_result` as above.
- Bench driver: `bench/learn_radial/run.py` — a standalone CLI (`--model`,
  `--data`, `--out`, plus the usual hyperparameters) that loads a model and
  dataset, converts/widens to the analytic branch, runs `fit_radial`, and
  writes `model.npz`/`rnl_Wnlq.npy`/`radial_info.json`/`summary.json`. This is
  the library's only CLI entry point for now.
- CLI: a `learn-radial` step in the fit pipeline still **depends on PR #8**
  (`fit/pipeline/`, CLI restructure) and has not landed; `bench/learn_radial/run.py`
  is the interim entry point until #8 merges.

## Phase 2 — MACE initialisation

### Extractor: `scripts/extract_mace_radial.py`

Runs in its own environment (`uv run --with mace-torch --with torch …`); never
imported by ace-jax. Pins a recent `mace-torch` floor in the script header.
(The local `~/gits/mace` checkout is v0.1.0 from 2022 with a different radial
API — not used.)

- Input: a foundation name (`mace_mp:medium`, …) or a `.model` path; loaded on
  CPU, float64, **cuequivariance disabled** (plain e3nn weight layout).
- On a dense `r` grid over `(0, r_max]`: `radial_embedding` →
  `interactions[0].conv_tp_weights`, applying the cutoff exactly as that
  version's `forward` does (newer: `(edge_feats, cutoff)` with the cutoff on
  the MLP output; older: cutoff folded into the edge features — detected by
  version).
- `h = interactions[0].linear_up(node_embedding(one_hot(Z)))` for requested Z.
- Reshape the flat `weight_numel` to `R[l, k, r]` using `conv_tp.instructions`
  (path → `l_sh`, `uvu` weight slices) — never by assumed ordering.
- **Self-check (hard error on failure):** rebuild the first-layer message for
  one small structure from the extracted `R`, `h` and compare with a forward
  hook on `interactions[0]` to 1e-6.
- Output `.npz`: `r`, `R[l,k,r]`, `h[Z,k]`, `Z`, `r_max`, model name/hash,
  mace-torch version.

### Loader/projector: `construct/mace_radial.py` (pure numpy)

1. For each neighbour species `zj` and each `l`: `F_k(r) = R_{k,l}(r) · h_{zj,k}`.
2. Density-weighted SVD over `k` (same pair density as the gauge); keep the top
   `n_max(l) + 1` modes.
3. Envelope reconciliation: on `r < min(rcut, r_max)` divide out the ACE
   envelope where it is non-negligible; **flag** pairs with `rcut > r_max`
   (radials are zero beyond `r_max`; learning must supply that range).
4. Order modes by roughness `∫|R''|²` so `n` runs smooth → rough and Γ's
   degree ordering roughly holds.
5. `from_table` → `Wnlq[zi, zj, n, :]`, broadcast over `zi` (MACE's first layer
   is `zi`-independent; learning may break the symmetry).
6. Report the projection residual per `(zj, l)`.

### Experiment

Four-way gate `{identity, learned←identity, mace, learned←mace}` on Si, SiGe,
CrMnFe; results table in `docs/` alongside the PR.

## Testing

`tests/test_gp_learn_radial.py`, `tests/test_mace_radial.py` (TDD).

- `with_radial` + identity init reproduces the #4 onehot model's descriptors
  bit-for-bit.
- `from_table` round-trip: sample a known analytic model's radials → recover
  `Wnlq` to 1e-10; spline→analytic conversion of a committed fixture matches
  descriptors to 1e-6.
- `projected_residual` equals `varpro.projected_residual` / `inner_solve` on a
  small in-memory Φ.
- Streamed `∂f/∂V` matches finite differences and autodiff through the
  in-memory `varpro.projected_residual` on a tiny Si fixture.
- Gauge invariance `f(sV) = f(V)`; `roughness` matches numerical quadrature.
- Recovery: data from a model with perturbed `Wnlq`, learning from identity
  lowers `f` and held-out RMSE; the gate selects `learned`. `steps=0` returns
  the init.
- Slow (moriarty): peak memory of value-and-grad < 3× value at realistic `L`.
- MACE: loader/projector on a synthetic npz (no torch needed); extractor
  self-check run manually on moriarty in the torch env, output recorded in the
  PR.

## Open PRs at time of writing

- **#7** (fast GPU forces; `ACEModel` → `EdgeSiteModel` subclass, radial
  refactor in `eval/model.py`). Keeps `rnl_Wnlq`; this design does not edit
  `eval/model.py`, so no textual conflict expected. Rebase after it merges and
  re-run the identity-descriptor test.
- **#8** (fit pipeline, multistart MAP, CLI). Touches `cli.py`,
  `bench/acegp_cantor/run.py`, `fit/predict.py`. Blocks only the CLI task.

## Out of scope

Agnesi transform learning (a later phase, same objective); joint learning
with the residual GP; Kaufman Gauss–Newton/LM; caching the raw-polynomial
atomic basis `Â` (A is linear in W) — only if profiling shows the restream
dominates; learning `pair_Wnlq`; per-config-type noise ratios (`log_ratios`)
in the radial objective (`projected_residual`/`combine` accept them, but the
radial learner never passes any).
