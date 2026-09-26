# Learned radial basis by VarPro (+ MACE-initialised radials) — design

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
- `from_table(r_grid, R_table, transform, envelope, polys, weights)` →
  `Wnlq`: density-weighted least squares of `R_table / envelope` onto `P_q(x(r))`
  per `(zi, zj, n)`. Returns `Wnlq` and the per-radial relative residual.
- Used for (a) converting a Julia-exported **splined** model to the analytic
  branch (sample its splines, project) so an existing fitted ACE can be the
  starting point, and (b) the MACE init (phase 2).

**`fit/radial_learn.py`** (new)
- `with_radial(model, W)` — `eqx.tree_at` replacement of `rnl_Wnlq`; raises
  unless `radial_kind == "analytic"`.
- `pair_density_weights(model, ds, x_grid)` — per-`(zi, zj)` quadrature
  weights on an `x` grid from the dataset's edge-distance histogram (once).
- `normalise(V, wq)` — `W = V / ‖R_n‖_{wq}` per `(zi, zj, n)`: unit norm of
  each radial under the empirical pair density. Fixes the scale gauge that
  would otherwise trade against Γ and the readout. The optimiser works on `V`.
- `roughness(W, D2, wn)` — `Σ_{zi,zj,n} wn[n] · W[zi,zj,n]ᵀ D2 W[zi,zj,n]`,
  `D2[q,q'] = ∫ P_q'' P_q'' dx` (precomputed, `x`-space), `wn[n] = 1/(1+n)²`.
- `projected_residual(W, theta, prob, ds, log_ratios=None)` —
  `yy − bᵀ(G+Λ)⁻¹b` from `combine(theta, linear_statistics(with_radial(...)))`
  and `prior_precision` with M = 0. This equals
  `min_c ‖Φc − y‖²_w + cᵀΛc` exactly, so no Φ is ever materialised.
  Float64; Cholesky of `G+Λ` with the jitter-retry of `solve.py`.
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

Small grid (default 4 values, log-spaced) chosen on the held-out split, not
optimised inside the loop.

### Held-out gate

A private helper in `radial_learn.py` (the removed `select_by_holdout` is not
revived as a shared API). Candidates `{init, learned}` (phase 2 adds `mace`,
`learned_from_mace`). Each is scored by refitting the linear model (M = 0,
θ re-MAP'd) on the fit split and computing weighted E+F RMSE on a **disjoint**
validation split, using the same split construction as the learned-embedding
gate (commit 516e0ba). Ties go to `init`.

### Outputs

`info`: objective trace, θ at each re-profile, λ_r grid scores, gate scores,
selected label, projection residuals (when `from_table` was used). The learned
model is saved as an npz (analytic branch) plus a JSON summary of `info`.

### Entry points

- Library: `learn_radial` as above.
- CLI: a `learn-radial` step in the fit pipeline. **Depends on PR #8**
  (`fit/pipeline/`, CLI restructure); if #8 has not merged when the CLI task
  is reached, the CLI task waits and the library ships first.

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
dominates; learning `pair_Wnlq`.
