# Hyperparameter routing + native POPS UQ + composable weights — design

**Date:** 2026-09-22. **Code:** `~/gits/ace-jax` (`ace_jax.fit`). **Status:** design, awaiting review.

## Motivation

Three threads converge on one abstraction:

1. **POPS UQ.** The linear (BLR, M=0) arm is over-confident under
   misspecification (ACE ≠ MACE). A local spike (`scratchpad/pops_spike/`)
   showed POPS (Perez–Swinburne, `POPSRegression`) calibrates **energy**
   natively but fails on **forces** — because sklearn uses **one global
   noise**, whereas our fit already has per-quantity σ_E/σ_F/σ_V. Whitening
   each row by `w_iq/σ_q` fixes this and lets a native POPS covariance
   calibrate E and F together, with no post-hoc scalar.
2. **Per-config-type precision.** We want per-config-type noise (bulk vs
   defect …). Evidence should pick it — i.e. it is just **more LML hypers**.
3. **Fitting route as a first-class choice.** The embedding is fit by an outer
   **VarOpt** loop (held-out); σ's are fit by the inner **LML/MAP** (evidence).
   Today the embedding is special-cased (`varopt_embed.py`) and `Hypers` is a
   fixed all-LML vector. Generalising the *route* per hyperparameter
   ({fixed, LML, VarOpt}) makes σ_{q,type}, the embedding, and future learnable
   weights all instances of one framework, and removes the special-casing.

## Goals / non-goals

**In scope (this spec):**
- A hyperparameter **routing framework**: heterogeneous param blocks, each
  `route ∈ {fixed, LML, VarOpt}`; inner MAP optimises the LML blocks, outer
  VarOpt optimises the VarOpt blocks; `varopt_embed.py` re-homed into it.
- **σ_{q,type}** per-config-type noise hypers (LML-routed).
- **Native POPS** UQ on the linear arm (whitened by `w_iq/σ_q`; posterior
  **samples** + **hypercube**, default samples; aleatoric optional).
- A **composable weight-factor pipeline** producing `w_iq` (structural +
  per-config-type + per-config), *learnable-ready* but shipped **frozen**.
- `run.py` plumbing for all of the above.

**Designed-for, not implemented here (future consumers of the framework):**
- Learnable weights as **VarOpt-routed** blocks against a **held-out** objective
  (RMSE/CRPS). The framework must accept them with no new machinery; the
  held-out outer objective is specified but only the profiled-LML objective
  (the embedding's) is wired now.

**Out of scope:** the GP (M>0) arm's math (unchanged); the production Cantor fit
itself (separate plan).

## Global constraints
- Backwards compatible: existing `--arm linear`/`--arm gp` runs reproduce
  today's numbers when no new flags are given (routing defaults to all-LML;
  weights default to structural×1; UQ defaults to BLR). The test suite stays
  green.
- fp64. Streamed statistics only — never materialise the full design.
- No new required deps (POPS ported, not imported at runtime).

---

## Component 1 — hyperparameter routing framework

### Param blocks
Replace the flat `Hypers` NamedTuple's role as "the thing both optimisers see"
with a registry of typed blocks. Each block:

```
ParamBlock:
  name:   str                     # "kernel", "sigma_q", "sigma_qtype", "embed", "weights"
  value:  Array                   # log-space where positive (σ, ℓ, A…); shape is the block's
  route:  {FIXED, LML, VAROPT}
  prior:  Prior | None            # required for LML blocks (log-normal, as today)
  anchor: AnchorSpec | None       # for VAROPT blocks (e.g. embedding → block-diagonal)
```

A `ParamSet` holds an ordered list of blocks and provides:
- `lml_vector()` / `set_lml_vector(x)` — flatten/unflatten the LML blocks.
- `varopt_vector()` / `set_varopt_vector(x)` — same for VarOpt blocks.
- `materialise()` → the concrete objects the model/statistics consume
  (a `Hypers`-shaped view for the kernel/σ blocks, an embedding matrix, a
  weight-params object), so downstream code (`stats.py`, `kernels.py`,
  `predict.py`) is unchanged — it receives the same materialised objects.

`Hypers` (the NamedTuple) is retained as the **materialised view** of the
kernel+σ blocks; `from_array`/`to_array` move onto `ParamSet`. `default_prior`
becomes per-block prior construction. Blocks with `route=FIXED` are held
constant by both optimisers.

### Inner MAP (LML route)
`ladder.run_map` today L-BFGS-optimises the whole `Hypers` vector on
`LML + log_prior`. It becomes: optimise `ParamSet.lml_vector()` on
`LML(materialise(ps)) + Σ_block log_prior_block`, holding FIXED and VAROPT
blocks constant. `objective.make_lml` / `log_marginal_likelihood` take the
materialised `Hypers`+embedding as today (no change to the LML math). Laplace /
NUTS / pathfinder / VI operate on the LML vector unchanged.

### Outer VarOpt (VarOpt route)
`varopt.learn` generalises from one block to `ParamSet.varopt_vector()`:
- **Objective (pluggable).** Two provided:
  - `profiled_logpost` — inner-MAP the LML blocks at the current VarOpt value,
    return `LML(θ*) + log_prior(θ*) − anchor(varopt blocks)`. Envelope theorem:
    the outer gradient is the partial w.r.t. the VarOpt blocks at θ* (validated
    for the embedding in `FINDINGS_embed_varopt_spike.md`). **This is what the
    embedding uses and the only objective wired in this spec.**
  - `heldout_score` — inner-MAP on a fit split, score (RMSE or CRPS, incl. the
    POPS predictive) on a disjoint val split. **Specified, not wired** (its
    first consumer is learnable weights, a later phase).
- **Acceptance gate.** `select_by_holdout` retained: keep best of
  {learned, anchor-init, identity} on the held-out split; ties → the simpler.
- **Envelope grad** and low-rank init (SVD from MACE) move from
  `varopt_embed.py` into the framework as the embedding block's methods.

### Re-homing the embedding
`varopt_embed.py` (`theta_map_at`, `embed_objective_and_grad`,
`learn_embedding`) is refactored so the embedding is *"a VAROPT block with a
block-diagonal anchor and an SVD init"*. `learn_embedding` becomes a thin
wrapper that builds a `ParamSet` with the embedding routed to VarOpt and calls
the generic outer loop. Numerical equivalence to the current embedding path is
a required test (bit-for-bit at `steps=0`; within tolerance for a fixed seed).

---

## Component 2 — σ_{q,type} noise hypers (LML)

Add per-config-type noise scales as LML-routed blocks: for config-type `t` and
quantity `q`, effective noise `σ_{q,t} = σ_q · exp(log_σ_type[t,q])`, with
`log_σ_type` an LML block (log-normal prior centred at 0 = "type like the
default"). The default/unlabelled type is pinned to 0 (identifiability: the
global per-quantity scale stays with `σ_q`). Enters `stats.py`/`objective.py`
exactly where `σ_q` does — the type index per observation is already available
(the weight pipeline resolves `config_type`). Absent types → prior-only (no
data → stays at 0).

---

## Component 3 — native POPS UQ (linear arm)

### Whitening
After the inner MAP gives `c*` (linear mean) and `σ_q` (and `σ_{q,t}`), and the
weight pipeline gives `w_iq`, define the **whitened** row and residual:
`φ̃_i = (w_iq / σ_{q,t(i)}) φ_i`, `r̃_i = (w_iq / σ_{q,t(i)}) (y_i − φ_iᵀc*)`.
On the whitened design the noise is homoscedastic (unit), so E and F contribute
at their correct relative scale — the fix for the spike's force failure.

### POPS covariance (ported from `popsregression`)
`pops.py` implements, on the whitened design:
- **Epistemic** `Σ_0 = A⁻¹` where `A = Σ_q G̃_q + Λ` (already the inner-MAP
  precision; `G̃` = whitened Gram, streamed as today).
- **Pointwise corrections** `{δ_i}` from `(Σ_0, φ̃_i, r̃_i)` — the
  parameter shift that optimally fits point `i` (the POPS "pointwise optimal
  parameter set"). Streamed; **leverage-subselected** (`leverage_percentile`,
  the package's knob) so the ensemble is tractable at L≈27k.
- **Two posterior forms** (both implemented):
  - `samples` (**default**): K weight samples `{c* + δ̃_k}` (a committee) — the
    package's `posterior_samples_`. Predict = spread over the committee; drops
    into MD/LAMMPS directly. K≈64; ~14 MB at production scale.
  - `hypercube`: the axis-aligned PCA box (`misspecification_sigma_`), analytic
    variance `φ*ᵀ Σ_POPS φ*`.
- **Predictive** (per quantity, physical units): σ = misspecification spread by
  default; **`--aleatoric`** adds `σ_{q,t}/w*` back for a label-predictive.

### Statistics
POPS needs the per-observation residual after `c*`, so one extra streamed pass
(`pops_statistics(c*, ...)`), analogous to `linear_statistics`; peak memory is
one more (L,L)-scale object during assembly (fits A100 for embedded bases).

### Scope guard
POPS is **linear-arm only** (`--arm linear`). `--arm gp` (joint linear+GP) is
unchanged. UQ path selected by `--uq {blr, pops}` (default `blr` for
backwards-compat; `pops` opt-in).

---

## Component 4 — composable weight pipeline

`weights.py`: a list of **factors**, each `weight(obs, q) -> float`, multiplied
per observation to the `w_E/w_F/w_V` the fitter already consumes (so
`stats.py` is unchanged; absent/padded → 0 = masking, as today).

```
WeightFactor (protocol): weight(config_meta, quantity) -> float
                          learnable: bool = False
                          params()  -> Array | None      # for a future VarOpt block
Built-ins:
  Structural(exp={E:0.5, V:0.5, F:0.0})     # 1/n^exp  (today's 1/√n, configurable)
  Quantity({E,F,V})                          # global per-quantity multiplier
  ConfigType(dict, key="config_type")        # existing per-type (E,F,V)
  PerConfig(key="weight")                    # NEW per-config override (info key)
  Custom(fn)                                 # e.g. force-magnitude / Boltzmann
w_iq = Π_factor factor.weight(obs_i, q)      # resolved once in load_configs
```

Resolution happens in `data.py:load_configs`, replacing the hard-coded
`1/√n × config_type dict` with `compose(factors)`. Default pipeline =
`[Structural(), ConfigType(default={E:1,F:1,V:1})]` — reproduces today.
Factors carry a `learnable` flag + `params()` so a future block can be routed
to VarOpt; **all factors ship frozen** here.

**Relationship to σ_{q,type}:** per-config-type *precision* is now carried by
the LML σ_{q,type} block (evidence-fit); the `ConfigType` weight factor stays
for cases a user wants a *fixed* multiplier, but the recommended path for
per-type behaviour is the σ hypers. Documented in the factor docstring.

---

## Component 5 — `run.py` integration

New flags (all default to today's behaviour):
- `--uq {blr, pops}` (default `blr`); `--pops-posterior {samples, hypercube}`
  (default `samples`); `--pops-leverage-pct <float>`; `--aleatoric`.
- `--weights <json>` — a declarative factor list, e.g.
  `[{"Structural":{}}, {"ConfigType":{"dict":{"defect":{"E":10,"F":10,"V":1}}}},
    {"PerConfig":{"key":"w"}}]`.
- `--sigma-type` — enable the σ_{q,type} LML block (reads `config_type`).
- `--route <json>` (advanced) — override any block's route
  (`{"embed":"varopt","sigma_qtype":"lml"}`); defaults set per feature.

---

## Data flow (linear arm, `--arm linear --uq pops`)

```
load_configs(weights=compose(factors))     -> per-config w_E/F/V (frozen)
build ParamSet: kernel+σ_q+σ_qtype = LML ; embed = VAROPT(if learnt) ; else FIXED
inner run_map (LML blocks)                 -> c*, σ_q, σ_qtype
[outer VarOpt (embed), if routed]          -> profiled-LML + anchor + gate
whiten rows by w_iq/σ_{q,t}
pops_statistics(c*)                        -> pointwise corrections (streamed, leverage-sel)
Σ_POPS: samples (default) | hypercube
predict: mean c*ᵀφ* ; σ = POPS spread (+ aleatoric if flagged)
```

## Testing / validation gate
- **Framework:** routing round-trips (LML/VarOpt vector flatten↔materialise);
  FIXED blocks never move; a run with all-LML routing reproduces current
  `run_map` bit-for-bit.
- **Embedding equivalence:** re-homed embedding == `varopt_embed` at `steps=0`
  (bit-for-bit) and within tol at a fixed seed.
- **σ_{q,type}:** on a 2-type synthetic set, evidence recovers the injected
  per-type noise ratio; default-type pinned.
- **POPS gate (the spike's failure, inverted):** on SiGe (`sige_embed_d16`,
  the spike's split), native POPS (no post-hoc scalar) gives **force** rms-z ∈
  [0.7, 1.5] and cov@90 ≈ 0.9, energy likewise, RMSE unchanged vs BLR, and
  CRPS ≤ BLR+scalar on **both** E and F. (Spike baseline: force rms-z 4.8.)
- **Weights:** factor composition unit tests; `PerConfig` override; a weighted
  fit changes `c*` in the expected direction; empty/absent → mask (weight 0).
- **Backwards-compat:** the existing suite passes unchanged with defaults.

## Migration / risks
- `Hypers` stays as a materialised view → `kernels.py`/`stats.py`/`predict.py`
  need no signature changes; the churn is in `hypers.py` (→ ParamSet),
  `ladder.run_map`, `varopt*.py`, `objective.make_lml` (takes ParamSet).
- **Risk:** POPS pointwise-correction ensemble at L≈27k. Mitigation: leverage
  subselection (default) + the streamed pass; the sandwich/low-rank form is a
  documented fallback if the ensemble is still too large.
- **Risk:** envelope-grad correctness after re-homing. Mitigation: the spike's
  fwd-vs-reverse JVP check is kept as a test on the framework path.

## Scope boundaries (explicit)
- Learnable-weight VarOpt: **designed-for, not built** (frozen factors + the
  `learnable`/`params()` hooks + the `heldout_score` objective stub).
- `heldout_score` outer objective: **specified, not wired** (embedding uses
  `profiled_logpost`).
- POPS on the GP arm: out of scope.
