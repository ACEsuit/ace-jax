# Hyper-routing + POPS UQ + composable weights — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every fit hyperparameter a fitting *route* ({fixed, LML, VarOpt}), add per-config-type σ hypers, a native POPS misspecification-UQ path on the linear arm, and a composable weight-factor pipeline.

**Architecture:** A `ParamSet` of typed blocks replaces the flat `Hypers` vector as the object both optimisers see; it *materialises* to the existing `Hypers`/embedding objects so `stats/kernels/predict` are untouched. The inner MAP optimises the LML blocks, the outer VarOpt the VarOpt blocks (the embedding re-homes into it). POPS runs on the linear arm over the σ-whitened design. Weights compose from small factor objects.

**Tech Stack:** Python 3.11+, JAX (fp64), numpyro/optax (ladder), pytest.

**Spec:** `docs/specs/2026-09-22-hyper-routing-pops-weights-design.md` (read it alongside this plan).

## Global Constraints
- **Backwards compatible:** with no new flags, `--arm linear`/`--arm gp` reproduce today's numbers; the existing suite stays green. Defaults: all blocks `route=LML`; weights `[Structural(), ConfigType(default={E:1,F:1,V:1})]`; `--uq blr`.
- **fp64**, streamed statistics only — never materialise the full design.
- **No new runtime deps** — POPS is *ported*, never imported at runtime.
- Code lives in `src/ace_jax/fit/`; tests in `tests/` named `test_gp_*.py`; use existing fixtures in `tests/conftest.py`.
- Every task ends green and is committed.

---

## File structure

| File | Responsibility |
|---|---|
| `src/ace_jax/fit/paramset.py` (new) | `ParamBlock`, `ParamSet`: routing, flatten/unflatten per route, `materialise()` |
| `src/ace_jax/fit/hypers.py` (mod) | keep `Hypers` as materialised view; add `sigma_type` field; per-block priors |
| `src/ace_jax/fit/ladder.py` (mod) | `run_map` takes a `ParamSet` (optimises LML blocks) |
| `src/ace_jax/fit/objective.py` (mod) | `make_lml` accepts materialised `Hypers`+embedding from a `ParamSet`; σ_type into `prior_precision`/noise |
| `src/ace_jax/fit/varopt.py` (mod) | generic outer loop over `ParamSet` VarOpt blocks + pluggable objective |
| `src/ace_jax/fit/varopt_embed.py` (mod) | embedding becomes a VarOpt block; `learn_embedding` a thin wrapper |
| `src/ace_jax/fit/pops.py` (new) | whitening, pointwise corrections, sample + hypercube posteriors, predictive var |
| `src/ace_jax/fit/weights.py` (new) | `WeightFactor` protocol + `Structural/Quantity/ConfigType/PerConfig/Custom` + `compose` |
| `src/ace_jax/fit/data.py` (mod) | `load_configs` uses `compose(factors)` |
| `bench/acegp_cantor/run.py` (mod) | `--uq/--pops-*/--aleatoric/--weights/--sigma-type/--route` |

---

# PHASE 1 — hyper-routing framework + re-home embedding

### Task 1: `ParamBlock` / `ParamSet` core

**Files:**
- Create: `src/ace_jax/fit/paramset.py`
- Test: `tests/test_gp_paramset.py`

**Interfaces:**
- Produces: `ParamBlock(name:str, value:jnp.ndarray, route:str, prior=None, anchor=None)` with `route ∈ {"fixed","lml","varopt"}`; `ParamSet(blocks:tuple[ParamBlock,...])` with `.lml_vector()`, `.set_lml_vector(x)->ParamSet`, `.varopt_vector()`, `.set_varopt_vector(x)->ParamSet`, `.block(name)->ParamBlock`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_gp_paramset.py
import jax.numpy as jnp
from ace_jax.fit.paramset import ParamBlock, ParamSet

def _ps():
    return ParamSet((
        ParamBlock("a", jnp.array([1.0, 2.0]), "lml"),
        ParamBlock("E", jnp.zeros((2, 2)), "varopt"),
        ParamBlock("k", jnp.array([9.0]), "fixed"),
    ))

def test_lml_vector_roundtrip_only_lml_blocks():
    ps = _ps()
    assert ps.lml_vector().shape == (2,)                 # only "a"
    ps2 = ps.set_lml_vector(jnp.array([3.0, 4.0]))
    assert jnp.allclose(ps2.block("a").value, jnp.array([3.0, 4.0]))
    assert jnp.allclose(ps2.block("k").value, jnp.array([9.0]))   # fixed untouched
    assert jnp.allclose(ps2.block("E").value, jnp.zeros((2, 2)))  # varopt untouched

def test_varopt_vector_roundtrip_flattens_matrix():
    ps = _ps()
    assert ps.varopt_vector().shape == (4,)              # 2x2 flattened
    ps2 = ps.set_varopt_vector(jnp.arange(4.0))
    assert jnp.allclose(ps2.block("E").value, jnp.arange(4.0).reshape(2, 2))
    assert jnp.allclose(ps2.block("a").value, jnp.array([1.0, 2.0]))  # lml untouched
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_gp_paramset.py -v` → FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**
```python
# src/ace_jax/fit/paramset.py
"""Typed hyperparameter blocks with a fitting route. The inner MAP optimises
the LML blocks, the outer VarOpt the VarOpt blocks; FIXED blocks never move.
Both optimisers see a flat vector of their route's blocks; `materialise` turns
the whole set back into the concrete objects stats/kernels/predict consume."""
from typing import NamedTuple, Optional
import jax.numpy as jnp

ROUTES = ("fixed", "lml", "varopt")

class ParamBlock(NamedTuple):
    name: str
    value: jnp.ndarray
    route: str
    prior: Optional[object] = None    # (mu, sigma) log-normal for LML blocks
    anchor: Optional[object] = None   # AnchorSpec for VarOpt blocks

class ParamSet(NamedTuple):
    blocks: tuple

    def block(self, name):
        return next(b for b in self.blocks if b.name == name)

    def _vec(self, route):
        parts = [b.value.reshape(-1) for b in self.blocks if b.route == route]
        return jnp.concatenate(parts) if parts else jnp.zeros((0,))

    def _set(self, route, x):
        out, i = [], 0
        for b in self.blocks:
            if b.route == route:
                n = b.value.size
                out.append(b._replace(value=x[i:i + n].reshape(b.value.shape)))
                i += n
            else:
                out.append(b)
        return ParamSet(tuple(out))

    def lml_vector(self):        return self._vec("lml")
    def set_lml_vector(self, x): return self._set("lml", x)
    def varopt_vector(self):     return self._vec("varopt")
    def set_varopt_vector(self, x): return self._set("varopt", x)
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_gp_paramset.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add src/ace_jax/fit/paramset.py tests/test_gp_paramset.py && git commit -m "feat(fit): ParamBlock/ParamSet routing core"`

---

### Task 2: `materialise` → Hypers + embedding view

**Files:**
- Modify: `src/ace_jax/fit/paramset.py` (add `materialise`), `src/ace_jax/fit/hypers.py` (build a `ParamSet` from a `Hypers`+prior)
- Test: `tests/test_gp_paramset.py` (extend)

**Interfaces:**
- Consumes: `Hypers` NamedTuple (10 fields incl. `log_sigma_E/F/V`), `Prior`, `default_prior(r0)`, `to_array/from_array` (`hypers.py`).
- Produces: `paramset.from_hypers(hypers, prior, embed=None) -> ParamSet` (kernel+σ = one `"hypers"` LML block; `"embed"` = FIXED unless routed); `ParamSet.materialise() -> (Hypers, embed_or_None)`.

- [ ] **Step 1: Write the failing test**
```python
def test_materialise_roundtrips_hypers():
    from ace_jax.fit.hypers import default_prior, to_array, from_array
    from ace_jax.fit.paramset import from_hypers
    pr = default_prior(2.5); h = from_array(to_array(pr.mu))
    ps = from_hypers(h, pr)
    h2, embed = ps.materialise()
    assert embed is None
    assert jnp.allclose(to_array(h2), to_array(h))
    # optimiser sees exactly the 10 hyper values as the lml vector
    assert jnp.allclose(ps.lml_vector(), to_array(h))
```

- [ ] **Step 2: Run** → FAIL (`from_hypers` missing).

- [ ] **Step 3: Implement** — in `paramset.py`:
```python
def from_hypers(hypers, prior, embed=None, hypers_route="lml", embed_route="fixed"):
    from .hypers import to_array
    blocks = [ParamBlock("hypers", to_array(hypers), hypers_route, prior=prior)]
    if embed is not None:
        blocks.append(ParamBlock("embed", jnp.asarray(embed), embed_route))
    return ParamSet(tuple(blocks))

def materialise(self):
    from .hypers import from_array
    h = from_array(self.block("hypers").value)
    embed = None
    try:
        embed = self.block("embed").value
    except StopIteration:
        pass
    return h, embed
ParamSet.materialise = materialise
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(fit): materialise ParamSet -> Hypers+embed"`

---

### Task 3: `run_map` optimises a `ParamSet`'s LML blocks

**Files:**
- Modify: `src/ace_jax/fit/ladder.py:64` (`run_map`), `src/ace_jax/fit/objective.py` (`make_lml` accepts a materialiser)
- Test: `tests/test_gp_ladder.py` (extend)

**Interfaces:**
- Consumes: `run_map(lml, prior, *, steps, lr, seed, init)` (current: lml/prior are fns of a flat array); `make_lml(prob, ds)` returns `lml(theta_array)`.
- Produces: `run_map_ps(ps, make_lml_of_hypers, ds, *, steps, lr, seed) -> ParamSet` — optimises only `ps.lml_vector()` on `LML(materialise) + log_prior`, returns the updated `ParamSet`. `make_lml` unchanged; a wrapper builds `lml(x) = make_lml(prob_with(ps.set_lml_vector(x)), ds)`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_gp_ladder.py  (add)
def test_run_map_ps_matches_flat_run_map(tiny_linear_problem):
    # tiny_linear_problem: existing fixture giving (prob, ds) at M=0. If absent,
    # build from tests/conftest.py helpers (see test_gp_objective.py).
    from ace_jax.fit.paramset import from_hypers
    from ace_jax.fit.ladder import run_map, run_map_ps
    from ace_jax.fit.objective import make_lml
    prob, ds = tiny_linear_problem
    lml = make_lml(prob, ds)
    flat = run_map(lml, prob.prior, steps=20, seed=0)          # existing path
    ps = from_hypers(prob.prior.mu, prob.prior)               # all-LML default
    ps_out = run_map_ps(ps, prob, ds, steps=20, seed=0)
    import jax.numpy as jnp
    assert jnp.allclose(ps_out.lml_vector(), flat, atol=1e-6)  # same optimum
```

- [ ] **Step 2: Run** → FAIL (`run_map_ps` missing).

- [ ] **Step 3: Implement** — add to `ladder.py` (reuse the existing `run_map` L-BFGS body; the executor reads `run_map` and wraps it):
```python
def run_map_ps(ps, prob, ds, *, steps=500, lr=0.02, seed=0):
    from .objective import make_lml
    def lml_of_x(x):
        ps_x = ps.set_lml_vector(x)
        h, embed = ps_x.materialise()
        prob_x = prob._replace(hypers=h) if embed is None else \
                 prob._replace(hypers=h, ind=prob.ind._replace(embed=embed))
        return make_lml(prob_x, ds)(to_array(h))     # or make_lml built once; see note
    from .hypers import to_array
    x0 = ps.lml_vector()
    x_star = run_map(lml_of_x, ps.block("hypers").prior, steps=steps, lr=lr, seed=seed, init=x0)
    return ps.set_lml_vector(x_star)
```
Note for the executor: `make_lml` currently caches linear stats over `ds`; build it **once** outside `lml_of_x` and pass the array through, mirroring `make_lml_embed` in `objective.py:128`. Match that pattern exactly (it already threads a mutated embed via `prob._replace(ind=...)`).

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(fit): run_map over a ParamSet's LML blocks"`

---

### Task 4: generic outer VarOpt over a `ParamSet`

**Files:**
- Modify: `src/ace_jax/fit/varopt.py`
- Test: `tests/test_gp_learn_embed.py` (extend), `tests/test_gp_paramset.py`

**Interfaces:**
- Consumes: `learn(psi0, objective, grad, *, steps, lr)`, `select_by_holdout(candidates, score)` (`varopt.py`); the envelope-grad pattern from `varopt_embed.embed_objective_and_grad`.
- Produces: `varopt_ps(ps, objective_and_grad, *, steps, lr, val_score=None) -> ParamSet` — optimises `ps.varopt_vector()` via `learn`, re-materialising each step; `objective_and_grad(x) -> (value, grad)` closes over the inner MAP; returns updated `ParamSet` (held-out gate applied if `val_score` given).

- [ ] **Step 1: Write the failing test**
```python
def test_varopt_ps_noop_when_no_varopt_blocks(tiny_linear_problem):
    from ace_jax.fit.paramset import from_hypers
    from ace_jax.fit.varopt import varopt_ps
    prob, ds = tiny_linear_problem
    ps = from_hypers(prob.prior.mu, prob.prior)              # no varopt blocks
    out = varopt_ps(ps, lambda x: (0.0, x * 0.0), steps=3)
    import jax.numpy as jnp
    assert jnp.allclose(out.varopt_vector(), jnp.zeros((0,)))  # nothing to do
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**
```python
def varopt_ps(ps, objective_and_grad, *, steps=30, lr=0.05, val_score=None):
    x0 = ps.varopt_vector()
    if x0.size == 0 or steps == 0:
        return ps
    obj = lambda x: objective_and_grad(x)[0]
    grad = lambda x: objective_and_grad(x)[1]
    x_star = learn(x0, obj, grad, steps=steps, lr=lr)
    cands = {"learned": x_star, "init": x0}
    best = select_by_holdout(cands, val_score) if val_score else x_star
    return ps.set_varopt_vector(best if not isinstance(best, str) else cands[best])
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(fit): generic VarOpt over ParamSet blocks"`

---

### Task 5: re-home the embedding as a VarOpt block

**Files:**
- Modify: `src/ace_jax/fit/varopt_embed.py` (`learn_embedding` → thin wrapper), `bench/acegp_cantor/run.py` (embedding path builds a `ParamSet`)
- Test: `tests/test_gp_learn_embed.py`

**Interfaces:**
- Consumes: `varopt_ps` (Task 4), `run_map_ps` (Task 3), `from_hypers` with `embed_route="varopt"`, existing `embed_objective_and_grad(prob, ds, lam)`.
- Produces: `learn_embedding(prob, ds, E0, *, lam, steps, lr, inner_steps, val_score, seed)` unchanged signature, now implemented via the framework; returns the same `(embed, accepted_tag)` it does today.

- [ ] **Step 1: Write the failing test** — reuse the existing `test_learn_embedding_steps0_returns_init` (must still pass) and add:
```python
def test_learn_embedding_uses_paramset_path(sige_embed_problem):
    from ace_jax.fit.varopt_embed import learn_embedding
    prob, ds, E0 = sige_embed_problem
    embed, tag = learn_embedding(prob, ds, E0, steps=0)   # steps=0 -> normalize_rows(E0)
    import jax.numpy as jnp
    from ace_jax.fit.embedding import normalize_rows
    assert jnp.allclose(embed, normalize_rows(E0))
```

- [ ] **Step 2: Run** → the steps=0 test must stay green; the new one FAILs only if you broke the return contract.
- [ ] **Step 3: Implement** — rewrite `learn_embedding` body to: build `ps = from_hypers(prob.prior.mu, prob.prior, embed=E0, embed_route="varopt")`; each outer step calls `run_map_ps` (inner θ) then `embed_objective_and_grad`'s value/grad (envelope); wrap with `varopt_ps(..., val_score=...)`; materialise → `(embed, tag)`. Keep `steps=0` returning `normalize_rows(E0)` exactly (guard before the loop).
- [ ] **Step 4: Run** — `pytest tests/test_gp_learn_embed.py -v` → PASS (both).
- [ ] **Step 5: Commit** — `git commit -m "refactor(fit): embedding VarOpt re-homed onto ParamSet"`

**Phase-1 gate:** `pytest tests/ -k "paramset or ladder or learn_embed or objective" -q` all green; the full suite unchanged.

---

# PHASE 2 — σ_{q,type} LML hypers

### Task 6: σ_type block + noise assembly

**Files:**
- Modify: `src/ace_jax/fit/hypers.py` (add `sigma_type` handling), `src/ace_jax/fit/objective.py` (per-type noise in the LML), `src/ace_jax/fit/data.py` (carry a per-observation type index)
- Test: `tests/test_gp_hypers.py`

**Interfaces:**
- Produces: an LML `ParamBlock("sigma_type", value=log_ratios (n_types,3), "lml", prior=...)`; effective noise for obs of quantity `q`, type `t`: `σ_{q,t} = σ_q · exp(log_ratios[t,q])`, with the default type row pinned to 0. `Config` gains `type_idx:int`; `Dataset`/batch carry `type_idx` per config.

- [ ] **Step 1: Write the failing test**
```python
def test_sigma_type_recovers_injected_ratio(two_type_synthetic):
    # fixture: 2 config-types, type-1 energies noisier by 3x (build in conftest).
    from ace_jax.fit.fit_api import fit_linear_with_sigma_type   # thin helper you add
    ratios = fit_linear_with_sigma_type(two_type_synthetic)      # returns exp(log_ratios)[:,0]
    assert 2.0 < ratios[1] / ratios[0] < 4.5      # ~3x recovered by evidence
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — add `type_idx` to `Config`/batch in `data.py` (default 0, from `weight_key` type ordering); in `objective.py` where per-quantity `σ_q` scales each type's `Stats` block, multiply by `exp(log_ratios[type, q])` per observation (the stats are already per-type via weights — extend `_type_stats` accumulation keyed by type, OR scale the noise per type-block). Add the block via `from_hypers`. Pin default row: exclude it from the LML vector (fix that entry) or add a strong zero-prior.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(fit): per-config-type sigma hypers (LML-routed)"`

**Phase-2 gate:** suite green; a run without `--sigma-type` is bit-identical to Phase 1.

---

# PHASE 3 — native POPS on the linear arm

### Task 7: whitening helper

**Files:**
- Create: `src/ace_jax/fit/pops.py`
- Test: `tests/test_gp_pops.py`

**Interfaces:**
- Produces: `whiten(phi, resid, w, sigma) -> (phi_t, resid_t)` = `(w/sigma) * phi`, `(w/sigma) * resid` (broadcast over rows); `whiten_rows(Rows, w_E/F/V, sigma_E/F/V) -> (Phi_tilde, r_tilde)` stacking E/F/V whitened rows.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_gp_pops.py
import jax.numpy as jnp
from ace_jax.fit.pops import whiten
def test_whiten_scales_rows_by_w_over_sigma():
    phi = jnp.ones((3, 2)); r = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([2.0, 2.0, 2.0]); sigma = 4.0
    pt, rt = whiten(phi, r, w, sigma)
    assert jnp.allclose(pt, 0.5 * phi) and jnp.allclose(rt, 0.5 * r)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**
```python
"""Native POPS misspecification UQ for the linear (M=0) arm, on the design
whitened per quantity by w/sigma_q so E and F share one homoscedastic scale."""
import jax.numpy as jnp
def whiten(phi, resid, w, sigma):
    s = (w / sigma)[:, None]
    return phi * s, resid * (w / sigma)
```

- [ ] **Step 2b–5:** run → PASS; commit `feat(fit): pops whitening`.

### Task 8: pointwise corrections + posterior (samples + hypercube)

**Files:**
- Modify: `src/ace_jax/fit/pops.py`
- Test: `tests/test_gp_pops.py`

**Interfaces:**
- Consumes: `Sigma0 (L,L)` (= `A⁻¹`, the inner-MAP precision inverse, available from `objective.posterior`), whitened `(phi_t, r_t)` streamed, `leverage_pct`.
- Produces: `pops_corrections(Sigma0, phi_t, r_t, leverage_pct) -> deltas (K,L)`; `pops_posterior(deltas, c_star, form) -> {"samples": (K,L)} | {"cov": (L,L)}`; `pops_var(phi_star, posterior) -> var (n,)`.

- [ ] **Step 1: Write the failing test** — synthetic misspecified linear (a quadratic target fit by a line): POPS var must EXCEED the epistemic `phi Sigma0 phi` on held-out points, and `pops_var` from samples ≈ from cov for a Gaussian-ish ensemble:
```python
def test_pops_var_exceeds_epistemic_under_misspecification(pops_synth):
    from ace_jax.fit.pops import pops_corrections, pops_posterior, pops_var
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    post = pops_posterior(d, c, form="samples")
    v = pops_var(phi_star, post)
    assert jnp.all(v >= 0.5 * epi_var)            # misspecification inflates
    assert v.mean() > epi_var.mean()
```

- [ ] **Step 2–5:** implement `pops_corrections` (δ_i = Sigma0 @ phi_t_i * r_t_i, leverage = diag(phi_t Sigma0 phi_tᵀ), keep top pct), `pops_posterior` (`samples`: `c + deltas`; `hypercube`: PCA box → cov), `pops_var` (samples: `var over c-samples of phi·c`; cov: `sum(phi @ cov * phi,1)`), port faithfully from `popsregression` `_build_posterior/_fit_hypercube/_sample_hypercube` (read the package source; keep both forms). Commit `feat(fit): POPS pointwise corrections + posteriors`.

### Task 9: streamed `pops_statistics` over the dataset

**Files:** Modify `src/ace_jax/fit/stats.py` (add `pops_statistics(c_star, model, cfg, ds, w, sigma) -> deltas`), `tests/test_gp_stats.py`.
- **Interfaces:** streams batches like `linear_statistics`, whitens each batch's `Rows` (E/F/V) with Task 7, accumulates leverage + corrections (Task 8). One extra pass; peak = one `(L,L)`-scale accumulator.
- [ ] Test: a 2-batch dataset gives the same `deltas` as the single-batch whitened design (streaming == monolithic). Implement mirroring `linear_statistics` (`jax.lax.scan`). Commit `feat(fit): streamed POPS statistics`.

### Task 10: predict + run.py `--uq pops`

**Files:** Modify `src/ace_jax/fit/predict.py` (a POPS variance branch), `bench/acegp_cantor/run.py` (`--uq {blr,pops}`, `--pops-posterior`, `--pops-leverage-pct`, `--aleatoric`), `tests/test_gp_predict.py`.
- **Interfaces:** `predict_fixed(..., uq="blr")` gains `uq="pops"`: mean = `c*ᵀφ*` (unchanged), var from `pops_var`; `+ (sigma_{q,t}/w*)²` iff `aleatoric`.
- [ ] Test: `uq="pops"` returns finite per-config E and per-component F σ with the right shapes; `aleatoric=True` strictly increases σ. Commit `feat(fit,cli): POPS predictive path + flags`.

### Task 11: SiGe calibration gate (the spike, inverted)

**Files:** `tests/test_gp_pops.py` (slow, marked).
- [ ] Test (uses `sige_embed_d16.npz` + `sige_mh1.xyz` via the run.py split r0=2.35 ntrain=200 test-start=250 ntest=100 seed=0): native POPS (no post-hoc scalar) gives **force** rms-z ∈ [0.7,1.5], cov@90 ∈ [0.85,0.95]; **energy** likewise; RMSE within 1% of BLR; CRPS ≤ BLR+scalar on both. Mark `@pytest.mark.slow`. This is the acceptance gate for Phase 3. Commit `test(fit): POPS SiGe per-quantity calibration gate`.

**Phase-3 gate:** Task 11 passes; `--uq blr` unchanged.

---

# PHASE 4 — composable weight pipeline

### Task 12: `WeightFactor` + built-ins + `compose`

**Files:** Create `src/ace_jax/fit/weights.py`; `tests/test_gp_weights.py`.
- **Interfaces:** `WeightFactor` protocol `weight(meta:dict, quantity:str)->float`, `learnable:bool=False`, `params()->None`; `Structural(exp={"E":0.5,"V":0.5,"F":0.0})`, `Quantity(w:dict)`, `ConfigType(dict, key="config_type", default=...)`, `PerConfig(key="weight")`, `Custom(fn)`; `compose(factors)-> fn(meta,q)->float`.
- [ ] **Test:**
```python
def test_structural_times_configtype():
    from ace_jax.fit.weights import Structural, ConfigType, compose
    f = compose([Structural(), ConfigType({"defect": {"E": 10, "F": 10, "V": 1}},
                                          default={"E": 1, "F": 1, "V": 1})])
    meta = {"n_atoms": 4, "config_type": "defect"}
    assert abs(f(meta, "E") - 10 * 4 ** -0.5) < 1e-9
    assert abs(f(meta, "F") - 10 * 1.0) < 1e-9        # F exp 0 -> structural 1
def test_perconfig_override():
    from ace_jax.fit.weights import PerConfig, compose
    f = compose([PerConfig(key="w")])
    assert f({"w": 2.5}, "E") == 2.5 and f({}, "E") == 1.0
```
- [ ] Implement each factor as a small class; `compose` returns the product; absent keys → neutral 1.0. Commit `feat(fit): composable weight-factor pipeline`.

### Task 13: wire `compose` into `load_configs` + run.py `--weights`

**Files:** Modify `src/ace_jax/fit/data.py:43` (`load_configs`), `bench/acegp_cantor/run.py` (`--weights <json>`), `tests/test_gp_data.py`.
- **Interfaces:** `load_configs(..., factors=None)` — default `[Structural(), ConfigType(default={E:1,F:1,V:1})]` reproduces today's `1/√n × type-dict`; `--weights` parses a JSON factor list into `factors`.
- [ ] **Test:** `load_configs` with default factors gives the same `w_E/w_F/w_V` as the current code on a 2-config fixture (backwards-compat), and a `PerConfig` factor changes one config's `w_E`. Implement: replace the inline weight computation with `compose(factors)` evaluated per config with `meta={"n_atoms":n, "config_type":...,**at.info}`. Commit `feat(fit,cli): weight factors in load_configs + --weights`.

### Task 14: `--route` override + docs

**Files:** Modify `bench/acegp_cantor/run.py` (`--route <json>` maps block→route), `docs/specs/...` unchanged; add `docs/plans/...` note; `tests/test_gp_cli.py`.
- [ ] **Test:** `--route '{"sigma_type":"lml"}'` produces a `ParamSet` with that block LML-routed; `--route '{"embed":"fixed"}'` freezes the embedding (VarOpt skipped). Implement a small parser mapping names→routes onto `from_hypers`/block construction. Commit `feat(cli): per-block route override`.

**Phase-4 gate:** full suite green; a default-flags run reproduces the pre-change numbers on the SiGe fixture.

---

## Self-review

**Spec coverage:** Component 1 → Tasks 1–5; Component 2 (σ_type) → Task 6; Component 3 (POPS: whiten/corrections/posteriors/streamed/predict/gate) → Tasks 7–11; Component 4 (weights + run.py) → Tasks 12–14; backwards-compat asserted in the Phase gates. Designed-for-not-built (learnable-weight VarOpt, held-out objective) correctly excluded — the hooks (`WeightFactor.learnable/params`, `varopt_ps` accepting any block) are present without wiring.

**Type consistency:** `ParamSet` methods (`lml_vector/set_lml_vector/varopt_vector/set_varopt_vector/block/materialise`) and `from_hypers(hypers, prior, embed, hypers_route, embed_route)` are used identically in Tasks 2–6. `run_map_ps(ps, prob, ds, ...)`, `varopt_ps(ps, objective_and_grad, *, val_score)`, `whiten(phi,resid,w,sigma)`, `pops_corrections/pops_posterior/pops_var`, `compose(factors)` names match across their producer and consumer tasks.

**Placeholder scan:** no TBD/TODO; each code step has real code or a precise, signature-level instruction pointing at the exact function to adapt (`make_lml_embed` as the caching template; `popsregression._build_posterior` as the port source). Two tasks (3, 6) tell the executor to read the existing body they extend — that is direction, not a placeholder, since the new signature, behaviour, and test are given.
