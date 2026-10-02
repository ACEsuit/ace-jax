# Joint radial + density VarPro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn P sqrt(density) features jointly with the ACE tensor radials by VarPro, then freeze them as extra linear columns. The final model stays linear in its readout and is exported and evaluated as an `FSModel`.

**Architecture:**

- `fit/density.py` gains masked, multi-density design columns and their streamed statistics.
- A new `fit/radial_density.py` holds:
  - the joint outer loop, an L-BFGS over the radial and density weights [V; H] with a block preconditioner (or alternating blocks as the fallback);
  - the held-out gate.
- A new `eval/fs_model.py` wraps an `ACEModel` with the frozen density term. Optional npz keys make it load transparently, so the calculators, `qoi.py` and MD need no change.

**Tech Stack:** Python 3.11+, JAX (float64), equinox, optax, pytest.

**Spec:** `docs/dev/specs/2026-09-28-radial-density-varpro-design.md`

## Global Constraints

- Float64 is required. Tests put `jax.config.update("jax_enable_x64", True)` at the top of the module. Library code never calls `jax.config.update`.
- P = 0, or an npz without fs keys, is exactly today's model: `aj.load` returns a plain `ACEModel`, and the statistics equal `linear_statistics` bit-for-bit.
- Density column order: column `L + p*NZ + z` is density p for species z, where `L = cfg.len_basis`.
- F is `ssqrt(ρ) = ρ (ρ² + ε)^(-1/4)` with ε = `fit.density.EPS` = 1e-6. The same ε is written to `meta["fs"]["eps"]` and used by `FSModel`.
- The prior weight on the density readout is γ_d, the geometric mean of Γ.
- `run.py --density` defaults to `none`, and today's behaviour is unchanged.
- Style: dense code with short math names, and trailing `#` comments that explain why. No f-strings with nested same-type quotes (Python 3.11). Plain pytest functions.
- Commits use conventional prefixes with a scope, for example `feat(fit):` or `test(fit):`. End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_019YVpzWuvJBLVWKjCpWr5ve
  ```
- Never use bare `git stash`.
- Do not run anything on moriarty; it is reserved for the user's benchmarks. Remote runs go on lestrade or Modal.
- Run targeted tests single-process: `uv run pytest tests/<file> -q -p no:xdist`. Before the final commit of each task, run the full suite: `uv run pytest -q`.

## Review Focus

1. **A species with no training sites** (for example Ge in `sige_nofit` on Si data). Its η must be exactly 0 and every quantity finite, never NaN from normalising a zero second moment. The test is in Task 4.
2. **A source npz that already carries fs keys,** passed to `patch_radial_npz` without `fs`. The stale density keys and `meta["fs"]` must be dropped. The test is in Task 3.
3. **A density npz given to `run.py --model`.** The driver must unwrap `FSModel.base` and say so, rather than crash in `to_analytic`. The test is in Task 6.
4. **Density flags with multi-valued `--lam-grid`, `--spec-grid` or `--gap-grid`.** These must fail fast with a clear message, not silently use the first value. The test is in Task 6.
5. **The edge-kind switch on an `FSModel`** (`with_edge_a_kind`, used by `ACECalculator`'s auto calibration). It must switch the base model and keep the density term. The test is in Task 3.

---

### Task 1: Size the prior from γ; allow precomputed statistics in θ-MAP and the holdout score

**Files:**
- Modify: `src/ace_jax/fit/objective.py` (`prior_precision`)
- Modify: `src/ace_jax/fit/solve.py` (`_prior_block`)
- Modify: `src/ace_jax/fit/radial_learn.py` (`theta_map_linear`, `holdout_score`)
- Test: `tests/test_radial_density.py` (new)

**Interfaces:**
- Produces:
  - `prior_precision(theta, prob)` sizes its linear block from `prob.gamma.shape[0]`.
  - `theta_map_linear(prob, ds, W, *, steps=300, seed=0, init=None, return_stats=False, lin=None)`: with `lin` given, W is unused (it may be None) and no streaming pass runs.
  - `holdout_score(W, ..., lin_fit=None, lin_val=None, ...)`: W may be None when both statistics are given.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_radial_density.py`:

```python
"""Joint radial + density VarPro (fit.density masked columns, fit.radial_density)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from conftest import FIXTURE_DIR
from test_gp_learn_radial import MODEL, THETA, XYZ, make_problem

pytestmark = pytest.mark.skipif(not (XYZ.exists() and MODEL.exists()), reason="missing fixtures")


@pytest.fixture(scope="module")
def small():
    return make_problem()


def test_prior_precision_sizes_from_gamma(small):
    from ace_jax.fit.objective import prior_precision
    prob, _, _ = small
    L = prob.cfg.len_basis
    Lam, logdet = prior_precision(THETA, prob._replace(gamma=jnp.full(L + 3, 2.0)))
    assert Lam.shape == (L + 3, L + 3)
    np.testing.assert_allclose(np.diag(Lam), 4.0 / 0.3 ** 2)
    Lam0, _ = prior_precision(THETA, prob)
    assert Lam0.shape == (L, L)


def test_theta_map_and_holdout_take_precomputed_stats(small):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_learn import holdout_score, theta_map_linear
    from ace_jax.fit.stats import linear_statistics
    prob, ds, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    W = prob.model.rnl_Wnlq
    lin = linear_statistics(prob.model, prob.cfg, ds)
    a1 = theta_map_linear(prob, ds, W, steps=20)
    a2 = theta_map_linear(prob, ds, None, steps=20, lin=lin)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    a = to_array(THETA)
    lv = linear_statistics(prob.model, prob.cfg, ds_val)
    s1 = holdout_score(W, a, a, prob, ds, ds_val)
    s2 = holdout_score(None, a, a, prob, ds, ds_val, lin_fit=lin, lin_val=lv)
    np.testing.assert_allclose(s1, s2, rtol=1e-12)
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_radial_density.py -q -p no:xdist`
  Expected: FAIL. The first test fails on a shape mismatch in `prior_precision` (it uses `len_basis`); the second fails because `lin` is an unexpected keyword.

- [ ] **Step 3: Implement.** In `objective.py`, `prior_precision`:

```python
def prior_precision(theta, prob):
    L = prob.gamma.shape[0]      # len_basis, or wider when fit.density columns are appended
```

In `solve.py`, `_prior_block`, replace `L, M = prob.cfg.len_basis, prob.ind.XM.shape[0]` with:

```python
    L, M = prob.gamma.shape[0], prob.ind.XM.shape[0]
```

In `radial_learn.py`, `theta_map_linear`: add `lin=None` to the signature and replace the first line of the body with:

```python
    if lin is None:
        lin = linear_statistics(with_radial(prob.model, W), prob.cfg, ds)
```

Extend its docstring: "`lin`: the statistics of ds if the caller already has them (then W is unused and may be None; e.g. the widened statistics of fit.density)."

In `holdout_score`, make the model lazy:

```python
    if lin_fit is None or lin_val is None:
        model = with_radial(prob.model, W)
    if lin_fit is None:
        lin_fit = linear_statistics(model, prob.cfg, ds_fit)
    if lin_val is None:
        lin_val = linear_statistics(model, prob.cfg, ds_val)
```

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run: `uv run pytest tests/test_radial_density.py tests/test_gp_learn_radial.py tests/test_gp_objective.py -q -p no:xdist`. If `test_gp_objective.py` doesn't exist, run `uv run pytest -q -k "objective or solve"` instead.
  Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/fit/objective.py src/ace_jax/fit/solve.py src/ace_jax/fit/radial_learn.py tests/test_radial_density.py
git commit -m "refactor(fit): size the prior from gamma; theta-MAP/holdout accept precomputed stats"
```

---

### Task 2: Masked multi-density columns and widened statistics

**Files:**
- Modify: `src/ace_jax/fit/density.py`
- Test: `tests/test_radial_density.py`

**Interfaces:**
- Consumes: `linear_rows(model, cfg, batch) -> (Rows, X (Ncap, D), J (Ncap*K, D, 3))`; `density_rows_pair(eta (NZ, w), cfg, batch, Bpair (Ncap, w), Jpair (Ncap, K, w, 3)) -> Rows`, which works for any width w; `stats._linear_type_stats`; `stats.Stats`.
- Produces:
  - `density_rows_masked(eta, mask, cfg, batch, X, J) -> Rows`, with E (C, P·NZ), F (Ncap, 3, P·NZ) and V (C, 6, P·NZ). eta is (P, NZ, D) and mask is (D,).
  - `density_mask(cfg, span) -> (D,) float`, where span is `"full"` or `"pair"`.
  - `compact_gamma(gamma, cfg) -> (NZ, D)`.
  - `density_gamma(gamma, P, NZ) -> (len_basis + P·NZ,)`.
  - `linear_density_statistics(model, eta, mask, cfg, ds) -> Stats` with Dt = len_basis + P·NZ. P = 0 returns `linear_statistics(model, cfg, ds)` itself.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_radial_density.py`:

```python
def _dens_fd_check(model, cfg, batch, eta, mask):
    """density_rows_masked F/V rows are minus the derivative of its E rows (autodiff through
    compact_basis), as in tests/test_gp_density.py."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.data import VOIGT, flat_edges
    from ace_jax.fit.density import density_rows_masked
    from ace_jax.fit.rows import linear_rows
    Ncap, K = batch.nbr.shape
    ncol = eta.shape[0] * cfg.NZ
    with highest_precision():
        _, X, J = linear_rows(model, cfg, batch)
        dr = density_rows_masked(eta, mask, cfg, batch, X, J)

        def e_dens(rij):
            b = batch._replace(rij=rij)
            r, send, recv, m = flat_edges(rij, batch.nbr, batch.nbr_mask)
            Xr = model.compact_basis(r, batch.node_z[send], batch.node_z[recv], send, Ncap, m)
            return density_rows_masked(eta, mask, cfg, b, Xr, J).E
        T = jax.jacrev(e_dens)(batch.rij)
    node_cfg = np.asarray(batch.node_cfg); nbr = np.asarray(batch.nbr).reshape(-1)
    own = np.minimum(node_cfg, cfg.C - 1)
    Tn = np.where(np.asarray(batch.node_mask)[:, None, None, None],
                  np.asarray(T)[own, :, np.arange(Ncap)], 0.0)               # (Ncap, ncol, K, 3)
    dEdr = np.zeros((Ncap, 3, ncol))
    dEdr -= Tn.sum(2).transpose(0, 2, 1)
    np.add.at(dEdr, nbr, Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, ncol))
    assert np.abs(dEdr).max() > 1e-3
    assert np.abs(np.asarray(dr.F) + dEdr).max() < 1e-9 * max(1.0, np.abs(dEdr).max())
    rij = np.asarray(batch.rij).reshape(Ncap * K, 3)
    Te = Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, ncol)
    Vref = np.zeros((cfg.C + 1, 6, ncol)); ecfg = node_cfg[np.repeat(np.arange(Ncap), K)]
    for v, (a, b_) in enumerate(VOIGT):
        np.add.at(Vref[:, v], ecfg, -0.5 * (Te[:, a] * rij[:, b_, None] + Te[:, b_] * rij[:, a, None]))
    assert np.abs(np.asarray(dr.V) - Vref[:cfg.C]).max() < 1e-9 * max(1.0, np.abs(Vref).max())
    return dr


@pytest.mark.parametrize("span,P", [("full", 2), ("pair", 1), ("pair", 2)])
def test_density_rows_masked_are_derivatives(small, span, P):
    from ace_jax.fit.density import density_mask
    prob, ds, _ = small
    cfg = prob.cfg
    mask = density_mask(cfg, span)
    eta = jnp.asarray(np.random.default_rng(1).standard_normal((P, cfg.NZ, cfg.D))) * mask
    dr = _dens_fd_check(prob.model, cfg, jax.tree.map(lambda a: a[0], ds), eta, mask)
    assert dr.E.shape[-1] == P * cfg.NZ


def test_masked_pair_P1_equals_density_rows(small):
    from ace_jax.fit.density import density_mask, density_rows, density_rows_masked
    from ace_jax.fit.rows import linear_rows
    prob, ds, _ = small
    cfg = prob.cfg
    b = jax.tree.map(lambda a: a[0], ds)
    _, X, J = linear_rows(prob.model, cfg, b)
    eta_p = jnp.asarray(np.random.default_rng(2).standard_normal((cfg.NZ, cfg.n_pair)))
    eta = jnp.zeros((1, cfg.NZ, cfg.D)).at[0, :, cfg.n_B:].set(eta_p)
    a = density_rows(eta_p, cfg, b, X, J)
    m = density_rows_masked(eta, density_mask(cfg, "pair"), cfg, b, X, J)
    for k in "EFV":
        np.testing.assert_allclose(np.asarray(getattr(m, k)), np.asarray(getattr(a, k)), rtol=1e-12, atol=1e-12)


def test_density_statistics_P0_is_linear_statistics(small):
    from ace_jax.fit.density import density_mask, linear_density_statistics
    from ace_jax.fit.stats import linear_statistics
    prob, ds, _ = small
    cfg = prob.cfg
    a = linear_statistics(prob.model, cfg, ds)
    b = linear_density_statistics(prob.model, jnp.zeros((0, cfg.NZ, cfg.D)), density_mask(cfg, "full"), cfg, ds)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _widened_rows(model, cfg, batch, eta, mask):
    from ace_jax.fit.density import density_rows_masked
    from ace_jax.fit.rows import Rows, linear_rows
    r, X, J = linear_rows(model, cfg, batch)
    d = density_rows_masked(eta, mask, cfg, batch, X, J)
    return Rows(*(jnp.concatenate([getattr(r, k), getattr(d, k)], -1) for k in "EFV"))


def test_density_statistics_match_in_memory(small):
    from ace_jax.fit.density import density_mask, linear_density_statistics
    prob, ds, _ = small
    cfg = prob.cfg
    mask = density_mask(cfg, "full")
    eta = jnp.asarray(np.random.default_rng(3).standard_normal((2, cfg.NZ, cfg.D)))
    st = linear_density_statistics(prob.model, eta, mask, cfg, ds)
    G = {k: 0.0 for k in "EFV"}; bb = {k: 0.0 for k in "EFV"}
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r = _widened_rows(prob.model, cfg, b, eta, mask)
        Dt = r.E.shape[-1]
        for k, R, y, w in (("E", r.E, b.y_E, b.w_E),
                           ("F", r.F.reshape(-1, Dt), b.y_F.reshape(-1), jnp.repeat(b.w_F, 3)),
                           ("V", r.V.reshape(-1, Dt), b.y_V.reshape(-1), jnp.repeat(b.w_V, 6))):
            Pw = R * w[:, None]
            G[k] = G[k] + Pw.T @ Pw; bb[k] = bb[k] + Pw.T @ (y * w)
    for k in "EFV":
        np.testing.assert_allclose(np.asarray(getattr(st, "G_" + k)), np.asarray(G[k]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(getattr(st, "b_" + k)), np.asarray(bb[k]), rtol=1e-10, atol=1e-10)


def test_widened_projected_residual_matches_qr(small):
    """VarPro objective on the widened statistics == min ||Phi~ c - y~||^2 by QR on the
    materialised prior-augmented [linear | density] design."""
    from ace_jax.fit.density import density_gamma, density_mask, linear_density_statistics
    from ace_jax.fit.radial_learn import projected_residual_from_stats
    prob, ds, _ = small
    cfg = prob.cfg
    mask = density_mask(cfg, "full")
    eta = jnp.asarray(np.random.default_rng(4).standard_normal((2, cfg.NZ, cfg.D))) * 1e-2
    gw = density_gamma(prob.gamma, 2, cfg.NZ)
    got = float(projected_residual_from_stats(THETA, linear_density_statistics(prob.model, eta, mask, cfg, ds), gw))
    se, sf, sv = (float(np.exp(getattr(THETA, "log_sigma_" + k))) for k in "EFV")
    rows, ys = [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r = _widened_rows(prob.model, cfg, b, eta, mask)
        Dt = r.E.shape[-1]
        w = jnp.concatenate([b.w_E / se, jnp.repeat(b.w_F, 3) / sf, jnp.repeat(b.w_V, 6) / sv])
        Phi = jnp.concatenate([r.E, r.F.reshape(-1, Dt), r.V.reshape(-1, Dt)])
        y = jnp.concatenate([b.y_E, b.y_F.reshape(-1), b.y_V.reshape(-1)])
        rows.append(Phi * w[:, None]); ys.append(y * w)
    Phi = jnp.concatenate(rows + [jnp.diag(gw / float(np.exp(THETA.log_sigma_c)))])
    y = jnp.concatenate(ys + [jnp.zeros(gw.shape[0])])
    Qm, _ = jnp.linalg.qr(Phi)
    v = Qm.T @ y
    ref = float(y @ y - v @ v)
    assert abs(got - ref) < 1e-7 * abs(ref)


def test_density_gamma_and_compact_gamma():
    from types import SimpleNamespace
    from ace_jax.fit.density import compact_gamma, density_gamma
    cfg = SimpleNamespace(n_B=2, n_pair=1, NZ=2, D=3)
    g = jnp.asarray([1., 2., 3., 4., 5., 6.])           # B z0 | B z1 | pair z0 | pair z1
    np.testing.assert_array_equal(np.asarray(compact_gamma(g, cfg)), [[1, 2, 5], [3, 4, 6]])
    gw = density_gamma(g, 2, 2)
    assert gw.shape == (10,)
    np.testing.assert_allclose(np.asarray(gw[6:]), np.exp(np.mean(np.log(np.arange(1, 7)))), rtol=1e-12)
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_radial_density.py -q -p no:xdist`
  Expected: FAIL with an ImportError on `density_rows_masked`.

- [ ] **Step 3: Implement.** Append to `src/ace_jax/fit/density.py`, and add `import jax.numpy as jnp` and friends only if they are missing:

```python
def density_rows_masked(eta, mask, cfg, batch, X, J):
    """P * NZ density columns over the masked compact basis: column p * NZ + z is
    species z's ssqrt(eta[p, z] . (mask * X_i)).  eta (P, NZ, D), mask (D,) 0/1,
    X (Ncap, D) and J (Ncap * K, D, 3) as linear_rows returns them.  Each p is
    density_rows_pair over the full width with eta masked (its algebra is
    width-agnostic), so P = 1 with the pair mask reproduces density_rows."""
    Ncap, K = batch.nbr.shape
    P = eta.shape[0]
    if P == 0:
        C = batch.y_E.shape[0]
        return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))
    Jr = J.reshape(Ncap, K, cfg.D, 3)
    rows = [density_rows_pair(eta[p] * mask, cfg, batch, X, Jr) for p in range(P)]
    return Rows(*(jnp.concatenate([getattr(r, k) for r in rows], axis=-1) for k in "EFV"))


def density_mask(cfg, span):
    """(D,) 0/1 span of the density over the compact basis [B | pair]."""
    if span == "full":
        return jnp.ones(cfg.D)
    if span == "pair":
        return jnp.concatenate([jnp.zeros(cfg.n_B), jnp.ones(cfg.n_pair)])
    raise ValueError(f"density span must be 'full' or 'pair', got {span!r}")


def compact_gamma(gamma, cfg):
    """The species-blocked prior diagonal (len_basis,) regrouped per species onto
    the compact basis, (NZ, D), in the layout of rows._place."""
    g, nB, nP, NZ = jnp.asarray(gamma), cfg.n_B, cfg.n_pair, cfg.NZ
    return jnp.stack([jnp.concatenate([g[z * nB:(z + 1) * nB], g[NZ * nB + z * nP:NZ * nB + (z + 1) * nP]])
                      for z in range(NZ)])


def density_gamma(gamma, P, NZ):
    """Prior diagonal of the widened readout [c | d]: Gamma, then the geometric
    mean of Gamma for each of the P * NZ density coefficients."""
    g = jnp.asarray(gamma)
    return jnp.concatenate([g, jnp.full(P * NZ, jnp.exp(jnp.mean(jnp.log(g))))])


def batch_density_stats(model, eta, mask, cfg, batch):
    from .stats import Stats, _linear_type_stats
    r, X, J = linear_rows(model, cfg, batch)
    d = density_rows_masked(eta, mask, cfg, batch, X, J)
    E, F, V = (jnp.concatenate([getattr(r, k), getattr(d, k)], -1) for k in "EFV")
    Dt = E.shape[-1]
    sE = _linear_type_stats(E, batch.y_E, batch.w_E)
    sF = _linear_type_stats(F.reshape(-1, Dt), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3))
    sV = _linear_type_stats(V.reshape(-1, Dt), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))
    return Stats(*(s[i] for i in range(5) for s in (sE, sF, sV)))


def linear_density_statistics(model, eta, mask, cfg, ds):
    """Streamed statistics of the widened design [linear | P * NZ density columns],
    Dt = len_basis + P * NZ, same layout and checkpointed scan as
    stats.linear_statistics.  P = 0 IS linear_statistics (bit-identical)."""
    from .stats import Stats, linear_statistics
    if eta.shape[0] == 0:
        return linear_statistics(model, cfg, ds)
    Dt = cfg.len_basis + eta.shape[0] * cfg.NZ
    z2, z1, z0 = jnp.zeros((Dt, Dt)), jnp.zeros(Dt), jnp.zeros(())
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda b: batch_density_stats(model, eta, mask, cfg, b))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(b)), None), zero, ds)[0]
```

`linear_rows` must be importable here: add `from .rows import Rows, linear_rows` (replacing `from .rows import Rows`). Check the `Stats` field order: `Stats(G_E, G_F, G_V, b_E, b_F, b_V, yy_E, yy_F, yy_V, n_E, n_F, n_V, logw_E, logw_F, logw_V)`, and `_linear_type_stats` returns `(G, b, yy, n, logw)`. So `Stats(*(s[i] for i in range(5) for s in (sE, sF, sV)))` yields exactly that order.

Update the module docstring's first paragraph to mention the masked, multi-density generalisation (`density_rows_masked`) used by `fit.radial_density`.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run: `uv run pytest tests/test_radial_density.py tests/test_gp_density.py -q -p no:xdist`
  Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/fit/density.py tests/test_radial_density.py
git commit -m "feat(fit): masked multi-density columns and widened linear statistics"
```

---

### Task 3: `FSModel`, npz keys and the loader

**Files:**
- Create: `src/ace_jax/eval/fs_model.py`
- Modify: `src/ace_jax/eval/io.py` (`load`, just before the final `return model, meta, z` of the ACEModel branch)
- Modify: `src/ace_jax/eval/edge_model.py` (`with_edge_a_kind`)
- Modify: `src/ace_jax/construct/export.py` (`patch_radial_npz`)
- Test: `tests/test_fs_model.py` (new)

**Interfaces:**
- Consumes:
  - `ACEModel.site_basis(rij, zi, zj, segment_ids, n_nodes, mask) -> (B, Apair)`;
  - `site_basis_dense(rij, zi, zj, mask) -> (B, Apair)`;
  - `_readout(B, Apair, node_z)`;
  - `fit.density.EPS`, which is used only in the tests.
- Produces:
  - `FSModel(base, eta (P,NZ,D), d (P,NZ), mask (D,) float, eps=1e-6)`, an `EdgeSiteModel`, and `FSModel.with_base(base)`.
  - `patch_radial_npz(src, dst, model, readout=None, fs=None)`, where `fs = (eta, d, mask, eps)`.
  - `load` returns an `FSModel` when `fs_eta` is in the npz.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_fs_model.py`:

```python
"""eval.fs_model.FSModel: frozen sqrt-density term on a linear ACE model; npz round trip."""
import json

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from conftest import FIXTURE_DIR
from ace_jax.eval import load

MODEL = FIXTURE_DIR / "si_ace_model.npz"
XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not (MODEL.exists() and XYZ.exists()), reason="missing fixtures")


def _fs(model, meta, P=2, seed=0):
    rng = np.random.default_rng(seed)
    NZ, D = len(meta["elements"]), meta["n_B"] + meta["n_pair"]
    mask = np.ones(D)
    return (1e-2 * rng.standard_normal((P, NZ, D)), rng.standard_normal((P, NZ)), mask, 1e-6)


def _atoms():
    from ace_jax.fit.xyz import read_extxyz
    return read_extxyz(XYZ)[0]


def test_plain_npz_loads_plain_model():
    from ace_jax.eval.fs_model import FSModel
    m, _, _ = load(MODEL)
    assert not isinstance(m, FSModel)


def test_npz_roundtrip_and_calculator(tmp_path):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval.fs_model import FSModel
    base, meta, _ = load(MODEL)
    eta, d, mask, eps = _fs(base, meta)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=(eta, d, mask, eps))
    m, meta2, z = load(tmp_path / "fs.npz")
    assert isinstance(m, FSModel) and meta2["fs"] == {"P": 2, "F": "ssqrt", "eps": eps}
    ref = FSModel(base, jnp.asarray(eta), jnp.asarray(d), jnp.asarray(mask), eps)
    at = _atoms()
    at.calc = ACECalculator(tmp_path / "fs.npz")
    e_fs = at.get_potential_energy()
    at.calc = ACECalculator(ref, meta)
    np.testing.assert_allclose(at.get_potential_energy(), e_fs, rtol=1e-12)
    at.calc = ACECalculator(MODEL)
    assert abs(at.get_potential_energy() - e_fs) > 1e-6          # the density term is live
    import sys
    from conftest import ROOT
    sys.path.insert(0, str(ROOT / "bench/learn_radial/md"))
    from padded_calc import PaddedACECalculator
    at.calc = PaddedACECalculator(tmp_path / "fs.npz")
    np.testing.assert_allclose(at.get_potential_energy(), e_fs, rtol=1e-12)


def test_fsmodel_forces_are_energy_derivative(tmp_path):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.construct.export import patch_radial_npz
    base, meta, _ = load(MODEL)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=_fs(base, meta))
    at = _atoms()
    at.calc = ACECalculator(tmp_path / "fs.npz")
    F = at.get_forces()
    h = 1e-5
    for i, a in ((0, 0), (1, 2)):
        p = at.get_positions()
        p[i, a] += h; at.set_positions(p); ep = at.get_potential_energy()
        p[i, a] -= 2 * h; at.set_positions(p); em = at.get_potential_energy()
        p[i, a] += h; at.set_positions(p)
        np.testing.assert_allclose(F[i, a], -(ep - em) / (2 * h), atol=1e-6)


def test_edge_kind_switch_keeps_density(tmp_path):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval.edge_model import with_edge_a_kind
    from ace_jax.eval.fs_model import FSModel
    base, meta, _ = load(MODEL)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=_fs(base, meta))
    m, _, _ = load(tmp_path / "fs.npz")
    mm = with_edge_a_kind(m, "matmul")
    assert isinstance(mm, FSModel) and mm.edge_a_kind == "matmul"
    np.testing.assert_array_equal(np.asarray(mm.eta), np.asarray(m.eta))


def test_patch_without_fs_drops_stale_density(tmp_path):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval.fs_model import FSModel
    base, meta, _ = load(MODEL)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=_fs(base, meta))
    patch_radial_npz(tmp_path / "fs.npz", tmp_path / "plain.npz", base)
    m, meta2, z = load(tmp_path / "plain.npz")
    assert not isinstance(m, FSModel) and "fs" not in meta2
    assert not any(k.startswith("fs_") for k in z.files)
```

**Note.** Check the helper names before running. If `ace_jax.fit.xyz` has no `read_extxyz` returning ASE Atoms, use `ase.io.read(XYZ, 0)`: the tests only need positions, cell and numbers, not the stored labels. If `ACECalculator(model, meta)` needs meta keys (`rcut`, `elements`), pass the `meta` from `load(MODEL)`.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_fs_model.py -q -p no:xdist`
  Expected: FAIL with an ImportError on `ace_jax.eval.fs_model`, or a TypeError on the `fs=` argument.

- [ ] **Step 3: Implement `src/ace_jax/eval/fs_model.py`.**

```python
"""A linear ACE model plus a frozen Finnis-Sinclair sqrt-density term.

    E_i = c_z . X_i + sum_p d[p, z] ssqrt(eta[p, z] . (mask * X_i)),
    ssqrt(rho) = rho (rho^2 + eps)^(-1/4),

X_i the compact site basis [B | Apair] of `base`.  eta (the density weights,
learned by fit.radial_density and then frozen) and d (their readout, fitted with
c) are data; forces and virial come from EdgeSiteModel's one value_and_grad over
the edge vectors, so there is no bespoke derivative code.  The linear part is
evaluated through `base._readout` on the materialised B (not the folded
readout), since the density needs B anyway.  Written by
construct.export.patch_radial_npz(fs=...), loaded by eval.io.load.
See docs/dev/specs/2026-09-28-radial-density-varpro-design.md.
"""
import dataclasses

import equinox as eqx
import jax.numpy as jnp

from .edge_model import EdgeSiteModel


def ssqrt(rho, eps):
    return rho * (rho * rho + eps) ** -0.25


class FSModel(EdgeSiteModel):
    base: EdgeSiteModel
    eta: jnp.ndarray            # (P, NZ, D), zero off the mask
    d: jnp.ndarray              # (P, NZ)
    mask: jnp.ndarray           # (D,) 0/1
    eps: float = eqx.field(static=True, default=1e-6)

    def _energies(self, B, Apair, node_z):
        X = jnp.concatenate([B, Apair], axis=1) * self.mask
        rho = jnp.einsum("nd,pnd->np", X, self.eta[:, node_z])
        fs = jnp.sum(ssqrt(rho, self.eps) * self.d[:, node_z].T, axis=1)
        return self.base._readout(B, Apair, node_z) + fs

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        return self._energies(*self.base.site_basis(rij, zi, zj, segment_ids, n_nodes, mask), node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        return self._energies(*self.base.site_basis_dense(rij, zi, zj, mask), node_z)

    def with_base(self, base):
        return dataclasses.replace(self, base=base)

    # the base's geometry, A-basis form and descriptors, for the calculators
    def pad_cutoff(self):
        return self.base.pad_cutoff()

    def edge_a_widths(self):
        return self.base.edge_a_widths()

    def edge_a_factors(self, *args, **kw):
        return self.base.edge_a_factors(*args, **kw)

    def site_basis(self, *args, **kw):
        return self.base.site_basis(*args, **kw)

    def site_basis_dense(self, *args, **kw):
        return self.base.site_basis_dense(*args, **kw)

    def compact_basis(self, *args, **kw):
        return self.base.compact_basis(*args, **kw)

    def site_descriptors(self, *args, **kw):
        return self.base.site_descriptors(*args, **kw)

    a_channels = property(lambda self: self.base.a_channels)
    aspec_r = property(lambda self: self.base.aspec_r)
    aspec_y = property(lambda self: self.base.aspec_y)
    edge_a_kind = property(lambda self: self.base.edge_a_kind)
    a_sel_r = property(lambda self: self.base.a_sel_r)
    a_sel_y = property(lambda self: self.base.a_sel_y)
    E0 = property(lambda self: self.base.E0)
    elements = property(lambda self: self.base.elements)
```

If equinox rejects class-level `property` objects alongside dataclass fields, use `@property` methods instead; they are equivalent. Check `ACEModel` for other attributes the calculator touches with `grep -n "self.model\.\|model\.[a-z_]*" src/ace_jax/calc/point.py src/ace_jax/eval/edge_model.py src/ace_jax/eval/api.py`, and delegate any that are missing the same way.

In `edge_model.with_edge_a_kind`, make the first statement of the body:

```python
    if hasattr(model, "with_base"):          # FSModel: switch the wrapped model's form
        return model.with_base(with_edge_a_kind(model.base, kind))
```

In `io.load`, immediately before the final `return model, meta, z`:

```python
    if "fs_eta" in z.files:                  # frozen sqrt-density term (fit.radial_density)
        from .fs_model import FSModel
        fs = meta.get("fs") or {}
        if fs.get("F", "ssqrt") != "ssqrt":
            raise ValueError(f"unsupported density embedding {fs.get('F')!r}")
        model = FSModel(model, A("fs_eta"), A("fs_d"), jnp.asarray(z["fs_mask"], dtype),
                        float(fs.get("eps", 1e-6)))
```

In `export.patch_radial_npz`, add `fs=None` to the signature and extend the docstring: "`fs`: (eta, d, mask, eps) of a frozen sqrt-density term (eval.fs_model); written as fs_eta/fs_d/fs_mask + meta['fs']. fs=None drops any density keys of the source, which belong to the old radials." Before `meta["radial_kind"] = "analytic"`, add:

```python
    for k in ("fs_eta", "fs_d", "fs_mask"):
        out.pop(k, None)
    meta.pop("fs", None)
    if fs is not None:
        eta, d, mask, eps = fs
        out["fs_eta"] = np.asarray(eta, np.float64)
        out["fs_d"] = np.asarray(d, np.float64)
        out["fs_mask"] = np.asarray(mask, bool)
        meta["fs"] = {"P": int(out["fs_eta"].shape[0]), "F": "ssqrt", "eps": float(eps)}
```

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run: `uv run pytest tests/test_fs_model.py tests/test_gp_learn_radial.py -q -p no:xdist -k "fs or save or readout or roundtrip"`
  Expected: PASS.

- [ ] **Step 5: Run the full suite, then commit.**
  Run: `uv run pytest -q` and expect PASS.

```bash
git add src/ace_jax/eval/fs_model.py src/ace_jax/eval/io.py src/ace_jax/eval/edge_model.py src/ace_jax/construct/export.py tests/test_fs_model.py
git commit -m "feat(eval): FSModel -- frozen sqrt-density term, npz keys, loader"
```

---

### Task 4: `learn_radial_density`, the joint and alternating outer loop

**Files:**
- Create: `src/ace_jax/fit/radial_density.py`
- Test: `tests/test_radial_density.py`

**Interfaces:**
- Consumes:
  - from Task 2: `linear_density_statistics`, `density_gamma`, `compact_gamma` and `density_mask`;
  - from Task 1: `theta_map_linear(..., lin=)`;
  - from `radial_learn`: `lbfgs_loop`, `projected_residual_from_stats`, `require_x64`, `require_linear`, `relative_lambda`, `relative_lambda_spec` and `relative_lambda_gap`;
  - from `radial_model`: `normalise`, `radial_gram`, `row_active`, `roughness`, `roughness_matrix`, `spectral_weights`, `spectral_penalty`, `gap_penalty`, `uniform_gram`, `data_r_range`, `require_analytic` and `with_radial`.
- Produces:
  - `rho_gram(model, W, mask, cfg, ds) -> S (NZ, D, D)`;
  - `normalise_rho(H, S) -> eta`;
  - `init_density(cfg, mask, P, seed=0) -> H (P, NZ, D)`;
  - `relative_lambda_eta(lam_eta, r0, n) -> float`;
  - `learn_radial_density(prob, ds, W0, *, mask, P=1, H0=None, mode="joint", theta0=None, profile=True, lam_rough=0.0, rough_weights=None, lam_spec=0.0, spec_p=4.0, lam_gap=0.0, lam_eta=0.0, steps=40, reprofile_every=10, tol=1e-6, patience=3, map_steps=300, seed=0, log=None, Q=None, D2=None, U=None, S=None, r0=None) -> (V, eta, info)`.

  V is the normalised radials and eta is normalised. `info` has these keys: `trace`, `reasons`, `round_lengths`, `theta`, `precond`, `r0`, `lam_abs`, `lam_spec_abs`, `lam_gap_abs`, `lam_eta`, `lam_eta_abs`, `P`, `mode`, `steps` and `theta_final`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_radial_density.py`:

```python
def _rd_args(prob, ds, P=2, span="full", seed=0):
    """Full positional args of _objective_rd after (xb, other), plus V, H, statics pieces."""
    from ace_jax.fit.density import compact_gamma, density_gamma, density_mask
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_density import init_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_model import normalise, radial_gram, roughness_matrix, row_active, spectral_weights
    cfg = prob.cfg
    W0 = prob.model.rnl_Wnlq
    active = row_active(W0); Q = radial_gram(prob.model, ds)
    V = normalise(W0, Q, active)
    mask = density_mask(cfg, span)
    S = rho_gram(prob.model, V, mask, cfg, ds)
    H = normalise_rho(init_density(cfg, mask, P, seed), S)
    nq = W0.shape[-1]
    z = lambda: jnp.asarray(0.0)
    common = (to_array(THETA), prob.model, ds, density_gamma(prob.gamma, P, cfg.NZ), Q, active,
              roughness_matrix(prob.model), jnp.ones(W0.shape[2]), z(), V, spectral_weights(nq, 4.0), z(),
              jnp.zeros((cfg.NZ, cfg.NZ, nq, nq)), z(), S, mask, compact_gamma(prob.gamma, cfg),
              jnp.asarray(1e-3))
    shapes = (tuple(V.shape), tuple(H.shape))
    return V, H, common, shapes


def test_rd_gradient_matches_fd(small):
    from ace_jax.fit.radial_density import _objective_rd, _pack
    prob, ds, _ = small
    V, H, common, shapes = _rd_args(prob, ds)
    r = jnp.asarray([1.0, 3.0])
    x, other = _pack(V, H, r, "joint")
    f = lambda y: _objective_rd(y, other, *common, r, prob.cfg, "joint", shapes)
    g = jax.grad(f)(x)
    D = jnp.asarray(np.random.default_rng(0).standard_normal(x.shape))
    h = 1e-6
    fd = (float(f(x + h * D)) - float(f(x - h * D))) / (2 * h)
    assert abs(float(g @ D) - fd) < 1e-5 * abs(fd)


def test_rd_objective_gauge_and_precond_invariant(small):
    from ace_jax.fit.radial_density import _objective_rd, _pack
    prob, ds, _ = small
    V, H, common, shapes = _rd_args(prob, ds)
    one = jnp.asarray([1.0, 1.0])
    f = lambda V_, H_, r: float(_objective_rd(*_pack(V_, H_, r, "joint"), *common, r, prob.cfg, "joint", shapes))
    f0 = f(V, H, one)
    scale = jnp.asarray(np.random.default_rng(1).uniform(0.2, 5.0, H.shape[:2]))[..., None]
    np.testing.assert_allclose(f(V, H * scale, one), f0, rtol=1e-10)               # per-(p, z) scale gauge
    np.testing.assert_allclose(f(V, H.at[1].multiply(-1.0), one), f0, rtol=1e-10)  # sign of density p
    np.testing.assert_allclose(f(V, H, jnp.asarray([1.0, 7.3])), f0, rtol=1e-12)    # preconditioner


def _relabel_rd(prob, ds, W, eta, mask, c):
    from ace_jax.fit.radial_model import with_radial
    m = with_radial(prob.model, W)
    yE, yF, yV = [], [], []
    for i in range(ds.n_batches):
        r = _widened_rows(m, prob.cfg, jax.tree.map(lambda a: a[i], ds), eta, mask)
        yE.append(r.E @ c); yF.append(r.F @ c); yV.append(r.V @ c)
    return ds._replace(y_E=jnp.stack(yE), y_F=jnp.stack(yF), y_V=jnp.stack(yV))


def _density_truth(prob, ds, P=1, span="full", seed=5):
    """Targets from the true radials plus a density whose weights differ from the init."""
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import init_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    cfg = prob.cfg
    rng = np.random.default_rng(seed)
    W0 = prob.model.rnl_Wnlq
    Wt = normalise(W0, radial_gram(prob.model, ds), row_active(W0))
    mask = density_mask(cfg, span)
    S = rho_gram(prob.model, Wt, mask, cfg, ds)
    Ht = init_density(cfg, mask, P, 0) + 0.5 * jnp.asarray(rng.standard_normal((P, cfg.NZ, cfg.D))) * mask
    eta_t = normalise_rho(Ht, S)
    c = jnp.concatenate([0.1 * jnp.asarray(rng.standard_normal(cfg.len_basis)), jnp.full(P * cfg.NZ, 0.5)])
    return Wt, eta_t, mask, c


@pytest.mark.parametrize("mode,factor", [("joint", 0.3), ("alternating", 0.5)])
def test_learn_radial_density_reduces_objective_on_density_data(mode, factor):
    from ace_jax.fit.density import density_gamma, linear_density_statistics
    from ace_jax.fit.radial_density import init_density, learn_radial_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_learn import projected_residual_from_stats
    from ace_jax.fit.radial_model import with_radial
    prob, ds, _ = make_problem(ncfg=12, per_batch=3)
    Wt, eta_t, mask, c = _density_truth(prob, ds)
    ds = _relabel_rd(prob, ds, Wt, eta_t, mask, c)
    gw = density_gamma(prob.gamma, 1, prob.cfg.NZ)
    f = lambda W, eta: float(projected_residual_from_stats(
        THETA, linear_density_statistics(with_radial(prob.model, W), eta, mask, prob.cfg, ds), gw))
    S = rho_gram(prob.model, Wt, mask, prob.cfg, ds)
    f0 = f(Wt, normalise_rho(init_density(prob.cfg, mask, 1), S))
    V, eta, info = learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, P=1, mode=mode,
                                        theta0=THETA, profile=False, steps=30, reprofile_every=10)
    f1 = f(V, eta)
    print(f"{mode}: f0={f0:.4e} f1={f1:.4e} reasons={info['reasons']} precond={info['precond']}")
    assert f1 < factor * f0
    assert info["mode"] == mode and info["P"] == 1 and len(info["precond"]) >= 1


def test_learn_radial_density_zero_steps_is_normalised_init(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import init_density, learn_radial_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    prob, ds, _ = small
    mask = density_mask(prob.cfg, "full")
    V, eta, info = learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, P=2, theta0=THETA,
                                        profile=False, steps=0)
    W0 = prob.model.rnl_Wnlq
    Vr = normalise(W0, radial_gram(prob.model, ds), row_active(W0))
    np.testing.assert_array_equal(np.asarray(V), np.asarray(Vr))
    S = rho_gram(prob.model, Vr, mask, prob.cfg, ds)
    np.testing.assert_array_equal(np.asarray(eta), np.asarray(normalise_rho(init_density(prob.cfg, mask, 2), S)))
    assert info["steps"] == 0


def test_normalise_rho_unit_mean_square(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import init_density, normalise_rho, rho_gram
    prob, ds, _ = small
    mask = density_mask(prob.cfg, "full")
    S = rho_gram(prob.model, prob.model.rnl_Wnlq, mask, prob.cfg, ds)
    eta = normalise_rho(init_density(prob.cfg, mask, 2, seed=3), S)
    q = jnp.einsum("pzd,zde,pze->pz", eta, S, eta)
    np.testing.assert_allclose(np.asarray(q), 1.0, rtol=1e-10)


def test_learn_radial_density_single_compile_joint(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import learn_radial_density
    from ace_jax.fit.radial_learn import _lbfgs_step
    prob, ds, _ = small
    kw = dict(mask=density_mask(prob.cfg, "full"), P=1, theta0=THETA, profile=False, reprofile_every=2)
    learn_radial_density(prob, ds, prob.model.rnl_Wnlq, steps=2, **kw)          # warm the cache
    n0 = _lbfgs_step._cache_size()
    learn_radial_density(prob, ds, prob.model.rnl_Wnlq, steps=6, **kw)          # 3 rounds, new precond each
    assert _lbfgs_step._cache_size() == n0


def test_learn_radial_density_absent_species_is_zero_and_finite():
    """SiGe basis on Si-only data: Ge has no sites, so its density weights are exactly 0."""
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem
    from ace_jax.fit.radial_density import learn_radial_density
    from ace_jax.fit.radial_model import to_analytic
    spline, meta, z = load(FIXTURE_DIR / "sige_nofit.npz")
    model, _ = to_analytic(spline, 12)
    ds = build_dataset(load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:6], meta,
                       np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.ones(cfg.len_basis), default_prior(2.35))
    ge = list(meta["elements"]).index(32)
    V, eta, info = learn_radial_density(prob, ds, model.rnl_Wnlq, mask=density_mask(cfg, "full"), P=1,
                                        theta0=THETA, profile=False, steps=3, reprofile_every=3)
    assert np.all(np.isfinite(np.asarray(eta))) and np.all(np.isfinite(np.asarray(V)))
    assert np.all(np.asarray(eta)[:, ge] == 0.0)
    assert all(np.isfinite(info["trace"]))


def test_learn_radial_density_rejects_bad_options(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import learn_radial_density
    prob, ds, _ = small
    mask = density_mask(prob.cfg, "full")
    with pytest.raises(ValueError, match="mode"):
        learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, mode="nope", theta0=THETA, steps=1)
    with pytest.raises(ValueError, match="P"):
        learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, P=0, theta0=THETA, steps=1)
    with pytest.raises(ValueError, match="lam_eta"):
        learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, lam_eta=-1.0, theta0=THETA, steps=1)
```

The value 32 in the absent-species test is germanium's atomic number. If `meta["elements"]` holds symbols, use `"Ge"` instead.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_radial_density.py -q -p no:xdist -k "rd or radial_density or normalise_rho"`
  Expected: FAIL with an ImportError on `ace_jax.fit.radial_density`.

- [ ] **Step 3: Implement `src/ace_jax/fit/radial_density.py`.**

```python
"""Joint VarPro learning of the tensor radials W and P sqrt-density features.

The site energy is E_i = c_z . X_i + sum_p d[p, z] ssqrt(eta[p, z] . (mask * X_i))
(eval.fs_model).  At fixed (W, eta) it is linear in [c | d], so the readout is
projected out exactly as in radial_learn, now on the widened statistics
fit.density.linear_density_statistics; the outer variables are the radial V
and the raw density weights H, with eta = normalise_rho(H, S) fixing each
density's scale gauge (unit mean-square rho on the training sites, S the
per-species second moment at the initial radials -- the analogue of the radial
gauge Gram Q).  d absorbs the scale and the sign of each density.

The optimiser is L-BFGS on u = [vec(V) ; r_eta vec(H)] ("joint"), r_eta the
ratio of the blocks' RMS gradients recomputed every round (one extra gradient
evaluation), or alternating V / H blocks ("alternating", the fallback), both
through radial_learn's single compiled `_lbfgs_step`.  theta is re-profiled on
the widened LML after every round.  After learning, eta is frozen and the
final fit is the ordinary Bayesian linear solve over [c | d].
See docs/dev/specs/2026-09-28-radial-density-varpro-design.md.
"""
import math
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .data import flat_edges
from .density import compact_gamma, density_gamma, linear_density_statistics
from .hypers import from_array, to_array
from .radial_learn import (lbfgs_loop, projected_residual_from_stats, relative_lambda, relative_lambda_gap,
                           relative_lambda_spec, require_linear, require_x64, theta_map_linear)
from .radial_model import (data_r_range, gap_penalty, normalise, radial_gram, require_analytic, roughness,
                           roughness_matrix, row_active, spectral_penalty, spectral_weights, uniform_gram,
                           with_radial)

MODES = ("joint", "alternating")


def rho_gram(model, W, mask, cfg, ds):
    """Per-species second moment of the masked compact basis at radials W,
    S[z] = sum_{i: z_i = z} (m X_i)(m X_i)^T / n_z, (NZ, D, D); one streaming pass.
    A species with no sites gets S[z] = 0."""
    m = with_radial(model, W)

    def body(carry, b):
        S, n = carry
        Ncap = b.nbr.shape[0]
        rij, send, recv, msk = flat_edges(b.rij, b.nbr, b.nbr_mask)
        X = m.compact_basis(rij, b.node_z[send], b.node_z[recv], send, Ncap, msk) * mask
        oh = jax.nn.one_hot(b.node_z, cfg.NZ) * b.node_mask[:, None]           # (Ncap, NZ)
        return (S + jnp.einsum("nz,nd,ne->zde", oh, X, X), n + oh.sum(0)), None

    D = cfg.D
    (S, n), _ = jax.lax.scan(body, (jnp.zeros((cfg.NZ, D, D)), jnp.zeros(cfg.NZ)), ds)
    return S / jnp.maximum(n, 1.0)[:, None, None]


def normalise_rho(H, S):
    """eta = H scaled per (p, z) to unit mean-square density eta^T S_z eta = 1;
    exactly zero where the second moment vanishes (absent species / dead weights)."""
    q = jnp.einsum("pzd,zde,pze->pz", H, S, H)
    live = q > 0
    return H * jnp.where(live, jax.lax.rsqrt(jnp.where(live, q, 1.0)), 0.0)[..., None]


def init_density(cfg, mask, P, seed=0):
    """p = 0: equal weights on the pair block for every species (the total pair
    density, the FS starting point) -- zero on the B block even under the full
    mask; p >= 1: that plus a seeded 0.1-scale perturbation over the mask, so the
    P columns start distinct."""
    pair = jnp.concatenate([jnp.zeros(cfg.n_B), jnp.ones(cfg.n_pair)]) * mask
    H = jnp.broadcast_to(pair, (P, cfg.NZ, cfg.D))
    if P > 1:
        noise = 0.1 * jnp.asarray(np.random.default_rng(seed).standard_normal((P - 1, cfg.NZ, cfg.D))) * mask
        H = H.at[1:].add(noise)
    return jnp.asarray(H)


def relative_lambda_eta(lam_eta, r0, n):
    """Absolute density-prior weight lam_eta * r0 / n, n = P * NZ densities."""
    if lam_eta < 0:
        raise ValueError(f"lam_eta must be >= 0, got {lam_eta}")
    return float(lam_eta) * r0 / n if lam_eta else 0.0


def _pack(V, H, r, block):
    """(xb, other): the optimised block and the frozen rest, preconditioned."""
    v, h = r[0] * V.ravel(), r[1] * H.ravel()
    if block == "joint":
        return jnp.concatenate([v, h]), jnp.zeros(0)
    return (v, h) if block == "V" else (h, v)


def _unpack(xb, other, r, block, shapes):
    (sV, sH) = shapes
    nV = math.prod(sV)
    x = xb if block == "joint" else (jnp.concatenate([xb, other]) if block == "V"
                                     else jnp.concatenate([other, xb]))
    return (x[:nV] / r[0]).reshape(sV), (x[nV:] / r[1]).reshape(sH)


def _objective_rd(xb, other, a, model, ds, gamma_w, Q, active, D2, wn, lam, W_ref, sw, lam_spec,
                  U, lam_gap, S, mask, gc, lam_eta, r, cfg, block, shapes):
    """VarPro residual of the widened design plus the radial priors and the density
    shape prior lam_eta * sum ||Gamma_z * eta_{p,z}||^2.  Module-level so it is a
    stable `f` for `_lbfgs_step` (one compile per block kind); cfg, block and
    shapes are statics, r (the preconditioner) is traced."""
    V, H = _unpack(xb, other, r, block, shapes)
    W = normalise(V, Q, active)
    eta = normalise_rho(H, S)
    lin = linear_density_statistics(with_radial(model, W), eta, mask, cfg, ds)
    return (projected_residual_from_stats(from_array(a), lin, gamma_w) + lam * roughness(W, D2, wn)
            + lam_spec * spectral_penalty(W, W_ref, sw) + lam_gap * gap_penalty(W, W_ref, U)
            + lam_eta * jnp.sum((gc[None] * eta) ** 2))


_grad_rd = jax.jit(jax.grad(_objective_rd), static_argnums=(21, 22, 23))


def _block_scale(g, nV):
    """r_eta = RMS(grad_H) / RMS(grad_V), so both blocks' gradients in u have the same
    RMS; 1 when either is zero or non-finite."""
    gV, gH = np.asarray(g[:nV]), np.asarray(g[nV:])
    rv, rh = float(np.sqrt(np.mean(gV ** 2))), float(np.sqrt(np.mean(gH ** 2)))
    ok = np.isfinite(rv) and np.isfinite(rh) and rv > 0 and rh > 0
    return rh / rv if ok else 1.0


def learn_radial_density(prob, ds, W0, *, mask, P=1, H0=None, mode="joint", theta0=None, profile=True,
                         lam_rough=0.0, rough_weights=None, lam_spec=0.0, spec_p=4.0, lam_gap=0.0,
                         lam_eta=0.0, steps=40, reprofile_every=10, tol=1e-6, patience=3, map_steps=300,
                         seed=0, log=None, Q=None, D2=None, U=None, S=None, r0=None):
    """VarPro-learn radials W and P density weights eta jointly (M = 0).  Same
    round structure, relative priors and stopping rules as radial_learn.learn_radial;
    lam_eta is relative too (relative_lambda_eta).  mode "joint" optimises
    [V; H] together, "alternating" spends each round's steps half on V (H
    fixed) then half on H.  Q, D2, U as learn_radial; S = rho_gram at the
    normalised init (computed when None); r0 = the widened projected residual at
    the start (computed when None; only valid with theta0).  Returns
    (V, eta, info), both normalised; steps = 0 returns the normalised init."""
    require_x64()
    require_analytic(prob.model)
    require_linear(prob)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if P < 1:
        raise ValueError(f"P must be >= 1 (P = 0 is radial_learn.learn_radial), got {P}")
    cfg = prob.cfg
    mask = jnp.asarray(mask, jnp.float64)
    prob_w = prob._replace(gamma=density_gamma(prob.gamma, P, cfg.NZ))
    W0 = jnp.asarray(W0, jnp.float64)
    active = row_active(W0)
    if Q is None:
        Q = radial_gram(prob.model, ds)
    if D2 is None:
        D2 = roughness_matrix(prob.model)
    NZw, n_q = W0.shape[0], W0.shape[-1]
    if U is None:
        if lam_gap:
            r_min, _ = data_r_range(ds)
            U = uniform_gram(prob.model, 0.8 * r_min, cfg.rcut)
        else:
            U = jnp.zeros((NZw, NZw, n_q, n_q), jnp.float64)
    wn = jnp.ones(W0.shape[2]) if rough_weights is None else jnp.asarray(rough_weights, jnp.float64)
    V = normalise(W0, Q, active)
    W_ref, sw = V, spectral_weights(n_q, spec_p)
    if S is None:
        S = rho_gram(prob.model, V, mask, cfg, ds)
    H = normalise_rho(init_density(cfg, mask, P, seed) if H0 is None else jnp.asarray(H0, jnp.float64), S)
    gc = compact_gamma(prob.gamma, cfg)
    stats = lambda V_, H_: linear_density_statistics(with_radial(prob.model, V_), normalise_rho(H_, S), mask, cfg, ds)
    lin0 = None
    if theta0 is not None:
        a = to_array(theta0)
    elif profile:
        lin0 = stats(V, H)
        a = theta_map_linear(prob_w, ds, None, steps=map_steps, seed=seed, lin=lin0)
    else:
        raise ValueError("learn_radial_density: profile=False needs theta0")
    if r0 is None:
        lin0 = stats(V, H) if lin0 is None else lin0
        r0 = projected_residual_from_stats(from_array(a), lin0, prob_w.gamma)
    r0 = float(r0)
    lam = relative_lambda(lam_rough, r0, float(roughness(V, D2, wn)))
    n_active = int(jnp.sum(active))
    lam_spec_abs = relative_lambda_spec(lam_spec, r0, n_active)
    lam_gap_abs = relative_lambda_gap(lam_gap, r0, n_active)
    lam_eta_abs = relative_lambda_eta(lam_eta, r0, P * cfg.NZ)
    info = {"trace": [], "reasons": [], "round_lengths": [], "theta": [np.asarray(a)], "precond": [],
            "r0": r0, "lam_abs": lam, "lam_spec_abs": lam_spec_abs, "lam_gap_abs": lam_gap_abs,
            "lam_eta": float(lam_eta), "lam_eta_abs": lam_eta_abs, "P": int(P), "mode": mode, "steps": 0}
    if log is not None:
        log(f"learn_radial_density: P={P} mode={mode} lam_eta={float(lam_eta):g} "
            f"lam_eta_abs={lam_eta_abs:.6e} r0={r0:.6e}")
    f64 = lambda v: jnp.asarray(v, jnp.float64)
    shapes = (tuple(V.shape), tuple(H.shape))
    one = f64([1.0, 1.0])
    done, round_idx = 0, 0
    while done < int(steps):
        round_idx += 1
        n = min(int(reprofile_every), int(steps) - done)
        t_round = time.perf_counter()
        common = (a, prob.model, ds, prob_w.gamma, Q, active, D2, wn, f64(lam), W_ref, sw, f64(lam_spec_abs),
                  U, f64(lam_gap_abs), S, mask, gc, f64(lam_eta_abs))
        g = _grad_rd(*_pack(V, H, one, "joint"), *common, one, cfg, "joint", shapes)
        r = f64([1.0, _block_scale(g, V.size)])
        info["precond"].append(float(r[1]))
        blocks = [("joint", n)] if mode == "joint" else [("V", (n + 1) // 2), ("H", n // 2)]
        reasons, obj = [], float("nan")
        for block, nb in blocks:
            if nb == 0:
                continue
            xb, other = _pack(V, H, r, block)
            xb, obj, trace, reason = lbfgs_loop(_objective_rd, xb, steps=nb, tol=tol, patience=patience,
                                                args=(other, *common, r), statics=(cfg, block, shapes))
            V, H = _unpack(xb, other, r, block, shapes)
            info["trace"].extend(trace)
            info["round_lengths"].append(len(trace))
            reasons.append(reason)
        info["reasons"].append(reasons[0] if len(reasons) == 1 else "+".join(reasons))
        done += n
        V, H = normalise(V, Q, active), normalise_rho(H, S)
        if profile:
            a = theta_map_linear(prob_w, ds, None, steps=map_steps, seed=seed, init=a, lin=stats(V, H))
            info["theta"].append(np.asarray(a))
        if log is not None:
            log(f"learn_radial_density: round {round_idx} steps={done}/{int(steps)} obj={obj:.6e} "
                f"reasons={reasons} precond={float(r[1]):.3e} time={time.perf_counter() - t_round:.1f}s")
        if "nonfinite" in reasons or all(x != "steps" for x in reasons):
            break
    info["steps"] = done
    info["theta_final"] = np.asarray(a)
    return V, H, info
```

**Notes for the implementer.**

- `_grad_rd`'s static positions (21, 22, 23) are `cfg`, `block` and `shapes` in `_objective_rd`'s positional order. Recount them if you change the signature.
- `H` is already normalised when it is returned (`H = normalise_rho(...)` at the end of every round, and at the start), so it is `eta`.
- The zero-step test requires `normalise_rho(H)` of the init to be returned bit-identically. This holds because no round runs when `steps == 0`.
- Check where `flat_edges` lives: `from .data import flat_edges` matches `tests/test_gp_density.py`.
- If the joint single-compile test shows a second compile, the cause is a weak-typed argument. Make sure every scalar in `common` and `r` goes through `f64(...)`.
- If `_lbfgs_step._cache_size()` doesn't exist in the installed JAX, mirror whatever `test_lbfgs_loop_no_recompile_across_rounds` in `tests/test_gp_learn_radial.py` uses.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run: `uv run pytest tests/test_radial_density.py -q -p no:xdist -s`
  Expected: PASS, with the printed f0/f1 showing the objective reduced.

  If the alternating recovery factor 0.5 isn't reached in 30 steps, print f1/f0 and record the ratio in the report. Only loosen the factor to the measured value plus a margin if that value is well below 1, and explain why in the report. The joint factor 0.3 matches the radial recovery test and must not be loosened without an explanation.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/fit/radial_density.py tests/test_radial_density.py
git commit -m "feat(fit): joint radial + density VarPro (learn_radial_density)"
```

---

### Task 5: `fit_radial_density` (the gate) and the density-aware `save_result`

**Files:**
- Modify: `src/ace_jax/fit/radial_density.py`
- Modify: `src/ace_jax/fit/radial_learn.py` (`save_result`)
- Test: `tests/test_radial_density.py`

**Interfaces:**
- Consumes: `learn_radial` and `gate` from `radial_learn`; `holdout_score(None, ..., lin_fit=, lin_val=, return_readout=True)` from Task 1; `patch_radial_npz(..., fs=)` from Task 3.
- Produces:
  - `fit_radial_density(prob, ds_fit, ds_val, W0, *, mask, P=1, mode="joint", lam_eta_grid=(0.0,), lam_rough=0.0, lam_spec=0.0, lam_gap=0.0, theta0=None, map_steps=300, log=None, checkpoint=None, **learn_kw) -> (W, eta_or_None, info)`.
  - The candidate labels are `"init"`, `"radials_only"` and `"density_lam_eta=<l:g>"`.
  - `info` has the keys `selected`, `scores`, `theta_fit`, `map_diag`, `runs`, `theta_init`, `readout` (length len_basis + P·NZ if a density is selected, else len_basis), `P` (0 if not selected), `mask` (a list of floats) and `mode`.
  - `checkpoint(key, W, eta_or_None, run_info)`.
  - `save_result(out_dir, W, info, *, src_npz=None, model=None, readout=None, eta=None, mask=None)` writes `eta.npy`, and writes the fs keys to model.npz when `eta` is given.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_radial_density.py`:

```python
def test_fit_radial_density_gate_prefers_density_on_density_data():
    from ace_jax.fit.radial_density import fit_radial_density
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, eta_t, mask, c = _density_truth(prob, ds_fit)
    ds_fit, ds_val = _relabel_rd(prob, ds_fit, Wt, eta_t, mask, c), _relabel_rd(prob, ds_val, Wt, eta_t, mask, c)
    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, mask=mask, P=1,
                                      theta0=None, profile=False, steps=20, map_steps=50)
    print(info["scores"])
    assert info["selected"].startswith("density") and eta is not None and info["P"] == 1
    assert info["readout"].shape == (prob.cfg.len_basis + prob.cfg.NZ,)
    assert set(info["scores"]) == {"init", "radials_only", "density_lam_eta=0"}


def test_fit_radial_density_gate_rejects_density_on_linear_data():
    from test_gp_learn_radial import _perturbed_truth, relabel
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import fit_radial_density
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, W0, c = _perturbed_truth(prob)
    ds_fit, ds_val = relabel(prob, ds_fit, Wt, c), relabel(prob, ds_val, Wt, c)
    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, W0, mask=density_mask(prob.cfg, "full"), P=1,
                                      theta0=None, profile=False, steps=20, map_steps=50)
    print(info["scores"])
    assert info["selected"] in ("init", "radials_only") and eta is None and info["P"] == 0
    assert info["readout"].shape == (prob.cfg.len_basis,)


def test_saved_density_model_matches_widened_rows(tmp_path):
    """model.npz from save_result(eta=...) evaluates to [linear | density] rows @ readout + E0."""
    from ace_jax.eval import load
    from ace_jax.eval.fs_model import FSModel
    from ace_jax.fit.radial_density import fit_radial_density
    from ace_jax.fit.radial_learn import save_result
    from ace_jax.fit.radial_model import with_radial
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, eta_t, mask, c = _density_truth(prob, ds_fit)
    ds_fit, ds_val = _relabel_rd(prob, ds_fit, Wt, eta_t, mask, c), _relabel_rd(prob, ds_val, Wt, eta_t, mask, c)
    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, mask=mask, P=1,
                                      theta0=None, profile=False, steps=5, map_steps=20)
    assert eta is not None
    save_result(tmp_path, W, info, src_npz=MODEL, model=prob.model, eta=eta, mask=mask)
    back, meta, z = load(tmp_path / "model.npz")
    assert isinstance(back, FSModel) and np.load(tmp_path / "eta.npy").shape == eta.shape
    E0 = np.asarray(z["E0"]); m = with_radial(prob.model, W); cr = jnp.asarray(info["readout"])
    worst = 0.0
    for i in range(ds_val.n_batches):
        b = jax.tree.map(lambda a: a[i], ds_val)
        r = _widened_rows(m, prob.cfg, b, eta, mask)
        C = b.y_E.shape[0]; Ncap, K = b.nbr.shape
        zi = jnp.broadcast_to(b.node_z[:, None], (Ncap, K))
        e = back.site_energies_dense(b.rij, zi, b.node_z[b.nbr], b.nbr_mask, b.node_z)
        E_model = np.asarray(jax.ops.segment_sum(e, b.node_cfg, num_segments=C + 1)[:C])
        E0sum = np.asarray(jax.ops.segment_sum(jnp.where(b.node_mask, jnp.asarray(E0)[b.node_z], 0.0),
                                               b.node_cfg, num_segments=C + 1)[:C])
        E_lin = np.asarray(r.E @ cr) + E0sum
        worst = max(worst, float(np.max(np.abs(E_model - E_lin) / np.abs(E_lin))))
    assert worst < 1e-8


def test_fit_radial_density_checkpoints(tmp_path, small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import fit_radial_density
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    seen = []
    fit_radial_density(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, mask=density_mask(prob.cfg, "pair"),
                       P=1, lam_eta_grid=(0.0, 1e-2), theta0=THETA, profile=False, steps=2, map_steps=20,
                       checkpoint=lambda k, W, eta, info: seen.append((k, eta is None)))
    assert seen == [("radials_only", True), ("density_lam_eta=0", False), ("density_lam_eta=0.01", False)]
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_radial_density.py -q -p no:xdist -k "fit_radial_density or saved_density"`
  Expected: FAIL with an ImportError on `fit_radial_density`.

- [ ] **Step 3: Implement.** Append to `src/ace_jax/fit/radial_density.py`, adding `from .radial_learn import gate, holdout_score, learn_radial` and `from .stats import linear_statistics` to the imports:

```python
def fit_radial_density(prob, ds_fit, ds_val, W0, *, mask, P=1, mode="joint", lam_eta_grid=(0.0,),
                       lam_rough=0.0, lam_spec=0.0, lam_gap=0.0, theta0=None, map_steps=300, log=None,
                       checkpoint=None, **learn_kw):
    """Held-out gate over {init, radials_only, density_lam_eta=<l> per l}: the
    radials alone (radial_learn.learn_radial) and the joint radials + density
    (learn_radial_density) from the same start and step budget, each candidate
    scored by radial_learn.fit_radial's one procedure (theta re-MAP on ds_fit
    warm-started at a0, posterior-mean readout, sigma-normalised SSE on ds_val
    with sigma from a0) on its own design width -- so density is kept only when
    it wins on held-out data; ties go to the earlier, simpler candidate.
    Returns (W, eta or None, info); info["readout"] is the selected readout,
    [c | d] (len_basis + P * NZ) for a density candidate."""
    require_x64()
    require_analytic(prob.model)
    require_linear(prob)
    if learn_kw.pop("learn_sigma_e_mult", 1.0) != 1.0:
        raise ValueError("fit_radial_density: learn_sigma_e_mult is not supported with a density")
    cfg = prob.cfg
    mask = jnp.asarray(mask, jnp.float64)
    W0 = jnp.asarray(W0, jnp.float64)
    n_prior = learn_kw.pop("n_prior", None)
    Q = radial_gram(prob.model, ds_fit) if n_prior is None else radial_gram(prob.model, ds_fit, n_prior=n_prior)
    D2 = roughness_matrix(prob.model)
    U = None
    if lam_gap:
        r_min, _ = data_r_range(ds_fit)
        U = uniform_gram(prob.model, 0.8 * r_min, cfg.rcut)
    W_init = normalise(W0, Q, row_active(W0))
    if theta0 is not None:
        a0 = to_array(theta0)
        lin0 = linear_statistics(with_radial(prob.model, W_init), cfg, ds_fit)
    else:
        a0, lin0, _ = theta_map_linear(prob, ds_fit, W_init, steps=map_steps, return_stats=True)
    r0 = float(projected_residual_from_stats(from_array(a0), lin0, prob.gamma))
    S = rho_gram(prob.model, W_init, mask, cfg, ds_fit)
    prob_w = prob._replace(gamma=density_gamma(prob.gamma, P, cfg.NZ))
    cands, runs = {"init": (W_init, None)}, {}
    kw = dict(lam_rough=lam_rough, lam_spec=lam_spec, lam_gap=lam_gap, map_steps=map_steps, log=log,
              Q=Q, D2=D2, U=U, **learn_kw)
    W_r, info_r = learn_radial(prob, ds_fit, W0, theta0=from_array(a0), r0=r0, **kw)
    cands["radials_only"], runs["radials_only"] = (W_r, None), info_r
    if checkpoint is not None:
        checkpoint("radials_only", W_r, None, info_r)
    for l in lam_eta_grid:
        key = f"density_lam_eta={float(l):g}"
        if log is not None:
            log(f"fit_radial_density: {key} starting")
        W, eta, info = learn_radial_density(prob, ds_fit, W0, mask=mask, P=P, mode=mode, theta0=from_array(a0),
                                            lam_eta=l, S=S, **kw)
        cands[key], runs[key] = (W, eta), info
        if checkpoint is not None:
            checkpoint(key, W, eta, info)
    theta_fit, map_diag, readouts = {}, {}, {}

    def score(label, cand):
        W, eta = cand
        pw = prob if eta is None else prob_w
        st = ((lambda d: linear_statistics(with_radial(prob.model, W), cfg, d)) if eta is None else
              (lambda d: linear_density_statistics(with_radial(prob.model, W), eta, mask, cfg, d)))
        lin_fit = st(ds_fit)
        a_fit, _, diag = theta_map_linear(pw, ds_fit, None, steps=map_steps, init=a0, return_stats=True,
                                          lin=lin_fit)
        s, c = holdout_score(None, a_fit, a0, pw, ds_fit, ds_val, lin_fit=lin_fit, lin_val=st(ds_val),
                             return_readout=True)
        theta_fit[label], map_diag[label], readouts[label] = np.asarray(a_fit), diag, np.asarray(c)
        if log is not None:
            log(f"fit_radial_density: gate {label} score={s:.6e}")
        return s

    label, scores = gate(cands, score)
    W_sel, eta_sel = cands[label]
    if log is not None:
        log(f"fit_radial_density: selected {label}")
    return W_sel, eta_sel, {"selected": label, "scores": scores, "theta_fit": theta_fit, "map_diag": map_diag,
                            "runs": runs, "theta_init": np.asarray(a0), "readout": readouts[label],
                            "P": 0 if eta_sel is None else int(P), "mask": np.asarray(mask).tolist(),
                            "mode": mode}
```

`theta_map_linear(..., return_stats=True, lin=...)` returns `(a, lin, diag)`, which is why it is unpacked as `a_fit, _, diag`.

In `radial_learn.save_result`, add `eta=None, mask=None` to the signature. Extend the docstring: "`eta`/`mask`: the frozen density of a fit_radial_density selection; then the readout is [c | d] (len_basis + P * NZ), eta is also saved as eta.npy, and model.npz carries the fs keys (eval.fs_model)." Replace the final `patch_radial_npz(...)` call, and add the `eta.npy` save next to `readout.npy`:

```python
    if eta is not None:
        np.save(out / "eta.npy", np.asarray(eta))
    if src_npz is not None:
        from ..construct.export import patch_radial_npz
        fs = None
        if eta is not None:
            from .density import EPS
            eta = np.asarray(eta)
            k = eta.shape[0] * eta.shape[1]
            readout = np.asarray(readout)
            fs = (eta, readout[-k:].reshape(eta.shape[0], eta.shape[1]), np.asarray(mask), EPS)
            readout = readout[:-k]
        patch_radial_npz(src_npz, out / "model.npz", with_radial(model, W), readout=readout, fs=fs)
```

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run: `uv run pytest tests/test_radial_density.py tests/test_gp_learn_radial.py -q -p no:xdist -s -k "fit_radial or save"`
  Expected: PASS.

  If the linear-data gate test selects a density candidate, look at the printed scores before concluding anything. On noiseless linear data the density column should be useless, and its extra parameters should score at best equal. If the density scores are lower, report them rather than weakening the test: it would mean the gate or the prior is wrong.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/fit/radial_density.py src/ace_jax/fit/radial_learn.py tests/test_radial_density.py
git commit -m "feat(fit): radial+density held-out gate and FS model export"
```

---

### Task 6: Driver flags, a model-level RMSE script and Modal pass-through

**Files:**
- Modify: `bench/learn_radial/run.py`
- Create: `bench/learn_radial/rmse_npz.py`
- Modify: `bench/learn_radial/modal_run.py`
- Test: `tests/test_radial_density.py`

**Interfaces:**
- Consumes: `fit_radial_density`, `save_result(eta=, mask=)` and `density_mask`.
- Produces:
  - `run.py` gains the flags `--density {none,pair,full}` (default none), `--P` (default 1), `--density-mode {joint,alternating}` and `--lam-eta-grid` (default "0").
  - `summary.json` gains `density`, `P_selected` and `density_mode`.
  - `rmse_npz.py --model-npz X.npz` (repeatable) prints and writes validation E/F RMSE and MAE for exported models, plain or FS, on the same split as `run.py`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_radial_density.py`:

```python
def _driver(tmp_path, *extra, model=MODEL):
    import subprocess, sys
    from conftest import ROOT
    return subprocess.run(
        [sys.executable, str(ROOT / "bench/learn_radial/run.py"), "--model", str(model),
         "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
         "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8", "--batch", "4",
         "--n-q", "20", "--steps", "3", "--lam-grid", "0", "--map-steps", "20",
         "--out", str(tmp_path), *extra], capture_output=True, text=True)


def test_bench_driver_density_smoke(tmp_path):
    import json, subprocess, sys
    from conftest import ROOT
    r = _driver(tmp_path, "--density", "full", "--P", "1")
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["density"] == "full" and s["selected"] in s["scores"] and "radials_only" in s["scores"]
    assert (tmp_path / "model.npz").exists()
    r2 = subprocess.run([sys.executable, str(ROOT / "bench/learn_radial/rmse_npz.py"), "--model", str(MODEL),
                         "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
                         "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8",
                         "--model-npz", str(tmp_path / "model.npz"), "--out", str(tmp_path / "rmse.json")],
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-3000:]
    js = json.loads((tmp_path / "rmse.json").read_text())
    (row,) = js.values()
    assert np.isfinite(row["E_rmse_meV_atom"]) and np.isfinite(row["F_rmse_meV_A"])


def test_bench_driver_density_rejects_grids(tmp_path):
    r = _driver(tmp_path, "--density", "full", "--spec-grid", "0,1e-5")
    assert r.returncode != 0 and "single-valued" in (r.stderr + r.stdout)


def test_bench_driver_unwraps_density_model(tmp_path):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval import load
    base, meta, _ = load(MODEL)
    NZ, D = len(meta["elements"]), meta["n_B"] + meta["n_pair"]
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base,
                     fs=(1e-2 * np.ones((1, NZ, D)), np.ones((1, NZ)), np.ones(D), 1e-6))
    r = _driver(tmp_path / "out", model=tmp_path / "fs.npz")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "density term of the input model is dropped" in r.stdout
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_radial_density.py -q -p no:xdist -k bench_driver`
  Expected: FAIL (unrecognised arguments / `rmse_npz.py` missing).

- [ ] **Step 3: Implement `run.py`.** First add these arguments after `--learn-sigma-e-mult`:

```python
p.add_argument("--density", choices=["none", "pair", "full"], default="none",
               help="also learn sqrt-density features jointly with the radials over this span of the basis "
                    "(docs/dev/specs/2026-09-28-radial-density-varpro-design.md); the gate keeps them only if "
                    "they win on the held-out split")
p.add_argument("--P", type=int, default=1, help="number of density features (with --density)")
p.add_argument("--density-mode", choices=["joint", "alternating"], default="joint")
p.add_argument("--lam-eta-grid", default="0", help="relative density shape-prior weights (with --density)")
```

After `model, meta, z = load(a.model)`, add:

```python
if hasattr(model, "base"):              # an FSModel: learning starts from its linear part
    model = model.base
    print("note: the density term of the input model is dropped (it belongs to the old radials)", flush=True)
```

Replace the `W, info = fit_radial(...)` call and the `save_result(...)` call with the following. Keep the existing `checkpoint` function for the non-density path.

```python
if a.density == "none":
    W, info = fit_radial(prob, ds_fit, ds_val, model.rnl_Wnlq, lam_grid=lam_grid, spec_grid=spec_grid,
                         gap_grid=gap_grid, rough_weights=wn, spec_p=a.spec_p, steps=a.steps,
                         reprofile_every=a.reprofile_every, map_steps=a.map_steps,
                         learn_sigma_e_mult=a.learn_sigma_e_mult,
                         log=lambda s: print(s, flush=True), checkpoint=checkpoint)
    eta = mask = None
else:
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import fit_radial_density
    if max(len(lam_grid), len(spec_grid), len(gap_grid)) > 1:
        raise SystemExit("--density needs single-valued --lam-grid, --spec-grid and --gap-grid "
                         "(sweep --lam-eta-grid instead)")
    if a.learn_sigma_e_mult != 1.0:
        raise SystemExit("--density does not support --learn-sigma-e-mult")
    mask = density_mask(cfg, a.density)

    def checkpoint_rd(label, W_c, eta_c, run_info):
        save_result(out / label, W_c, run_info, eta=eta_c)
        print(f"checkpoint: {out / label}", flush=True)

    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, model.rnl_Wnlq, mask=mask, P=a.P,
                                      mode=a.density_mode,
                                      lam_eta_grid=tuple(float(x) for x in a.lam_eta_grid.split(",")),
                                      lam_rough=lam_grid[0], lam_spec=spec_grid[0], lam_gap=gap_grid[0],
                                      rough_weights=wn, spec_p=a.spec_p, steps=a.steps,
                                      reprofile_every=a.reprofile_every, map_steps=a.map_steps,
                                      log=lambda s: print(s, flush=True), checkpoint=checkpoint_rd)
info["to_analytic_relres_max"] = relres_max
save_result(out, W, info, src_npz=a.model, model=model, eta=eta, mask=mask)
```

Add these keys to `summary`: `"density": a.density, "P_selected": int(info.get("P", 0)), "density_mode": a.density_mode, "lam_eta_grid": a.lam_eta_grid`.

The `radials_only` checkpoint in `checkpoint_rd` calls `save_result(..., eta=None)`, which writes only `rnl_Wnlq.npy` and `radial_info.json`, since there is no `src_npz`. Update the module docstring: add the `--density` usage line and the fact that density runs checkpoint to `DIR/radials_only/` and `DIR/density_lam_eta=<l>/` (each with `rnl_Wnlq.npy`, and `eta.npy` for density runs).

- [ ] **Step 4: Implement `bench/learn_radial/rmse_npz.py`.**

```python
"""Validation errors of exported models (model.npz, plain or with a frozen
sqrt-density term) on the run.py split: energy RMSE/MAE in meV/atom and force
RMSE/MAE in meV/Å, evaluated with the model itself (energy_forces_virial), so
it measures exactly what qoi.py and MD will use.

    uv run python bench/learn_radial/rmse_npz.py --model BASE.npz --data D.xyz \
        --model-npz runs/x/model.npz [--model-npz ...] --out rmse_npz.json
--model (the run's input model) only fixes the element order and E0 for the
dataset, exactly as run.py builds it.
"""
import argparse
import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--energy-key", default="energy"); p.add_argument("--force-key", default="forces")
p.add_argument("--virial-key", default="virial")
p.add_argument("--ntrain", type=int, default=200); p.add_argument("--nval", type=int, default=200)
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--model-npz", action="append", required=True); p.add_argument("--out", required=True)
a = p.parse_args()

_, meta, z = load(a.model)
NZ = len(meta["elements"])
configs = load_configs(a.data, a.energy_key, a.force_key, a.virial_key)
perm = np.random.default_rng(a.seed).permutation(len(configs))
E0 = np.asarray(z["E0"]) if "E0" in z else np.zeros(NZ)
ds = build_dataset([configs[i] for i in perm[a.ntrain:a.ntrain + a.nval]], meta, E0, configs_per_batch=a.batch)


@jax.jit
def efv(m, b):
    """Per-config energies (C,) and per-node forces (Ncap, 3) of one padded batch."""
    Ncap, K = b.nbr.shape
    C = b.y_E.shape[0]
    zi = jnp.broadcast_to(b.node_z[:, None], (Ncap, K))
    E, F, _ = m.energy_forces_virial_dense(b.rij, zi, b.node_z[b.nbr], b.nbr, b.nbr_mask, b.node_z)
    e = m.site_energies_dense(b.rij, zi, b.node_z[b.nbr], b.nbr_mask, b.node_z)
    return jax.ops.segment_sum(e, b.node_cfg, num_segments=C + 1)[:C], F


res = {}
for path in a.model_npz:
    m, _, _ = load(path)
    dE, dF = [], []
    with highest_precision():
        for i in range(ds.n_batches):
            b = jax.tree.map(lambda x: x[i], ds)
            Ec, F = (np.asarray(x) for x in efv(m, b))
            E0sum = np.asarray(jax.ops.segment_sum(jnp.where(b.node_mask, jnp.asarray(E0)[b.node_z], 0.0),
                                                   b.node_cfg, num_segments=b.y_E.shape[0] + 1)[:b.y_E.shape[0]])
            live_E = np.asarray(b.cfg_mask) & (np.asarray(b.w_E) > 0)
            n = np.maximum(np.asarray(b.n_atoms), 1)
            dE.append(((Ec - E0sum - np.asarray(b.y_E)) / n)[live_E])
            live_F = np.asarray(b.node_mask) & (np.asarray(b.w_F) > 0)
            dF.append((F - np.asarray(b.y_F))[live_F].ravel())
    dE, dF = np.concatenate(dE), np.concatenate(dF)
    res[str(path)] = {"E_rmse_meV_atom": float(1e3 * np.sqrt(np.mean(dE ** 2))),
                      "E_mae_meV_atom": float(1e3 * np.mean(np.abs(dE))),
                      "F_rmse_meV_A": float(1e3 * np.sqrt(np.mean(dF ** 2))),
                      "F_mae_meV_A": float(1e3 * np.mean(np.abs(dF)))}
    print(f"{path}: E {res[str(path)]['E_rmse_meV_atom']:.3f} meV/atom  F {res[str(path)]['F_rmse_meV_A']:.2f} meV/A",
          flush=True)
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
```

**Check before running.** `build_dataset` targets: confirm from `fit/data.py` whether `y_E` has E0 already subtracted. `run.py` passes E0 to `build_dataset`, so `y_E` is very likely the E0-subtracted binding energy; then subtracting `E0sum` from the model's total energy, as above, is right. If `y_E` holds total energies instead, drop the `- E0sum`. Also confirm the `energy_forces_virial_dense` argument order: it is `(rij, zi, zj, idx, mask, node_z)`, where `idx` is the neighbour index `b.nbr`. Use the `tests/test_gp_learn_radial.py::test_saved_model_energies_match_fitted_readout` pattern as the reference for energies.

Sanity check: for the plain `model.npz` from a non-density run, the RMSE here must equal `rmse.py`'s validation E/F RMSE for the selected candidate to about 1e-6 relative. Check this once by hand on the smoke output and state the result in the report.

- [ ] **Step 5: Implement the `modal_run.py` pass-through.** Add `density: str = "none", P: int = 1, density_mode: str = "joint", lam_eta_grid: str = "0"` to both `learn(...)` and `main(...)`. In `learn`, append `["--density", density, "--P", str(P), "--density-mode", density_mode, "--lam-eta-grid", lam_eta_grid]` to the `run.py` command. The output dir must include the density configuration: `out = pathlib.Path(f"/tmp/out_{system}_s{steps}_m{mult:g}_{density}{P}")`.

After `rmse.py`, also run `rmse_npz.py` on `out / "model.npz"`, writing `out / "rmse_npz.json"`. In the `rmse.py` candidate glob, when `density != "none"`, use `out.glob("*/rnl_Wnlq.npy")` (the checkpoint dirs are `radials_only` and `density_*`); keep `lam_*` otherwise. In `main`, pass the new args through `calls`. The local directory name gets the suffix `_{density}{P}` when `density != "none"`.

- [ ] **Step 6: Run the tests and confirm they pass.**
  Run: `uv run pytest tests/test_radial_density.py tests/test_gp_learn_radial.py -q -p no:xdist -k "bench_driver"`
  Expected: PASS.

- [ ] **Step 7: Run the full suite and lint, then commit.**
  Run: `uv run pytest -q && uv run ruff check` and expect PASS, then:

```bash
git add bench/learn_radial/run.py bench/learn_radial/rmse_npz.py bench/learn_radial/modal_run.py tests/test_radial_density.py
git commit -m "feat(bench): --density flags, model-level RMSE, Modal pass-through"
```

---

### Task 7: Benchmarks and results document

This task is run by the controller, not a subagent. It needs Modal credentials, lestrade and judgement on the results.

**Files:**
- Create: `docs/dev/learn-radial-density-results.md`
- Create: `docs/dev/figures/learn-radial/density/` (JSON results)

- [ ] **Step 1: Launch the Modal runs.** The data dir is the one PR #14 used. Launch SiGe and Cantor, each at `--steps 40,200`:

```bash
LEARN_RADIAL_DATA=<data dir> modal run bench/learn_radial/modal_run.py --system sige --steps 40,200 --lam-grid 0.1 --density full --P 1 --out runs/density
LEARN_RADIAL_DATA=<data dir> modal run bench/learn_radial/modal_run.py --system sige --steps 40,200 --lam-grid 0.1 --density full --P 2 --out runs/density
LEARN_RADIAL_DATA=<data dir> modal run bench/learn_radial/modal_run.py --system sige --steps 40,200 --lam-grid 0.1 --density pair --P 1 --out runs/density
```

Repeat for `--system cantor`. Radials-only baselines at 40/200 steps exist from PR #14 (`docs/dev/figures/learn-radial/comparison/`). Reuse them, or rerun with `--density none` if the numbers aren't on the same split. `--lam-eta-grid` stays `0` for the first pass. Add `0,1e-2,1e-1` only if the learned η are visibly rough: plot `eta.npy` against basis degree.

- [ ] **Step 2: Compute QoIs.** On lestrade, with the ace-jax env (not moriarty), run `qoi.py --calc ace:<run>/model.npz --system {sige,cantor}` for every run's selected model. Then run `qoi_compare.py --dir <qoi dir> --ref mace` against the existing MACE reference JSONs.

- [ ] **Step 3: Write `docs/dev/learn-radial-density-results.md`.** Include:

  - a table with the RMSE (`rmse_npz.json`), the gate selection and scores, and the SiGe elastic mean deviation;
  - Si and Ge vacancy errors, and Cantor B, C44 and vacancy MAE, for each configuration against radials-only and pacemaker sqrt(ρ);
  - wall time per step compared with radials-only;
  - a verdict against each of the four success criteria in the spec;
  - the step-count recommendation (40 vs 200).

  If criterion 1 fails, the document says so and lists the nonlinear-correction fallback as the open decision. Do not implement the fallback.

- [ ] **Step 4: Commit.**

```bash
git add docs/dev/learn-radial-density-results.md docs/dev/figures/learn-radial/density
git commit -m "docs(bench): radial + density learning results"
```
