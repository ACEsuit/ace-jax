# Calibrated force UQ on the GP arm (`--uq ard-gp`): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the schema-3 calibrated per-atom force UQ (`forces_std`, `forces_cov`, `forces_q`,
`forces_group`) from a GP-arm fit. The jackknife sandwich runs over the joint design [B | k_θ(B, B_M)] at
the fixed θ_MAP. The new option is `--uq ard-gp`. A second shape, `--ard-variance dtc`, adds the GP's
derivative-DTC variance.

**Architecture:**
- **One new abstraction.** `fit/prior_root.py::PriorRoot` is the square root R0 of the ARD prior, as
  blockdiag(diag Γ, chol(K_MM)ᵀ). With M = 0 it is today's `dinv` exactly.
- **Routing.** Every place that multiplies or divides by `dinv` goes through `PriorRoot`: the evidence, the
  posterior, PRESS, the shape factor, the atom shape and the served variances. With M = 0 each method takes
  the existing diagonal expression verbatim, so the linear arm stays bit-identical.
- **Rows.** The design rows come from a rows function chosen once: the chunked linear rows (M = 0), or
  `rows.batch_rows` at θ_MAP (M > 0).
- **Serving.** `GPCalculator` gains `posterior=`. The fit writes `gp_model.npz` with the ARD mean, so the
  served mean and the served UQ belong to one posterior, as `model.npz` does for `--uq ard`.

**Tech Stack:** JAX (x64), NumPy, SciPy, ASE, pytest, ruff, `uv run`; Modal for the Cantor acceptance
(Task 8).

**Spec:** `docs/dev/specs/2026-10-05-gp-discrepancy-design.md`, Phase 3. Code map and the evidence behind the
design: `docs/dev/gp-discrepancy-exploration.md` (0.1 "Can `solve._prior_block`'s R0 stand in for D?",
and the D3 addendum).

## Global Constraints

- Branch `feat/gp-sandwich` (worktree `~/projects/ace-jax/.worktrees/gp-discrepancy`, from main 0e58bfd +
  the cherry-picked Phase 0 commits). One PR, squash-merged.
- **The linear arm is bit-identical.**
  - `--uq ard` results, `tests/test_ard*.py`, `tests/test_jackknife.py`, `tests/test_conformal.py`,
    `tests/test_shape_committee.py` and the pipeline goldens are unchanged.
  - Every `PriorRoot` method has an `if self.M == 0:` branch holding today's expression, character for
    character.
- **Option values:**
  - `FitConfig.uq` ∈ {"blr", "pops", "ard", "ard-gp"}; `"ard-gp"` requires `arm == "gp"`, and `"ard"` still
    requires `arm == "linear"`.
  - `FitConfig.ard_variance` ∈ {"sandwich", "kappa", "dtc"}; `"dtc"` requires `uq == "ard-gp"` and
    `kernel == "cosine"`.
  - CLI: `--uq ard-gp`, `--ard-variance dtc`. The `fit.yaml` keys are the existing `uq`, `ard_variance`.
- **Prior.** On the GP block, Λ_GP = exp(a_GP)·K_MM(θ_MAP), with **one** ARD scale a_GP. Its start value is
  0, which is the MAP's own prior. The body-order groups and the joint-E0 columns are unchanged.
- **Refused with a ValueError** under `uq="ard-gp"`: `_shape_variant="legacy"` / `_score_source="mixed"`;
  `ACECalculator(posterior=<ard-gp posterior>)`; `GPCalculator(posterior=<linear posterior>)`;
  `shape_path="committee"`.
- **Schema:**
  - `posterior.npz` stays `SCHEMA = 3`.
  - An ard-gp posterior adds `gp_U` (M, M), `gp_theta` (10,) and `gp_cols` (int M). Their absence means a
    linear posterior.
  - Old posteriors load unchanged.
- Python 3.11 syntax; float64 at the top of tests and scripts; optional imports inside functions; no
  reformatting of unrelated code; `uv run ruff check` and `uv run pytest -q` green after every task
  (`ACEJAX_TEST_WORKERS=4`, see the session memory cap).
- The **dtc** shape is diagonal in the DTC part: V_dtc = V_κ + diag(dtc_x, dtc_y, dtc_z).
  `_dtc_deriv_residual` gives per-component variances, not the 3×3 cross-covariance; say so in the
  docstring and the docs.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Prefixes
  `feat(ard):`, `test(ard):`, `docs(...)`, `feat(bench):`.

## Review Focus

1. **A posterior paired with the wrong calculator or the wrong fit.** Examples: an ard-gp `posterior.npz`
   given to `ACECalculator`, a linear one to `GPCalculator`, or a posterior from a different GP fit. Each must
   be refused with a message naming the fix, never served silently. Owner: Task 6
   (`test_gp_calculator_refuses_mismatched_posteriors`).
2. **A near-singular K_MM** (two inducing points almost identical). R0's triangular solves must stay finite,
   and the evidence must agree with the dense reference to the same relative tolerance. Owner: Task 1
   (`test_prior_root_near_duplicate_inducing_points`).
3. **Shared noise** (`noise="shared"`, which forces `ard_mode="sequential"`) with `uq="ard-gp"` must run
   end to end. It is the configuration Phase 1 recommends with explicit weights. Owner: Task 5
   (`test_stage_ard_gp_sequential_shared_noise`).
4. **A big cell through `GPCalculator(posterior=)`.** The (N, 3, L + M) force rows are built under
   `ROWS_EDGE_BUDGET`, not as one array. Calling with N past the budget must give the same values as small
   chunks. Owner: Task 6 (`test_gp_calculator_served_rows_are_chunked`).
5. **Rotating the input cell** must rotate `forces_cov` as R V Rᵀ, and leave `forces_std`, `forces_q` and
   `forces_group` invariant, for both shapes. Owner: Task 4 (`test_ard_gp_shape_is_rotation_equivariant`)
   and Task 7 (`test_dtc_shape_is_rotation_equivariant`).

---

### Task 1: `PriorRoot`, the square root of the ARD prior

**Files:**
- Create: `src/ace_jax/fit/prior_root.py`
- Modify: `tests/conftest.py`: add a `tiny_gp_problem` fixture, the M > 0 twin of `tiny_linear_problem`
- Test: `tests/test_prior_root.py`

**Interfaces:**
- Produces:
  - `PriorRoot(dinv: np.ndarray (L,), U: jax.Array (M, M) upper | None)`, a NamedTuple with properties `M`
    (int) and `width` (L + M), and methods:
    - `rows(X)` = X R0⁻¹ on the last axis (any leading shape);
    - `dual(G)` = R0⁻ᵀ G on the first axis ((Dt,) or (Dt, k));
    - `primal(x)` = R0⁻¹ x on the first axis;
    - `lift(z)` = R0ᵀ z on the first axis;
    - `gram(Mx)` = R0⁻ᵀ Mx R0⁻¹.
  - `PriorRoot.diag(dinv)`: the M = 0 root.
  - `prior_root(prob, theta)`: dinv from `ard.ard_gamma(prob)`; U = chol(K_MM(θ))ᵀ, or None when M = 0.
  - Fixture `tiny_gp_problem` → (prob, ds, theta): si_tiny, 6 configs, `m_per_species=8`, θ = the default
    prior mean.

- [ ] **Step 1: Write the failing tests** (`tests/test_prior_root.py`)

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from ace_jax.fit.prior_root import PriorRoot, prior_root


def _dense(root):
    L, M = len(root.dinv), root.M
    R0 = np.zeros((L + M, L + M))
    R0[np.arange(L), np.arange(L)] = 1.0 / np.asarray(root.dinv)
    if M:
        R0[L:, L:] = np.asarray(root.U)
    return R0


def _random_root(M, seed=0):
    rng = np.random.default_rng(seed)
    dinv = rng.uniform(0.5, 2.0, 7)
    if M == 0:
        return PriorRoot.diag(dinv)
    A = rng.normal(size=(M, M)); K = A @ A.T + M * np.eye(M)
    return PriorRoot(dinv, jnp.asarray(np.linalg.cholesky(K).T))


@pytest.mark.parametrize("M", [0, 4])
def test_prior_root_ops_match_dense(M):
    root = _random_root(M)
    R0 = _dense(root); Ri = np.linalg.inv(R0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 3, root.width)); G = rng.normal(size=(root.width, 5)); x = rng.normal(size=root.width)
    Mx = rng.normal(size=(root.width, root.width)); Mx = Mx + Mx.T
    np.testing.assert_allclose(root.rows(X), X @ Ri, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.dual(G), Ri.T @ G, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.dual(x), Ri.T @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.primal(x), Ri @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.lift(x), R0.T @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.gram(Mx), Ri.T @ Mx @ Ri, rtol=1e-11, atol=1e-11)


def test_prior_root_diag_is_todays_dinv_bitwise():
    root = _random_root(0)
    d = jnp.asarray(root.dinv)
    X = jnp.asarray(np.random.default_rng(2).normal(size=(5, 7)))
    assert np.array_equal(np.asarray(root.rows(X)), np.asarray(X * d[None, :]))
    assert np.array_equal(np.asarray(root.gram(X.T @ X)), np.asarray(d[:, None] * (X.T @ X) * d[None, :]))
    assert np.array_equal(np.asarray(root.lift(X[0])), np.asarray(X[0] / d))


def test_prior_root_from_problem(tiny_gp_problem):
    from ace_jax.fit.ard import ard_gamma
    from ace_jax.fit.kernels import K_MM
    prob, _, theta = tiny_gp_problem
    root = prior_root(prob, theta)
    assert root.M == prob.ind.XM.shape[0] > 0
    np.testing.assert_array_equal(root.dinv, 1.0 / ard_gamma(prob))
    K = np.asarray(K_MM(theta, prob.spec, prob.ind.XM, prob.ind.SM, prob.ind.ZM, prob.ind.embed))
    np.testing.assert_allclose(np.asarray(root.U).T @ np.asarray(root.U), K, rtol=1e-10, atol=1e-14)


def test_prior_root_near_duplicate_inducing_points(tiny_gp_problem):
    """Two inducing points 1e-9 apart: K_MM's jitter keeps chol finite, and rows/dual stay finite."""
    prob, _, theta = tiny_gp_problem
    ind = prob.ind
    XM = ind.XM.at[1].set(ind.XM[0] + 1e-9); SM = ind.SM.at[1].set(ind.SM[0]); ZM = ind.ZM.at[1].set(ind.ZM[0])
    root = prior_root(prob._replace(ind=ind._replace(XM=XM, SM=SM, ZM=ZM)), theta)
    v = np.ones(root.width)
    assert np.isfinite(np.asarray(root.U)).all()
    assert np.isfinite(np.asarray(root.dual(v))).all() and np.isfinite(np.asarray(root.rows(v[None]))).all()
```

- [ ] **Step 2: Add the fixture** to `tests/conftest.py`, after `tiny_linear_problem`:

```python
@pytest.fixture(scope="module")
def tiny_gp_problem():
    """M > 0 twin of tiny_linear_problem: 8 inducing sites (FPS), theta = the default prior mean.
    Returns (prob, ds, theta)."""
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem

    xyz, fitted = FIXTURE_DIR / "si_tiny_train.xyz", FIXTURE_DIR / "si_fitted.npz"
    if not (xyz.exists() and fitted.exists()):
        pytest.skip("missing GP fixtures")
    model, meta, z = load(fitted)
    configs = load_configs(xyz, "dft_energy", "dft_force", "dft_virial")[:6]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 8, descriptor_scale(X, ds.node_mask))
    prior = default_prior(2.35)
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]), prior)
    return prob, ds, prior.mu
```

- [ ] **Step 3: Run, expect failure**

Run: `uv run pytest tests/test_prior_root.py -q`
Expected: `ModuleNotFoundError: No module named 'ace_jax.fit.prior_root'`.

- [ ] **Step 4: Implement `src/ace_jax/fit/prior_root.py`**

```python
"""The square root R0 of the ARD prior precision, Lambda = R0^T diag(lambda) R0.

R0 = blockdiag(diag(Gamma), chol(K_MM)^T): diagonal on the L linear columns (the readout's smoothness
prior Gamma, then the joint-E0 columns' sqrt(e0_prec): ard.ard_gamma), upper triangular on the M inducing
columns of a GP-arm fit (K_MM at theta_MAP).  The ARD posterior works in the prior-scaled system
S = R0^-T A R0^-1, where the R0 terms of log|A| and log|Lambda| cancel for any invertible R0
(docs/dev/gp-discrepancy-exploration.md, 0.1).  M = 0 is the diagonal D = diag(Gamma) of the linear arm:
every method then evaluates today's dinv expression verbatim, so --uq ard stays bit-identical."""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular


class PriorRoot(NamedTuple):
    dinv: np.ndarray                 # (L,) 1 / Gamma on the linear columns
    U: object = None                 # (M, M) upper triangular chol(K_MM)^T, or None (M = 0)

    @staticmethod
    def diag(dinv):
        return PriorRoot(np.asarray(dinv, np.float64), None)

    @property
    def M(self):
        return 0 if self.U is None else int(self.U.shape[0])

    @property
    def width(self):
        return len(self.dinv) + self.M

    def _split(self, a, axis):
        L = len(self.dinv)
        return jnp.take(a, jnp.arange(L), axis=axis), jnp.take(a, jnp.arange(L, L + self.M), axis=axis)

    def rows(self, X):
        """X R0^-1 on the last axis."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return X * d
        Xl, Xg = self._split(jnp.asarray(X), -1)
        sh = Xg.shape
        g = solve_triangular(self.U, Xg.reshape(-1, self.M).T, lower=False, trans="T").T.reshape(sh)
        return jnp.concatenate([Xl * d, g], -1)

    def dual(self, G):
        """R0^-T G on the first axis (G (Dt,) or (Dt, k))."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return d * G if G.ndim == 1 else d[:, None] * G
        Gl, Gg = self._split(jnp.asarray(G), 0)
        lin = d * Gl if Gl.ndim == 1 else d[:, None] * Gl
        return jnp.concatenate([lin, solve_triangular(self.U, Gg, lower=False, trans="T")], 0)

    def primal(self, x):
        """R0^-1 x on the first axis (the mean coefficients from the scaled solution)."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return d * x
        xl, xg = self._split(jnp.asarray(x), 0)
        return jnp.concatenate([d * xl if xl.ndim == 1 else d[:, None] * xl,
                                solve_triangular(self.U, xg, lower=False)], 0)

    def lift(self, z):
        """R0^T z on the first axis (press_scores' push-through: g~ = R0^T L z)."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return z / d if z.ndim == 1 else z / d[:, None]
        zl, zg = self._split(jnp.asarray(z), 0)
        return jnp.concatenate([zl / d if zl.ndim == 1 else zl / d[:, None], self.U.T @ zg], 0)

    def gram(self, Mx):
        """R0^-T Mx R0^-1."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return d[:, None] * Mx * d[None, :]
        return self.rows(self.dual(Mx).T).T


def prior_root(prob, theta):
    """The ARD prior root of prob at theta: Gamma (ard_gamma) on the linear columns, chol(K_MM(theta))^T on
    the M inducing columns (None when M = 0)."""
    from .ard import ard_gamma
    from .kernels import K_MM
    dinv = 1.0 / ard_gamma(prob)
    if prob.ind.XM.shape[0] == 0:
        return PriorRoot.diag(dinv)
    ind = prob.ind
    K = K_MM(theta, prob.spec, ind.XM, ind.SM, ind.ZM, ind.embed)
    return PriorRoot(dinv, jnp.linalg.cholesky(K).T)
```

**Note:** `gram`'s M = 0 branch is `ard._ev_parts`'s `dinv[:, None] * M * dinv[None, :]` verbatim, and
`rows` is `X * dinv[None, :]` broadcast. Task 2 swaps the call sites, and `test_prior_root_diag_is_todays_dinv_bitwise`
pins the equality.

- [ ] **Step 5: Run the tests, then the whole file set**

Run: `uv run pytest tests/test_prior_root.py -q && uv run ruff check src/ace_jax/fit/prior_root.py tests/test_prior_root.py`
Expected: 6 passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/ace_jax/fit/prior_root.py tests/test_prior_root.py tests/conftest.py
git commit -m "feat(ard): PriorRoot, the ARD prior's square root blockdiag(diag Gamma, chol(K_MM)^T)"
```

---

### Task 2: The ARD evidence and posterior over the joint design

**Files:**
- Modify: `src/ace_jax/fit/ard.py`:
  - `body_order_columns` (`:32`), plus a new `GP_GROUP` and `gp_body_columns`;
  - `ard_statistics` (`:60`);
  - `ARDEvidence.__init__`/`h0`/`_parts` (`:115-150`), `_ev_parts` (`:199-211`);
  - `ard_posterior` (`:506-520`); `ARDPosterior` fields, `save`/`load` (`:288-500`).
- Test: `tests/test_ard_gp.py` (new)

**Interfaces:**
- Consumes: `PriorRoot`, `prior_root`, fixture `tiny_gp_problem` (Task 1).
- Produces:
  - `GP_GROUP = -1`: the body_col marker of an inducing column, with one shared ARD scale a_GP, start 0.
  - `gp_body_columns(body_col, M) -> np.ndarray`: body_col with M `GP_GROUP` entries appended.
  - `ard_statistics(theta, prob, ds, mode, columns="linear")`. `columns="joint"` streams
    `stats.sufficient_statistics` (Dt = L + M). Sequential mode combines the joint Gram at the MAP σ_q.
  - `ARDEvidence(stats, gamma, body_col, root=None)`. With `root` given, it replaces `PriorRoot.diag(1/gamma)`;
    `self.root` is stored and `self.dinv` stays (= root.dinv) for the linear code.
  - `ARDPosterior` gains `root: PriorRoot | None = None` (None = linear: `PriorRoot.diag(dinv)`) and
    `gp_theta: np.ndarray | None = None`. Property `post.prior_root` returns the effective root.
  - `ard_posterior(ev, h, kappa, meta)`: `mean = ev.root.primal(x)`, and it carries `root`/`gp_theta` when
    M > 0.

- [ ] **Step 1: Write the failing tests** (`tests/test_ard_gp.py`)

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from ace_jax.eval import highest_precision


def _meta(prob):
    from conftest import _orders
    return {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}


@pytest.fixture
def gp_ev(tiny_gp_problem):
    from ace_jax.fit.ard import (ARDEvidence, ard_gamma, ard_statistics, body_order_columns,
                                 gp_body_columns)
    from ace_jax.fit.prior_root import prior_root
    prob, ds, theta = tiny_gp_problem
    M = prob.ind.XM.shape[0]
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint", columns="joint")
        ev = ARDEvidence(st, ard_gamma(prob), gp_body_columns(body_order_columns(_meta(prob), prob.cfg), M),
                         root=prior_root(prob, theta))
    return prob, ds, theta, ev


def test_joint_statistics_have_the_inducing_columns(gp_ev):
    prob, _, _, ev = gp_ev
    L, M = prob.cfg.len_basis, prob.ind.XM.shape[0]
    assert ev._data[0][1].shape == (L + M, L + M)          # G_F over the joint design
    assert ev.groups[0] == -1 and len(ev.groups) == 1 + len(set(ev.body_col[:L]) - {0})


def test_evidence_at_h0_is_the_gp_lml(gp_ev):
    """h0 (a_k = -2 log sigma_c on the body orders, a_GP = 0, the MAP's sigma_q) reproduces the hybrid
    model's own log marginal likelihood at theta: the ARD prior at h0 IS blkdiag(Gamma^2/sigma_c^2, K_MM).
    ARDEvidence's constant leaves out the structural log weights and the 2 pi term, which the LML keeps:
    LML = logev + 1/2 sum_q logw_q - 1/2 N log(2 pi)."""
    from ace_jax.fit.objective import log_marginal_likelihood
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds, theta, ev = gp_ev
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        lml = float(log_marginal_likelihood(theta, st, prob))
        v, _ = ev.value_and_grad(ev.h0(theta))
    N = sum(float(getattr(st, f"n_{q}")) for q in "EFV")
    logw = sum(float(getattr(st, f"logw_{q}")) for q in "EFV")
    assert v + 0.5 * logw - 0.5 * N * np.log(2 * np.pi) == pytest.approx(lml, rel=1e-9)


def test_posterior_mean_at_h0_is_the_gp_map_mean(gp_ev):
    from ace_jax.fit.ard import ard_posterior
    from ace_jax.fit.objective import posterior
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds, theta, ev = gp_ev
    with highest_precision():
        mu, _ = posterior(theta, sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds), prob)
        post = ard_posterior(ev, ev.h0(theta), 1.0, _meta(prob))
    np.testing.assert_allclose(post.mean, np.asarray(mu), rtol=1e-7, atol=1e-9 * np.abs(mu).max())


def test_gp_evidence_gradient_matches_finite_differences(gp_ev):
    _, _, theta, ev = gp_ev
    h = ev.h0(theta) + 0.1
    _, g = ev.value_and_grad(h)
    for i in range(len(h)):
        e = np.zeros_like(h); e[i] = 1e-5
        fd = (ev.value_and_grad(h + e)[0] - ev.value_and_grad(h - e)[0]) / 2e-5
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_gp_posterior_save_load_roundtrip(gp_ev, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, ard_posterior
    prob, _, theta, ev = gp_ev
    post = ard_posterior(ev, ev.h0(theta), 1.0, _meta(prob))
    post.save(tmp_path / "p.npz", dtype=np.float64)
    back = ARDPosterior.load(tmp_path / "p.npz")
    assert back.prior_root.M == prob.ind.XM.shape[0]
    np.testing.assert_array_equal(np.asarray(back.prior_root.U), np.asarray(post.prior_root.U))
    np.testing.assert_array_equal(back.gp_theta, post.gp_theta)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_ard_gp.py -q`
Expected: `ImportError: cannot import name 'gp_body_columns'`.

- [ ] **Step 3: Implement in `src/ace_jax/fit/ard.py`**

1. After `E0_GROUP = 0`:

```python
GP_GROUP = -1     # body_col marker of an inducing column (uq "ard-gp"): one shared scale a_GP, start 0


def gp_body_columns(body_col, M):
    """body_col of the joint design [B | k(B, B_M)]: the linear columns' groups, then M GP_GROUP columns."""
    return np.concatenate([np.asarray(body_col, int), np.full(int(M), GP_GROUP)])
```

2. `ard_statistics(theta, prob, ds, mode, columns="linear")`. Keep the linear path as it is. For
   `columns == "joint"`:

```python
    if columns == "joint":
        from .stats import sufficient_statistics
        st = _joint_statistics_jit(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        if mode == "joint":
            return joint_ard_stats(st)
        ls = np.array([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
        w = np.exp(-2 * ls)
        M = sum(w[i] * getattr(st, f"G_{q}") for i, q in enumerate("EFV"))
        bv = sum(w[i] * getattr(st, f"b_{q}") for i, q in enumerate("EFV"))
        return ARDStats((M,), (bv,), np.zeros(3), np.zeros(3), ls)
```

   with a module-level `_joint_statistics_jit = eqx.filter_jit(sufficient_statistics)` placed next to
   `_linear_statistics_jit`. Same reason as the comment there: one executable per shape, not per call.
3. `ARDEvidence.__init__(self, stats, gamma, body_col, root=None)`:
   - `self.root = PriorRoot.diag(1.0 / np.asarray(gamma)) if root is None else root`, and
     `self.dinv = jnp.asarray(self.root.dinv)`.
   - Check `len(gamma) + self.root.M == len(body_col)`; the existing message is extended.
   - `fixed = body_col == E0_GROUP`.
   - `groups = tuple(sorted(set(body_col[~fixed])))`, so `GP_GROUP = -1` sorts first.
   - Pass `self.root` to `_ev_parts` in place of `dinv`.
4. `h0`: per group, `0.0 if g == GP_GROUP else a_blr`.
5. `_ev_parts(h, data, root, gidx, joint)`: `Ms = root.gram(M)`, `bs = root.dual(bv)`. Everything else is
   unchanged.
   - `root` must be a jit argument. Register `PriorRoot` as a pytree: a NamedTuple whose `U` may be None is
     one already. Mark nothing static.
   - **Bit-identity check:** with M = 0, `root.gram(M)` is the old expression verbatim and `root.dual(bv)`
     is `dinv * bv`, as before.
   - Update `_ev_logev`, `_ev_value_and_grad`, `_ev_hvp` and `cond` to pass `self.root`.
6. `ard_posterior`:
   - `mean = np.asarray(ev.root.primal(x))`; `dinv = np.asarray(ev.root.dinv)`.
   - With M > 0, also set `root=ev.root` and `gp_theta=np.asarray(to_array(theta))`. To get θ, give
     `ard_posterior` a keyword `theta=None`; Task 5 passes it.
   - `keep["gp_cols"] = M` when M > 0.
7. `ARDPosterior`:
   - Add the trailing fields `root: object = None` and `gp_theta: np.ndarray | None = None`, plus the
     property

```python
    @property
    def prior_root(self):
        from .prior_root import PriorRoot
        return self.root if self.root is not None else PriorRoot.diag(self.dinv)
```

   - `save`: when `self.root is not None and self.root.M`, add `gp_U=np.asarray(self.root.U, np.float64)`
     (always float64: it is M × M, small, and a triangular solve in float32 loses the jitter scale) and
     `gp_theta=self.gp_theta`.
   - `load`: if `"gp_U" in z.files`, rebuild `root=PriorRoot(z["dinv"], jnp.asarray(z["gp_U"]))`,
     `gp_theta=z["gp_theta"]`.

- [ ] **Step 4: Run the new tests and every existing ARD test (bit-identity of the linear path)**

Run: `uv run pytest tests/test_ard_gp.py tests/test_ard.py tests/test_jackknife.py tests/test_prior_root.py -q`
Expected: all pass; no change to any existing assertion.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/ard.py tests/test_ard_gp.py
git commit -m "feat(ard): the ARD evidence and posterior over the joint design [B | k(B, B_M)]"
```

---

### Task 3: PRESS, shape factor and atom shape through `PriorRoot`, with the GP rows

**Files:**
- Modify: `src/ace_jax/fit/jackknife.py`:
  - `press_scores` (`:43-87`): `W`, the push-through `G`, and a rows-function parameter;
  - `shape_factor` (`:89-106`); `atom_shape` (`:116-128`).
- Modify: `src/ace_jax/fit/ard.py`: `ARDPosterior.atom_shape`, `var_rows`, `misspec_var_rows`,
  `_force_width`, `_val_atoms`, `predict_ard`: each `* dinv` becomes `prior_root.rows(...)`, and the rows
  function becomes a parameter.
- Create: `rows_fn_for(prob, theta)` in `src/ace_jax/fit/ard.py`.
- Test: `tests/test_ard_gp.py` (append)

**Interfaces:**
- Consumes: `PriorRoot`, `ARDPosterior.prior_root`, `gp_ev` (Task 2).
- Produces:
  - `rows_fn_for(prob, theta) -> Callable[[batch], Rows]`: `chunked_rows_fn(prob.model, prob.cfg)` when M = 0;
    otherwise a jitted `lambda b: rows.batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, b)`.
  - `press_scores(post, prob, ds, clusters, K, sig, mode="exact", rows_fn=None)`, defaulting to
    `rows_fn_for(prob, None)` (linear).
  - `atom_shape(R, root, Frows, chunk=None)`. The second argument is now a `PriorRoot`. Callers that pass
    a `dinv` array are rewritten to pass `PriorRoot.diag(dinv)`, and `committee_shape` keeps its own `dinv`
    argument because it is linear-only.
  - `_val_atoms(post, prob, ds, r1=None, shape=True, own_col=None, rows_fn=None)` and
    `predict_ard(post, prob, ds, node_chunk=None, rows_fn=None)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ard_gp.py`)

```python
def _gp_dense_rows(prob, ds, theta, sig):
    """Whitened joint rows Psi (n, L+M), targets y~ and the whole-configuration label of each row."""
    from ace_jax.fit.rows import batch_rows
    P, Y, cf, g = [], [], [], 0
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, b)
        for c in np.flatnonzero(np.asarray(b.cfg_mask)):
            w = float(b.w_E[c]) / sig[0]; P.append(np.asarray(r.E[c]) * w); Y.append(float(b.y_E[c]) * w); cf.append(g)
            for n in np.flatnonzero(np.asarray(b.node_cfg) == c):
                w = float(b.w_F[n]) / sig[1]
                for a in range(3):
                    P.append(np.asarray(r.F[n, a]) * w); Y.append(float(b.y_F[n, a]) * w); cf.append(g)
            w = float(b.w_V[c]) / sig[2]
            for v in range(6):
                P.append(np.asarray(r.V[c, v]) * w); Y.append(float(b.y_V[c, v]) * w); cf.append(g)
            g += 1
    return np.array(P), np.array(Y), np.array(cf)


def _A_gp(post):
    R0 = np.linalg.inv(np.asarray(post.prior_root.rows(np.eye(post.prior_root.width))))   # R0 from R0^-1
    S = np.asarray(post.chol) @ np.asarray(post.chol).T
    return R0.T @ S @ R0


@pytest.mark.parametrize("ell", [float("inf"), 18.75])
def test_gp_press_equals_exact_deletion(gp_ev, ell):
    """c - c_(-k) = A^-1 g~_k for every cluster (whole configurations, and 3 r_cut spatial blocks), h fixed,
    over the joint design: the PRESS/DFBETA identity of the spec, to the reference solve's noise."""
    from ace_jax.fit.ard import ard_posterior, rows_fn_for
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, theta, ev = gp_ev
    h = ev.h0(theta)
    post = ard_posterior(ev, h, 1.0, _meta(prob), theta=theta)
    sig = ev.sigmas(h)
    Psi, y, cf = _gp_dense_rows(prob, ds, theta, sig)
    A = _A_gp(post)
    c = np.linalg.solve(A, Psi.T @ y)
    np.testing.assert_allclose(c, post.mean, rtol=1e-6, atol=1e-7 * np.abs(c).max())
    rc, K = row_clusters(ds, None, ell)
    G, _ = press_scores(post, prob, ds, rc, K, sig, rows_fn=rows_fn_for(prob, theta))
    labels = _row_cluster_labels(ds, rc)          # the cluster id of each dense row, in _gp_dense_rows' order
    for k in range(K):
        m = labels == k
        ck = np.linalg.solve(A - Psi[m].T @ Psi[m], Psi[~m].T @ y[~m])
        np.testing.assert_allclose(c - ck, np.linalg.solve(A, G[:, k]), rtol=1e-5, atol=1e-7 * np.abs(c).max())


def _row_cluster_labels(ds, rc):
    out = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        for c in np.flatnonzero(np.asarray(b.cfg_mask)):
            out.append(rc[i]["E"][c])
            for n in np.flatnonzero(np.asarray(b.node_cfg) == c):
                out += [rc[i]["F"][n]] * 3
            out += [rc[i]["V"][c]] * 6
    return np.array(out)


def test_gp_push_through_matches_exact(gp_ev):
    from ace_jax.fit.ard import ard_posterior, rows_fn_for
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, theta, ev = gp_ev
    h = ev.h0(theta); post = ard_posterior(ev, h, 1.0, _meta(prob), theta=theta)
    rc, K = row_clusters(ds, None, float("inf")); f = rows_fn_for(prob, theta)
    Ge, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="exact", rows_fn=f)
    Gp, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="pushthrough", rows_fn=f)
    np.testing.assert_allclose(Gp, Ge, rtol=1e-7, atol=1e-10 * np.abs(Ge).max())


def test_ard_gp_shape_is_rotation_equivariant(gp_ev):
    """V(R x) = R V(x) R^T for the PRESS shape over the joint rows (Review Focus 5)."""
    from scipy.spatial.transform import Rotation
    from ace_jax.fit.ard import ard_posterior, rows_fn_for
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores, shape_factor
    prob, ds, theta, ev = gp_ev
    h = ev.h0(theta); post = ard_posterior(ev, h, 1.0, _meta(prob), theta=theta)
    rc, K = row_clusters(ds, None, float("inf")); f = rows_fn_for(prob, theta)
    G, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), rows_fn=f)
    post = post._replace(R=shape_factor(post, G))
    b = jax.tree.map(lambda a: a[0], ds)
    Rm = Rotation.from_euler("zyx", [0.3, -0.7, 1.1]).as_matrix()
    br = b._replace(rij=b.rij @ Rm.T)
    V0 = post.atom_shape(np.asarray(f(b).F)); V1 = post.atom_shape(np.asarray(f(br).F))
    live = np.asarray(b.node_mask)
    np.testing.assert_allclose(V1[live], np.einsum("ab,nbc,dc->nad", Rm, V0[live], Rm), rtol=1e-8,
                               atol=1e-12 * np.abs(V0).max())
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_ard_gp.py -q -k "press or push or rotation"`
Expected: `ImportError: cannot import name 'rows_fn_for'`.

- [ ] **Step 3: Implement**

`jackknife.press_scores`:
- `rows = chunked_rows_fn(prob.model, prob.cfg) if rows_fn is None else rows_fn`;
- `root = post.prior_root`;
- `W = solve_triangular(Lj, root.rows(Pk).T, lower=True)`. With M = 0, `root.rows(Pk)` is
  `Pk * dinv[None, :]`, the old expression;
- push-through: `G[:, kk] = np.asarray(root.lift(Lj @ jnp.asarray(z)))`. With M = 0 that is `(Lj z) / dinv`.

`shape_factor`: `Qt = cho_solve((Lc, True), post.prior_root.dual(G - G.mean(1, keepdims=True)))`.

`atom_shape(R, root, Frows, chunk=None)`: `U = root.rows(jnp.asarray(Frows[i:i + c], jnp.float64))`.

`ard.py`:
- `ARDPosterior.atom_shape` passes `self.prior_root` and truncates with `self._force_width`.
- `_force_width` accepts `n_cols in (full, full - n_e0)`, where `full = self.prior_root.width`. The joint-E0
  columns sit before the M inducing columns in the joint layout, so readout-only rows are refused for M > 0:
  `GPCalculator` always builds full rows.
- `var_rows`: `v = solve_triangular(c, self.prior_root.rows(P).T, lower=True)`.
- `misspec_var_rows`: `(self.prior_root.rows(P)) @ Q`.
- `_val_atoms`: `rows_fn` parameter, `L = post.prior_root.width`, and `own_col` uses `post.prior_root.rows(F)`.
- `predict_ard`: `rows_fn` parameter, defaulting to `chunked_rows_fn(prob.model, prob.cfg, node_chunk)`;
  `L = post.prior_root.width`.

`rows_fn_for(prob, theta)`:

```python
def rows_fn_for(prob, theta, node_chunk=None):
    """The ARD stage's design-row function: the linear arm's node-chunked rows (M = 0, theta unused), or the
    joint rows [B | k(B, B_M)] at theta (uq "ard-gp"), jitted once per call site."""
    from .rows import batch_rows, chunked_rows_fn
    if prob.ind.XM.shape[0] == 0:
        return chunked_rows_fn(prob.model, prob.cfg, node_chunk)
    f = eqx.filter_jit(lambda b: batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, b))
    return f
```

- [ ] **Step 4: Run the new tests and the full ARD set (bit-identity)**

Run: `uv run pytest tests/test_ard_gp.py tests/test_ard.py tests/test_jackknife.py tests/test_shape_committee.py tests/test_ard_calc_cli.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/jackknife.py src/ace_jax/fit/ard.py tests/test_ard_gp.py
git commit -m "feat(ard): PRESS, shape and served variances over the joint design via PriorRoot"
```

---

### Task 4: `run_ard_stage` for `uq="ard-gp"`, and the M = 0 reduction

**Files:**
- Modify: `src/ace_jax/fit/ard.py::run_ard_stage` (`:792-985`), `_holdout_posterior` (`:763-790`)
- Modify: `src/ace_jax/fit/pipeline/config.py`: `uq` choices, `ard_variance` choices, validation
  (`:117-150`)
- Modify: `src/ace_jax/fit/pipeline/run.py:58-71`: run the stage for `cfg.uq in ("ard", "ard-gp")`, and pass
  the full joint statistics
- Test: `tests/test_ard_gp.py` (append)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `run_ard_stage(cfg, data, built, theta, log=print, full_stats=None)`. Under `cfg.uq == "ard-gp"` it:
    - builds the stats with `columns="joint"`;
    - uses `gp_body_columns`, the `prior_root(prob, theta)` root and `rows_fn_for(prob, theta)`;
    - passes `theta` to `ard_posterior`.
  - Under `uq="ard"`, nothing changes.
  - `FitConfig.validate()` accepts `uq="ard-gp"` with `arm="gp"`, and gives the refusals of Global
    Constraints.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _gp_pipe_cfg(**kw):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="gp", m_per_species=8,
                uq="ard-gp", opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False,
                ard_val_frac=0.4, ard_n_min=50, ard_force_shape="aniso")
    return FitConfig(**{**base, **kw})


def _gp_stage(**kw):
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _gp_pipe_cfg(**kw).validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        pred = ard.predict_ard(res.posterior, b.prob, d.ds_test, rows_fn=ard.rows_fn_for(b.prob, theta))
    return d, b, theta, res, pred


def test_stage_ard_gp_press_shape_and_group_scales():
    d, b, theta, res, pred = _gp_stage()
    post, rep = res.posterior, res.report
    M = b.prob.ind.XM.shape[0]
    assert post.prior_root.M == M and post.R.shape[0] == b.prob.cfg.len_basis + M
    assert "a_GPbody" in rep["h_names"] or any(n.startswith("a_GP") for n in rep["h_names"])
    assert np.isfinite(post.group_table["q"]).all()
    assert np.isfinite(pred.F_var).all() and np.all(pred.F_var >= 0)


def test_stage_ard_gp_sequential_shared_noise():
    """Review Focus 3: shared noise forces sequential ARD; the GP-arm stage runs end to end."""
    _, _, _, res, pred = _gp_stage(noise="shared", ard_mode="sequential")
    assert res.report["mode"] == "sequential" and np.isfinite(pred.F_var).all()


def test_ard_gp_with_no_inducing_columns_is_the_linear_stage(ard_map):
    """M = 0 reduces to the linear-arm result: the ard-gp stage (joint statistics via
    sufficient_statistics, the PriorRoot path, rows_fn_for) on a problem with no inducing columns gives the
    --uq ard posterior.  The two stream the statistics through different code (sufficient_statistics against
    the bounded linear statistics), so equality is to summation-order roundoff, not bitwise."""
    from ace_jax.fit import ard
    from conftest import ard_pipe_cfg
    d, b, theta = ard_map
    lin = ard.run_ard_stage(ard_pipe_cfg(ard_variance="sandwich").validate(), d, b, theta, log=lambda *a: None)
    gp_cfg = ard_pipe_cfg(ard_variance="sandwich")
    gp_cfg.uq = "ard-gp"                    # the stage's ard-gp branch, bypassing validate()'s arm check
    gp = ard.run_ard_stage(gp_cfg, d, b, theta, log=lambda *a: None)
    P, Q = lin.posterior, gp.posterior
    np.testing.assert_allclose(Q.mean, P.mean, rtol=1e-9, atol=1e-12 * np.abs(P.mean).max())
    S = lambda p: np.asarray(p.R) @ np.asarray(p.R).T
    np.testing.assert_allclose(S(Q), S(P), rtol=1e-7, atol=1e-10 * np.abs(S(P)).max())
    for k in ("lam_rms", "q"):
        np.testing.assert_allclose(Q.group_table[k], P.group_table[k], rtol=1e-7)


@pytest.mark.parametrize("kw, msg", [(dict(arm="linear", m_per_species=0), "ard-gp"),
                                     (dict(_shape_variant="legacy", _score_source="mixed"), "legacy"),
                                     (dict(ard_variance="dtc", kernel="matern32"), "cosine")])
def test_ard_gp_config_refusals(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _gp_pipe_cfg(**kw).validate()
```

`ard_map` and `_pipe_cfg` live in `tests/test_ard.py`. Move both to `tests/conftest.py` unchanged:
- `_pipe_cfg` is renamed `ard_pipe_cfg`; `test_ard.py` imports it as `from conftest import ard_pipe_cfg as
  _pipe_cfg`, so its body is untouched;
- `ard_map` becomes a conftest fixture with the same name.

That way both files share one module-scoped fit. `FitConfig` is a plain (non-frozen) dataclass, so the test
assigns `uq` directly.

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_ard_gp.py -q -k "stage or reduces or refusals"`
Expected: `ValueError: uq must be 'blr', 'pops' or 'ard', got 'ard-gp'`.

- [ ] **Step 3: Implement**

`config.py`:
- `uq` ∈ ("blr", "pops", "ard", "ard-gp").
- Run the existing `if self.uq == "ard":` block for `self.uq in ("ard", "ard-gp")`, with the arm check
  split:
  - `ard`: `arm == "linear"`, as now;
  - `ard-gp`: `arm == "gp"`, else `ValueError("uq='ard-gp' is the GP-arm sandwich posterior: use arm gp (m_per_species > 0)")`.
- `ard_variance` ∈ ("sandwich", "kappa", "dtc"). `"dtc"` needs `uq == "ard-gp"` (message names
  `--uq ard-gp`) and `kernel == "cosine"` (message names the cosine kernel).
- Under `ard-gp`, `_shape_variant != "press"` or `_score_source != "fit"` raises a ValueError naming the
  legacy ablation.
- `joint_e0` (line 101) includes `"ard-gp"`.
- Nothing else changes; the shared-noise refusal (joint ARD) applies to both.

`run.py:58`: `if cfg.uq in ("ard", "ard-gp"):`.
- `full = obj.lin` stays for `"ard"`.
- For `"ard-gp"` in joint mode, `full = obj.stats(mf.theta)`. This is the full Stats: host-cache assembles
  the cached linear part with one residual pass, so no second linear pass is made. Take it **before** the
  `obj._replace(... stats=None ...)` line.
- The model-file stage (`linear_arrays_from_mean`) runs only for `"ard"`. Task 6 adds the GP export.

`ard.run_ard_stage` and `_holdout_posterior`:
- `gp = cfg.uq == "ard-gp"` (no M = 0 guard: the reduction test drives the joint path with M = 0, and every
  piece of it must accept M = 0: `PriorRoot.diag`, `gp_body_columns(..., 0)`, and `rows_fn_for` returning
  the linear rows);
- `cols = "joint" if gp else "linear"`;
- `body = gp_body_columns(body_col, M) if gp else body_col`;
- `root = prior_root(prob, theta) if gp else None`;
- `rows_fn = rows_fn_for(prob, theta) if gp else None`.

Thread these through:
- `ard_statistics(..., columns=cols)`;
- `ARDEvidence(..., root=root)`;
- `ard_posterior(..., theta=theta if gp else None)`;
- `press_scores(..., rows_fn=rows_fn)`;
- `_val_atoms(..., rows_fn=rows_fn)`.

With `full_stats` given in joint mode, the full refit uses `joint_ard_stats(full_stats)` for both
columns. For `ard-gp` it already holds Dt columns.

The report gets `"gp_cols": M`. The hyperparameter names for GP_GROUP are `"a_GP"`: build `names` from
`ev.groups` as `"a_GP" if g == GP_GROUP else f"a_{g}body"`, and update the test's assertion to
`"a_GP" in rep["h_names"]`.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_ard_gp.py tests/test_ard.py tests/test_pipeline_units.py tests/test_shared_noise.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/ard.py src/ace_jax/fit/pipeline/config.py src/ace_jax/fit/pipeline/run.py tests/test_ard_gp.py tests/test_ard.py tests/conftest.py
git commit -m "feat(ard): uq='ard-gp', the jackknife-sandwich stage on the GP arm at fixed theta"
```

---

### Task 5: Prediction and outputs of an `ard-gp` fit

**Files:**
- Modify: `src/ace_jax/fit/pipeline/predict.py:90-115`: the `ard` branch for `ard-gp`, with
  `rows_fn_for(prob, theta)`
- Modify: `src/ace_jax/fit/pipeline/export.py`: `gp_model_arrays(res, ...)` writes the ARD mean and
  chol(A) when `res.ard` is set
- Modify: `src/ace_jax/fit/pipeline/outputs.py:30-75`: write `posterior.npz` for `ard-gp` too
- Test: `tests/test_ard_gp.py` (append)

**Interfaces:**
- Consumes: Task 4's `run_ard_stage` result.
- Produces:
  - An `ard-gp` fit writes `gp_model.npz` whose single posterior is (ARD mean, L_A), with L_A L_Aᵀ = A =
    R0ᵀ S R0, plus `posterior.npz` (schema 3 with `gp_U`, `gp_theta`).
  - `draws` = θ_MAP.
  - `metrics.csv` F rows use the served calibrated `F_var` (as `--uq ard`).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_ard_gp_fit_writes_ard_mean_gp_model(tmp_path):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import fit, load_fit_data, write_outputs
    from ace_jax.fit.pipeline.export import load_gp_model
    cfg = _gp_pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    res = fit(cfg, d, log=lambda *a: None)
    write_outputs(res, tmp_path, layout=("run", "cli"), log=lambda *a: None)
    fg, _ = load_gp_model(tmp_path / "gp_model.npz")
    mu, L = fg.posteriors[0]
    np.testing.assert_allclose(np.asarray(mu), res.ard.posterior.mean, rtol=0, atol=0)
    root = res.ard.posterior.prior_root
    R0 = np.linalg.inv(np.asarray(root.rows(np.eye(root.width))))
    S = np.asarray(res.ard.posterior.chol) @ np.asarray(res.ard.posterior.chol).T
    np.testing.assert_allclose(np.asarray(L) @ np.asarray(L).T, R0.T @ S @ R0, rtol=1e-8,
                               atol=1e-10 * np.abs(R0.T @ S @ R0).max())
    assert (tmp_path / "posterior.npz").exists()
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_ard_gp.py -q -k writes`
Expected: `posterior.npz` missing, or `mu` ≠ the ARD mean (the export writes the MAP posterior).

- [ ] **Step 3: Implement**

`predict.py`: the branch `if cfg.uq == "ard":` becomes `if cfg.uq in ("ard", "ard-gp"):`, with
`rows_fn=rows_fn_for(b.prob, theta)` passed to `predict_ard`.

`export.gp_model_arrays`: when `getattr(res, "ard", None) is not None`:
- `post = res.ard.posterior; root = post.prior_root`;
- `L_A = R0ᵀ Lc`. R0ᵀ is lower triangular and Lc is lower, so the product is lower with a positive
  diagonal. Compute it blockwise:
  - linear rows: `Lc[:L] / dinv[:, None]`;
  - inducing rows: `U.T @ Lc[L:]`;
- write `mu=[post.mean]`, `L=[L_A]`, `draws=[to_array(res.theta)]`.

`outputs.py`: the `posterior.npz` write condition `cfg.uq == "ard"` → `cfg.uq in ("ard", "ard-gp")`.
The "ARD mean in the model file" branch (line 70) stays linear-only.

- [ ] **Step 4: Run** `uv run pytest tests/test_ard_gp.py tests/test_pipeline_units.py tests/test_ard.py -q`.
  Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/pipeline/predict.py src/ace_jax/fit/pipeline/export.py src/ace_jax/fit/pipeline/outputs.py tests/test_ard_gp.py
git commit -m "feat(ard): ard-gp fits write the ARD-mean gp_model.npz and posterior.npz"
```

---

### Task 6: `GPCalculator(posterior=)` and `aj eval --posterior` for GP models

**Files:**
- Modify: `src/ace_jax/calc/gp.py`: `__init__(..., posterior=None)`, `implemented_properties`, a
  `_posterior_quantities` mirroring `ACECalculator`'s
- Modify: `src/ace_jax/calc/point.py:221-253`: refuse a posterior with `gp_U`
- Modify: `src/ace_jax/cli.py:260-285`: allow `--posterior` with a `gp_model.npz` when the posterior has
  `gp_U`
- Test: `tests/test_ard_gp.py` (append)

**Interfaces:**
- Consumes: Task 5's files.
- Produces:
  - `GPCalculator(fitted, meta, deriv_dtc=True, posterior=None, **kw)` and
    `GPCalculator.from_file(path, posterior=...)`.
  - With a posterior, `forces_std`, `forces_cov`, `forces_q`, `forces_q_mahal` (aniso) and `forces_group`
    are served from the ARD posterior over the joint rows at the posterior's `gp_theta`. `energy`/`forces`
    stay the `gp_model.npz` mean, which is the ARD mean.

- [ ] **Step 1: Write the failing tests** (append)

```python
@pytest.fixture(scope="module")
def gp_fit_dir(tmp_path_factory):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import fit, load_fit_data, write_outputs
    out = tmp_path_factory.mktemp("ardgp")
    cfg = _gp_pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    res = fit(cfg, d, log=lambda *a: None)
    write_outputs(res, out, layout=("run", "cli"), log=lambda *a: None)
    return out, d, res


def test_gp_calculator_serves_the_ard_posterior(gp_fit_dir):
    from ace_jax.calc.gp import GPCalculator
    from ace_jax.fit.ard import predict_ard, rows_fn_for
    from ace_jax.fit.xyz import read_extxyz
    from conftest import FIXTURE_DIR
    out, d, res = gp_fit_dir
    calc = GPCalculator.from_file(out / "gp_model.npz", posterior=out / "posterior.npz")
    at = read_extxyz(FIXTURE_DIR / "si_tiny_train.xyz")[0]
    at.calc = calc
    sd = calc.get_property("forces_std", at)
    cov = calc.get_property("forces_cov", at)
    assert sd.shape == (len(at),) and cov.shape == (len(at), 3, 3)
    np.testing.assert_allclose(sd ** 2, np.trace(cov, axis1=1, axis2=2), rtol=1e-10)
    assert np.isfinite(calc.get_property("forces_q", at)).all()


def test_gp_calculator_refuses_mismatched_posteriors(gp_fit_dir, tmp_path):
    """Review Focus 1."""
    from conftest import FIXTURE_DIR
    from ace_jax import ACECalculator
    from ace_jax.calc.gp import GPCalculator
    out, _, _ = gp_fit_dir
    with pytest.raises(ValueError, match="ard-gp"):
        ACECalculator(str(FIXTURE_DIR / "si_fitted.npz"), posterior=str(out / "posterior.npz"))
    lin = tmp_path / "lin_post.npz"
    _write_linear_posterior(lin)                     # a --uq ard posterior of the tiny fixture (helper below)
    with pytest.raises(ValueError, match="linear"):
        GPCalculator.from_file(out / "gp_model.npz", posterior=lin)
    other = tmp_path / "other.npz"
    z = dict(np.load(out / "posterior.npz")); z["mean"] = z["mean"] * 1.01; np.savez(other, **z)
    with pytest.raises(ValueError, match="same fit"):
        GPCalculator.from_file(out / "gp_model.npz", posterior=other)


def _write_linear_posterior(path):
    from test_ard import _stage           # tests/ is on sys.path under pytest (conftest's directory)
    _stage(ard_force_shape="aniso", ard_val_frac=0.4, ard_n_min=50)[1].posterior.save(path)


def test_gp_calculator_served_rows_are_chunked(gp_fit_dir, monkeypatch):
    """Review Focus 4: a tiny ROWS_EDGE_BUDGET forces the node-chunked rows; served values are unchanged."""
    from ace_jax.calc.gp import GPCalculator
    from ace_jax.fit import rows
    from ace_jax.fit.xyz import read_extxyz
    from conftest import FIXTURE_DIR
    out, _, _ = gp_fit_dir
    at = read_extxyz(FIXTURE_DIR / "si_tiny_train.xyz")[0]
    ref = GPCalculator.from_file(out / "gp_model.npz", posterior=out / "posterior.npz")
    at.calc = ref; a = ref.get_property("forces_cov", at)
    monkeypatch.setattr(rows, "ROWS_EDGE_BUDGET", 1000)
    small = GPCalculator.from_file(out / "gp_model.npz", posterior=out / "posterior.npz")
    at.calc = small; b = small.get_property("forces_cov", at)
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-14)
```

- [ ] **Step 2: Run, expect failure** (`TypeError: unexpected keyword argument 'posterior'`).

- [ ] **Step 3: Implement**

`GPCalculator.__init__`:
- Load `ARDPosterior.load(posterior)`. Refuse it if `post.prior_root.M == 0` ("a linear --uq ard
  posterior: serve it with ACECalculator(model.npz, posterior=...)").
- Refuse it if `post.prior_root.width != mu.shape[0]`, or if `not np.allclose(post.mean, mu, rtol=0,
  atol=1e-12 * |mu|max)` ("not the gp_model.npz written by the same fit").
- Store `self.posterior`, `self._post_theta = from_array(jnp.asarray(post.gp_theta))` and
  `self._rows_fn = rows_fn_for(fitted.prob, self._post_theta)`.
- Extend `implemented_properties` with `forces_cov`, `forces_q`, `forces_q_mahal` and `forces_group`.

`calculate`: after the existing results, when a posterior is set:
- compute `F = self._rows_fn(batch).F[live]`;
- compute `groups = post.groups_of(batch)[live]`;
- update the results with `post.served(F, groups, which)` and `forces_group`;
- replace `forces_std` with the served one;
- serve `forces_q_mahal` only if aniso.

`rows_fn_for`'s M > 0 path must use the bounded rows (`batch_rows`, which goes through `batch_rows_parts`,
already bounded by `ROWS_EDGE_BUDGET`). It must not use a whole-batch `linear_rows` call.

`ACECalculator`: after `post = ARDPosterior.load(posterior)`, if `post.prior_root.M > 0`, raise
`ValueError("this posterior is from --uq ard-gp (a GP fit): serve it with GPCalculator.from_file(gp_model.npz, posterior=...)")`.

`cli.py` `aj eval`: the check at `:265` becomes "refuse `--posterior` with a `gp_model.npz` unless the
posterior has `gp_U`"; then construct `GPCalculator.from_file(model, posterior=...)`. The served columns are
written as for the linear path.

- [ ] **Step 4: Run** `uv run pytest tests/test_ard_gp.py tests/test_ard_calc_cli.py tests/test_calc*.py -q`.
  Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/calc/gp.py src/ace_jax/calc/point.py src/ace_jax/cli.py tests/test_ard_gp.py
git commit -m "feat(calc): GPCalculator(posterior=) serves the ard-gp calibrated force UQ; aj eval --posterior for GP models"
```

---

### Task 7: `--ard-variance dtc`

**Files:**
- Modify: `src/ace_jax/fit/ard.py`: a `dtc` shape in `run_ard_stage` (the hold-out shape and the served
  shape), `ARDPosterior.atom_shape` (`dtc` flag, stored), and `predict_ard`
- Modify: `src/ace_jax/calc/gp.py`: add the DTC diagonal when `post.variance == "dtc"`
- Test: `tests/test_ard_gp.py` (append)

**Interfaces:**
- Consumes: `predict._dtc_deriv_residual(theta, prob, batch, X, res=, JU0=)` → (Fv (Ncap, 3), Vv).
- Produces:
  - `ARDPosterior.variance: str = "sandwich"` (saved; old files load as "sandwich" or, with no `R`/`Q`,
    "kappa").
  - For `variance == "dtc"` the shape is V = W Wᵀ with W = L⁻¹ R0⁻ᵀ φᵀ (the κ shape over the joint rows),
    plus diag(Fv) from the derivative DTC at `gp_theta`.
  - `dtc_shape(post, prob, theta, batch) -> np.ndarray (Ncap, 3, 3)`: the one implementation, used by the
    stage, `predict_ard` and `GPCalculator`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_dtc_shape_is_kappa_shape_plus_dtc_diagonal(gp_fit_dir):
    from ace_jax.fit.ard import dtc_shape
    from ace_jax.fit.hypers import from_array
    from ace_jax.fit.predict import _dtc_deriv_residual
    out, d, res = gp_fit_dir
    post, prob = res.ard.posterior, res.built.prob
    theta = from_array(np.asarray(post.gp_theta))
    b = jax.tree.map(lambda a: a[0], d.ds_test)
    V = dtc_shape(post._replace(variance="dtc", R=None), prob, theta, b)
    Fv, _ = _dtc_deriv_residual(theta, prob, b)
    Vk = post._replace(variance="kappa", R=None).atom_shape(np.asarray(_rows(prob, theta, b).F))
    live = np.asarray(b.node_mask)
    np.testing.assert_allclose(V[live], Vk[live] + np.einsum("na,ab->nab", np.asarray(Fv)[live], np.eye(3)),
                               rtol=1e-10, atol=1e-14)


def _rows(prob, theta, b):
    from ace_jax.fit.ard import rows_fn_for
    return rows_fn_for(prob, theta)(b)


def test_dtc_shape_is_rotation_equivariant(gp_fit_dir):
    """Review Focus 5 for the dtc shape: a diagonal DTC term rotates only up to its anisotropy, so the test
    holds the trace (forces_std) invariant and V's kappa part equivariant."""
    from scipy.spatial.transform import Rotation
    from ace_jax.fit.ard import dtc_shape
    from ace_jax.fit.hypers import from_array
    out, d, res = gp_fit_dir
    post, prob = res.ard.posterior._replace(variance="dtc", R=None), res.built.prob
    theta = from_array(np.asarray(post.gp_theta))
    b = jax.tree.map(lambda a: a[0], d.ds_test)
    Rm = Rotation.from_euler("zyx", [0.4, 0.2, -0.9]).as_matrix()
    V0 = dtc_shape(post, prob, theta, b); V1 = dtc_shape(post, prob, theta, b._replace(rij=b.rij @ Rm.T))
    live = np.asarray(b.node_mask)
    np.testing.assert_allclose(np.trace(V1[live], axis1=1, axis2=2), np.trace(V0[live], axis1=1, axis2=2), rtol=1e-8)


def test_stage_ard_gp_dtc_runs():
    _, _, _, res, pred = _gp_stage(ard_variance="dtc")
    assert res.posterior.variance == "dtc" and np.isfinite(pred.F_var).all()
```

The diagonal DTC term is not rotation-equivariant as a 3×3 tensor: per-component variances along the lab
axes are a basis-dependent approximation of the true DTC covariance. The test therefore pins the invariant
part, the trace, which is what `forces_std` serves. Say this in `dtc_shape`'s docstring and in the user
docs: `forces_cov` and aniso `forces_q` from `dtc` depend on the lab frame at the level of the DTC term's
anisotropy. **Flag this at review.** The alternative is a full 3×3 derivative-DTC block. That needs the
mixed second derivatives per component pair in `_dtc_deriv_residual`, about 3× its cost, and is out of scope
unless review asks for it.

- [ ] **Step 2: Run, expect failure** (`ImportError: cannot import name 'dtc_shape'`).

- [ ] **Step 3: Implement**

`dtc_shape(post, prob, theta, batch)`:
- `lin, res, X, JU0 = batch_rows_parts(theta, ..., with_X=True, with_JU0=True)`;
- `F = _cat_rows(lin, res).F`;
- `W = solve_triangular(chol, post.prior_root.rows(F).reshape(-1, Dt).T, lower=True)`, reshaped to
  (Ncap, 3, Dt), then `Vk = einsum("nar,nbr->nab", W, W)`;
- `Fv, _ = _dtc_deriv_residual(theta, prob, batch, X, res=res, JU0=JU0)`;
- return `Vk + Fv[..., None] * eye(3)`.

Use `atom_chunk` for the solve, as `ARDPosterior.atom_shape`'s κ branch does.

In `run_ard_stage`, under `variance == "dtc"`:
- The hold-out posterior keeps no R/Q (as κ).
- `_val_atoms` builds V from `dtc_shape` per batch.
- The served posterior gets `variance="dtc"`.

`predict_ard` and `GPCalculator` call `dtc_shape` when `post.variance == "dtc"`. `_warn_deriv_dtc_size`
already fires on big batches.

- [ ] **Step 4: Run** `uv run pytest tests/test_ard_gp.py -q`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/ard.py src/ace_jax/calc/gp.py tests/test_ard_gp.py
git commit -m "feat(ard): --ard-variance dtc, the kappa shape plus the derivative-DTC diagonal (ard-gp only)"
```

---

### Task 8: CLI, docs and the Cantor acceptance

**Files:**
- Modify: `src/ace_jax/cli.py:97-110` (`--uq` choices + help, `--ard-variance` choices + help)
- Modify: `docs/user/howto/force-uncertainty.md`, `docs/user/concepts/force-uncertainty-maths.md`
  (ASD-STE100; Terms table in `docs/user/concepts.md` if a new term appears), `skills/ace-jax/SKILL.md`,
  `CHANGELOG.md`
- Modify: `bench/defect_uq/modal/ard_arms.py`: GP arms `ard_gp_sw`, `ard_gp_dtc` on top of the `gp_conv`
  settings (PCA-16, host-cache, θ re-fitted)
- Modify: `bench/defect_uq/modal/modal_bench365.py`: `big_errors` serves a `gp_model.npz` run through
  `GPCalculator(posterior=)`, and records peak device memory per cell
- Create: `bench/defect_uq/results/2026-10-xx_ard_gp_acceptance.md`, and a section in
  `docs/dev/gp-discrepancy-exploration.md`
- Test: `tests/test_ard_calc_cli.py` (append a `--uq ard-gp` CLI round trip on the tiny fixture)

- [ ] **Step 1: CLI test** (append to `tests/test_ard_calc_cli.py`)

```python
def test_cli_fit_and_eval_ard_gp(tmp_path):
    import subprocess, sys
    from conftest import FIXTURE_DIR
    out = tmp_path / "fit"
    subprocess.run([sys.executable, "-m", "ace_jax.cli", "fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"),
                    "--data", str(FIXTURE_DIR / "si_tiny_train.xyz"), "--energy-key", "dft_energy",
                    "--force-key", "dft_force", "--virial-key", "dft_virial", "--ntrain", "30", "--ntest", "8",
                    "--m-per-species", "8", "--uq", "ard-gp", "--ard-val-frac", "0.4", "--ard-n-min", "50",
                    "--map-steps", "5", "--r0", "2.35", "--out", str(out)], check=True)
    assert (out / "gp_model.npz").exists() and (out / "posterior.npz").exists()
    r = subprocess.run([sys.executable, "-m", "ace_jax.cli", "eval", "--model", str(out / "gp_model.npz"),
                        "--posterior", str(out / "posterior.npz"), "--data", str(FIXTURE_DIR / "si_tiny_train.xyz"),
                        "--force-key", "dft_force", "--energy-key", "dft_energy", "--out", str(tmp_path / "e.xyz")],
                       check=True, capture_output=True, text=True)
    from ace_jax.fit.xyz import read_extxyz
    a = read_extxyz(tmp_path / "e.xyz")[0]
    assert "ace_forces_std" in a.arrays or any(k.endswith("forces_std") for k in a.arrays)
```

Match the flag names to `aj fit --help` and the output-key prefix that `aj eval` writes today. Run
`uv run aj eval --help` first, and keep the test to the existing conventions.

- [ ] **Step 2: Implement the CLI**
- `--uq` choices `["blr", "pops", "ard", "ard-gp"]`, with help:
  "`ard-gp`: the GP arm's jackknife-sandwich posterior over [B | k(B, B_M)] at the MAP hyperparameters,
  calibrated as --uq ard; needs --m-per-species > 0".
- `--ard-variance` choices `["sandwich", "kappa", "dtc"]`, with help for `dtc`: "the posterior shape plus
  the GP's derivative-DTC variance (ard-gp, cosine kernel)".
- Run the test: `uv run pytest tests/test_ard_calc_cli.py -q -k ard_gp`.

- [ ] **Step 3: Docs.**
- `force-uncertainty.md`: a numbered procedure "Calibrated force uncertainty from a GP fit"
  (`aj fit --m-per-species 100 --uq ard-gp`, then `GPCalculator.from_file(..., posterior=...)` /
  `aj eval --posterior`). Add a warning admonition: `dtc`'s `forces_cov` depends on the axes at the level
  of the DTC term (Task 7).
- `force-uncertainty-maths.md`: the prior root R0 = blockdiag(diag Γ, chol K_MMᵀ), the scaled system, and
  why the PRESS identity carries over.
- `SKILL.md`: the option and the calculator.
- `CHANGELOG.md` "Unreleased": `--uq ard-gp` and `--ard-variance dtc`; the linear `--uq ard` is unchanged.
- Build: `uv pip install -r docs/requirements.txt && ACEJAX_DOCS_NOTEBOOKS=skip uv run --no-sync mkdocs build --strict`.

- [ ] **Step 4: Full suite**

Run: `ACEJAX_TEST_WORKERS=4 uv run pytest -q && uv run ruff check`
Expected: green. Commit:

```bash
git add src/ace_jax/cli.py docs skills CHANGELOG.md tests/test_ard_calc_cli.py
git commit -m "docs(ard): --uq ard-gp and --ard-variance dtc in the CLI, how-to, maths page, SKILL and CHANGELOG"
```

- [ ] **Step 5: Bench arms and the Modal `big_errors` for GP runs**
- `ard_arms.py`: a `GP_ARMS` dict, built from `gp_conv`'s `FitConfig` kwargs plus
  `{"uq": "ard-gp", "ard_force_shape": "aniso"}` and `"ard_variance": "sandwich"` (arm `ard_gp_sw`) or
  `"dtc"` (arm `ard_gp_dtc`). `make_config` resolves it like `D4_ARMS`, and the bench test checks the
  configs validate.
- `modal_bench365.py::big_errors`: if `/out/{run}/gp_model.npz` exists, use
  `GPCalculator.from_file(..., posterior=...)`. Record `jax.devices()[0].memory_stats()["peak_bytes_in_use"]`
  per configuration into the err file as `peak_bytes` (the spec's memory report for 3–4k-atom cells).

- [ ] **Step 6: Ask James before launching.**
- Estimate: 2 GP fits at ~3.5 B200-h each (`gp_conv` took 3.5 h, and the ARD stage adds ~1 h) plus
  `big_errors` on 5 files × 2 runs (A100-80GB; the GP rows are L + M = 15 540 wide, about 1.5× the linear
  run's time).
- **≈ 9–10 GPU-h in all, above the "few GPU-hours" line.**

- [ ] **Step 7: After approval, run, score and decide.**
- Score: `scoring/validate_shape.py` and `scoring/d4_score.py` (the ID column) on `bench365_ard_gp_sw`,
  `bench365_ard_gp_dtc` and `bench365_ard_default_pol`.
- Write the table to `bench/defect_uq/results/2026-10-xx_ard_gp_acceptance.md` and a summary section in
  `docs/dev/gp-discrepancy-exploration.md`.
- Include: tip coverage, ρ on the big cells, ID coverage, test RMSE, fit time and the peak device memory
  per cell.
- **Acceptance (spec):** either GP shape matches or beats the linear rev2 result on crack-tip coverage
  (0.893) **and** on Spearman ρ on the large cells (0.37 / 0.30 / 0.32). If neither does, record the result
  and keep `ard-gp` experimental: no default change, and the CHANGELOG entry says experimental.
- Commit with `feat(bench):` / `docs(dev):` and stop for review.
