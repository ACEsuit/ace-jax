# Tempered ARD Posterior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A linear ACE fit reports a calibrated per-atom force uncertainty `forces_std`: the tempered ARD-per-body-order BLR posterior. It is available from `aj fit --uq ard`, from the saved `posterior.npz`, and from `ACECalculator(..., posterior=...)`, including on single cells of >2.8k atoms.

**Architecture:**
- A new module `fit/ard.py` holds:
  - the evidence of the linear model in the prior-scaled system;
  - its joint (or sequential) type-II maximisation;
  - the posterior object (`ArdPosterior`);
  - the held-out temperature κ;
  - batched prediction.
- `fit/rows.py` gains `linear_rows_chunked`, which evaluates the edge Jacobian in node chunks so no tensor exceeds int32 indexing.
- The pipeline adds an ARD stage after the linear MAP when `uq == "ard"`, and writes `posterior.npz` and `ard.json`.
- `ACECalculator` serves `forces_std` from `posterior.npz`.

**Tech Stack:** Python ≥ 3.11, JAX (float64), NumPy, SciPy (L-BFGS-B), ASE; pytest (+ xdist); ruff.

**Spec:** `docs/specs/2026-09-28-tempered-ard-uq-design.md`. Evidence for the design: PR #12, `bench/defect_uq/README.md`.

**Deviations from the spec (rulings):**
- The spec routes the linear-arm training statistics through `linear_rows_chunked` as well. This
  plan keeps `linear_rows` there: training cells are small, and that path is proven and cached.
  Chunked rows serve every *prediction* path (ARD prediction, the validation errors for κ, and the
  calculator), which is where big cells occur.
  - Cost if wrong: a training set containing a >2.8k-atom cell would hit the int32 limit.
    `stats.batch_rows` would then switch to the chunked rows, a one-line change.
- The spec's gradient test asks for "forward-mode vs reverse-mode". The plan uses a directional
  central finite difference of the value against the JAX gradient. That checks the same thing and
  does not need a second AD mode through a Cholesky.

## Global Constraints

- Python ≥ 3.11. All numerics in float64 with `jax.config.update("jax_enable_x64", True)`, except the stored posterior factor, which is float32 by default.
- No new dependencies. SciPy, JAX and ASE are already dependencies.
- Code passes `uv run ruff check`. Style is dense one-line where the surrounding code is (see `[tool.ruff]` in `pyproject.toml`).
- `skills/ace-jax/SKILL.md` and `README.md` must describe any CLI or API change (CLAUDE.md rule).
- The GP arm is untouched. `uq="ard"` is valid only for the linear arm (`m_per_species 0`).
- `fixtures/` are bit-exact goldens: do not edit them.
- Evidence (spec): log p(D|h) = −½ Σ_q yᵀy_q/σ_q² + ½ bᵀA⁻¹b − ½ log|A| + ½ log|Λ| − Σ_q n_q log σ_q.
  - It is evaluated in the prior-scaled system S = D⁻¹AD⁻¹, D = diag(Γ).
  - h = (log σ_E, log σ_F, log σ_V, a_k per body order); Λ = Γ² ⊙ exp(a_{k(j)}).
- Optimiser (spec, measured):
  - L-BFGS-B on the objective *relative* to its start and *divided by the initial gradient norm*;
  - bounds log σ ∈ h₀ ± 3 and a ∈ [a_floor, 10], with a_floor = log(λ_max(Ms(h₀)) / cond_max) and cond_max = 1e14;
  - a non-finite evaluation is a rejected step (returns 1e30, zero gradient).
- κ: the closed form κ² = mean(|ΔF|² / (s²/3)) / 3 over held-out atoms, with s² = Σ_c φ_c A⁻¹ φ_cᵀ. κ is fitted on an internal train hold-out (`ard_val_frac`, default 0.2), never on the test set. The final posterior is refit on all the training data with the same κ.
- Defaults: `ard_mode="joint"` (alternative `"sequential"`), `ard_val_frac=0.2`, `ard_cond_max=1e14`.

## Review Focus

1. **Atoms with no neighbours** (isolated atom, vacuum). A force row of zeros must give `forces_std == 0`, finite, not NaN. Task 5 tests it.
2. **A `posterior.npz` that does not match the model** (different basis size or species). The calculator must refuse with a clear error, not silently mis-index. Task 5 tests it.
3. **Chunk sizes that do not divide the number of nodes, a chunk larger than the batch, and padded nodes.** The chunked rows must equal the unchunked rows exactly. Task 1 tests it.
4. **A validation split with no force labels** (tiny data, or energy-only configs). The ARD stage must raise a ValueError naming `ard_val_frac`, not a NaN κ. Task 3 tests it.
5. **Float32 posterior storage.** A reloaded posterior's σ must agree with float64 to ≤ 1e-4 relative. Task 2 tests it.

---

## File Structure

- `src/ace_jax/fit/rows.py`: **modify**. Add `linear_rows_chunked(model, cfg, batch, node_chunk=256) -> Rows`.
- `src/ace_jax/fit/ard.py`: **create**. It contains:
  - `body_order_columns`, `ArdStats`, `ard_statistics`, `ArdEvidence`, `fit_ard`, `laplace_hypers`;
  - `ArdPosterior` (with `save`/`load`), `ard_posterior`;
  - `predict_ard`, `kappa_closed_form`, `run_ard_stage`, `ArdResult`.
- `src/ace_jax/fit/pipeline/config.py`: **modify**. `uq` gains `"ard"`; add `ard_mode`, `ard_val_frac`, `ard_cond_max`, `ard_laplace`, with validation.
- `src/ace_jax/fit/pipeline/run.py`: **modify**. The ARD stage, and `FitResult.ard`.
- `src/ace_jax/fit/pipeline/predict.py`: **modify**. The ARD predictive branch.
- `src/ace_jax/fit/pipeline/outputs.py`, `export.py`: **modify**. Write `posterior.npz` and `ard.json`; `model.npz` uses the ARD mean.
- `src/ace_jax/calc/point.py`: **modify**. `ACECalculator(..., posterior=None)` and the `forces_std` result.
- `src/ace_jax/cli.py`: **modify**. `fit --uq ard --ard-mode --ard-val-frac`; `eval --posterior --per-atom`.
- Tests:
  - `tests/test_rows_chunked.py`, `tests/test_ard.py`, `tests/test_ard_pipeline.py`, `tests/test_ard_calc_cli.py`: **create**.
- Docs:
  - `README.md`, `skills/ace-jax/SKILL.md`: **modify**.

---

### Task 1: Node-chunked linear design rows

**Files:**
- Modify: `src/ace_jax/fit/rows.py` (add after `linear_rows`, ~line 68)
- Test: `tests/test_rows_chunked.py`

**Interfaces:**
- Consumes: `model.edge_jacobian_dense(rij (n,K,3), zi (n,K), zj (n,K), mask (n,K)) -> (X (n,D), J (n*K,D,3))`; `_place`, `_voigt`, `Rows` (same file).
- Produces: `linear_rows_chunked(model, cfg, batch, node_chunk=256) -> Rows`. Identical values to `linear_rows(model, cfg, batch)[0]`; never materialises the batch's full J.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rows_chunked.py
"""linear_rows_chunked == linear_rows: the node-chunked edge Jacobian (one chunk's J at a time,
so a >2.8k-atom cell never builds a >2^31-element tensor) gives the same design rows."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def _problem(n_cfg=8, per_batch=4):
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    cfgs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    big = [c for c in cfgs if len(c.numbers) == 64][:1]           # a 64-atom cell among small ones
    cfgs = big + [c for c in cfgs if len(c.numbers) < 64][: n_cfg - 1]
    ds = build_dataset(cfgs, meta, np.asarray(z["E0"]), per_batch)
    g = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                 NZ=len(meta["elements"]), C=per_batch)
    return model, g, ds


@pytest.mark.parametrize("chunk", [1, 7, 64, 10_000])
def test_chunked_rows_equal_unchunked(chunk):
    from ace_jax.fit.rows import linear_rows, linear_rows_chunked
    model, g, ds = _problem()
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        ref = linear_rows(model, g, b)[0]
        got = linear_rows_chunked(model, g, b, node_chunk=chunk)
        for k in ("E", "F", "V"):
            np.testing.assert_allclose(np.asarray(getattr(got, k)), np.asarray(getattr(ref, k)),
                                       rtol=1e-12, atol=1e-12, err_msg=f"{k} chunk={chunk}")


@pytest.mark.slow
def test_chunked_rows_big_cell():
    """A 4096-atom diamond Si cell: chunked rows are finite and satisfy the force/energy
    consistency sum_i F_i = 0 row-wise (translation invariance), without building the
    whole-cell edge Jacobian."""
    from ase.build import bulk
    from ace_jax.eval import load
    from ace_jax.fit.data import Config, build_dataset
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import linear_rows_chunked
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((8, 8, 8))
    at.rattle(0.05, seed=1)
    c = Config(at.positions, at.numbers, at.cell.array, at.pbc, None, None, None, 1.0, 1.0, 1.0)
    ds = build_dataset([c], meta, np.asarray(z["E0"]), 1)
    g = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=1, C=1)
    r = linear_rows_chunked(model, g, jax.tree.map(lambda a: a[0], ds), node_chunk=512)
    F = np.asarray(r.F)
    assert np.all(np.isfinite(F))
    np.testing.assert_allclose(F.sum(axis=0), 0.0, atol=1e-8)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_rows_chunked.py -v -m "not slow"`
Expected: FAIL with `ImportError: cannot import name 'linear_rows_chunked'`

- [ ] **Step 3: Write the implementation** (in `src/ace_jax/fit/rows.py`, after `linear_rows`)

```python
def linear_rows_chunked(model, cfg, batch, node_chunk=256):
    """`linear_rows(model, cfg, batch)[0]` computed a chunk of centre nodes at a time.

    `linear_rows` materialises the edge Jacobian J (Ncap*K, D, 3) of the whole batch.  For one
    cell of more than ~2.8k atoms at the production Cantor basis (D = 3007, K ~ 85) that tensor
    and the GEMM building it exceed 2^31 elements, and XLA's int32-indexed GEMM autotuning fails.
    Here each chunk's J (node_chunk*K, D, 3) is built, scattered into E/F/V and dropped; the rows
    are identical (tests/test_rows_chunked.py).  Returns Rows only (no X, J)."""
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]
    L = cfg.len_basis
    nc = int(min(node_chunk, Ncap))
    n_chunks = -(-Ncap // nc)
    pad = n_chunks * nc - Ncap
    Np = Ncap + pad
    rij = jnp.concatenate([batch.rij, jnp.broadcast_to(batch.rij[-1:], (pad, K, 3))])
    nbr = jnp.concatenate([batch.nbr, jnp.zeros((pad, K), batch.nbr.dtype)])
    msk = jnp.concatenate([batch.nbr_mask, jnp.zeros((pad, K), bool)])
    node_z = jnp.concatenate([batch.node_z, jnp.zeros(pad, batch.node_z.dtype)])
    node_cfg = jnp.concatenate([batch.node_cfg, jnp.full(pad, C, batch.node_cfg.dtype)])
    seg = lambda a, ids, n: jax.ops.segment_sum(a, ids, num_segments=n)
    local = jnp.repeat(jnp.arange(nc), K)

    def body(i, acc):
        Enodes, F, V = acc
        s0 = i * nc
        r_c = jax.lax.dynamic_slice_in_dim(rij, s0, nc)
        nb_c = jax.lax.dynamic_slice_in_dim(nbr, s0, nc)
        m_c = jax.lax.dynamic_slice_in_dim(msk, s0, nc)
        z_c = jax.lax.dynamic_slice_in_dim(node_z, s0, nc)
        X, J = model.edge_jacobian_dense(r_c, jnp.broadcast_to(z_c[:, None], (nc, K)), node_z[nb_c], m_c)
        send, recv = s0 + local, nb_c.reshape(-1)
        zi, r_flat, edge_cfg = z_c[local], r_c.reshape(-1, 3), node_cfg[s0 + local]
        E_c = jnp.zeros((nc, L))
        for z in range(cfg.NZ):
            E_c = _place(E_c, jnp.where((z_c == z)[:, None], X, 0.0), z, cfg)
            Jz = jnp.where((zi == z)[:, None, None], J, 0.0)
            dEdr = seg(Jz, recv, Np) - seg(Jz, send, Np)                      # (Np, D, 3)
            F = _place(F, -jnp.swapaxes(dEdr, 1, 2), z, cfg)
            V = _place(V, jnp.swapaxes(seg(_voigt(Jz, r_flat), edge_cfg, C + 1), 1, 2), z, cfg)
        Enodes = jax.lax.dynamic_update_slice_in_dim(Enodes, E_c, s0, 0)
        return Enodes, F, V

    init = (jnp.zeros((Np, L)), jnp.zeros((Np, 3, L)), jnp.zeros((C + 1, 6, L)))
    Enodes, F, V = jax.lax.fori_loop(0, n_chunks, body, init)
    E = seg(Enodes, node_cfg, C + 1)[:C]
    return Rows(E, F[:Ncap], V[:C])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rows_chunked.py -v -m "not slow"` and then `uv run pytest tests/test_rows_chunked.py -v -m slow`
Expected: 4 passed (not slow); 1 passed (slow).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/ace_jax/fit/rows.py tests/test_rows_chunked.py
git add src/ace_jax/fit/rows.py tests/test_rows_chunked.py
git commit -m "feat(rows): linear_rows_chunked -- node-chunked edge Jacobian for >2.8k-atom cells"
```

---

### Task 2: ARD evidence, joint/sequential type-II ML, posterior object

**Files:**
- Create: `src/ace_jax/fit/ard.py`
- Test: `tests/test_ard.py`

**Interfaces:**
- Consumes:
  - `stats.sufficient_statistics(theta, spec, model, ind, cfg, ds) -> Stats`, with fields G_E, G_F, G_V, b_E, b_F, b_V, yy_E, yy_F, yy_V, n_E, n_F, n_V;
  - `rows.linear_rows(model, cfg, batch)`;
  - `prob.gamma` (L,);
  - `theta.log_sigma_{E,F,V}` and `theta.log_sigma_c`.
- Produces:
  - `body_order_columns(meta, cfg) -> np.ndarray (L,) int`: body order 2, 3 or 4 per column, in `_place` layout.
  - `ArdStats(G: tuple, b: tuple, yy: np.ndarray, n: np.ndarray, ls_fixed: np.ndarray | None)`.
    - Joint mode: G and b have 3 entries, one per quantity.
    - Sequential mode: 1 entry each (the combined Gram at `ls_fixed`).
  - `ard_statistics(theta, prob, ds, mode) -> ArdStats`.
  - `ArdEvidence(stats, gamma, body_col)`:
    - `.groups` is a sorted tuple of body orders;
    - `.h0(theta) -> np.ndarray`;
    - `.value_and_grad(h) -> (float, np.ndarray)`;
    - `.free` is the free-parameter index mask;
    - `.lower` and `.upper` are the bounds;
    - `.a_floor`.
  - `fit_ard(ev, h0, cond_max=1e14) -> (h, logev, info)`.
  - `laplace_hypers(ev, h, eps=1e-3) -> dict(std, eigs, cov, interior)`.
  - `ArdPosterior`:
    - fields `mean` (L,), `chol` (L,L), `dinv` (L,), `kappa` (float), `h` (P,), `groups` (tuple), `body_col` (L,) and `meta` (dict with n_B, n_pair, NZ, rcut, elements);
    - `var_rows(Phi (n,L)) -> (n,)`;
    - `forces_std(Frows (N,3,L)) -> (N,)`;
    - `.save(path, dtype=np.float32)` and the static `ArdPosterior.load(path)`.
  - `ard_posterior(ev, h, kappa, meta) -> ArdPosterior`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ard.py
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision


def _dense_rows(prob, ds):
    """Weighted live rows (w*phi, w*y) and the quantity index (0 E, 1 F, 2 V) of every observation."""
    from ace_jax.fit.rows import linear_rows
    rows, ys, qs = [], [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r = linear_rows(prob.model, prob.cfg, b)[0]
        L = r.E.shape[-1]
        for k, (Phi, y, w) in enumerate(((r.E, b.y_E, b.w_E),
                                         (r.F.reshape(-1, L), b.y_F.reshape(-1), jnp.repeat(b.w_F, 3)),
                                         (r.V.reshape(-1, L), b.y_V.reshape(-1), jnp.repeat(b.w_V, 6)))):
            w = np.asarray(w); live = w > 0
            rows.append(np.asarray(Phi)[live] * w[live, None]); ys.append(np.asarray(y)[live] * w[live])
            qs.append(np.full(live.sum(), k))
    return np.concatenate(rows), np.concatenate(ys), np.concatenate(qs)


def _dense_logev(Phi, y, q, h, gamma, gidx, groups_n):
    s2 = np.exp(2 * h[:3])[q]
    lam = gamma ** 2 * np.exp(h[3:3 + groups_n])[gidx]
    Cov = (Phi / lam) @ Phi.T + np.diag(s2)
    _, ld = np.linalg.slogdet(Cov)
    return -0.5 * y @ np.linalg.solve(Cov, y) - 0.5 * ld


def test_body_order_columns_layout():
    from ace_jax.fit.ard import body_order_columns
    from ace_jax.fit.inducing import GPConfig
    meta = {"nnll": [[(1, 0)], [(1, 0), (1, 1)], [(1, 0), (1, 1), (2, 1)]]}
    g = GPConfig(r0=2.35, rcut=5.0, n_B=3, n_pair=2, NZ=2, C=1)
    col = body_order_columns(meta, g)
    # species blocks of B first (orders 1,2,3 -> body 2,3,4), then pair blocks (body 2)
    assert col.tolist() == [2, 3, 4, 2, 3, 4, 2, 2, 2, 2]


def test_evidence_matches_dense_marginal_likelihood(tiny_linear_problem):
    """Differences of log p between two h equal those of the explicit Gaussian marginal likelihood
    N(y_w | 0, Phi_w Lambda^-1 Phi_w^T + diag(sigma_q^2)) (constants cancel)."""
    from ace_jax.fit.ard import ArdEvidence, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint")
        meta = {"nnll": [[None] * o for o in np.asarray(_orders(prob))]}
        ev = ArdEvidence(st, np.asarray(prob.gamma), body_order_columns(meta, prob.cfg))
        h1 = ev.h0(theta); h2 = h1 + np.r_[0.1, -0.2, 0.05, [0.3, -0.4, 0.2][: len(ev.groups)]]
        v1, v2 = ev.value_and_grad(h1)[0], ev.value_and_grad(h2)[0]
        Phi, y, q = _dense_rows(prob, ds)
        gidx = np.searchsorted(np.asarray(ev.groups), body_order_columns(meta, prob.cfg))
        d1 = _dense_logev(Phi, y, q, h1, np.asarray(prob.gamma), gidx, len(ev.groups))
        d2 = _dense_logev(Phi, y, q, h2, np.asarray(prob.gamma), gidx, len(ev.groups))
    assert abs((v2 - v1) - (d2 - d1)) < 1e-6 * max(1.0, abs(d2 - d1))


def _orders(prob):
    """Correlation order of each B column of the tiny problem's model (all body orders present)."""
    import json
    from conftest import FIXTURE_DIR
    z = np.load(FIXTURE_DIR / "si_fitted.npz")
    return [len(x) for x in json.loads(bytes(z["meta_json"]).decode())["nnll"]]


def test_gradient_matches_finite_differences(tiny_linear_problem):
    from ace_jax.fit.ard import ArdEvidence, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint")
        meta = {"nnll": [[None] * o for o in _orders(prob)]}
        ev = ArdEvidence(st, np.asarray(prob.gamma), body_order_columns(meta, prob.cfg))
        h = ev.h0(theta); v, g = ev.value_and_grad(h)
        d = np.random.default_rng(0).standard_normal(len(h)); d /= np.linalg.norm(d); e = 1e-5
        fd = (ev.value_and_grad(h + e * d)[0] - ev.value_and_grad(h - e * d)[0]) / (2 * e)
    assert abs(fd - g @ d) < 1e-5 * max(1.0, abs(fd))


def test_joint_fit_beats_sequential_beats_start(tiny_linear_problem):
    from ace_jax.fit.ard import ArdEvidence, ard_statistics, body_order_columns, fit_ard
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)]}
    with highest_precision():
        evJ = ArdEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                          body_order_columns(meta, prob.cfg))
        evS = ArdEvidence(ard_statistics(theta, prob, ds, "sequential"), np.asarray(prob.gamma),
                          body_order_columns(meta, prob.cfg))
        h0J, h0S = evJ.h0(theta), evS.h0(theta)
        hS, vS, _ = fit_ard(evS, h0S)
        hJ, vJ, info = fit_ard(evJ, h0J)
        v0 = evJ.value_and_grad(h0J)[0]
        # the sequential optimum, embedded in the joint parameterisation, is a feasible joint point
        vS_in_J = evJ.value_and_grad(np.concatenate([h0J[:3], hS]))[0]
    assert vS_in_J >= v0 - 1e-6
    assert vJ >= vS_in_J - 1e-4
    assert np.all(hJ >= evJ.lower - 1e-12) and np.all(hJ <= evJ.upper + 1e-12)


def test_posterior_variance_matches_blr_path_at_gamma_prior(tiny_linear_problem):
    """With a_k = log(1/sigma_c^2) for every group and sigma_q at theta, the ARD posterior is the
    existing BLR posterior: same mean, same row variances."""
    from ace_jax.fit.ard import ArdEvidence, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.objective import posterior
    from ace_jax.fit.predict import _rows_mean_var
    from ace_jax.fit.rows import linear_rows
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}
    with highest_precision():
        ev = ArdEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        post = ard_posterior(ev, ev.h0(theta), 1.0, meta)
        mu, Lc = posterior(theta, sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds), prob)
        b = jax.tree.map(lambda a: a[0], ds)
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, b)[0].F).reshape(-1, prob.cfg.len_basis)
        m_ref, v_ref = _rows_mean_var(jnp.asarray(Fr), mu, Lc)
    np.testing.assert_allclose(Fr @ post.mean, np.asarray(m_ref), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(post.var_rows(Fr), np.asarray(v_ref), rtol=1e-6, atol=1e-14)


def test_posterior_save_load_float32(tiny_linear_problem, tmp_path):
    from ace_jax.fit.ard import ArdEvidence, ArdPosterior, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.rows import linear_rows
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}
    with highest_precision():
        ev = ArdEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        post = ard_posterior(ev, ev.h0(theta), 2.5, meta)
        post.save(tmp_path / "posterior.npz")
        back = ArdPosterior.load(tmp_path / "posterior.npz")
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    a, b = post.forces_std(Fr), back.forces_std(Fr)
    assert back.kappa == 2.5 and back.meta["n_B"] == prob.cfg.n_B
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-12)


def test_laplace_reports_interior_psd(tiny_linear_problem):
    from ace_jax.fit.ard import ArdEvidence, ard_statistics, body_order_columns, fit_ard, laplace_hypers
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)]}
    with highest_precision():
        ev = ArdEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        h, _, _ = fit_ard(ev, ev.h0(theta))
        rep = laplace_hypers(ev, h)
    assert np.all(np.linalg.eigvalsh(rep["cov"]) >= -1e-12)
    assert np.all(np.isfinite(rep["std"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.ard'`

- [ ] **Step 3: Write the implementation** (`src/ace_jax/fit/ard.py`, the first part; Task 3 appends the rest)

```python
"""Tempered ARD posterior: calibrated per-atom force uncertainty for linear ACE.

Spec: docs/specs/2026-09-28-tempered-ard-uq-design.md.  For the linear model the weighted design
rows do not depend on the hyperparameters, so one statistics pass (G_q, b_q, y^T y_q, n_q per
quantity) gives the exact evidence for any h = (log sigma_E, log sigma_F, log sigma_V, a_k):

    log p(D|h) = -1/2 sum_q yy_q/s_q^2 + 1/2 b^T A^-1 b - 1/2 log|A| + 1/2 log|Lambda| - sum_q n_q log s_q
    A = sum_q G_q/s_q^2 + Lambda,  Lambda = Gamma^2 exp(a_k(j))

evaluated in the prior-scaled system S = D^-1 A D^-1 (D = diag Gamma), where the Gamma terms of
log|A| and log|Lambda| cancel: cond(A) reached 1e17 on the production Cantor basis, cond(S) 1e13.
"""
import json
import pathlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

SCHEMA = 1


def body_order_columns(meta, cfg):
    """Body order (2, 3, 4, ...) of every design column, in the `rows._place` layout: species blocks
    of the n_B many-body columns (correlation order nu -> body order nu + 1), then species blocks of
    the n_pair pair columns (body order 2)."""
    order_B = np.array([len(x) for x in meta["nnll"]]) + 1
    return np.concatenate([np.tile(order_B, cfg.NZ), np.full(cfg.n_pair * cfg.NZ, 2)])


class ArdStats(NamedTuple):
    G: tuple                 # (G_E, G_F, G_V) [joint] or (M,) combined at ls_fixed [sequential]
    b: tuple
    yy: np.ndarray           # (3,) [joint]; unused [sequential]
    n: np.ndarray            # (3,)
    ls_fixed: np.ndarray | None


def ard_statistics(theta, prob, ds, mode):
    """joint: per-quantity statistics (3 L^2 matrices).  sequential: the combined Gram at the
    linear MAP noise scales, accumulated in one pass (1 L^2 matrix) -- the low-memory mode."""
    from .stats import sufficient_statistics
    if mode == "joint":
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        return ArdStats(tuple(getattr(st, f"G_{q}") for q in "EFV"), tuple(getattr(st, f"b_{q}") for q in "EFV"),
                        np.array([float(getattr(st, f"yy_{q}")) for q in "EFV"]),
                        np.array([float(getattr(st, f"n_{q}")) for q in "EFV"]), None)
    if mode != "sequential":
        raise ValueError(f"ard mode must be 'joint' or 'sequential', got {mode!r}")
    from .rows import linear_rows
    ls = np.array([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
    inv = jnp.asarray(np.exp(-ls))
    L = prob.cfg.len_basis

    def body(acc, batch):
        M, bv = acc
        r = linear_rows(prob.model, prob.cfg, batch)[0]
        for k, (Phi, y, w) in enumerate(((r.E, batch.y_E, batch.w_E),
                                         (r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3)),
                                         (r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6)))):
            Pw = Phi * (w * inv[k])[:, None]
            M = M + Pw.T @ Pw
            bv = bv + Pw.T @ (y * w * inv[k])
        return (M, bv), None

    (M, bv), _ = jax.lax.scan(jax.checkpoint(body), (jnp.zeros((L, L)), jnp.zeros(L)), ds)
    return ArdStats((M,), (bv,), np.zeros(3), np.zeros(3), ls)


class ArdEvidence:
    """log p(D|h) and its gradient in the prior-scaled system.  h = (log sigma_q [joint only], a_k)."""

    def __init__(self, stats, gamma, body_col):
        self.joint = stats.ls_fixed is None
        self.groups = tuple(int(g) for g in np.unique(body_col))
        self.body_col = np.asarray(body_col)
        gidx = jnp.asarray(np.searchsorted(np.asarray(self.groups), self.body_col))
        dinv = jnp.asarray(1.0 / np.asarray(gamma))
        self.dinv = dinv
        Gs = tuple(dinv[:, None] * G * dinv[None, :] for G in stats.G)
        bs = tuple(dinv * b for b in stats.b)
        yy, nq = jnp.asarray(stats.yy), jnp.asarray(stats.n)
        nls = 3 if self.joint else 0

        def parts(h):
            if self.joint:
                w = jnp.exp(-2 * h[:3])
                Ms = w[0] * Gs[0] + w[1] * Gs[1] + w[2] * Gs[2]
                bv = w[0] * bs[0] + w[1] * bs[1] + w[2] * bs[2]
                const = -0.5 * jnp.sum(yy * w) - jnp.sum(nq * h[:3])
            else:
                Ms, bv, const = Gs[0], bs[0], 0.0
            return Ms, bv, jnp.exp(h[nls:])[gidx], const

        def logev(h):
            Ms, bv, lam, const = parts(h)
            c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
            x = cho_solve((c, low), bv)
            return const + 0.5 * bv @ x - jnp.sum(jnp.log(jnp.diag(c))) + 0.5 * jnp.sum(jnp.log(lam))

        self._parts, self._vg = parts, jax.jit(jax.value_and_grad(logev))
        self._nls = nls

    def h0(self, theta):
        a_blr = float(-2 * theta.log_sigma_c)
        ls = [float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"] if self.joint else []
        return np.array(ls + [a_blr] * len(self.groups))

    def value_and_grad(self, h):
        v, g = self._vg(jnp.asarray(h, float))
        return float(v), np.asarray(g)

    def bounds(self, h0, cond_max):
        """log sigma within h0 +- 3; a in [a_floor, 10] with a_floor = log(lambda_max(Ms(h0)) / cond_max):
        the prior never lets cond(S) exceed ~cond_max (at the unfloored Cantor ARD optimum cond(S) was
        4e17 and repeated evaluations differed by 0.3 nats)."""
        Ms = self._parts(jnp.asarray(h0, float))[0]
        v = jnp.ones(Ms.shape[0]) / np.sqrt(Ms.shape[0])
        for _ in range(50):                                  # power iteration: lambda_max(Ms)
            v = Ms @ v; v = v / jnp.linalg.norm(v)
        lam_max = float(v @ (Ms @ v))
        self.a_floor = float(np.log(max(lam_max, 1e-300) / cond_max))
        k = self._nls
        lo = np.concatenate([np.asarray(h0[:k]) - 3.0, np.full(len(self.groups), self.a_floor)])
        hi = np.concatenate([np.asarray(h0[:k]) + 3.0, np.full(len(self.groups), 10.0)])
        self.lower, self.upper = lo, hi
        return lo, hi


def fit_ard(ev, h0, cond_max=1e14, maxiter=500):
    """Type-II ML by L-BFGS-B, bounded (ev.bounds), on the objective relative to its start and divided
    by the initial gradient norm: L-BFGS-B's first bounded step is the full gradient (~3e3 on the
    Cantor problem), which lands on the box corner; a non-finite point is a rejected step."""
    from scipy.optimize import minimize
    lo, hi = ev.bounds(h0, cond_max)
    x0 = np.clip(np.asarray(h0, float), lo, hi)
    v0, g0 = ev.value_and_grad(x0)
    gs = max(1.0, float(np.linalg.norm(g0)))

    def f(z):
        v, g = ev.value_and_grad(z)
        if not (np.isfinite(v) and np.all(np.isfinite(g))):
            return 1e30, np.zeros_like(z)
        return -(v - v0) / gs, -g / gs

    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                 options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6})
    v, _ = ev.value_and_grad(r.x)
    return r.x, v, {"message": str(r.message), "nit": int(r.nit), "gain": v - v0,
                    "at_bound": ((np.isclose(r.x, lo)) | (np.isclose(r.x, hi))).tolist()}


def laplace_hypers(ev, h, eps=1e-3):
    """Diagnostic: Laplace approximation of the hyperparameter posterior at h (central differences of
    the exact gradient); the covariance uses interior coordinates only and is PSD by construction."""
    P = len(h)
    H = np.zeros((P, P))
    for i in range(P):
        e = np.zeros(P); e[i] = eps
        H[:, i] = (ev.value_and_grad(h + e)[1] - ev.value_and_grad(h - e)[1]) / (2 * eps)
    H = 0.5 * (H + H.T)
    interior = ~(np.isclose(h, ev.lower) | np.isclose(h, ev.upper))
    cov = np.zeros((P, P))
    w, V = np.linalg.eigh(-H[np.ix_(interior, interior)])
    cov[np.ix_(interior, interior)] = (V / np.clip(w, 1e-12, None)) @ V.T
    return {"std": np.sqrt(np.diag(cov)), "eigs": np.linalg.eigvalsh(-H), "cov": cov,
            "interior": interior.tolist()}


class ArdPosterior(NamedTuple):
    mean: np.ndarray          # (L,) posterior mean coefficients (the readout)
    chol: np.ndarray          # (L, L) lower Cholesky factor of S = D^-1 A D^-1
    dinv: np.ndarray          # (L,) 1 / Gamma
    kappa: float              # temperature (sigma inflation)
    h: np.ndarray
    groups: tuple
    body_col: np.ndarray
    meta: dict                # n_B, n_pair, NZ, rcut, elements

    def var_rows(self, Phi, chunk=4096):
        """Untempered posterior variance phi A^-1 phi^T of each row of Phi (n, L)."""
        c = jnp.asarray(self.chol, jnp.float64)
        out = []
        for i in range(0, len(Phi), chunk):
            v = solve_triangular(c, (jnp.asarray(Phi[i:i + chunk]) * jnp.asarray(self.dinv)[None, :]).T, lower=True)
            out.append(np.asarray(jnp.sum(v * v, axis=0)))
        return np.concatenate(out) if out else np.zeros(0)

    def forces_std(self, Frows):
        """Tempered per-atom force std kappa * sqrt(sum_c phi_c A^-1 phi_c^T) from force rows (N, 3, L)."""
        Frows = np.asarray(Frows)
        v = self.var_rows(Frows.reshape(-1, Frows.shape[-1])).reshape(-1, 3)
        return self.kappa * np.sqrt(np.maximum(v.sum(1), 0.0))

    def save(self, path, dtype=np.float32):
        np.savez(path, mean=self.mean, chol=np.asarray(self.chol, dtype), dinv=self.dinv, kappa=self.kappa,
                 h=self.h, groups=np.asarray(self.groups), body_col=self.body_col, schema=SCHEMA,
                 meta_json=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))

    @staticmethod
    def load(path):
        z = np.load(pathlib.Path(path))
        if int(z["schema"]) != SCHEMA:
            raise ValueError(f"unsupported posterior schema {int(z['schema'])}")
        return ArdPosterior(z["mean"], z["chol"].astype(np.float64), z["dinv"], float(z["kappa"]), z["h"],
                            tuple(int(g) for g in z["groups"]), z["body_col"],
                            json.loads(bytes(z["meta_json"]).decode()))


def ard_posterior(ev, h, kappa, meta):
    Ms, bv, lam, _ = ev._parts(jnp.asarray(h, float))
    c, low = cho_factor(Ms + jnp.diag(lam), lower=True)
    x = cho_solve((c, low), bv)
    keep = {k: meta[k] for k in ("n_B", "n_pair", "NZ", "rcut", "elements") if k in meta}
    if "NZ" not in keep and "elements" in keep:          # model meta carries elements, not NZ
        keep["NZ"] = len(keep["elements"])
    return ArdPosterior(np.asarray(ev.dinv * x), np.asarray(c), np.asarray(ev.dinv), float(kappa),
                        np.asarray(h, float), ev.groups, ev.body_col, keep)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ard.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/ace_jax/fit/ard.py tests/test_ard.py
git add src/ace_jax/fit/ard.py tests/test_ard.py
git commit -m "feat(ard): prior-scaled evidence, joint/sequential type-II ML, Laplace diagnostic, ArdPosterior"
```

---

### Task 3: Temperature κ, batched ARD prediction, the ARD stage

**Files:**
- Modify: `src/ace_jax/fit/ard.py` (append)
- Test: `tests/test_ard.py` (append)

**Interfaces:**
- Consumes:
  - Task 2's API;
  - `rows.linear_rows_chunked` (Task 1);
  - `predict._pack(outs, prob, ds)` and `predict.Prediction`;
  - `data.build_dataset(configs, meta, E0, per_batch)`;
  - `pipeline FitData` (`train` configs, `meta`, `E0`, `ds_train`);
  - `Built.prob`.
- Produces:
  - `kappa_closed_form(e2, s2) -> float`, where e2 is the per-atom |ΔF|² and s2 = Σ_c var.
  - `predict_ard(post, prob, ds, node_chunk=256) -> Prediction`. F_var is tempered by κ²; E_var and V_var are untempered.
  - `ArdResult(posterior: ArdPosterior, report: dict)`.
  - `run_ard_stage(cfg, data, built, theta, log=print) -> ArdResult`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ard.py`)

```python
def test_kappa_closed_form_minimises_nll():
    from scipy.optimize import minimize_scalar
    from ace_jax.fit.ard import kappa_closed_form
    rng = np.random.default_rng(1)
    s2 = rng.uniform(0.01, 1.0, 500)
    e2 = (3.2 ** 2) * s2 / 3 * rng.chisquare(3, 500)          # true kappa 3.2
    k = kappa_closed_form(e2, s2)
    nll = lambda lk: np.mean(e2 / (2 * np.exp(2 * lk) * s2 / 3) + 1.5 * np.log(np.exp(2 * lk) * s2 / 3))
    k_bf = float(np.exp(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x))
    assert abs(k - k_bf) < 1e-4 * k_bf and 2.9 < k < 3.5


def _pipe_cfg(**kw):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear", uq="ard",
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False)
    return FitConfig(**{**base, **kw})


def test_ard_stage_fits_kappa_and_refits_on_all_training_data():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.ard import run_ard_stage
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
    rep = res.report
    assert res.posterior.kappa > 0 and np.isfinite(res.posterior.kappa)
    assert rep["n_val_atoms"] > 0 and rep["n_fit_configs"] + rep["n_val_configs"] == len(d.train)
    assert abs(rep["val_rms_z_tempered"] - 1.0) < 1e-6          # kappa closed form on the val set
    assert rep["logev_full"] >= rep["logev_full_start"] - 1e-6


def test_ard_stage_rejects_val_split_without_forces():
    from ace_jax.fit.ard import run_ard_stage
    from ace_jax.fit.pipeline import load_fit_data
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg(ard_val_frac=0.2).validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    # strip every force label: the validation split then has nothing to fit kappa on
    d = d._replace(train=[c._replace(forces=None, w_F=0.0) for c in d.train])
    with pytest.raises(ValueError, match="ard_val_frac"):
        run_ard_stage(cfg, d, build_problem(cfg, d), None, log=lambda *a: None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ard.py -v -k "kappa or ard_stage"`
Expected: FAIL with `ImportError: cannot import name 'kappa_closed_form'` (and `FitConfig` rejecting `uq="ard"`, which Task 4 adds; the two `ard_stage` tests stay red until Task 4, as recorded in Step 4).

- [ ] **Step 3: Write the implementation** (append to `src/ace_jax/fit/ard.py`)

```python
def kappa_closed_form(e2, s2):
    """NLL-optimal kappa for dF ~ N(0, kappa^2 s^2/3 I_3) per atom: kappa^2 = mean(e2 / (s2/3)) / 3."""
    e2, s2 = np.asarray(e2, float), np.asarray(s2, float)
    ok = s2 > 0
    return float(np.sqrt(np.mean(e2[ok] / (s2[ok] / 3.0)) / 3.0))


def predict_ard(post, prob, ds, node_chunk=256):
    """Posterior predictive on a Dataset: means from the ARD mean; F_var tempered by kappa^2 (the
    calibrated quantity), E_var/V_var the untempered posterior variances."""
    from .predict import _pack
    from .rows import linear_rows_chunked
    L, k2 = prob.cfg.len_basis, post.kappa ** 2
    outs = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r = linear_rows_chunked(prob.model, prob.cfg, b, node_chunk=node_chunk)
        E, F, V = np.asarray(r.E), np.asarray(r.F).reshape(-1, L), np.asarray(r.V).reshape(-1, L)
        outs.append((E @ post.mean, post.var_rows(E), (F @ post.mean).reshape(-1, 3),
                     k2 * post.var_rows(F).reshape(-1, 3), (V @ post.mean).reshape(-1, 6),
                     post.var_rows(V).reshape(-1, 6)))
    return _pack(outs, prob, ds)


class ArdResult(NamedTuple):
    posterior: ArdPosterior
    report: dict


def _val_errors(post, prob, ds):
    """Per-atom squared force error and untempered s2 on the live force rows of ds."""
    from .rows import linear_rows_chunked
    L, e2, s2 = prob.cfg.len_basis, [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        live = np.asarray(b.w_F) > 0
        if not live.any():
            continue
        F = np.asarray(linear_rows_chunked(prob.model, prob.cfg, b).F)[live]          # (n, 3, L)
        pred = (F.reshape(-1, L) @ post.mean).reshape(-1, 3)
        e2.append(np.sum((np.asarray(b.y_F)[live] - pred) ** 2, 1))
        s2.append(post.var_rows(F.reshape(-1, L)).reshape(-1, 3).sum(1))
    if not e2:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(e2), np.concatenate(s2)


def run_ard_stage(cfg, data, built, theta, log=print):
    """Fit ARD on a train subset, choose kappa on the held-out rest (never the test set), then refit
    on ALL training data (started from the subset optimum) and keep kappa."""
    import time
    from .data import build_dataset
    t0 = time.time()
    prob = built.prob
    body_col = body_order_columns(data.meta, prob.cfg)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(data.train))
    nval = max(1, int(round(cfg.ard_val_frac * len(data.train))))
    val = [data.train[i] for i in idx[:nval]]
    fit_ = [data.train[i] for i in idx[nval:]]
    if not any(c.forces is not None and c.w_F > 0 for c in val):
        raise ValueError(f"the ARD validation split ({nval} configs, ard_val_frac={cfg.ard_val_frac}) has "
                         f"no force labels to fit the temperature kappa on")
    ds_fit = build_dataset(fit_, data.meta, data.E0, cfg.batch)
    ds_val = build_dataset(val, data.meta, data.E0, cfg.batch)
    ev = ArdEvidence(ard_statistics(theta, prob, ds_fit, cfg.ard_mode), np.asarray(prob.gamma), body_col)
    h_fit, v_fit, info_fit = fit_ard(ev, ev.h0(theta), cfg.ard_cond_max)
    post_fit = ard_posterior(ev, h_fit, 1.0, data.meta)
    e2, s2 = _val_errors(post_fit, prob, ds_val)
    ok = s2 > 0                            # atoms with zero force rows (isolated, 1-atom configs): no information
    e2, s2 = e2[ok], s2[ok]
    if len(e2) == 0:
        raise ValueError(f"the ARD validation split (ard_val_frac={cfg.ard_val_frac}) has no atom with a "
                         f"non-zero force row to fit the temperature kappa on")
    kappa = kappa_closed_form(e2, s2)
    log(f"ARD: fit-subset logev {v_fit:.2f} ({info_fit['message']}, nit {info_fit['nit']}); "
        f"kappa {kappa:.3f} from {len(e2)} held-out atoms")
    del ev
    ev = ArdEvidence(ard_statistics(theta, prob, data.ds_train, cfg.ard_mode), np.asarray(prob.gamma), body_col)
    v_start = ev.value_and_grad(np.clip(h_fit, *ev.bounds(h_fit, cfg.ard_cond_max)))[0]   # refit's start
    h, v, info = fit_ard(ev, h_fit, cfg.ard_cond_max)
    post = ard_posterior(ev, h, kappa, data.meta)
    report = {"mode": cfg.ard_mode, "groups": list(ev.groups), "h": h.tolist(),
              "h_names": (["log_sigma_E", "log_sigma_F", "log_sigma_V"] if ev.joint else [])
              + [f"a_{g}body" for g in ev.groups],
              "logev_full": v, "logev_full_start": v_start, "optimiser": info, "a_floor": ev.a_floor,
              "tempered_quantities": ["F"],        # E_var / V_var are the untempered posterior variances
              "kappa": kappa, "n_val_atoms": int(len(e2)), "n_val_configs": len(val), "n_fit_configs": len(fit_),
              "val_rms_z_untempered": float(np.sqrt(np.mean(e2 / (s2 / 3)) / 3)),
              "val_rms_z_tempered": float(np.sqrt(np.mean(e2 / (kappa ** 2 * s2 / 3)) / 3)),
              "seconds": time.time() - t0}
    if cfg.ard_laplace:
        lap = laplace_hypers(ev, h)
        report["laplace_std_h"] = lap["std"].tolist(); report["laplace_eigs"] = lap["eigs"].tolist()
    return ArdResult(post, report)
```

- [ ] **Step 4: Run the tests** — `kappa_closed_form` passes now; the two `ard_stage` tests need `FitConfig.uq="ard"` (Task 4)

Run: `uv run pytest tests/test_ard.py -v -k kappa`
Expected: 1 passed. Record in the ledger that the `ard_stage` tests are completed by Task 4.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/ace_jax/fit/ard.py tests/test_ard.py
git add src/ace_jax/fit/ard.py tests/test_ard.py
git commit -m "feat(ard): held-out temperature, batched tempered prediction, ARD stage"
```

---

### Task 4: Pipeline integration (`uq="ard"`)

**Files:**
- Modify:
  - `src/ace_jax/fit/pipeline/config.py` (fields near line 49, validation near line 63);
  - `src/ace_jax/fit/pipeline/run.py` (`FitResult`, `fit`);
  - `src/ace_jax/fit/pipeline/predict.py` (`predict_splits`, the branch near line 101);
  - `src/ace_jax/fit/pipeline/outputs.py` (`write_outputs`);
  - `src/ace_jax/fit/pipeline/export.py` (`linear_model_arrays`).
- Test: `tests/test_ard_pipeline.py`

**Interfaces:**
- Consumes: `run_ard_stage`, `predict_ard`, `ArdResult` (Task 3).
- Produces:
  - `FitConfig.uq ∈ {"blr", "pops", "ard"}`;
  - `FitConfig.ard_mode="joint"`, `ard_val_frac=0.2`, `ard_cond_max=1e14`, `ard_laplace=False`;
  - `FitResult.ard: ArdResult | None` (last field, default None);
  - `write_outputs` writes `posterior.npz` and `ard.json` when `res.ard`;
  - `model.npz` holds the ARD mean when `res.ard`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ard_pipeline.py
import json

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR


def _cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear", uq="ard",
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False, predict_stats="recompute")
    return FitConfig(**{**base, **kw})


def test_config_validates_ard():
    _cfg().validate()
    with pytest.raises(ValueError, match="linear"):
        _cfg(arm="gp", m_per_species=6).validate()
    with pytest.raises(ValueError, match="ard_mode"):
        _cfg(ard_mode="both").validate()
    with pytest.raises(ValueError, match="ard_val_frac"):
        _cfg(ard_val_frac=1.0).validate()


@pytest.fixture(scope="module")
def ard_fit():
    from ace_jax.fit.pipeline import fit, load_fit_data
    cfg = _cfg().validate()
    return fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None)


def test_fit_ard_predicts_with_tempered_force_variance(ard_fit):
    from ace_jax.fit.ard import predict_ard
    res = ard_fit
    assert res.ard is not None and res.ard.posterior.kappa > 0
    p = res.preds.arrays["test/map"]
    again = predict_ard(res.ard.posterior, res.built.prob, res.data.ds_test)
    np.testing.assert_allclose(p["F_var"], np.asarray(again.F_var), rtol=1e-10)
    np.testing.assert_allclose(p["F_mean"], np.asarray(again.F_mean), rtol=1e-10, atol=1e-12)


def test_write_outputs_saves_posterior_and_report(ard_fit, tmp_path):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ArdPosterior
    from ace_jax.fit.pipeline import write_outputs
    from ase import Atoms
    write_outputs(ard_fit, tmp_path, layout=("cli",), argv={})
    post = ArdPosterior.load(tmp_path / "posterior.npz")
    rep = json.load(open(tmp_path / "ard.json"))
    assert post.kappa == pytest.approx(ard_fit.ard.posterior.kappa) and rep["mode"] == "joint"
    calc = ACECalculator(str(tmp_path / "model.npz"))                  # model.npz = the ARD mean
    E = []
    for c in ard_fit.data.test:
        at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc); at.calc = calc
        E.append(at.get_potential_energy())
    np.testing.assert_allclose(E, ard_fit.preds.arrays["test/map"]["E_mean"], rtol=1e-9, atol=1e-9)


def test_sequential_mode_runs(tmp_path):
    from ace_jax.fit.pipeline import fit, load_fit_data
    cfg = _cfg(ard_mode="sequential").validate()
    res = fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None)
    assert res.ard.report["mode"] == "sequential" and len(res.ard.report["h"]) == len(res.ard.report["groups"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ard_pipeline.py -v`
Expected: FAIL. `FitConfig.__init__()` got an unexpected keyword argument (or validate rejects `uq="ard"`).

- [ ] **Step 3: Write the implementation**

`config.py`: after the `uq` field:
```python
    uq: str = "blr"                      # "blr" | "pops" | "ard"
    ard_mode: str = "joint"              # "joint" (sigma_q + ARD scales) | "sequential" (low memory)
    ard_val_frac: float = 0.2            # train hold-out for the temperature kappa
    ard_cond_max: float = 1e14           # prior floor: cond(S) <= ard_cond_max
    ard_laplace: bool = False            # Laplace diagnostic of the hyperparameters (ard.json)
```
and in `validate()`, next to the POPS check:
```python
        if self.uq not in ("blr", "pops", "ard"):
            raise ValueError(f"uq must be 'blr', 'pops' or 'ard', got {self.uq!r}")
        if self.uq == "ard":
            if self.arm != "linear":
                raise ValueError("uq='ard' is the linear-arm tempered posterior: use arm linear (m_per_species 0)")
            if self.ard_mode not in ("joint", "sequential"):
                raise ValueError(f"ard_mode must be 'joint' or 'sequential', got {self.ard_mode!r}")
            if not 0.0 < self.ard_val_frac < 1.0:
                raise ValueError(f"ard_val_frac must be in (0, 1), got {self.ard_val_frac}")
```

`run.py`:
```python
class FitResult(NamedTuple):
    config: object; data: object; built: object; theta: object
    map: object; rungs: object; preds: object; timings: dict
    ard: object = None
```
and in `fit()`, after `stage("rungs", rg)`:
```python
        ard = None
        if cfg.uq == "ard":
            from ..ard import run_ard_stage
            obj = obj._replace(lik=None, vg=None, host_cache=None)
            release()
            ard = run_ard_stage(cfg, data, b, mf.theta, log=log)
            stage("ard", ard)
```
Pass `ard` into `predict_splits` (new keyword `ard=None`). Add `ard.report["seconds"]` to the timings as `"ard"`, and return `FitResult(..., tm, ard)`.

`predict.py` (`predict_splits(cfg, d, b, stats, theta, draws, log=print, ard=None)`), in the per-split branch:
```python
            if cfg.uq == "ard":
                from ..ard import predict_ard
                pred = predict_ard(ard.posterior, prob, ds)
            elif cfg.uq == "pops":
```

`outputs.py` (`write_outputs`), before `if save_model:`:
```python
    if res.ard is not None:
        res.ard.posterior.save(out / "posterior.npz")
        _dump(out / "ard.json", res.ard.report)
```

`export.py` (`linear_model_arrays`), first branch:
```python
    if res.ard is not None:
        mu = res.ard.posterior.mean           # ARD: the posterior mean the predictions use
    elif "mean" in res.preds.pops:
```

- [ ] **Step 4: Run the tests to verify they pass** (including Task 3's stage tests)

Run: `uv run pytest tests/test_ard_pipeline.py tests/test_ard.py -v`
Expected: all pass (4 + 10).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/ace_jax/fit/pipeline tests/test_ard_pipeline.py
git commit -m "feat(pipeline): uq='ard' -- ARD stage after the linear MAP, tempered predictions, posterior.npz + ard.json"
```

---

### Task 5: Serving: ACECalculator `forces_std`, CLI

**Files:**
- Modify: `src/ace_jax/calc/point.py` (`ACECalculator.__init__`, `implemented_properties`, the end of `calculate`)
- Modify: `src/ace_jax/cli.py` (the `fit` parser near line 59, `_fit_config`, the `eval` parser, `cmd_eval`)
- Test: `tests/test_ard_calc_cli.py`

**Interfaces:**
- Consumes: `ArdPosterior.load/forces_std` (Task 2), `linear_rows_chunked` (Task 1), `build_dataset`, `Config`, `GPConfig`, `ace_jax.eval.load`.
- Produces:
  - `ACECalculator(model_path, posterior=None, ...)`. When given a posterior, `results["forces_std"]` is an (N,) array.
  - `aj fit --uq ard [--ard-mode joint|sequential] [--ard-val-frac F]`.
  - `aj eval --posterior P [--per-atom out.xyz]`, which adds an `fmax_std` column.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ard_calc_cli.py
import jax
import numpy as np
import pytest
from ase import Atoms

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    from ace_jax.cli import main
    out = tmp_path_factory.mktemp("ard")
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "30",
                 "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                 "dft_virial", "--m-per-species", "0", "--uq", "ard", "--opt", "lbfgs", "--map-steps", "5",
                 "--configs-per-batch", "4", "--r0", "2.35", "--out", str(out)]) == 0
    return out


def test_calculator_forces_std_matches_pipeline(fitted):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ArdPosterior, predict_ard
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.pipeline import load_fit_data
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    cfgs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:3]
    post = ArdPosterior.load(fitted / "posterior.npz")
    cfg = FitConfig(model=str(fitted / "model.npz"), arm="linear", r0=2.35, batch=1, energy_key="dft_energy",
                    force_key="dft_force", virial_key="dft_virial", e0="model").validate()
    d = load_fit_data(cfg, train=str(XYZ), test=str(XYZ))
    prob = build_problem(cfg, d).prob
    ds = build_dataset(cfgs, d.meta, d.E0, 1)
    ref = np.sqrt(np.asarray(predict_ard(post, prob, ds).F_var).sum(1))
    got = []
    for c in cfgs:
        at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc); at.calc = calc
        at.get_forces(); got.append(calc.results["forces_std"])
    np.testing.assert_allclose(np.concatenate(got), ref, rtol=1e-6, atol=1e-12)


def test_isolated_atom_has_zero_finite_std(fitted):
    from ace_jax import ACECalculator
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at = Atoms("Si", positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=False); at.calc = calc
    at.get_forces()
    s = calc.results["forces_std"]
    assert s.shape == (1,) and np.all(np.isfinite(s)) and s[0] == 0.0


def test_mismatched_posterior_is_refused(fitted, tmp_path):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ArdPosterior
    p = ArdPosterior.load(fitted / "posterior.npz")
    bad = p._replace(mean=p.mean[:-1], dinv=p.dinv[:-1], chol=p.chol[:-1, :-1], body_col=p.body_col[:-1])
    bad.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="posterior"):
        ACECalculator(str(fitted / "model.npz"), posterior=str(tmp_path / "bad.npz"))


def test_cli_eval_with_posterior_writes_per_atom_std(fitted, tmp_path):
    import csv
    from ase.io import read, write
    from ace_jax.cli import main
    data = tmp_path / "d.xyz"
    write(data, read(XYZ, ":3"))
    assert main(["eval", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
                 "--data", str(data), "--energy-key", "dft_energy", "--force-key", "dft_force", "--forces",
                 "--out", str(tmp_path / "p.csv"), "--per-atom", str(tmp_path / "atoms.xyz")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "p.csv")))
    assert len(rows) == 3 and float(rows[0]["fmax_std"]) > 0
    ats = read(tmp_path / "atoms.xyz", ":")
    assert len(ats) == 3 and ats[0].arrays["forces_std"].shape == (len(ats[0]),)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ard_calc_cli.py -v`
Expected: FAIL (argparse `invalid choice: 'ard'`).

- [ ] **Step 3: Write the implementation**

`cli.py`, in the fit parser:
```python
    p.add_argument("--uq", choices=["blr", "pops", "ard"], default="blr",
                   help="pops/ard: linear arm (--m-per-species 0); ard = tempered ARD posterior "
                        "(calibrated per-atom forces_std, writes posterior.npz)")
    p.add_argument("--ard-mode", choices=["joint", "sequential"], default="joint",
                   help="joint: noise + ARD scales by evidence; sequential: ARD only, one Gram (low memory)")
    p.add_argument("--ard-val-frac", type=float, default=0.2,
                   help="train fraction held out to fit the temperature kappa")
```
In `_fit_config`, pass `ard_mode=a.ard_mode, ard_val_frac=a.ard_val_frac`.

In the eval parser:
```python
    ev.add_argument("--posterior", default=None, help="posterior.npz from `fit --uq ard`: adds forces_std")
    ev.add_argument("--per-atom", default=None, help="extxyz with per-atom forces and forces_std arrays")
```
In `cmd_eval`, replace the model-selection block and the per-config evaluation:
```python
    gp = str(a.model).endswith(".npz") and "gp_json" in np.load(a.model).files   # gp_model.npz from `fit`
    ard = getattr(a, "posterior", None) is not None
    if gp or ard:
        from ase import Atoms
        if gp:
            from .calc.gp import GPCalculator
            calc = GPCalculator.from_file(a.model)
        else:
            from .calc.point import ACECalculator
            calc = ACECalculator(a.model, posterior=a.posterior)
    else:
        model, meta, z = load(a.model)
        rcut = float(meta["rcut"])
    esq = ecnt = fsq = fcnt = 0.0
    rows, per_atom = [], []
    with highest_precision():
        for i, c in enumerate(configs):
            if gp or ard:
                at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
                at.calc = calc
                E, F = at.get_potential_energy(), at.get_forces()
            else:
                g = sparse_graph(c.positions, c.cell, c.pbc, rcut)
                nz = jnp.asarray(species_indices(meta, c.numbers))
                send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
                E, F, V = model.energy_forces_virial(jnp.asarray(g.rij), nz[send], nz[recv],
                                                     send, recv, g.n_nodes, nz)
            E = float(E); F = np.asarray(F); nat = len(c.numbers)
            rows.append({"config": i, "natoms": nat, "energy": E,
                         "energy_per_atom": E / nat, "fmax": float(np.abs(F).max())})
            if gp:
                rows[-1]["energy_std"] = float(calc.results["energy_std"])
            if ard:
                s = np.asarray(calc.results["forces_std"])
                rows[-1]["fmax_std"] = float(s.max())
                if a.per_atom:
                    out_at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
                    out_at.arrays["forces_pred"] = F; out_at.arrays["forces_std"] = s
                    per_atom.append(out_at)
            if c.energy is not None:
                esq += ((E - c.energy) / nat) ** 2; ecnt += 1
            if c.forces is not None:
                fsq += float(((F - c.forces) ** 2).sum()); fcnt += c.forces.size
    if per_atom:
        from ase.io import write as _write
        _write(a.per_atom, per_atom)
        print(f"wrote per-atom forces_std for {len(per_atom)} configs to {a.per_atom}")
```
(the rest of `cmd_eval`, i.e. CSV output and RMSE lines, is unchanged).

`point.py`, in `ACECalculator`:
```python
    implemented_properties = ["energy", "free_energy", "forces", "stress", "site_descriptors", "forces_std"]
```
Add `posterior=None` to `__init__`'s keyword arguments. As the first line of `__init__`'s body
(before `_resolve` replaces `model` with the loaded object), keep the path:
`model_path = model`. At the end of `__init__`:
```python
        self.posterior = None
        if posterior is not None:
            from ..eval import load as _load_fit_model
            from ..fit.ard import ArdPosterior
            from ..fit.inducing import GPConfig
            if not isinstance(model_path, (str, bytes)) and not hasattr(model_path, "__fspath__"):
                raise ValueError("posterior= needs the model FILE path (the design rows use the fit model)")
            post = ArdPosterior.load(posterior)
            L = (meta["n_B"] + meta["n_pair"]) * len(meta["elements"])
            if len(post.mean) != L or post.meta.get("n_B") != meta["n_B"] or post.meta.get("NZ") != len(meta["elements"]):
                raise ValueError(f"posterior {posterior} does not match the model: basis {len(post.mean)} vs {L}")
            self.posterior = post
            self._fit_model = _load_fit_model(model_path)[0]
            self._fit_cfg = GPConfig(r0=1.0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                                     NZ=len(meta["elements"]), C=1)
```
At the end of `calculate`, after the stress block:
```python
        if self.posterior is not None:
            self.results["forces_std"] = self._forces_std()
```
And the method:
```python
    def _forces_std(self):
        """Tempered ARD per-atom force std (posterior.npz), from node-chunked design rows."""
        import jax
        from ..fit.data import Config, build_dataset
        from ..fit.rows import linear_rows_chunked
        at = self.atoms
        c = Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                   None, None, None, 1.0, 1.0, 1.0)
        ds = build_dataset([c], self.meta, np.zeros(len(self.meta["elements"])), 1)
        b = jax.tree.map(lambda a: a[0], ds)
        if b.nbr.shape[1] == 0 or not bool(np.asarray(b.nbr_mask).any()):
            return np.zeros(len(at))                     # no neighbours: forces are identically zero
        with highest_precision():
            F = np.asarray(linear_rows_chunked(self._fit_model, self._fit_cfg, b).F)
        F = F[np.asarray(b.node_mask)]
        return self.posterior.forces_std(F)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ard_calc_cli.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests
git add src/ace_jax/calc/point.py src/ace_jax/cli.py tests/test_ard_calc_cli.py
git commit -m "feat(calc,cli): ACECalculator(posterior=) forces_std; aj fit --uq ard; aj eval --posterior --per-atom"
```

---

### Task 6: Docs, full suite

**Files:**
- Modify: `README.md` (the section "Command line (`ace-jax` / `aj`)", after the POPS example)
- Modify: `skills/ace-jax/SKILL.md` (the "Choosing options" table and "Outputs")

- [ ] **Step 1: README**. Add after the POPS example block:

````markdown
# calibrated per-atom force uncertainty: tempered ARD posterior (linear model)
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --uq ard --opt lbfgs --r0 2.35 --out out_ard
aj eval --model out_ard/model.npz --posterior out_ard/posterior.npz --data big.xyz $K \
    --forces --per-atom atoms_std.xyz          # per-atom forces_std, e.g. to colour a crack tip
````

and a short paragraph:
- `--uq ard` fits prior scales per body order and the noise scales by evidence (joint type-II ML).
- A single temperature κ, from a train hold-out, calibrates the force σ. On the Cantor defect
  benchmark it held rms-z 0.91–1.02 on held-out defect combinations.
- `posterior.npz` stores the float32 posterior factor, ~0.9 GB at L = 15k.
- `--ard-mode sequential` is the low-memory fallback.

- [ ] **Step 2: SKILL.md**. In the "Choosing options" table, add the row
`| Calibrated per-atom force uncertainty (e.g. big-cell fracture) | --m-per-species 0 --uq ard (posterior.npz; ACECalculator(model, posterior=...)) |`.
Under Outputs, add `posterior.npz`, `ard.json` and the `forces_std` calculator result.

- [ ] **Step 3: Run the full suite and lint**

Run: `uv run pytest` and then `uv run pytest -m slow tests/test_rows_chunked.py` and then `uv run ruff check`
Expected: all pass; no ruff errors.

- [ ] **Step 4: Commit**

```bash
git add README.md skills/ace-jax/SKILL.md
git commit -m "docs: --uq ard (tempered ARD posterior), posterior.npz, forces_std"
```

---

### Task 7: Acceptance on the defect benchmark (manual, GPU)

The spec's acceptance criterion cannot be a unit test: it needs the bench365 data and a B200. It
uses the harness from PR #12 (`bench/defect_uq`).

- [ ] **Step 1:** In `bench/defect_uq/modal/fit_bench.py` (PR #12 branch, rebased on this branch), add an `ard` arm: `FitConfig(**common, arm="linear", uq="ard")`. Launch it with `modal run modal_bench365.py::launch --arms ard`.
- [ ] **Step 2:** Score it with `scoring/score_atoms.py` and `scoring/calibrate.py`. The acceptance bar:
  - AUROC on the four held-out combination families within ±0.03 of the benchmark's `sdF_ardG` row (0.75 / 0.90 / 0.85 / 0.88);
  - rms-z within 0.85–1.15 and 90 % coverage within 0.85–0.95 on each held-out family.
- [ ] **Step 3:** Big cells: `aj eval --model model.npz --posterior posterior.npz --data big.xyz --per-atom`. Report rms-z and 90 % coverage of `forces_std` on the crack and dislocation interiors. This is the first big-cell calibration result; record it in `bench/defect_uq/README.md` whatever it shows.
- [ ] **Step 4:** Record the outcome in the ledger and in PR #12's README.
