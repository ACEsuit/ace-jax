# Learned Radial Basis by VarPro — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn the analytic tensor-radial weights `rnl_Wnlq` of a linear ACE model by variable projection (readout projected out, L-BFGS outer loop), with a held-out gate against the initial radials.

**Architecture:** Two new modules. `fit/radial_model.py` holds model-level helpers (swap/widen/convert `Wnlq`, the empirical radial Gram `Q` for the gauge, the roughness matrix `D2`). `fit/radial_learn.py` holds the VarPro objective `yy − bᵀ(G+Λ)⁻¹b` computed from the existing streamed `linear_statistics`, the L-BFGS loop with θ re-profiling, and the held-out gate. `construct/radial_init.py` gains `n_q_factor` and a numpy `from_table` projector; `construct/export.py` gains `patch_radial_npz` so a learned model can be written from any loaded npz. A bench driver runs the whole thing on real data.

**Tech Stack:** JAX (float64), equinox, optax 0.2.8 (`optax.lbfgs`), numpyro (θ-MAP via `fit.ladder.run_map`), numpy, pytest via `uv run`.

**Spec:** `docs/dev/specs/2026-09-26-learned-radial-varpro-design.md`

## Global Constraints

- Float64 throughout; library entry points raise if `jax_enable_x64` is off.
- No torch / mace / mace-jax dependency anywhere in ace-jax.
- Do not edit `src/ace_jax/eval/model.py` (PR #7 rewrites it); only use `eqx.tree_at` / `dataclasses.replace` on `ACEModel`.
- Do not edit `cli.py` or `bench/acegp_cantor/run.py` (PR #8 touches them); CLI integration is deferred until #8 merges.
- Learned parameters: `rnl_Wnlq` only. Transform, envelopes, pair basis fixed.
- Residual GP not involved: every problem here has M = 0 inducing points.
- Defaults that must not change behaviour: `n_q_factor=1.5` reproduces today's `n_q = ceil(1.5·maxn)` exactly.
- Run tests with `uv run pytest …` from the worktree root `/Users/u1470235/gits/ace-jax/.worktrees/learn-radial` (env: `uv sync --extra gp`).
- Commit messages: conventional (`feat(radial_learn): …`), ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Identity (onehot) init with more than one species** has structurally-zero radial rows (`R_n(zi,zj) = 0` for `zj ≠ z'(n)`). Normalising them must not give 0/0 NaNs; they stay zero and frozen. → Task 4, `test_normalise_zero_rows_stay_zero_and_finite`.
2. **A species pair absent from the training data** (e.g. Ge–Ge in an all-Si set) leaves its empirical Gram empty; `Q` must still be positive definite. → Task 4, `test_radial_gram_absent_pair_is_positive_definite`.
3. **A splined Julia-exported model** passed to the learner must fail with an error naming `to_analytic`, not silently learn nothing. → Task 3, `test_non_analytic_model_raises`.
4. **The objective going non-finite mid-run** (Cholesky of `G+Λ` fails, or float32) must leave the best finite iterate, never NaN weights; float32 must be refused up front. → Task 7, `test_lbfgs_loop_nonfinite_keeps_best` and `test_learn_radial_requires_x64`.
5. **A validation split with no force rows** (energies only) must give a finite gate score, skipping the empty type. → Task 8, `test_holdout_score_energy_only_split`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ace_jax/construct/radial_init.py` (modify) | `n_q_factor`; numpy `from_table` (sampled radials → `Wnlq`) |
| `src/ace_jax/construct/model.py` (modify) | thread `n_q_factor` through `build_model` |
| `src/ace_jax/construct/export.py` (modify) | `patch_radial_npz(src, dst, model)` |
| `src/ace_jax/fit/radial_model.py` (create) | `require_analytic`, `with_radial`, `widen_radial`, `to_analytic`, `poly_env`, `radial_gram`, `row_active`, `normalise`, `roughness_matrix`, `roughness`, `rnl_degrees` |
| `src/ace_jax/fit/radial_learn.py` (create) | `projected_residual_from_stats`, `projected_residual`, `lbfgs_loop`, `theta_map_linear`, `learn_radial`, `holdout_score`, `gate`, `fit_radial`, `save_result` |
| `bench/learn_radial/run.py` (create) | end-to-end driver on an xyz dataset |
| `tests/test_radial_init_table.py` (create) | Tasks 1–2 |
| `tests/test_gp_radial_model.py` (create) | Tasks 3–4 |
| `tests/test_gp_learn_radial.py` (create) | Tasks 5–8 |

---

### Task 1: `n_q_factor` knob

**Files:**
- Modify: `src/ace_jax/construct/radial_init.py:256-276` (`tensor_radial_init`)
- Modify: `src/ace_jax/construct/model.py:92-95` (signature), `:128-129` (call), meta `"authoring"` dict
- Test: `tests/test_radial_init_table.py`

**Interfaces:**
- Produces: `tensor_radial_init(..., n_q_factor=1.5)`; `build_model(..., n_q_factor=1.5)`; `meta["authoring"]["n_q_factor"]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_radial_init_table.py`:

```python
"""radial_init: the n_q_factor knob and the from_table projector."""
import math

import numpy as np
import pytest

from ace_jax.construct.radial_init import tensor_radial_init
from ace_jax.construct.spec import build_spec


def _rnl():
    _, Rnl, _ = build_spec(1, 3, 10, 1.5)
    return Rnl, max(n for n, _ in Rnl)


def test_tensor_radial_init_default_n_q_unchanged():
    Rnl, maxn = _rnl()
    d = tensor_radial_init([14], Rnl, rcut=5.5)
    assert d["rnl_Wnlq"].shape[-1] == math.ceil(1.5 * maxn) == len(d["polys_A"])


def test_tensor_radial_init_n_q_factor_widens_and_keeps_onehot():
    Rnl, maxn = _rnl()
    d1 = tensor_radial_init([14], Rnl, rcut=5.5, mode="onehot")
    d3 = tensor_radial_init([14], Rnl, rcut=5.5, mode="onehot", n_q_factor=3.0)
    n1 = d1["rnl_Wnlq"].shape[-1]
    assert d3["rnl_Wnlq"].shape[-1] == math.ceil(3.0 * maxn) == len(d3["polys_A"])
    np.testing.assert_array_equal(d3["rnl_Wnlq"][..., :n1], d1["rnl_Wnlq"])
    assert not d3["rnl_Wnlq"][..., n1:].any()
    np.testing.assert_allclose(d3["polys_A"][:n1], d1["polys_A"], rtol=0, atol=0)


def test_build_model_n_q_factor(tmp_path):
    from test_python_authoring import _primed_cache
    from ace_jax.construct.model import build_model
    _, maxn = _rnl()
    auth = build_model([14], 3, 10, coupling_cache_dir=_primed_cache(tmp_path),
                       radial_mode="onehot", n_q_factor=3.0)
    assert auth.model.rnl_Wnlq.shape[-1] == math.ceil(3.0 * maxn)
    assert auth.model.polys_A.shape == (math.ceil(3.0 * maxn),)
    assert auth.meta["authoring"]["n_q_factor"] == 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_radial_init_table.py -v`
Expected: first test PASSES (default unchanged); the other two FAIL with `TypeError: ... unexpected keyword argument 'n_q_factor'`.

- [ ] **Step 3: Implement**

In `src/ace_jax/construct/radial_init.py`, change `tensor_radial_init`:

```python
def tensor_radial_init(elements, Rnl_spec, *, rcut,
                       r0=None, rin=0.0, p=2, q=2, mode="glorot_normal", seed=0,
                       n_q_factor=1.5):
    """Coefficients for the many-body (tensor) radial basis.

    elements: atomic numbers; Rnl_spec: (n, l) list from `build_spec` (defines
    n_rnl and the onehot convention); r0: None (per-pair bond-length default),
    a scalar or an (NZ, NZ) table; n_q_factor: polynomial span
    n_q = ceil(n_q_factor * max n) (1.5 is ACEpotentials' default; learned
    radials want 2-3).  Returns dict with rnl_transform (NZ,NZ,7),
    rnl_envelope (NZ,NZ,5), rnl_Wnlq (NZ,NZ,n_rnl,n_q) and polys_A/B/C (n_q,)."""
    NZ = len(elements)
    n_rnl = len(Rnl_spec)
    actual_maxn = max(n for n, _ in Rnl_spec)
    n_q = math.ceil(actual_maxn * n_q_factor)
```

(rest of the function unchanged). In `src/ace_jax/construct/model.py`, add `n_q_factor=1.5` to the `build_model` keyword list (after `coupling_cache_dir=None`), pass it on:

```python
    tinit = ri.tensor_radial_init(zs, Rnl, rcut=rcut, r0=r0, rin=rin,
                                  mode=radial_mode, seed=seed, n_q_factor=n_q_factor)
```

and add `"n_q_factor": float(n_q_factor),` to the `meta["authoring"]` dict next to `"radial_mode"`. Add one line to the `build_model` docstring: `n_q_factor: tensor-radial polynomial span, n_q = ceil(n_q_factor * max n).`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_radial_init_table.py tests/test_python_authoring.py -v`
Expected: all PASS (existing authoring tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/construct/radial_init.py src/ace_jax/construct/model.py tests/test_radial_init_table.py
git commit -m "feat(construct): n_q_factor knob for the tensor-radial polynomial span

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `from_table` — sampled radials → `Wnlq`

**Files:**
- Modify: `src/ace_jax/construct/radial_init.py` (append after `_init_Wnlq`)
- Test: `tests/test_radial_init_table.py`

**Interfaces:**
- Produces: `from_table(x, R, envelope, polys, weights=None) -> (Wnlq (NZ,NZ,n_rnl,n_q), relres (NZ,NZ,n_rnl))`. `x` is the transformed coordinate in [-1, 1]; `R` includes the envelope; basis is `env(x)·P_q(x)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_radial_init_table.py`:

```python
from conftest import FIXTURE_DIR


def _analytic_fixture():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.eval import load
    p = FIXTURE_DIR / "si_ace_model.npz"
    if not p.exists():
        pytest.skip("missing si_ace_model.npz")
    m, meta, _ = load(p)
    return m


def _sample(m, x):
    """env(x) P(x) W^T per pair, numpy, (NZ, NZ, n_x, n_rnl)."""
    from ace_jax.construct.radial_init import envelope2sx_eval, poly_eval
    W = np.asarray(m.rnl_Wnlq)
    P = poly_eval(x, np.asarray(m.polys_A), np.asarray(m.polys_B), np.asarray(m.polys_C))
    NZ = W.shape[0]
    env = np.asarray(m.rnl_envelope)
    return np.stack([np.stack([envelope2sx_eval(None, x, env[i, j])[:, None] * P @ W[i, j].T
                               for j in range(NZ)]) for i in range(NZ)])


def test_from_table_roundtrip_recovers_Wnlq():
    from ace_jax.construct.radial_init import from_table
    m = _analytic_fixture()
    x = np.linspace(-1, 1, 801)
    R = _sample(m, x)
    polys = tuple(np.asarray(a) for a in (m.polys_A, m.polys_B, m.polys_C))
    W, rel = from_table(x, R, np.asarray(m.rnl_envelope), polys)
    Wt = np.asarray(m.rnl_Wnlq)
    assert np.abs(W - Wt).max() < 1e-10 * np.abs(Wt).max()
    assert rel.max() < 1e-10


def test_from_table_zero_radial_has_zero_residual():
    from ace_jax.construct.radial_init import from_table
    m = _analytic_fixture()
    x = np.linspace(-1, 1, 201)
    R = _sample(m, x)
    R[..., 0] = 0.0
    polys = tuple(np.asarray(a) for a in (m.polys_A, m.polys_B, m.polys_C))
    W, rel = from_table(x, R, np.asarray(m.rnl_envelope), polys)
    assert np.all(W[..., 0, :] == 0.0) and np.all(rel[..., 0] == 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_radial_init_table.py -k from_table -v`
Expected: FAIL with `ImportError: cannot import name 'from_table'`.

- [ ] **Step 3: Implement**

Append to `src/ace_jax/construct/radial_init.py`:

```python
# ---------------------------------------------------------------------------
#  sampled radials -> analytic Wnlq (spline conversion, MACE init)
# ---------------------------------------------------------------------------

def from_table(x, R, envelope, polys, weights=None):
    """Project sampled tensor radials onto the analytic basis env(x) P_q(x).

    x: (n_x,) grid in the transformed coordinate [-1, 1]; R: (NZ, NZ, n_x,
    n_rnl) radial values, envelope INCLUDED; envelope: (NZ, NZ, 5)
    PolyEnvelope2sX params; polys: (A, B, C) recursion arrays (n_q = len(A));
    weights: optional (NZ, NZ, n_x) nonnegative quadrature/density weights
    (default uniform).  Per species pair solves the weighted least squares
    min_W sum_x w (env P W^T - R)^2 -- the envelope is part of the basis, so
    nothing is divided by it.  Returns (Wnlq (NZ, NZ, n_rnl, n_q), relres
    (NZ, NZ, n_rnl)): relres is ||weighted residual|| / ||weighted R|| per
    radial, 0 for an all-zero radial."""
    x = np.asarray(x, float)
    R = np.asarray(R, float)
    envelope = np.asarray(envelope, float)
    NZ, _, n_x, n_rnl = R.shape
    P = poly_eval(x, *polys)                                   # (n_x, n_q)
    n_q = P.shape[1]
    W = np.zeros((NZ, NZ, n_rnl, n_q))
    rel = np.zeros((NZ, NZ, n_rnl))
    for i in range(NZ):
        for j in range(NZ):
            basis = envelope2sx_eval(None, x, envelope[i, j])[:, None] * P
            w = np.ones(n_x) if weights is None else np.asarray(weights[i, j], float)
            sw = np.sqrt(w)[:, None]
            Wij = np.linalg.lstsq(sw * basis, sw * R[i, j], rcond=None)[0]     # (n_q, n_rnl)
            W[i, j] = Wij.T
            res = np.linalg.norm(sw * (basis @ Wij - R[i, j]), axis=0)
            nrm = np.linalg.norm(sw * R[i, j], axis=0)
            rel[i, j] = np.where(nrm > 0, res / np.where(nrm > 0, nrm, 1.0), 0.0)
    return W, rel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_radial_init_table.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/construct/radial_init.py tests/test_radial_init_table.py
git commit -m "feat(construct): from_table projects sampled radials onto env*P_q (Wnlq)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `radial_model` — swap, widen, convert

**Files:**
- Create: `src/ace_jax/fit/radial_model.py`
- Modify: `src/ace_jax/construct/export.py` (append `patch_radial_npz`)
- Test: `tests/test_gp_radial_model.py`

**Interfaces:**
- Consumes: `from_table`, `legendre_3term` (`construct.radial_init`).
- Produces:
  - `require_analytic(model) -> None` (raises `ValueError` mentioning `to_analytic`)
  - `with_radial(model, W) -> ACEModel`
  - `widen_radial(model, n_q) -> ACEModel`
  - `to_analytic(model, n_q, n_x=2001) -> (ACEModel, relres np.ndarray (NZ,NZ,n_rnl))`
  - `poly_env(model, r, zi, zj) -> (E, n_q)` jnp
  - `construct.export.patch_radial_npz(src, dst, model) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gp_radial_model.py`:

```python
"""fit.radial_model: analytic-radial helpers (swap / widen / convert, Gram, gauge, roughness)."""
import dataclasses

import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, flat_edges, load_configs
from ace_jax.fit.inducing import GPConfig
from ace_jax.fit.rows import linear_rows

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
SI_ANALYTIC = FIXTURE_DIR / "si_ace_model.npz"
SI_SPLINE = FIXTURE_DIR / "si_fitted.npz"
SIGE_SPLINE = FIXTURE_DIR / "sige_nofit.npz"
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (XYZ, SI_ANALYTIC, SI_SPLINE, SIGE_SPLINE)), reason="missing fixtures")


def _setup(path, ncfg=6, per_batch=3):
    model, meta, z = load(path)
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:ncfg]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=per_batch)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=per_batch)
    return model, meta, ds, cfg


def _batch(ds, i=0):
    return jax.tree.map(lambda a: a[i], ds)


def _live_edges(ds):
    """(rij, zi, zj) over every live edge of every batch."""
    out = []
    for i in range(ds.n_batches):
        b = _batch(ds, i)
        rij, send, recv, mask = flat_edges(b.rij, b.nbr, b.nbr_mask)
        m = np.asarray(mask)
        out.append((np.asarray(rij)[m], np.asarray(b.node_z[send])[m], np.asarray(b.node_z[recv])[m]))
    return tuple(jnp.asarray(np.concatenate(k)) for k in zip(*out))


@pytest.fixture(scope="module")
def si():
    return _setup(SI_ANALYTIC)


def test_with_radial_identity_is_exact(si):
    from ace_jax.fit.radial_model import with_radial
    model, _, ds, cfg = si
    b = _batch(ds)
    r0, _, _ = linear_rows(model, cfg, b)
    r1, _, _ = linear_rows(with_radial(model, model.rnl_Wnlq), cfg, b)
    np.testing.assert_array_equal(np.asarray(r1.E), np.asarray(r0.E))
    with pytest.raises(ValueError, match="shape"):
        with_radial(model, model.rnl_Wnlq[..., :-1])


def test_poly_env_reproduces_model_radial(si):
    from ace_jax.fit.radial_model import poly_env
    model, _, ds, _ = si
    rij, zi, zj = _live_edges(ds)
    Rnl, _ = model.radial(rij, zi, zj)
    r = jnp.linalg.norm(rij, axis=-1)
    mine = jnp.einsum("eq,enq->en", poly_env(model, r, zi, zj), model.rnl_Wnlq[zi, zj])
    np.testing.assert_allclose(np.asarray(mine), np.asarray(Rnl), rtol=1e-12, atol=1e-13)


def test_widen_radial_preserves_descriptors(si):
    from ace_jax.fit.radial_model import widen_radial
    model, _, ds, cfg = si
    wide = widen_radial(model, 30)
    assert wide.rnl_Wnlq.shape[-1] == 30 and wide.polys_A.shape == (30,)
    b = _batch(ds)
    r0, _, _ = linear_rows(model, cfg, b)
    r1, _, _ = linear_rows(wide, cfg, b)
    np.testing.assert_allclose(np.asarray(r1.E), np.asarray(r0.E), rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match="n_q"):
        widen_radial(model, 5)
    with pytest.raises(ValueError, match="Legendre"):
        widen_radial(dataclasses.replace(model, polys_A=2.0 * model.polys_A), 30)


def test_non_analytic_model_raises():
    from ace_jax.fit.radial_model import with_radial
    model, *_ = _setup(SI_SPLINE)
    with pytest.raises(ValueError, match="to_analytic"):
        with_radial(model, model.rnl_Wnlq)


def test_to_analytic_matches_spline_radials(si):
    from ace_jax.fit.radial_model import to_analytic
    _, _, ds, _ = si
    spline, *_ = _setup(SI_SPLINE)
    ana, rel = to_analytic(spline, 30)
    assert ana.radial_kind == "analytic" and ana.rnl_Wnlq.shape[-1] == 30
    rij, zi, zj = _live_edges(ds)
    Rs, Ps = spline.radial(rij, zi, zj)
    Ra, Pa = ana.radial(rij, zi, zj)
    err = float(jnp.abs(Ra - Rs).max() / jnp.abs(Rs).max())
    print(f"to_analytic max rel radial error {err:.3e}, max relres {rel.max():.3e}")
    assert err < 1e-6
    np.testing.assert_array_equal(np.asarray(Pa), np.asarray(Ps))   # pair basis untouched


def test_patch_radial_npz_roundtrip(tmp_path, si):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.fit.radial_model import to_analytic
    spline, *_ = _setup(SI_SPLINE)
    ana, _ = to_analytic(spline, 30)
    patch_radial_npz(SI_SPLINE, tmp_path / "m.npz", ana)
    back, meta, _ = load(tmp_path / "m.npz")
    assert back.radial_kind == "analytic" and meta["radial_kind"] == "analytic"
    np.testing.assert_array_equal(np.asarray(back.rnl_Wnlq), np.asarray(ana.rnl_Wnlq))
    _, _, ds, _ = si
    rij, zi, zj = _live_edges(ds)
    np.testing.assert_allclose(np.asarray(back.radial(rij, zi, zj)[0]),
                               np.asarray(ana.radial(rij, zi, zj)[0]), rtol=0, atol=1e-14)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gp_radial_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.radial_model'` (and `ImportError` for `patch_radial_npz`).

- [ ] **Step 3: Implement `radial_model.py` (part 1)**

Create `src/ace_jax/fit/radial_model.py`:

```python
"""Model-level helpers for learnable tensor-basis radials (analytic branch).

The analytic tensor radial is

    R_n(r; zi, zj) = env(x) * sum_q Wnlq[zi, zj, n, q] P_q(x),   x = T_{zi,zj}(r),

so for a fixed transform, envelope and polynomial set it is LINEAR in Wnlq.
These helpers swap, widen and convert Wnlq, and build the fixed quadratic forms
the learner needs: the empirical radial Gram Q (gauge normalisation) and the
roughness matrix D2.  See docs/dev/specs/2026-09-26-learned-radial-varpro-design.md.
"""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..construct.radial_init import from_table, legendre_3term
from ..eval.radial import agnesi_normalized, env_poly2sx, poly_recursion, spline_eval
from .data import flat_edges


def require_analytic(model):
    if model.radial_kind != "analytic":
        raise ValueError(
            f"learned radials need the analytic radial branch, got radial_kind="
            f"{model.radial_kind!r}; convert with fit.radial_model.to_analytic(model, n_q)")


def with_radial(model, W):
    """`model` with its tensor-radial weights replaced by W (same shape)."""
    require_analytic(model)
    W = jnp.asarray(W)
    if W.shape != model.rnl_Wnlq.shape:
        raise ValueError(f"W shape {W.shape} != rnl_Wnlq shape {model.rnl_Wnlq.shape}")
    return eqx.tree_at(lambda m: m.rnl_Wnlq, model, W)


def _legendre(n_q):
    return tuple(jnp.asarray(a) for a in legendre_3term(n_q))


def widen_radial(model, n_q):
    """Extend the polynomial span to n_q normalized-Legendre polynomials by
    zero-padding Wnlq: every radial (hence every descriptor) is unchanged, but
    the learner can now reach degrees up to n_q - 1."""
    require_analytic(model)
    old = model.rnl_Wnlq.shape[-1]
    if n_q < old:
        raise ValueError(f"widen_radial: n_q={n_q} < current n_q={old}")
    A, B, C = _legendre(n_q)
    for name, ref in (("polys_A", A), ("polys_B", B), ("polys_C", C)):
        if not np.allclose(np.asarray(getattr(model, name)), np.asarray(ref[:old]),
                           rtol=1e-12, atol=1e-14):
            raise ValueError(f"widen_radial assumes normalized-Legendre polys; {name} differs")
    W = jnp.pad(model.rnl_Wnlq, ((0, 0), (0, 0), (0, 0), (0, n_q - old)))
    return dataclasses.replace(model, rnl_Wnlq=W, polys_A=A, polys_B=B, polys_C=C)


def poly_env(model, r, zi, zj):
    """(E, n_q): env(x) P_q(x) per edge; the tensor radials are
    einsum('eq,enq->en', poly_env, Wnlq[zi, zj])."""
    x = agnesi_normalized(r, model.rnl_transform[zi, zj])
    env = env_poly2sx(x, model.rnl_envelope[zi, zj])
    return env[:, None] * poly_recursion(x, model.polys_A, model.polys_B, model.polys_C)


def to_analytic(model, n_q, n_x=2001):
    """Convert a splined tensor radial to the analytic branch.  Each pair's
    spline S(x) (the envelope-free part) is sampled on a uniform x-grid, env*S
    is projected onto env*P_q(x) (radial_init.from_table), and the branch is
    swapped.  An analytic model is only widened.  Returns (model, relres) with
    relres (NZ, NZ, n_rnl) the per-radial relative projection residual."""
    if model.radial_kind == "analytic":
        wide = widen_radial(model, max(n_q, model.rnl_Wnlq.shape[-1]))
        return wide, np.zeros(model.rnl_Wnlq.shape[:3])
    if model.radial_kind != "spline":
        raise ValueError(f"to_analytic: unsupported radial_kind {model.radial_kind!r}")
    x = np.linspace(-1.0, 1.0, n_x)
    x0, h, n = model.rnl_grid
    NZ = model.rnl_coefs.shape[0]
    envp = np.asarray(model.rnl_envelope)
    env = np.asarray(env_poly2sx(jnp.asarray(x)[None, None, :],
                                 jnp.asarray(envp)[:, :, None, :]))             # (NZ, NZ, n_x)
    S = np.stack([np.stack([np.asarray(spline_eval(jnp.asarray(x), model.rnl_coefs[i, j], x0, h, n))
                            for j in range(NZ)]) for i in range(NZ)])         # (NZ, NZ, n_x, n_rnl)
    polys = legendre_3term(n_q)
    W, rel = from_table(x, env[..., None] * S, envp, polys)
    A, B, C = (jnp.asarray(a) for a in polys)
    out = dataclasses.replace(model, radial_kind="analytic", rnl_Wnlq=jnp.asarray(W),
                              polys_A=A, polys_B=B, polys_C=C)
    return out, rel
```

- [ ] **Step 4: Implement `patch_radial_npz`**

Append to `src/ace_jax/construct/export.py`:

```python
def patch_radial_npz(src, dst, model):
    """Copy the npz at `src` to `dst` with the tensor radial replaced by
    `model`'s analytic one (rnl_Wnlq + polys_A/B/C; any rnl spline arrays are
    dropped and meta radial_kind/rnl_spline updated).  Everything else --
    coupling, pair basis, readout, E0 -- is copied verbatim.  This is how a
    learned radial is written back from a model that was loaded rather than
    authored (save_npz needs an Authoring)."""
    import json
    if model.radial_kind != "analytic":
        raise ValueError(f"patch_radial_npz: model radial_kind {model.radial_kind!r} is not analytic")
    with np.load(src, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    meta = json.loads(bytes(out["meta_json"]).decode())
    for k in ("rnl_spline_coefs", "rnl_spline_coefs_single", "rnl_embedding",
              "rnl_emb_nidx", "rnl_emb_kidx"):
        out.pop(k, None)
    out["rnl_Wnlq"] = np.asarray(model.rnl_Wnlq, np.float64)
    out["polys_A"] = np.asarray(model.polys_A, np.float64)
    out["polys_B"] = np.asarray(model.polys_B, np.float64)
    out["polys_C"] = np.asarray(model.polys_C, np.float64)
    meta["radial_kind"] = "analytic"
    meta["rnl_spline"] = None
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(dst, **out)
```

(`meta_json` is written exactly as `save_npz` writes it — `np.frombuffer(json.dumps(meta).encode(), np.uint8)` — which `eval.io.load` decodes with `bytes(z["meta_json"]).decode()`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp_radial_model.py -v -s`
Expected: all PASS. `test_to_analytic_matches_spline_radials` prints the measured error. If `err` is above 1e-6 because the ACE1 fixture's spline is not a polynomial of degree < 30 in x, do not loosen blindly: print `rel.max(axis=-1)` per radial, and if the residual is uniform and ≤ 1e-4 set the tolerance to 10× the measured `err` with a comment giving the measured value; otherwise stop and report.

- [ ] **Step 6: Commit**

```bash
git add src/ace_jax/fit/radial_model.py src/ace_jax/construct/export.py tests/test_gp_radial_model.py
git commit -m "feat(radial_model): with_radial/widen_radial/to_analytic + patch_radial_npz

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `radial_model` — Gram, gauge, roughness

**Files:**
- Modify: `src/ace_jax/fit/radial_model.py` (append)
- Test: `tests/test_gp_radial_model.py` (append)

**Interfaces:**
- Consumes: `poly_env`, `to_analytic`, `require_analytic` (Task 3).
- Produces:
  - `radial_gram(model, ds, n_prior=10.0, n_uniform=401) -> Q (NZ,NZ,n_q,n_q)` jnp
  - `row_active(W) -> (NZ,NZ,n_rnl) bool` jnp
  - `normalise(V, Q, active) -> W` jnp (differentiable)
  - `roughness_matrix(model) -> D2 (n_q,n_q)` jnp
  - `roughness(W, D2, wn) -> scalar` jnp
  - `rnl_degrees(meta) -> (n_rnl,) int np.ndarray`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gp_radial_model.py`:

```python
def test_radial_gram_gives_unit_empirical_norm(si):
    from ace_jax.fit.radial_model import normalise, poly_env, radial_gram, row_active
    model, _, ds, _ = si
    Q = radial_gram(model, ds, n_prior=0.0)
    W = normalise(model.rnl_Wnlq, Q, row_active(model.rnl_Wnlq))
    rij, zi, zj = _live_edges(ds)
    R = jnp.einsum("eq,enq->en", poly_env(model, jnp.linalg.norm(rij, axis=-1), zi, zj), W[zi, zj])
    np.testing.assert_allclose(np.asarray(jnp.mean(R ** 2, axis=0)), 1.0, rtol=1e-10)


def test_normalise_is_scale_invariant(si):
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    model, _, ds, _ = si
    Q = radial_gram(model, ds)
    V = model.rnl_Wnlq
    act = row_active(V)
    np.testing.assert_allclose(np.asarray(normalise(3.7 * V, Q, act)),
                               np.asarray(normalise(V, Q, act)), rtol=1e-12, atol=1e-14)


def test_normalise_zero_rows_stay_zero_and_finite(si):
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    model, _, ds, _ = si
    Q = radial_gram(model, ds)
    V0 = model.rnl_Wnlq.at[0, 0, 3].set(0.0)            # a structurally-zero radial
    act = row_active(V0)
    assert not bool(act[0, 0, 3]) and bool(act[0, 0, 2])
    W = normalise(V0, Q, act)
    assert bool(jnp.all(W[0, 0, 3] == 0.0)) and bool(jnp.all(jnp.isfinite(W)))
    g = jax.grad(lambda V: jnp.sum(normalise(V, Q, act) ** 2))(V0)
    assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all(g[0, 0, 3] == 0.0))


def test_radial_gram_absent_pair_is_positive_definite():
    from ace_jax.fit.radial_model import radial_gram, to_analytic
    spline, _, ds, _ = _setup(SIGE_SPLINE)              # elements Si, Ge; data is all-Si
    model, _ = to_analytic(spline, 12)
    Q = radial_gram(model, ds)
    ev = np.linalg.eigvalsh(np.asarray(Q))              # (NZ, NZ, n_q)
    assert ev.min() > 0.0


def test_roughness_matrix_properties(si):
    from ace_jax.fit.radial_model import roughness_matrix
    model, *_ = si
    D2 = np.asarray(roughness_matrix(model))
    np.testing.assert_allclose(D2, D2.T, atol=1e-10)
    assert np.abs(D2[:2]).max() < 1e-9                  # degrees 0, 1 have zero curvature
    assert np.linalg.eigvalsh(D2).min() > -1e-8 * np.abs(D2).max()
    # finite-difference check of one entry: int P_5'' P_7'' dx
    from ace_jax.construct.radial_init import poly_eval
    x = np.linspace(-1, 1, 20001)
    P = poly_eval(x, *(np.asarray(a) for a in (model.polys_A, model.polys_B, model.polys_C)))
    d2 = np.gradient(np.gradient(P, x, axis=0), x, axis=0)
    ref = np.trapezoid(d2[:, 5] * d2[:, 7], x)
    assert abs(D2[5, 7] - ref) < 1e-3 * abs(ref)


def test_roughness_is_weighted_quadratic_form(si):
    from ace_jax.fit.radial_model import roughness, roughness_matrix
    model, *_ = si
    W, D2 = model.rnl_Wnlq, roughness_matrix(model)
    wn = jnp.linspace(1.0, 0.1, W.shape[2])
    ref = sum(float(wn[n]) * float(W[0, 0, n] @ D2 @ W[0, 0, n]) for n in range(W.shape[2]))
    assert abs(float(roughness(W, D2, wn)) - ref) < 1e-10 * abs(ref)


def test_rnl_degrees(si):
    from ace_jax.fit.radial_model import rnl_degrees
    _, meta, _, _ = si
    d = rnl_degrees(meta)
    assert d.shape == (meta["n_rnl"],) and d.min() == 0
    with pytest.raises(ValueError, match="n_rnl"):
        rnl_degrees({**meta, "n_rnl": meta["n_rnl"] + 1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gp_radial_model.py -v`
Expected: the new tests FAIL with `ImportError` (`radial_gram`, `normalise`, …).

- [ ] **Step 3: Implement**

Append to `src/ace_jax/fit/radial_model.py`:

```python
def _uniform_moment(model, n_uniform):
    """(NZ, NZ, n_q, n_q): mean over a uniform x-grid of p p^T, p = env(x) P(x)."""
    x = jnp.linspace(-1.0, 1.0, n_uniform)
    P = poly_recursion(x, model.polys_A, model.polys_B, model.polys_C)          # (n_x, n_q)
    env = env_poly2sx(x[None, None, :], model.rnl_envelope[:, :, None, :])     # (NZ, NZ, n_x)
    p = env[..., None] * P                                                      # (NZ, NZ, n_x, n_q)
    return jnp.einsum("abxq,abxp->abqp", p, p) / n_uniform


def radial_gram(model, ds, n_prior=10.0, n_uniform=401):
    """Q (NZ, NZ, n_q, n_q): per (centre, neighbour) species pair, the second
    moment E_e[p(r_e) p(r_e)^T] of the enveloped polynomials p = env * P over
    the dataset's live edges, so ||R_n||^2 under the empirical pair-distance
    density is W_n^T Q W_n.  Each pair is shrunk towards the uniform-in-x
    moment with `n_prior` pseudo-edges, so a pair absent from the data (or
    with very few edges) still has a positive-definite Q.  Depends on the
    transform/envelope/polys only, never on Wnlq."""
    require_analytic(model)
    NZ, n_q = model.rnl_Wnlq.shape[0], model.rnl_Wnlq.shape[-1]
    S = jnp.zeros((NZ * NZ, n_q, n_q))
    cnt = jnp.zeros(NZ * NZ)
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        rij, send, recv, mask = flat_edges(b.rij, b.nbr, b.nbr_mask)
        zi, zj = b.node_z[send], b.node_z[recv]
        r = jnp.where(mask, jnp.linalg.norm(rij, axis=-1), 1.0)      # padded slots: any finite r
        p = poly_env(model, r, zi, zj) * mask[:, None]
        pair = zi * NZ + zj
        S = S + jax.ops.segment_sum(p[:, :, None] * p[:, None, :], pair, num_segments=NZ * NZ)
        cnt = cnt + jax.ops.segment_sum(mask.astype(S.dtype), pair, num_segments=NZ * NZ)
    S = S.reshape(NZ, NZ, n_q, n_q)
    cnt = cnt.reshape(NZ, NZ)[..., None, None]
    return (S + n_prior * _uniform_moment(model, n_uniform)) / (cnt + n_prior)


def row_active(W):
    """(NZ, NZ, n_rnl) bool: radials that are not identically zero.  The onehot
    (identity) init has structurally-zero rows for NZ > 1; they stay frozen."""
    return jnp.any(jnp.asarray(W) != 0.0, axis=-1)


def normalise(V, Q, active):
    """W = V with each active radial (zi, zj, n) scaled to unit norm under Q
    (||R||^2 = V Q V^T); inactive rows are returned as exact zeros.  Invariant
    to a positive rescaling of any row -- the gauge the readout and the prior
    would otherwise trade against.  The `where` keeps 0/0 out of the gradient."""
    nrm2 = jnp.einsum("abnq,abqp,abnp->abn", V, Q, V)
    safe = jnp.where(active, nrm2, 1.0)
    return jnp.where(active[..., None], V * jax.lax.rsqrt(safe)[..., None], 0.0)


def roughness_matrix(model):
    """D2 (n_q, n_q) = int_{-1}^{1} P_q''(x) P_p''(x) dx by Gauss-Legendre with
    n_q + 2 nodes or more (exact for these polynomial degrees)."""
    n_q = model.polys_A.shape[0]
    xg, wg = np.polynomial.legendre.leggauss(max(64, n_q + 2))
    p = lambda x: poly_recursion(x, model.polys_A, model.polys_B, model.polys_C)
    d2 = jax.vmap(jax.jacfwd(jax.jacfwd(p)))(jnp.asarray(xg))     # (n_nodes, n_q)
    return (d2 * jnp.asarray(wg)[:, None]).T @ d2


def roughness(W, D2, wn):
    """sum_{zi, zj, n} wn[n] * W[zi, zj, n] D2 W[zi, zj, n]^T -- the curvature
    of the polynomial part of each radial, weighted per radial by wn."""
    return jnp.einsum("n,abnq,qp,abnp->", wn, W, D2, W)


def rnl_degrees(meta, wL=1.5):
    """(n_rnl,) polynomial degree of each tensor radial under the identity
    (onehot) convention, n' = (n - 1) // NZ, from the model's (n, l) spec.
    Rebuilds the spec with build_spec; raises if its length disagrees with
    meta["n_rnl"] (e.g. a Julia export with a different wL)."""
    from ..construct.spec import build_spec
    NZ = len(meta["elements"])
    wL = meta.get("authoring", {}).get("wL", wL)
    _, Rnl, _ = build_spec(NZ, meta["order"], meta["totaldegree"], wL)
    if len(Rnl) != meta["n_rnl"]:
        raise ValueError(f"rnl_degrees: rebuilt spec has {len(Rnl)} radials, meta n_rnl={meta['n_rnl']}")
    return np.array([(n - 1) // NZ for n, _ in Rnl], dtype=int)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp_radial_model.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/radial_model.py tests/test_gp_radial_model.py
git commit -m "feat(radial_model): empirical radial Gram, gauge normalise, roughness

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: VarPro objective from streamed statistics

**Files:**
- Create: `src/ace_jax/fit/radial_learn.py`
- Test: `tests/test_gp_learn_radial.py`

**Interfaces:**
- Consumes: `with_radial` (Task 3); `linear_statistics` (`fit.stats`); `combine` (`fit.objective`).
- Produces:
  - `projected_residual_from_stats(theta, lin, gamma) -> scalar` (theta a `Hypers`)
  - `projected_residual(W, theta, prob, ds) -> scalar`
  - test helpers in `tests/test_gp_learn_radial.py`: `THETA`, `make_problem(ncfg, per_batch, start)`, `relabel(prob, ds, W, c)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gp_learn_radial.py`:

```python
"""fit.radial_learn: VarPro objective, L-BFGS loop, held-out gate."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_ace_model.npz"
pytestmark = pytest.mark.skipif(not (XYZ.exists() and MODEL.exists()), reason="missing fixtures")

THETA = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
               log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
               log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))


def make_problem(ncfg=6, per_batch=3, start=0, force_key="dft_force"):
    """Linear (M = 0) problem on si_tiny with the analytic Si model; gamma = 1."""
    model, meta, z = load(MODEL)
    configs = load_configs(XYZ, "dft_energy", force_key, "dft_virial")[start:start + ncfg]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=per_batch)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=per_batch)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.ones(cfg.len_basis), default_prior(2.35))
    return prob, ds, meta


def relabel(prob, ds, W, c):
    """ds with targets replaced by the noiseless linear ACE with radials W and readout c."""
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.rows import linear_rows
    m = with_radial(prob.model, W)
    yE, yF, yV = [], [], []
    for i in range(ds.n_batches):
        r, _, _ = linear_rows(m, prob.cfg, jax.tree.map(lambda a: a[i], ds))
        yE.append(r.E @ c); yF.append(r.F @ c); yV.append(r.V @ c)
    return ds._replace(y_E=jnp.stack(yE), y_F=jnp.stack(yF), y_V=jnp.stack(yV))


@pytest.fixture(scope="module")
def small():
    return make_problem()


def _in_memory_residual(W, prob, ds, theta):
    """min_c ||Phi~ c - y~||^2 on the materialised prior-augmented design."""
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.solve import stacked_design
    Phi, y = stacked_design(prob._replace(model=with_radial(prob.model, W)), ds, theta)
    c = jnp.linalg.lstsq(Phi, y, rcond=None)[0]
    r = Phi @ c - y
    return r @ r


def test_projected_residual_matches_in_memory_lstsq(small):
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds, _ = small
    W = prob.model.rnl_Wnlq
    got = float(projected_residual(W, THETA, prob, ds))
    ref = float(_in_memory_residual(W, prob, ds, THETA))
    assert abs(got - ref) < 1e-7 * abs(ref)


def test_projected_residual_gradient(small):
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds, _ = small
    W = prob.model.rnl_Wnlq
    f = lambda X: projected_residual(X, THETA, prob, ds)
    g = jax.grad(f)(W)
    D = jnp.asarray(np.random.default_rng(0).standard_normal(W.shape))
    h = 1e-5
    fd = (float(f(W + h * D)) - float(f(W - h * D))) / (2 * h)
    assert abs(float(jnp.vdot(g, D)) - fd) < 1e-5 * abs(fd)
    g_mem = jax.grad(lambda X: _in_memory_residual(X, prob, ds, THETA))(W)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_mem), rtol=1e-6,
                               atol=1e-8 * float(jnp.abs(g_mem).max()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gp_learn_radial.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.radial_learn'`.

- [ ] **Step 3: Implement**

Create `src/ace_jax/fit/radial_learn.py`:

```python
"""Learned tensor-basis radials by variable projection (VarPro).

The optimiser moves the analytic-branch mixing weights Wnlq; the linear readout
is projected out exactly.  At fixed hyperparameters theta the projected ridge
residual

    r(W) = min_c ||Phi(W) c - y||^2_w + c^T Lambda c = yy - b^T (G + Lambda)^{-1} b,
    Lambda = diag(gamma^2 / sigma_c^2),

is closed form in the streamed linear statistics (G, b, yy) of the model with
radials W, so no design matrix is ever materialised.  Its W-gradient is the
exact Golub-Pereyra/Kaufman gradient (envelope theorem), taken by autodiff
through the checkpointed `linear_statistics` scan.  M = 0 throughout: the
residual GP is fitted afterwards on the frozen learned model.
See docs/dev/specs/2026-09-26-learned-radial-varpro-design.md.
"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .objective import combine
from .radial_model import with_radial
from .stats import linear_statistics


def projected_residual_from_stats(theta, lin, gamma):
    """yy - b^T (G + Lambda)^{-1} b from linear statistics `lin` (M = 0).
    A failed Cholesky yields NaN; the learner treats that as a stop signal."""
    G, b, yy, _, _ = combine(theta, lin)
    lam = gamma ** 2 * jnp.exp(-2.0 * theta.log_sigma_c)
    L = jnp.linalg.cholesky(G + jnp.diag(lam))
    v = solve_triangular(L, b, lower=True)
    return yy - v @ v


def projected_residual(W, theta, prob, ds):
    """VarPro objective of the linear ACE with tensor radials W (one full
    streaming pass over ds)."""
    lin = linear_statistics(with_radial(prob.model, W), prob.cfg, ds)
    return projected_residual_from_stats(theta, lin, prob.gamma)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp_learn_radial.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/radial_learn.py tests/test_gp_learn_radial.py
git commit -m "feat(radial_learn): VarPro projected residual from streamed linear statistics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Gradient memory gate

This checks the spec's key assumption: reverse mode through the checkpointed `linear_statistics` scan must not store per-batch `L × L` carries. It is checked on compiled memory stats, so it runs on CPU in seconds, and it **gates Tasks 7–9**.

**Files:**
- Test: `tests/test_gp_learn_radial.py` (append)
- Modify (only if the test fails): `src/ace_jax/fit/radial_learn.py`

**Interfaces:**
- Produces (only in the fallback case): `projected_residual_2pass(theta, prob, ds) -> f(W)` with a `jax.custom_vjp`; Task 7 then uses it in place of `projected_residual` inside `_objective`.

- [ ] **Step 1: Write the test**

Append to `tests/test_gp_learn_radial.py`:

```python
def _temp_bytes(fn, *args):
    ma = jax.jit(fn).lower(*args).compile().memory_analysis()
    if ma is None:
        pytest.skip("backend exposes no memory_analysis")
    return ma.temp_size_in_bytes


def test_gradient_memory_does_not_scale_with_batches():
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds6, _ = make_problem(ncfg=12, per_batch=2)          # 6 equal-shape batches
    ds2 = jax.tree.map(lambda a: a[:2], ds6)
    W = prob.model.rnl_Wnlq
    grad = lambda X, d: jax.grad(lambda Y: projected_residual(Y, THETA, prob, d))(X)
    t2, t6 = _temp_bytes(grad, W, ds2), _temp_bytes(grad, W, ds6)
    val6 = _temp_bytes(lambda X, d: projected_residual(X, THETA, prob, d), W, ds6)
    print(f"temp bytes: grad(2 batches)={t2}  grad(6 batches)={t6}  value(6)={val6}")
    assert t6 <= 1.25 * t2 + 1_000_000
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_gp_learn_radial.py::test_gradient_memory_does_not_scale_with_batches -v -s`
Expected: PASS (temp memory roughly independent of the batch count). Record the three printed numbers in the commit message.

- [ ] **Step 3 (only if Step 2 FAILS): two-pass streamed adjoint**

Append to `src/ace_jax/fit/radial_learn.py`:

```python
def projected_residual_2pass(theta, prob, ds):
    """f(W) = projected_residual with a hand-written streamed adjoint: pass 1
    streams (G, b, yy); the cotangent of the stats is closed form; pass 2
    scans vjp(batch_linear_stats) with it.  Memory: one batch + O(L^2)."""
    from .stats import batch_linear_stats

    @jax.custom_vjp
    def f(W):
        return projected_residual(W, theta, prob, ds)

    def fwd(W):
        lin = linear_statistics(with_radial(prob.model, W), prob.cfg, ds)
        val, st_bar = jax.value_and_grad(
            lambda st: projected_residual_from_stats(theta, st, prob.gamma), allow_int=True)(lin)
        return val, (W, st_bar)

    def bwd(res, gbar):
        W, st_bar = res

        def body(acc, batch):
            _, vjp = jax.vjp(lambda Wb: batch_linear_stats(with_radial(prob.model, Wb), prob.cfg, batch), W)
            return acc + vjp(st_bar)[0], None
        g, _ = jax.lax.scan(body, jnp.zeros_like(W), ds)
        return (gbar * g,)

    f.defvjp(fwd, bwd)
    return f
```

Change the test to measure `jax.grad(projected_residual_2pass(THETA, prob, d))`, add a check that its gradient equals `jax.grad` of `projected_residual` on `ds2` to `rtol=1e-10`, and re-run. In Task 7, `_objective` then calls `projected_residual_2pass(theta, prob_like, ds)(W)` in place of `projected_residual(W, …)`. If the fallback also fails the memory assertion, stop and report; do not continue to Task 7.

- [ ] **Step 4: Manual realistic-size check on moriarty**

After the local test passes, run the same measurement at production size (the Cantor model and data used by `bench/acegp_cantor/run.py`, `--batch 4`). This is recorded, not asserted. Connect with plain `ssh moriarty` (never `-o BatchMode=yes`); the repo is shared under `/home/eng/essswb`. Run `uv run python - <<'EOF'` with a script that builds the problem as in `bench/learn_radial/run.py` (Task 9) and prints `_temp_bytes` for value and grad. Paste the numbers into the Task 9 results note. If this step can't be run yet, add a line to the commit message saying so.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gp_learn_radial.py src/ace_jax/fit/radial_learn.py
git commit -m "test(radial_learn): gradient temp memory independent of batch count

temp bytes: grad(2)=<t2> grad(6)=<t6> value(6)=<val6>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Replace `<t2>`, `<t6>`, `<val6>` with the printed values.)

---

### Task 7: L-BFGS loop, θ profiling, `learn_radial`

**Files:**
- Modify: `src/ace_jax/fit/radial_learn.py`
- Test: `tests/test_gp_learn_radial.py` (append)

**Interfaces:**
- Consumes: `projected_residual_from_stats`, `projected_residual` (Task 5, or the Task 6 fallback); `radial_gram`, `row_active`, `normalise`, `roughness_matrix`, `roughness`, `require_analytic`, `with_radial` (Tasks 3–4); `run_map` (`fit.ladder`); `log_marginal_likelihood` (`fit.objective`); `to_array`, `from_array` (`fit.hypers`).
- Produces:
  - `require_x64() -> None`
  - `lbfgs_loop(f, x0, *, steps, tol=1e-6, patience=3, memory_size=10) -> (x_best, f_best, trace list[float], reason str)`, where reason ∈ {`"steps"`, `"converged"`, `"linesearch"`, `"nonfinite"`}
  - `theta_map_linear(prob, ds, W, *, steps=300, seed=0, init=None) -> a (theta array)`
  - `learn_radial(prob, ds, W0, *, theta0=None, profile=True, lam_rough=0.0, rough_weights=None, steps=200, reprofile_every=10, tol=1e-6, patience=3, map_steps=300, n_prior=10.0, seed=0) -> (W (normalised), info dict)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gp_learn_radial.py`:

```python
def test_lbfgs_loop_quadratic():
    from ace_jax.fit.radial_learn import lbfgs_loop
    t = jnp.arange(5.0)
    s = jnp.array([1.0, 10.0, 100.0, 0.1, 3.0])
    f = jax.jit(lambda x: jnp.sum(s * (x - t) ** 2))
    x, fx, trace, reason = lbfgs_loop(f, jnp.zeros(5), steps=100)
    np.testing.assert_allclose(np.asarray(x), np.asarray(t), atol=1e-6)
    assert reason in ("converged", "linesearch", "steps") and fx <= trace[0]


def test_lbfgs_loop_zero_steps_returns_start():
    from ace_jax.fit.radial_learn import lbfgs_loop
    x0 = jnp.ones(3)
    x, fx, trace, reason = lbfgs_loop(lambda x: jnp.sum(x ** 2), x0, steps=0)
    assert bool(jnp.all(x == x0)) and trace == [] and reason == "steps" and fx == 3.0


def test_lbfgs_loop_nonfinite_keeps_best():
    from ace_jax.fit.radial_learn import lbfgs_loop
    f = jax.jit(lambda x: jnp.where(x[0] > 0.5, jnp.sum(x ** 2), jnp.nan))
    x0 = jnp.array([2.0, 1.0])
    x, fx, _, _ = lbfgs_loop(f, x0, steps=50)
    assert bool(jnp.all(jnp.isfinite(x))) and np.isfinite(fx)
    assert fx <= float(f(x0)) and float(f(x)) == fx


def test_learn_radial_requires_x64(small, monkeypatch):
    import types
    from ace_jax.fit import radial_learn
    prob, ds, _ = small
    # require_x64 is learn_radial's first statement, so only its read of
    # jax.config sees the stub; the global flag is never touched
    monkeypatch.setattr(radial_learn, "jax", types.SimpleNamespace(
        config=types.SimpleNamespace(jax_enable_x64=False)))
    with pytest.raises(RuntimeError, match="float64"):
        radial_learn.learn_radial(prob, ds, prob.model.rnl_Wnlq, theta0=THETA, steps=1)


def test_learn_radial_zero_steps_is_normalised_init(small):
    from ace_jax.fit.radial_learn import learn_radial
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    prob, ds, _ = small
    W0 = prob.model.rnl_Wnlq
    W, info = learn_radial(prob, ds, W0, theta0=THETA, profile=False, steps=0)
    ref = normalise(W0, radial_gram(prob.model, ds), row_active(W0))
    np.testing.assert_array_equal(np.asarray(W), np.asarray(ref))
    assert info["steps"] == 0


def _perturbed_truth(prob, seed=0, eps=0.2):
    rng = np.random.default_rng(seed)
    Wt = prob.model.rnl_Wnlq
    W0 = Wt + eps * jnp.abs(Wt).mean() * jnp.asarray(rng.standard_normal(Wt.shape))
    c = jnp.asarray(0.1 * rng.standard_normal(prob.cfg.len_basis))
    return Wt, W0, c


def test_learn_radial_recovers_perturbed_radials():
    from ace_jax.fit.radial_learn import learn_radial, projected_residual
    prob, ds, _ = make_problem(ncfg=12, per_batch=3)
    Wt, W0, c = _perturbed_truth(prob)
    ds = relabel(prob, ds, Wt, c)
    W, info = learn_radial(prob, ds, W0, theta0=THETA, profile=False, steps=30)
    f0 = float(projected_residual(W0, THETA, prob, ds))
    f1 = float(projected_residual(W, THETA, prob, ds))
    print(f"recovery: f0={f0:.4e} f1={f1:.4e} reasons={info['reasons']}")
    assert f1 < 0.3 * f0
    assert all(b <= a * (1 + 1e-12) for a, b in zip(info["trace"], info["trace"][1:]))


def test_learn_radial_profiles_theta(small):
    from ace_jax.fit.radial_learn import learn_radial
    prob, ds, _ = small
    W, info = learn_radial(prob, ds, prob.model.rnl_Wnlq, steps=4, reprofile_every=2, map_steps=50)
    assert len(info["theta"]) >= 2 and all(np.all(np.isfinite(t)) for t in info["theta"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gp_learn_radial.py -v -k "lbfgs or learn_radial"`
Expected: FAIL with `ImportError` (`lbfgs_loop`, `learn_radial`).

- [ ] **Step 3: Implement**

Add these imports at the top of `src/ace_jax/fit/radial_learn.py` (merge with the existing ones):

```python
from functools import partial

import numpy as np
import optax

from .hypers import from_array, to_array
from .ladder import run_map
from .objective import log_marginal_likelihood
from .radial_model import (normalise, radial_gram, require_analytic, roughness,
                           roughness_matrix, row_active)
```

Append:

```python
def require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError("learned radials need float64: jax.config.update('jax_enable_x64', True)")


def lbfgs_loop(f, x0, *, steps, tol=1e-6, patience=3, memory_size=10):
    """Minimise f (x -> scalar, jittable) by optax L-BFGS with zoom line search.

    Stops after `steps` iterations ("steps"); when the relative decrease stays
    below `tol` for `patience` consecutive iterations ("converged"); when an
    iteration fails to decrease f ("linesearch"); or when f, its gradient or
    the iterate go non-finite ("nonfinite").  Always returns the best finite
    iterate seen: (x_best, f_best, trace, reason), trace = f after each
    accepted step (strictly decreasing)."""
    x_best, f_best = x0, float(f(x0))
    if int(steps) <= 0:
        return x_best, f_best, [], "steps"
    if not np.isfinite(f_best):
        return x_best, f_best, [], "nonfinite"
    opt = optax.lbfgs(memory_size=memory_size)
    vg = optax.value_and_grad_from_state(f)
    x, state = x0, opt.init(x0)
    trace, prev, small = [], f_best, 0
    for _ in range(int(steps)):
        value, grad = vg(x, state=state)
        if not (np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))):
            return x_best, f_best, trace, "nonfinite"
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=f)
        x = optax.apply_updates(x, updates)
        fx = float(optax.tree_utils.tree_get(state, "value"))
        if not (np.isfinite(fx) and bool(jnp.all(jnp.isfinite(x)))):
            return x_best, f_best, trace, "nonfinite"
        if fx >= prev:                       # rejected step: not recorded, best kept
            return x_best, f_best, trace, "linesearch"
        trace.append(fx)
        x_best, f_best = x, fx
        small = small + 1 if (prev - fx) <= tol * max(abs(prev), 1e-300) else 0
        prev = fx
        if small >= patience:
            return x_best, f_best, trace, "converged"
    return x_best, f_best, trace, "steps"


def theta_map_linear(prob, ds, W, *, steps=300, seed=0, init=None):
    """theta-MAP of the M = 0 LML for the model with radials W (one streaming
    pass for the statistics, then run_map on the cached Gram).  init: optional
    theta array to warm-start from.  Returns the theta array."""
    lin = linear_statistics(with_radial(prob.model, W), prob.cfg, ds)
    lml = jax.jit(lambda a: log_marginal_likelihood(from_array(a), lin, prob))
    h = run_map(lml, prob.prior, steps=steps, seed=seed,
                init=None if init is None else from_array(jnp.asarray(init)))
    return to_array(h)


@partial(jax.jit, static_argnames=("cfg",))
def _objective(V, a, model, ds, gamma, Q, active, D2, wn, lam, cfg):
    theta = from_array(a)
    W = normalise(V, Q, active)
    lin = linear_statistics(with_radial(model, W), cfg, ds)
    return projected_residual_from_stats(theta, lin, gamma) + lam * roughness(W, D2, wn)


def learn_radial(prob, ds, W0, *, theta0=None, profile=True, lam_rough=0.0, rough_weights=None,
                 steps=200, reprofile_every=10, tol=1e-6, patience=3, map_steps=300,
                 n_prior=10.0, seed=0):
    """VarPro-learn the tensor radials of prob.model (analytic branch, M = 0).

    Minimises  r(W; theta) + lam * roughness(W)  over V with W = normalise(V)
    (unit empirical norm per radial; rows that are zero in W0 stay zero), by
    L-BFGS in rounds of `reprofile_every` steps.  With profile=True theta is
    re-MAP'd on the M = 0 LML after every round (and at the start unless
    theta0 is given), and L-BFGS restarts because the objective changed.
    lam_rough is RELATIVE: lam = lam_rough * r(W0) / roughness(W0).  Stops at
    `steps` total or at the first round that ends early (converged, line
    search, non-finite).  Returns (W, info); steps=0 returns normalise(W0)."""
    require_x64()
    require_analytic(prob.model)
    W0 = jnp.asarray(W0, jnp.float64)
    active = row_active(W0)
    Q = radial_gram(prob.model, ds, n_prior=n_prior)
    D2 = roughness_matrix(prob.model)
    wn = jnp.ones(W0.shape[2]) if rough_weights is None else jnp.asarray(rough_weights, jnp.float64)
    V = normalise(W0, Q, active)
    if theta0 is not None:
        a = to_array(theta0)
    elif profile:
        a = theta_map_linear(prob, ds, V, steps=map_steps, seed=seed)
    else:
        raise ValueError("learn_radial: profile=False needs theta0")
    r0 = float(projected_residual_from_stats(
        from_array(a), linear_statistics(with_radial(prob.model, V), prob.cfg, ds), prob.gamma))
    rough0 = float(roughness(V, D2, wn))
    lam = float(lam_rough) * r0 / max(rough0, 1e-300) if lam_rough else 0.0
    info = {"trace": [], "reasons": [], "theta": [np.asarray(a)], "lam_abs": lam,
            "lam_rough": float(lam_rough), "steps": 0}
    done = 0
    while done < int(steps):
        n = min(int(reprofile_every), int(steps) - done)
        f = lambda X, a=a: _objective(X, a, prob.model, ds, prob.gamma, Q, active, D2, wn,
                                      lam, prob.cfg)
        V, _, trace, reason = lbfgs_loop(f, V, steps=n, tol=tol, patience=patience)
        info["trace"].extend(trace)
        info["reasons"].append(reason)
        done += n
        if profile:
            a = theta_map_linear(prob, ds, normalise(V, Q, active), steps=map_steps, seed=seed, init=a)
            info["theta"].append(np.asarray(a))
        if reason != "steps":
            break
    info["steps"] = done
    info["theta_final"] = np.asarray(a)
    return normalise(V, Q, active), info
```

(If Task 6 needed the fallback, replace the body of `_objective` with `f2 = projected_residual_2pass(theta, _P(model, cfg, gamma), ds)` where `_P` is a small `NamedTuple(model, cfg, gamma)` stand-in for `prob`, then return `f2(W) + lam * roughness(W, D2, wn)`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp_learn_radial.py -v -s`
Expected: all PASS; the recovery test prints `f0`, `f1` and the round reasons.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/radial_learn.py tests/test_gp_learn_radial.py
git commit -m "feat(radial_learn): L-BFGS VarPro loop with theta re-profiling and gauge/roughness

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Held-out gate, λ grid, saving

**Files:**
- Modify: `src/ace_jax/fit/radial_learn.py`
- Test: `tests/test_gp_learn_radial.py` (append)

**Interfaces:**
- Consumes: `learn_radial`, `theta_map_linear` (Task 7); `posterior` (`fit.objective`); `patch_radial_npz` (Task 3).
- Produces:
  - `holdout_score(W, a_fit, a_norm, prob, ds_fit, ds_val) -> float`
  - `gate(candidates: dict[str, W], score: Callable[[W], float]) -> (label, scores dict)`
  - `fit_radial(prob, ds_fit, ds_val, W0, *, lam_grid=(0.0, 1e-3, 1e-2, 1e-1), theta0=None, map_steps=300, **learn_kw) -> (W_sel, info)`
  - `save_result(out_dir, W, info, *, src_npz=None, model=None) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gp_learn_radial.py`:

```python
def test_gate_ties_go_to_first():
    from ace_jax.fit.radial_learn import gate
    label, scores = gate({"init": 1, "learned": 2}, lambda w: 0.5)
    assert label == "init" and scores == {"init": 0.5, "learned": 0.5}
    label, _ = gate({"init": 1, "learned": 2}, lambda w: 1.0 if w == 1 else 0.1)
    assert label == "learned"


def test_holdout_score_energy_only_split(small):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_learn import holdout_score
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6, force_key="__no_such_key__")
    a = to_array(THETA)
    s = holdout_score(prob.model.rnl_Wnlq, a, a, prob, ds_fit, ds_val)
    assert np.isfinite(s) and s >= 0.0


def test_fit_radial_gate_prefers_learned_on_recoverable_problem():
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, W0, c = _perturbed_truth(prob)
    ds_fit, ds_val = relabel(prob, ds_fit, Wt, c), relabel(prob, ds_val, Wt, c)
    W, info = fit_radial(prob, ds_fit, ds_val, W0, lam_grid=(0.0,), theta0=THETA,
                         profile=False, steps=30, map_steps=50)
    print(info["scores"])
    assert info["selected"].startswith("learned")
    assert info["scores"][info["selected"]] < info["scores"]["init"]


def test_save_result_roundtrip(tmp_path, small):
    from ace_jax.fit.radial_learn import save_result
    prob, _, _ = small
    W = 1.5 * prob.model.rnl_Wnlq
    save_result(tmp_path, W, {"selected": "learned", "trace": [1.0, 0.5], "theta": [np.zeros(3)]},
                src_npz=MODEL, model=prob.model)
    assert np.array_equal(np.load(tmp_path / "rnl_Wnlq.npy"), np.asarray(W))
    back, meta, _ = load(tmp_path / "model.npz")
    np.testing.assert_array_equal(np.asarray(back.rnl_Wnlq), np.asarray(W))
    import json
    assert json.loads((tmp_path / "radial_info.json").read_text())["selected"] == "learned"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gp_learn_radial.py -v -k "gate or holdout or fit_radial or save_result"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add `import json`, `import pathlib` and `from .objective import posterior` to the imports of `src/ace_jax/fit/radial_learn.py`, then append:

```python
def holdout_score(W, a_fit, a_norm, prob, ds_fit, ds_val):
    """Validation error of the M = 0 linear fit with radials W: the readout is
    the posterior mean on ds_fit at theta a_fit; the score is
    sum_{t in E, F} SSE_t / (n_t sigma_t^2) on ds_val with sigma from a_norm
    (fixed across candidates so scores are comparable), SSE_t = yy - 2 c.b +
    c.G.c from ds_val's weighted linear statistics -- no design matrix.  A
    type with no rows in ds_val is skipped."""
    model = with_radial(prob.model, W)
    c, _ = posterior(from_array(a_fit), linear_statistics(model, prob.cfg, ds_fit), prob)
    val = linear_statistics(model, prob.cfg, ds_val)
    th = from_array(a_norm)
    score = 0.0
    for t in "EF":
        n = float(getattr(val, f"n_{t}"))
        if n == 0:
            continue
        G, b, yy = getattr(val, f"G_{t}"), getattr(val, f"b_{t}"), getattr(val, f"yy_{t}")
        sse = float(yy - 2.0 * c @ b + c @ G @ c)
        score += sse / (n * float(jnp.exp(2.0 * getattr(th, f"log_sigma_{t}"))))
    return score


def gate(candidates, score):
    """Score every candidate (lower is better); ties resolve to insertion order,
    so put the conservative choice ("init") first.  Returns (label, scores)."""
    scores = {k: float(score(w)) for k, w in candidates.items()}
    order = list(scores)
    return min(order, key=lambda k: (scores[k], order.index(k))), scores


def fit_radial(prob, ds_fit, ds_val, W0, *, lam_grid=(0.0, 1e-3, 1e-2, 1e-1), theta0=None,
               map_steps=300, **learn_kw):
    """learn_radial on ds_fit once per relative roughness weight in lam_grid,
    then keep the best of {init, learned per lam} on the disjoint ds_val
    (ties -> init).  Each candidate's readout is refitted at its own theta-MAP
    (warm-started from the init's); scores are normalised by the init's sigmas.
    Returns (W_sel, info)."""
    require_x64()
    require_analytic(prob.model)
    W0 = jnp.asarray(W0, jnp.float64)
    Q = radial_gram(prob.model, ds_fit, n_prior=learn_kw.get("n_prior", 10.0))
    W_init = normalise(W0, Q, row_active(W0))
    a0 = to_array(theta0) if theta0 is not None else theta_map_linear(prob, ds_fit, W_init, steps=map_steps)
    cands, runs = {"init": W_init}, {}
    for lam in lam_grid:
        W, info = learn_radial(prob, ds_fit, W0, theta0=from_array(a0), lam_rough=lam,
                               map_steps=map_steps, **learn_kw)
        cands[f"learned_lam={lam:g}"] = W
        runs[f"{lam:g}"] = info

    def score(W):
        a_fit = a0 if W is W_init else theta_map_linear(prob, ds_fit, W, steps=map_steps, init=a0)
        return holdout_score(W, a_fit, a0, prob, ds_fit, ds_val)

    label, scores = gate(cands, score)
    return cands[label], {"selected": label, "scores": scores, "runs": runs, "theta_init": np.asarray(a0)}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def save_result(out_dir, W, info, *, src_npz=None, model=None):
    """Write rnl_Wnlq.npy and radial_info.json to out_dir; with src_npz and
    model (the analytic model W belongs to, e.g. after widen_radial /
    to_analytic) also write model.npz = src_npz with the learned radial."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rnl_Wnlq.npy", np.asarray(W))
    (out / "radial_info.json").write_text(json.dumps(_jsonable(info), indent=1))
    if src_npz is not None:
        if model is None:
            raise ValueError("save_result: src_npz needs the analytic `model` W belongs to")
        from ..construct.export import patch_radial_npz
        patch_radial_npz(src_npz, out / "model.npz", with_radial(model, W))
```

Note: when `profile=False` is forwarded through `learn_kw` (as in the test), `learn_radial` uses `theta0` and never re-profiles; that is intended.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gp_learn_radial.py -v -s`
Expected: all PASS.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q -x --ignore=tests/test_gp_ladder_real.py`
Expected: all PASS / SKIP as on `main` (the baseline from the start of the branch), plus the new tests.

- [ ] **Step 6: Commit**

```bash
git add src/ace_jax/fit/radial_learn.py tests/test_gp_learn_radial.py
git commit -m "feat(radial_learn): held-out gate over a roughness grid; save learned radials

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Bench driver, benchmark, spec amendments

**Files:**
- Create: `bench/learn_radial/run.py`
- Create: `docs/dev/learn-radial-results.md`
- Modify: `docs/dev/specs/2026-09-26-learned-radial-varpro-design.md`
- Test: `tests/test_gp_learn_radial.py` (append a smoke test)

**Interfaces:**
- Consumes: everything above; `smoothness_prior` (`construct.prior`); `load_configs`, `build_dataset` (`fit.data`).
- Produces: CLI `uv run python bench/learn_radial/run.py --model M.npz --data D.xyz --out DIR [...]`, writing `DIR/{model.npz, rnl_Wnlq.npy, radial_info.json, summary.json}`.

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_gp_learn_radial.py`:

```python
def test_bench_driver_smoke(tmp_path):
    import json, subprocess, sys
    from conftest import ROOT
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench/learn_radial/run.py"), "--model", str(MODEL),
         "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
         "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8", "--batch", "4",
         "--n-q", "20", "--steps", "3", "--lam-grid", "0", "--map-steps", "20",
         "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["selected"] in s["scores"] and (tmp_path / "model.npz").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_gp_learn_radial.py::test_bench_driver_smoke -v`
Expected: FAIL (script missing, non-zero return code).

- [ ] **Step 3: Implement the driver**

Create `bench/learn_radial/run.py`:

```python
"""Learn the tensor radials of an ACE model by VarPro, gate on a held-out split.

    uv run python bench/learn_radial/run.py --model M.npz --data D.xyz --out DIR \
        [--n-q 30] [--steps 200] [--lam-grid 0,1e-3,1e-2,1e-1]

A splined (Julia-exported) model is converted to the analytic branch first
(to_analytic); an analytic one is widened to --n-q.  Writes DIR/model.npz (the
selected radials patched into a copy of --model), rnl_Wnlq.npy,
radial_info.json and summary.json (gate scores, selected label).  The residual
GP / UQ fit then runs on DIR/model.npz as usual.
"""
import argparse
import json
import pathlib
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ace_jax.construct.prior import smoothness_prior
from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import default_prior
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem
from ace_jax.fit.radial_learn import fit_radial, save_result
from ace_jax.fit.radial_model import rnl_degrees, to_analytic

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--energy-key", default="energy"); p.add_argument("--force-key", default="forces")
p.add_argument("--virial-key", default="virial")
p.add_argument("--ntrain", type=int, default=800); p.add_argument("--nval", type=int, default=200)
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--r0", type=float, default=2.35, help="hyperprior length scale (default_prior)")
p.add_argument("--n-q", type=int, default=30, help="tensor-radial polynomial span after widening")
p.add_argument("--steps", type=int, default=200); p.add_argument("--reprofile-every", type=int, default=10)
p.add_argument("--lam-grid", default="0,1e-3,1e-2,1e-1", help="relative roughness weights")
p.add_argument("--map-steps", type=int, default=300)
a = p.parse_args()

t0 = time.time()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
model, meta, z = load(a.model)
model, relres = to_analytic(model, a.n_q)
NZ = len(meta["elements"])

configs = load_configs(a.data, a.energy_key, a.force_key, a.virial_key)
perm = np.random.default_rng(a.seed).permutation(len(configs))
if a.ntrain + a.nval > len(configs):
    raise SystemExit(f"--ntrain + --nval = {a.ntrain + a.nval} > {len(configs)} configs")
fit_c = [configs[i] for i in perm[:a.ntrain]]
val_c = [configs[i] for i in perm[a.ntrain:a.ntrain + a.nval]]
E0 = np.asarray(z["E0"]) if "E0" in z else np.zeros(NZ)
ds_fit = build_dataset(fit_c, meta, E0, configs_per_batch=a.batch)
ds_val = build_dataset(val_c, meta, E0, configs_per_batch=a.batch)

cfg = GPConfig(r0=a.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=NZ, C=a.batch)
X, S = site_features(model, cfg, ds_fit)
ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
if "gamma" in z:
    gamma = np.asarray(z["gamma"])
else:   # same construction as construct.model.build_model
    tensor_nnll = [[tuple(b) for b in bb] for bb in meta["nnll"]]
    pair_nnll = [[(n, 0)] for n in range(1, meta["n_pair"] + 1)]
    gamma = smoothness_prior(tensor_nnll * NZ + pair_nnll * NZ)
prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(gamma), default_prior(a.r0))

wn = 1.0 / (1.0 + rnl_degrees(meta)) ** 2
lam_grid = tuple(float(x) for x in a.lam_grid.split(","))
W, info = fit_radial(prob, ds_fit, ds_val, model.rnl_Wnlq, lam_grid=lam_grid,
                     rough_weights=wn, steps=a.steps, reprofile_every=a.reprofile_every,
                     map_steps=a.map_steps)
info["to_analytic_relres_max"] = float(np.max(relres))
save_result(out, W, info, src_npz=a.model, model=model)
summary = {"selected": info["selected"], "scores": info["scores"], "n_q": a.n_q,
           "ntrain": a.ntrain, "nval": a.nval, "lam_grid": list(lam_grid),
           "seconds": time.time() - t0}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
```

Before relying on `meta["nnll"]` for gamma, confirm its layout by comparing with `construct.model.build_model` (`tensor_nnll = list(nnll)` there). If a fitted Julia npz already carries `gamma`, that path is used instead.

- [ ] **Step 4: Run the smoke test**

Run: `uv run pytest tests/test_gp_learn_radial.py::test_bench_driver_smoke -v`
Expected: PASS.

- [ ] **Step 5: Amend the spec to match what was built**

In `docs/dev/specs/2026-09-26-learned-radial-varpro-design.md`:
- "Components → `from_table`": the projector works in the transformed coordinate `x` (grid on [-1, 1]) with the envelope as part of the basis (`env·P_q`), not "divide out the envelope".
- "Components → `normalise`": rows that are zero in `W0` (onehot for NZ > 1) are frozen at zero.
- "`projected_residual`": no jitter-retry exists in `solve.py`; a failed Cholesky gives NaN and the loop keeps the best finite iterate.
- "λ_r selection": `lam_rough` is relative (`λ = lam_rough · r(W0)/roughness(W0)`), default grid `(0, 1e-3, 1e-2, 1e-1)`.
- "Held-out gate": candidates are refitted at their own θ-MAP (warm-started from the init's); scores use the init's σ.
- "Out of scope": add per-config-type noise ratios (`log_ratios`) in the radial objective.
- "Entry points": library + `bench/learn_radial/run.py`; CLI still waits for #8.

- [ ] **Step 6: Benchmark on moriarty and record results**

On moriarty (plain `ssh moriarty`; the worktree is on the shared home), for each system for which a model and training xyz exist (Si, SiGe, CrMnFe; reuse the data and models that `bench/acegp_cantor/run.py` was run with, with the same energy/force/virial keys), run:

```bash
uv run python bench/learn_radial/run.py --model <model.npz> --data <train.xyz> \
    --energy-key <key> --force-key <key> --virial-key <key> \
    --ntrain 800 --nval 200 --n-q 30 --steps 200 --out runs/learn_radial/<system>
```

Also record the Task 6 Step 4 memory numbers. Write `docs/dev/learn-radial-results.md` with one table row per system: `n_q`, selected label, the init and learned gate scores, wall time, peak memory, and a one-line verdict against the spec's success criterion (learned wins on ≥ 2 of 3 systems). Record what was actually run; if a system's data isn't available, say so in its row rather than leaving it out.

- [ ] **Step 7: Commit**

```bash
git add bench/learn_radial/run.py tests/test_gp_learn_radial.py docs/dev/specs/2026-09-26-learned-radial-varpro-design.md docs/dev/learn-radial-results.md
git commit -m "feat(bench): learn_radial driver; spec amendments; benchmark results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Deferred (not in this plan)

- CLI subcommand for radial learning: waits for PR #8 (`fit/pipeline/`), then becomes a pipeline stage calling `fit_radial` + `save_result`.
- Plan B (MACE initialisation): `scripts/extract_mace_radial.py` + `construct/mace_radial.py`, reusing `from_table` (Task 2) and `fit_radial` (Task 8) with an extra `mace` candidate.
- Agnesi transform learning, Kaufman-LM, `Â` caching, `pair_Wnlq`, per-type noise in the objective.
