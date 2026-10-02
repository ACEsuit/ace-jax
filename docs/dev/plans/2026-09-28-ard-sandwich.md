# Cluster-sandwich ARD force variance: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** served per-atom force σ = λ·sqrt(Σ_c ‖Qᵀφ̃_c‖²), with the configuration-clustered sandwich factor Q, as an `ARDPosterior` option (the default), λ fitted on the train hold-out.

**Architecture:**
- `fit/ard.py` gains:
  - the cluster scores of the training configurations at the ARD mean;
  - the factor Q = S⁻¹G̃;
  - `ARDPosterior.Q/lam` and the variance methods;
  - schema 2.
- The stage computes Q after the full refit and λ with the κ refit rule.
- `predict_ard`, `forces_std` and the calculator go through one `force_var_rows`.

**Tech stack:** JAX (x64), NumPy, pytest, ruff; `uv run`.

**Spec:** `docs/dev/specs/2026-09-28-tempered-ard-uq-design.md`, section "Addendum (2026-09-28): cluster-sandwich force variance".

## Global Constraints

- Worktree `/Users/u1470235/gits/ace-jax/.worktrees/ard-uq`, branch `feat/ard-uq`. Run everything with `uv run`.
- float64 for all fitting maths. Q is stored as float32 in `posterior.npz` and loaded back as float64.
- Names are binding: `ARDEvidence`, `ARDPosterior`, `ARDStats`, `ARDResult`, `sandwich_scores`, `sandwich_factor`, `force_var_rows`, `misspec_var_rows`.
- `FitConfig.ard_variance` ∈ {"sandwich", "kappa"}, default "sandwich". CLI flag `--ard-variance`.
- `posterior.npz` SCHEMA = 2. A schema-1 file still loads (Q = None, lam = 1.0).
- E/V variances stay untempered. `var_rows` stays the epistemic φA⁻¹φᵀ.
- Tests are fast by default. The full suite (`uv run pytest -q`) and `uv run ruff check` must stay green.
- Commit trailer lines, exactly:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01P4oGcUAsifmZRFR3Kkg3Ei`

## Review Focus

1. **Padded configs and nodes** in a batch must contribute no cluster column. Padded configs are dropped via `cfg_mask`; padded nodes carry `node_cfg == C` and are dropped by the extra segment. Task 1 tests the column count.
2. **Sequential mode** has no σ_q in `h`. The residuals must be whitened with the fixed MAP σ_q (`ls_fixed`). Task 2 tests a sequential stage.
3. **A schema-1 `posterior.npz`** from an earlier run must still load and serve κ variance. Task 1 tests it.
4. **Atoms with no neighbours** (zero force rows) must give σ = 0, not NaN, on the sandwich path. Task 1 tests `forces_std` on a zero row.
5. **The calculator's `forces_std` must equal the pipeline's F σ with a sandwich posterior.** Task 2 re-runs the existing calculator-matches-pipeline test with the default variance.

---

### Task 1: sandwich factor and posterior variance (`fit/ard.py`)

**Files:** modify `src/ace_jax/fit/ard.py`; test `tests/test_ard.py`.

**Interfaces:**
- Produces:
  - `ARDEvidence.sigmas(h) -> np.ndarray (3,)`;
  - `sandwich_scores(post, prob, ds, sig) -> np.ndarray (L, n_cfg)`;
  - `sandwich_factor(post, G) -> np.ndarray (L, n_cfg)`;
  - `ARDPosterior` fields `Q=None`, `lam=1.0`;
  - `ARDPosterior.misspec_var_rows(Phi)`, `ARDPosterior.force_var_rows(Phi)`;
  - `forces_std` routed through `force_var_rows`;
  - SCHEMA 2.

- [ ] **Step 1: write the failing tests** (append to `tests/test_ard.py`; `_orders` and `highest_precision` are already imported there)

```python
def _sandwich_setup(tiny_linear_problem):
    from ace_jax.fit.ard import ARDEvidence, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}
    ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                     body_order_columns(meta, prob.cfg))
    h = ev.h0(theta)
    return prob, ds, ev, h, ard_posterior(ev, h, 2.0, meta)


def test_sandwich_scores_sum_to_the_prior_force_at_the_mean(tiny_linear_problem):
    """Stationarity: sum_c g~_c = D^-1 Lambda c = lam_prior * x at the posterior mean, so the
    residuals, their whitening and the cluster sums are exactly the posterior's own."""
    from ace_jax.fit.ard import sandwich_scores
    with highest_precision():
        prob, ds, ev, h, post = _sandwich_setup(tiny_linear_problem)
        G = sandwich_scores(post, prob, ds, ev.sigmas(h))
        _, _, lam, _ = ev._parts(jnp.asarray(h, float))
        x = post.mean / post.dinv                                                    # scaled mean D c
    n_cfg = int(np.asarray(ds.cfg_mask).sum())
    assert G.shape == (prob.cfg.len_basis, n_cfg)                                    # padded configs dropped
    np.testing.assert_allclose(G.sum(1), np.asarray(lam) * x, rtol=1e-6, atol=1e-8 * np.abs(G).max())


def test_sandwich_variance_matches_dense_reference(tiny_linear_problem):
    """lam^2 ||Q^T phi~||^2 == lam^2 phi A^-1 M A^-1 phi^T with A and M built densely in the original
    coordinates (M = sum over configs of the outer product of the summed residual-weighted rows)."""
    from ace_jax.fit.ard import sandwich_factor, sandwich_scores
    with highest_precision():
        prob, ds, ev, h, post = _sandwich_setup(tiny_linear_problem)
        G = sandwich_scores(post, prob, ds, ev.sigmas(h))
        post = post._replace(Q=sandwich_factor(post, G), lam=1.7)
        Ms, _, lam, _ = ev._parts(jnp.asarray(h, float))
        D = 1.0 / post.dinv
        A = np.asarray(Ms + jnp.diag(lam)) * D[:, None] * D[None, :]                 # unscaled A
        Gu = G * D[:, None]                                                          # unscaled scores
        Sig = np.linalg.solve(A, np.linalg.solve(A, Gu @ Gu.T).T)                    # A^-1 M A^-1
        b0 = jax.tree.map(lambda a: a[0], ds)
        from ace_jax.fit.rows import linear_rows
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, b0)[0].F).reshape(-1, prob.cfg.len_basis)
        got = post.force_var_rows(Fr)
    ref = 1.7 ** 2 * np.einsum("nl,lm,nm->n", Fr, Sig, Fr)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-12 * ref.max())


def test_posterior_schema2_roundtrip_and_schema1_loads(tiny_linear_problem, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, sandwich_factor, sandwich_scores
    from ace_jax.fit.rows import linear_rows
    with highest_precision():
        prob, ds, ev, h, post = _sandwich_setup(tiny_linear_problem)
        post = post._replace(Q=sandwich_factor(post, sandwich_scores(post, prob, ds, ev.sigmas(h))), lam=3.0)
        post.save(tmp_path / "p.npz")
        back = ARDPosterior.load(tmp_path / "p.npz")
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
        np.testing.assert_allclose(back.forces_std(Fr), post.forces_std(Fr), rtol=1e-3)
        assert back.lam == 3.0 and back.Q.dtype == np.float64
        # a schema-1 file (no Q / lam) still loads and serves kappa variance
        z = dict(np.load(tmp_path / "p.npz"))
        z.pop("Q"); z.pop("lam"); z["schema"] = np.array(1)
        np.savez(tmp_path / "p1.npz", **z)
        old = ARDPosterior.load(tmp_path / "p1.npz")
        assert old.Q is None and old.lam == 1.0
        np.testing.assert_allclose(old.forces_std(Fr), old.kappa * np.sqrt(
            old.var_rows(Fr.reshape(-1, Fr.shape[-1])).reshape(-1, 3).sum(1)), rtol=1e-12)
        zero = np.zeros((2, 3, prob.cfg.len_basis))
        assert np.all(post.forces_std(zero) == 0.0)                                  # no neighbours: 0, not NaN
```

- [ ] **Step 2: run them:** `uv run pytest tests/test_ard.py -q -k "sandwich or schema2"`. Expected: FAIL (ImportError on `sandwich_scores`).

- [ ] **Step 3: implement** in `src/ace_jax/fit/ard.py`.
  - Set `SCHEMA = 2`.
  - In `ARDEvidence.__init__` add `self.ls_fixed = stats.ls_fixed` and the method below.
  - Add the fields and methods to `ARDPosterior`, update `save` and `load`, and add the two functions after `ard_posterior`:

```python
    def sigmas(self, h):
        """Noise scales (sigma_E, sigma_F, sigma_V) of hyperparameters h: fitted [joint] or the fixed
        linear MAP ones [sequential]."""
        return np.exp(np.asarray(h[:3], float)) if self.joint else np.exp(np.asarray(self.ls_fixed, float))
```

```python
    # ARDPosterior: new trailing fields (defaults keep every existing positional construction valid)
    Q: np.ndarray | None = None   # (L, n_cfg) S^-1 G~: configuration-clustered sandwich factor
    lam: float = 1.0              # sandwich scale (fitted like kappa)

    def misspec_var_rows(self, Phi, chunk=4096):
        """Unscaled cluster-sandwich variance ||Q^T (D^-1 phi)||^2 = phi A^-1 M A^-1 phi^T per row."""
        Q = jnp.asarray(self.Q, jnp.float64)
        out = []
        for i in range(0, len(Phi), chunk):
            v = (jnp.asarray(Phi[i:i + chunk]) * jnp.asarray(self.dinv)[None, :]) @ Q
            out.append(np.asarray(jnp.sum(v * v, axis=1)))
        return np.concatenate(out) if out else np.zeros(0)

    def force_var_rows(self, Phi):
        """The served (calibrated) variance of each force-component row: lam^2 x sandwich when Q is
        set, else kappa^2 x the epistemic posterior variance."""
        if self.Q is not None:
            return self.lam ** 2 * self.misspec_var_rows(Phi)
        return self.kappa ** 2 * self.var_rows(Phi)

    def forces_std(self, Frows):
        """Per-atom force std sqrt(sum_c force_var_rows) from force rows (N, 3, L), numpy or device."""
        v = self.force_var_rows(Frows.reshape(-1, Frows.shape[-1])).reshape(-1, 3)
        return np.sqrt(np.maximum(v.sum(1), 0.0))
```

  In `save`, add `**({} if self.Q is None else {"Q": np.asarray(self.Q, dtype)})` and `lam=self.lam`.

  In `load`:
  - accept `int(z["schema"]) in (1, 2)`;
  - pass `Q=z["Q"].astype(np.float64) if "Q" in z.files else None`;
  - pass `lam=float(z["lam"]) if "lam" in z.files else 1.0`.

```python
def sandwich_scores(post, prob, ds, sig):
    """Prior-scaled cluster scores G~ (L, n_cfg) of the configurations of ds at the posterior mean:
    g~_c = D^-1 sum_{i in c} rho_i psi_i over every E/F/V row i of config c, psi_i = phi_i w_i/sigma_q,
    rho_i = (y_i - phi_i c) w_i/sigma_q.  Padded configs (cfg_mask) are dropped; padded nodes carry
    node_cfg == C and land in a discarded extra segment."""
    from .rows import linear_rows
    L = prob.cfg.len_basis
    c, dinv = jnp.asarray(post.mean), jnp.asarray(post.dinv)
    inv = jnp.asarray(1.0 / np.asarray(sig, float))

    @jax.jit
    def scores(bt):
        r = linear_rows(prob.model, prob.cfg, bt)[0]
        C = r.E.shape[0]
        P = jnp.concatenate([r.E, r.F.reshape(-1, L), r.V.reshape(-1, L)])
        y = jnp.concatenate([bt.y_E, bt.y_F.reshape(-1), bt.y_V.reshape(-1)])
        w = jnp.concatenate([bt.w_E * inv[0], jnp.repeat(bt.w_F, 3) * inv[1], jnp.repeat(bt.w_V, 6) * inv[2]])
        cid = jnp.concatenate([jnp.arange(C), jnp.repeat(bt.node_cfg, 3), jnp.repeat(jnp.arange(C), 6)])
        g = jax.ops.segment_sum(P * ((y - P @ c) * w * w)[:, None], cid, num_segments=C + 1)[:C]
        return g * dinv[None, :]

    out = []
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        out.append(np.asarray(scores(bt))[np.asarray(bt.cfg_mask)])
    return np.concatenate(out).T


def sandwich_factor(post, G):
    """Q = S^-1 G~ (L, n_cfg), so that phi A^-1 M A^-1 phi^T = ||Q^T (D^-1 phi)||^2."""
    c = jnp.asarray(post.chol, jnp.float64)
    return np.asarray(cho_solve((c, True), jnp.asarray(G)))
```

- [ ] **Step 4: run** `uv run pytest tests/test_ard.py -q`. Expected: all pass, including the 3 new tests.
  - If the stationarity tolerance fails narrowly on the fixture's conditioning, look at `np.abs(G.sum(1) - lam*x).max() / np.abs(G).max()` before loosening anything, and record it in the report.
- [ ] **Step 5: full suite + ruff:** `uv run pytest -q` and `uv run ruff check`.
- [ ] **Step 6: commit** `feat(ard): cluster-sandwich factor Q, ARDPosterior.Q/lam, force_var_rows, schema 2`.

### Task 2: stage, prediction, config and CLI

**Files:** modify `src/ace_jax/fit/ard.py` (`run_ard_stage`, `_val_errors`, `predict_ard`), `src/ace_jax/fit/pipeline/config.py`, `src/ace_jax/cli.py`; test `tests/test_ard.py`, `tests/test_ard_pipeline.py`, `tests/test_ard_calc_cli.py`.

**Interfaces:**
- Consumes the Task 1 API.
- Produces:
  - `FitConfig.ard_variance`;
  - `ard.json` keys `variance`, `lam`, `n_clusters`, `val_rms_z_sandwich`;
  - `--ard-variance`.

- [ ] **Step 1: failing tests.**
  - In `tests/test_ard.py`, `_pipe_cfg` gains `ard_variance="kappa"` in its base dict, so the existing κ-tempering tests keep testing the κ path.
  - Append:

```python
def test_ard_stage_sandwich_variance_and_lam_rule(monkeypatch):
    """Default variance: Q from the full refit's training residuals, lam by the kappa refit rule
    (held-out subset errors against the served posterior's sandwich variance), F_var = lam^2 sandwich."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    calls = []
    orig = ard.kappa_closed_form
    monkeypatch.setattr(ard, "kappa_closed_form", lambda e2, s2: calls.append((np.array(e2), np.array(s2))) or orig(e2, s2))
    cfg = _pipe_cfg(ard_variance="sandwich").validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        pred = ard.predict_ard(res.posterior, b.prob, d.ds_test)
        pred1 = ard.predict_ard(res.posterior._replace(lam=1.0), b.prob, d.ds_test)
    post, rep = res.posterior, res.report
    assert post.Q is not None and post.Q.shape == (b.prob.cfg.len_basis, len(d.train))
    assert rep["variance"] == "sandwich" and rep["n_clusters"] == len(d.train)
    assert len(calls) == 3 and np.array_equal(calls[2][0], calls[0][0])       # kappa_sub, kappa, lam: same e2
    assert post.lam == orig(*calls[2]) == rep["lam"] and abs(rep["val_rms_z_sandwich"] - 1.0) < 1e-6
    np.testing.assert_allclose(pred.F_var, post.lam ** 2 * pred1.F_var, rtol=1e-12)


def test_ard_stage_sandwich_in_sequential_mode():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.ard import run_ard_stage
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg(ard_variance="sandwich", ard_mode="sequential").validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
    assert res.posterior.Q is not None and np.isfinite(res.posterior.lam) and res.posterior.lam > 0
```

  - Config validation: extend `test_config_validates_ard` in `tests/test_ard_pipeline.py` with `pytest.raises(ValueError, match="ard_variance")` for `ard_variance="nope"`, and assert that the default is `"sandwich"`.
  - CLI: extend the existing `aj fit --uq ard` CLI test in `tests/test_ard_calc_cli.py` to pass `--ard-variance kappa` once and assert `json.load(open(out/"ard.json"))["variance"] == "kappa"`. Leave the other CLI tests on the default.

- [ ] **Step 2: run** `uv run pytest tests/test_ard.py tests/test_ard_pipeline.py tests/test_ard_calc_cli.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: implement.**
  - `config.py`:
    - add `ard_variance: str = "sandwich"   # "sandwich" (configuration-clustered, spec addendum) | "kappa"`;
    - validate it next to `ard_mode` with the message `f"ard_variance must be 'sandwich' or 'kappa', got {self.ard_variance!r}"`.
  - `cli.py`:
    - `p.add_argument("--ard-variance", choices=["sandwich", "kappa"], default="sandwich", help=...)`;
    - pass `ard_variance=a.ard_variance` where `ard_mode=a.ard_mode` is passed.
  - `_val_errors(post, prob, ds)`: also return a third array `m2`, the per-atom sum of `post.misspec_var_rows` over the 3 components when `post.Q is not None`, else `None`.
    - Update its two call sites (`e2, s2, _ = ...`).
    - Stay inside the existing loop so no extra rows pass is made.
  - `run_ard_stage`, after `post = ard_posterior(ev, h, 1.0, data.meta)` and the κ refit:

```python
    variance = getattr(cfg, "ard_variance", "kappa")
    lam, n_clusters, m2_full = 1.0, 0, None
    if variance == "sandwich":
        G = sandwich_scores(post, prob, data.ds_train, ev.sigmas(h))
        post = post._replace(Q=sandwich_factor(post, G))
        n_clusters = int(G.shape[1])
        del G
        m2_full = _val_errors(post, prob, ds_val)[2][ok]
        lam = kappa_closed_form(e2, m2_full)
        log(f"ARD: sandwich over {n_clusters} training configs; lam {lam:.3f}")
    post = post._replace(kappa=kappa, lam=lam)
```

    Merge this with the existing κ refit lines so there is exactly one `_val_errors(post, ...)` call on the full posterior. Its `s2` feeds κ and its `m2` feeds λ, so `_val_errors` must be called after `Q` is set. Keep the call order `kappa_closed_form(subset)`, then `kappa_closed_form(full κ)`, then `kappa_closed_form(λ)`.

    Report additions:
    - `"variance": variance`, `"lam": lam`, `"n_clusters": n_clusters`;
    - when sandwich: `"val_rms_z_sandwich": float(np.sqrt(np.mean(e2 / (lam ** 2 * m2_full / 3)) / 3))`.
  - `predict_ard`: replace `k2 * post.var_rows(F).reshape(-1, 3)` with `post.force_var_rows(F).reshape(-1, 3)`. Delete the now-unused `k2`; keep its docstring accurate.

- [ ] **Step 4: run** the three test files. Expected: pass.
  - The existing `test_calculator_forces_std_matches_pipeline` now runs with the default (sandwich) variance, which is Review Focus 5. It must pass unchanged.
- [ ] **Step 5: full suite + ruff.**
- [ ] **Step 6: commit** `feat(ard): configuration-clustered sandwich force variance (default), lam from the train hold-out; --ard-variance`.

### Task 3: docs

**Files:** `README.md`, `skills/ace-jax/SKILL.md`.

- [ ] **Step 1: README.** In the `--uq ard` paragraph, say that the default `--ard-variance sandwich` serves the configuration-clustered sandwich variance, as follows:
  - σ² = λ²·φA⁻¹MA⁻¹φᵀ, the misspecification-robust covariance, with λ from the train hold-out;
  - on the bench365 prototype it ranked local errors better than the tempered posterior (ρ 0.26–0.37 against 0.15–0.26) with the same calibration and OOD detection;
  - `--ard-variance kappa` keeps the single-temperature posterior;
  - `posterior.npz` gains an (L, n_train_configs) float32 factor.

  Keep the existing prototype and acceptance caveats.
- [ ] **Step 2: SKILL.md.** Add one gotcha line: the default sandwich variance needs the training data at fit time and stores an (L, n_cfg) factor; use `--ard-variance kappa` for the smaller posterior.
- [ ] **Step 3:** `uv run pytest -q`, `uv run ruff check`.
- [ ] **Step 4: commit** `docs: --ard-variance sandwich (default)`.
