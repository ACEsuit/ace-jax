# Jackknife shape + Mondrian conformal scale: implementation plan (revision 2)

> **Note (2026-10-03):** the default force shape was changed to aniso after validation; the plan text below is historical and still says iso.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve per-atom force uncertainty with a stated coverage. The shape is an exact, centred, delete-one-cluster (PRESS/CR3) jackknife covariance, with spatial sub-clustering and an optional anisotropic 3×3 block. Two per-group, configuration-weighted scales sit on top: an rms factor for `forces_std`/`forces_cov` and a conformal quantile for `forces_q`. Calibration comes from a stratified hold-out posterior, plus `aj calibrate` and a support diagnostic.

**Architecture:** Four new pure modules hold the numerics:
- `fit/clusters.py`: which rows form each sandwich cluster;
- `fit/jackknife.py`: PRESS scores, the centred factor R and the per-atom shape V;
- `fit/conformal.py`: features, groups, stratified split, weighted scales and the per-group table;
- `fit/support.py`: the covariate-shift diagnostic.

`run_ard_stage` (in `fit/ard.py`) orchestrates them: stratified split → P_fit with its own shape → T_val scores → served P with stored R → per-group scales. `ARDPosterior` schema 3 stores the result. `ACECalculator` and the CLI serve it.

**Tech Stack:** JAX (x64), NumPy, SciPy (`scipy.stats.chi`, `scipy.linalg`), ASE, pytest, ruff; `uv run`.

**Spec:** `docs/dev/specs/2026-09-30-conformal-force-sigma-design.md` (revision 2). Mathematical reference (binding): `docs/dev/specs/tex/force-uq-math-pipeline.tex` (revision 2).

## Global Constraints

- Worktree `~/projects/ace-jax/.worktrees/conformal-uq`, branch `feat/conformal-uq`. Run everything with `uv run`. `uv run pytest -q` and `uv run ruff check` must stay green after every task.
- **Unchanged:** the mean c̄, E/V predictions and variances, the evidence fit, `--ard-variance kappa`'s epistemic shape A⁻¹, and the lean/full split (σ rows always from the full `calc.model`).
- `FitConfig` new fields and defaults:
  - `ard_force_shape = "iso"` (∈ {"iso", "aniso"}), `ard_shape_eps = 1e-3`;
  - `ard_coverage = 0.9`, `ard_groups = "distortion"` (∈ {"distortion", "none"});
  - `ard_cluster_size = 3.0` (× r_cut; `float("inf")` = whole configurations), `ard_press = "exact"` (∈ {"exact", "block"}), `ard_shape_tau = 1.0`;
  - `ard_n_min = 20`;
  - `ard_support = True`, `ard_support_max_atoms = 50000`;
  - private, bench-only: `_shape_variant = "press"` (∈ {"press", "legacy"}) and `_score_source = "fit"` (∈ {"fit", "mixed"});
  - `ard_val_frac = 0.2` is kept.
- **Weighting and quantiles.** Each calibration atom i in configuration c and group g has weight 1/n_{c,g}. Then:
  - `lam_rms_g² = Σ_c (1/n_{c,g}) Σ_i s_i² / (3 n_cfg,g)`;
  - `q_g = inf{t : F̂_g(t) ≥ 1−α}` with `F̂_g(t) = [Σ_c (1/n_{c,g}) Σ_i 1{s_i ≤ t} + 1{∞ ≤ t}] / (n_cfg,g + 1)`;
  - `r_g = q_g / (lam_rms_g · chi.ppf(1−α, 3))`.

  Groups with `n_cfg,g < ard_n_min` take the values of the nearest band with the same z-flag, then across the flag, then the all-groups values.
- **Groups:** `g = band(d)·2 + (z ≠ z*)`, with G = 2·(len(edges)+1); G = 8 for `distortion`, 2 for `none`. `band` uses the edges at the p50/p90/p99 of d over **all atoms of T**; NaN d goes in the top band. r₁ is the first RDF minimum (1.25 × the first peak if there is none), and z* the modal coordination.
- **Shape.** With H_kk = Ψ_k A⁻¹ Ψ_kᵀ:
  - g̃_k = Ψ_kᵀ (I − H_kk)⁻¹ ρ_k;
  - the centred Q̃ = S⁻¹D⁻¹[g̃_k − ḡ], and R with RRᵀ = Q̃Q̃ᵀ (R = Q̃ if K ≤ L and τ = 1, else U_rΣ_r of the thin SVD, truncated to τ of Σσ²);
  - V_αβ(x) = (Rᵀu_α)·(Rᵀu_β), with u_α = D⁻¹φ_α(x)ᵀ.
- **Scores:** iso `s = |e| / √(v/3)`, with v = tr V; aniso `s = √(eᵀ(V + ε·(v/3)·I)⁻¹e)`.
- **Served quantities:**
  - `forces_std = lam_rms_g · √v`;
  - `forces_cov = lam_rms_g² · V`;
  - `forces_q` = `q_g √(v/3)` (iso) or `q_g √λ_max(V + ε(v/3)I)` (aniso);
  - `forces_q_mahal = q_g` (aniso);
  - `forces_group = g`.
- **Schema:** `posterior.npz` `SCHEMA = 3`. Schema 1 and 2 still load and serve their scalar `forces_std`. `forces_q`, `forces_cov` and `forces_group` then raise a `ValueError` whose message contains "refit with --uq ard".
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`

## Review Focus

1. **A target atom outside all calibrated territory:** its group was merged away, d is NaN (vacuum or an isolated atom), or d is beyond every edge. It gets a finite top-band scale, never NaN or a crash. Owner: Task 1 (`test_assign_groups_extremes`, `test_scales_merge_small_groups`).
2. **A cluster with leverage close to 1** (e.g. one configuration dominates a direction). The PRESS solve stays finite, matches the refit, and the reported λ_max(H_kk) is < 1. Owner: Task 3 (`test_press_high_leverage_cluster`).
3. **A large training cell with `--ard-cluster-size inf`** (n_k > L). The push-through path equals the exact deletion. Owner: Task 3 (`test_push_through_matches_exact`).
4. **`aj calibrate` given a U that covers only some groups.** Only groups with ≥ n_min U-configurations are replaced; the rest pool T_val ∪ U; the counts say which. Owner: Task 8 (`test_calibrate_per_group_replace`).
5. **A schema-2 posterior given to the calculator, eval or calibrate.** `forces_std` serves the scalar λ; the new properties raise a clear error; calibrate refuses. Owner: Task 4 (`test_schema2_serves_scalar`) and Task 8 (`test_calibrate_refuses_schema2`).

---

## File Structure

| file | role |
|---|---|
| `src/ace_jax/fit/conformal.py` (new) | shell features, groups, edges, stratified split, configuration-weighted λ_rms/q, merging, `GroupTable` |
| `src/ace_jax/fit/clusters.py` (new) | per-configuration cluster layout (one cluster, or spatial blocks plus an E/V cluster) and row → cluster ids |
| `src/ace_jax/fit/jackknife.py` (new) | PRESS scores (exact, push-through, block), centring, factor R, per-atom V |
| `src/ace_jax/fit/support.py` (new) | per-species descriptor PCA, a grouped-CV logistic density ratio, the weighted quantile |
| `src/ace_jax/fit/ard.py` (modify) | `ARDPosterior` schema 3 and its served quantities; `run_ard_stage`; `predict_ard` |
| `src/ace_jax/fit/pipeline/config.py` (modify) | the new fields and their validation |
| `src/ace_jax/calc/point.py` (modify) | the served properties |
| `src/ace_jax/cli.py` (modify) | fit flags, `calibrate`, eval outputs |
| `bench/defect_uq/scoring/validate_shape.py` (new), `bench/defect_uq/fit_bench.py`, `bench/defect_uq/modal/modal_bench365.py` (modify) | the validation programme |
| `tests/conftest.py` (modify) | the fixtures `ard_setup` and `one_config_batch` |
| `tests/test_conformal.py`, `tests/test_jackknife.py` (new); `tests/test_ard.py`, `tests/test_ard_pipeline.py`, `tests/test_ard_calc_cli.py` (modify) | tests |

---

### Task 1: `fit/conformal.py` — features, groups, stratified split, weighted scales

**Files:**
- Create: `src/ace_jax/fit/conformal.py`, `tests/test_conformal.py`
- Modify: `tests/conftest.py` (add `one_config_batch`)

**Interfaces:**
- Consumes: `ace_jax.fit.data.Config`, `build_dataset(configs, meta, E0, configs_per_batch)` (meta needs `rcut`; pass `elements` too, as `calc/point.py:451` does). The `Dataset` fields are `rij, nbr, nbr_mask, node_z, node_cfg, node_mask, y_E, w_E, y_F, w_F, y_V, w_V, n_atoms, cfg_mask, ...`.
- Produces:
  - `chi3_ppf(p) -> float`;
  - `shell_reference(dists) -> float` (r₁);
  - `shell_features(batch, r1) -> (z int64 (Ncap,), d float (Ncap,))`;
  - `band_edges(d, qs=(0.5, 0.9, 0.99)) -> ndarray`;
  - `n_groups(edges) -> int`;
  - `assign_groups(z, d, z_star, edges) -> int64 ndarray`;
  - `config_strata(groups_per_cfg: list[ndarray]) -> ndarray`;
  - `stratified_split(strata, f, seed) -> (fit_idx, val_idx)`;
  - `group_scales(scores, groups, cfg, G, alpha, n_min, src=None) -> GroupTable`;
  - `GroupTable`, a frozen dataclass: `alpha: float, n_min: int, lam_rms, q, r (G,) float, n_cfg, n_cfg_val, n_cfg_cal, n_atoms (G,) int, merged: list[[g, g_src]], sources: list = []`, with `to_dict()` / `from_dict(d)`.

- [ ] **Step 1: Add the fixture to `tests/conftest.py`**

```python
@pytest.fixture
def one_config_batch():
    """Factory: ASE Atoms -> the one-config Dataset batch (the calculator's construction)."""
    import jax
    from ace_jax.fit.data import Config, build_dataset

    def make(atoms, rcut=6.25):
        meta = {"elements": sorted({int(z) for z in atoms.numbers}), "rcut": rcut}
        c = Config(atoms.get_positions(), atoms.get_atomic_numbers(), atoms.get_cell().array, atoms.get_pbc(),
                   None, None, None, 1.0, 1.0, 1.0)
        return jax.tree.map(lambda a: a[0], build_dataset([c], meta, np.zeros(len(meta["elements"])), 1))
    return make
```

- [ ] **Step 2: Write the failing tests** in `tests/test_conformal.py`

```python
import numpy as np
import pytest
from ase.build import bulk, fcc111


def _wq(s, w, alpha):
    """Brute-force pooled-CDF quantile: smallest t with (sum_{s_i<=t} w_i) / (W + 1) >= 1 - alpha."""
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / (w.sum() + 1.0)
    k = np.searchsorted(cw, 1 - alpha - 1e-15)
    return np.inf if k >= len(s) else s[o][k]


def test_shell_features_fcc_strained_slab(one_config_batch):
    from ace_jax.fit.conformal import shell_features
    a0 = 3.65
    r1 = 0.5 * (a0 / np.sqrt(2) + a0)
    z, d = shell_features(one_config_batch(bulk("Ni", "fcc", a=a0, cubic=True).repeat(3)), r1)
    assert np.all(z[:108] == 12) and np.allclose(d[:108], 0.0, atol=1e-12)
    st = bulk("Ni", "fcc", a=a0, cubic=True).repeat(3)
    st.set_cell(st.cell.array @ np.diag([1.04, 1, 1]), scale_atoms=True)
    z2, d2 = shell_features(one_config_batch(st), r1)
    assert np.all(z2[:108] == 12) and np.all(d2[:108] > 0.01)
    sl = fcc111("Ni", size=(3, 3, 6), a=a0, vacuum=8.0)
    sl.pbc = True
    z3, _ = shell_features(one_config_batch(sl), r1)
    assert z3[:len(sl)].min() == 9 and z3[:len(sl)].max() == 12


def test_shell_reference_fcc_bcc_and_fallback():
    from ace_jax.fit.conformal import shell_reference
    from ase.neighborlist import neighbor_list
    for at, lo, hi in ((bulk("Ni", "fcc", a=3.65, cubic=True).repeat(4), 2.6, 3.6),
                       (bulk("Fe", "bcc", a=2.87, cubic=True).repeat(5), 2.9, 4.0)):
        at.rattle(0.03, seed=1)
        assert lo < shell_reference(neighbor_list("d", at, 6.0)) < hi
    flat = np.full(1000, 2.5) + np.random.default_rng(0).normal(0, 1e-3, 1000)
    assert shell_reference(flat) == pytest.approx(1.25 * 2.5, rel=0.02)


def test_assign_groups_extremes():
    from ace_jax.fit.conformal import assign_groups, n_groups
    edges = np.array([0.01, 0.02, 0.05])
    g = assign_groups(np.array([12, 12, 11, 3, 1]), np.array([0.0, 0.03, 0.9, np.nan, np.nan]), 12, edges)
    assert g.tolist() == [0, 4, 7, 7, 7] and n_groups(edges) == 8
    assert assign_groups(np.array([12, 3]), np.array([0.5, np.nan]), 12, np.array([])).tolist() == [0, 1]
    assert n_groups(np.array([])) == 2


def test_stratified_split_guarantees():
    from ace_jax.fit.conformal import config_strata, stratified_split
    assert config_strata([np.array([0, 4, 1]), np.array([], int), np.array([7])]).tolist() == [4, 0, 7]
    strata = np.r_[np.zeros(100, int), np.full(10, 6), np.full(2, 7), [5]]  # a lone config in stratum 5
    fit, val = stratified_split(strata, 0.2, seed=0)
    assert len(np.intersect1d(fit, val)) == 0 and len(fit) + len(val) == len(strata)
    for s in (0, 6, 7):
        m = np.flatnonzero(strata == s)
        assert np.isin(m, val).any() and np.isin(m, fit).any(), s
    assert abs(np.isin(np.flatnonzero(strata == 0), val).sum() - 20) <= 1
    f2, v2 = stratified_split(strata, 0.2, seed=0)
    assert np.array_equal(v2, val)                                              # deterministic in seed


def test_config_weighted_scales_against_brute_force():
    from ace_jax.fit.conformal import chi3_ppf, group_scales
    rng = np.random.default_rng(3)
    n_cfg = 60
    sizes = rng.integers(1, 40, n_cfg)                                          # very unequal atom counts
    cfg = np.repeat(np.arange(n_cfg), sizes)
    s = rng.chisquare(3, len(cfg)) ** 0.5 * (1 + 0.5 * (cfg % 3))
    g = np.zeros(len(cfg), int)
    t = group_scales(s, g, cfg, G=2, alpha=0.1, n_min=20)
    w = 1.0 / np.bincount(cfg)[cfg]
    assert t.lam_rms[0] == pytest.approx(np.sqrt(np.sum(w * s ** 2) / (3 * n_cfg))) and t.n_cfg[0] == n_cfg
    assert t.q[0] == pytest.approx(_wq(s, w, 0.1))
    assert t.r[0] == pytest.approx(t.q[0] / (t.lam_rms[0] * chi3_ppf(0.9)))
    assert chi3_ppf(0.9) == pytest.approx(2.50028, abs=1e-4)


def test_scales_merge_small_groups():
    from ace_jax.fit.conformal import GroupTable, group_scales
    rng = np.random.default_rng(4)
    cfg = np.r_[np.arange(400) // 10, 1000 + np.arange(30) // 10]               # 40 cfgs in g=0, 3 in g=2
    g = np.r_[np.zeros(400, int), np.full(30, 2)]
    src = np.r_[np.zeros(400, int), np.ones(30, int)]
    s = rng.chisquare(3, 430) ** 0.5
    t = group_scales(s, g, cfg, G=8, alpha=0.1, n_min=20, src=src)
    assert [2, 0] in t.merged and t.q[2] == t.q[0]                              # band 1 flag 0 -> band 0
    assert np.isfinite(t.q).all() and np.isfinite(t.lam_rms).all()             # empty groups served too
    assert t.n_cfg_val[0] == 40 and t.n_cfg_cal[2] == 3 and t.n_min == 20
    back = GroupTable.from_dict(t.to_dict())
    assert np.array_equal(back.q, t.q) and back.merged == t.merged and back.n_min == 20
```

- [ ] **Step 3: Run** `uv run pytest tests/test_conformal.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.conformal'`.

- [ ] **Step 4: Implement** `src/ace_jax/fit/conformal.py`

```python
"""Mondrian groups, stratified split and the two configuration-weighted per-group scales of the
ARD force sigma (math rev. 2)."""
import dataclasses

import numpy as np
from scipy.stats import chi


def chi3_ppf(p):
    return float(chi.ppf(p, 3))


def shell_reference(dists, bins=400):
    """r1: first minimum of the radial density after its first peak; 1.25 x the first peak if none."""
    d = np.asarray(dists, float)
    h, e = np.histogram(d, bins=bins, range=(0.0, d.max()))
    c = 0.5 * (e[1:] + e[:-1])
    g = np.convolve(h / np.maximum(c ** 2, 1e-12), np.ones(5) / 5, mode="same")
    p = int(np.argmax(g > 0.2 * g.max()))
    while p + 1 < len(g) and g[p + 1] >= g[p]:
        p += 1
    m = p
    while m + 1 < len(g) and g[m + 1] <= g[m]:
        m += 1
    return float(c[m] if (m + 1 < len(g) and g[m] < 0.5 * g[p]) else 1.25 * c[p])


def shell_features(batch, r1):
    """Per node: z = #{r_ij < r1}, d = std/mean of those r_ij (NaN if z < 2), from the batch's neighbour list."""
    r = np.linalg.norm(np.asarray(batch.rij), axis=-1)
    m = np.asarray(batch.nbr_mask) & (r < r1)
    z = m.sum(1)
    mean = np.where(m, r, 0.0).sum(1) / np.maximum(z, 1)
    var = np.where(m, (r - mean[:, None]) ** 2, 0.0).sum(1) / np.maximum(z, 1)
    return z.astype(np.int64), np.where(z >= 2, np.sqrt(var) / np.maximum(mean, 1e-12), np.nan)


def band_edges(d, qs=(0.5, 0.9, 0.99)):
    d = np.asarray(d, float)
    return np.quantile(d[np.isfinite(d)], qs)


def n_groups(edges):
    return 2 * (len(edges) + 1)


def assign_groups(z, d, z_star, edges):
    band = np.searchsorted(np.asarray(edges, float), np.nan_to_num(np.asarray(d, float), nan=np.inf), side="right")
    return (band * 2 + (np.asarray(z) != z_star)).astype(np.int64)


def config_strata(groups_per_cfg):
    """Stratum of each configuration: its most extreme group, max over its atoms of band*2 + flag (0 if empty)."""
    return np.array([int(np.max(g)) if len(g) else 0 for g in groups_per_cfg])


def stratified_split(strata, f, seed):
    """(fit_idx, val_idx): within each stratum a fraction f to val, with >= 1 config on each side.
    Strata with < 2 configs merge into the next less extreme populated stratum first."""
    s = np.asarray(strata).copy()
    for k in sorted(set(s.tolist()), reverse=True):
        lower = [j for j in sorted(set(s.tolist()), reverse=True) if j < k]
        if np.sum(s == k) < 2 and lower:
            s[s == k] = lower[0]
    rng = np.random.default_rng(seed)
    val = []
    for k in sorted(set(s.tolist())):
        idx = rng.permutation(np.flatnonzero(s == k))
        if len(idx) < 2:
            continue
        nv = min(max(1, int(round(f * len(idx)))), len(idx) - 1)
        val.extend(idx[:nv].tolist())
    val = np.sort(np.asarray(val, int))
    return np.setdiff1d(np.arange(len(s)), val), val


def _pooled_q(s, w, alpha):
    if len(s) == 0:
        return np.inf
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / (w.sum() + 1.0)
    k = int(np.searchsorted(cw, 1 - alpha - 1e-15))
    return float(s[o][k]) if k < len(s) else np.inf


@dataclasses.dataclass(frozen=True)
class GroupTable:
    alpha: float
    n_min: int
    lam_rms: np.ndarray
    q: np.ndarray
    r: np.ndarray
    n_cfg: np.ndarray
    n_cfg_val: np.ndarray
    n_cfg_cal: np.ndarray
    n_atoms: np.ndarray
    merged: list
    sources: list = dataclasses.field(default_factory=list)

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["alpha"]), int(d["n_min"]),
                   *(np.asarray(d[k], float) for k in ("lam_rms", "q", "r")),
                   *(np.asarray(d[k], int) for k in ("n_cfg", "n_cfg_val", "n_cfg_cal", "n_atoms")),
                   [list(m) for m in d["merged"]], list(d.get("sources", [])))


def group_scales(scores, groups, cfg, G, alpha, n_min, src=None):
    """Configuration-weighted per-group lam_rms and pooled-CDF q (atom weights 1/n_{c,g}).  Groups below
    n_min configurations take the nearest qualifying band's values (same z-flag, then the other flag,
    then all groups pooled).  src per atom: 0 = T_val, >= 1 = a calibrate set (composition counts only)."""
    s, g, c = np.asarray(scores, float), np.asarray(groups), np.asarray(cfg)
    src = np.zeros(len(s), int) if src is None else np.asarray(src)
    _, inv, cnt = np.unique(c * G + g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]

    def stats(m):
        nc = len(np.unique(c[m]))
        if nc == 0:
            return np.nan, np.inf, 0
        return float(np.sqrt(np.sum(w[m] * s[m] ** 2) / (3 * nc))), _pooled_q(s[m], w[m], alpha), nc

    lam, q, ncfg = np.full(G, np.nan), np.full(G, np.inf), np.zeros(G, int)
    nval, ncal, nat = np.zeros(G, int), np.zeros(G, int), np.zeros(G, int)
    for k in range(G):
        m = g == k
        lam[k], q[k], ncfg[k] = stats(m)
        nval[k] = len(np.unique(c[m & (src == 0)]))
        ncal[k] = len(np.unique(c[m & (src > 0)]))
        nat[k] = int(m.sum())
    ok = ncfg >= n_min
    lam_all, q_all, _ = stats(np.ones(len(s), bool))
    lam_s, q_s, merged = lam.copy(), q.copy(), []
    for k in np.flatnonzero(~ok):
        band, flag = divmod(int(k), 2)
        cand = [(abs(b - band), 0, b * 2 + flag) for b in range(G // 2) if ok[b * 2 + flag]]
        cand += [(abs(b - band), 1, b * 2 + 1 - flag) for b in range(G // 2) if ok[b * 2 + 1 - flag]]
        cand.sort(key=lambda t: (t[1], t[0]))
        k_src = cand[0][2] if cand else -1
        lam_s[k], q_s[k] = (lam[k_src], q[k_src]) if k_src >= 0 else (lam_all, q_all)
        merged.append([int(k), int(k_src)])
    r = q_s / (lam_s * chi3_ppf(1 - alpha))
    return GroupTable(float(alpha), int(n_min), lam_s, q_s, r, ncfg, nval, ncal, nat, merged)
```

- [ ] **Step 5: Run** `uv run pytest tests/test_conformal.py -q`. Expected: 6 passed. If `shell_reference` misses bcc (whose first two shells are close together), print the peak and minimum bins before changing any threshold.
- [ ] **Step 6: Commit.** First run `uv run pytest -q && uv run ruff check`. Then:

```bash
git add src/ace_jax/fit/conformal.py tests/test_conformal.py tests/conftest.py
git commit -m "feat(conformal): shell features, groups, stratified split, configuration-weighted per-group scales"
```

### Task 2: `fit/clusters.py` — sandwich cluster layout

**Files:**
- Create: `src/ace_jax/fit/clusters.py`, `tests/test_jackknife.py`

**Interfaces:**
- Consumes: `Config`, `build_dataset` (configurations are batched in order, `configs_per_batch` per batch; padded configurations have `cfg_mask == False`, and padded nodes have `node_cfg == C`).
- Produces:
  - `config_blocks(config, ell) -> ndarray | None`: a block id per atom, or None when the configuration is one cluster.
  - `row_clusters(ds, configs, ell) -> (list[dict], K)`: per batch, `{"E": (C,), "F": (Ncap,), "V": (C,)}` int64 cluster ids. Each force row of atom n shares `F[n]`, and each of a configuration's 6 virial rows shares `V[c]`. Padded entries are −1.
    - Ids are dense 0..K−1, assigned in Dataset order of the live configurations. A small configuration takes one id. A large one takes one id per block, then one for its E+V rows.
    - With `ell == inf`, `configs` may be None.

- [ ] **Step 1: Write the failing tests** in `tests/test_jackknife.py`

```python
import numpy as np
from ase.build import bulk


def _cfg(at):
    from ace_jax.fit.data import Config
    return Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                  0.0, np.zeros((len(at), 3)), np.zeros((3, 3)), 1.0, 1.0, 1.0)


def test_config_blocks_small_and_large():
    from ace_jax.fit.clusters import config_blocks
    ell = 3 * 6.25
    assert config_blocks(_cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(4)), ell) is None   # 14.6 A
    big = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4))                              # 43.8 A in x
    b = config_blocks(_cfg(big), ell)
    frac = big.get_scaled_positions(wrap=True)[:, 0]
    assert len(np.unique(b)) == 2                                                               # floor(43.8/18.75)
    assert len({(int(fx * 2), int(bi)) for fx, bi in zip(frac, b)}) == 2                        # id = f(floor(2 fx))


def test_config_blocks_nonperiodic_uses_bounding_box():
    from ace_jax.fit.clusters import config_blocks
    at = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((16, 16, 2))                              # 58.4 A in x, y
    at.pbc = (False, False, True)
    at.center(vacuum=10.0, axis=(0, 1))
    assert len(np.unique(config_blocks(_cfg(at), 3 * 6.25))) == 9                               # 3 x 3 x 1


def test_row_clusters_ids():
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.data import build_dataset
    small = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(2))
    large = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4)))
    ds = build_dataset([small, large], {"elements": [28], "rcut": 6.25}, np.zeros(1), 2)
    rc, K = row_clusters(ds, [small, large], 3 * 6.25)
    b, ns, nl = rc[0], len(small.numbers), len(large.numbers)
    assert b["E"][0] == b["V"][0] == 0 and np.all(b["F"][:ns] == 0)                            # small: one id
    fl = b["F"][ns:ns + nl]
    assert set(fl.tolist()) == {1, 2} and b["E"][1] == b["V"][1] == 3 and K == 4               # blocks, then E/V
    assert np.all(b["F"][ns + nl:] == -1)
    rc_inf, K_inf = row_clusters(ds, None, float("inf"))
    assert K_inf == 2 and np.all(rc_inf[0]["F"][ns:ns + nl] == 1)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_jackknife.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.clusters'`.
- [ ] **Step 3: Implement** `src/ace_jax/fit/clusters.py`

```python
"""Sandwich cluster layout (math rev. 2): a configuration is one cluster unless it spans > ell in some
direction; then its force rows split into spatial blocks of side ~ell and its E+V rows form one more."""
import numpy as np


def _widths_and_frac(cfg):
    cell, pos, pbc = np.asarray(cfg.cell, float), np.asarray(cfg.positions, float), np.asarray(cfg.pbc, bool)
    w, frac = np.zeros(3), np.zeros((len(pos), 3))
    vol = abs(np.linalg.det(cell)) if np.all(np.linalg.norm(cell, axis=1) > 0) else 0.0
    inv = np.linalg.inv(cell) if vol > 0 else None
    for a in range(3):
        nrm = np.cross(cell[(a + 1) % 3], cell[(a + 2) % 3])
        nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
        if pbc[a] and inv is not None:
            w[a] = vol / max(np.linalg.norm(np.cross(cell[(a + 1) % 3], cell[(a + 2) % 3])), 1e-12)
            frac[:, a] = (pos @ inv)[:, a] % 1.0
        else:
            p = pos @ nrm
            w[a] = p.max() - p.min()
            frac[:, a] = (p - p.min()) / max(w[a], 1e-12)
    return w, frac


def config_blocks(cfg, ell):
    if not np.isfinite(ell):
        return None
    w, frac = _widths_and_frac(cfg)
    n = np.maximum(1, np.floor(w / ell)).astype(int)
    if np.all(n == 1):
        return None
    idx = np.minimum(np.floor(frac * n).astype(int), n - 1)
    _, ids = np.unique(np.ravel_multi_index(idx.T, n), return_inverse=True)
    return ids.astype(np.int64)


def row_clusters(ds, configs, ell):
    out, k = [], 0
    gc = 0
    for i in range(ds.n_batches):
        cm = np.asarray(ds.cfg_mask[i])
        ncfg = np.asarray(ds.node_cfg[i])
        C, N = len(cm), len(ncfg)
        E, F, V = np.full(C, -1, np.int64), np.full(N, -1, np.int64), np.full(C, -1, np.int64)
        for c in np.flatnonzero(cm):
            nodes = np.flatnonzero(ncfg == c)
            blk = config_blocks(configs[gc], ell) if np.isfinite(ell) else None
            if blk is None:
                E[c] = V[c] = k
                F[nodes] = k
                k += 1
            else:
                F[nodes] = k + blk
                k += int(blk.max()) + 1
                E[c] = V[c] = k
                k += 1
            gc += 1
        out.append({"E": E, "F": F, "V": V})
    return out, k
```

  Nodes are config-major within a batch, in atom order, so `nodes` lines up with `config_blocks`' atom order. Check this against `fit/data.py:_batch` (it appends each configuration's atoms in order); if not, map through the batch's node order.

- [ ] **Step 4: Run** `uv run pytest tests/test_jackknife.py -q`. Expected: 3 passed. Then `uv run pytest -q && uv run ruff check`.
- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/clusters.py tests/test_jackknife.py
git commit -m "feat(clusters): sandwich cluster layout -- whole configs or spatial blocks + an E/V cluster"
```

### Task 3: `fit/jackknife.py` — PRESS scores, centred factor R, per-atom V

**Files:**
- Create: `src/ace_jax/fit/jackknife.py`
- Modify: `tests/conftest.py` (move `_sandwich_setup` from `tests/test_ard.py` into a fixture `ard_setup`, and update that file's 3 callers to use the fixture), `tests/test_jackknife.py`

**Interfaces:**
- Consumes:
  - `row_clusters` (Task 2);
  - `ARDPosterior.mean, .chol (lower Cholesky of S), .dinv`;
  - `linear_rows(prob.model, prob.cfg, batch)[0]` with fields `.E (C,L)`, `.F (Ncap,3,L)`, `.V (C,6,L)`, exactly as `sandwich_scores` uses it (`src/ace_jax/fit/ard.py:273`);
  - `ev.sigmas(h)` → (σ_E, σ_F, σ_V).
- Produces:
  - `press_scores(post, prob, ds, clusters, K, sig, mode="exact") -> (G (L,K) ndarray, lev (K,) ndarray)`: unscaled g̃_k and λ_max(H_kk). Modes: "exact" (switches to push-through automatically when n_k > L), "pushthrough" (forced), "block".
  - `shape_factor(post, G, tau=1.0) -> R`, a jax float64 (L, r) array.
  - `atom_shape(R, dinv, Frows) -> V`, an (N,3,3) ndarray.

**Algebra:** S = D⁻¹AD⁻¹ = LLᵀ, and Ps = Ψ_k D⁻¹ (columns × dinv).
- H_kk = WᵀW, with W = L⁻¹Psᵀ.
- Push-through: S_k = S − PsᵀPs, and x = S_k⁻¹Psᵀρ. Then A⁻¹g̃_k = A_(−k)⁻¹Ψ_kᵀρ = dinv ⊙ x, so g̃_k = A(dinv ⊙ x) = (S x) / dinv.

- [ ] **Step 1: Move the fixture.** Add to `tests/conftest.py` (with `_orders` moved alongside it if `_sandwich_setup` uses it):

```python
@pytest.fixture
def ard_setup(tiny_linear_problem):
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
```

  Run `uv run pytest tests/test_ard.py -q`. Expected: everything passes as before.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_jackknife.py`)

```python
import jax
import jax.numpy as jnp
import pytest


def _rows(prob, ds, sig, rc=None):
    """Dense whitened rows Psi (n, L) and targets y~ (n,), with each row's label: the global live-config
    index, or (rc given) the row's cluster id from row_clusters."""
    from ace_jax.fit.rows import linear_rows
    P, Y, cf = [], [], []
    g = 0
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = linear_rows(prob.model, prob.cfg, b)[0]
        for c in np.flatnonzero(np.asarray(b.cfg_mask)):
            lab = (lambda kind, j: rc[i][kind][j]) if rc is not None else (lambda kind, j, g=g: g)
            w = float(b.w_E[c]) / sig[0]
            P.append(np.asarray(r.E[c]) * w); Y.append(float(b.y_E[c]) * w); cf.append(lab("E", c))
            for n in np.flatnonzero(np.asarray(b.node_cfg) == c):
                w = float(b.w_F[n]) / sig[1]
                for a in range(3):
                    P.append(np.asarray(r.F[n, a]) * w); Y.append(float(b.y_F[n, a]) * w); cf.append(lab("F", n))
            w = float(b.w_V[c]) / sig[2]
            for v in range(6):
                P.append(np.asarray(r.V[c, v]) * w); Y.append(float(b.y_V[c, v]) * w); cf.append(lab("V", c))
            g += 1
    return np.array(P), np.array(Y), np.array(cf)


def _A(post):
    D = 1.0 / np.asarray(post.dinv)
    S = np.asarray(post.chol) @ np.asarray(post.chol).T
    return S * D[:, None] * D[None, :]


def test_press_equals_exact_deletion(ard_setup):
    """c - c_(-k) = A^-1 g~_k for every whole-configuration cluster, h fixed."""
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    sig = ev.sigmas(h)
    Psi, y, cf = _rows(prob, ds, sig)
    A = _A(post)
    c = np.linalg.solve(A, Psi.T @ y)
    np.testing.assert_allclose(c, post.mean, rtol=1e-7, atol=1e-10 * np.abs(c).max())
    rc, K = row_clusters(ds, None, float("inf"))
    G, _ = press_scores(post, prob, ds, rc, K, sig)
    assert K == cf.max() + 1
    for k in range(K):
        m = cf == k
        ck = np.linalg.solve(A - Psi[m].T @ Psi[m], Psi[~m].T @ y[~m])
        np.testing.assert_allclose(c - ck, np.linalg.solve(A, G[:, k]), rtol=1e-6, atol=1e-9 * np.abs(c).max())


def test_push_through_matches_exact(ard_setup):
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    Ge, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="exact")
    Gp, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="pushthrough")
    np.testing.assert_allclose(Gp, Ge, rtol=1e-7, atol=1e-10 * np.abs(Ge).max())


def test_press_high_leverage_cluster(ard_setup):
    """Prior precision > 0 keeps every lambda_max(H_kk) < 1; a near-1 cluster stays finite and exact."""
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    G, lev = press_scores(post, prob, ds, rc, K, ev.sigmas(h))
    assert np.all(lev >= 0) and np.all(lev < 1) and np.isfinite(G).all()
    hh = np.array(h, float)
    hh[1] -= 3.0                                                                # sigma_F / e^3: data dominate
    post2 = post._replace(**_posterior_at(ev, hh, post))
    G2, lev2 = press_scores(post2, prob, ds, rc, K, ev.sigmas(hh))
    assert lev2.max() > lev.max() and np.all(lev2 < 1) and np.isfinite(G2).all()


def _posterior_at(ev, h, post):
    from ace_jax.fit.ard import ard_posterior
    p = ard_posterior(ev, h, 2.0, post.meta)
    return {"mean": p.mean, "chol": p.chol, "dinv": p.dinv}
```

  Large cells: one rattled Si supercell, wide enough for 2 blocks at ell = 1.2 r_cut, with random labels. The posterior is built on that Dataset alone (h = `ev.h0(theta)`), and the per-row cluster id comes from `rc` via `_rows(..., rc=rc)`.

```python
def test_press_spatial_and_ev_clusters(tiny_linear_problem):
    from ace_jax.fit.ard import ARDEvidence, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.data import Config, build_dataset
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.jackknife import press_scores
    from ase.build import bulk as _bulk
    prob, _ = tiny_linear_problem
    rcut = float(prob.cfg.rcut)
    ell = 1.2 * rcut
    n = int(np.ceil(2 * ell / 5.43)) + 1
    at = _bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, 1, 1))
    at.rattle(0.05, seed=0)
    rng = np.random.default_rng(0)
    cfg = Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                 float(rng.normal()), rng.normal(size=(len(at), 3)), rng.normal(size=(3, 3)), 1.0, 1.0, 1.0)
    ds = build_dataset([cfg], {"elements": [14], "rcut": rcut}, np.zeros(1), 1)
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": rcut, "elements": [14]}
    theta = default_prior(2.35).mu
    ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                     body_order_columns(meta, prob.cfg))
    h = ev.h0(theta)
    post = ard_posterior(ev, h, 2.0, meta)
    rc, K = row_clusters(ds, [cfg], ell)
    assert K == 3
    Psi, y, kid = _rows(prob, ds, ev.sigmas(h), rc=rc)
    A = _A(post)
    c = np.linalg.solve(A, Psi.T @ y)
    G, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h))
    for k in range(K):
        m = kid == k
        ck = np.linalg.solve(A - Psi[m].T @ Psi[m], Psi[~m].T @ y[~m])
        np.testing.assert_allclose(c - ck, np.linalg.solve(A, G[:, k]), rtol=1e-6, atol=1e-9 * np.abs(c).max())
```

  `_orders` is imported from `conftest` (moved there in Step 1).

```python
def test_centring_and_factor(ard_setup):
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import atom_shape, press_scores, shape_factor
    from ace_jax.fit.rows import linear_rows
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    G, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h))
    R = np.asarray(shape_factor(post, G))
    S = np.asarray(post.chol) @ np.asarray(post.chol).T
    Qt = np.linalg.solve(S, (G - G.mean(1, keepdims=True)) * np.asarray(post.dinv)[:, None])
    M = Qt @ Qt.T
    np.testing.assert_allclose(R @ R.T, M, rtol=1e-8, atol=1e-12 * np.abs(M).max())
    Rt = np.asarray(shape_factor(post, G, tau=0.9))
    assert Rt.shape[1] <= min(K, len(post.mean)) and np.trace(Rt @ Rt.T) >= 0.9 * np.trace(M) * (1 - 1e-12)
    Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    V = atom_shape(jnp.asarray(R), post.dinv, Fr)
    u = Fr * np.asarray(post.dinv)[None, None, :]
    Vref = np.einsum("nal,lm,nbm->nab", u, M, u)
    np.testing.assert_allclose(V, Vref, rtol=1e-8, atol=1e-14 * max(np.abs(Vref).max(), 1e-300))
    np.testing.assert_allclose(V, np.swapaxes(V, 1, 2))
```

- [ ] **Step 3: Run** `uv run pytest tests/test_jackknife.py -q`. Expected: the new tests FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.jackknife'`.
- [ ] **Step 4: Implement** `src/ace_jax/fit/jackknife.py`

```python
"""Exact centred delete-one-cluster (PRESS / CR3) jackknife shape (math rev. 2)."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, solve_triangular


def _batch_rows(prob, bt, ids, sig):
    """Whitened rows Psi (n, L), targets y~ (n,), cluster id (n,), row-block id (n,) of one batch's live rows.
    Row blocks (for mode="block"): an energy row, an atom's 3 force rows, a config's 6 virial rows."""
    from .rows import linear_rows
    r = linear_rows(prob.model, prob.cfg, bt)[0]
    C, L = r.E.shape
    N = r.F.shape[0]
    P = np.concatenate([np.asarray(r.E), np.asarray(r.F).reshape(-1, L), np.asarray(r.V).reshape(-1, L)])
    y = np.concatenate([np.asarray(bt.y_E), np.asarray(bt.y_F).reshape(-1), np.asarray(bt.y_V).reshape(-1)])
    w = np.concatenate([np.asarray(bt.w_E) / sig[0], np.repeat(np.asarray(bt.w_F), 3) / sig[1],
                        np.repeat(np.asarray(bt.w_V), 6) / sig[2]])
    k = np.concatenate([ids["E"], np.repeat(ids["F"], 3), np.repeat(ids["V"], 6)])
    blk = np.concatenate([np.arange(C), C + np.repeat(np.arange(N), 3), C + N + np.repeat(np.arange(C), 6)])
    live = (k >= 0) & (w > 0)
    return P[live] * w[live, None], y[live] * w[live], k[live], blk[live]


def _collect(prob, ds, clusters, sig):
    P, y, k, b = [], [], [], []
    off = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        p, q, kk, bb = _batch_rows(prob, bt, clusters[i], sig)
        P.append(p); y.append(q); k.append(kk); b.append(bb + off)
        off += int(bb.max()) + 1 if len(bb) else 0
    P, y, k, b = map(np.concatenate, (P, y, k, b))
    o = np.argsort(k, kind="stable")
    return P[o], y[o], k[o], b[o]


def press_scores(post, prob, ds, clusters, K, sig, mode="exact"):
    """G (L, K) with g~_k = Psi_k^T (I - H_kk)^-1 rho_k (rho_k = y~_k - Psi_k c), and lambda_max(H_kk)."""
    Lc = np.asarray(post.chol, np.float64)
    dinv = np.asarray(post.dinv)
    c = np.asarray(post.mean)
    L = len(c)
    S = Lc @ Lc.T
    P, y, k, blk = _collect(prob, ds, clusters, sig)
    starts = np.searchsorted(k, np.arange(K + 1))
    G, lev = np.zeros((L, K)), np.zeros(K)
    for kk in range(K):
        sl = slice(starts[kk], starts[kk + 1])
        Pk, rho, bk = P[sl], y[sl] - P[sl] @ c, blk[sl]
        if len(Pk) == 0:
            continue
        Ps = Pk * dinv[None, :]
        W = np.asarray(solve_triangular(jnp.asarray(Lc), jnp.asarray(Ps.T), lower=True))
        if len(Pk) <= L:
            H = W.T @ W
            lev[kk] = float(np.linalg.eigvalsh(H).max())
        else:
            lev[kk] = float(np.linalg.eigvalsh(W @ W.T).max())          # same nonzero spectrum, L x L
        if mode == "pushthrough" or (mode == "exact" and len(Pk) > L):
            x = np.linalg.solve(S - Ps.T @ Ps, Ps.T @ rho)
            G[:, kk] = (S @ x) / dinv
        elif mode == "block":
            H = W.T @ W
            z = np.empty_like(rho)
            for b in np.unique(bk):
                m = bk == b
                z[m] = np.linalg.solve(np.eye(m.sum()) - H[np.ix_(m, m)], rho[m])
            G[:, kk] = Pk.T @ z
        else:
            G[:, kk] = Pk.T @ np.linalg.solve(np.eye(len(Pk)) - W.T @ W, rho)
    return G, lev


def shape_factor(post, G, tau=1.0):
    """R with R R^T = Q~ Q~^T, Q~ = S^-1 D^-1 (G - g-bar).  R = Q~ when K <= L and tau = 1, else
    U_r Sigma_r of the thin SVD with r the smallest rank holding >= tau of sum sigma^2."""
    Lc = jnp.asarray(post.chol, jnp.float64)
    G = jnp.asarray(G, jnp.float64)
    Qt = cho_solve((Lc, True), (G - G.mean(1, keepdims=True)) * jnp.asarray(post.dinv)[:, None])
    if Qt.shape[1] <= Qt.shape[0] and tau >= 1.0:
        return Qt
    U, s, _ = jnp.linalg.svd(Qt, full_matrices=False)
    e = np.cumsum(np.asarray(s) ** 2)
    r = int(np.searchsorted(e / e[-1], min(tau, 1.0) - 1e-12) + 1)
    return U[:, :r] * s[None, :r]


def atom_shape(R, dinv, Frows):
    """V (N, 3, 3): V_ab = (R^T u_a) . (R^T u_b), u_a = D^-1 phi_a^T, from force rows (N, 3, L)."""
    U = jnp.asarray(Frows) * jnp.asarray(dinv)[None, None, :]
    Pr = U @ jnp.asarray(R)
    return np.asarray(jnp.einsum("nar,nbr->nab", Pr, Pr))
```

  **Performance (keep it simple until the bench says otherwise):** the loop above is the reference. If `press_scores` takes more than ~10 % of the ARD stage on bench365 (log it), batch the n_k ≤ L solves with `jax.vmap` over power-of-two n_k buckets. Padded rows carry zero ψ and ρ, so the result does not change. Keep the loop as the tested reference.

- [ ] **Step 5: Run** `uv run pytest tests/test_jackknife.py -q`. Expected: all pass. Then `uv run pytest -q && uv run ruff check`.
- [ ] **Step 6: Commit**

```bash
git add src/ace_jax/fit/jackknife.py tests/test_jackknife.py tests/conftest.py tests/test_ard.py
git commit -m "feat(jackknife): exact centred PRESS cluster jackknife shape, factor R, per-atom V"
```

### Task 4: `ARDPosterior` schema 3 — served quantities

**Files:**
- Modify: `src/ace_jax/fit/ard.py` (`ARDPosterior`, `SCHEMA`, `save`, `load`), `tests/test_ard.py`

**Interfaces:**
- Consumes: `jackknife.atom_shape`, and `conformal.{GroupTable, assign_groups, shell_features, chi3_ppf}`.
- Produces:
  - **New `ARDPosterior` fields** after `lam`: `R=None`, `force_shape="iso"`, `eps=1e-3`, `group_consts=None` (dict `{"r1", "z_star", "edges"}`), `group_table=None` (a `GroupTable.to_dict()` dict), `cal=None` (dict of arrays `scores` f32, `groups` i8, `cfg` i64, `src` i8), `support=None` (dict; Task 6).
  - **Methods:**
    - `groups_of(batch) -> (Ncap,) int64`;
    - `atom_shape(Frows) -> (N,3,3)`;
    - `forces_std(Frows, groups=None) -> (N,)`;
    - `forces_cov(Frows, groups) -> (N,3,3)`;
    - `forces_q(Frows, groups) -> (N,)`;
    - `forces_q_mahal(groups) -> (N,)`.
  - `SCHEMA = 3`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ard.py`)

```python
def _table(G=2):
    from ace_jax.fit.conformal import GroupTable
    return GroupTable(alpha=0.1, n_min=20, lam_rms=np.array([2.0, 3.0]), q=np.array([5.0, 9.0]),
                      r=np.array([1.0, 1.2]), n_cfg=np.array([30, 30]), n_cfg_val=np.array([30, 30]),
                      n_cfg_cal=np.array([0, 0]), n_atoms=np.array([100, 100]), merged=[])


def test_schema3_served_quantities(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores, shape_factor
    from ace_jax.fit.rows import linear_rows
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    R = shape_factor(post, press_scores(post, prob, ds, rc, K, ev.sigmas(h))[0])
    tab = _table()
    p = post._replace(R=R, group_consts={"r1": 3.0, "z_star": 4, "edges": []}, group_table=tab.to_dict())
    Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    g = np.arange(Fr.shape[0]) % 2
    V = p.atom_shape(Fr)
    v = np.trace(V, axis1=1, axis2=2)
    np.testing.assert_allclose(p.forces_std(Fr, g), tab.lam_rms[g] * np.sqrt(v), rtol=1e-12)
    np.testing.assert_allclose(p.forces_cov(Fr, g), tab.lam_rms[g, None, None] ** 2 * V, rtol=1e-12)
    np.testing.assert_allclose(np.trace(p.forces_cov(Fr, g), axis1=1, axis2=2), p.forces_std(Fr, g) ** 2,
                               rtol=1e-12)
    np.testing.assert_allclose(p.forces_q(Fr, g), tab.q[g] * np.sqrt(v / 3), rtol=1e-12)
    pa = p._replace(force_shape="aniso")
    lam_max = np.linalg.eigvalsh(V + p.eps * (v / 3)[:, None, None] * np.eye(3)).max(1)
    np.testing.assert_allclose(pa.forces_q(Fr, g), tab.q[g] * np.sqrt(lam_max), rtol=1e-10)
    np.testing.assert_allclose(pa.forces_q_mahal(g), tab.q[g])
    p.save(tmp_path / "p.npz")
    back = ARDPosterior.load(tmp_path / "p.npz")
    np.testing.assert_allclose(back.forces_std(Fr, g), p.forces_std(Fr, g), rtol=1e-3)   # R stored float32
    assert back.group_table["q"] == tab.q.tolist() and back.force_shape == "iso"


def test_schema2_serves_scalar(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, sandwich_factor, sandwich_scores
    from ace_jax.fit.rows import linear_rows
    prob, ds, ev, h, post = ard_setup
    old = post._replace(Q=sandwich_factor(post, sandwich_scores(post, prob, ds, ev.sigmas(h))), lam=2.0)
    old.save(tmp_path / "p.npz")
    z = dict(np.load(tmp_path / "p.npz"))
    z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("R", "group_", "cal_", "support", "force_shape", "eps"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    p2 = ARDPosterior.load(tmp_path / "p2.npz")
    Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    np.testing.assert_allclose(p2.forces_std(Fr), old.forces_std(Fr), rtol=1e-3)
    for f in (lambda: p2.forces_q(Fr, None), lambda: p2.forces_cov(Fr, None), lambda: p2.forces_q_mahal(None)):
        with pytest.raises(ValueError, match="refit with --uq ard"):
            f()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ard.py -q -k "schema3 or schema2"`. Expected: FAIL with `TypeError: ... unexpected keyword argument 'R'` from `_replace`.
- [ ] **Step 3: Implement** in `src/ace_jax/fit/ard.py`:

```python
_NEED3 = "this posterior predates schema 3; refit with --uq ard to get forces_q/forces_cov/forces_group"

    # --- inside class ARDPosterior (after the existing fields Q=None, lam=1.0) ---
    R: object = None
    force_shape: str = "iso"
    eps: float = 1e-3
    group_consts: dict | None = None
    group_table: dict | None = None
    cal: dict | None = None
    support: dict | None = None

    def _tab(self):
        if self.group_table is None:
            raise ValueError(_NEED3)
        return self.group_table

    def groups_of(self, batch):
        from .conformal import assign_groups, shell_features
        if self.group_consts is None:
            raise ValueError(_NEED3)
        gc = self.group_consts
        z, d = shell_features(batch, gc["r1"])
        return assign_groups(z, d, gc["z_star"], np.asarray(gc["edges"], float))

    def atom_shape(self, Frows):
        from .jackknife import atom_shape
        if self.R is not None:
            return atom_shape(self.R, self.dinv, Frows)
        if self.Q is not None:                                   # schema 2: uncentred sandwich factor
            return atom_shape(self.Q, self.dinv, Frows)
        U = np.asarray(Frows) * np.asarray(self.dinv)[None, None, :]          # kappa: V = u^T S^-1 u
        W = np.asarray(solve_triangular(jnp.asarray(self.chol), jnp.asarray(U.reshape(-1, U.shape[-1]).T),
                                        lower=True)).T.reshape(U.shape[0], 3, -1)
        return np.einsum("nar,nbr->nab", W, W)

    def forces_std(self, Frows, groups=None):
        if self.group_table is not None and groups is not None:
            v = np.trace(self.atom_shape(Frows), axis1=1, axis2=2)
            return np.asarray(self.group_table["lam_rms"])[groups] * np.sqrt(v)
        ...  # the existing schema-1/2 body, unchanged

    def forces_cov(self, Frows, groups):
        lam = np.asarray(self._tab()["lam_rms"])[groups]
        return lam[:, None, None] ** 2 * self.atom_shape(Frows)

    def forces_q(self, Frows, groups):
        q = np.asarray(self._tab()["q"])[groups]
        V = self.atom_shape(Frows)
        v = np.trace(V, axis1=1, axis2=2)
        if self.force_shape == "aniso":
            lm = np.linalg.eigvalsh(V + self.eps * (v / 3)[:, None, None] * np.eye(3)).max(1)
            return q * np.sqrt(lm)
        return q * np.sqrt(v / 3)

    def forces_q_mahal(self, groups):
        return np.asarray(self._tab()["q"])[groups]
```

  `forces_q_mahal` with `groups=None` in the schema-2 test must raise through `_tab()` before indexing, which the order above ensures.

  **`save`:** in addition to what it writes now, store:
  - `schema=3`;
  - `R` as float32 when set;
  - `force_shape` (str array), `eps`;
  - `group_consts_json` and `group_table_json` (`np.frombuffer(json.dumps(d).encode(), np.uint8)`);
  - `cal_scores` (f32), `cal_groups` (i8), `cal_cfg` (i64), `cal_src` (i8);
  - `support_*` arrays (Task 6 fills these).

  **`load`:** accept schema ∈ {1, 2, 3}. Fields that are absent default as above. Decode the JSON with `json.loads(bytes(z[k]).decode())`. Build `R` as a jax float64 array, as `Q` is.

  The `forces_std` signature keeps `Frows` first, so existing callers still work.

- [ ] **Step 4: Run** `uv run pytest tests/test_ard.py -q`. Expected: all pass, including the existing schema-1/2 tests. Then `uv run pytest -q && uv run ruff check`.
- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/ard.py tests/test_ard.py
git commit -m "feat(ard): schema 3 -- R factor, per-group tables, forces_std/cov/q/q_mahal"
```

### Task 5: `run_ard_stage` orchestration, `predict_ard`, config

**Files:**
- Modify: `src/ace_jax/fit/ard.py` (`run_ard_stage`, `_val_errors`, `predict_ard`), `src/ace_jax/fit/pipeline/config.py`
- Test: `tests/test_ard.py`, `tests/test_ard_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–4. `data.train` (the list of `Config`), `data.ds_train` / `data.ds_test`, `build_dataset`, `ard_posterior`, `ard_statistics`, `ARDEvidence`, `kappa_closed_form`, and the existing `sandwich_scores` / `sandwich_factor` (legacy variant).
- Produces:
  - `run_ard_stage(...)` returns as now, with `report` keys `shape`, `groups`, `split` and `transfer` added.
  - `predict_ard(post, prob, ds).F_var = lam_rms_g² · diag V` for schema-3 posteriors.
  - The `FitConfig` fields from Global Constraints.

- [ ] **Step 1: Write the failing tests.** In `tests/test_ard.py`, `_pipe_cfg` gains `ard_n_min=1` (the tiny fixture has about 30 configurations). Append:

```python
def _stage(**kw):
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg(ard_variance="sandwich", **kw).validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        pred = ard.predict_ard(res.posterior, b.prob, d.ds_test)
    return d, res, pred


def test_stage_default_press_shape_and_group_scales():
    from ace_jax.fit.conformal import chi3_ppf
    d, res, pred = _stage()
    post, rep = res.posterior, res.report
    assert post.R is not None and post.group_table is not None and post.cal is not None
    assert set(rep) >= {"shape", "groups", "split", "transfer"}
    assert rep["split"]["n_val"] + rep["split"]["n_fit"] == len(d.train) and rep["split"]["n_val"] >= 1
    assert rep["transfer"]["score_source"] == "fit"
    assert rep["shape"]["K_fit"] < rep["shape"]["K"]                 # T_val scores used P_fit's own shape
    assert 0 <= rep["shape"]["lev_max"] < 1
    t = post.group_table
    np.testing.assert_allclose(np.asarray(t["r"]),
                               np.asarray(t["q"]) / (np.asarray(t["lam_rms"]) * chi3_ppf(0.9)), rtol=1e-10)
    assert len(post.cal["scores"]) == sum(t["n_atoms"])
    assert np.isfinite(pred.F_var).all() and np.all(pred.F_var >= 0)


def test_stage_aniso_variant():
    _, res, _ = _stage(ard_force_shape="aniso")
    assert res.posterior.force_shape == "aniso" and np.isfinite(res.posterior.group_table["q"]).all()


def test_stage_legacy_ablation_variant():
    _, res, _ = _stage(_shape_variant="legacy", _score_source="mixed")
    post = res.posterior
    assert post.R is None and post.Q is not None and res.report["transfer"]["score_source"] == "mixed"
    assert np.isfinite(post.group_table["q"]).all()


def test_stage_groups_none_is_two_groups():
    _, res, _ = _stage(ard_groups="none")
    assert len(res.posterior.group_table["q"]) == 2 and res.posterior.group_consts["edges"] == []
```

  In `tests/test_ard_pipeline.py::test_config_validates_ard`, append:

```python
    from ace_jax.fit.pipeline.config import FitConfig
    c = FitConfig(uq="ard").validate()
    assert (c.ard_force_shape, c.ard_coverage, c.ard_groups, c.ard_cluster_size, c.ard_press, c.ard_n_min,
            c.ard_support, c._shape_variant, c._score_source) == ("iso", 0.9, "distortion", 3.0, "exact", 20,
                                                                  True, "press", "fit")
    for field, bad in (("ard_force_shape", "x"), ("ard_groups", "x"), ("ard_press", "x"),
                       ("_shape_variant", "x"), ("_score_source", "x"), ("ard_coverage", 1.0),
                       ("ard_n_min", 0), ("ard_cluster_size", 0.0)):
        with pytest.raises(ValueError, match=field):
            FitConfig(uq="ard", **{field: bad}).validate()
```

  Adjust `FitConfig(uq="ard")` to whatever the existing test uses to construct a valid ARD config.

- [ ] **Step 2: Run** `uv run pytest tests/test_ard.py tests/test_ard_pipeline.py -q`. Expected: the new tests FAIL (`TypeError: unexpected keyword argument 'ard_n_min'`, or similar).
- [ ] **Step 3: Implement.**
  - **`config.py`:** add the fields and validate them. Enums are checked by membership; `0 < ard_coverage < 1`; `ard_n_min >= 1`; `ard_cluster_size > 0` (inf allowed); `0 < ard_shape_tau <= 1`. Every message names the field.
  - **`run_ard_stage`**, replacing the uniform permutation split:

```python
    from .clusters import row_clusters
    from .conformal import (assign_groups, band_edges, config_strata, group_scales, n_groups,
                            shell_features, shell_reference, stratified_split)
    from .jackknife import press_scores, shape_factor

    # 1. groups and strata from T (training atoms only, all of them for the edges)
    feats = _train_shell_features(data.ds_train, prob)            # per batch: dists sample, z, d, node_cfg
    r1 = shell_reference(feats["dists"])
    z_all, d_all, cfg_all = feats_at(r1)                            # one pass over ds_train
    z_star = int(np.bincount(z_all).argmax())
    edges = band_edges(d_all) if cfg.ard_groups == "distortion" else np.array([])
    G = n_groups(edges)
    g_all = assign_groups(z_all, d_all, z_star, edges)
    strata = config_strata([g_all[cfg_all == c] for c in range(len(data.train))])
    fit_, val = stratified_split(strata, cfg.ard_val_frac, cfg.seed)
```

  `_train_shell_features` loops over `data.ds_train` batches. Per batch it collects the live (`node_mask`) neighbour distances `|rij|` (up to 200 batches sampled, for r₁) and, after r₁ is known, `shell_features(batch, r1)` and each live node's global config index (`batch_idx * C + node_cfg`, mapped to the live config counter). Write it as two small private helpers in `ard.py`, `_shell_dists(ds, max_batches=200)` and `_shell_table(ds, r1)`. The latter returns `(z, d, cfg_index, batch_idx, node_idx)` for every live node.

  Step 2 continues:

```python
    # 2. P_fit and its own shape
    ds_fit = build_dataset([data.train[i] for i in fit_], meta, E0, cpb)
    ds_val = build_dataset([data.train[i] for i in val], meta, E0, cpb)
    ev_fit = ARDEvidence(ard_statistics(theta, prob, ds_fit, mode), gamma, cols); h_fit = fit_hypers(ev_fit)
    post_fit = ard_posterior(ev_fit, h_fit, kappa_floor, meta)
    ell = cfg.ard_cluster_size * prob.cfg.rcut
    if cfg._shape_variant == "press":
        rc, K_fit = row_clusters(ds_fit, [data.train[i] for i in fit_], ell)
        Gs, _ = press_scores(post_fit, prob, ds_fit, rc, K_fit, ev_fit.sigmas(h_fit), mode=cfg.ard_press)
        post_fit = post_fit._replace(R=shape_factor(post_fit, Gs, cfg.ard_shape_tau))
    else:
        Gs = sandwich_scores(post_fit, prob, ds_fit, ev_fit.sigmas(h_fit)); K_fit = Gs.shape[1]
        post_fit = post_fit._replace(Q=sandwich_factor(post_fit, Gs))
```

  `build_dataset` argument names, `meta`, `E0`, `cpb`, `mode`, `gamma`, `cols`, `fit_hypers` and `kappa_floor` are whatever the current `run_ard_stage` already uses for its subset fit. Keep those lines and change only the index sets.

```python
    # 3. T_val scores
    E = _val_atoms(post_fit if cfg._score_source == "fit" else None, prob, ds_val)  # e (n,3), Frows, z, d, cfg
```

  `_val_atoms(post, prob, ds)` replaces `_val_errors`. Over the batches of ds it returns, for live atoms with w_F > 0:
  - e = F_label − F_pred(post.mean), as `_val_errors` computes it now;
  - V = `post.atom_shape(F rows)` when a posterior is given;
  - `shell_features(batch, r1)` (z, d);
  - the configuration index into `val`.

  For `_score_source == "mixed"`, call it after step 4 with the full posterior. V is then computed with the own-cluster-out rule (the existing `misspec_var_rows(..., own=)` path), and that branch reuses the current code verbatim.

  Then:
  - **Scores.** iso: `v = trace V`, `s = |e| / sqrt(v/3)`. aniso: `s = sqrt(e^T (V + eps (v/3) I)^-1 e)`, using `np.linalg.solve` per atom on stacked (n,3,3). Drop atoms with v ≤ 0.

```python
    # 4. served posterior on all of T and its shape
    post = ard_posterior(ev, h, kappa_floor, meta)       # existing full refit
    if cfg._shape_variant == "press":
        rc, K = row_clusters(data.ds_train, data.train, ell)
        Gf, lev = press_scores(post, prob, data.ds_train, rc, K, ev.sigmas(h), mode=cfg.ard_press)
        post = post._replace(R=shape_factor(post, Gf, cfg.ard_shape_tau))
    else:
        Gf = sandwich_scores(post, prob, data.ds_train, ev.sigmas(h)); K = Gf.shape[1]; lev = np.zeros(K)
        post = post._replace(Q=sandwich_factor(post, Gf))

    # 5. per-group scales on T_val
    gv = assign_groups(E.z, E.d, z_star, edges)
    tab = group_scales(s, gv, E.cfg, G, 1 - cfg.ard_coverage, cfg.ard_n_min)
    post = post._replace(force_shape=cfg.ard_force_shape, eps=cfg.ard_shape_eps,
                         group_consts={"r1": r1, "z_star": z_star, "edges": edges.tolist()},
                         group_table=tab.to_dict(),
                         cal={"scores": s.astype(np.float32), "groups": gv.astype(np.int8),
                              "cfg": E.cfg.astype(np.int64), "src": np.zeros(len(s), np.int8)})

    # 6. report
    report |= {"shape": {"variant": cfg._shape_variant, "mode": cfg.ard_press, "ell": ell, "K": K,
                         "K_fit": K_fit, "rank_R": int(np.shape(post.R if post.R is not None else post.Q)[1]),
                         "lev_p50": float(np.median(lev)), "lev_p99": float(np.quantile(lev, 0.99)),
                         "lev_max": float(lev.max())},
               "groups": tab.to_dict(),
               "split": {"n_fit": len(fit_), "n_val": len(val),
                         "strata": np.bincount(strata, minlength=G).tolist()},
               "transfer": {"f": cfg.ard_val_frac, "N_fit": len(fit_), "N": len(data.train),
                            "score_source": cfg._score_source}}
```

  Keep the existing κ and scalar-λ report entries, computing λ as `kappa_closed_form` over the new T_val (e², v) for comparison. Free `post_fit` after step 3 (`del post_fit`, as now).

  - **`predict_ard`:** when `post.group_table is not None`, take per batch `g = post.groups_of(b)` and F rows as now, then `V = post.atom_shape(F)` and `F_var = lam_rms[g][:, None] ** 2 * np.diagonal(V, axis1=1, axis2=2)`, masked by `node_mask`. Otherwise the existing path.
- [ ] **Step 4: Run** `uv run pytest tests/test_ard.py tests/test_ard_pipeline.py -q`. Expected: all pass. Then `uv run pytest -q`. `tests/test_ard_calc_cli.py` now fits through the default press path, and its existing tests must still pass; if `test_calculator_forces_std_matches_pipeline` fails, Task 7 fixes the calculator side, so note it in the report and continue. Then `uv run ruff check`.
- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/ard.py src/ace_jax/fit/pipeline/config.py tests/test_ard.py tests/test_ard_pipeline.py
git commit -m "feat(ard): stratified hold-out, PRESS jackknife shapes for P_fit and P, per-group scales"
```

### Task 6: `fit/support.py` — support diagnostic

**Files:**
- Create: `src/ace_jax/fit/support.py`
- Modify: `src/ace_jax/fit/ard.py` (`run_ard_stage` builds the reference; `save`/`load` store it), `tests/test_conformal.py`

**Interfaces:**
- Consumes: `fit/inducing.py:site_features(model, cfg, ds)` (site descriptors per node), and the T_val atoms from Task 5 (scores, configuration index).
- Produces:
  - `fit_pca(X_by_species: dict[int, ndarray], var=0.99, cap=64) -> dict[int, (mu, sd, W)]`;
  - `build_support(pca, X, Z, scores, cfg, max_atoms, seed) -> dict` (`{z: {"Xc": whitened cal, "s": scores, "m": masses}}` plus `"pca"`);
  - `support_check(ref, X_t, Z_t, alpha) -> {"support_ok": bool (n,), "support_q": float (n,), "n_eff": {z: float}}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_conformal.py`)

```python
def test_support_flags_out_of_support_with_config_masses():
    from ace_jax.fit.support import build_support, fit_pca, support_check
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 10))
    Z = np.zeros(4000, int)
    cfg = np.arange(4000) // 20
    s = rng.chisquare(3, 4000) ** 0.5
    ref = build_support(fit_pca({0: X}), X, Z, s, cfg, max_atoms=50000, seed=0)
    Xin, Xout = rng.normal(size=(400, 10)), rng.normal(size=(400, 10)) + 10.0
    a = support_check(ref, np.r_[Xin, Xout], np.zeros(800, int), 0.1)
    assert a["support_ok"][:400].mean() > 0.95 and a["support_ok"][400:].mean() < 0.05
    assert np.isinf(a["support_q"][400:][~a["support_ok"][400:]]).all()
    b = support_check(ref, Xin, np.zeros(400, int), 0.1)
    assert b["n_eff"][0] > 0.4 * 200 and np.isfinite(b["support_q"]).all()      # over 200 config masses


def test_support_pca_cap_and_variance():
    from ace_jax.fit.support import fit_pca
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3)) @ rng.normal(size=(3, 100)) + 1e-6 * rng.normal(size=(500, 100))
    mu, sd, W = fit_pca({0: X})[0]
    assert W.shape[1] == 3
    mu, sd, W = fit_pca({0: rng.normal(size=(2000, 200))})[0]
    assert W.shape[1] == 64
```

- [ ] **Step 2: Run** `uv run pytest tests/test_conformal.py -q -k support`. Expected: FAIL with `ModuleNotFoundError: No module named 'ace_jax.fit.support'`.
- [ ] **Step 3: Implement** `src/ace_jax/fit/support.py`

```python
"""Covariate-shift support diagnostic (math rev. 2): per species, PCA of the site descriptor,
a grouped-CV L2 logistic density ratio (target vs calibration), weighted conformal quantile."""
import numpy as np


def fit_pca(X_by_species, var=0.99, cap=64):
    out = {}
    for z, X in X_by_species.items():
        X = np.asarray(X, float)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        _, s, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        k = min(int(np.searchsorted(e, var) + 1), cap, len(s))
        out[int(z)] = (mu, sd, Vt[:k].T / (s[:k] / np.sqrt(len(X))))       # whitened components
    return out


def _proj(pca, z, X):
    mu, sd, W = pca[int(z)]
    return ((np.asarray(X, float) - mu) / sd) @ W


def build_support(pca, X, Z, scores, cfg, max_atoms, seed):
    rng = np.random.default_rng(seed)
    ref = {"pca": pca}
    for z in np.unique(Z):
        m = np.flatnonzero(Z == z)
        cs = rng.permutation(np.unique(cfg[m]))
        keep, n = [], 0
        for c in cs:
            mc = m[cfg[m] == c]
            if n + len(mc) > max_atoms and keep:
                break
            keep.append(mc); n += len(mc)
        m = np.concatenate(keep)
        _, inv, cnt = np.unique(cfg[m], return_inverse=True, return_counts=True)
        ref[int(z)] = {"Xc": _proj(pca, z, X[m]), "s": np.asarray(scores, float)[m],
                       "m": 1.0 / cnt[inv], "g": inv}
    return ref


def _logistic(X, y, w, l2, iters=200):
    """Weighted L2 logistic regression by Newton's method; returns (beta, b0)."""
    n, p = X.shape
    Xa = np.c_[X, np.ones(n)]
    beta = np.zeros(p + 1)
    reg = np.r_[np.full(p, l2), 0.0]
    for _ in range(iters):
        t = 1 / (1 + np.exp(-Xa @ beta))
        gr = Xa.T @ (w * (t - y)) + reg * beta
        H = (Xa * (w * t * (1 - t))[:, None]).T @ Xa + np.diag(reg)
        step = np.linalg.solve(H, gr)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def _ratio(Xc, gc, Xt, seed):
    """log density ratio log p_t/p_c at calibration and target points, L2 chosen by 5-fold CV grouped by
    calibration configuration (target atoms split at random)."""
    rng = np.random.default_rng(seed)
    X = np.r_[Xc, Xt]
    y = np.r_[np.zeros(len(Xc)), np.ones(len(Xt))]
    w = np.r_[np.full(len(Xc), 0.5 / len(Xc)), np.full(len(Xt), 0.5 / len(Xt))] * len(X)
    ug = np.unique(gc)
    fold_c = dict(zip(ug, rng.integers(0, 5, len(ug))))
    fold = np.r_[np.array([fold_c[g] for g in gc]), rng.integers(0, 5, len(Xt))]
    best, l2b = np.inf, 1e-2
    for l2 in (1e-3, 1e-2, 1e-1, 1.0):
        nll = 0.0
        for f in range(5):
            tr, te = fold != f, fold == f
            if not te.any():
                continue
            beta = _logistic(X[tr], y[tr], w[tr], l2 * tr.sum())
            t = np.clip(1 / (1 + np.exp(-np.c_[X[te], np.ones(te.sum())] @ beta)), 1e-12, 1 - 1e-12)
            nll -= np.sum(w[te] * (y[te] * np.log(t) + (1 - y[te]) * np.log(1 - t)))
        if nll < best:
            best, l2b = nll, l2
    beta = _logistic(X, y, w, l2b * len(X))
    f = np.c_[X, np.ones(len(X))] @ beta
    return f[:len(Xc)], f[len(Xc):]


def support_check(ref, X_t, Z_t, alpha, seed=0):
    n = len(Z_t)
    ok, q, neff = np.zeros(n, bool), np.full(n, np.inf), {}
    for z in np.unique(Z_t):
        mt = np.flatnonzero(Z_t == z)
        r = ref.get(int(z))
        if r is None:
            neff[int(z)] = 0.0
            continue
        Xt = _proj(ref["pca"], z, X_t[mt])
        lc, lt = _ratio(r["Xc"], r["g"], Xt, seed)
        sh = max(lc.max(), lt.max())
        pc = r["m"] * np.exp(lc - sh)
        neff[int(z)] = float(pc.sum() ** 2 / np.sum(pc ** 2))
        o = np.argsort(r["s"])
        cs = np.cumsum(pc[o])
        for j, i in enumerate(mt):
            pt = np.exp(lt[j] - sh)
            k = int(np.searchsorted(cs / (cs[-1] + pt), 1 - alpha - 1e-15))
            if k < len(o):
                ok[i], q[i] = True, r["s"][o][k]
    return {"support_ok": ok, "support_q": q, "n_eff": neff}
```

  **Integrate it in `run_ard_stage`** (step 5 of Task 5, when `cfg.ard_support`):
  - X_cal = `site_features(prob.model, prob.cfg, ds_val)` restricted to the same live atoms `_val_atoms` returns. Have `_val_atoms` return each atom's (batch, node) index so this can be matched.
  - The PCA is fitted on up to 50 000 training atoms per species from `site_features` over `data.ds_train` (a random sample of batches).
  - Store `post.support = build_support(...)`.

  **In `save`/`load`:** flatten to `support_pca_{z}_{mu,sd,W}` and `support_{z}_{Xc,s,m,g}` (float32 except `g`), and rebuild the dict on load.
- [ ] **Step 4: Run** `uv run pytest tests/test_conformal.py -q`. Expected: all pass. Then `uv run pytest -q && uv run ruff check`.
- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/fit/support.py src/ace_jax/fit/ard.py tests/test_conformal.py
git commit -m "feat(support): descriptor-PCA covariate-shift support flag with configuration masses"
```

### Task 7: Calculator — the served properties

**Files:**
- Modify: `src/ace_jax/calc/point.py` (`_forces_std` → `_posterior_quantities`, `implemented_properties`, the fast-path property set)
- Test: `tests/test_ard_calc_cli.py`

**Interfaces:**
- Consumes: Task 4's methods, `support_check` (Task 6), `site_features`.
- Produces: `ACECalculator` properties `forces_std (N,)`, `forces_cov (N,3,3)`, `forces_q (N,)`, `forces_q_mahal (N,)` (aniso only), `forces_group (N,) int` and `forces_support` (dict: `support_ok`, `support_q`, `n_eff`), each computed only when requested.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ard_calc_cli.py`; the module fixture `fitted` uses defaults)

```python
def test_calculator_served_quantities_and_lean_parity(fitted):
    from ase.io import read
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    tab = ARDPosterior.load(fitted / "posterior.npz").group_table
    at = max(read(XYZ, ":8"), key=len)
    out = {}
    for lean in (True, False):
        calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), lean=lean)
        a = at.copy()
        a.calc = calc
        a.get_forces()
        sd = np.asarray(calc.get_property("forces_std", a))
        cov = np.asarray(calc.get_property("forces_cov", a))
        q = np.asarray(calc.get_property("forces_q", a))
        g = np.asarray(calc.get_property("forces_group", a))
        assert sd.shape == (len(a),) and cov.shape == (len(a), 3, 3) and g.dtype.kind == "i"
        np.testing.assert_allclose(np.trace(cov, axis1=1, axis2=2), sd ** 2, rtol=1e-10)
        v = sd ** 2 / np.asarray(tab["lam_rms"])[g] ** 2
        np.testing.assert_allclose(q, np.asarray(tab["q"])[g] * np.sqrt(v / 3), rtol=1e-8)
        out[lean] = (sd, q, g)
    np.testing.assert_allclose(out[True][0], out[False][0], rtol=LEAN_RTOL)
    np.testing.assert_allclose(out[True][1], out[False][1], rtol=LEAN_RTOL)
    assert np.array_equal(out[True][2], out[False][2])


def test_calculator_forces_support(fitted):
    from ase.io import read
    from ace_jax import ACECalculator
    at = max(read(XYZ, ":8"), key=len)
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"))
    at.calc = calc
    sup = calc.get_property("forces_support", at)
    ok = sup["support_ok"]
    assert ok.shape == (len(at),) and ok.dtype == bool and all(v > 0 for v in sup["n_eff"].values())
    assert np.isfinite(sup["support_q"][ok]).all() and np.isinf(sup["support_q"][~ok]).all()


def test_calculator_schema2_new_properties_raise(fitted, tmp_path):
    from ase.io import read
    from ace_jax import ACECalculator
    z = dict(np.load(fitted / "posterior.npz"))
    z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("R", "group_", "cal_", "support", "force_shape", "eps"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    at = read(XYZ, "0")
    calc = ACECalculator(str(fitted / "model.npz"), posterior=str(tmp_path / "p2.npz"))
    at.calc = calc
    assert np.isfinite(calc.get_property("forces_std", at)).all()
    with pytest.raises(ValueError, match="refit with --uq ard"):
        calc.get_property("forces_q", at)
```

  The existing `test_calculator_forces_std_matches_pipeline` must keep passing: the calculator's `forces_std` equals √(Σ F_var) from `predict_ard`.

  The schema-2 file in `test_calculator_schema2_new_properties_raise` lacks `Q` (the default fit stores R), so `forces_std` falls back to κ. That is the intended schema-2/κ path.

- [ ] **Step 2: Run** `uv run pytest tests/test_ard_calc_cli.py -q`. Expected: the new tests FAIL (`PropertyNotImplementedError` or `KeyError` on `forces_cov`).
- [ ] **Step 3: Implement.** Replace `_forces_std` with `_posterior_quantities(atoms, which: set[str]) -> dict`:
  - Build `b` exactly as `_forces_std` does now (`Config` → `build_dataset` → batch 0), and take the F rows from `chunked_rows_fn(self._fit_model, self._fit_cfg)` (the full model, never `eval_model`).
  - Let `live = np.asarray(b.node_mask)` and `F = Frows[live]`.
  - `groups = post.groups_of(b)[live]` when `post.group_consts` is set, else None.
  - Then fill only what `which` asks for:
    - `forces_std`: `post.forces_std(F, groups)`;
    - `forces_cov`: `post.forces_cov(F, groups)`;
    - `forces_q`: `post.forces_q(F, groups)`;
    - `forces_q_mahal`: `post.forces_q_mahal(groups)`, only if `post.force_shape == "aniso"`, else `PropertyNotImplementedError`;
    - `forces_group`: `groups`, raising `ValueError(_NEED3)` if None;
    - `forces_support`: `support_check(post.support, site_features(self._fit_model, self._fit_cfg, ds)[0][live], b.node_z[live], 1 - post.group_table["alpha"])`.

  Add the new names to `implemented_properties` and to the fast-path set that already routes `forces_std`. Cache the results in `self.results` like `forces_std`.
- [ ] **Step 4: Run** `uv run pytest tests/test_ard_calc_cli.py -q`. Expected: all pass. Then `uv run pytest -q && uv run ruff check`.
- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/calc/point.py tests/test_ard_calc_cli.py
git commit -m "feat(calc): forces_cov/forces_q/forces_q_mahal/forces_group/forces_support"
```

### Task 8: CLI — fit flags, `aj calibrate`, eval

**Files:**
- Modify: `src/ace_jax/cli.py`
- Test: `tests/test_ard_calc_cli.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 6 and 7. `load_configs` (as `cmd_eval` uses it), `ACECalculator`, `group_scales`, `GroupTable`.
- Produces:
  - **`aj fit` flags:** `--force-shape {iso,aniso}`, `--ard-coverage FLOAT`, `--ard-groups {distortion,none}`, `--ard-cluster-size FLOAT|inf`, `--ard-press {exact,block}`, `--ard-n-min INT`, `--no-ard-support`.
  - **`aj calibrate`:** `--model M --posterior P --data U [--energy-key k] [--force-key k] [--virial-key k] [--append | --replace] --out O`.
  - **`aj eval --per-atom`** adds `forces_q` and `forces_group`. `--support` adds `support_ok` and `support_q`. Aniso posteriors also write `forces_cov` (9 columns) and `forces_q_mahal`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ard_calc_cli.py`)

```python
def _calib_args(fitted, data, extra=()):
    return ["calibrate", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
            "--data", str(data), "--energy-key", "dft_energy", "--force-key", "dft_force",
            "--virial-key", "dft_virial", *extra]


@pytest.fixture(scope="module")
def calib_set(tmp_path_factory):
    """25 rattled copies of one training config: >= n_min (20) configurations in its groups."""
    from ase.io import read, write
    base = read(XYZ, "0")
    out = []
    for i in range(25):
        a = base.copy()
        a.rattle(0.01, seed=i)
        out.append(a)
    p = tmp_path_factory.mktemp("cal") / "U.xyz"
    write(p, out)
    return p


def test_calibrate_per_group_replace(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    base = ARDPosterior.load(fitted / "posterior.npz")
    assert main(_calib_args(fitted, calib_set, ["--out", str(tmp_path / "d.npz")])) == 0
    new = ARDPosterior.load(tmp_path / "d.npz")
    t = new.group_table
    replaced = [g for g, nc in enumerate(t["n_cfg_cal"]) if nc >= t["n_min"]]
    assert replaced                                                       # the 25 copies cover >= 1 group
    for g in replaced:
        assert t["n_cfg_val"][g] == 0, g
    for g, (nv0, nc) in enumerate(zip(base.group_table["n_cfg_val"], t["n_cfg_cal"])):
        if nc < t["n_min"]:
            assert t["n_cfg_val"][g] == nv0, g                             # short groups keep T_val
    np.testing.assert_array_equal(new.mean, base.mean)
    assert t["sources"] and t["sources"][-1]["n_cfg"] == 25


def test_calibrate_append_and_replace(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    assert main(_calib_args(fitted, calib_set, ["--append", "--out", str(tmp_path / "a.npz")])) == 0
    assert main(_calib_args(fitted, calib_set, ["--replace", "--out", str(tmp_path / "r.npz")])) == 0
    a, r = ARDPosterior.load(tmp_path / "a.npz"), ARDPosterior.load(tmp_path / "r.npz")
    assert set(np.unique(a.cal["src"]).tolist()) == {0, 1} and set(np.unique(r.cal["src"]).tolist()) == {1}
    assert sum(r.group_table["n_cfg_val"]) == 0


def test_calibrate_refuses_schema2(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    z = dict(np.load(fitted / "posterior.npz"))
    z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("R", "group_", "cal_", "support", "force_shape", "eps"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    args = _calib_args(fitted, calib_set, ["--out", str(tmp_path / "o.npz")])
    args[args.index("--posterior") + 1] = str(tmp_path / "p2.npz")
    with pytest.raises(ValueError, match="refit with --uq ard"):
        main(args)


def test_calibrate_requires_forces(fitted, calib_set, tmp_path):
    from ace_jax.cli import main
    args = _calib_args(fitted, calib_set, ["--out", str(tmp_path / "o.npz")])
    args[args.index("--force-key") + 1] = "no_such_force"
    with pytest.raises(ValueError, match="no_such_force"):
        main(args)


def test_eval_per_atom_writes_served_arrays(fitted, tmp_path):
    from ase.io import read
    from ace_jax.cli import main
    assert main(["eval", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
                 "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
                 "--per-atom", str(tmp_path / "pa.xyz"), "--support"]) == 0
    a = read(tmp_path / "pa.xyz", "0")
    for k in ("forces_std", "forces_q", "forces_group", "support_ok", "support_q"):
        assert k in a.arrays, k
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ard_calc_cli.py -q -k "calibrate or served_arrays"`. Expected: FAIL (`argparse` error: invalid choice 'calibrate', exit 2 raised as `SystemExit`).
- [ ] **Step 3: Implement** `cmd_calibrate` in `cli.py`:

```python
_NEED3 = "this posterior predates schema 3; refit with --uq ard"


def cmd_calibrate(a):
    import hashlib
    from .calc.point import ACECalculator
    from .fit.ard import ARDPosterior
    from .fit.conformal import GroupTable, group_scales, n_groups
    post = ARDPosterior.load(a.posterior)
    if post.group_table is None or post.cal is None:
        raise ValueError(_NEED3)
    configs = load_configs(a.data, energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key)
    if not any(c.forces is not None for c in configs):
        raise ValueError(f"calibrate: no forces under --force-key {a.force_key!r} in {a.data}")
    calc = ACECalculator(a.model, posterior=a.posterior)
    s_u, g_u, c_u = [], [], []
    for ci, (at, c) in enumerate(zip(_atoms_of(a.data), configs)):
        if c.forces is None:
            continue
        at.calc = calc
        e = np.asarray(c.forces) - at.get_forces()
        cov = np.asarray(calc.get_property("forces_cov", at))
        g = np.asarray(calc.get_property("forces_group", at))
        lam = np.asarray(post.group_table["lam_rms"])[g]
        V = cov / lam[:, None, None] ** 2
        v = np.trace(V, axis1=1, axis2=2)
        ok = v > 0
        if post.force_shape == "aniso":
            M = V + post.eps * (v / 3)[:, None, None] * np.eye(3)
            s = np.sqrt(np.einsum("na,na->n", e, np.linalg.solve(M, e[..., None])[..., 0]))
        else:
            s = np.linalg.norm(e, axis=1) / np.sqrt(np.maximum(v, 1e-300) / 3)
        s_u.append(s[ok]); g_u.append(g[ok]); c_u.append(np.full(ok.sum(), ci))
    s_u, g_u, c_u = map(np.concatenate, (s_u, g_u, c_u))
    t0 = post.group_table
    G, n_min = len(t0["q"]), int(t0["n_min"])
    cal = post.cal
    src_new = int(cal["src"].max()) + 1
    c_u = c_u + int(cal["cfg"].max()) + 1
    u_cfg = np.array([len(np.unique(c_u[g_u == g])) for g in range(G)])
    keep_old = np.ones(len(cal["scores"]), bool)
    if a.replace:
        keep_old[:] = False
    elif not a.append:
        keep_old = u_cfg[cal["groups"].astype(int)] < n_min                 # per-group replace
    S = np.r_[cal["scores"][keep_old], s_u]
    Gg = np.r_[cal["groups"][keep_old].astype(int), g_u]
    Cc = np.r_[cal["cfg"][keep_old], c_u]
    Ss = np.r_[cal["src"][keep_old].astype(int), np.full(len(s_u), src_new)]
    tab = group_scales(S, Gg, Cc, G, t0["alpha"], n_min, src=Ss)
    sha = hashlib.sha256(open(a.data, "rb").read()).hexdigest()
    sources = list(t0.get("sources", [])) + [{"path": str(a.data), "sha256": sha, "n_cfg": len(np.unique(c_u)),
                                             "n_atoms": int(len(s_u)), "mode": "replace" if a.replace else
                                             ("append" if a.append else "per-group")}]
    tab = GroupTable(**{**tab.__dict__, "sources": sources})
    new = post._replace(group_table=tab.to_dict(),
                        cal={"scores": S.astype(np.float32), "groups": Gg.astype(np.int8),
                             "cfg": Cc.astype(np.int64), "src": Ss.astype(np.int8)})
    new.save(a.out)
    _print_group_table(tab)
    return 0
```

  - `_atoms_of(path)` is `ase.io.read(path, ":")`.
  - `load_configs` is the same loader `cmd_eval` uses, with the same key arguments; reuse its exact call signature. If it raises on a missing force key itself, make sure the message names the key.
  - `_print_group_table` prints one row per group: `g`, `n_cfg (T_val/U)`, `lam_rms`, `q`, `r`, and `-> g_src` if merged.
  - **Support:** if `post.support` is present, rebuild it from the pooled atoms. That needs their descriptors, so collect `site_features` for the U atoms in the loop (one more property request per configuration). Then rebuild with `build_support` over T_val's stored calibration points (kept atoms) together with U's.
  - Add the `calibrate` subparser. Make `--append` and `--replace` mutually exclusive.
  - **`aj fit` flags:** each maps to its `FitConfig` field. `--ard-cluster-size` is parsed with `float` (accepts `inf`). `--no-ard-support` sets `ard_support=False`.
  - **`cmd_eval --per-atom`:** next to `forces_std`, write `forces_q` and `forces_group`. With `--support`, write `support_ok` and `support_q`. With aniso, write `forces_cov` reshaped (N, 9) and `forces_q_mahal`. Skip the new arrays with a one-line notice for schema-2 posteriors.
- [ ] **Step 4: Run** `uv run pytest tests/test_ard_calc_cli.py -q`. Expected: all pass. Then `uv run pytest -q && uv run ruff check`.
- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/cli.py tests/test_ard_calc_cli.py
git commit -m "feat(cli): fit flags for the jackknife shape and scales; aj calibrate (per-group replace); eval outputs"
```

### Task 9: Docs

**Files:**
- Modify: `README.md`, `skills/ace-jax/SKILL.md`

- [ ] **Step 1: README, `--uq ard` section.** Replace the λ/sandwich paragraph with these points:
  - **The shape:** the centred PRESS cluster jackknife; large cells split into ~3 r_cut blocks; `--force-shape aniso` for a 3×3 shape.
  - **The two scales:** `forces_std`/`forces_cov` use a per-group rms factor; `forces_q` is the per-group conformal radius at `--ard-coverage` (default 0.9). `forces_group` names the group, and the posterior's per-group table gives λ_rms, q and r = q/(λ_rms·χ₃⁻¹(0.9)).
  - **`aj calibrate`:** per-group replace by default, `--append`, `--replace`, with an example on labelled crack cells.
  - **`forces_support`:** atoms the calibration cannot certify.

  Link `docs/dev/specs/tex/force-uq-math-pipeline.tex`.
- [ ] **Step 2: SKILL.md gotchas.** Add three bullets:
  - Coverage holds for atoms exchangeable with their group's calibration configurations. For a new regime (cracks, interfaces), run `aj calibrate` on a few labelled cells like it.
  - `support_ok = False` marks candidates for labelling.
  - Schema-2 posteriors serve only the old scalar σ.
- [ ] **Step 3:** run `uvx pre-commit run --all-files`. Expected: Passed. Then commit:

```bash
git add README.md skills/ace-jax/SKILL.md
git commit -m "docs: jackknife shape and conformal scales for --uq ard"
```

### Task 10: Validation programme and acceptance (Modal)

**Files:**
- Create: `bench/defect_uq/scoring/validate_shape.py`
- Modify: `bench/defect_uq/fit_bench.py` (arms, `--train-extra`), `bench/defect_uq/scoring/big_errors.py` (save the served arrays), `bench/defect_uq/modal/modal_bench365.py`, `bench/defect_uq/README.md`
- Data: `/storage/eng/essswb/acegp-data/cantor/big3/` (v3 big cells, crack realisations r2–r9), results under `/storage/eng/essswb/acegp-data/results/2026-10-xx/`

This task needs PR #21's `bench/line-period` files (`big3*.py`, `conformal_cv.py`). Rebase `feat/conformal-uq` onto `origin/main` after #21 merges, or cherry-pick e530459..fcad3cc at the start of this task.

- [ ] **Step 1: Arms in `fit_bench.py`.** Each name maps to `FitConfig` overrides:

```python
ARD_ARMS = {
    "ard_legacy": {"_shape_variant": "legacy", "_score_source": "mixed", "ard_groups": "none"},
    "ard_A":      {"_shape_variant": "legacy", "_score_source": "fit"},
    "ard_AB":     {"ard_cluster_size": float("inf")},
    "ard_ABblk":  {},                                                     # defaults: press, ell = 3 r_cut
    "ard_aniso":  {"ard_force_shape": "aniso"},
    **{f"ard_ell{k}": {"ard_cluster_size": float(k)} for k in (2, 4, 6)},
    **{f"ard_f{int(f * 10)}": {"ard_val_frac": f} for f in (0.1, 0.3)},
}
```

  `--train-extra PATH[,PATH...]` appends those xyz configurations to the training set before the fit. It is used to put the crack realisations not held out into T for the ℓ sweep.
- [ ] **Step 2: `big_errors.py`.** Next to the existing per-atom `forces_std`, also save `forces_q`, `forces_group` and `forces_cov` (aniso arms) into `*_err.npz`. A unit test is in `bench/defect_uq/tests/` if that directory exists; otherwise check by hand on one cell and record that in the README.
- [ ] **Step 3: `validate_shape.py`.** CLI: `--runs DIR [DIR...] --out report.md`. Per run, read `ard.json`, `posterior.npz` and `big3*_err.npz`. Report:
  1. the leverage summary (`shape.lev_*`);
  2. global λ_rms (T_val pooled, configuration-weighted) per variant;
  3. Spearman ρ(σ, |e|) per family;
  4. per-group coverage of `forces_q`, configuration-weighted and atom-weighted, with configuration-bootstrap 90 % intervals (B = 1000, resampling whole cells);
  5. crack-tip coverage (atoms within 10 Å of the tip);
  6. mean region volume (`(4π/3) forces_q³` iso, or the ellipsoid volume aniso) at equal coverage;
  7. for the f arms, log λ_g against log N_fit with a least-squares slope;
  8. for the ℓ arms, median v and λ_g against ℓ.

  Reuse `bench/defect_uq/scoring/conformal_cv.py`'s bootstrap and tip-mask helpers by import; do not copy them.
- [ ] **Step 4: Run on Modal** (manual, B200; record the exact commands in `bench/defect_uq/README.md`):
  1. the 5 ablation arms on bench365;
  2. the ℓ sweep {2, 3, 4, 6, ∞}, with the crack realisations not held out in T per leave-one-realisation fold (start with 3 representative folds; extend to all if the spread between folds is > 1 point), reporting tip coverage with tip data in training against calibration only;
  3. the f sweep {0.1, 0.2, 0.3};
  4. `aj calibrate` leave-one-realisation-out on the v3 crack cells with the default arm.
- [ ] **Step 5: Acceptance.**
  - In-distribution held-out coverage 0.90 ± 0.01.
  - Crack whole cell ≥ 0.89, tip ≥ 0.88, edge/screw ≥ 0.90 (calibrate, leave-one-realisation-out).

  Record the results in `bench/defect_uq/README.md` and `/storage/eng/essswb/acegp-data/results/.../ACCEPTANCE.md`. A miss is reported as a miss, with the per-group table.
- [ ] **Step 6: Commit**

```bash
git add bench/defect_uq
git commit -m "bench(defect_uq): validation programme for the jackknife shape and conformal scales"
```
