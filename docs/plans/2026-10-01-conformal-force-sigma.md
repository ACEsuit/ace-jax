# Conformal force-σ calibration: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scalar λ/κ scale of the ARD force σ with Mondrian split-conformal factors λ_g, one per geometric group, giving a stated coverage. Add `aj calibrate`, which recalibrates a posterior on labelled target-regime data, and a covariate-shift support flag as a diagnostic.

**Architecture:**
- The **shape** v(x) is unchanged: the configuration-clustered sandwich, or the epistemic variance.
- A new pure-numpy module `fit/conformal.py` holds the geometry features, the groups and the Mondrian quantiles. `fit/support.py` holds the weighted-conformal diagnostic.
- `ARDPosterior` gains schema-3 fields (`conformal`, `cal`, `support`). It serves `forces_std = λ_g·√v` and `forces_q = χ₃⁻¹(1−α)·forces_std/√3`.
- The ARD stage fits λ_g from the existing hold-out. The calculator and CLI expose the results.

**Tech Stack:** JAX (x64), NumPy, SciPy (`scipy.stats.chi`, `scipy.optimize`), ASE, pytest, ruff; `uv run`.

**Spec:** `docs/specs/2026-09-30-conformal-force-sigma-design.md` (interfaces, decisions, acceptance). Mathematical reference: `docs/specs/2026-10-01-force-uq-math-pipeline.md` (§5 shape, §6 scale, §7 groups, §8 calibration sets, §9 support).

## Global Constraints

- Worktree `~/projects/ace-jax/.worktrees/conformal-uq`, branch `feat/conformal-uq`. Run everything with `uv run`; `uv run pytest -q` and `uv run ruff check` must stay green.
- float64 for every fitting quantity. Stored calibration arrays are float32 for scores and distortions, integer for groups, coordinations and configs.
- **Unchanged:** the mean c̄, E/V variances, v(x) (`misspec_var_rows` / `var_rows`), the hold-out protocol, the λ/κ fits, and the leave-own-cluster-out rule.
- **Score:** `s = |ΔF| / sqrt(v / 3)`. `lam_g = q_g / chi3_ppf(1 - alpha)`, where `chi3_ppf(p) = scipy.stats.chi.ppf(p, 3)` (2.5003 at p = 0.9). `forces_q = chi3_ppf(coverage) * forces_std / sqrt(3)`.
- **Finite-sample level:** `min(ceil((n + 1) * (1 - alpha)) / n, 1.0)`.
- **`FitConfig` defaults:**
  - `ard_calibration = "conformal"` (∈ {"conformal", "gaussian"});
  - `ard_coverage = 0.9`;
  - `ard_groups = "distortion"` (∈ {"distortion", "none"});
  - `ard_group_bands = (0.5, 0.9, 0.99)`;
  - `ard_n_min_cfg = 5`, `ard_n_min_atoms = 200`;
  - `ard_support = True`, `ard_support_max_atoms = 50000`.
- **CLI:** `--ard-calibration`, `--ard-coverage`, `--ard-groups`, `--no-ard-support`; a new subcommand `calibrate`; `eval` adds `forces_q` and `ard_group`, and `--support` adds `support_ok` / `support_q`.
- **`posterior.npz` `SCHEMA = 3`.** Schema 1 and 2 still load and serve the λ/κ scale (no `forces_q`).
- **Groups:** `g = band(d) * 2 + (z != z_star)`. Band edges are fixed at fit time from the hold-out pool's d quantiles; `aj calibrate` keeps them. The top band holds NaN d.
- **Names are binding:**
  - `fit/conformal.py`: `chi3_ppf`, `conformal_level`, `shell_reference`, `shell_features`, `band_edges`, `assign_groups`, `Mondrian`;
  - `fit/support.py`: `SupportReference`, `build_support`, `support_check`;
  - `ARDPosterior`: `.conformal`, `.cal`, `.support`, `.row_scale2`, `.forces_q`, `.unscaled_atom_var`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`

## Review Focus

1. **A target atom whose distortion is beyond every calibration band, or NaN** (fewer than 2 first-shell neighbours: vacuum, isolated atoms). It must land in the top band and get a finite λ_g, not crash or get NaN. Owner: Task 1 (`test_assign_groups_extremes`).
2. **A group with too few calibration configurations or atoms.** It is merged into the nearest populated band of the same coordination class, and failing that into the global quantile, with the merge recorded. It is never served from 3 atoms. Owner: Task 1 (`test_mondrian_merges_small_groups`).
3. **A schema-2 `posterior.npz` passed to `aj calibrate` or to the calculator.** The calculator serves the old λ scale (`forces_q` raises a clear error). `aj calibrate` refuses with a message to refit with `--ard-calibration conformal`. Owner: Task 2 (`test_schema2_posterior_serves_lambda`) and Task 6 (`test_calibrate_refuses_schema2`).
4. **A non-fcc training set**, where the RDF first minimum is not at fcc distances, or has no clear minimum. `shell_reference` must find bcc's minimum, and fall back to `1.25 × first peak` when there is none. Owner: Task 1 (`test_shell_reference_fcc_bcc_and_fallback`).
5. **`aj calibrate` given labelled data without forces.** A `ValueError` naming the force key, not a silent no-op recalibration. Owner: Task 6 (`test_calibrate_requires_forces`).

---

## File Structure

| file | role |
|---|---|
| `src/ace_jax/fit/conformal.py` (new) | geometry features from a Dataset batch, groups, finite-sample quantiles, the `Mondrian` fit |
| `src/ace_jax/fit/support.py` (new) | `SupportReference` (whitened 2-body pool), per-species logistic density ratio, weighted quantile |
| `src/ace_jax/fit/ard.py` (modify) | `ARDPosterior` schema 3 and served scale; `run_ard_stage` fits λ_g and builds the support reference; `predict_ard` applies per-atom λ_g² |
| `src/ace_jax/fit/pipeline/config.py` (modify) | the new `ard_*` fields and their validation |
| `src/ace_jax/calc/point.py` (modify) | groups at serve time; properties `forces_q`, `ard_group`, `forces_support` |
| `src/ace_jax/cli.py` (modify) | fit flags, the `calibrate` subcommand, eval outputs |
| `tests/test_conformal.py` (new) | Task 1 |
| `tests/test_ard.py`, `tests/test_ard_pipeline.py`, `tests/test_ard_calc_cli.py` (modify) | Tasks 2–6 |
| `README.md`, `skills/ace-jax/SKILL.md` | Task 7 |

---

### Task 1: `fit/conformal.py` — features, groups and Mondrian quantiles

**Files:** create `src/ace_jax/fit/conformal.py`; test `tests/test_conformal.py`.

**Interfaces:**
- Produces:
  - `chi3_ppf(p: float) -> float`
  - `conformal_level(n: int, alpha: float) -> float`
  - `shell_reference(dists: np.ndarray) -> tuple[float, int]`, taking all pair distances (< r_cut) from training nodes and returning `(r1, z_star)`. `z_star` is computed by the caller: see `shell_features`.
  - `shell_features(batch, r1: float) -> tuple[np.ndarray, np.ndarray]` returning per node of the batch (Ncap): `z` (int) and `d` (float, NaN if z < 2).
  - `band_edges(d: np.ndarray, quantiles) -> np.ndarray`
  - `assign_groups(z, d, z_star, edges) -> np.ndarray`, with G = 2·(len(edges)+1)
  - `class Mondrian`, a frozen dataclass with fields `alpha`, `edges`, `z_star`, `r1`, `q_g`, `lam_g`, `n_atoms_g`, `n_cfg_g`, `merged`; classmethod `fit(scores, groups, cfg, alpha, edges, z_star, r1, n_min_cfg, n_min_atoms)`; `to_dict()` / `from_dict(d)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_conformal.py`

```python
import numpy as np
import pytest
from ase.build import bulk, fcc111


def _batch(atoms, rcut=6.25):
    """A one-config Dataset batch for `atoms` (the calculator's construction).  If `build_dataset` needs
    more meta keys than elements/rcut, take them from `ace_jax.eval.load(FIXTURE_DIR/"si_fitted.npz")[1]`
    with `elements` and `rcut` overridden -- only the neighbour list matters here."""
    import jax
    from ace_jax.fit.data import Config, build_dataset
    meta = {"elements": sorted(set(int(z) for z in atoms.numbers)), "rcut": rcut}
    c = Config(atoms.get_positions(), atoms.get_atomic_numbers(), atoms.get_cell().array, atoms.get_pbc(),
               None, None, None, 1.0, 1.0, 1.0)
    ds = build_dataset([c], meta, np.zeros(len(meta["elements"])), 1)
    return jax.tree.map(lambda a: a[0], ds)


def test_chi3_and_level():
    from ace_jax.fit.conformal import chi3_ppf, conformal_level
    assert chi3_ppf(0.9) == pytest.approx(2.50028, abs=1e-4)
    assert conformal_level(99, 0.1) == pytest.approx(90 / 99)
    assert conformal_level(5, 0.1) == 1.0


def test_split_conformal_coverage_is_valid_and_tight():
    """Exchangeable synthetic scores: coverage >= 1 - alpha and <= 1 - alpha + 1/(n+1) on average."""
    from ace_jax.fit.conformal import conformal_level
    rng = np.random.default_rng(0)
    n, alpha, reps, hits = 199, 0.1, 4000, 0
    for _ in range(reps):
        s = rng.chisquare(3, n + 1) ** 0.5
        q = np.quantile(s[:n], conformal_level(n, alpha), method="higher")
        hits += s[n] <= q
    assert 1 - alpha - 0.01 <= hits / reps <= 1 - alpha + 1 / (n + 1) + 0.01


def test_shell_features_fcc_strained_slab():
    from ace_jax.fit.conformal import shell_features
    a0 = 3.65; r1 = 0.5 * (a0 / np.sqrt(2) + a0)        # between 1st and 2nd fcc shells
    z, d = shell_features(_batch(bulk("Ni", "fcc", a=a0, cubic=True).repeat(3)), r1)
    assert np.all(z[:108] == 12) and np.allclose(d[:108], 0.0, atol=1e-12)
    st = bulk("Ni", "fcc", a=a0, cubic=True).repeat(3); st.set_cell(st.cell @ np.diag([1.04, 1, 1]), scale_atoms=True)
    z2, d2 = shell_features(_batch(st), r1)
    assert np.all(z2[:108] == 12) and np.all(d2[:108] > 0.01)
    sl = fcc111("Ni", size=(3, 3, 6), a=a0, vacuum=8.0); sl.pbc = True
    z3, _ = shell_features(_batch(sl), r1)
    assert z3[:len(sl)].min() == 9 and z3[:len(sl)].max() == 12      # surface atoms are 9-fold


def test_shell_reference_fcc_bcc_and_fallback():
    from ace_jax.fit.conformal import shell_reference
    from ase.neighborlist import neighbor_list
    for at, lo, hi in ((bulk("Ni", "fcc", a=3.65, cubic=True).repeat(4), 2.6, 3.6),
                       (bulk("Fe", "bcc", a=2.87, cubic=True).repeat(5), 2.9, 4.0)):
        at.rattle(0.03, seed=1)
        r1, _ = shell_reference(neighbor_list("d", at, 6.0))
        assert lo < r1 < hi, (at.get_chemical_formula(), r1)
    r1, _ = shell_reference(np.full(1000, 2.5) + np.random.default_rng(0).normal(0, 1e-3, 1000))
    assert r1 == pytest.approx(1.25 * 2.5, rel=0.02)                   # no minimum: 1.25 x first peak


def test_assign_groups_extremes():
    from ace_jax.fit.conformal import assign_groups
    edges = np.array([0.01, 0.02, 0.05])
    g = assign_groups(np.array([12, 12, 11, 3, 1]), np.array([0.0, 0.03, 0.9, np.nan, np.nan]), 12, edges)
    assert g.tolist() == [0, 4, 7, 7, 7]                                # band*2 + (z != z*); NaN -> top band


def test_mondrian_quantiles_and_lam():
    from ace_jax.fit.conformal import Mondrian, chi3_ppf, conformal_level
    rng = np.random.default_rng(1)
    s = np.r_[rng.chisquare(3, 3000) ** 0.5, 2 * rng.chisquare(3, 3000) ** 0.5]
    g = np.r_[np.zeros(3000, int), np.full(3000, 2)]
    cfg = np.arange(6000) // 30
    m = Mondrian.fit(s, g, cfg, 0.1, np.array([0.1]), 12, 3.0, n_min_cfg=5, n_min_atoms=200)
    q0 = np.quantile(s[:3000], conformal_level(3000, 0.1), method="higher")
    assert m.q_g[0] == pytest.approx(q0) and m.lam_g[2] == pytest.approx(m.q_g[2] / chi3_ppf(0.9))
    assert m.lam_g[2] == pytest.approx(2 * m.lam_g[0], rel=0.1)
    back = Mondrian.from_dict(m.to_dict())
    assert np.array_equal(back.q_g, m.q_g) and back.merged == m.merged


def test_mondrian_merges_small_groups():
    """Group 1 (band 0, z != z*) has 3 atoms in 1 config: served from its band's other class or global."""
    from ace_jax.fit.conformal import Mondrian
    rng = np.random.default_rng(2)
    s = np.r_[rng.chisquare(3, 2000) ** 0.5, [9.0, 9.0, 9.0]]
    g = np.r_[np.zeros(2000, int), [1, 1, 1]]
    cfg = np.r_[np.arange(2000) // 20, [999, 999, 999]]
    m = Mondrian.fit(s, g, cfg, 0.1, np.array([0.1]), 12, 3.0, n_min_cfg=5, n_min_atoms=200)
    assert m.q_g[1] == m.q_g[0] and [1, 0] in m.merged           # merged into band 0's z == z* group
    assert np.isfinite(m.q_g).all()                              # empty groups fall back to the global quantile
```

- [ ] **Step 2: Run them:** `uv run pytest tests/test_conformal.py -q`. Expected: FAIL (`ModuleNotFoundError: ace_jax.fit.conformal`).

- [ ] **Step 3: Implement** `src/ace_jax/fit/conformal.py`

```python
"""Mondrian split-conformal calibration of the ARD force sigma (spec 2026-09-30; math 2026-10-01 sec 6-7).

The shape v(x) (sandwich or epistemic variance) is unchanged; this module provides the scale: one
conformal factor lam_g = q_g / chi_3^{-1}(1 - alpha) per group g of a reference-free geometric feature.
"""
import dataclasses

import numpy as np
from scipy.stats import chi


def chi3_ppf(p):
    """Quantile of the chi distribution with 3 dof (the |dF| / (sigma/sqrt 3) law of a calibrated Gaussian)."""
    return float(chi.ppf(p, 3))


def conformal_level(n, alpha):
    """Finite-sample split-conformal quantile level ceil((n+1)(1-alpha))/n, capped at 1."""
    return min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0) if n else 1.0


def shell_reference(dists, bins=400):
    """r1 = first minimum of the pair-distance histogram after its first peak (smoothed); 1.25 x the
    first peak when the histogram has no minimum before r_max.  Returns (r1, n_pairs_below_r1)."""
    d = np.asarray(dists, float)
    h, e = np.histogram(d, bins=bins, range=(0.0, d.max()))
    c = 0.5 * (e[1:] + e[:-1])
    g = h / np.maximum(c ** 2, 1e-12)                                 # radial density ~ RDF
    k = np.ones(5) / 5
    g = np.convolve(g, k, mode="same")
    p = int(np.argmax(g > 0.2 * g.max()))                             # first rise
    while p + 1 < len(g) and g[p + 1] >= g[p]:
        p += 1                                                        # first peak
    m = p
    while m + 1 < len(g) and g[m + 1] <= g[m]:
        m += 1                                                        # first minimum after it
    r1 = c[m] if (m + 1 < len(g) and g[m] < 0.5 * g[p]) else 1.25 * c[p]
    return float(r1), int(np.sum(d < r1))


def shell_features(batch, r1):
    """Per node of a Dataset batch: first-shell coordination z = #{r_ij < r1} and distortion
    d = std/mean of those r_ij (NaN when z < 2).  Uses the batch's own neighbour list (rij, nbr_mask)."""
    r = np.linalg.norm(np.asarray(batch.rij), axis=-1)               # (Ncap, K)
    m = np.asarray(batch.nbr_mask) & (r < r1)
    z = m.sum(1)
    rs = np.where(m, r, 0.0)
    mean = rs.sum(1) / np.maximum(z, 1)
    var = np.where(m, (r - mean[:, None]) ** 2, 0.0).sum(1) / np.maximum(z, 1)
    d = np.where(z >= 2, np.sqrt(var) / np.maximum(mean, 1e-12), np.nan)
    return z.astype(np.int64), d


def band_edges(d, quantiles=(0.5, 0.9, 0.99)):
    d = np.asarray(d, float)
    return np.nanquantile(d[np.isfinite(d)], quantiles)


def assign_groups(z, d, z_star, edges):
    """g = band(d) * 2 + (z != z_star); NaN distortion (fewer than 2 first-shell neighbours) is the top band."""
    band = np.searchsorted(np.asarray(edges), np.nan_to_num(np.asarray(d, float), nan=np.inf), side="right")
    return (band * 2 + (np.asarray(z) != z_star)).astype(np.int64)


@dataclasses.dataclass(frozen=True)
class Mondrian:
    alpha: float
    edges: np.ndarray
    z_star: int
    r1: float
    q_g: np.ndarray
    lam_g: np.ndarray
    n_atoms_g: np.ndarray
    n_cfg_g: np.ndarray
    merged: list                         # [g, g_source] for groups served from another group (-1 = global)

    @property
    def coverage(self):
        return 1.0 - self.alpha

    @classmethod
    def fit(cls, scores, groups, cfg, alpha, edges, z_star, r1, n_min_cfg=5, n_min_atoms=200):
        s, g, c = np.asarray(scores, float), np.asarray(groups), np.asarray(cfg)
        G = 2 * (len(edges) + 1)
        n_at = np.bincount(g, minlength=G)[:G]
        n_cf = np.array([len(np.unique(c[g == k])) for k in range(G)])
        ok = (n_at >= n_min_atoms) & (n_cf >= n_min_cfg)

        def q_of(mask):
            x = s[mask]
            return float(np.quantile(x, conformal_level(len(x), alpha), method="higher"))

        q_glob = q_of(np.ones(len(s), bool))
        q = np.full(G, np.nan); merged = []
        for k in range(G):
            if ok[k]:
                q[k] = q_of(g == k)
        for k in range(G):
            if ok[k]:
                continue
            band, cls_ = divmod(k, 2)
            # nearest populated band of the same coordination class, then the other class of the band
            cands = sorted((abs(b - band), b * 2 + cls_) for b in range(G // 2) if ok[b * 2 + cls_])
            src = cands[0][1] if cands else (band * 2 + (1 - cls_) if ok[band * 2 + (1 - cls_)] else -1)
            q[k] = q[src] if src >= 0 else q_glob
            merged.append([int(k), int(src)])
        return cls(float(alpha), np.asarray(edges, float), int(z_star), float(r1), q, q / chi3_ppf(1 - alpha),
                   n_at, n_cf, merged)

    def to_dict(self):
        return {"alpha": self.alpha, "edges": self.edges.tolist(), "z_star": self.z_star, "r1": self.r1,
                "q_g": self.q_g.tolist(), "lam_g": self.lam_g.tolist(), "n_atoms_g": self.n_atoms_g.tolist(),
                "n_cfg_g": self.n_cfg_g.tolist(), "merged": self.merged}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["alpha"]), np.asarray(d["edges"], float), int(d["z_star"]), float(d["r1"]),
                   np.asarray(d["q_g"], float), np.asarray(d["lam_g"], float), np.asarray(d["n_atoms_g"]),
                   np.asarray(d["n_cfg_g"]), [list(m) for m in d["merged"]])
```

  Note on the merge test: group 1 (band 0, z ≠ z*) has no populated band of its own class. The fallback is the other class of the same band, which is group 0, so `[1, 0]` is recorded.

- [ ] **Step 4: Run** `uv run pytest tests/test_conformal.py -q`. Expected: all pass.
  - If `test_shell_reference_fcc_bcc_and_fallback` fails, print the histogram's peak and minimum bins before touching the thresholds.
  - Keep the 1.25 × peak fallback.
- [ ] **Step 5: Commit:** `feat(conformal): reference-free shell features, Mondrian split-conformal quantiles`.

### Task 2: `ARDPosterior` schema 3 — the served per-group scale

**Files:** modify `src/ace_jax/fit/ard.py` (the `ARDPosterior` class, `SCHEMA`); test `tests/test_ard.py`.

**Interfaces:**
- Consumes `Mondrian` from Task 1.
- Produces:
  - `ARDPosterior` trailing fields: `conformal: dict | None = None` (a `Mondrian.to_dict()`), `cal: dict | None = None` (arrays `scores` f32, `z` i16, `d` f32, `cfg` i64, `src` i8), `support: dict | None = None` (Task 4's arrays);
  - `row_scale2(groups) -> np.ndarray`, the per-row variance multiplier: `lam_g[groups]**2` if `conformal`, else `lam**2` (with Q) or `kappa**2`;
  - `unscaled_atom_var(Frows) -> (N,)`, which is Σ_c of `misspec_var_rows` (with Q) or of `var_rows`;
  - `forces_std(Frows, groups=None)`, `forces_q(Frows, groups)`.

- [ ] **Step 1: Failing tests** (append to `tests/test_ard.py`; reuse the existing `_sandwich_setup` helper):

```python
def test_forces_std_uses_group_lambdas_and_forces_q(tiny_linear_problem, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, sandwich_factor, sandwich_scores
    from ace_jax.fit.conformal import chi3_ppf
    from ace_jax.fit.rows import linear_rows
    with highest_precision():
        prob, ds, ev, h, post = _sandwich_setup(tiny_linear_problem)
        post = post._replace(Q=sandwich_factor(post, sandwich_scores(post, prob, ds, ev.sigmas(h))), lam=3.0)
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
        v = post.unscaled_atom_var(Fr)
        conf = {"alpha": 0.1, "edges": [0.01], "z_star": 12, "r1": 3.0, "q_g": [2.0, 2.5, 5.0, 6.0],
                "lam_g": [2.0 / chi3_ppf(0.9), 2.5 / chi3_ppf(0.9), 5.0 / chi3_ppf(0.9), 6.0 / chi3_ppf(0.9)],
                "n_atoms_g": [1, 1, 1, 1], "n_cfg_g": [1, 1, 1, 1], "merged": []}
        pc = post._replace(conformal=conf)
        g = np.arange(len(v)) % 4
        sd = pc.forces_std(Fr, g)
        np.testing.assert_allclose(sd, np.asarray(conf["lam_g"])[g] * np.sqrt(v), rtol=1e-12)
        np.testing.assert_allclose(pc.forces_q(Fr, g), np.asarray(conf["q_g"])[g] * np.sqrt(v / 3), rtol=1e-12)
        np.testing.assert_allclose(post.forces_std(Fr), 3.0 * np.sqrt(v), rtol=1e-12)      # no conformal: lam
        cal = {"scores": np.ones(5, np.float32), "z": np.full(5, 12, np.int16), "d": np.zeros(5, np.float32),
               "cfg": np.arange(5), "src": np.zeros(5, np.int8)}
        pc = pc._replace(cal=cal)
        pc.save(tmp_path / "p.npz")
        back = ARDPosterior.load(tmp_path / "p.npz")
        assert back.conformal["q_g"] == conf["q_g"] and np.array_equal(back.cal["cfg"], cal["cfg"])
        np.testing.assert_allclose(back.forces_std(Fr, g), sd, rtol=1e-3)


def test_schema2_posterior_serves_lambda(tiny_linear_problem, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, sandwich_factor, sandwich_scores
    from ace_jax.fit.rows import linear_rows
    with highest_precision():
        prob, ds, ev, h, post = _sandwich_setup(tiny_linear_problem)
        post = post._replace(Q=sandwich_factor(post, sandwich_scores(post, prob, ds, ev.sigmas(h))), lam=2.0)
        post.save(tmp_path / "p.npz")
        z = dict(np.load(tmp_path / "p.npz")); z["schema"] = np.array(2)
        for k in [k for k in z if k.startswith(("conformal", "cal_", "support"))]:
            z.pop(k)
        np.savez(tmp_path / "p2.npz", **z)
        old = ARDPosterior.load(tmp_path / "p2.npz")
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
        assert old.conformal is None
        np.testing.assert_allclose(old.forces_std(Fr, np.zeros(Fr.shape[0], int)), post.forces_std(Fr), rtol=1e-3)
        with pytest.raises(ValueError, match="conformal"):
            old.forces_q(Fr, np.zeros(Fr.shape[0], int))
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ard.py -q -k "group_lambdas or schema2_posterior"`. Expected: FAIL.

- [ ] **Step 3: Implement** in `ard.py`.
  - Set `SCHEMA = 3`.
  - Add the three trailing fields after `lam`.
  - Add these methods:

```python
    def row_scale2(self, groups):
        """Per-atom variance multiplier of the served scale: lam_g^2 (conformal), else lam^2 (sandwich)
        or kappa^2 (epistemic)."""
        if self.conformal is not None:
            return np.asarray(self.conformal["lam_g"], float)[np.asarray(groups)] ** 2
        s = self.lam if self.Q is not None else self.kappa
        return np.full(len(groups), s ** 2)

    def unscaled_atom_var(self, Frows):
        """v(x) per atom: sum over the 3 force components of the sandwich (Q set) or epistemic variance."""
        P = Frows.reshape(-1, Frows.shape[-1])
        v = self.misspec_var_rows(P) if self.Q is not None else self.var_rows(P)
        return np.maximum(v.reshape(-1, 3).sum(1), 0.0)

    def forces_std(self, Frows, groups=None):
        """Per-atom served force std: lam_g * sqrt(v) (conformal; groups required), else the scalar scale."""
        v = self.unscaled_atom_var(Frows)
        if self.conformal is None or groups is None:
            s = self.lam if self.Q is not None else self.kappa
            return s * np.sqrt(v)
        return np.sqrt(self.row_scale2(groups) * v)

    def forces_q(self, Frows, groups):
        """Guaranteed radius q_g * sqrt(v/3): P(|dF| <= forces_q) >= coverage for atoms exchangeable with g."""
        if self.conformal is None:
            raise ValueError("forces_q needs a conformally calibrated posterior (fit --ard-calibration conformal)")
        q = np.asarray(self.conformal["q_g"], float)[np.asarray(groups)]
        return q * np.sqrt(self.unscaled_atom_var(Frows) / 3.0)
```

  - Keep `force_var_rows(Phi)` as is: E/V paths and older callers use it. Add `force_var_rows(Phi, groups=None)`, which when `conformal` and `groups` (per atom) are set returns `np.repeat(self.row_scale2(groups), 3) * unscaled_row_var`.
  - **`save`** adds:
    - `conformal_json` (uint8 JSON bytes) when set;
    - `cal_scores`, `cal_z`, `cal_d`, `cal_cfg`, `cal_src` when set;
    - Task 4's `support_*` keys.
  - **`load`** accepts schema ∈ (1, 2, 3) and rebuilds the dicts when present.
- [ ] **Step 4: Run** `uv run pytest tests/test_ard.py -q`. Expected: all pass (the old tests are unaffected because the defaults are None).
- [ ] **Step 5:** full suite + ruff; commit `feat(ard): schema 3 -- per-group conformal scale, forces_q, unscaled_atom_var`.

### Task 3: The ARD stage fits λ_g; `predict_ard` serves it; config

**Files:** modify `src/ace_jax/fit/ard.py` (`_val_errors`, `run_ard_stage`, `predict_ard`) and `src/ace_jax/fit/pipeline/config.py`; tests in `tests/test_ard.py` and `tests/test_ard_pipeline.py`.

**Interfaces:**
- Consumes Tasks 1–2.
- Produces:
  - `_val_errors(..., features_r1=None)`, which also returns per held-out atom `(z, d, cfg_index_in_ds)` when `features_r1` is given (same order as e2);
  - `run_ard_stage` sets `post.conformal` and `post.cal`;
  - `report["conformal"]` = the Mondrian dict plus `val_coverage` (in-sample, by construction ≥ 1 − α);
  - `predict_ard` uses per-atom λ_g.

- [ ] **Step 1: Failing tests.**
  - In `tests/test_ard.py`, `_pipe_cfg` keeps `ard_calibration="gaussian"` in its base dict, so the existing κ/λ tests still test those paths unchanged. Append:

```python
def test_ard_stage_conformal_groups_and_coverage():
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.conformal import assign_groups, chi3_ppf, shell_features
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg(ard_variance="sandwich", ard_calibration="conformal", ard_n_min_cfg=1, ard_n_min_atoms=5).validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        pred = ard.predict_ard(res.posterior, b.prob, d.ds_test)
    post, rep = res.posterior, res.report
    c = post.conformal
    assert c is not None and len(c["q_g"]) == 2 * (len(c["edges"]) + 1) and np.isfinite(c["q_g"]).all()
    np.testing.assert_allclose(c["lam_g"], np.asarray(c["q_g"]) / chi3_ppf(0.9))
    # the stored calibration scores reproduce the per-group quantiles
    g = assign_groups(post.cal["z"], post.cal["d"], c["z_star"], np.asarray(c["edges"]))
    assert rep["conformal"]["val_coverage"] >= 0.9 - 1e-12
    for k in set(g.tolist()) - {m[0] for m in c["merged"]}:
        s = np.sort(post.cal["scores"][g == k].astype(float))
        assert np.isclose(c["q_g"][k], s, rtol=1e-6).any()
    # predict_ard: F_var rows = lam_g(atom)^2 * unscaled rows, groups from the test batch geometry
    assert np.all(pred.F_var >= 0) and np.isfinite(pred.F_var).all()
```

  - In `tests/test_ard_pipeline.py`, extend `test_config_validates_ard` with these, plus the defaults (`"conformal"`, 0.9, `"distortion"`):

```python
    with pytest.raises(ValueError, match="ard_calibration"):
        _cfg(ard_calibration="nope").validate()
    with pytest.raises(ValueError, match="ard_coverage"):
        _cfg(ard_coverage=1.0).validate()
    with pytest.raises(ValueError, match="ard_groups"):
        _cfg(ard_groups="nope").validate()
```

- [ ] **Step 2: Run** the two files. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **`config.py`:** add the fields from Global Constraints, validated next to `ard_variance` (`0 < ard_coverage < 1`, and the enums). Each message must name its field.
  - **`_val_errors`:** add `features_r1=None`. In the same per-batch loop, when set, compute `z, d = shell_features(b, features_r1)` and keep `z[live]`, `d[live]`, `off + node_cfg[live]`. Return them as a 4th element `(z, d, cfg)`, and keep the 3-tuple when `features_r1` is None so existing callers are untouched.
  - **`run_ard_stage`:**
    1. **Before the full-posterior held-out pass**, when `cfg.ard_calibration == "conformal"`, compute
       the shell reference from the training batches:

```python
    r1 = z_star = None
    if cfg.ard_calibration == "conformal":
        from .conformal import shell_features, shell_reference
        nb = min(data.ds_train.n_batches, 200)                     # a sample of training batches
        bts = [jax.tree.map(lambda a, i=i: a[i], data.ds_train) for i in range(nb)]
        dists = np.concatenate([np.linalg.norm(np.asarray(bt.rij), axis=-1)[np.asarray(bt.nbr_mask)] for bt in bts])
        r1, _ = shell_reference(dists)
        zs = np.concatenate([shell_features(bt, r1)[0][np.asarray(bt.node_mask)] for bt in bts])
        z_star = int(np.bincount(zs).argmax())
```

    2. **Make the existing single held-out pass** (the `_val_errors(post, prob, ds_val, own_col=...)`
       call) also return features when conformal:

```python
    vals = _val_errors(post, prob, ds_val, own_col=idx[:nval] if variance == "sandwich" else None,
                       features_r1=r1)
    _, s2_full, m2_full = vals[:3]
    zv, dv, cv = vals[3] if r1 is not None else (None, None, None)
```

    3. **Leave the κ and λ fits exactly as they are.** After them, when conformal:

```python
    if r1 is not None:
        from .conformal import Mondrian, assign_groups, band_edges
        vcal = m2_full if variance == "sandwich" else s2_full        # already masked by ok above
        zv, dv, cv = zv[ok], dv[ok], cv[ok]
        scores = np.sqrt(e2 / (vcal / 3.0))
        if cfg.ard_groups == "distortion":
            edges = band_edges(dv, cfg.ard_group_bands)
            g = assign_groups(zv, dv, z_star, edges)
        else:
            edges, g = np.zeros(0), np.zeros(len(scores), np.int64)
        mond = Mondrian.fit(scores, g, cv, 1 - cfg.ard_coverage, edges, z_star, r1,
                            cfg.ard_n_min_cfg, cfg.ard_n_min_atoms)
        post = post._replace(conformal=mond.to_dict(),
                             cal={"scores": scores.astype(np.float32), "z": zv.astype(np.int16),
                                  "d": dv.astype(np.float32), "cfg": cv.astype(np.int64),
                                  "src": np.zeros(len(scores), np.int8)})
        cov = float(np.mean(scores <= mond.q_g[g]))
        log(f"ARD: conformal ({cfg.ard_coverage:.2f}) over {len(scores)} held-out atoms, {len(mond.q_g)} groups; "
            f"lam_g {np.round(mond.lam_g, 2).tolist()} (scalar lam {lam:.2f})")
```

    Here `m2_full` is the leave-own-out sandwich array that the λ fit already uses (`m2_full[1][ok]`
    in the current code). Bind `vcal` to that same masked array, not the with-own one.

    Add `report["conformal"] = {**conf, "val_coverage": cov, "n_val_atoms": len(scores)}`.
  - **`predict_ard`:** when `post.conformal`, compute `z, d = shell_features(b, r1)` per batch, the groups `g` per node (all Ncap nodes; padding is harmless), and `F_var = (np.repeat(post.row_scale2(g), 3) * unscaled_rows).reshape(-1, 3)`, where `unscaled_rows = post.misspec_var_rows(F) if post.Q is not None else post.var_rows(F)` (the shape v, per force row, before any scale). Otherwise keep `force_var_rows(F)`.
- [ ] **Step 4: Run** the ARD test files. Expected: pass. Then run the full suite: with the new default `"conformal"`, the CLI/calculator tests now take that path and must still pass. Fix anything that assumed scalar λ in served σ, using its fixture's `ard_calibration`.
- [ ] **Step 5:** ruff; commit `feat(ard): Mondrian conformal lam_g from the train hold-out (default); predict_ard serves it`.

### Task 4: `fit/support.py` — covariate-shift support reference and check

**Files:** create `src/ace_jax/fit/support.py`; modify `run_ard_stage` (to build the reference) and `ARDPosterior` save/load (the `support_*` keys); test `tests/test_conformal.py`.

**Interfaces:**
- Produces:
  - `SupportReference`, a NamedTuple: `species` (S,), `mu`/`sd`/`W` (per species), `P` (per species, whitened pool, float32), `scores` (per species), `pair_cols` (int array);
  - `build_support(X, Z, scores, pair_cols, max_atoms=50000, seed=0) -> SupportReference`;
  - `support_check(ref, X_t, Z_t, alpha) -> dict(support_ok (N,) bool, support_q (N,) float, n_eff {species: float})`;
  - `ARDPosterior.support` holds `SupportReference._asdict()` arrays.

- [ ] **Step 1: Failing tests** (append to `tests/test_conformal.py`)

```python
def test_support_check_flags_out_of_support():
    from ace_jax.fit.support import build_support, support_check
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 6)); Z = np.zeros(4000, int); s = rng.chisquare(3, 4000) ** 0.5
    ref = build_support(X, Z, s, np.arange(6))
    Xin = rng.normal(size=(500, 6))
    Xout = rng.normal(size=(500, 6)) + 12.0                          # disjoint target cluster
    a = support_check(ref, np.r_[Xin, Xout], np.zeros(1000, int), 0.1)
    assert a["support_ok"][:500].mean() > 0.95 and a["support_ok"][500:].mean() < 0.05
    b = support_check(ref, Xin, np.zeros(500, int), 0.1)
    assert b["n_eff"][0] > 0.5 * 4000                                # in distribution: weights ~flat
    assert np.all(np.isfinite(b["support_q"]))
```

- [ ] **Step 2: Run** `uv run pytest tests/test_conformal.py -q -k support`. Expected: FAIL.
- [ ] **Step 3: Implement** `src/ace_jax/fit/support.py`:
  - the whitener per species: standardise, SVD, keep components with variance > 1e-10 × max;
  - an L2 logistic regression via `scipy.optimize.minimize` (L-BFGS-B, λ = 1e-2, class-balanced, as in `bench/defect_uq/scoring/conformal_shift.py` lines `def logistic`);
  - weights `w = exp(clip(logodds, -30, 30))`;
  - the per-target weighted quantile with mass `w(x)/(Σw + w(x))` at +∞ (`weighted_q` from the same script);
  - `n_eff = (Σw)² / Σw²` of the pool weights;
  - pool subsampling to `max_atoms`, uniformly per species, with the seed.

  In `run_ard_stage`, when `cfg.ard_support` and conformal:
  - `X_val = site_features(prob.model, prob.cfg, ds_val)` (as in `bench/defect_uq/scoring/descriptors.py`), with live rows selected exactly as in `_val_errors`;
  - `pair_cols` = the columns with `len(nnll[b]) == 1` plus `arange(n_B, n_B + n_pair)`, from `data.meta["nnll"]`;
  - store `build_support(X_val, Z_val, scores, pair_cols, cfg.ard_support_max_atoms)._asdict()` in `post.support`;
  - `save`/`load` write and read the `support_*` arrays.
- [ ] **Step 4: Run** the conformal and ARD tests, then the full suite.
- [ ] **Step 5:** ruff; commit `feat(support): covariate-shift weighted-conformal support flag (diagnostic)`.

### Task 5: Calculator — groups, `forces_q`, `ard_group`, `forces_support`

**Files:** modify `src/ace_jax/calc/point.py` (`implemented_properties`, `calculate`'s on-request path, `_forces_std`); test `tests/test_ard_calc_cli.py`.

**Interfaces:**
- Consumes Tasks 2–4.
- Produces the properties `forces_q` (N,), `ard_group` (N,) int and `forces_support` (a dict, via `get_property`). `forces_std` now uses λ_g.

- [ ] **Step 1: Failing tests** (append to `tests/test_ard_calc_cli.py`; the module fixture `fitted` now has the default conformal calibration):

```python
def test_calculator_conformal_matches_pipeline_and_lean(fitted):
    """forces_std / forces_q / ard_group agree with predict_ard and between lean=True/False."""
    from ase.io import read
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.conformal import chi3_ppf
    post = ARDPosterior.load(fitted / "posterior.npz")
    assert post.conformal is not None
    at = max(read(XYZ, ":8"), key=len)
    out = {}
    for lean in (True, False):
        calc = ACECalculator(str(fitted / "model.npz"), posterior=str(fitted / "posterior.npz"), lean=lean)
        a = at.copy(); a.calc = calc
        a.get_forces()
        sd = np.asarray(calc.get_property("forces_std", a)); q = np.asarray(calc.get_property("forces_q", a))
        g = np.asarray(calc.get_property("ard_group", a))
        out[lean] = (sd, q, g)
        # the guaranteed radius is the Gaussian (1 - alpha) ball of forces_std, group by group
        np.testing.assert_allclose(q, chi3_ppf(1 - post.conformal["alpha"]) * sd / np.sqrt(3), rtol=1e-10)
        assert g.shape == (len(at),) and g.min() >= 0 and g.max() < len(post.conformal["q_g"])
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
    assert sup["support_ok"].shape == (len(at),) and sup["support_q"].shape == (len(at),)
    assert sup["support_ok"].dtype == bool and len(sup["n_eff"]) >= 1
    assert all(v > 0 for v in sup["n_eff"].values())
    ok = sup["support_ok"]
    assert np.all(np.isfinite(sup["support_q"][ok])) and np.all(np.isinf(sup["support_q"][~ok]))
```

  The pipeline-equality check is the existing `test_calculator_forces_std_matches_pipeline`, which must keep passing on the conformal path: `predict_ard`'s F σ = √(ΣF_var) equals the calculator's `forces_std` for the same configs.

- [ ] **Step 2: Run** them. Expected: FAIL (unknown property `forces_q`).
- [ ] **Step 3: Implement.**
  - **`_forces_std`:** compute `z, d = shell_features(b, post.conformal["r1"])` and `g = assign_groups(...)` on `b` (all nodes), then mask with `node_mask`. Return `(sd, q, g)` to the callers, which set `results["forces_std"]`, `results["forces_q"]` (only when conformal) and `results["ard_group"]`.
  - **The on-request fast path** in `calculate` accepts `properties ⊆ {"forces_std", "forces_q", "ard_group", "forces_support"}`.
  - **`forces_support`** (on request only):
    - target descriptors from the full `self._fit_model` via `site_features(self._fit_model, self._fit_cfg, ds)` on the same one-config batch, masked;
    - then `support_check(SupportReference(**post.support), X, Z, post.conformal["alpha"])`;
    - this raises `ValueError` when `post.support` is None ("fit with --ard-support").
  - Add the new names to `implemented_properties`.
- [ ] **Step 4: Run** `uv run pytest tests/test_ard_calc_cli.py -q`, then the full suite.
- [ ] **Step 5:** ruff; commit `feat(calc): conformal forces_std/forces_q, ard_group, forces_support`.

### Task 6: CLI — fit flags, `aj calibrate`, eval outputs

**Files:** modify `src/ace_jax/cli.py`; test `tests/test_ard_calc_cli.py`.

**Interfaces:**
- Consumes Tasks 1–5.
- Produces:
  - `aj calibrate --model M --posterior P --data X [--energy-key k] [--force-key k] [--virial-key k] [--coverage c] [--replace] --out O`;
  - `aj eval --per-atom` writes `forces_q` and `ard_group`; `--support` adds `support_ok` and `support_q`.

- [ ] **Step 1: Failing tests**

```python
def test_calibrate_appends_and_replaces(fitted, tmp_path):
    from ace_jax.cli import main
    from ace_jax.fit.ard import ARDPosterior
    base = ARDPosterior.load(fitted / "posterior.npz")
    args = ["calibrate", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
            "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial"]
    assert main(args + ["--out", str(tmp_path / "app.npz")]) == 0
    app = ARDPosterior.load(tmp_path / "app.npz")
    assert len(app.cal["scores"]) > len(base.cal["scores"]) and set(np.unique(app.cal["src"])) == {0, 1}
    assert app.conformal["edges"] == base.conformal["edges"]            # band edges fixed at fit time
    assert main(args + ["--replace", "--out", str(tmp_path / "rep.npz")]) == 0
    rep = ARDPosterior.load(tmp_path / "rep.npz")
    assert set(np.unique(rep.cal["src"])) == {1}
    np.testing.assert_array_equal(rep.mean, base.mean)                 # calibration never touches the readout


def test_calibrate_refuses_schema2(fitted, tmp_path):
    from ace_jax.cli import main
    z = dict(np.load(fitted / "posterior.npz")); z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("conformal", "cal_", "support"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    with pytest.raises(ValueError, match="ard-calibration conformal"):
        main(["calibrate", "--model", str(fitted / "model.npz"), "--posterior", str(tmp_path / "p2.npz"),
              "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force", "--out", str(tmp_path / "o.npz")])


def test_calibrate_requires_forces(fitted, tmp_path):
    from ace_jax.cli import main
    with pytest.raises(ValueError, match="no_such_force"):
        main(["calibrate", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
              "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "no_such_force", "--out", str(tmp_path / "o.npz")])


def test_eval_per_atom_writes_conformal_arrays(fitted, tmp_path):
    from ase.io import read
    from ace_jax.cli import main
    assert main(["eval", "--model", str(fitted / "model.npz"), "--posterior", str(fitted / "posterior.npz"),
                 "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
                 "--per-atom", str(tmp_path / "pa.xyz"), "--support"]) == 0
    a = read(tmp_path / "pa.xyz", "0")
    for k in ("forces_std", "forces_q", "ard_group", "support_ok", "support_q"):
        assert k in a.arrays, k
```

- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Fit flags:** `--ard-calibration {conformal,gaussian}` (default conformal), `--ard-coverage` (0.9), `--ard-groups {distortion,none}`, `--no-ard-support`. Thread them into `FitConfig` like `--ard-variance`.
  - **`cmd_calibrate`:**
    1. Load the posterior; if `conformal` or `cal` is None, raise `ValueError("... refit with --ard-calibration conformal")`.
    2. Read the data with `load_configs(data, ek, fk, vk)`. If no config has forces, raise `ValueError(f"... no forces under --force-key {fk!r}")`.
    3. Build an `ACECalculator(model, posterior=…)`. Per config, compute the served-model forces (`get_forces`), `v` (`post.unscaled_atom_var` of the calculator's rows) and `z, d` from the batch.
    4. Compute the scores `|ΔF| / √(v/3)`, using only atoms with v > 0.
    5. Combine with the stored set (`--replace` uses the new one only), with `src = max(src)+1` and `cfg` offset by `max(cfg)+1`.
    6. Rerun `Mondrian.fit` with the stored edges, z*, r1 and coverage (`--coverage` overrides alpha).
    7. When the posterior has support, rebuild it on the combined atoms (needs their descriptors from `site_features`).
    8. Save to `--out`, print the before/after `q_g` and `n_atoms_g` table and the sha256 of `--data`, and record the provenance in `conformal["sources"]` (list of `{path, sha256, n_cfg, n_atoms}`).
  - **eval:** with `--per-atom`, also write `forces_q` and `ard_group`; with `--support`, `support_ok` and `support_q`.
- [ ] **Step 4: Run** the file, then the full suite.
- [ ] **Step 5:** ruff; commit `feat(cli): aj calibrate (target-regime conformal); eval forces_q / ard_group / --support`.

### Task 7: Docs and the acceptance run

**Files:** `README.md`, `skills/ace-jax/SKILL.md`, `bench/defect_uq/scoring/conformal_cv.py` (port), `bench/defect_uq/README.md`.

- [ ] **Step 1: README `--uq ard` paragraph.**
  - The served scale is conformal per group by default (`--ard-coverage 0.9`); `forces_q` is the guaranteed radius.
  - `aj calibrate` recalibrates on a labelled target-regime set.
  - `forces_support` is the diagnostic.
  - `--ard-calibration gaussian` keeps the scalar λ.
  - Link the math doc.
- [ ] **Step 2: SKILL.md gotchas.**
  - The guarantee holds for atoms exchangeable with a group's calibration atoms. For a new regime, run `aj calibrate` on a few labelled cells like it.
  - `support_ok = False` marks atoms the calibration cannot certify; they are candidates to label.
- [ ] **Step 3: Port `conformal_cv.py`'s method M** to call the library (`fit/conformal.py`). Then run the acceptance. On Modal, fit arm `ard` (now conformal by default) with the `bench365` launcher; then `big_errors` on the v3 files; then `aj calibrate` leave-one-realisation-out. Expected, per the spec:
  - in-distribution held-out coverage 0.90 ± 0.01;
  - crack whole cell ≥ 0.89;
  - tip ≥ 0.88;
  - edge/screw ≥ 0.90.

  If the reference-free distortion misses these, record it in `bench/defect_uq/README.md` and add `--ard-groups lattice` (the benchmark's a0/√2 feature) as a follow-up ruling.
- [ ] **Step 4:** full suite, `uvx pre-commit run --all-files`; commit `docs: conformal force sigma (aj calibrate, forces_q, support flag)`.
