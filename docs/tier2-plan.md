# Tier 2: in-memory hand-off + coupling cache

**Goal.** After Tier 1 (PR #4) a model is authored in Python, but it still
round-trips through `save_npz` → `eval.io.load`, and every `couple()` call
launches Julia. Tier 2 (points 1–2 of the roadmap; the pure-JAX coupling
endgame stays out of scope, and the EquivariantTensors bridge is kept):

1. **In-memory hand-off** — Python callers hold the `ACEModel` directly; the
   npz writer becomes strictly compatibility-only.
2. **Coupling cache** — authoring an *existing* shape never launches Julia:
   the shim runs once per new shape, results are persisted per shape, and a
   pip install + cache dir is Julia-free.

## 1. In-memory hand-off

### State of play

`ACECalculator` and `site_descriptors` already accept a `(model, meta)` pair
(`eval/api.py::_resolve`); the eval graph never reads the npz itself. The
actual gap: nothing guarantees *meta* agrees with the (possibly patched)
model tree. Tier 1 put that derivation in `save_npz` — so today the npz file
is paradoxically the "trust anchor" for branch selectors, and the in-memory
tree is second-class.

### Design

- **`Authoring.eval_pair()`** (`construct/model.py`): returns `(model, meta)`
  with the four branch-selector keys (`radial_kind`, `pair_radial_kind`,
  `pair_envelope_kind`, `ybasis_kind`) derived from the tree — the same
  derivation `save_npz` performs — plus light structural checks (`elements`
  width vs `WB`, `n_B`/`n_AA` vs `A2B`, `aa_lens` vs `len(aa_specs)`,
  `lmax`). Meta is deep-copied; callers may mutate freely.
- **`save_npz` refactors to call `eval_pair()`** — one source of truth for
  the derivation, so the file writer and the in-memory path can never drift.
- **Consumers** (no new machinery, just the documented entry points):
  `ACECalculator(*auth.eval_pair())`, `site_descriptors(auth.model, …,
  meta=meta)`, and `model.energy_forces_virial(...)` directly. The tree's
  dtype is whatever it holds (f64 by default); `load()`'s dtype handling
  applies only to the file path.
- **`save_npz` demoted to compat-only**: ACEfit-fitted interchange and shell
  hand-off (`ace-jax construct --out`). The module docstring keeps the
  round-trip test as the schema drift alarm.

### Tests

- pip-only (no Julia): `load()` the committed Si fixture, evaluate through
  the pair path (`ACECalculator(model, meta)`, `site_descriptors(model, …,
  meta=…)`) and assert identical E/F/stress/descriptors to the path-based
  calculator. Pins the pair path with zero Julia.
- Authoring extra (bridge subprocess): after coefficient injection, evaluate
  the injected tree **in memory** and assert agreement with the
  round-tripped-file results to 1e-12 — proves the loader applies no
  transformation the in-memory path would skip.

## 2. Coupling cache

### Design (`construct/coupling.py`)

- **Key**: canonical JSON of `{"schema": 1, "mb": …, "rnl": …, "ylm": …}` —
  order-preserving (B-row order and spec order are meaningful; `sort_keys`
  only normalises dict key order) — hashed with sha256 → filename
  `<hash16>.npz`.
- **Entry** (npz, no `allow_pickle`): `A2B`, `aa_spec_1..K`, `aspec` as int
  arrays; a JSON blob (the `np.frombuffer` pattern from `export.py`) for the
  ragged `aa_sig` / `nnll_spec`, the **input specs** (hit-time re-check
  against the request — insurance beyond the hash), and the `juliapkg.json`
  **pin hash**.
- **Pin hash without Julia**: scan `sys.path` for `juliapkg.json` (the same
  discovery rule juliapkg uses) and hash its bytes. On a cache *hit* nothing
  imports juliacall — that is the payoff: pip install + a populated cache
  dir authors a known shape Julia-free. Pin mismatch → recompute.
- **Locations**: `ACEJAX_COUPLING_CACHE` (value `none` disables) wins, else
  `$XDG_CACHE_HOME/ace-jax/coupling` (default `~/.cache/...`). Writes are
  atomic (tmp + `os.replace`) and best-effort (an unwritable cache dir
  degrades to always-recompute, never an error).
- **API**: `couple_cached(mb_spec, Rnl_spec, Ylm_spec, cache_dir=None)`
  wrapping the unchanged `couple()` (kept as the uncached parity oracle —
  `tests/test_coupling_parity.py` keeps calling it directly);
  `build_model(..., coupling_cache=True, coupling_cache_dir=None)` routes
  through the cache.
- **CLI**: `ace-jax construct --no-coupling-cache` escape hatch; cache on by
  default. Shipping = copy the entry npz into a team's cache dir (entries
  are self-describing).

### Tests

- pip-only, monkeypatched `couple` (no juliacall anywhere): miss → raw called
  once, entry written; hit → identical `Coupling`, raw not called; mb_spec
  order change → different key; stored-specs mismatch or pin-hash change →
  recompute; env handling of the cache dir.
- Authoring extra (subprocess): run 1 builds a small shape (order 2,
  degree 5) with a temp cache dir; run 2 repeats with `ACEJAX_NO_JULIA=1`
  set (a check in `_jl()` raises if Julia is ever touched) — the build must
  succeed from cache alone and produce the same `A2B`.

## Sequencing & traps

- Commit A = point 1 (small, pip-testable); commit B = point 2. One branch,
  both fold into PR #4.
- Traps respected: juliacall's `sys.path` scan applies to *misses* only;
  JAX clamps OOB branch indices silently (why `eval_pair` derives rather
  than trusts meta); cache entries are shape-exact, so the degenerate-nnll
  row-rotation issue never arises; ragged specs are stored as JSON, not
  pickled object arrays.
- Docs: this file; `python-authoring.md` gains "In-memory hand-off" and
  "Coupling cache" sections; its roadmap shrinks to the pure-JAX coupling
  endgame.
