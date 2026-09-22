# Python model authoring (Tier 1)

**Goal.** Author a complete frozen `ace_model`-family model — `ACEModel` + `meta`
— entirely from Python: `ace-jax construct --elements Si --order 3 --max-degree 10
--out si.npz`. Julia's role shrinks to the one thing it still owns, the SO(3)
coupling via the [EquivariantTensors shim](coupling-etshim-spec.md); radial
init, pair basis, readout, packaging and evaluation are pure Python/NumPy/JAX.

## The authoring ladder

`build_model(elements, order, totaldegree, *, wL, rcut, rin, radial_mode,
pair_mode, seed, with_gamma, edge_a_kind)` (`construct/model.py`) walks the same
steps the Julia exporter walks, returning an `Authoring` NamedTuple:

1. **`resolve_elements`** — atomic numbers or symbols (`"14"`, `14`, `"Si"`) →
   sorted `zs`.
2. **`build_spec(NZ, order, totaldegree, wL)`** (`construct/spec.py`) — the three
   integer specs ET consumes: `mb_spec` (per-B `(n,l)` tuples under
   `TotalDegree` + `rpe_admissible`), `Rnl_spec`, `Ylm_spec`.
3. **`couple(mb_spec, Rnl_spec, Ylm_spec)`** (`construct/coupling.py`) — the
   in-process JuliaCall shim; returns `A2B` `(n_B, n_AA)`, per-order `aa_specs`,
   `aspec`, and ET's `aa_sig` per AA column (signature-sorted, **not**
   evaluation-ordered — see below).
4. **nnll cross-check** — `nnll_from_coupling` recomputes each B row's body
   list from the block-diagonal `A2B` (first nonzero column → `aa_sig`, with
   multiplicity) and asserts it matches the shim dump as a row-multiset.
   Catches any ET-version drift in one place.
5. **`tensor_radial_init` / `pair_radial_init`** (`construct/radial_init.py`) —
   seeded radial coefficients, frozen. `mode="glorot_normal"` (tensor),
   `"onehot"` (pair, `spl_n = δ(n, 2Z)`-style identity rows) or `"zeros"`;
   the 7-tuple Agnesi transform `(n_deg, l_shift, a, y0=0, r0, rcut, ycut)` is
   derived from `(n_deg, l_shift, r0, rcut)` and never stored independently.
6. **Zero readout** — `WB`, `Wpair`, `E0` zeros (the `acefit!`-fresh init), then
   `fold_readout` bakes the species pooling into the coefficient tables.
7. **`smoothness_prior`** (`construct/prior.py`, ported in PR #3) — the
   `len_basis` γ vector, over tensor rows repeated per species and pair rows
   `[(n,0)]`; `--no-gamma` / `with_gamma=False` skips it.

`save_npz(path, authoring)` (`construct/export.py`) writes the disposable
npz bridge file.

## Writer routes on the model, not on defaults

`save_npz` **derives the branch-selector meta keys from the model tree being
saved**: `radial_kind = model.radial_kind`, `pair_radial_kind =
model.pair_radial_kind`, `pair_envelope_kind = model.pair_envelope_kind`,
`ybasis_kind` from `model.ysolid`. Everything else in `meta` is copied
(deep-copied through JSON, so callers may mutate their copy freely).

Rationale: the loader (`eval/io.py`) reconstructs the eval graph from
`meta` alone — `radial_kind`/`pair_radial_kind` pick the radial branch,
`pair_envelope_kind` the pair envelope formula, `rnl_spline`/`pair_spline`
the `(x0, h, n)` grids. If the writer echoed authoring defaults while the
arrays came from a patched model (e.g. the fixture-injection test swaps
analytic radials for spline ones), the loader would silently wire the
placeholder branch `(0.0, 1.0, 2)` and the wrong envelope — coefficients
round-trip but energies diverge. Deriving from the tree makes the writer
self-consistent with *any* patched tree, which is what lets the bridge test
inject fitted coefficients.

JAX caveat this guards against: `env_ace1_poly1sr` reads `params[..., 2]`
silently clamped for a 2-column `PolyEnvelope1sR`, and OOB indices are clamped,
not errors — a wrong branch selector fails numerically, not loudly.

## Evaluation parity (the bridge test)

`tests/test_python_authoring.py::test_bridge_wellformed_subprocess` covers the
whole chain against `fixtures/si_ace_model.npz` (Si, order 3, TotalDegree 10):

- **Structural**: authored `nnll_spec` matches the fixture as a row-multiset;
  feeding the fixture's own nnll row order back through `couple` reproduces
  `A2B` bit-for-bit (the injection premise — coefficient rows are only
  interchangeable when the map is).
- **Injected + evaluated**: the fixture's fitted coefficients and fitted
  branch kinds (`radial_kind="analytic"`, `pair_radial_kind="spline"`,
  `pair_envelope_kind="poly1sr"`, spline grid `(x0, h, n)` from
  `meta["pair_spline"]`) are patched in via `dataclasses.replace`
  (eqx `ACEModel` supports it; the `Authoring` tuple via `._replace`),
  packaged with `save_npz`, reloaded with `eval.io.load`, and every table is
  compared against the fixture. Then the round-tripped file runs through the
  native eval path (`ACECalculator`) and must reproduce Julia's energies,
  forces, stress and descriptors to float noise (`< 1e-8`, E exact).

Two conventions to know when reading fixtures:

- **Probe rows are tensor-side** (`probe_r = range(0.9, maxrcut, 24)`, seeded):
  `probe_x` is `rbasis.transforms[1,1](r)` (the *tensor* basis transform —
  `agnesi_normalized(r, rnl_transform)`, NOT the pair transform),
  `probe_env` the tensor envelope, `probe_Rnl` the tensor radial. The one
  pair-basis row is `probe_Rpair = spl(x_ij) * e_ij` from
  `evaluate_batched(m.pairbasis, ...)`.
- **Virial vs stress**: the fixture's `test_V` is the *Julia virial*
  (`site_virial = -sum(dv_i r_i')`, i.e. `V = -dE/dε`); ASE's
  `get_stress(voigt=False)` is `stress = -virial / volume`. The eval path
  (`calc/point.py`) stores ASE-convention stress; compare with
  `S + V/vol ≈ 0`.

## In-memory hand-off (Tier 2, point 1)

`auth.eval_pair()` returns `(model, meta)` with the branch-selector keys
derived from the tree (the same derivation `save_npz` performs — one source
of truth) plus structural meta/tree checks. The pair evaluates directly:
`ACECalculator(*auth.eval_pair())`, `site_descriptors(auth.model, ...,
meta=meta)`, `model.energy_forces_virial(...)` — no npz round-trip.
`save_npz` is the compatibility path (ACEfit-fitted interchange, shell
hand-off), not a required step.

Trap this pins: replacing `WB` on a **folded** model leaves a stale `ctilde`
(the bridge test hit exactly this — the round-trip masked it because
`load()` rebuilds with `folded=False`, but the in-memory tree is evaluated
as-is). Re-fold after any `dataclasses.replace` that touches `WB`.

## Coupling cache (Tier 2, point 2)

The coupling depends only on the three integer specs, so
`couple_cached` (default inside `build_model`) persists one entry per shape —
keyed by a sha256 of the order-preserving spec JSON — under
`$ACEJAX_COUPLING_CACHE` (or `~/.cache/ace-jax/coupling`). A **hit
reconstructs the `Coupling` without importing juliacall**: pip install +
populated cache dir = Julia-free authoring of a known shape. Entries store
their input specs (hit-time re-check) and the `juliapkg.json` pin hash (a
pin change invalidates); writes are atomic and best-effort. `couple()` stays
the uncached parity oracle; `--no-coupling-cache` / `cache_dir="none"` opt
out. Entries are self-describing npz files — ship one by copying it into a
team's cache dir.

## Known traps

- **juliapkg scans `sys.path` for `juliapkg.json`.** When running a script
  outside the repo, `sys.path[0]` is the *script's* directory — a scratch
  `juliapkg.json` there gets scanned and can poison `.venv/julia_env` with an
  unresolvable spec (recovery: rename the file, delete `.venv/julia_env`).
  Run probes with `PYTHONPATH=.` from the worktree root instead.
- **The shim env has only EquivariantTensors** (bundling StaticArrays); no
  `Interpolations`/`OffsetArrays`. Reconstructing Julia's spline evaluator
  through a scratch Julia env does not resolve and is not needed —
  `spline_eval` (`eval/radial.py`) was validated exactly against the Julia
  batched evaluator (per-column ratio 1.0) on the fixture's prefiltered
  B-spline control points.
- **angular table width differs from the fixture** (mine 64 / lmax 7 vs
  fixture 25 / lmax 4): compare angular quantities through the `aspec_y`
  gathers, not positionally.

## Roadmap after Tier 1

- ~~Tier 2 point 1 — in-memory hand-off~~: done (`Authoring.eval_pair`).
- ~~Tier 2 point 2 — coupling cache~~: done (`couple_cached`, see
  [tier2-plan.md](tier2-plan.md)) — authoring an existing shape runs
  Julia-free from a populated cache dir.
- **Endgame — pure-JAX coupling**: reimplement ET's `SparseSymmProd`
  symmetrisation in JAX, dropping the `authoring` extra's juliacall
  dependency entirely. The hard part: degenerate nnll blocks are only unique
  up to a row-space rotation, so coefficient interchange needs the
  subspace-matching discipline the parity tests already use.
