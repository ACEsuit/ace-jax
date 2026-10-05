# Basis construction internals

Maintainer notes on how `build_model` builds a basis, how `save_npz` writes it,
and the parity test against the ACEpotentials fixture. These notes were part
of the user how-to [Build a basis](../user/howto/basis.md) until the
ASD-STE100 review of the user documentation moved them here.

## How a basis is built

`build_model(elements, order, totaldegree, *, wL, rcut, r0, rin, radial_mode,
pair_mode, seed, with_gamma, edge_a_kind, coupling_cache, coupling_cache_dir,
n_q_factor)` (`basis/model.py`) walks the same
steps the ACEpotentials exporter walks, returning a `Basis` NamedTuple:

1. **`resolve_elements`** — atomic numbers or symbols (`14`, `"Si"`) → `zs`
   in the order given (it is the species index order, as ACEpotentials'
   `_convert_zlist` keeps it); duplicates are rejected.
2. **`build_spec(NZ, order, totaldegree, wL)`** (`basis/spec.py`) — the three
   integer specs ET consumes: `mb_spec` (per-B `(n,l)` tuples under
   `TotalDegree` + `rpe_admissible`), `Rnl_spec`, `Ylm_spec`.
3. **`couple(mb_spec, Rnl_spec, Ylm_spec)`** (`basis/coupling.py`) — the
   compiled EquivariantTensors library (`ace-jax-coupling`); returns `A2B` `(n_B, n_AA)`, per-order `aa_specs`,
   `aspec`, and ET's `aa_sig` per AA column (signature-sorted, **not**
   evaluation-ordered — see below).
4. **nnll cross-check** — `nnll_from_coupling` recomputes each B row's body
   list from the block-diagonal `A2B` (first nonzero column → `aa_sig`, with
   multiplicity) and asserts it matches the shim dump as a row-multiset.
   Catches any ET-version drift in one place.
5. **`tensor_radial_init` / `pair_radial_init`** (`basis/radial_init.py`) —
   seeded radial coefficients, frozen. `mode="glorot_normal"` (tensor),
   `"onehot"` (pair, `spl_n = δ(n, 2Z)`-style identity rows) or `"zero"`;
   the 7-tuple Agnesi transform `(p, q, a, rin, r0, yin, ycut)` is derived
   from `(p, q, rin, r0, rcut)` per species pair and never stored
   independently. `r0` defaults to ACEpotentials' `_default_rin0cuts`,
   `(bond_len(zi) + bond_len(zj)) / 2` per pair (`r0=` overrides it with a
   scalar or an `(NZ, NZ)` table). The Legendre recurrence uses
   Polynomials4ML's `OrthPolyBasis1D3T` convention (`P1 = A[1] x + B[1]`,
   no `P0` factor, so `A[1] = sqrt(3/2)`) — the one `eval/radial.py`'s
   `poly_recursion` reads, pinned coefficient-for-coefficient against the
   fixture.
6. **Zero readout** — `WB`, `Wpair`, `E0` zeros (the `acefit!`-fresh init), then
   `fold_readout` bakes the species pooling into the coefficient tables.
7. **`smoothness_prior`** (`basis/prior.py`, ported in PR #3) — the
   `len_basis` γ vector, over tensor rows repeated per species and pair rows
   `[(n,0)]`; `--no-gamma` / `with_gamma=False` skips it.

`save_npz(path_or_file, basis)` (`basis/export.py`) writes the
npz bridge file, for any of the three radial layouts the loader reads
(`analytic`, `spline`, `spline_factorised`).

## Writer routes on the model, not on defaults

`save_npz` **derives the branch-selector meta keys from the model tree being
saved**: `radial_kind = model.radial_kind`, `pair_radial_kind =
model.pair_radial_kind`, `pair_envelope_kind = model.pair_envelope_kind`,
`ybasis_kind` from `model.ysolid`. Everything else in `meta` is copied
(deep-copied through JSON, so callers may mutate their copy freely).

Rationale: the loader (`eval/io.py`) reconstructs the eval graph from
`meta` alone — `radial_kind`/`pair_radial_kind` pick the radial branch,
`pair_envelope_kind` the pair envelope formula, `rnl_spline`/`pair_spline`
the `(x0, h, n)` grids. If the writer echoed build defaults while the
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

`tests/test_basis_build.py::test_bridge_wellformed_subprocess` covers the
whole chain against `fixtures/si_ace_model.npz` (Si, order 3, TotalDegree 10):

- **Structural**: the built `nnll_spec` matches the fixture as a row-multiset;
  feeding the fixture's own nnll row order back through `couple` reproduces
  `A2B` up to a per-B-row scale (EquivariantTensors main normalises B rows
  differently from the ET 0.4.3 the ACEpotentials 0.10.1 fixture used; the
  injected `WB` and the descriptor reference are rescaled accordingly).
- **Injected + evaluated**: the fixture's fitted coefficients and fitted
  branch kinds (`radial_kind="analytic"`, `pair_radial_kind="spline"`,
  `pair_envelope_kind="poly1sr"`, spline grid `(x0, h, n)` from
  `meta["pair_spline"]`) are patched in via `dataclasses.replace`
  (eqx `ACEModel` supports it; the `Basis` tuple via `._replace`),
  packaged with `save_npz`, reloaded with `eval.io.load`, and every table is
  compared against the fixture. Then the round-tripped file runs through the
  native eval path (`ACECalculator`) and must reproduce ACEpotentials' energies,
  forces, stress and descriptors to float noise (`< 1e-8`, E exact).

Two conventions to know when reading fixtures:

- **Probe rows are tensor-side** (`probe_r = range(0.9, maxrcut, 24)`, seeded):
  `probe_x` is `rbasis.transforms[1,1](r)` (the *tensor* basis transform —
  `agnesi_normalized(r, rnl_transform)`, NOT the pair transform),
  `probe_env` the tensor envelope, `probe_Rnl` the tensor radial. The one
  pair-basis row is `probe_Rpair = spl(x_ij) * e_ij` from
  `evaluate_batched(m.pairbasis, ...)`.
- **Virial vs stress**: the fixture's `test_V` is the *ACEpotentials virial*
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
reconstructs the `Coupling` without importing the coupling library**: a
known shape builds from a populated cache dir even where the coupling library
is unavailable. Entries store their input specs (hit-time re-check) and the
coupling backend id `coupling.backend_id()` (`ace-jax-coupling==<version>`; a
library change invalidates); writes are atomic (a unique tmp file per writer,
then `os.replace`) and best-effort, and any entry that fails to read for
whatever reason — torn zip, schema drift — is a miss, never an error.
`couple()` stays
the uncached parity oracle; `--no-coupling-cache` / `coupling_cache=False`
(or `coupling_cache_dir="none"`, `ACEJAX_COUPLING_CACHE=none`) opt out. Entries are self-describing npz files — ship one by copying it into a
team's cache dir.

## Embedded (species-compressed) models

`build_embedding_model(elements, order, totaldegree, embedding, *, d_max, wL,
maxl, rcut, reduction, ...)` authors ACEpotentials' ace1-compatible
`ace_embedding_model` the same way (`ace-jax basis --embedding <table.json |
identity> [--d-max N]`); its design record is
[docs/dev/plans/embedded-model-authoring.md](plans/embedded-model-authoring.md).

Choosing the table and `d_max`:

- `identity` is the one-hot element table: one channel per element, so it is
  lossless only. A `d_max` below the number of elements is an error, because a
  one-hot table has no preferred directions to compress onto; compress a table
  of element properties instead.
- The rows are normalised to unit length, so with `d_max=1` each element is
  either +1 or −1 and the elements fall into at most two groups. ace-jax warns
  when two elements end up with the same row: the basis cannot tell them apart.

## Known traps

- **The coupling library contains only EquivariantTensors' construction code**,
  nothing else to evaluate. Reconstructing ACEpotentials' spline evaluator is
  not needed — `spline_eval` (`eval/radial.py`) was validated exactly against
  ACEpotentials' batched evaluator (per-column ratio 1.0) on the fixture's
  prefiltered B-spline control points.
- **angular table width differs from the fixture** (mine 64 / lmax 7 vs
  fixture 25 / lmax 4): compare angular quantities through the `aspec_y`
  gathers, not positionally.

## Roadmap after Tier 1

- ~~Tier 2 point 1 — in-memory hand-off~~: done (`Basis.eval_pair`).
- ~~Tier 2 point 2 — coupling cache~~: done (`couple_cached`, see
  [tier2-plan.md](tier2-plan.md)) — a known shape builds from a populated
  cache dir.
- ~~Precompiled coupling~~: done — `ace-jax-coupling` is a core dependency
  (`coupling/`, [coupling-etshim-spec.md](coupling-etshim-spec.md)).
- ~~`aj fit` builds the basis~~: done (`FitConfig(model=BasisSpec|Basis)`,
  `fit.yaml`).
- **Endgame — pure-JAX coupling**: reimplement ET's `SparseSymmProd`
  symmetrisation in JAX, dropping the compiled library too. The hard part: degenerate nnll blocks are only unique
  up to a row-space rotation, so coefficient interchange needs the
