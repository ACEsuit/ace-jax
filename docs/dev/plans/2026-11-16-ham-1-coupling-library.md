# Hamiltonians 1: `ace-jax-coupling` 0.3.0 (L > 0, both parities) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The compiled coupling library returns equivariant (L ≤ 6, parity ±1) couplings through a new, additive entry point, and ace-jax can call and cache them.

**Architecture:** The ET fork forwards `refl_sym` through its type-stable `Val(L)` path. `ETCouple.jl` gains `etc_couple_equivariant`, which dispatches over a static table of L and shares all input parsing, validation and output writing with `etc_couple`. Python gains `couple_equivariant_raw` (ctypes) and `basis.coupling.couple_equivariant[_cached]`.

**Tech Stack:** Julia 1.13.1, JuliaC `--trim=safe`, EquivariantTensors fork, ctypes, numpy, pytest.

**Spec:** `docs/dev/specs/2026-11-16-hamiltonians-design.md` §1; ABI v1 in `docs/dev/coupling-etshim-spec.md`.

This is plan 1 of 6 (design "Order of work"). Plans 2–6 (framework refactor; data IO; model; fit and UQ; site-bond provider and results) are written when their predecessor is done, because each depends on interfaces the previous one fixes.

## Global Constraints

- `etc_couple`: signature and results unchanged, bit for bit. Existing `coupling/python/tests` pass unmodified.
- `etc_abi_version()` returns 2. L_MAX = 6. Correlation order ≤ 8.
- `parity` is +1 or −1 = (−1)^Σl of the products; +1 ↔ ET `refl_sym = :sym`, −1 ↔ `:asym`.
- `nullspace_solver = :dense` always (no UMFPACK in the bundle).
- No exception crosses the C boundary; every failure is a return code.
- Build with exactly Julia 1.13.1 (`build.jl` checks the Manifests).
- **Not without the owner's go-ahead:** pushing the fork branch, touching ET PR #146, publishing 0.3.0 to PyPI. Until the fork is pushed, `coupling/julia/Project.toml` cannot name the new rev; local builds use a path source that is not committed.
- Commits: conventional prefixes with scope, e.g. `feat(coupling):`.

## Review Focus

1. A (L, parity) with no coupled function for the given `mb_spec` (e.g. L = 6, parity −1 at order ≤ 3): returns `NO_INVARIANTS`, Python raises `ValueError`; `couple_equivariant` callers must be able to treat it as "empty", not an error (Task 4 returns an empty `Coupling`).
2. L outside 0…6 or parity not ±1: `INVALID_INPUT`, never a crash (Task 2).
3. An old library (ABI 1) with the new Python: `couple_equivariant_raw` raises `CouplingLibError` naming the ABI; `couple_raw` keeps working (Task 3).
4. A cache entry for (L, parity) must never be read for another (L, parity) or as an L = 0 `etc_couple` entry (Task 4: key test).
5. The AA columns returned for different (L, parity) of one `mb_spec` are different subsets; after alignment each must index the same evaluation order of the union (Task 4: union test).

---

### Task 1: Fork: `refl_sym` through the `Val(L)` path, trim test for L ≤ 6

**Files** (in the fork checkout `/storage/eng/essswb/projects/ace-jax/ham-ref/ET-fork`, branch `jk/trim-equivariant` off `jk/trim-coupling`):
- Modify: `src/utils/symmop.jl` (`symmetrisation_matrix(::Val{L}, …)`), `src/O3/O3.jl` (`coupling_coeffs(::Val{L}, …)`, `_coupling_coeffs_static`), `src/ace/sparse_ace_utils.jl` (`sparse_equivariant_tensor_spec(::Val{L}; …)`)
- Modify: `test/test_utils/trim_cases.jl`, `test/trim/entry.jl`, `test/test_trim_api.jl`

**Interfaces:**
- Produces: `ET.symmetrisation_matrix(Val(L), mb; prune, PI, basis, nullspace_solver, refl_sym::Union{Symbol,Nothing} = nothing)`; same keyword on `O3.coupling_coeffs(Val(L), …)` and `sparse_equivariant_tensor_spec(Val(L); …)`. `nothing` keeps today's default (L even `:sym`, L odd `:asym`).

- [ ] **Step 1:** In `test/test_trim_api.jl` add a testset: for L in 0:4 and `rs` in (:sym, :asym), on `trim_mb_spec(2, 2, 3)` without its even-Σl filter, `symmetrisation_matrix(Val(L), mb; prune=true, PI=true, basis=real, refl_sym=rs)` equals `symmetrisation_matrix(L, mb; …, refl_sym=rs)` exactly (same `findnz`, same 𝔸spec).
- [ ] **Step 2:** Run `julia +1.13.1 --startup-file=no --project=. -e 'using Pkg; Pkg.test(test_args=["trim_api"])'` (or the file directly); expect a MethodError on the `refl_sym` keyword.
- [ ] **Step 3:** Forward the keyword through the three functions (already prototyped in the working tree for the first two).
- [ ] **Step 4:** Extend `test/trim/entry.jl`'s `checksum_lines` with an L loop written as explicit `Val(0)`…`Val(6)` calls × both `refl_sym`, on an `mb_spec` that includes odd-Σl bodies; checksums hash every component of each `SVector`. No `string` call with more than a few arguments (the verifier rejects the vararg fallback).
- [ ] **Step 5:** Run `julia +1.13.1 --startup-file=no --project=test/trim test/trim/build_and_run.jl`; expect `trim test OK`. Remove the spike file `test/trim/entry_L.jl`.
- [ ] **Step 6:** Commit in the fork (local only): `feat: refl_sym in the Val(L) coupling API; trim test for L ≤ 6`.

### Task 2: `ETCouple.jl` ABI v2: `etc_couple_equivariant`

**Files:**
- Modify: `coupling/julia/src/ETCouple.jl`, `coupling/julia/test/runtests.jl`, `coupling/julia/build.jl` (`build_info.json`: `"abi" => 2`, `"max_L" => 6`)
- Modify (uncommitted, local only): `coupling/julia/Project.toml` `[sources]` → `path` to the fork checkout

**Interfaces:**
- Produces (C):
  - `int32 etc_abi_version()` → 2; `int32 etc_max_L()` → 6.
  - `int32 etc_couple_equivariant(int64 L, int64 parity, <the 10 input args of etc_couple>, int64* sizes, int64* a2b_i, int64* a2b_j, int64* a2b_m, double* a2b_v, <the 7 remaining output args of etc_couple>)`.
  - `sizes` has the same 7 slots; slot 0 (`nnz`) counts (row, col, m) entries with a nonzero value, m ∈ 0…2L. Entries are ordered by `findnz` of the sparse matrix, then m ascending.
  - Return codes as v1, plus `IMAGINARY_COUPLING = 6` (reserved for ET's purely-imaginary real-basis case; today ET handles it internally, so the code is returned only if ET throws that specific error).
  - L = 0, parity = +1 returns exactly `etc_couple`'s arrays with `a2b_m` all 0.

- [ ] **Step 1:** In `coupling/julia/test/runtests.jl` add tests calling the Julia-level `ETCouple.compute_equivariant(L, parity, mb, rnl, ylm)`: (a) L=0, parity=+1 equals `compute` field by field; (b) for L in 1:4, both parities, the (I, J, m, V) expansion equals `ET.symmetrisation_matrix(L, mb; refl_sym=…)` through `ET._tensor_specs`; (c) L=7 and parity=0 give `INVALID_INPUT` from a `validate_equivariant`; (d) L=6, parity=−1 on an order ≤ 3 spec gives `nothing`.
- [ ] **Step 2:** Run `julia +1.13.1 --startup-file=no --project=coupling/julia coupling/julia/test/runtests.jl`; expect `UndefVarError: compute_equivariant`.
- [ ] **Step 3:** Implement. `Result` gains `M::Vector{Int}` (empty for v1 writes). Factor the body of `etc_couple` into `_read_inputs` and `_write_outputs(r, sizes, …; a2b_m = C_NULL)` used by both entry points. `compute_equivariant` dispatches `L == 0 … L == 6` to `_compute(Val(L), rs, …)` explicitly (a static table, as `coupling_coeffs` does for the order).
- [ ] **Step 4:** Run the Julia tests; expect all pass.
- [ ] **Step 5:** Build: `julia +1.13.1 --startup-file=no --project=coupling/julia/build coupling/julia/build.jl /storage/eng/essswb/projects/ace-jax/ham-ref/work/coupling-build`; expect `built …/libetcouple.so` and 0 `Verifier error` lines.
- [ ] **Step 6:** Commit `ETCouple.jl`, tests, `build.jl` (not `Project.toml`): `feat(coupling): ABI v2, etc_couple_equivariant for L <= 6 and both parities`.

### Task 3: Python package: `couple_equivariant_raw`

**Files:**
- Modify: `coupling/python/src/ace_jax_coupling/_api.py`, `_loader.py`, `__init__.py`
- Test: `coupling/python/tests/test_coupling_equivariant.py` (new)

**Interfaces:**
- Consumes: Task 2's C functions.
- Produces: `ace_jax_coupling.couple_equivariant_raw(L: int, parity: int, mb_spec, Rnl_spec, Ylm_spec) -> RawEquivariantCoupling` (the fields of `RawCoupling` plus `L`, `parity`, `A2B_m: np.ndarray[int64]`); `ace_jax_coupling.MAX_L = 6`. Raises `ValueError` for bad L/parity and for `NO_INVARIANTS`; `CouplingLibError` if the loaded library's ABI is < 2.

- [ ] **Step 1:** Write tests (run with `ACEJAX_COUPLING_LIB` set to Task 2's build; skip via `conftest.require_coupling_lib` otherwise): L=0/+1 equals `couple_raw` array for array; `A2B_m` within 0…2L; L=7, parity=2 → `ValueError`; a monkeypatched ABI 1 → `CouplingLibError`; `couple_raw` results unchanged against the committed `et_reference.npz`.
- [ ] **Step 2:** Run `ACEJAX_COUPLING_LIB=… uv run --no-sync pytest coupling/python/tests/test_coupling_equivariant.py -n 0`; expect ImportError/AttributeError.
- [ ] **Step 3:** Implement, sharing input marshalling with `couple_raw` through one private helper. `_loader` declares the new argtypes only when `etc_abi_version() >= 2`.
- [ ] **Step 4:** Run the new file and the whole `coupling/python/tests` directory; expect pass, no existing test modified.
- [ ] **Step 5:** Commit: `feat(coupling): couple_equivariant_raw in the Python package`.

### Task 4: Parity reference for L > 0, and `basis.coupling.couple_equivariant`

**Files:**
- Modify: `coupling/julia/reference/reference.jl`, `coupling/tools/gen_cases.py` (equivariant cases), `coupling/python/tests/data/` (new `et_reference_equivariant.npz`, `cases_equivariant.json`)
- Modify: `src/ace_jax/basis/coupling.py`
- Test: `coupling/python/tests/test_coupling_parity.py` (new tests only), `tests/test_coupling_equivariant.py` (new)

**Interfaces:**
- Produces:
  - `couple_equivariant(L, parity, mb_spec, Rnl_spec, Ylm_spec) -> Coupling` with `A2B` of shape `(n_B, 2L+1, n_AA)`, columns in that call's own AA evaluation order (the library's A and AA sets depend on L and parity). `n_B = 0`, empty specs, when the library reports `NO_INVARIANTS`.
  - `couple_equivariant_set(lps, mb_spec, Rnl_spec, Ylm_spec, cache_dir=None) -> EquivariantSet(aspec, aa_specs, aa_sig, A2B)` for `lps` a sequence of `(L, parity)`: one shared A list and AA evaluation order (the union over `lps`, by (n, l, m) signature; AA grouped by order, first-seen order within a group) and `A2B: dict[(L, parity)] -> (n_B, 2L+1, n_AA_union)`. This is what the model evaluates: one AA vector for all L.
  - `equivariant_key(L, parity, mb, rnl, ylm) -> str`; `couple_equivariant_cached(L, parity, mb, rnl, ylm, cache_dir=None)`; cache files `cple-<key16>.npz`.
  - `COUPLING_LIB_VERSION` stays `"0.2.0"` until the release commit; `couple_equivariant` checks `ace_jax_coupling.MAX_L` exists and raises `BasisUnavailable` naming 0.3.0 otherwise.

- [ ] **Step 1:** Reference: extend `reference.jl` to write, for each equivariant case × L in 0:4 × both parities, the (rows, cols, m, vals) of unpatched ET's `symmetrisation_matrix(L, …; refl_sym)` with `:dense`. Generate the npz with the reference project (ET at the fork base + the dense-solver commit only).
- [ ] **Step 2:** Parity tests: index arrays identical; values within 4 ulp; for blocks with multiplicity > 1, `subspace_residual` < 1e-12 on the (n_B, (2L+1)·n_AA) flattening.
- [ ] **Step 3:** `tests/test_coupling_equivariant.py`:
  - *Equivariance:* build A from random neighbour vectors with `eval/harmonics.py` and a random radial table, form AA, B = einsum(A2B, AA); for a random rotation Q, B(QR) = D_L(Q) B(R) to 1e-10, with D_L from the same harmonics; under inversion B → parity·B.
  - *Union:* `couple_equivariant_set` on [(0,+1),(1,−1),(1,+1),(2,+1)]: contracting each `A2B[(L,p)]` with the shared AA equals contracting the single-call `couple_equivariant` result with its own AA, to 1e-13.
  - *Keys:* `equivariant_key` differs across L, across parity, and from `coupling_key`.
  - *Empty:* L=6, parity=−1, order ≤ 3 → `A2B.shape[0] == 0`, no exception.
  - *Cache:* second call does not import `ace_jax_coupling` (as the existing cache test asserts).
- [ ] **Step 4:** Run; expect failures on the missing functions.
- [ ] **Step 5:** Implement in `basis/coupling.py`, reusing `align_columns`' signature lookup; the set scatters each call's columns into the union order.
- [ ] **Step 6:** Run `ACEJAX_COUPLING_LIB=… uv run pytest tests/test_coupling_equivariant.py coupling/python/tests -n 0` and the existing `tests/test_coupling*.py`; expect pass.
- [ ] **Step 7:** Commit: `feat(basis): equivariant couplings (L, parity) with a per-shape cache`.

### Task 5: Local wheel, docs, and the stop before release

**Files:**
- Modify: `docs/dev/coupling-etshim-spec.md` (ABI v2 section), `coupling/RELEASING.md`, `coupling/python/README.md`, `CLAUDE.md` (coupling lines), `coupling/python/tests/test_coupling_notices.py` only if the bundle's library list changes

- [ ] **Step 1:** `coupling/tools/prune_bundle.py` and `check_bundle.py` on the Task 2 bundle: the trace-based pruning must be driven by a trace that includes equivariant calls (extend the trace driver), or L > 0 code paths' libraries are pruned. Expect `check_bundle` OK.
- [ ] **Step 2:** `coupling/tools/test_wheel.sh` on a locally built wheel versioned `0.3.0.dev0`; extend its smoke call with one `couple_equivariant_raw`. Expect pass in the clean environment.
- [ ] **Step 3:** Write the ABI v2 documentation.
- [ ] **Step 4:** Commit: `docs(coupling): ABI v2`.
- [ ] **Step 5: STOP.** Report to the owner: fork branch ready to push, the diff for ET PR #146 (or a follow-up PR), and the release steps. On go-ahead: push the fork, set `Project.toml` `rev`, version 0.3.0, run `coupling-wheels.yml`, publish, bump `COUPLING_LIB_VERSION` and the `pyproject.toml` pin, commit the ham coupling cache.
