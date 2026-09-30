# Coupling generation via the compiled EquivariantTensors library

> **Update 2026-09-30.** juliacall/juliapkg are gone: the `basis` extra is
> now `ace-jax-coupling`, EquivariantTensors' coupling construction compiled
> with `juliac --trim=safe` and shipped as a platform wheel (plan:
> [plans/2026-09-30-trim-coupling-lib.md](plans/2026-09-30-trim-coupling-lib.md)).
> Radials, pair basis and embedding are authored in Python
> ([basis.md](basis.md)), with a per-shape coupling cache.

**Goal.** Generate the SO(3) coupling artifacts (`A2B` map + `aa_spec`) — the one
piece [EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl)
owns — from Python with **no Julia at runtime**, so a *new basis shape* can be
authored without the ACEpotentials.jl stack or a Julia install. Refitting or
evaluating an existing model never runs any of this.

## The contract (layout ace-jax consumes)

`julia/export_model.jl` builds the whole basis with one ET call and writes:

- `aspec_r`, `aspec_y` — the A-basis spec `m.tensor.abasis.spec` = `(Rnl_idx,
  Ylm_idx)` per A function, **0-based**.
- `aa_spec_k` (k = 1..order) — `m.tensor.aabasis.specs`, each `(n_v, k)` int
  matrix of A-indices per AA function, **0-based**.
- `A2B` — `Matrix(m.tensor.A2Bmaps[1])`, shape `(n_B, n_AA)`, stored as sparse
  triplets `A2B_rows/cols/vals/A2B_shape`, **0-based**.
- `nnll` — the `(n,l)` per B function.

The eval side (`ace_jax/eval/io.py`, `eval/model.py`) reads `A2B` (triplets →
dense `(n_B, n_AA)`) and the per-order `aa_spec_k`, and contracts `AA → B` with
`A2B`. **The row/column ordering of `A2B`, the A-index numbering used inside
`aa_spec`, and the B-function order must all match the exporter**, or fitted
coefficients stop being interchangeable and the ACEfit-parity guarantee breaks.
`sparse_equivariant_tensor` is deterministic given identical integer specs, so
parity reduces to: **feed ET the same `mb_spec`/`Rnl_spec`/`Ylm_spec` the Julia
path fed it.**

## Python spec builder (`basis/spec.py`)

`build_spec(NZ, order, totaldegree, wL)` produces the three integer specs ET
consumes, and `spec_from_export` / `spec_from_reference` recover them from an
existing `.npz` (used by the parity test):

- `Rnl_spec`: list of `(n, l)` radial-channel tuples, `n ≥ 1`, `l ≤ lmax`,
  admitted by the level function.
- `Ylm_spec`: `(l, m)` for `l ≤ lmax`, `-l ≤ m ≤ l` (real-harmonics ordering).
- `mb_spec`: per-B-function `(n, l)` tuples up to `order`, filtered to
  `totaldegree` under `TotalDegree` (the level weights `l` by `1/wL`, `n` by
  `NZ`) and by `rpe_admissible`.

`rpe_admissible(bb, L=0)`: a basis tuple `bb` (list of `(n,l,m)`) is admissible
iff a signed `m`-combination sums to `≤ L` in absolute value **and**
`iseven(sum(l) + L)` **and** the single-entry special case (`len == 1 ⇒ l == 0`).
Pure Python, no deps.

## Architecture (three layers)

1. **ET fork** — [jameskermode/EquivariantTensors.jl](https://github.com/jameskermode/EquivariantTensors.jl),
   branch `jk/trim-safe-coupling`, on ACEsuit `main` (v0.5.1). It adds a
   type-stable construction API with unchanged public behaviour:
   `symmetrisation_matrix(Val(L), mb_spec; ...)`, `O3.coupling_coeffs(Val(L),
   ll::Vector{Int}, nn::Vector{Int}; ...)` (correlation order resolved through a
   static table, `N ≤ O3.MAX_STATIC_ORDER = 8`) and
   `sparse_equivariant_tensor_spec(Val(L); ...)` (the layer-free half of
   `sparse_equivariant_tensor`), plus trim-safety fixes (`basis`/`hashfcn`
   argument specialisation, concrete `SetLl` containers, a trim-safe sparse
   transpose and UMFPACK `L'`/`p`/`Rs` extraction in `solver_inner`). The fork's
   CI compiles `test/trim/entry.jl` with `juliac --trim=safe` on Julia 1.13 and
   compares the executable with ordinary Julia. The rev ace-jax uses is pinned
   (full sha) in `coupling/julia/Project.toml` `[sources]`.
2. **Shim + build** — `coupling/julia/src/ETCouple.jl` exposes one stateless
   C function over that API; `coupling/julia/build.jl` compiles it with JuliaC
   0.3.10 (`--trim=safe --compile-ccallable`, `JULIA_CPU_TARGET=generic`,
   `--jl-option handle-signals=no`, `--privatize`, `MACOSX_DEPLOYMENT_TARGET=11.0`,
   ad-hoc re-signing on macOS) into a relocatable bundle with
   `build_info.json`. Julia **1.13** is required (1.12's SparseArrays is not
   trim-clean).
3. **Python package** — `coupling/python` (dist `ace-jax-coupling`, import
   `ace_jax_coupling`): ctypes loader (lazy, `RTLD_LOCAL`, calls serialised by a
   lock), Python-side input validation, and `couple_raw(...)`.
   `basis.coupling.couple()` converts its arrays to the `Coupling` tuple.

### C ABI v1 (`libetcouple`)

```c
int32_t etc_abi_version(void);          // 1
int32_t etc_max_order(void);            // 8  (max correlation order = body length)
int32_t etc_couple(
    int64_t n_mb, const int64_t *mb_off /* n_mb+1, mb_off[0]==0 */, const int64_t *mb_n, const int64_t *mb_l,
    int64_t n_r,  const int64_t *r_n, const int64_t *r_l,        // Rnl_spec (n, l)
    int64_t n_y,  const int64_t *y_l, const int64_t *y_m,        // Ylm_spec (l, m)
    int64_t *sizes /* [7] in: capacities, out: required */,
    int64_t *a2b_i, int64_t *a2b_j, double *a2b_v,               // [nnz]   A2B triplets, 0-based, findnz order
    int64_t *sig_off /* [n_AA+1] */, int64_t *sig /* [3*n_sig] */, // per A2B column (n,l,m) bodies = meta 𝔸spec
    int64_t *aspec /* [2*n_A] */,                                  // (Rnl_idx, Ylm_idx) 0-based = abasis.spec
    int64_t *aa_off /* [n_AA+1] */, int64_t *aa_idx /* [n_aaidx] */, // AA basis, SparseSymmProd evaluation order
    int64_t *nnll_off /* [n_B+1] */, int64_t *nnll /* [2*n_nnll] */); // per B row (n,l) = get_nnll_spec(tensor, 1)
// sizes = [nnz, n_B, n_AA, n_sig, n_A, n_aaidx, n_nnll]
// returns 0 OK, 1 BUFFERS_TOO_SMALL (sizes filled), 2 INVALID_INPUT, 3 ORDER_TOO_HIGH (> 8)
//         4 NO_INVARIANTS (no body admits an L=0 invariant), 5 INTERNAL_ERROR (ET threw / inconsistent sizes)
```

Two-phase and stateless: call 1 with zero capacities returns the sizes, call 2
fills caller-owned buffers; no Julia object outlives a call (ACEpotentials PR
#309 found runtime-rooted objects in a trimmed image collected under the
caller). Inputs are validated before any ET call, so no exception unwinds
across the boundary.

The per-column signatures come from ET's `meta["𝔸spec"]` (the spec returned
*with* the symmetrisation matrix), not `aabasis.specs`: `SparseSymmProd`
re-sorts its input, so the evaluation order (`aa_idx`) is a different
permutation from the `A2B` column order. The shim reproduces
`SparseSymmProd`'s ordering (stable sort by length) and `get_nnll_spec` (each
row's first stored column), both checked against the real ET objects in
`coupling/julia/test/runtests.jl`.

### Bundle and wheels

`coupling/tools/prune_bundle.py` keeps only the shared libraries actually
loaded, traced (`DYLD_PRINT_LIBRARIES` / `LD_DEBUG`) in two process states --
every test case with numpy imported first, and a bare `ctypes` load (numpy
pulls in the system `libgcc_s`, hiding the bundled one Julia's loader opens
otherwise) -- in an empty HOME so JLL artifacts resolve from the bundle and not
a local depot. Linux keeps exactly the names `LD_DEBUG` reports as opened and
strips debug info from the unmodified third-party libraries (never from
`libetcouple` or the patchelf'd `libjulia*`, which `strip` corrupts). macOS
(dyld reports resolved files) keeps a library's alias names only when another
traced binary or libjulia's loader list names them. Wheels cannot hold symlinks, and two
copies of one library under different names load as two images with separate
state (a duplicated `libblastrampoline` hung the solver; a second
`libjulia-internal` is an uninitialised runtime). So each library ships once,
under the name libjulia's loader opens by path, and every other reference is
rewritten to it (`install_name_tool -change` on macOS, keeping install names so
`dlopen("@rpath/<install name>")` still matches the loaded image;
`patchelf --replace-needed` on Linux). `check_bundle.py` rejects any duplicate
copy; `test_no_library_loaded_twice` checks the loaded image list and
`test_largest_case_is_fast` catches a slow double load. `check_bundle.py`
asserts ABI, macOS `minos ≤ 11.0` and Linux `GLIBC ≤ 2.28`. Wheels are
`py3-none-{manylinux_2_28_x86_64, manylinux_2_28_aarch64, macosx_11_0_arm64}`
(macOS arm64 ~18 MB, Linux aarch64 ~24 MB); without a bundle the build makes a lib-less `py3-none-any` dev wheel
whose `build_info()` raises `CouplingLibError`. `coupling/tools/test_wheel.sh`
installs a wheel into a fresh venv with an empty HOME and no Julia on PATH and
runs the package tests (and checks HOME is untouched). Bundled third-party
licences: `coupling/python/THIRD_PARTY_NOTICES.md` (includes GPL-2.0+
SuiteSparse modules; the wheel's licence declaration is still open).

## ET version vs the ACEpotentials fixtures

The library is ET main; ACEpotentials 0.10.1 (which generated
`fixtures/coupling_ref_*.npz` via `julia/coupling_reference.jl`) pins ET 0.4.3.
Aligned by column signature, the two have identical nnll blocks and identical
per-block row spaces (degenerate order-4 blocks included) but a different
per-B-row normalisation (row-norm ratio main/0.4.3 in [0.225, 1]). Python-
authored models therefore span the same function space as 0.10.1 exports, but
their coefficients are not interchangeable (coefficients scale by the inverse
row factor; descriptors by the factor).

## Tests

- `coupling/python/tests` — against unpatched upstream ET at the fork's base
  commit (reference computed on macOS arm64): every index array identical, A2B
  values within 4 ulp (bit-identical on macOS arm64 and Linux aarch64; x86_64
  differs in the last bit or two where ET's sparse LU runs) on 8 shapes (orders 1–8, up to
  6216×17302), via `coupling/julia/reference/` → `tests/data/et_reference.npz`;
  input validation, threads, SIGINT after load, lazy loading.
- `tests/test_coupling_parity.py` — vs the ACEpotentials fixtures, in-process:
  multiplicity-1 blocks equal up to a per-row scale, degenerate blocks on the
  row subspace (`subspace_residual`), all residuals < 1e-12.
- `tests/test_basis_build.py` — authored model vs the Si fixture with the
  fixture's coefficients rescaled: energies, forces, stress, descriptors.
- All of them skip without a compiled `ace_jax_coupling`
  (`conftest.require_coupling_lib`); CI (`coupling-wheels.yml`) builds the
  wheels, tests them in clean containers and runs the parity with
  `ACEJAX_REQUIRE_OPTIONAL=1`.

## Caching per shape

`A2B`/`aa_spec` depend only on the three integer specs. `couple_cached` stores
one entry per shape (sha256 of the order-preserving spec JSON) stamped with
`coupling.backend_id()` (`ace-jax-coupling==<version>`); a hit never imports the
library, a backend change invalidates entries (see `docs/basis.md`,
"Coupling cache"). `couple()` remains the uncached parity oracle.
