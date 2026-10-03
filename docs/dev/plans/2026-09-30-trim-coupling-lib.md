# Trim-compiled coupling library (drop juliacall) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ace-jax's `authoring` extra (juliacall + juliapkg + a runtime Julia with EquivariantTensors/Lux) with a `juliac --trim=safe`-compiled shared library, shipped as a Julia-free platform wheel `ace-jax-coupling`, producing the coupling bit-exactly equal to EquivariantTensors (ET) `main`.

**Architecture:** Three layers. (1) A branch on the user's ET fork makes ET's coupling *construction* trim-safe (a `Val(L)` API plus type-stability patches; public behaviour unchanged). (2) A small Julia shim in ace-jax (`coupling/julia/src/ETCouple.jl`) exposes one stateless, two-phase C function over that API; `coupling/julia/build.jl` compiles it with JuliaC into a bundled, pruned, relocatable library. (3) A Python package (`coupling/python`, dist `ace-jax-coupling`) wraps the library with ctypes and is packaged as `py3-none-<platform>` wheels; `ace_jax.construct.coupling.couple()` calls it instead of juliacall.

**Tech Stack:** Julia 1.13 + JuliaC 0.3.10 (`--trim=safe`), EquivariantTensors.jl (fork), Python 3.11–3.13, ctypes, numpy, hatchling (custom build hook), uv, GitHub Actions (manylinux_2_28 containers, macos-14).

**Spec:** this document (the "Design" section below) + the feasibility probe recorded in memory note `et-trim-feasibility` (2026-09-30). `docs/dev/coupling-etshim-spec.md` is the *old* design and is rewritten in Task 12.

## Design (decisions already taken with the user)

- **ET base = ET `main` on the fork** `jameskermode/EquivariantTensors.jl`, branch `jk/trim-safe-coupling`, based on ACEsuit `main` (probe used 5342310, v0.5.1). Consequence, accepted: ACEpotentials 0.10.1 pins ET 0.4.3–0.4, and ET main differs from it by a **per-B-row rescaling** only (identical nnll blocks, identical per-block row spaces incl. degenerate order-4 blocks, row-norm ratio main/0.4.3 in [0.225, 1.0]). Python-authored couplings therefore stop being *coefficient*-identical to ACEpotentials 0.10.1 exports; they span the same function space. Parity vs the committed ACEpotentials fixtures becomes "same blocks, same span, per-row scale"; bit-exactness is asserted against **unpatched upstream ET at the base sha**.
- **Packaging = separate binary distribution** `ace-jax-coupling` (import `ace_jax_coupling`). ace-jax stays a pure-Python wheel; `ace-jax[authoring]` depends on it. No juliacall/juliapkg anywhere afterwards.
- **Platforms (first wheels):** `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macosx_11_0_arm64`. Windows and Intel macOS unsupported (extra is marker-gated).
- **Stateless C ABI, two-phase:** no Julia object survives a call (ACEpotentials PR 309 found runtime-rooted objects in a trimmed image are collected under the caller). Call 1 with zero capacities returns required sizes; call 2 fills caller-owned buffers. Compute runs twice; measured cost ≤150 ms for the largest case (6216×17302), ≤25 ms for everything else.
- **Julia 1.13 is required** to build (1.12's SparseArrays still has unresolvable `sparse(I,J,V)` / UMFPACK-init calls under trim).

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
    int64_t *aa_off /* [n_AA+1] */, int64_t *aa_idx /* [n_aaidx] */, // AA basis, SparseSymmProd evaluation order, 0-based A idx
    int64_t *nnll_off /* [n_B+1] */, int64_t *nnll /* [2*n_nnll] */); // per B row (n,l) = get_nnll_spec(tensor, 1)
// sizes = [nnz, n_B, n_AA, n_sig, n_A, n_aaidx, n_nnll]
// returns 0 OK, 1 BUFFERS_TOO_SMALL (sizes filled), 2 INVALID_INPUT, 3 ORDER_TOO_HIGH (> 8)
```

### Repository layout (new/changed)

```
ET fork (~/gits/EquivariantTensors.jl, branch jk/trim-safe-coupling)
  src/O3/O3.jl                 Val(L) lifts; basis::B; bounded-N coupling_coeffs(Val(L), ::Vector{Int}, ::Vector{Int}); trim-safe sparse helpers
  src/utils/symmop.jl          symmetrisation_matrix(::Val{L}, ...) + shared body
  src/utils/invmap.jl          hashfcn::F
  src/ace/sparse_ace_utils.jl  sparse_equivariant_tensor_spec(::Val{L}; ...) + shared _tensor_specs
  test/test_utils/trim_cases.jl  shared small-case generator
  test/test_trim_api.jl        Val API == Integer API (normal Julia, in runtests.jl)
  test/trim/{Project.toml, juliac_env/Project.toml, entry.jl, build_and_run.jl}   juliac compile test
  .github/workflows/CI.yml     + trim job (Julia 1.13, ubuntu + macos)

ace-jax (branch feat/trim-coupling, worktree .worktrees/trim-coupling)
  coupling/julia/Project.toml, Manifest.toml   shim project (ET = fork @ ET_REV)
  coupling/julia/src/ETCouple.jl               C ABI
  coupling/julia/test/runtests.jl              shim vs real ET (uncompiled)
  coupling/julia/build/Project.toml            JuliaC build env
  coupling/julia/build.jl                      compile → bundle → privatize → codesign → build_info.json
  coupling/julia/reference/{Project.toml,reference.jl}   UNPATCHED upstream ET @ BASE_SHA oracle
  coupling/tools/gen_cases.py                  cases.json from ace_jax.construct.spec
  coupling/tools/check_bundle.py               loadability, ABI, macOS minos / glibc floor
  coupling/tools/prune_bundle.py               trace-based library allowlist
  coupling/tools/test_wheel.sh                 clean-env wheel install + tests
  coupling/python/pyproject.toml, hatch_build.py, LICENSE, THIRD_PARTY_NOTICES.md, README.md
  coupling/python/src/ace_jax_coupling/{__init__.py,_loader.py,_api.py}
  coupling/python/tests/{test_api.py, test_runtime.py, data/cases.json, data/et_reference.npz}
  src/ace_jax/construct/coupling.py            couple() via ace_jax_coupling; backend_id cache stamp
  pyproject.toml, uv.lock                      authoring extra -> ace-jax-coupling; uv path source
  juliapkg.json                                DELETED
  tests/test_coupling_{parity,etshim,cache}.py, tests/test_python_authoring.py   updated
  .github/workflows/coupling-wheels.yml        NEW (build matrix, clean-env tests, ace-jax parity)
  .github/workflows/coupling-parity.yml        DELETED
  README.md, CLAUDE.md, docs/dev/coupling-etshim-spec.md (→ coupling-lib spec), docs/python-authoring.md
```

## Global Constraints

- Build toolchain: Julia **1.13.x** (`julia +1.13`), JuliaC **=0.3.10**. Never Julia 1.12 for trim builds.
- Local Julia 1.13 work uses a fresh depot: prefix every local `julia +1.13` command with `JULIA_DEPOT_PATH=$HOME/.julia-trim` (the default `~/.julia` has a stale ACE registry that crashes 1.13's Pkg with `KeyError: key "StrideArrays" not found`). One-time: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 -e 'using Pkg; Pkg.Registry.add("General")'`. CI uses the default depot.
- ET fork: `https://github.com/jameskermode/EquivariantTensors.jl`, branch `jk/trim-safe-coupling`. `BASE_SHA` = the ACEsuit `main` commit the branch starts from; `ET_REV` = the pushed fork commit used by ace-jax. Both are recorded in Task 1/3 and written into the Project.toml `[sources]` entries; never a branch name in `[sources]`.
- ET changes must not alter any existing public behaviour: `Pkg.test()` of the fork passes unchanged on Julia 1.11 and 1.13.
- Numbers: the compiled library's output is **bit-identical** (`np.array_equal`) to unpatched upstream `sparse_equivariant_tensor(L=0, …, basis=real)` at `BASE_SHA` for every case in `cases.json`.
- Max correlation order (body length) = **8**. Longer bodies → code 3 / `ValueError`.
- No exception may unwind across the C ABI: inputs are validated before any ET call; the Julia side returns codes, never throws.
- Distribution `ace-jax-coupling`, import `ace_jax_coupling`, version `0.1.0`, `requires-python >=3.11`, runtime deps `numpy>=1.24` only. Wheel tags: `py3-none-manylinux_2_28_x86_64`, `py3-none-manylinux_2_28_aarch64`, `py3-none-macosx_11_0_arm64`.
- Every Mach-O in the macOS bundle has `minos ≤ 11.0` (build links with `MACOSX_DEPLOYMENT_TARGET=11.0`; the probe's unset build produced `minos 26.0`, unloadable on older macOS). Linux bundle needs no glibc symbol newer than `GLIBC_2.28`.
- `cpu_target = "generic"` for the image (portable; coupling is not hot code). Build with `--jl-option handle-signals=no` (Python owns signals).
- ace-jax core (no extras) installs and its full test suite passes without `ace-jax-coupling` installed and without Julia.
- `authoring` extra: `ace-jax-coupling==0.1.0` with marker `(sys_platform == "linux" and (platform_machine == "x86_64" or platform_machine == "aarch64")) or (sys_platform == "darwin" and platform_machine == "arm64")`.
- `ACEJAX_NO_JULIA=1` keeps its meaning: any attempt to compute a coupling (i.e. to use the library) raises.
- No PyPI/TestPyPI publication in this plan. Before any publication the licence declaration of the wheel must be decided with the user (the bundle includes SuiteSparse UMFPACK/CHOLMOD, believed GPL-2.0-or-later — verify in Task 8 — plus BSD/MIT libraries); this plan only ships complete third-party notices inside the wheel.
- Commit only on `feat/trim-coupling` (ace-jax) and `jk/trim-safe-coupling` (fork). Pushing to the fork is authorised by the user; do not open PRs against ACEsuit.

## Review Focus

1. **Wheel built on a new macOS loaded on an older one** — must load (minos ≤ 11.0 on every Mach-O). Test: Task 6 `check_bundle.py` minos assertion; Task 8 re-runs it on the pruned tree.
2. **Ctrl-C in a Python process after the library is loaded** — must still raise `KeyboardInterrupt`, not be swallowed or crash (Julia installs signal handlers by default). Test: Task 7 `test_sigint_still_raises_keyboardinterrupt`.
3. **Malformed specs from Python** (an mb `(n,l)` missing from `Rnl_spec`, a missing `(l,m)` in `Ylm_spec`, body length 9, empty `mb_spec`, negative `l`, `|m|>l`, numpy integer scalars / tuples / lists mixed) — must raise `ValueError` with a message, never segfault; and a raw ctypes call with bad input returns 2, not a crash. Tests: Task 5 (Julia `validate` codes) and Task 7 (`test_invalid_inputs_raise`, `test_raw_invalid_returns_code`).
4. **Clean machine: no Julia, empty HOME** — the wheel must work without touching `~/.julia` or the network. Test: Task 8 `test_wheel.sh` runs with `HOME=<empty tmp>` and asserts the directory is still empty afterwards and `julia` is not on PATH.
5. **Concurrent calls from several Python threads** — results identical, no crash (Julia runtime entered from foreign threads). Test: Task 7 `test_threads_concurrent_calls` (the Python wrapper serialises with a lock; the test pins that).

---

## Phase A — ET fork

### Task 1: Fork branch + `Val(L)` construction API (functional, normal Julia)

**Files (repo `~/gits/EquivariantTensors.jl`):**
- Modify: `src/O3/O3.jl` (`_coupling_coeffs`, `coupling_coeffs_new`, new `coupling_coeffs(::Val{L}, ::Vector{Int}, ::Vector{Int})`)
- Modify: `src/utils/symmop.jl` (split into shared body + `Integer` and `Val` methods)
- Modify: `src/ace/sparse_ace_utils.jl` (`sparse_equivariant_tensor_spec`, shared `_tensor_specs`)
- Create: `test/test_utils/trim_cases.jl`, `test/test_trim_api.jl`
- Modify: `test/runtests.jl` (include the new test)

**Interfaces:**
- Produces:
  - `EquivariantTensors.O3.MAX_STATIC_ORDER::Int == 8`
  - `EquivariantTensors.O3.coupling_coeffs(::Val{L}, ll::Vector{Int}, nn::Vector{Int}; PI::Bool=true, basis=complex) -> (Matrix{T}, Vector{Vector{Int}})`, `T = L==0 ? Float64 : SVector{2L+1,Float64}`; `ArgumentError` for `length(ll) > 8` or length mismatch.
  - `EquivariantTensors.symmetrisation_matrix(::Val{L}, mb_spec; prune=false, PI=true, basis=complex) -> (SparseMatrixCSC, Vector{Vector{NLM}})`, identical results to `symmetrisation_matrix(L::Integer, mb_spec; prune, PI, basis)`.
  - `EquivariantTensors.sparse_equivariant_tensor_spec(::Val{L}; mb_spec, Rnl_spec, Ylm_spec, basis=real) -> NamedTuple{(:symm, :𝔸spec, :Aspec, :Aspec_raw, :𝔸spec_raw)}` — the layer-free half of `sparse_equivariant_tensor`.
  - `test/test_utils/trim_cases.jl`: `TRIM_CASES::Vector{NTuple{3,Int}}`, `trim_mb_spec(maxn, maxl, ord)`, `trim_rnl(maxn, maxl)`, `trim_ylm(maxl)`.

- [ ] **Step 1: Sync the fork and branch**

```bash
gh repo sync jameskermode/EquivariantTensors.jl --source ACEsuit/EquivariantTensors.jl --branch main
cd ~/gits && git clone https://github.com/jameskermode/EquivariantTensors.jl.git && cd EquivariantTensors.jl
git remote add upstream https://github.com/ACEsuit/EquivariantTensors.jl.git && git fetch upstream
git checkout -b jk/trim-safe-coupling upstream/main
git rev-parse HEAD      # record this as BASE_SHA (probe: 5342310...)
```
Write `BASE_SHA` into the task's commit message and keep it for Tasks 4 and 5.

- [ ] **Step 2: Write the shared case generator**

`test/test_utils/trim_cases.jl`:
```julia
# Small L = 0 coupling cases shared by test_trim_api.jl and test/trim/entry.jl.
const TRIM_NL = @NamedTuple{n::Int, l::Int}
const TRIM_LM = @NamedTuple{l::Int, m::Int}

# (maxn, maxl, max correlation order); (1, 2, 4) has degenerate nnll blocks,
# (1, 0, 8) reaches the table bound.
const TRIM_CASES = [(2, 2, 2), (2, 2, 3), (1, 2, 4), (1, 1, 5), (1, 0, 8)]

"""All sorted bodies of (n,l) channels (n ≤ maxn, l ≤ maxl) of length 1..ord with even ∑l."""
function trim_mb_spec(maxn::Int, maxl::Int, ord::Int)
   chans = [(n = n, l = l) for l in 0:maxl for n in 1:maxn]
   mb = Vector{TRIM_NL}[]
   bb = TRIM_NL[]
   function rec(start::Int)
      if !isempty(bb) && iseven(sum(b.l for b in bb))
         push!(mb, copy(bb))
      end
      length(bb) == ord && return
      for i in start:length(chans)
         push!(bb, chans[i]); rec(i); pop!(bb)
      end
   end
   rec(1)
   return mb
end

trim_rnl(maxn::Int, maxl::Int) = TRIM_NL[(n = n, l = l) for l in 0:maxl for n in 1:maxn]
trim_ylm(maxl::Int) = TRIM_LM[(l = l, m = m) for l in 0:maxl for m in -l:l]
```

- [ ] **Step 3: Write the failing test**

`test/test_trim_api.jl`:
```julia
using EquivariantTensors, Test
import EquivariantTensors as ET
include(joinpath(@__DIR__, "test_utils", "trim_cases.jl"))

@testset "Val(L) construction API == Integer API" begin
   for (maxn, maxl, ord) in TRIM_CASES
      mb = trim_mb_spec(maxn, maxl, ord); R = trim_rnl(maxn, maxl); Y = trim_ylm(maxl)
      S1, A1 = ET.symmetrisation_matrix(0, mb; prune = true, PI = true, basis = real)
      S2, A2 = ET.symmetrisation_matrix(Val(0), mb; prune = true, PI = true, basis = real)
      @test S1 == S2
      @test A1 == A2
      ref = ET.sparse_equivariant_tensor(L = 0, mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      t = ET.sparse_equivariant_tensor_spec(Val(0); mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      @test Matrix(t.symm) == Matrix(ref.A2Bmaps[1])
      @test t.𝔸spec == ref.meta["𝔸spec"]
      @test t.Aspec == ref.meta["Aspec"]
      @test collect(t.Aspec_raw) == collect(ref.abasis.spec)
   end
   # complex basis and L = 1 go through the same table
   mb = trim_mb_spec(1, 2, 3)
   @test ET.symmetrisation_matrix(Val(0), mb; PI = true, basis = complex) ==
         ET.symmetrisation_matrix(0, mb; PI = true, basis = complex)
   @test ET.symmetrisation_matrix(Val(1), mb; PI = true, basis = real) ==
         ET.symmetrisation_matrix(1, mb; PI = true, basis = real)
   # the static table is bounded
   mb9 = [[(n = 1, l = 0) for _ in 1:9]]
   @test_throws ArgumentError ET.symmetrisation_matrix(Val(0), mb9; PI = true, basis = real)
   @test_throws ArgumentError ET.O3.coupling_coeffs(Val(0), [0, 0], [1]; PI = true, basis = real)
end
```
Add to `test/runtests.jl` inside the `"O3-Coupling"` testset:
```julia
    @testset "Trim-safe Val(L) API" begin include("test_trim_api.jl"); end
```

- [ ] **Step 4: Run it to verify it fails**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=. -e 'using Pkg; Pkg.instantiate(); include("test/test_trim_api.jl")'`
Expected: FAIL — `MethodError: no method matching symmetrisation_matrix(::Val{0}, ...)`.

- [ ] **Step 5: Lift `L` into the type domain in `src/O3/O3.jl`**

Replace the `_coupling_coeffs` signature (keep the body) and add the `Int` forwarder:
```julia
_coupling_coeffs(L::Int, ll::SVector{N, Int}, nn::SVector{N, Int}; kwargs...) where {N} =
      _coupling_coeffs(Val(L), ll, nn; kwargs...)

function _coupling_coeffs(::Val{L}, ll::SVector{N, Int}, nn::SVector{N, Int};
                          PI = true, basis::B = complex, refl_sym::Union{Symbol,Nothing} = nothing) where {L, N, B}
```
Inside that body change the three recursive/new-solver calls:
- `C, MM = coupling_coeffs_new(L, ll, nn)` → `C, MM = coupling_coeffs_new(Val(L), ll, nn)`
- `_coupling_coeffs(L, ll, nn; PI = false, basis=complex, refl_sym = refl_sym)` → `_coupling_coeffs(Val(L), ll, nn; PI = false, basis=complex, refl_sym = refl_sym)`
- `_coupling_coeffs(L, ll, nn, PI = PI, basis=complex, refl_sym = refl_sym)` → `_coupling_coeffs(Val(L), ll, nn, PI = PI, basis=complex, refl_sym = refl_sym)`

Replace the `coupling_coeffs_new` signature (keep the body):
```julia
coupling_coeffs_new(K::Int, ll::SVector{N,Int}, nn::SVector{N,Int}) where N =
      coupling_coeffs_new(Val(K), ll, nn)

function coupling_coeffs_new(::Val{K}, ll::SVector{N,Int}, nn::SVector{N,Int}) where {K, N}
```
Add, directly after the existing `coupling_coeffs(L::Integer, ll, nn = nothing; …)` method:
```julia
"""
    O3.coupling_coeffs(Val(L), ll::Vector{Int}, nn::Vector{Int}; PI = true, basis = complex)

Type-stable variant of `coupling_coeffs` for ahead-of-time compilation
(`juliac --trim`): `L` is a compile-time constant and the correlation order
`N = length(ll)` is resolved through a fixed table, `N ≤ MAX_STATIC_ORDER`.
Returns `(C::Matrix{T}, MM::Vector{Vector{Int}})` with
`T = L == 0 ? Float64 : SVector{2L+1, Float64}`.
"""
function coupling_coeffs(::Val{L}, ll::Vector{Int}, nn::Vector{Int};
                         PI::Bool = true, basis::B = complex) where {L, B}
   N = length(ll)
   length(nn) == N || throw(ArgumentError("coupling_coeffs: ll and nn must have the same length"))
   N == 1 && return _coupling_coeffs_static(Val(L), Val(1), ll, nn, PI, basis)
   N == 2 && return _coupling_coeffs_static(Val(L), Val(2), ll, nn, PI, basis)
   N == 3 && return _coupling_coeffs_static(Val(L), Val(3), ll, nn, PI, basis)
   N == 4 && return _coupling_coeffs_static(Val(L), Val(4), ll, nn, PI, basis)
   N == 5 && return _coupling_coeffs_static(Val(L), Val(5), ll, nn, PI, basis)
   N == 6 && return _coupling_coeffs_static(Val(L), Val(6), ll, nn, PI, basis)
   N == 7 && return _coupling_coeffs_static(Val(L), Val(7), ll, nn, PI, basis)
   N == 8 && return _coupling_coeffs_static(Val(L), Val(8), ll, nn, PI, basis)
   throw(ArgumentError("coupling_coeffs(::Val{L}, ...): correlation order must be ≤ 8"))
end

const MAX_STATIC_ORDER = 8

function _coupling_coeffs_static(::Val{L}, ::Val{N}, ll::Vector{Int}, nn::Vector{Int},
                                 PI::Bool, basis::B) where {L, N, B}
   T = L == 0 ? Float64 : SVector{2L+1, Float64}
   C, MM = _coupling_coeffs(Val(L), SVector{N, Int}(ntuple(i -> ll[i], Val(N))),
                            SVector{N, Int}(ntuple(i -> nn[i], Val(N))); PI = PI, basis = basis)
   return Matrix{T}(C), Vector{Int}[Vector{Int}(mm) for mm in MM]
end
```

- [ ] **Step 6: Split `symmetrisation_matrix` in `src/utils/symmop.jl`**

Replace the whole existing `function symmetrisation_matrix(L::Integer, mb_spec; prune = false, kwargs...) … end` with the three definitions below. The shared body `_symmetrisation_matrix` is the existing body verbatim except: `TVAL` comes from the argument, and the coupling call is `cc, MM = ccfun(ll, nn)` (note the existing code passes `(L, ll, nn)` to `O3.coupling_coeffs`). Keep the existing docstring on the `Integer` method and add the new one on the `Val` method.
```julia
function symmetrisation_matrix(L::Integer, mb_spec; prune = false, kwargs...)
   _L = Int(L)
   TVAL = _L == 0 ? Float64 : SVector{2*_L+1, Float64}
   ccfun = (ll, nn) -> O3.coupling_coeffs(_L, ll, nn; kwargs...)
   return _symmetrisation_matrix(TVAL, mb_spec, ccfun; prune = prune)
end

"""
   symmetrisation_matrix(Val(L), mb_spec; prune = false, PI = true, basis = complex)

Same result as `symmetrisation_matrix(L, mb_spec; prune, PI, basis)`, but
type-stable for ahead-of-time compilation (`juliac --trim`): `L` is a
compile-time constant and the correlation order is limited to
`O3.MAX_STATIC_ORDER`.
"""
function symmetrisation_matrix(::Val{L}, mb_spec; prune = false, PI::Bool = true,
                               basis::B = complex) where {L, B}
   TVAL = L == 0 ? Float64 : SVector{2*L+1, Float64}
   ccfun = (ll, nn) -> O3.coupling_coeffs(Val(L), ll, nn; PI = PI, basis = basis)
   return _symmetrisation_matrix(TVAL, mb_spec, ccfun; prune = prune)
end

function _symmetrisation_matrix(::Type{TVAL}, mb_spec, ccfun::F; prune = false) where {TVAL, F}
   function _vecnt2nnll(bb)
      nn = Int[ b.n for b in bb ]
      ll = Int[ b.l for b in bb ]
      return nn, ll
   end
   nnll = unique(_vecnt2nnll.(mb_spec))
   irow = Int[]; jcol = Int[]; val = TVAL[]
   𝔸spec = Vector{@NamedTuple{n::Int64, l::Int64, m::Int64}}[]
   num𝔹 = 0
   num𝔸 = 0
   for (nn, ll) in nnll
      cc, MM = ccfun(ll, nn)
      # ... the remainder of the existing loop body and post-processing, unchanged ...
   end
   # ... unchanged: sparse assembly, zero-row check (@warn), prune, return ...
end
```
(`SVector` is available in `symmop.jl` via the module's `using StaticArrays`; if not, add `using StaticArrays: SVector` at the top of the file.) Copy the remainder of the body literally from the existing function — do not re-type it.

- [ ] **Step 7: Add `sparse_equivariant_tensor_spec` in `src/ace/sparse_ace_utils.jl`**

In `sparse_equivariant_tensor`, replace the four lines from `Aspec = sort( unique( reduce(vcat, 𝔸spec) ) )` through `𝔸spec_raw = _make_idx_AA_spec(𝔸spec, Aspec)` with:
```julia
   sp = _tensor_specs(symm, 𝔸spec, Rnl_spec, Ylm_spec)
   Aspec, Aspec_raw, 𝔸spec_raw = sp.Aspec, sp.Aspec_raw, sp.𝔸spec_raw
```
and add after the function:
```julia
"""
   sparse_equivariant_tensor_spec(Val(L); mb_spec, Rnl_spec, Ylm_spec, basis = real)

The layer-free, type-stable half of `sparse_equivariant_tensor` (usable under
`juliac --trim`): returns `(symm, 𝔸spec, Aspec, Aspec_raw, 𝔸spec_raw)`, where
`symm` equals `sparse_equivariant_tensor(...).A2Bmaps[1]`, `𝔸spec` its
`meta["𝔸spec"]` and `Aspec_raw` its `abasis.spec`.
"""
function sparse_equivariant_tensor_spec(::Val{L}; mb_spec, Rnl_spec, Ylm_spec,
                                        basis::B = real) where {L, B}
   symm, 𝔸spec = symmetrisation_matrix(Val(L), mb_spec; prune = true, PI = true, basis = basis)
   return _tensor_specs(symm, 𝔸spec, Rnl_spec, Ylm_spec)
end

function _tensor_specs(symm, 𝔸spec, Rnl_spec, Ylm_spec)
   Aspec = sort( unique( reduce(vcat, 𝔸spec) ) )
   Aspec_raw = _make_idx_A_spec(Aspec, Rnl_spec, Ylm_spec)
   𝔸spec_raw = _make_idx_AA_spec(𝔸spec, Aspec)
   return (symm = symm, 𝔸spec = 𝔸spec, Aspec = Aspec, Aspec_raw = Aspec_raw, 𝔸spec_raw = 𝔸spec_raw)
end
```

- [ ] **Step 8: Run the new test, then the whole suite**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=. -e 'include("test/test_trim_api.jl")'`
Expected: PASS (all `@test`s).
Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=. -e 'using Pkg; Pkg.test()'` and `julia +1.11 --project=. -e 'using Pkg; Pkg.test()'`
Expected: both PASS with the same test counts as on `BASE_SHA` plus the new testset (run `Pkg.test()` on `BASE_SHA` first if you need the baseline count).

- [ ] **Step 9: Commit**

```bash
git add src/O3/O3.jl src/utils/symmop.jl src/ace/sparse_ace_utils.jl test/test_utils/trim_cases.jl test/test_trim_api.jl test/runtests.jl
git commit -m "feat: Val(L) coupling-construction API (type-stable half of sparse_equivariant_tensor)

Base: ACEsuit/main @ <BASE_SHA>"
```

---

### Task 2: juliac `--trim=safe` compile test + trim-safety patches

**Files (fork):**
- Create: `test/trim/Project.toml`, `test/trim/juliac_env/Project.toml`, `test/trim/entry.jl`, `test/trim/build_and_run.jl`
- Modify: `src/O3/O3.jl`, `src/utils/invmap.jl` (and any further site the verifier names — see Step 5)
- Modify: `.gitignore` (ignore `test/trim/**/Manifest.toml` if Manifests are not already ignored)

**Interfaces:**
- Consumes: Task 1's `sparse_equivariant_tensor_spec(Val(0); …)`, `trim_cases.jl`.
- Produces: `julia +1.13 --project=test/trim test/trim/build_and_run.jl` exits 0 iff the trimmed executable reproduces the uncompiled checksums.

- [ ] **Step 1: Write the trim test harness**

`test/trim/Project.toml`:
```toml
[deps]
EquivariantTensors = "5e107534-7145-4f8f-b06f-47a52840c895"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[sources]
EquivariantTensors = {path = "../.."}
```
`test/trim/juliac_env/Project.toml`:
```toml
[deps]
JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"

[compat]
JuliaC = "=0.3.10"
```
`test/trim/entry.jl`:
```julia
# juliac --trim=safe entry point exercising the trim-safe construction API.
# test/trim/build_and_run.jl compiles this file and compares the executable's
# output with checksum_lines() evaluated by ordinary Julia.
import EquivariantTensors as ET
using SparseArrays: findnz
include(joinpath(@__DIR__, "..", "test_utils", "trim_cases.jl"))

_mix(h::UInt64, x::UInt64) = (h ⊻ x) * 0x100000001b3

function checksum_lines()::Vector{String}
   out = String[]
   for (maxn, maxl, ord) in TRIM_CASES
      mb = trim_mb_spec(maxn, maxl, ord); R = trim_rnl(maxn, maxl); Y = trim_ylm(maxl)
      t = ET.sparse_equivariant_tensor_spec(Val(0); mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      I, J, V = findnz(t.symm)
      h = 0xcbf29ce484222325
      for k in eachindex(V)
         h = _mix(h, UInt64(I[k])); h = _mix(h, UInt64(J[k])); h = _mix(h, reinterpret(UInt64, V[k]))
      end
      for (r, y) in t.Aspec_raw
         h = _mix(h, UInt64(r)); h = _mix(h, UInt64(y))
      end
      for bb in t.𝔸spec, b in bb
         h = _mix(h, UInt64(b.n)); h = _mix(h, UInt64(b.l)); h = _mix(h, UInt64(b.m + 1024))
      end
      push!(out, string(size(t.symm, 1), " ", size(t.symm, 2), " ", length(V), " ", h))
   end
   return out
end

function (@main)(args::Vector{String})::Cint
   for line in checksum_lines()
      println(Core.stdout, line)
   end
   return 0
end
```
`test/trim/build_and_run.jl`:
```julia
# Compile test/trim/entry.jl with `juliac --trim=safe` and check the executable
# reproduces the uncompiled results.  Needs Julia >= 1.13:
#     julia +1.13 --project=test/trim test/trim/build_and_run.jl
VERSION >= v"1.13" || error("trim test needs Julia >= 1.13 (got $VERSION)")
const HERE = @__DIR__
const JC_ENV = joinpath(HERE, "juliac_env")
const JULIA = Base.julia_cmd()
run(`$JULIA --project=$HERE -e "using Pkg; Pkg.instantiate()"`)
run(`$JULIA --project=$JC_ENV -e "using Pkg; Pkg.instantiate()"`)
out = mktempdir()
run(Cmd(`$JULIA --project=$JC_ENV -e "using JuliaC; JuliaC.main(ARGS)" -- --output-exe trim_entry --project=$HERE --trim=safe --experimental --bundle $out $(joinpath(HERE, "entry.jl"))`; dir = out))
exe = joinpath(out, "bin", Sys.iswindows() ? "trim_entry.exe" : "trim_entry")
isfile(exe) || error("no executable at $exe; bundle has $(readdir(out))")
got = readlines(`$exe`)
ref = Module(:TrimEntryRef)
Base.include(ref, joinpath(HERE, "entry.jl"))
want = Base.invokelatest(getfield(ref, :checksum_lines))
got == want || error("trimmed executable differs from Julia:\n got  = $got\n want = $want")
println("trim test OK: $(length(got)) cases")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=test/trim test/trim/build_and_run.jl 2>&1 | tee /tmp/trim1.log | tail -20; grep -c '^Verifier error' /tmp/trim1.log`
Expected: FAIL — `ERROR: Failed to compile entry.jl` with several hundred `Verifier error` lines (probe: 606 on unpatched main; fewer now that Task 1 lifted `L`). Typical sites: `SetLl` (`Vector{Any}`), `basis::Function`, `invmap`, `ftranspose`, `UmfpackLU getproperty`.

- [ ] **Step 3: Apply the probe-validated patches**

In `src/O3/O3.jl`:
- `GCG(ll, mm, LL; vectorize::Bool=true, basis = complex) where N` → `GCG(ll::SVector{N,Int64}, mm::SVector{N,Int64}, LL::SVector{N,Int64}; vectorize::Bool=true, basis::B = complex) where {N, B}`.
- `mm_generate(L::Int, ll::SVector{N,Int}, nn::SVector{N,Int}; basis = complex, PI = false) where {N}` → `...; basis::B = complex, PI = false) where {N, B}`.
- `coupling_coeffs(L::Integer, ll, nn = nothing; PI = !(isnothing(nn)), basis = complex, refl_sym::Union{Symbol,Nothing} = nothing)` → `basis::B = complex, ...) where {B}`.
- Every `error("Unknown basis type: $basis")` → `error("Unknown basis type")` (three sites; interpolating a `Function` is not trim-safe).
- In `SetLl`: `set = Vector{Any}[]` → `set = Vector{Int}[]`.
- Replace the head of `solver_inner` and its last-block LU as follows (add the two helpers directly above `solver_inner`):
```julia
# trim-safe sparse transpose: `sparse(M')` goes through SparseArrays.ftranspose(A, f::Function),
# which --trim cannot resolve. (Real entries, so adjoint == transpose.)
function _spT(M::SparseMatrixCSC)
    I, J, V = findnz(M)
    return sparse(J, I, V, size(M, 2), size(M, 1))
end
_spT(M::AbstractMatrix) = sparse(transpose(M))

# trim-safe (L', p, Rs) of a sparse LU. UmfpackLU's getproperty is one method over all
# fields and its :L branch transposes via ftranspose(f::Function), which --trim cannot
# resolve -- so even F.p / F.Rs are unusable. One get_numeric call returns all three
# (L' is exactly the CSR L that getproperty(:L) would transpose).
function _umf_Lt_p_Rs(F::SparseArrays.UMFPACK.UmfpackLU{Float64, Int})
    U = SparseArrays.UMFPACK
    U.umfpack_numeric!(F)
    lnz, unz, n_row, n_col, nz_diag = U.umf_lunz(F)
    Lp = Vector{Int}(undef, n_row + 1); Lj = Vector{Int}(undef, lnz); Lx = Vector{Float64}(undef, lnz)
    P = Vector{Int}(undef, n_row); Rs = Vector{Float64}(undef, n_row)
    SparseArrays.LibSuiteSparse.umfpack_dl_get_numeric(Lp, Lj, Lx, C_NULL, C_NULL, C_NULL,
        P, C_NULL, C_NULL, C_NULL, Rs, getfield(F, :numeric))
    Lt = SparseMatrixCSC(min(n_row, n_col), n_row, U.increment!(Lp), U.increment!(Lj), Lx)
    return Lt, U.increment!(P), Rs
end
```
In `solver_inner`: `M = sparse(M')` → `M = _spT(M)`; and
```julia
        F = lu(B')
        invp = invperm(F.p)
        sparse_ns = nullspace_upper_sparse(sparse(F.L'))
        C[:, prev_col_block] .= (Diagonal(F.Rs) * sparse_ns[invp, :])'
```
→
```julia
        F = lu(_spT(B))
        Lt, Fp, FRs = _umf_Lt_p_Rs(F)
        invp = invperm(Fp)
        sparse_ns = nullspace_upper_sparse(Lt)
        C[:, prev_col_block] .= (Diagonal(FRs) * sparse_ns[invp, :])'
```
Leave the commented-out "method III" lines untouched. If `SparseArrays` is not a bound name inside `module O3`, add `import SparseArrays` next to its `using SparseArrays`.

In `src/utils/invmap.jl`: `function invmap(a::AbstractVector, hashfcn = identity)` → `function invmap(a::AbstractVector, hashfcn::F = identity) where {F}`.

- [ ] **Step 4: Re-run the trim test**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=test/trim test/trim/build_and_run.jl 2>&1 | tee /tmp/trim2.log | tail -5; grep -c '^Verifier error' /tmp/trim2.log`
Expected: either `trim test OK: 5 cases`, or a short list of remaining verifier errors inside `symmetrisation_matrix`'s post-processing (the probe replaced that code, so it was never verified under trim).

- [ ] **Step 5: Fix any remaining sites (only if Step 4 still fails)**

Fix each reported ET source line with the smallest type-stabilising change — never edit Base or stdlib. The two expected cases, with ready fixes:
- `sum(norm, symm; dims = 2)` / `dims = 1` in `_symmetrisation_matrix` (sparse `mapreduce` with a `Function`): replace the two `findall(!iszero, sum(norm, symm; dims = …)[:])` expressions by
```julia
_nz_rows(A::SparseMatrixCSC) = (m = falses(size(A, 1));
   for (k, v) in enumerate(nonzeros(A)); iszero(v) || (m[rowvals(A)[k]] = true); end; findall(m))
_nz_cols(A::SparseMatrixCSC) = [j for j in 1:size(A, 2) if any(!iszero, view(nonzeros(A), nzrange(A, j)))]
```
  (`i_nz_rows = _nz_rows(symm)`, `i_nz_cols = _nz_cols(symm)`; both already sorted.)
- `@warn(...)` in `_symmetrisation_matrix`: replace with `println(Core.stderr, "Warning: symmetrization matrix has all-zero rows; this indicates a bug in `coupling_coeffs`")`.
Re-run Step 4 after each fix until `trim test OK`. List every extra fix in the commit message.

- [ ] **Step 6: Full ET suite still passes**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=. -e 'using Pkg; Pkg.test()'` and `julia +1.11 --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, same counts as after Task 1.

- [ ] **Step 7: Commit**

```bash
git add src/ test/trim .gitignore
git commit -m "fix: make L=0 coupling construction juliac --trim=safe (Julia 1.13)

- specialise basis/hashfcn arguments (were widened to ::Function)
- SetLl: concrete Vector{Int}; no Function interpolation in error strings
- solver_inner: trim-safe sparse transpose and UMFPACK L'/p/Rs extraction
- test/trim: juliac compile test comparing the executable to Julia"
```

---

### Task 3: Fork CI trim job, push, record `ET_REV`

**Files (fork):** Modify `.github/workflows/CI.yml`

- [ ] **Step 1: Add the job** (append under `jobs:`; keep existing jobs unchanged)

```yaml
  trim:
    name: juliac --trim (Julia 1.13 - ${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    timeout-minutes: 60
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-14]
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.13'
      - uses: julia-actions/cache@v3
      - run: julia --project=test/trim test/trim/build_and_run.jl
```
Check the workflow's `on:` block. If pushes to non-default branches do not trigger it, add `workflow_dispatch:` to `on:`.

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/CI.yml
git commit -m "ci: juliac --trim job (Julia 1.13, ubuntu + macos)"
git push -u origin jk/trim-safe-coupling
```

- [ ] **Step 3: Verify CI**

Run: `gh run list -R jameskermode/EquivariantTensors.jl --branch jk/trim-safe-coupling --limit 3` then `gh run watch -R jameskermode/EquivariantTensors.jl <run-id>` (if no run started: `gh workflow run CI.yml -R jameskermode/EquivariantTensors.jl --ref jk/trim-safe-coupling`).
Expected: the existing test jobs and both `trim` jobs green. A red existing job that is also red on `BASE_SHA` upstream is not ours — note it in the task report; any other red is a blocker.

- [ ] **Step 4: Record `ET_REV`**

Run: `git rev-parse HEAD` → `ET_REV`. Tasks 4–6 use it.

---

## Phase B — ace-jax: shim library

All ace-jax work happens in `/Users/u1470235/gits/ace-jax/.worktrees/trim-coupling` (branch `feat/trim-coupling`, from `origin/main` a198a93).

### Task 4: Test cases + unpatched-ET reference data

**Files:**
- Create: `coupling/tools/gen_cases.py`, `coupling/python/tests/data/cases.json`
- Create: `coupling/julia/reference/Project.toml`, `coupling/julia/reference/reference.jl`, `coupling/julia/reference/Manifest.toml`, `coupling/python/tests/data/et_reference.npz`

**Interfaces:**
- Produces: `cases.json` = `{case_name: {"mb": [[[n,l],...],...], "R": [[n,l],...], "Y": [[l,m],...]}}`; `et_reference.npz` with, per case `C`, keys `C__A2B_shape (2,)`, `C__A2B_rows`, `C__A2B_cols` (int64, 0-based, `findnz` order), `C__A2B_vals` (float64), `C__aa_sig_off (n_AA+1,)`, `C__aa_sig (n_sig,3)`, `C__aspec (n_A,2)`, `C__aa_off (n_AA+1,)`, `C__aa_idx (n_aaidx,)`, `C__nnll_off (n_B+1,)`, `C__nnll (n_nnll,2)` — exactly the `RawCoupling` layout of Task 7.

- [ ] **Step 1: Write the case generator**

`coupling/tools/gen_cases.py`:
```python
"""Write coupling/python/tests/data/cases.json: the specs every coupling
backend test runs.  Run from the repo root with the ace-jax env:
    uv run python coupling/tools/gen_cases.py"""
import json
import pathlib

from ace_jax.construct.spec import build_spec, spec_from_reference, ylm_spec

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "coupling/python/tests/data/cases.json"


def _case(mb, R, Y):
    return {"mb": [[[int(n), int(l)] for n, l in bb] for bb in mb],
            "R": [[int(n), int(l)] for n, l in R],
            "Y": [[int(l), int(m)] for l, m in Y]}


def main():
    cases = {"tiny": _case([[(1, 0)], [(1, 1), (1, 1)]], [(1, 0), (1, 1), (2, 0)], ylm_spec(1))}
    for f in ["SiGe_o2d6", "CrMnFe_o2d5", "CrMnFe_o3d6", "SiGe_o4d5w05"]:
        mb, R, Y, _ = spec_from_reference(str(ROOT / f"fixtures/coupling_ref_{f}.npz"))
        cases[f] = _case(mb, R, Y)
    cases["build_NZ2_o5_d8"] = _case(*build_spec(2, 5, 8, 1.5))
    cases["build_NZ3_o4_d10"] = _case(*build_spec(3, 4, 10, 1.5))
    # reach the order-8 table bound: largest totaldegree with <= 3000 bodies
    for td in (3.0, 2.5, 2.0):
        mb, R, Y = build_spec(4, 8, td, 1.5)
        if len(mb) <= 3000 and max(map(len, mb)) == 8:
            cases["build_NZ4_o8"] = _case(mb, R, Y)
            break
    else:
        raise SystemExit("no order-8 case with <= 3000 bodies; lower NZ or totaldegree")
    for k, v in cases.items():
        print(f"{k:18s} n_mb={len(v['mb']):5d} max_order={max(map(len, v['mb']))}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, separators=(",", ":")))


if __name__ == "__main__":
    main()
```
Run: `uv run python coupling/tools/gen_cases.py`
Expected: 8 lines (tiny, 4 fixture shapes, 3 build_spec shapes); `build_NZ4_o8 ... max_order=8`; the file exists.

- [ ] **Step 2: Reference environment pinned to UNPATCHED upstream**

`coupling/julia/reference/Project.toml` (substitute the recorded `BASE_SHA`):
```toml
[deps]
EquivariantTensors = "5e107534-7145-4f8f-b06f-47a52840c895"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[sources]
EquivariantTensors = {url = "https://github.com/ACEsuit/EquivariantTensors.jl", rev = "<BASE_SHA>"}
```

- [ ] **Step 3: Write the reference generator**

`coupling/julia/reference/reference.jl`:
```julia
# Oracle for ace-jax-coupling: UNPATCHED upstream EquivariantTensors at BASE_SHA.
#   julia +1.13 --project=coupling/julia/reference coupling/julia/reference/reference.jl \
#         coupling/python/tests/data/cases.json coupling/python/tests/data/et_reference.npz
import EquivariantTensors as ET
using JSON, NPZ, SparseArrays

cases = JSON.parsefile(ARGS[1])
out = Dict{String, Any}()
for name in sort(collect(keys(cases)))
   c = cases[name]
   mb = [[(n = Int(b[1]), l = Int(b[2])) for b in bb] for bb in c["mb"]]
   R = [(n = Int(b[1]), l = Int(b[2])) for b in c["R"]]
   Y = [(l = Int(b[1]), m = Int(b[2])) for b in c["Y"]]
   t = ET.sparse_equivariant_tensor(L = 0, mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
   A2B = sparse(t.A2Bmaps[1])
   I, J, V = findnz(A2B)
   sig = t.meta["𝔸spec"]
   aa = [collect(s) for s in vcat(t.aabasis.specs...)]
   nnll = ET.get_nnll_spec(t, 1)
   cs(v) = Int64[0; cumsum(length.(v))]
   out["$(name)__A2B_shape"] = Int64[size(A2B)...]
   out["$(name)__A2B_rows"] = Int64.(I .- 1)
   out["$(name)__A2B_cols"] = Int64.(J .- 1)
   out["$(name)__A2B_vals"] = Float64.(V)
   out["$(name)__aa_sig_off"] = cs(sig)
   # (n_sig, 3) etc.: build (k, n) with hcat and transpose, so numpy reads row = one body
   out["$(name)__aa_sig"] = permutedims(reduce(hcat, [Int64[b.n, b.l, b.m] for bb in sig for b in bb]))
   out["$(name)__aspec"] = permutedims(reduce(hcat, [Int64[r - 1, y - 1] for (r, y) in t.abasis.spec]))
   out["$(name)__aa_off"] = cs(aa)
   out["$(name)__aa_idx"] = Int64[i - 1 for s in aa for i in s]
   out["$(name)__nnll_off"] = cs(nnll)
   out["$(name)__nnll"] = permutedims(reduce(hcat, [Int64[b.n, b.l] for bb in nnll for b in bb]))
   println(rpad(name, 18), " A2B ", size(A2B), " nnz=", length(V))
end
npzwrite(ARGS[2], out)
```

- [ ] **Step 4: Generate and sanity-check**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=coupling/julia/reference -e 'using Pkg; Pkg.instantiate()' && JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=coupling/julia/reference coupling/julia/reference/reference.jl coupling/python/tests/data/cases.json coupling/python/tests/data/et_reference.npz`
Expected: 8 lines; `SiGe_o2d6 A2B (57, 75) nnz=75`, `SiGe_o4d5w05 A2B (252, 987) nnz=1024`, `build_NZ3_o4_d10 A2B (6216, 17302) nnz=17917` (probe values at 5342310).
Run: `uv run python -c "import numpy as np; z=np.load('coupling/python/tests/data/et_reference.npz'); print(z['tiny__aa_sig'].shape, z['tiny__aspec'].shape, z['tiny__nnll'].shape)"`
Expected: three `(k, 3)`, `(k, 2)`, `(k, 2)` shapes (2-D, last dim 3/2/2).

- [ ] **Step 5: Commit**

```bash
git add coupling/tools/gen_cases.py coupling/python/tests/data coupling/julia/reference
git commit -m "test(coupling): cases + unpatched-upstream ET reference (BASE_SHA)"
```

---

### Task 5: Julia shim `ETCouple.jl` (C ABI) + uncompiled tests

**Files:**
- Create: `coupling/julia/Project.toml`, `coupling/julia/Manifest.toml`, `coupling/julia/src/ETCouple.jl`, `coupling/julia/test/runtests.jl`

**Interfaces:**
- Consumes: fork `ET_REV`: `ET.sparse_equivariant_tensor_spec(Val(0); …)`.
- Produces (Julia module `ETCouple`): `validate(mb, rnl, ylm)::Cint`, `compute(mb, rnl, ylm)::Result`, `required_sizes(::Result)::NTuple{7,Int}`, and the `@ccallable`s `etc_abi_version`, `etc_max_order`, `etc_couple` with the ABI in the Design section. Codes `OK=0, BUFFERS_TOO_SMALL=1, INVALID_INPUT=2, ORDER_TOO_HIGH=3`.

- [ ] **Step 1: Project**

`coupling/julia/Project.toml` (substitute `ET_REV` as the full 40-character sha — `build_info()["et_rev"]` is checked for length 40):
```toml
[deps]
EquivariantTensors = "5e107534-7145-4f8f-b06f-47a52840c895"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[sources]
EquivariantTensors = {url = "https://github.com/jameskermode/EquivariantTensors.jl", rev = "<ET_REV>"}

[compat]
julia = "1.13"
```
Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=coupling/julia -e 'using Pkg; Pkg.instantiate()'` (creates `Manifest.toml`; commit it — builds must be reproducible).

- [ ] **Step 2: Write the failing tests**

`coupling/julia/test/runtests.jl`:
```julia
# Uncompiled checks of the shim: `compute` against the real (patched-fork) ET
# objects, the validation codes, and the two-phase C entry called from Julia.
#   julia +1.13 --project=coupling/julia coupling/julia/test/runtests.jl
using Test
import EquivariantTensors as ET
include(joinpath(@__DIR__, "..", "src", "ETCouple.jl"))
using .ETCouple: validate, compute, required_sizes, etc_couple, OK, BUFFERS_TOO_SMALL, INVALID_INPUT, ORDER_TOO_HIGH

nl(n, l) = (n = n, l = l); lm(l, m) = (l = l, m = m)
ylm(L) = [lm(l, m) for l in 0:L for m in -l:l]
bodies(maxn, maxl, ord) = begin                      # sorted bodies, even ∑l
   ch = [nl(n, l) for l in 0:maxl for n in 1:maxn]; mb = Vector{ETCouple.NL}[]; bb = ETCouple.NL[]
   rec(s) = (!isempty(bb) && iseven(sum(b.l for b in bb)) && push!(mb, copy(bb));
             length(bb) == ord && return; for i in s:length(ch); push!(bb, ch[i]); rec(i); pop!(bb); end)
   rec(1); mb
end
CASES = [(2, 2, 2), (2, 2, 3), (1, 2, 4), (1, 0, 8)]

@testset "compute == sparse_equivariant_tensor" begin
   for (maxn, maxl, ord) in CASES
      mb = bodies(maxn, maxl, ord); R = [nl(n, l) for l in 0:maxl for n in 1:maxn]; Y = ylm(maxl)
      @test validate(mb, R, Y) == OK
      r = compute(mb, R, Y)
      t = ET.sparse_equivariant_tensor(L = 0, mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      A = Matrix(t.A2Bmaps[1])
      B = zeros(r.nB, r.nAA); for k in eachindex(r.V); B[r.I[k], r.J[k]] = r.V[k]; end
      @test B == A
      @test r.sig == t.meta["𝔸spec"]
      @test r.aspec == collect(t.abasis.spec)
      @test r.aa == [collect(s) for s in vcat(t.aabasis.specs...)]      # SparseSymmProd order
      @test r.nnll == ET.get_nnll_spec(t, 1)
   end
end

@testset "validate" begin
   R = [nl(1, 0), nl(1, 1)]; Y = ylm(1)
   @test validate([[nl(1, 0)]], R, Y) == OK
   @test validate(Vector{ETCouple.NL}[], R, Y) == INVALID_INPUT          # empty mb
   @test validate([ETCouple.NL[]], R, Y) == INVALID_INPUT                # empty body
   @test validate([[nl(2, 0)]], R, Y) == INVALID_INPUT                   # (2,0) not in Rnl
   @test validate([[nl(1, 1), nl(1, 1)]], R, [lm(0, 0), lm(1, 0)]) == INVALID_INPUT   # (1,±1) missing
   @test validate([[nl(1, 0)]], [nl(1, 0), nl(1, 0)], Y) == INVALID_INPUT             # duplicate Rnl
   @test validate([[nl(1, 0)]], R, [lm(0, 0), lm(0, 0)]) == INVALID_INPUT             # duplicate Ylm
   @test validate([[nl(1, 0)]], R, [lm(1, 2)]) == INVALID_INPUT                       # |m| > l
   @test validate([[nl(1, -1)]], [nl(1, -1)], Y) == INVALID_INPUT                     # l < 0
   @test validate([[nl(1, 0) for _ in 1:9]], R, Y) == ORDER_TOO_HIGH
end

@testset "etc_couple two-phase from Julia" begin
   mb = [[nl(1, 0)], [nl(1, 1), nl(1, 1)]]; R = [nl(1, 0), nl(1, 1), nl(2, 0)]; Y = ylm(1)
   off = Int64[0, 1, 3]; mn = Int64[1, 1, 1]; ml = Int64[0, 1, 1]
   rn = Int64[1, 1, 2]; rl = Int64[0, 1, 0]; yl = Int64[y.l for y in Y]; ym = Int64[y.m for y in Y]
   sizes = zeros(Int64, 7); z = zeros(Int64, 1); zf = zeros(Float64, 1)
   call(ai, aj, av, so, sg, asp, ao, ax, no, nn) = GC.@preserve off mn ml rn rl yl ym sizes ai aj av so sg asp ao ax no nn etc_couple(
      Int64(2), pointer(off), pointer(mn), pointer(ml), Int64(3), pointer(rn), pointer(rl),
      Int64(length(Y)), pointer(yl), pointer(ym), pointer(sizes),
      pointer(ai), pointer(aj), pointer(av), pointer(so), pointer(sg), pointer(asp),
      pointer(ao), pointer(ax), pointer(no), pointer(nn))
   @test call(z, z, zf, z, z, z, z, z, z, z) == BUFFERS_TOO_SMALL
   r = compute(mb, R, Y)
   @test Tuple(sizes) == required_sizes(r)
   nnz, nB, nAA, nsig, nA, naa, nnl = sizes
   bufs = (zeros(Int64, nnz), zeros(Int64, nnz), zeros(nnz), zeros(Int64, nAA + 1), zeros(Int64, 3nsig),
           zeros(Int64, 2nA), zeros(Int64, nAA + 1), zeros(Int64, naa), zeros(Int64, nB + 1), zeros(Int64, 2nnl))
   @test call(bufs...) == OK
   @test bufs[1] == r.I .- 1 && bufs[2] == r.J .- 1 && bufs[3] == r.V
   @test bufs[4][end] == nsig && bufs[7][end] == naa && bufs[9][end] == nnl
   bad_off = Int64[0, 2, 1]                                               # non-monotone offsets (in bounds)
   @test GC.@preserve bad_off mn ml rn rl yl ym sizes etc_couple(Int64(2), pointer(bad_off), pointer(mn), pointer(ml),
      Int64(3), pointer(rn), pointer(rl), Int64(length(Y)), pointer(yl), pointer(ym), pointer(sizes),
      C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL) == INVALID_INPUT
end
```
Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=coupling/julia coupling/julia/test/runtests.jl`
Expected: FAIL — `SystemError: opening file ".../src/ETCouple.jl": No such file`.

- [ ] **Step 3: Implement the shim**

`coupling/julia/src/ETCouple.jl`:
```julia
"""
    ETCouple

C entry point (compiled with `juliac --trim=safe`, see ../build.jl) for the L = 0,
real-basis SO(3) coupling of EquivariantTensors.  Stateless: every call computes
from its inputs and writes only into caller-owned buffers; no Julia object
outlives a call.  Two-phase: a call whose capacities (`sizes`) are too small
returns BUFFERS_TOO_SMALL with the required sizes written back.  Inputs are
validated first so no exception can unwind across the C boundary.
ABI: docs/dev/coupling-etshim-spec.md.
"""
module ETCouple

import EquivariantTensors as ET
using SparseArrays: findnz

const ABI_VERSION = Cint(1)
const MAX_ORDER = 8                       # == ET.O3.MAX_STATIC_ORDER
const OK = Cint(0)
const BUFFERS_TOO_SMALL = Cint(1)
const INVALID_INPUT = Cint(2)
const ORDER_TOO_HIGH = Cint(3)
const NSIZES = 7                          # [nnz, n_B, n_AA, n_sig, n_A, n_aaidx, n_nnll]

const NL = @NamedTuple{n::Int, l::Int}
const LM = @NamedTuple{l::Int, m::Int}
const NLM = @NamedTuple{n::Int, l::Int, m::Int}

"""Everything the ABI returns, 1-based (converted to 0-based on write)."""
struct Result
   nB::Int
   nAA::Int
   I::Vector{Int}                         # A2B triplets, findnz order
   J::Vector{Int}
   V::Vector{Float64}
   sig::Vector{Vector{NLM}}               # per A2B column: meta["𝔸spec"]
   aspec::Vector{Tuple{Int, Int}}         # abasis.spec: (Rnl_idx, Ylm_idx)
   aa::Vector{Vector{Int}}                # aabasis evaluation order (SparseSymmProd)
   nnll::Vector{Vector{NL}}               # get_nnll_spec(tensor, 1)
end

function validate(mb::Vector{Vector{NL}}, rnl::Vector{NL}, ylm::Vector{LM})::Cint
   (isempty(mb) || isempty(rnl) || isempty(ylm)) && return INVALID_INPUT
   length(unique(rnl)) == length(rnl) || return INVALID_INPUT
   length(unique(ylm)) == length(ylm) || return INVALID_INPUT
   for r in rnl
      r.l >= 0 || return INVALID_INPUT
   end
   for y in ylm
      (y.l >= 0 && abs(y.m) <= y.l) || return INVALID_INPUT
   end
   rset = Set{NL}(rnl)
   yset = Set{LM}(ylm)
   for bb in mb
      isempty(bb) && return INVALID_INPUT
      length(bb) > MAX_ORDER && return ORDER_TOO_HIGH
      for b in bb
         b.l >= 0 || return INVALID_INPUT
         b in rset || return INVALID_INPUT
         for m in -b.l:b.l
            (l = b.l, m = m) in yset || return INVALID_INPUT
         end
      end
   end
   return OK
end

function compute(mb::Vector{Vector{NL}}, rnl::Vector{NL}, ylm::Vector{LM})::Result
   t = ET.sparse_equivariant_tensor_spec(Val(0); mb_spec = mb, Rnl_spec = rnl, Ylm_spec = ylm, basis = real)
   I, J, V = findnz(t.symm)
   nB, nAA = size(t.symm)
   # SparseSymmProd(𝔸spec_raw) evaluation order: the constructor's own
   # `sort(spec, by = length)` (stable); each body is already sorted by
   # _make_idx_AA_spec.
   aa = sort(t.𝔸spec_raw; by = length)
   # get_nnll_spec: (n, l) of each row's first stored column
   first_col = fill(typemax(Int), nB)
   for k in eachindex(I)
      J[k] < first_col[I[k]] && (first_col[I[k]] = J[k])
   end
   nnll = Vector{NL}[NL[(n = b.n, l = b.l) for b in t.𝔸spec[first_col[i]]] for i in 1:nB]
   aspec = Tuple{Int, Int}[(Int(p[1]), Int(p[2])) for p in t.Aspec_raw]
   return Result(nB, nAA, I, J, Vector{Float64}(V), t.𝔸spec, aspec, aa, nnll)
end

required_sizes(r::Result) = (length(r.V), r.nB, r.nAA, sum(length, r.sig; init = 0),
                             length(r.aspec), sum(length, r.aa; init = 0), sum(length, r.nnll; init = 0))

# read mb from CSR offsets; `nothing` if the offsets are malformed
function _read_mb(n_mb::Int64, off::Ptr{Int64}, pn::Ptr{Int64}, pl::Ptr{Int64})
   unsafe_load(off, 1) == 0 || return nothing
   mb = Vector{Vector{NL}}(undef, n_mb)
   for b in 1:n_mb
      lo = unsafe_load(off, b); hi = unsafe_load(off, b + 1)
      hi >= lo || return nothing
      body = Vector{NL}(undef, hi - lo)
      for k in 1:(hi - lo)
         body[k] = (n = Int(unsafe_load(pn, lo + k)), l = Int(unsafe_load(pl, lo + k)))
      end
      mb[b] = body
   end
   return mb
end

Base.@ccallable function etc_abi_version()::Cint
   return ABI_VERSION
end

Base.@ccallable function etc_max_order()::Cint
   return Cint(MAX_ORDER)
end

Base.@ccallable function etc_couple(
      n_mb::Int64, mb_off::Ptr{Int64}, mb_n::Ptr{Int64}, mb_l::Ptr{Int64},
      n_r::Int64, r_n::Ptr{Int64}, r_l::Ptr{Int64},
      n_y::Int64, y_l::Ptr{Int64}, y_m::Ptr{Int64},
      sizes::Ptr{Int64},
      a2b_i::Ptr{Int64}, a2b_j::Ptr{Int64}, a2b_v::Ptr{Float64},
      sig_off::Ptr{Int64}, sig::Ptr{Int64}, aspec::Ptr{Int64},
      aa_off::Ptr{Int64}, aa_idx::Ptr{Int64},
      nnll_off::Ptr{Int64}, nnll::Ptr{Int64})::Cint
   (n_mb < 1 || n_r < 1 || n_y < 1) && return INVALID_INPUT
   mb = _read_mb(n_mb, mb_off, mb_n, mb_l)
   mb === nothing && return INVALID_INPUT
   rnl = NL[(n = Int(unsafe_load(r_n, k)), l = Int(unsafe_load(r_l, k))) for k in 1:n_r]
   ylm = LM[(l = Int(unsafe_load(y_l, k)), m = Int(unsafe_load(y_m, k))) for k in 1:n_y]
   code = validate(mb, rnl, ylm)
   code == OK || return code
   r = compute(mb, rnl, ylm)
   need = required_sizes(r)
   short = false
   for k in 1:NSIZES
      unsafe_load(sizes, k) < need[k] && (short = true)
      unsafe_store!(sizes, need[k], k)
   end
   short && return BUFFERS_TOO_SMALL
   for k in eachindex(r.V)
      unsafe_store!(a2b_i, r.I[k] - 1, k); unsafe_store!(a2b_j, r.J[k] - 1, k); unsafe_store!(a2b_v, r.V[k], k)
   end
   p = 0; unsafe_store!(sig_off, 0, 1)
   for (c, bb) in enumerate(r.sig)
      for b in bb
         unsafe_store!(sig, b.n, 3p + 1); unsafe_store!(sig, b.l, 3p + 2); unsafe_store!(sig, b.m, 3p + 3)
         p += 1
      end
      unsafe_store!(sig_off, p, c + 1)
   end
   for (k, (ri, yi)) in enumerate(r.aspec)
      unsafe_store!(aspec, ri - 1, 2k - 1); unsafe_store!(aspec, yi - 1, 2k)
   end
   p = 0; unsafe_store!(aa_off, 0, 1)
   for (c, s) in enumerate(r.aa)
      for i in s
         p += 1; unsafe_store!(aa_idx, i - 1, p)
      end
      unsafe_store!(aa_off, p, c + 1)
   end
   p = 0; unsafe_store!(nnll_off, 0, 1)
   for (i, bb) in enumerate(r.nnll)
      for b in bb
         unsafe_store!(nnll, b.n, 2p + 1); unsafe_store!(nnll, b.l, 2p + 2)
         p += 1
      end
      unsafe_store!(nnll_off, p, i + 1)
   end
   return OK
end

end # module
```

- [ ] **Step 4: Run tests**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=coupling/julia coupling/julia/test/runtests.jl`
Expected: PASS, three testsets. (If `r.aa == …` fails, compare `t.aabasis.specs` construction in the fork's `src/ace/sparsesymmprod.jl` — the shim must mirror `SparseSymmProd(spec)` exactly; fix the shim, not the test.)

- [ ] **Step 5: Commit**

```bash
git add coupling/julia/Project.toml coupling/julia/Manifest.toml coupling/julia/src coupling/julia/test
git commit -m "feat(coupling): ETCouple C-ABI shim over ET's trim-safe construction API"
```

---

### Task 6: `build.jl` — compile, bundle, privatize, codesign, build_info

**Files:**
- Create: `coupling/julia/build/Project.toml`, `coupling/julia/build/Manifest.toml`, `coupling/julia/build.jl`, `coupling/tools/check_bundle.py`
- Modify: `.gitignore` (add `coupling/build/`)

**Interfaces:**
- Consumes: Task 5 shim.
- Produces: `julia +1.13 --project=coupling/julia/build coupling/julia/build.jl <out>` creates `<out>/bundle/` with `lib/libetcouple.{dylib,so}`, `lib/julia/…`, `share/…`, and `<out>/bundle/build_info.json` = `{"abi": 1, "max_order": 8, "et_repo": "...", "et_rev": "<ET_REV>", "julia": "1.13.x", "juliac": "0.3.10", "platform": "<os>-<arch>", "cpu_target": "generic"}`. `python coupling/tools/check_bundle.py <out>/bundle` exits 0 iff the bundle loads, ABI == 1, and the platform floor holds.

- [ ] **Step 1: Write `check_bundle.py` (the test)**

```python
"""Check a libetcouple bundle: loads, ABI 1, max order 8, and the platform floor
(macOS: every Mach-O minos <= 11.0; Linux: no GLIBC_ symbol newer than 2.28).
    python coupling/tools/check_bundle.py coupling/build/bundle"""
import ctypes
import json
import pathlib
import platform
import re
import subprocess
import sys

MACOS_MAX = (11, 0)
GLIBC_MAX = (2, 28)


def _libs(root):
    pats = ("*.dylib",) if sys.platform == "darwin" else ("*.so", "*.so.*")
    return sorted({p for pat in pats for p in root.rglob(pat) if p.is_file() and not p.is_symlink()})


def check(root):
    root = pathlib.Path(root).resolve()
    info = json.loads((root / "build_info.json").read_text())
    ext = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(str(root / "lib" / f"libetcouple.{ext}"), mode=ctypes.RTLD_LOCAL)
    assert lib.etc_abi_version() == 1 == info["abi"], "ABI mismatch"
    assert lib.etc_max_order() == 8 == info["max_order"], "max order mismatch"
    bad = []
    for f in _libs(root):
        if sys.platform == "darwin":
            out = subprocess.run(["otool", "-l", str(f)], capture_output=True, text=True).stdout
            for v in re.findall(r"minos (\d+)\.(\d+)", out):
                if tuple(map(int, v)) > MACOS_MAX:
                    bad.append(f"{f.relative_to(root)} minos {'.'.join(v)}")
        else:
            out = subprocess.run(["objdump", "-T", str(f)], capture_output=True, text=True).stdout
            for v in set(re.findall(r"GLIBC_(\d+)\.(\d+)", out)):
                if tuple(map(int, v)) > GLIBC_MAX:
                    bad.append(f"{f.relative_to(root)} GLIBC_{'.'.join(v)}")
    assert not bad, "platform floor violated:\n  " + "\n  ".join(bad)
    print(f"bundle OK: {info['platform']} et_rev={info['et_rev'][:12]} libs={len(_libs(root))} {platform.machine()}")


if __name__ == "__main__":
    check(sys.argv[1])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python coupling/tools/check_bundle.py coupling/build/bundle`
Expected: FAIL — `FileNotFoundError: .../coupling/build/bundle/build_info.json`.

- [ ] **Step 3: Build env and script**

`coupling/julia/build/Project.toml`:
```toml
[deps]
JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[compat]
JuliaC = "=0.3.10"
julia = "1.13"
```
`coupling/julia/build.jl`:
```julia
# Build the ace-jax-coupling native library.
#   julia +1.13 --project=coupling/julia/build coupling/julia/build.jl coupling/build
# Produces <out>/bundle/{lib/libetcouple.<ext>, lib/julia/..., share/..., build_info.json}.
VERSION >= v"1.13" || error("build needs Julia >= 1.13 (got $VERSION)")
using JuliaC, TOML

const HERE = @__DIR__
const LIBPROJ = HERE                                   # coupling/julia (the shim project)
const ENTRY = joinpath(HERE, "src", "ETCouple.jl")
out = abspath(length(ARGS) >= 1 ? ARGS[1] : joinpath(HERE, "..", "build"))
bundle = joinpath(out, "bundle")
rm(bundle; force = true, recursive = true); mkpath(out)

ENV["JULIA_CPU_TARGET"] = "generic"
Sys.isapple() && (ENV["MACOSX_DEPLOYMENT_TARGET"] = "11.0")

run(`$(Base.julia_cmd()) --project=$LIBPROJ -e "using Pkg; Pkg.instantiate()"`)
JuliaC.main(["--output-lib", joinpath(out, "libetcouple"), "--project=$LIBPROJ",
             "--trim=safe", "--compile-ccallable", "--experimental",
             "--jl-option", "handle-signals=no", "--privatize",
             "--bundle", bundle, ENTRY])

ext = Sys.isapple() ? "dylib" : "so"
lib = joinpath(bundle, "lib", "libetcouple.$ext")
isfile(lib) || error("expected $lib; bundle/lib has $(readdir(joinpath(bundle, "lib")))")

# privatize/bundling rewrites load commands, which invalidates macOS signatures;
# arm64 macOS refuses to load unsigned modified code, so re-sign ad hoc.
if Sys.isapple()
   for (root, _, files) in walkdir(bundle), f in files
      p = joinpath(root, f)
      (!islink(p) && endswith(f, ".dylib")) && run(`codesign --force --sign - $p`)
   end
end

src = TOML.parsefile(joinpath(LIBPROJ, "Project.toml"))["sources"]["EquivariantTensors"]
info = Dict("abi" => 1, "max_order" => 8, "et_repo" => src["url"], "et_rev" => src["rev"],
            "julia" => string(VERSION), "juliac" => string(pkgversion(JuliaC)),
            "platform" => "$(Sys.KERNEL)-$(Sys.ARCH)", "cpu_target" => "generic")
open(joinpath(bundle, "build_info.json"), "w") do io
   print(io, "{", join(["\"$k\": " * (v isa Number ? string(v) : "\"$v\"") for (k, v) in sort(collect(info))], ", "), "}")
end
println("built $lib")
```
Add `coupling/build/` to `.gitignore`.

- [ ] **Step 4: Build and check**

Run: `JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=coupling/julia/build -e 'using Pkg; Pkg.instantiate()' && JULIA_DEPOT_PATH=$HOME/.julia-trim julia +1.13 --project=coupling/julia/build coupling/julia/build.jl coupling/build 2>&1 | tail -5`
Expected: `built .../coupling/build/bundle/lib/libetcouple.dylib`, no `Verifier error`.
Run: `uv run python coupling/tools/check_bundle.py coupling/build/bundle`
Expected: `bundle OK: Darwin-aarch64 et_rev=<12 chars> libs=<N> arm64`.
If minos > 11.0 is reported for `libetcouple.dylib` despite `MACOSX_DEPLOYMENT_TARGET`: pass the flag to the link step explicitly — build `LinkRecipe(...; ld_flags = ["-mmacosx-version-min=11.0"])` via the recipe API (`ImageRecipe`/`LinkRecipe`/`BundleRecipe` fields shown in JuliaC's `src/JuliaC.jl`) instead of `JuliaC.main`, and re-run.

- [ ] **Step 5: Commit**

```bash
git add coupling/julia/build coupling/julia/build.jl coupling/tools/check_bundle.py .gitignore
git commit -m "build(coupling): juliac --trim bundle (generic CPU, handle-signals=no, privatized, macOS 11 floor)"
```

---

## Phase C — Python package `ace-jax-coupling`

### Task 7: `ace_jax_coupling` loader + API + tests (against the local bundle)

**Files:**
- Create: `coupling/python/src/ace_jax_coupling/__init__.py`, `_loader.py`, `_api.py`
- Create: `coupling/python/tests/conftest.py`, `coupling/python/tests/test_api.py`, `coupling/python/tests/test_runtime.py`

**Interfaces:**
- Consumes: bundle from Task 6 (`ACEJAX_COUPLING_LIB=<bundle>/lib/libetcouple.<ext>` overrides the packaged location); `tests/data/{cases.json, et_reference.npz}` (Task 4).
- Produces (public, `ace_jax_coupling`):
  - `__version__ = "0.1.0"`, `ABI_VERSION = 1`, `MAX_ORDER = 8`
  - `class CouplingLibError(RuntimeError)`
  - `build_info() -> dict` — reads `build_info.json`, **never loads the library**
  - `couple_raw(mb_spec, Rnl_spec, Ylm_spec) -> RawCoupling` where `RawCoupling` is a `NamedTuple` with fields `A2B_shape: tuple[int,int]`, `A2B_rows, A2B_cols: int64[nnz]`, `A2B_vals: float64[nnz]`, `aa_sig_off: int64[n_AA+1]`, `aa_sig: int64[n_sig,3]`, `aspec: int64[n_A,2]`, `aa_off: int64[n_AA+1]`, `aa_idx: int64[n_aaidx]`, `nnll_off: int64[n_B+1]`, `nnll: int64[n_nnll,2]` (all 0-based). Raises `ValueError` for invalid specs (before touching the library), `CouplingLibError` if the library is missing/incompatible.
  - The library is loaded lazily on the first `couple_raw` call, with `RTLD_NOW | RTLD_LOCAL`; calls are serialised by a module lock.

- [ ] **Step 1: Write the failing tests**

`coupling/python/tests/conftest.py`:
```python
import json
import pathlib

import numpy as np
import pytest

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cases():
    return json.loads((DATA / "cases.json").read_text())


@pytest.fixture(scope="session")
def reference():
    return np.load(DATA / "et_reference.npz")
```
`coupling/python/tests/test_api.py`:
```python
"""ace_jax_coupling against unpatched upstream EquivariantTensors (BASE_SHA)."""
import numpy as np
import pytest

import ace_jax_coupling as ajc

FIELDS = ["A2B_rows", "A2B_cols", "A2B_vals", "aa_sig_off", "aa_sig", "aspec",
          "aa_off", "aa_idx", "nnll_off", "nnll"]


def test_bit_exact_vs_upstream_et(cases, reference):
    for name, c in cases.items():
        r = ajc.couple_raw(c["mb"], c["R"], c["Y"])
        assert r.A2B_shape == tuple(int(v) for v in reference[f"{name}__A2B_shape"]), name
        for f in FIELDS:
            got, want = getattr(r, f), reference[f"{name}__{f}"]
            assert got.dtype == want.dtype and got.shape == want.shape, (name, f, got.shape, want.shape)
            assert np.array_equal(got, want), (name, f)


def test_accepts_tuples_lists_and_numpy_ints(cases):
    c = cases["tiny"]
    a = ajc.couple_raw(c["mb"], c["R"], c["Y"])
    mb = [tuple((np.int32(n), np.int64(l)) for n, l in bb) for bb in c["mb"]]
    b = ajc.couple_raw(mb, np.array(c["R"]), [tuple(y) for y in c["Y"]])
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in FIELDS)


def test_deterministic(cases):
    c = cases["SiGe_o4d5w05"]
    a, b = ajc.couple_raw(c["mb"], c["R"], c["Y"]), ajc.couple_raw(c["mb"], c["R"], c["Y"])
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in FIELDS)


@pytest.mark.parametrize("mb,R,Y,msg", [
    ([], [(1, 0)], [(0, 0)], "empty"),
    ([[]], [(1, 0)], [(0, 0)], "empty body"),
    ([[(2, 0)]], [(1, 0)], [(0, 0)], "not in Rnl_spec"),
    ([[(1, 1), (1, 1)]], [(1, 1)], [(1, 0)], "not in Ylm_spec"),
    ([[(1, 0)] * 9], [(1, 0)], [(0, 0)], "order"),
    ([[(1, -1)]], [(1, -1)], [(0, 0)], "negative"),
    ([[(1, 0)]], [(1, 0)], [(1, 2)], r"\|m\|"),
    ([[(1, 0)]], [(1, 0), (1, 0)], [(0, 0)], "duplicate"),
    ([[(1.5, 0)]], [(1, 0)], [(0, 0)], "integer"),
])
def test_invalid_inputs_raise(mb, R, Y, msg):
    with pytest.raises(ValueError, match=msg):
        ajc.couple_raw(mb, R, Y)


def test_raw_invalid_returns_code():
    """A malformed direct ctypes call is rejected by the library (code 2), not a crash."""
    from ace_jax_coupling import _loader
    lib = _loader.lib()
    z = np.zeros(8, np.int64)
    off = np.array([0, 2, 1], np.int64)                      # non-monotone offsets (in bounds)
    p = lambda a: a.ctypes.data_as(_loader.c_void_p)
    code = lib.etc_couple(2, p(off), p(z), p(z), 1, p(z), p(z), 1, p(z), p(z), p(z),
                          *([None] * 10))
    assert code == 2


def test_build_info_does_not_load_library(monkeypatch):
    from ace_jax_coupling import _loader
    monkeypatch.setattr(_loader, "_LIB", None)
    info = ajc.build_info()
    assert info["abi"] == ajc.ABI_VERSION and info["max_order"] == ajc.MAX_ORDER
    assert len(info["et_rev"]) == 40
    assert _loader._LIB is None
```
`coupling/python/tests/test_runtime.py`:
```python
"""Process-level behaviour of the embedded Julia runtime."""
import concurrent.futures as cf
import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np

import ace_jax_coupling as ajc


def test_import_does_not_load_library():
    code = "import sys, ace_jax_coupling; from ace_jax_coupling import _loader; sys.exit(0 if _loader._LIB is None else 1)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_threads_concurrent_calls(cases):
    c = cases["CrMnFe_o3d6"]
    ref = ajc.couple_raw(c["mb"], c["R"], c["Y"])
    with cf.ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: ajc.couple_raw(c["mb"], c["R"], c["Y"]), range(8)))
    assert all(np.array_equal(o.A2B_vals, ref.A2B_vals) and np.array_equal(o.aa_idx, ref.aa_idx) for o in outs)


def test_sigint_still_raises_keyboardinterrupt(cases, tmp_path):
    """Loading the library must not take over SIGINT (built with handle-signals=no)."""
    c = cases["tiny"]
    script = textwrap.dedent(f"""
        import sys, time, ace_jax_coupling as ajc
        ajc.couple_raw({c['mb']!r}, {c['R']!r}, {c['Y']!r})
        print("ready", flush=True)
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("kbi", flush=True); sys.exit(7)
    """)
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=os.environ)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.2)
    p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 7 and "kbi" in out
```
Run: `ACEJAX_COUPLING_LIB=$PWD/coupling/build/bundle/lib/libetcouple.dylib PYTHONPATH=coupling/python/src uv run --no-project --with numpy --with pytest python -m pytest coupling/python/tests -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ace_jax_coupling'`.

- [ ] **Step 2: Implement `_loader.py`**

```python
"""Locate, load and type the libetcouple shared library (lazily)."""
import ctypes
import json
import os
import pathlib
import sys
import threading
from ctypes import c_int32, c_int64, c_void_p

ABI_VERSION = 1
_EXT = "dylib" if sys.platform == "darwin" else "so"
_PKG = pathlib.Path(__file__).resolve().parent
_LIB = None
LOCK = threading.Lock()


class CouplingLibError(RuntimeError):
    """The native coupling library is missing, unloadable or incompatible."""


def lib_path() -> pathlib.Path:
    env = os.environ.get("ACEJAX_COUPLING_LIB")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return _PKG / "_lib" / "lib" / f"libetcouple.{_EXT}"


def bundle_root() -> pathlib.Path:
    return lib_path().parent.parent


def build_info() -> dict:
    """The bundle's build_info.json. Never loads the library."""
    p = bundle_root() / "build_info.json"
    if not p.is_file():
        raise CouplingLibError(
            f"no coupling library build found ({p} missing). This is a source/development "
            "build of ace-jax-coupling without the compiled bundle: install a platform wheel, "
            "or set ACEJAX_COUPLING_LIB to <bundle>/lib/libetcouple." + _EXT)
    return json.loads(p.read_text())


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with LOCK:
        if _LIB is not None:
            return _LIB
        build_info()                                   # clear error if the bundle is absent
        p = lib_path()
        try:
            h = ctypes.CDLL(str(p), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        except OSError as e:
            raise CouplingLibError(f"cannot load {p}: {e}") from e
        h.etc_abi_version.restype = c_int32
        h.etc_max_order.restype = c_int32
        if h.etc_abi_version() != ABI_VERSION:
            raise CouplingLibError(f"{p}: ABI {h.etc_abi_version()} != expected {ABI_VERSION}")
        h.etc_couple.restype = c_int32
        h.etc_couple.argtypes = ([c_int64, c_void_p, c_void_p, c_void_p,
                                  c_int64, c_void_p, c_void_p,
                                  c_int64, c_void_p, c_void_p,
                                  c_void_p] + [c_void_p] * 10)
        _LIB = h
        return _LIB
```

- [ ] **Step 3: Implement `_api.py`**

```python
"""couple_raw: validated, two-phase call into etc_couple."""
import numbers
from typing import NamedTuple

import numpy as np

from . import _loader

MAX_ORDER = 8
_OK, _SHORT, _INVALID, _ORDER = 0, 1, 2, 3


class RawCoupling(NamedTuple):
    A2B_shape: tuple
    A2B_rows: np.ndarray
    A2B_cols: np.ndarray
    A2B_vals: np.ndarray
    aa_sig_off: np.ndarray
    aa_sig: np.ndarray
    aspec: np.ndarray
    aa_off: np.ndarray
    aa_idx: np.ndarray
    nnll_off: np.ndarray
    nnll: np.ndarray


def _int(v, what):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise ValueError(f"{what}: expected an integer, got {v!r}")
    return int(v)


def _pairs(xs, what):
    out = []
    for x in xs:
        x = tuple(x)
        if len(x) != 2:
            raise ValueError(f"{what}: expected pairs, got {x!r}")
        out.append((_int(x[0], what), _int(x[1], what)))
    return out


def _validate(mb, R, Y):
    if len(mb) == 0:
        raise ValueError("mb_spec is empty")
    if len(set(R)) != len(R):
        raise ValueError("Rnl_spec has duplicate (n, l) entries")
    if len(set(Y)) != len(Y):
        raise ValueError("Ylm_spec has duplicate (l, m) entries")
    for n, l in R:
        if l < 0:
            raise ValueError(f"Rnl_spec: negative l in {(n, l)}")
    for l, m in Y:
        if l < 0 or abs(m) > l:
            raise ValueError(f"Ylm_spec: need l >= 0 and |m| <= l, got {(l, m)}")
    rs, ys = set(R), set(Y)
    for i, bb in enumerate(mb):
        if len(bb) == 0:
            raise ValueError(f"mb_spec[{i}] is an empty body")
        if len(bb) > MAX_ORDER:
            raise ValueError(f"mb_spec[{i}]: correlation order {len(bb)} > {MAX_ORDER}")
        for n, l in bb:
            if l < 0:
                raise ValueError(f"mb_spec[{i}]: negative l in {(n, l)}")
            if (n, l) not in rs:
                raise ValueError(f"mb_spec[{i}]: {(n, l)} not in Rnl_spec")
            for m in range(-l, l + 1):
                if (l, m) not in ys:
                    raise ValueError(f"mb_spec[{i}]: {(l, m)} not in Ylm_spec")


def couple_raw(mb_spec, Rnl_spec, Ylm_spec) -> RawCoupling:
    mb = [_pairs(bb, "mb_spec") for bb in mb_spec]
    R, Y = _pairs(Rnl_spec, "Rnl_spec"), _pairs(Ylm_spec, "Ylm_spec")
    _validate(mb, R, Y)
    i64 = lambda xs: np.ascontiguousarray(np.asarray(xs, np.int64))
    off = i64(np.cumsum([0] + [len(bb) for bb in mb]))
    flat = [b for bb in mb for b in bb]
    mn, ml = i64([n for n, _ in flat]), i64([l for _, l in flat])
    rn, rl = i64([n for n, _ in R]), i64([l for _, l in R])
    yl, ym = i64([l for l, _ in Y]), i64([m for _, m in Y])
    p = lambda a: a.ctypes.data
    h = _loader.lib()
    sizes = np.zeros(7, np.int64)

    def call(bufs):
        return h.etc_couple(len(mb), p(off), p(mn), p(ml), len(R), p(rn), p(rl), len(Y), p(yl), p(ym),
                            p(sizes), *[p(b) if b is not None else None for b in bufs])

    with _loader.LOCK:
        code = call([None] * 10)
        if code not in (_OK, _SHORT):
            raise _error(code)
        nnz, nB, nAA, nsig, nA, naa, nnl = (int(v) for v in sizes)
        bufs = [np.zeros(nnz, np.int64), np.zeros(nnz, np.int64), np.zeros(nnz, np.float64),
                np.zeros(nAA + 1, np.int64), np.zeros(3 * nsig, np.int64), np.zeros(2 * nA, np.int64),
                np.zeros(nAA + 1, np.int64), np.zeros(naa, np.int64),
                np.zeros(nB + 1, np.int64), np.zeros(2 * nnl, np.int64)]
        code = call(bufs)
        if code != _OK:
            raise _error(code)
    return RawCoupling((nB, nAA), bufs[0], bufs[1], bufs[2], bufs[3], bufs[4].reshape(-1, 3),
                       bufs[5].reshape(-1, 2), bufs[6], bufs[7], bufs[8], bufs[9].reshape(-1, 2))


def _error(code):
    if code == _INVALID:
        return ValueError("coupling library rejected the specs (INVALID_INPUT)")
    if code == _ORDER:
        return ValueError(f"correlation order > {MAX_ORDER} (ORDER_TOO_HIGH)")
    return _loader.CouplingLibError(f"etc_couple returned unexpected code {code}")
```
Note: `_validate` runs before the library is touched, so the Julia checks are only a backstop; `test_invalid_inputs_raise` pins the Python messages (`"empty"`, `"empty body"`, `"not in Rnl_spec"`, `"not in Ylm_spec"`, `"order"`, `"negative"`, `"|m|"`, `"duplicate"`, `"integer"`).

- [ ] **Step 4: Implement `__init__.py`**

```python
"""EquivariantTensors.jl SO(3) coupling tables for ace-jax, from a
juliac-compiled library (no Julia needed at runtime)."""
from ._api import MAX_ORDER, RawCoupling, couple_raw
from ._loader import ABI_VERSION, CouplingLibError, build_info

__version__ = "0.1.0"
__all__ = ["ABI_VERSION", "MAX_ORDER", "CouplingLibError", "RawCoupling", "build_info", "couple_raw", "__version__"]
```

- [ ] **Step 5: Run tests**

Run: `ACEJAX_COUPLING_LIB=$PWD/coupling/build/bundle/lib/libetcouple.dylib PYTHONPATH=coupling/python/src uv run --no-project --with numpy --with pytest python -m pytest coupling/python/tests -v`
Expected: PASS (all of `test_api.py`, `test_runtime.py`). If `test_sigint_still_raises_keyboardinterrupt` fails, confirm `handle-signals=no` reached the image (`strings <lib> | grep -i handle` is not reliable — instead rebuild with `--jl-option handle-signals=no` placed before `--output-lib` and re-test); do not weaken the test.

- [ ] **Step 6: Commit**

```bash
git add coupling/python/src coupling/python/tests/conftest.py coupling/python/tests/test_api.py coupling/python/tests/test_runtime.py
git commit -m "feat(ace-jax-coupling): ctypes wrapper, validation, bit-exact tests vs upstream ET"
```

---

### Task 8: Bundle pruning, wheel build hook, clean-environment wheel test

**Files:**
- Create: `coupling/tools/prune_bundle.py`, `coupling/python/pyproject.toml`, `coupling/python/hatch_build.py`, `coupling/python/LICENSE`, `coupling/python/THIRD_PARTY_NOTICES.md`, `coupling/python/README.md`, `coupling/tools/test_wheel.sh`

**Interfaces:**
- Consumes: Task 6 bundle, Task 7 package + tests.
- Produces: `python coupling/tools/prune_bundle.py <bundle> <pruned>` → minimal relocatable bundle; `ACEJAX_COUPLING_BUNDLE=<pruned> ACEJAX_COUPLING_PLAT=<tag> uv build --wheel coupling/python -o dist` → `ace_jax_coupling-0.1.0-py3-none-<tag>.whl`; without `ACEJAX_COUPLING_BUNDLE` the same command builds a lib-less `py3-none-any` wheel (development stub; `build_info()` raises `CouplingLibError`). `bash coupling/tools/test_wheel.sh dist/<wheel>` exits 0 iff the wheel passes the package tests in a clean env.

- [ ] **Step 1: Write the clean-env wheel test (it fails: no wheel yet)**

`coupling/tools/test_wheel.sh`:
```bash
#!/usr/bin/env bash
# Install a built wheel into a fresh venv with an EMPTY HOME and no Julia on
# PATH, run the package tests, and check nothing was written to HOME.
#   bash coupling/tools/test_wheel.sh dist/ace_jax_coupling-0.1.0-py3-none-<tag>.whl [python]
set -euo pipefail
WHEEL=$(realpath "$1"); PY=${2:-python3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mktemp -d); export HOME="$WORK/home"; mkdir -p "$HOME"
unset JULIA_DEPOT_PATH JULIA_PROJECT ACEJAX_COUPLING_LIB
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -i julia | paste -sd: -)
if command -v julia >/dev/null; then echo "julia still on PATH: $(command -v julia)"; exit 1; fi
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q "$WHEEL" pytest
cp -r "$REPO/coupling/python/tests" "$WORK/tests"          # run from outside the source tree
BEFORE=$(find "$HOME" | sort)                              # after pip: only the library's writes count
(cd "$WORK" && "$WORK/venv/bin/python" -m pytest -q tests -p no:cacheprovider)
AFTER=$(find "$HOME" | sort)
if [ "$BEFORE" != "$AFTER" ]; then echo "library wrote to HOME:"; diff <(echo "$BEFORE") <(echo "$AFTER"); exit 1; fi
du -h "$WHEEL"; echo "wheel OK"
```
Run: `bash coupling/tools/test_wheel.sh dist/nonexistent.whl`
Expected: FAIL (`realpath: dist/nonexistent.whl: No such file or directory`).

- [ ] **Step 2: Trace-based pruning**

`coupling/tools/prune_bundle.py`:
```python
"""Copy a libetcouple bundle keeping only the shared libraries the runtime
actually loads (traced while running every case in cases.json) plus all
non-library files.  Symlinked names that were loaded become regular files
(wheels cannot carry symlinks).
    python coupling/tools/prune_bundle.py coupling/build/bundle coupling/build/pruned"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EXT = "dylib" if sys.platform == "darwin" else "so"
DRIVER = """
import json, sys
sys.path.insert(0, sys.argv[1])
import ace_jax_coupling as ajc
for c in json.load(open(sys.argv[2])).values():
    ajc.couple_raw(c["mb"], c["R"], c["Y"])
print("traced-ok")
"""


def _is_lib(p):
    n = p.name
    return n.endswith(".dylib") if sys.platform == "darwin" else (n.endswith(".so") or ".so." in n)


def traced(src):
    env = dict(os.environ, ACEJAX_COUPLING_LIB=str(src / "lib" / f"libetcouple.{EXT}"))
    env["DYLD_PRINT_LIBRARIES" if sys.platform == "darwin" else "LD_DEBUG"] = "1" if sys.platform == "darwin" else "files"
    r = subprocess.run([sys.executable, "-c", DRIVER, str(REPO / "coupling/python/src"),
                        str(REPO / "coupling/python/tests/data/cases.json")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "traced-ok" in r.stdout, r.stderr[-3000:]
    root = str(src) + os.sep
    used = set()
    for line in r.stderr.splitlines():
        for tok in line.replace("=", " ").split():
            if tok.startswith(root):
                used.add(pathlib.Path(tok).relative_to(src))
    return used


def main(src, dst):
    src, dst = pathlib.Path(src).resolve(), pathlib.Path(dst).resolve()
    used = traced(src)
    assert pathlib.Path("lib") / f"libetcouple.{EXT}" in used, "trace did not see libetcouple"
    shutil.rmtree(dst, ignore_errors=True)
    kept = dropped = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir():
            continue
        if _is_lib(p) and rel not in used:
            dropped += 1
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p.resolve(), out)                  # dereference symlinks
        kept += 1
    print(f"kept {kept} files ({len(used)} traced libs), dropped {dropped} libs -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```
Run: `uv run --no-project --with numpy python coupling/tools/prune_bundle.py coupling/build/bundle coupling/build/pruned && uv run python coupling/tools/check_bundle.py coupling/build/pruned`
Expected: `kept … (≈31 traced libs), dropped …` then `bundle OK`. Then run the Task 7 tests against the pruned tree: `ACEJAX_COUPLING_LIB=$PWD/coupling/build/pruned/lib/libetcouple.dylib PYTHONPATH=coupling/python/src uv run --no-project --with numpy --with pytest python -m pytest coupling/python/tests -q` → PASS.
(Linux `LD_DEBUG=files` prints `file=<path> [0]; ...`; the `replace("=", " ")` split handles it. On macOS, `uv run` strips `DYLD_*` — the script sets it on its own child process, so this works.)

- [ ] **Step 3: Packaging files**

`coupling/python/pyproject.toml`:
```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "ace-jax-coupling"
version = "0.1.0"
description = "EquivariantTensors.jl SO(3) coupling tables for ace-jax, as a juliac-compiled library (no Julia needed)"
readme = "README.md"
requires-python = ">=3.11"
license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]
authors = [{ name = "ACEsuit contributors" }]
dependencies = ["numpy>=1.24"]

[project.urls]
Homepage = "https://github.com/ACEsuit/ace-jax"

[tool.hatch.build.targets.wheel]
packages = ["src/ace_jax_coupling"]

[tool.hatch.build.targets.wheel.hooks.custom]
path = "hatch_build.py"

[tool.hatch.build.targets.sdist]
include = ["src", "hatch_build.py", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"]
```
(No `license =` expression yet: the declaration is a user decision — see Global Constraints. `LICENSE` is ace-jax's MIT text for the Python/Julia glue.)
`coupling/python/hatch_build.py`:
```python
"""Wheel hook: embed the compiled bundle and tag the wheel for its platform.

ACEJAX_COUPLING_BUNDLE  path to a (pruned) bundle dir with build_info.json
ACEJAX_COUPLING_PLAT    wheel platform tag, e.g. manylinux_2_28_x86_64, macosx_11_0_arm64
Without ACEJAX_COUPLING_BUNDLE a lib-less py3-none-any development wheel is built."""
import os
import pathlib
import warnings

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        bundle = os.environ.get("ACEJAX_COUPLING_BUNDLE")
        if not bundle:
            warnings.warn("ACEJAX_COUPLING_BUNDLE unset: building a lib-less development wheel")
            return
        root = pathlib.Path(bundle).resolve()
        if not (root / "build_info.json").is_file():
            raise RuntimeError(f"{root} is not a coupling bundle (no build_info.json)")
        plat = os.environ.get("ACEJAX_COUPLING_PLAT")
        if not plat:
            raise RuntimeError("ACEJAX_COUPLING_PLAT must be set with ACEJAX_COUPLING_BUNDLE")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{plat}"
        for p in root.rglob("*"):
            if p.is_file():
                build_data["force_include"][str(p)] = f"ace_jax_coupling/_lib/{p.relative_to(root)}"
```
`coupling/python/README.md`: 10–20 lines: what the package is, that it contains a trimmed Julia runtime + EquivariantTensors (fork rev in `build_info()`), supported platforms, `ACEJAX_COUPLING_LIB` override, that ace-jax uses it through `ace-jax[authoring]`.
`coupling/python/LICENSE`: copy of the repo root `LICENSE`.
`coupling/python/THIRD_PARTY_NOTICES.md`: one section per bundled component, generated from the pruned bundle's file list (`find coupling/build/pruned -type f -name '*.dylib' -o -name '*.so*'`). For each: component name, upstream URL, SPDX licence as stated by its upstream (check each upstream's own licence file — do not guess), and the licence text. Components to expect: Julia runtime (MIT), EquivariantTensors.jl (MIT), OpenBLAS (BSD-3-Clause), libblastrampoline (MIT), SuiteSparse AMD/CAMD/COLAMD/CCOLAMD/BTF/LDL/SuiteSparse_config (BSD-3-Clause), KLU (LGPL-2.1-or-later), UMFPACK/CHOLMOD/SPQR/RBio (verify: UMFPACK and parts of CHOLMOD are believed GPL-2.0-or-later), GCC runtime libs libgfortran/libgcc_s/libquadmath/libgomp/libatomic/libssp/libstdc++ (GPL-3.0-or-later WITH GCC-exception-3.1), openlibm (MIT/BSD), GMP/MPFR (LGPL-3.0-or-later), libunwind (MIT), zlib (Zlib), zstd (BSD-3-Clause), PCRE2 (BSD-3-Clause). Julia's licence texts for these live under the Julia install's `share/licenses` (or each `*_jll` artifact's `share/licenses/<Name>/`) — copy from there. Record in the task report the list of components whose licence is GPL (not LGPL), for the user's decision.

- [ ] **Step 4: Build the wheel and run the clean-env test**

Run:
```bash
rm -rf dist && ACEJAX_COUPLING_BUNDLE=$PWD/coupling/build/pruned ACEJAX_COUPLING_PLAT=macosx_11_0_arm64 uv build --wheel coupling/python -o dist
unzip -l dist/*.whl | tail -3
bash coupling/tools/test_wheel.sh dist/ace_jax_coupling-0.1.0-py3-none-macosx_11_0_arm64.whl
```
Expected: one wheel `ace_jax_coupling-0.1.0-py3-none-macosx_11_0_arm64.whl`, ≈16 MB (probe: 16.4 MB); `test_wheel.sh` prints the test summary (all pass), `wheel OK`, and no "library wrote to HOME".
Also: `uv build --wheel coupling/python -o /tmp/devwheel` (no env) → `ace_jax_coupling-0.1.0-py3-none-any.whl` with a warning; `python -c "import ace_jax_coupling as a; a.build_info()"` from it raises `CouplingLibError`.

- [ ] **Step 5: Commit**

```bash
git add coupling/tools/prune_bundle.py coupling/tools/test_wheel.sh coupling/python/pyproject.toml coupling/python/hatch_build.py coupling/python/LICENSE coupling/python/THIRD_PARTY_NOTICES.md coupling/python/README.md
git commit -m "build(ace-jax-coupling): trace-pruned bundle, platform wheel hook, clean-env wheel test"
```

---

## Phase D — ace-jax integration

### Task 9: `construct/coupling.py` on `ace_jax_coupling`; drop juliacall/juliapkg

**Files:**
- Modify: `src/ace_jax/construct/coupling.py`
- Modify: `pyproject.toml` (authoring extra, `[tool.uv.sources]`), `uv.lock`
- Delete: `juliapkg.json`
- Modify: `tests/test_coupling_cache.py`

**Interfaces:**
- Consumes: `ace_jax_coupling.couple_raw`, `ace_jax_coupling.__version__`.
- Produces: `couple(mb_spec, Rnl_spec, Ylm_spec) -> Coupling` (same NamedTuple and field semantics as before); `COUPLING_LIB_VERSION = "0.1.0"`; `backend_id() -> str` (`"ace-jax-coupling==0.1.0"`, computed without importing the lib); cache schema `_CACHE_SCHEMA = 2`, entry meta key `"backend"` replacing `"juliapkg_hash"`; `juliapkg_hash` and `_jl` removed. `couple_cached`, `coupling_key`, `default_cache_dir`, `_entry_path`, `_write_entry(path, cpl, key, mb, rnl, ylm)`, `_read_entry(path, key)` keep their signatures.

- [ ] **Step 1: Update the cache tests first**

In `tests/test_coupling_cache.py`:
- Rename `test_pin_hash_change_recomputes` → `test_backend_change_recomputes`; replace `monkeypatch.setattr(C, "juliapkg_hash", lambda: "changed-pin")` with `monkeypatch.setattr(C, "backend_id", lambda: "ace-jax-coupling==9.9.9")`; update its docstring ("a coupling-library version change invalidates entries").
- Replace `test_juliapkg_hash_finds_repo_pin` with:
```python
def test_backend_id_matches_extra_pin():
    """The cache stamp, the version check on a miss, and the `authoring` extra
    pin must name the same ace-jax-coupling version."""
    import re
    import tomllib
    pp = tomllib.loads((pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    extra = " ".join(pp["project"]["optional-dependencies"]["authoring"])
    m = re.search(r"ace-jax-coupling==([0-9][^;\s]*)", extra)
    assert m and m.group(1) == C.COUPLING_LIB_VERSION
    assert C.backend_id() == f"ace-jax-coupling=={C.COUPLING_LIB_VERSION}"


def test_schema1_entry_is_a_miss(tmp_path, shim):
    """A pre-migration (juliapkg-stamped, schema 1) entry is recomputed, not an error."""
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    key = C.coupling_key(_MB, _RNL, _YLM)
    p = C._entry_path(tmp_path, key)
    z = dict(np.load(p))
    meta = json.loads(bytes(z["meta_json"]).decode())
    meta["schema"] = 1; meta.pop("backend"); meta["juliapkg_hash"] = "0" * 64
    z["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(p, **z)
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 2
```
- Update the module docstring: "pin-hash change" → "coupling-library version change".
Run: `uv run pytest tests/test_coupling_cache.py -q`
Expected: FAIL — `AttributeError: module 'ace_jax.construct.coupling' has no attribute 'backend_id'` (and `COUPLING_LIB_VERSION`).

- [ ] **Step 2: Rewrite the shim half of `coupling.py`**

Module docstring → describe `couple` as calling the compiled `ace-jax-coupling` library (EquivariantTensors via juliac; no Julia at runtime) and remove the juliacall/`sys.path` note. Delete `_jl()` and `juliapkg_hash()` and the `import sys`. Add, above `couple`:
```python
COUPLING_LIB_VERSION = "0.1.0"   # == the `authoring` extra pin (test_backend_id_matches_extra_pin)


def backend_id():
    """Identity of the coupling backend, stored in cache entries so a library
    change invalidates them.  Computed WITHOUT importing the library (a cache
    hit must not need it installed)."""
    return f"ace-jax-coupling=={COUPLING_LIB_VERSION}"


def _lib():
    if os.environ.get("ACEJAX_NO_JULIA"):
        raise RuntimeError("ACEJAX_NO_JULIA is set but a coupling was computed -- "
                           "the coupling cache missed where it should have hit")
    try:
        import ace_jax_coupling
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "coupling generation needs the 'authoring' extra: pip install 'ace-jax[authoring]' "
            "(Linux x86_64/aarch64, macOS arm64)") from e
    if ace_jax_coupling.__version__ != COUPLING_LIB_VERSION:
        raise RuntimeError(f"ace-jax-coupling {ace_jax_coupling.__version__} installed, this ace-jax "
                           f"needs =={COUPLING_LIB_VERSION}: pip install 'ace-jax-coupling=={COUPLING_LIB_VERSION}'")
    return ace_jax_coupling
```
Replace the body of `couple` (keep its docstring, dropping juliacall wording) with:
```python
    import numpy as np
    raw = _lib().couple_raw(mb_spec, Rnl_spec, Ylm_spec)
    A2B = np.zeros(raw.A2B_shape, np.float64)
    A2B[raw.A2B_rows, raw.A2B_cols] = raw.A2B_vals
    so = raw.aa_sig_off
    aa_sig = tuple(tuple((int(n), int(l), int(m)) for n, l, m in raw.aa_sig[so[k]:so[k + 1]])
                   for k in range(len(so) - 1))
    aspec = tuple((int(r), int(y)) for r, y in raw.aspec)
    ao = raw.aa_off
    rows = [raw.aa_idx[ao[k]:ao[k + 1]] for k in range(len(ao) - 1)]
    max_ord = max(len(r) for r in rows)
    aa_specs = tuple(np.asarray([r for r in rows if len(r) == k], np.int64).reshape(-1, k)
                     for k in range(1, max_ord + 1))
    no = raw.nnll_off
    nnll_spec = tuple(tuple((int(n), int(l)) for n, l in raw.nnll[no[i]:no[i + 1]])
                      for i in range(len(no) - 1))
    return Coupling(A2B=A2B, aa_sig=aa_sig, aspec=aspec, aa_specs=aa_specs, nnll_spec=nnll_spec)
```
Cache: `_CACHE_SCHEMA = 2`; in `_write_entry` replace `"juliapkg_hash": juliapkg_hash(),` with `"backend": backend_id(),`; in `_read_entry` replace `and meta.get("juliapkg_hash") == juliapkg_hash())` with `and meta.get("backend") == backend_id())`.

- [ ] **Step 3: Packaging**

`pyproject.toml`: replace the `authoring = [...]` line with
```toml
# Coupling tables for NEW basis shapes: a juliac-compiled EquivariantTensors (no Julia at runtime).
# Refitting / evaluating never needs it.  Platform wheels: Linux x86_64/aarch64, macOS arm64.
authoring = ["ace-jax-coupling==0.1.0; (sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')) or (sys_platform == 'darwin' and platform_machine == 'arm64')"]
```
and add
```toml
[tool.uv.sources]
# Until ace-jax-coupling is on PyPI.  A plain `uv sync --extra authoring` builds a lib-less dev
# wheel; for the real library build the bundle and reinstall:
#   ACEJAX_COUPLING_BUNDLE=$PWD/coupling/build/pruned ACEJAX_COUPLING_PLAT=<tag> \
#     uv sync --extra authoring --reinstall-package ace-jax-coupling
ace-jax-coupling = { path = "coupling/python" }
```
Delete `juliapkg.json`: `git rm juliapkg.json`.
Run: `uv lock && uv sync && uv run python -c "import ace_jax"`
Expected: lock updates (juliacall, juliapkg and their deps gone; `ace-jax-coupling` path source present); sync succeeds without any bundle.

- [ ] **Step 4: Tests**

Run: `uv run pytest tests/test_coupling_cache.py -q`
Expected: PASS.
Run: `ACEJAX_COUPLING_BUNDLE=$PWD/coupling/build/pruned ACEJAX_COUPLING_PLAT=macosx_11_0_arm64 uv sync --extra authoring --reinstall-package ace-jax-coupling && uv run python -c "import numpy as np; from ace_jax.construct.coupling import couple; from ace_jax.construct.spec import ylm_spec; c=couple([[(1,0)],[(1,1),(1,1)]],[(1,0),(1,1),(2,0)],ylm_spec(1)); z=np.load('coupling/python/tests/data/et_reference.npz'); assert c.A2B.shape == tuple(z['tiny__A2B_shape']); print('ok', c.A2B.shape, c.nnll_spec)"`
Expected: `ok (2, <n_AA>) (((1, 0),), ((1, 1), (1, 1)))`.
Run: `grep -rn "juliacall\|juliapkg" src pyproject.toml` → no output.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/construct/coupling.py pyproject.toml uv.lock tests/test_coupling_cache.py
git commit -m "feat(construct)!: coupling via ace-jax-coupling (compiled ET); drop juliacall/juliapkg"
```

---

### Task 10: Parity tests on the compiled backend

**Files:**
- Modify: `tests/test_coupling_parity.py`, `tests/test_coupling_etshim.py`, `tests/test_python_authoring.py`

**Interfaces:**
- Consumes: Task 9 `couple`; `conftest.require_optional`.
- Produces: tests that run in-process (no subprocess), skip without `ace_jax_coupling` (fail under `ACEJAX_REQUIRE_OPTIONAL=1`).

- [ ] **Step 1: Skip helper + `test_coupling_parity.py` → scale-aware, in-process**

Add to `tests/conftest.py` (next to `require_optional`):
```python
def require_coupling_lib():
    """Skip (or fail under ACEJAX_REQUIRE_OPTIONAL) unless ace_jax_coupling is
    installed WITH its compiled library (the uv path source builds a lib-less
    dev wheel when no bundle is given)."""
    ajc = require_optional("ace_jax_coupling")
    try:
        ajc.build_info()
    except ajc.CouplingLibError as e:
        if os.environ.get("ACEJAX_REQUIRE_OPTIONAL"):
            pytest.fail(f"ace_jax_coupling has no compiled library: {e}")
        pytest.skip(f"ace_jax_coupling has no compiled library: {e}")
    return ajc
```
Use `require_coupling_lib()` wherever a test needs the backend (Steps 1–3).

- Replace `pytest.importorskip("juliacall", …)` with a module-level `from conftest import FIXTURE_DIR, require_coupling_lib` and, in each test, `require_coupling_lib()`.
- Remove `_run` and the `_PARITY` subprocess string; turn its body into a function `_parity(ref_path, mode) -> dict` executed in-process (same code, `REF`/`MODE` become arguments, `print("RESULT", …)` becomes `return out`).
- Change the multiplicity-1 comparison from "equal up to sign" to "equal up to a nonzero scale": replace
```python
                    res = float(min(np.abs(Mp - Mr).max(), np.abs(Mp + Mr).max()))
```
with
```python
                    s = float((Mp * Mr).sum() / (Mr * Mr).sum())      # ET main rescales B rows vs 0.4.3
                    res = float(np.abs(Mp - s * Mr).max() / max(np.abs(Mp).max(), 1e-300))
                    scales.append(abs(s))
```
  (initialise `scales = []` next to `worst`, and add `out["scale_range"] = [min(scales), max(scales)] if scales else None`). Degenerate blocks keep `subspace_residual` (scale-invariant).
- Rewrite the module docstring's "PARITY IS ET-VERSION-SENSITIVE" paragraph: the backend is ET main (fork `ET_REV`), the fixtures are ACEpotentials 0.10.1 / ET 0.4.3; the two agree block-by-block up to a per-B-row scale (0.225–1.0 observed), which the comparison allows; bit-exactness vs ET itself is asserted in `coupling/python/tests`.
- Keep every existing assertion on `shape_ok`, `mset_ok`, uniqueness, `blocks_ok`, multiplicity match and `max_residual < 1e-12`; add `assert 0.2 < r["scale_range"][0] and r["scale_range"][1] <= 1.0 + 1e-12` in the tests.
Run: `uv run pytest tests/test_coupling_parity.py -v`
Expected: PASS for all 4 fixtures × both modes.

- [ ] **Step 2: `test_coupling_etshim.py` → in-process**

Replace the `importorskip("juliacall")` and `_BRIDGE` subprocess with a direct call inside `test_bridge_wellformed` (renamed from `test_bridge_wellformed_subprocess`), guarded by `require_coupling_lib()`; keep all assertions. `test_rpe_admissible_port` must run without the library (move the guard into the bridge test only). Update the module docstring.
Run: `uv run pytest tests/test_coupling_etshim.py -v` → PASS.

- [ ] **Step 3: `test_python_authoring.py` bridge + no-Julia tests**

In `_BRIDGE` (still a subprocess script — it writes files and is long; keep it, but import `ace_jax_coupling` instead of `juliacall` at its top):
- replace `res["a2b_exact"] = bool(np.array_equal(cpl.A2B, np.asarray(zf["A2B"], float)))` with a column alignment by (n,l,m) signature (ET main may enumerate a block's m-columns in a different order than 0.4.3) followed by a per-row scale fit:
```python
# fixture A2B columns are in its aa_spec evaluation order; identify each by its (n,l,m) body
ar, ay = zf["aspec_r"], zf["aspec_y"]
fsig = [tuple(sorted((Rnl[ar[i]][0], Rnl[ar[i]][1], Ylm[ay[i]][1]) for i in row))
        for k in range(len(fmeta["aa_lens"])) for row in zf[f"aa_spec_{k+1}"]]
col = {sg: j for j, sg in enumerate(fsig)}
perm = np.array([col[tuple(sorted(sg))] for sg in cpl.aa_sig])
res["col_perm_identity"] = bool((perm == np.arange(len(perm))).all())   # informational
A2B_f = np.asarray(zf["A2B"], float)[:, perm]                # fixture columns in cpl's column order
s = (cpl.A2B * A2B_f).sum(1) / (A2B_f * A2B_f).sum(1)        # ET main rescales B rows vs the 0.4.3 export
res["a2b_rowscale"] = float(np.abs(cpl.A2B - s[:, None] * A2B_f).max())
res["scale_range"] = [float(np.abs(s).min()), float(np.abs(s).max())]
```
- where the fixture coefficients are injected, use `WB=jnp.asarray(np.asarray(zf["WB"]) / s[:, None])` (same function: `B_new = s·B_old` ⇒ `WB_new = WB_old / s`);
- in the `res["arrays"]` comparison list, compare `("A2B", model.A2B, s[:, None] * A2B_f)` and `("WB", model.WB, np.asarray(zf["WB"]) / s[:, None])` instead of the raw fixture arrays. (Rows need no alignment: `cpl` is built from the fixture's own nnll row order `mb_ref`. The injected model is self-consistent: `m`'s `aa_specs` and `cpl.A2B` both come from the same backend.)
In `test_bridge_wellformed_subprocess`: replace the juliacall skip with `require_coupling_lib()`, replace `r["a2b_exact"]` with `r["a2b_rowscale"] < 1e-12 and 0.2 < r["scale_range"][0] <= r["scale_range"][1] <= 1.0 + 1e-12`; keep all energy/force/stress/descriptor tolerances unchanged.
In `_NOJULIA` and `test_coupling_cache_no_julia_on_hit`: `import juliacall` → `import ace_jax_coupling`; skip guard → `require_coupling_lib()`.
Run: `uv run pytest tests/test_python_authoring.py -v`
Expected: PASS (including `efv`/`inmem` < 1e-8 — the rescaled model is the same potential).

- [ ] **Step 4: Full suite, with and without the library**

Run: `uv run pytest -q` (library installed from Task 9 Step 4) → PASS.
Run: `uv sync --reinstall-package ace-jax-coupling && uv run pytest -q` (dev stub, no bundle) → PASS with the coupling-backend tests skipped by `require_coupling_lib()`.
Run: `grep -rn "juliacall\|juliapkg" tests src` → no output.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: coupling parity on the compiled backend (in-process; per-row scale vs ACEpotentials 0.10.1)"
```

---

### Task 11: CI — wheel matrix, clean-environment tests, ace-jax parity

**Files:**
- Create: `.github/workflows/coupling-wheels.yml`
- Delete: `.github/workflows/coupling-parity.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: coupling-wheels
# Build the juliac-compiled coupling library, package platform wheels, test them
# in clean environments (no Julia), and run ace-jax's coupling parity on them.
on:
  pull_request:
    paths:
      - "coupling/**"
      - "src/ace_jax/construct/**"
      - "tests/test_coupling_*.py"
      - "tests/test_python_authoring.py"
      - "fixtures/coupling_ref_*.npz"
      - ".github/workflows/coupling-wheels.yml"
  workflow_dispatch:
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { name: linux-x86_64,  runner: ubuntu-latest,    container: "quay.io/pypa/manylinux_2_28_x86_64",  plat: manylinux_2_28_x86_64 }
          - { name: linux-aarch64, runner: ubuntu-24.04-arm, container: "quay.io/pypa/manylinux_2_28_aarch64", plat: manylinux_2_28_aarch64 }
          - { name: macos-arm64,   runner: macos-14,         container: "",                                     plat: macosx_11_0_arm64 }
    name: build ${{ matrix.name }}
    runs-on: ${{ matrix.runner }}
    container: ${{ matrix.container || null }}
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - name: Julia 1.13 (container)
        if: matrix.container != ''
        run: |
          curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.13
          echo "$HOME/.juliaup/bin" >> $GITHUB_PATH
      - name: Julia 1.13 (macOS)
        if: matrix.container == ''
        uses: julia-actions/setup-julia@v2
        with: { version: "1.13" }
      - uses: actions/cache@v4
        with:
          path: ~/.julia
          key: julia-${{ matrix.name }}-${{ hashFiles('coupling/julia/Manifest.toml', 'coupling/julia/build/Manifest.toml') }}
      - name: Build library
        run: |
          julia --project=coupling/julia/build -e 'using Pkg; Pkg.instantiate()'
          julia --project=coupling/julia/build coupling/julia/build.jl coupling/build
      - name: Python for pruning/packaging
        run: |
          PY=$( [ -x /opt/python/cp312-cp312/bin/python ] && echo /opt/python/cp312-cp312/bin/python || echo python3 )
          $PY -m venv .buildenv && .buildenv/bin/pip install -q numpy build
      - name: Prune + check bundle
        run: |
          .buildenv/bin/python coupling/tools/prune_bundle.py coupling/build/bundle coupling/build/pruned
          .buildenv/bin/python coupling/tools/check_bundle.py coupling/build/pruned
      - name: Wheel
        env:
          ACEJAX_COUPLING_BUNDLE: ${{ github.workspace }}/coupling/build/pruned
          ACEJAX_COUPLING_PLAT: ${{ matrix.plat }}
        run: .buildenv/bin/python -m build --wheel coupling/python -o dist
      - name: auditwheel (Linux)
        if: matrix.container != ''
        run: |
          .buildenv/bin/pip install -q auditwheel
          .buildenv/bin/auditwheel show dist/*.whl || true   # informational; the gates are check_bundle.py (GLIBC <= 2.28) and the manylinux_2_28 test job
      - uses: actions/upload-artifact@v4
        with: { name: "wheel-${{ matrix.name }}", path: dist/*.whl, retention-days: 7 }

  test-wheel:
    needs: build
    strategy:
      fail-fast: false
      matrix:
        include:
          - { name: linux-x86_64,  runner: ubuntu-latest,    container: "quay.io/pypa/manylinux_2_28_x86_64",  py: /opt/python/cp311-cp311/bin/python }
          - { name: linux-aarch64, runner: ubuntu-24.04-arm, container: "quay.io/pypa/manylinux_2_28_aarch64", py: /opt/python/cp313-cp313/bin/python }
          - { name: macos-arm64,   runner: macos-14,         container: "",                                     py: python3 }
    name: test wheel ${{ matrix.name }} (no Julia)
    runs-on: ${{ matrix.runner }}
    container: ${{ matrix.container || null }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { name: "wheel-${{ matrix.name }}", path: dist }
      - run: bash coupling/tools/test_wheel.sh dist/*.whl ${{ matrix.py }}

  ace-jax-parity:
    needs: build
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - uses: actions/download-artifact@v4
        with: { name: wheel-linux-x86_64, path: dist }
      - name: Coupling parity on the built wheel
        env: { JAX_PLATFORM_NAME: cpu, JAX_ENABLE_X64: "1", ACEJAX_REQUIRE_OPTIONAL: "1" }
        run: |
          uv sync --python 3.12 --extra gp
          uv pip install --reinstall dist/*.whl
          uv run --no-sync pytest -n 0 -v tests/test_coupling_parity.py tests/test_coupling_etshim.py tests/test_coupling_cache.py tests/test_python_authoring.py
```
Delete the old job: `git rm .github/workflows/coupling-parity.yml`.
Note: `ACEJAX_REQUIRE_OPTIONAL=1` would also demand lammps-jax/matscipy for tests using `require_optional` elsewhere — the job only runs the four coupling test files, none of which use those.

- [ ] **Step 2: Lint the workflow locally**

Run: `uvx --from actionlint-py actionlint .github/workflows/coupling-wheels.yml` (or `uv run pre-commit run --all-files` if the repo hooks include actionlint)
Expected: no errors.

- [ ] **Step 3: Commit, push the branch, dispatch**

```bash
git add .github/workflows/coupling-wheels.yml
git commit -m "ci: coupling-wheels (manylinux x86_64/aarch64 + macOS arm64 build, clean-env wheel tests, parity); drop coupling-parity"
git push -u origin feat/trim-coupling
gh workflow run coupling-wheels.yml --ref feat/trim-coupling
gh run watch "$(gh run list --workflow coupling-wheels.yml --branch feat/trim-coupling --limit 1 --json databaseId -q '.[0].databaseId')"
```
Expected: all 7 jobs green. Record the three wheel sizes from the `test_wheel.sh` output. Typical first-run failures and where to fix: node-based actions inside the aarch64 container (switch that job's checkout to `git clone` if `actions/checkout` cannot run); missing `objdump` in the container (`dnf install -y binutils`); `LD_DEBUG` trace missing libs (check `prune_bundle.py`'s parsing against the raw `LD_DEBUG=files` output).
(Pushing `feat/trim-coupling` to `origin` = ACEsuit/ace-jax creates a remote branch there — confirm with the user before this step if not already agreed; alternatively push to the user's ace-jax fork.)

---

### Task 12: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/python-authoring.md`
- Rewrite: `docs/dev/coupling-etshim-spec.md` (keep the file name; title "Coupling generation via the compiled EquivariantTensors library")
- Modify: `skills/ace-jax/SKILL.md` only if it mentions juliacall/`authoring` (grep first)

- [ ] **Step 1: Find every stale reference**

Run: `grep -rn -i "juliacall\|juliapkg\|julia_env\|authoring extra\|coupling-parity" README.md CLAUDE.md docs/*.md skills/ | grep -v "docs/dev/plans/"`
Expected: a list of lines; each is rewritten below (historical plans under `docs/dev/plans/` stay untouched).

- [ ] **Step 2: Rewrite**

- `docs/dev/coupling-etshim-spec.md`: the contract section stays (layout ace-jax consumes); replace the JuliaCall bridge, packaging and parity sections with: the three-layer architecture; the C ABI v1 block (copy from this plan's Design); build (`coupling/julia/build.jl`, Julia 1.13, JuliaC 0.3.10, generic CPU, `handle-signals=no`, privatize, macOS 11 floor, trace pruning); the ET-version note (fork `ET_REV` on ET main vs ACEpotentials 0.10.1 / ET 0.4.3: same blocks and spans, per-row scale); the test matrix (bit-exact vs upstream ET in `coupling/python/tests`; scale-aware vs ACEpotentials in `tests/test_coupling_parity.py`); cache stamp = `backend_id()`.
- `README.md`: install line → `pip install ace-jax[authoring]  # + compiled EquivariantTensors coupling (no Julia needed; Linux x86_64/aarch64, macOS arm64)`; replace "juliacall/juliapkg auto-provision a pinned Julia" wording; state the per-row-scale caveat vs ACEpotentials 0.10.1 exports in one sentence where "bit-for-bit with ACEpotentials" is claimed for authoring.
- `CLAUDE.md`: layout line for `construct/coupling.py` ("the EquivariantTensors shim via the compiled ace-jax-coupling library"); add `coupling/` to the layout list (julia shim, build, python package); Extras: `authoring` = `ace-jax-coupling` platform wheel; skip list: coupling tests skip without a working `ace_jax_coupling` (`conftest.require_coupling_lib`); CI list: `coupling-wheels` replaces `coupling-parity`; add the local build recipe (the `JULIA_DEPOT_PATH=$HOME/.julia-trim` note and the `uv sync --extra authoring --reinstall-package ace-jax-coupling` command).
- `docs/python-authoring.md`: remove the "juliapkg scans sys.path" gotcha; the cache paragraph mentions `backend_id()` instead of the `juliapkg.json` pin hash.

- [ ] **Step 3: Verify and commit**

Run: `grep -rn -i "juliacall\|juliapkg" README.md CLAUDE.md docs/*.md skills/ src tests pyproject.toml` → no output.
Run: `uv run pre-commit run --all-files` → PASS.
```bash
git add README.md CLAUDE.md docs/ skills/
git commit -m "docs: compiled coupling library replaces juliacall authoring"
```

---

## Out of scope (follow-ups)

- Publishing `ace-jax-coupling` to (Test)PyPI — needs the licence decision (GPL components) and a release workflow.
- Upstreaming the ET patches to ACEsuit (the fork branch is PR-ready; maintainer capacity permitting).
- A 0.4.3-backport build for coefficient-identical parity with ACEpotentials 0.10.1.
- Intel macOS / Windows wheels.
- Replacing the UMFPACK-internals helper once SparseArrays' `ftranspose(f::Function)` / `UmfpackLU` getproperty are trim-safe upstream.
