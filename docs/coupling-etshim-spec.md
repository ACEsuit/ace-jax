# Coupling generation via an EquivariantTensors-only JuliaCall shim

**Goal.** Generate the SO(3) coupling artifacts (`A2B` map + `aa_spec`) — the one
piece EquivariantTensors.jl owns — from Python, in-process via `juliacall`, so a
*new basis shape* can be authored without ACEpotentials.jl. Radials, pair basis,
embedding and splining stay exported for now (separate follow-up, out of scope).
Refitting on an existing exported model stays 100% Julia-free (nothing here runs).

## The contract (must match bit-for-bit)

`acejax/julia/export_model.jl` builds the whole basis with one ET call and writes:

- `aspec_r`, `aspec_y` — the A-basis spec `m.tensor.abasis.spec` = `(Rnl_idx,
  Ylm_idx)` per A function, **0-based** (`export_model.jl:198-200`).
- `aa_spec_k` (k = 1..order) — `m.tensor.aabasis.specs`, each `(n_v, k)` int
  matrix of A-indices per AA function, **0-based** (`:409-410`).
- `A2B` — `Matrix(m.tensor.A2Bmaps[1])`, shape `(n_B, n_AA)`, stored as sparse
  triplets `A2B_rows/cols/vals/A2B_shape`, **0-based** (`:344-347`).
- `nnll` — `get_nnll_spec(m.tensor)` = the `(n,l)` per B function (`:331`).

The eval side (`ace_jax/eval/io.py`, `eval/model.py`) reads `A2B` (triplets →
dense `(n_B, n_AA)`) and the per-order `aa_spec_k`, and contracts `AA → B` with
`A2B`. **The row/column ordering of `A2B`, the A-index numbering used inside
`aa_spec`, and the B-function order must all match the exporter**, or fitted
coefficients stop being interchangeable and the ACEfit-parity guarantee breaks.
`sparse_equivariant_tensor` is deterministic given identical integer specs, so
parity reduces to: **feed ET the same `mb_spec`/`Rnl_spec`/`Ylm_spec` the Julia
path fed it.**

## Phase 1a — Python spec builder (`construct/spec.py`)

Inputs `(elements, order, totaldegree, wL, lmax)` → the three integer specs ET
consumes:
- `Rnl_spec`: list of `(n, l)` radial-channel tuples, `n≥1`, `l≤lmax`, admitted
  by the level function.
- `Ylm_spec`: `(l, m)` for `l≤lmax`, `-l≤m≤l` (real harmonics ordering).
- `mb_spec`: per-B-function `Vector{(n=…, l=…)}` up to `order`, filtered to
  `totaldegree` under `TotalDegree(NZ, 1/wL)` (`ace1_compat.jl:104,229`: the level
  weights `l` by `1/wL`, `n` by `NZ`), and by the admissibility predicate below.

**Ported `_rpe_filter_real(L=0)`** (from `ACEpotentials/src/models/utils.jl:37`,
~23 lines): a basis tuple `bb` (list of `(n,l,m)`) is admissible iff
`_mm_filter([m…], L)` (some signed m-combination sums to ≤L in abs) **and**
`iseven(sum(l)+L)` **and** the L=0/single-entry special case (`len==1 ⇒ l==0`).
For L=0: `sum(m)`-based m-filter + even `sum(l)` + the singleton rule. Pure
Python, ~25 lines, no deps.

> **Known gap (honest).** A *complete* from-scratch `mb_spec` requires
> reproducing ACEpotentials' `TotalDegree`/`wL` enumeration and its exact
> tie-break/sort order (`ace.jl:_make_idx_AA_spec` sorts; ordering feeds the
> A-index numbering). That is the bulk of the ~400–600 LOC "row 2" work and is
> where a subtle ordering mismatch would hide. For **parity testing now** we
> instead reconstruct the specs from an existing export (Phase 3), which
> exercises the *coupling bridge and the layout contract* without first having to
> get the enumerator's ordering bit-identical.

## Phase 1b — JuliaCall bridge (`construct/coupling.py`)

In-process via `juliacall` (no subprocess, no npz shuffling):

```
from juliacall import Main as jl
jl.seval("using EquivariantTensors")
tensor = jl.EquivariantTensors.sparse_equivariant_tensor(
    L=0, mb_spec=<jl vector of NamedTuples>, Rnl_spec=<…>, Ylm_spec=<…>, basis=jl.real)
A2B  = jl.Matrix(tensor.A2Bmaps[1])          # -> numpy via juliacall
specs = tensor.aabasis.specs                  # per-order
nnll  = jl.EquivariantTensors.get_nnll_spec(tensor)   # or ACEpotentials.Models — see note
```

Returns `A2B` (dense `(n_B, n_AA)` numpy), `aa_spec` (tuple of 0-based int
arrays), `aspec_r/y` (0-based). The bridge converts Julia 1-based → 0-based and
emits exactly the exporter's layout. **Note:** `get_nnll_spec` currently lives in
`ACEpotentials.Models` (`ace.jl:142`); the shim must either (i) call the ET-level
equivalent if ET exposes one, or (ii) recompute `nnll` in Python from `mb_spec`
(it is just the per-B `(n,l)` list) — preferred, to stay ACEpotentials-free.

## Phase 1c — Packaging (optional `authoring` extra)

- `pyproject.toml`: `[project.optional-dependencies] authoring = ["juliacall",
  "juliapkg"]`. Core install (`pip install ace-jax`) pulls **neither** — refitting
  users stay Julia-free.
- `juliapkg.json` (repo root or package): declares **only** `EquivariantTensors`
  (+ the ACE registry if ET is not in General). `juliapkg` auto-provisions a
  private Julia + the package on first `import juliacall`. No system Julia needed.
- The whole ET/Lux/GPU toolchain is pulled **only** when someone `pip install
  ace-jax[authoring]` and authors a new shape. Honest caveat: ET drags Lux +
  KernelAbstractions + GPUArraysCore, so `authoring` is a heavy extra (long first
  precompile), not a featherweight CG lib.

## Phase 2/3 — Parity test + caching

- `tests/test_coupling_etshim.py`: **skips cleanly** (`importorskip("juliacall")`)
  so the 197-test core suite is unaffected. When the extra is present: reconstruct
  the ET inputs for the shape of an existing fixture (`si_ace_model.npz` /
  `si_fitted.npz`), call the bridge, and assert the resulting `A2B` + `aa_spec`
  equal the fixture's — **bit-for-bit up to a basis-ordering permutation**. If a
  permutation is needed, resolve it (match B-functions by their `nnll` signature +
  the A2B column support) and document it.
- **Caching per shape.** `A2B`/`aa_spec` depend only on `(elements, order,
  totaldegree, wL, lmax)` — not on data or the fit. So the shim runs **once per
  new shape**; cache/ship the tiny table (a few KB–MB). New coefficients on an
  existing shape never invoke it → refitting is 100% Julia-free.
