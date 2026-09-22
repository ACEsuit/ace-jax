# Coupling generation via an EquivariantTensors-only JuliaCall shim

**Goal.** Generate the SO(3) coupling artifacts (`A2B` map + `aa_spec`) — the one
piece [EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl)
owns — from Python, in-process via `juliacall`, so a *new basis shape* can be
authored without the full ACEpotentials.jl stack. Radials, pair basis, embedding
and splining are still taken from a Julia export (a separate follow-up). Refitting
or evaluating an existing exported model stays 100% Julia-free — nothing here runs.

## The contract (must match bit-for-bit)

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

## Python spec builder (`construct/spec.py`)

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

## JuliaCall bridge (`construct/coupling.py`)

`couple(mb_spec, Rnl_spec, Ylm_spec)` calls ET in-process via `juliacall` (no
subprocess, no npz shuffling):

```python
from juliacall import Main as jl
jl.seval("using EquivariantTensors")
tensor = jl.EquivariantTensors.sparse_equivariant_tensor(
    L=0, mb_spec=<jl vector of NamedTuples>, Rnl_spec=<…>, Ylm_spec=<…>, basis=jl.real)
A2B = jl.Matrix(tensor.A2Bmaps[1])   # -> numpy via juliacall
```

It returns `A2B` (dense `(n_B, n_AA)` numpy), `aa_spec` (0-based int arrays), and
the A-basis spec, converting Julia 1-based → 0-based to emit exactly the
exporter's layout. The per-B `nnll` is recomputed in Python from `mb_spec` (it is
just the per-B `(n,l)` list), keeping the shim ACEpotentials-free — it depends on
EquivariantTensors alone.

The column identities used to align `A2B` come from ET's `meta["𝔸spec"]` (the
spec returned *with* the symmetrisation matrix), not `aabasis.specs`:
`SparseSymmProd` re-sorts its input, so the evaluation order is a different
permutation from the `A2B` column order.

## Packaging (the `authoring` extra)

- `pyproject.toml`: `authoring = ["juliacall", "juliapkg"]`. Core install
  (`pip install ace-jax`) pulls **neither** — refitting/evaluating users stay
  Julia-free.
- `juliapkg.json` declares **only** `EquivariantTensors`, pinned to the
  registered version the committed reference fixtures were generated with.
  `juliapkg` auto-provisions a private Julia + the package on first
  `import juliacall`; no system Julia is needed.
- The ET/Lux/GPU toolchain is pulled **only** when someone installs
  `ace-jax[authoring]` and authors a new shape. ET drags Lux +
  KernelAbstractions + GPUArraysCore, so `authoring` is a heavy extra (long first
  precompile), not a featherweight CG library.

## Parity test + caching

- `tests/test_coupling_parity.py` **skips cleanly** (`importorskip("juliacall")`)
  so the pip-only core suite is unaffected. With the extra present, it feeds the
  bridge the specs of the committed reference fixtures
  (`fixtures/coupling_ref_*.npz`) and asserts the resulting `A2B` + `aa_spec`
  equal the reference **bit-for-bit** for orders **2, 3, and 4**. Multiplicity-1
  `nnll` blocks match exactly; degenerate blocks (order ≥ 4, e.g. SiGe `wL=0.5`)
  are unique only up to a within-block orthogonal rotation, so those are matched
  on the **row subspace** (a row-space projector — `subspace_residual`,
  `tests/test_coupling_subspace.py`). CI regenerates the references from
  ACEpotentials at the same pinned ET (`julia/coupling_reference.jl`); a
  registered-ET pin keeps both sides self-consistent with no ACE registry needed.
- **Caching per shape.** `A2B`/`aa_spec` depend only on
  `(elements, order, totaldegree, wL, lmax)` — not on data or the fit. So the
  shim runs **once per new shape**; the tiny table (a few KB–MB) is cached/shipped.
  New coefficients on an existing shape never invoke it → refitting is 100%
  Julia-free. This is now implemented as `couple_cached` — a per-shape disk
  cache keyed by a sha256 of the specs, with hits that never import juliacall
  (see `docs/python-authoring.md`, "Coupling cache"; `couple()` remains the
  uncached parity oracle).
