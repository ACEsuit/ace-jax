# Post mortem: misaligned A2B columns in built bases (#57)

**Bug.** From the compiled coupling library (#20, 0.2.0) until #57, every basis
built in Python (`build_basis`, so `aj basis` and `aj fit --order/--max-degree`)
multiplied A2B column j against the wrong AA product. The library returns A2B
columns in ET's `𝔸spec` order (the `mb_spec` body order) but `aa_specs` in
`SparseSymmProd` order (bodies stable-sorted by length). `build_spec`'s DFS
interleaves body orders, so the two orders differ. B functions were not
rotation invariant, and γ was attached to the wrong rows. Found from a fit
(GAP-18 Si o4d12 force RMSE floor 0.19 vs 0.13 eV/Å for ACEpotentials), not by
any test.

**Question.** Why did the Julia parity CI, which compares the full design
matrix against ACEpotentials, not catch it?

## Short answer

The Julia parity job never ran ace-jax's basis construction. Every test that
compared against ACEpotentials either took the basis from Julia, or fed the
coupling library the *ACEpotentials* `mb_spec` order (on which the bug cannot
occur), or re-aligned the columns itself before comparing. The one ordering
that triggers the bug — `build_spec`'s — was never pushed through to an
evaluated quantity.

Measured on the `si_ace_model.npz` shape (Si, o3, d10, 230 AA columns), with
`align_columns` spied on:

| `mb_spec` fed to `couple` | grouped by body length? | columns permuted by the fix |
|---|---|---|
| ACEpotentials (fixture `nnll`) | yes | **0 / 230** |
| `build_spec(1, 3, 10, wL=1.5)` | no (`[1, 2, 3, 3, …, 3, 2, 3, …]`) | **229 / 230** |

## The five layers that each missed it

### 1. `julia-parity` tests the evaluator, not the builder

`.github/workflows/julia-parity.yml` runs `julia/export_model.jl`, which builds
**and couples** the model in ACEpotentials (`ace1_model`), fits it, and writes
A2B, `aa_spec_*`, radials and the design matrix to `fixtures/si_fitted.npz` /
`si_tiny_design.npz`. `tests/test_gp_rows.py` then `load`s that npz and checks
ace-jax's rows against ACEfit's. A2B and `aa_specs` both come from Julia, so they
are consistent by construction. The test proves "ace-jax evaluates a given ACE
model exactly like ACEpotentials"; it says nothing about whether ace-jax can
*build* that model.

Its path filter (`eval/model.py`, `fit/rows.py`, `fit/solve.py`, `julia/…`) also
does not include `src/ace_jax/basis/**`, so a change to the builder would not
trigger it even if it did cover it.

### 2. Coupling parity compares coefficients up to column permutation

`tests/test_coupling_parity.py` (run by `coupling-wheels` on every coupling or
basis change) does compare a `build_spec`-generated coupling with ACEpotentials —
the end-to-end mode. But it first builds a bijection from the Python columns to
the reference columns **by (n, l, m) signature** (`perm = ref_col[s] for s in
aa_sig_py`) and compares in the reference's order. That is exactly the
alignment `align_columns` now does in production. The test was correct about the
question it asked (are the coefficients and the `mb` set right?) and blind by
design to the question that mattered (does column j of A2B match AA product j
*in evaluation order*?). It reads `aa_sig`, which was in `𝔸spec` order, and
never looks at `aa_specs`.

The docstring of `couple` even recorded the hazard ("`SparseSymmProd` re-sorts
its input, so the aabasis EVALUATION order is a different permutation from the
A2B column order"), but the conclusion drawn was about which spec to use for
column *identity*, not that `build_model` paired the columns with the wrong
products.

### 3. The one evaluated end-to-end check used the safe `mb_spec`

`tests/test_basis_build.py::test_bridge_wellformed_subprocess` is the only
pre-#57 test that builds a model and checks E, F, virial and descriptors against
ACEpotentials. But:

- it calls `couple(mb_ref, …)` with the **fixture's** `nnll` (ACEpotentials'
  length-grouped order), not `build_spec`'s;
- the model from `build_model([14], 3, 10)` (which did use `build_spec`) is used
  only for shapes and meta, and its A2B is **replaced** by `cpl.A2B` before
  evaluation;
- it computed `col_perm_identity` and marked it `# informational` — never
  asserted. On the safe order it is the identity anyway.

So the evaluated path in this test is "ACEpotentials order → library → eval",
which was correct.

### 4. Most builder tests never call the live library

`_primed_cache` (in `test_basis_build.py`, reused by `test_basis_spec.py`,
`test_fit_basis_pipeline.py`, `test_lean.py`, `test_radial_init_table.py`) writes
the **Julia fixture's** coupling into the coupling cache under the key that
`build_spec(1, 3, 10)` produces. `build_model`/`build_basis` then hit the cache
and receive a coupling whose A2B columns already follow its own `aa_specs` — a
correct coupling the live library would never have produced for that key. This
made the suite Julia-free and fast, but it substituted a known-good answer for the
exact component that was wrong.

### 5. No reference-free property test on built bases

Rotation/permutation invariance of site descriptors would have caught this with
no reference at all (o2d6: 5 / 23 columns broken, max relative change 1.04). The
pre-#57 invariance tests (`test_ard_equivariance`, `test_pace_model`, …) ran on
Julia-exported or PACE models only. Embedding bases (`build_embedding_spec`
length-sorts its spec) and every Julia fixture are on the safe path, so every
model a test happened to evaluate was correct.

## Why it stayed invisible in use

- A misaligned basis is still a valid linear model: fits converge, the LML is
  finite, energies look plausible. The damage shows only as an accuracy floor
  (and rotation-dependent predictions), which reads as "ACE is a bit worse here".
- The coupling cache made a wrong answer persistent and fast.
- The parity CI was green and described as "full design matrix vs
  ACEpotentials", which was true for the imported-model path and was taken to
  cover the built-model path.

## What #57 added, and what is still open

Added (`tests/test_basis_invariance.py`, `test_coupling_etshim.py`):
rotation and neighbour-permutation invariance of built bases (including after an
npz round trip), γ against the evaluated body of each row, column alignment of
the live coupling against `aa_specs` on an interleaved `build_spec`, repair of a
stale cache entry, and per-column parity of a `build_spec` basis with
`si_ace_model.npz`.

Still recommended:

1. **Make `julia-parity` cover the builder.** Done after this post mortem:
   `julia/built_basis_reference.jl` builds the `ace_model` that `build_model`
   reproduces (Si o3d10, Si o4d12 with a multiplicity-2 block, SiGe o3d6) and
   exports its radials and basis E/F/V on three rattled bulk cells.
   `tests/test_built_basis_parity.py` builds the same basis with the live
   coupling library (no cache), grafts on only the radials, and requires each
   (species, body) block of design columns to span the Julia block (to 1e-9;
   observed <= 2e-14), the pair columns and γ to match. With `align_columns`
   disabled it fails 109/110, 336/337 and 244/246 blocks (residual ~1). The job
   now also runs on `src/ace_jax/basis/**` and fails, not skips, on a missing
   fixture or coupling library; `coupling-wheels` runs the test on each new
   wheel.
2. **Compare in evaluation order, never re-align in the test.** In
   `test_coupling_parity`, assert that `aa_sig` agrees with the signatures read
   from `aa_specs` *before* mapping to the reference. A test that applies the
   production fix-up itself cannot detect its absence.
3. **Assert, don't log.** Turn `col_perm_identity`-style "informational" values
   into assertions, or delete them. Run the bridge test on `build_spec`'s order
   as well as the fixture's.
4. **Limit `_primed_cache`.** Keep it for tests of cache mechanics and eval
   plumbing, but every builder-correctness test should run the live library at
   least once (it is a core dependency, so this costs seconds, not Julia).
5. **Property tests for every model source.** Rotation and permutation
   invariance (and, cheaply, `E(R·x) = E(x)`, `F(R·x) = R·F(x)`) on each way of
   making a model: Julia export, `build_basis`, embedding, PACE, lean, splined.
6. **Test the non-canonical input.** Here the bug needed an `mb_spec` that is not
   grouped by body length. When a reference is used, also feed a shuffled or
   interleaved version of its input, so the test cannot pass only because the
   reference happens to use a canonical order.
