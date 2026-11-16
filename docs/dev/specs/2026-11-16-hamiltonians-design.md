# ACE Hamiltonians in ace-jax: design for phases 0, 1 and 2a

Status: design, awaiting approval (2026-11-16). Branch `feat/hamiltonians`, one long-lived branch.
Background, physics and the route comparison: `docs/dev/acehamiltonians-port-scoping.md` (revision 3).
This document records what was decided on top of it and what is built in this branch. Where the two
differ, this one wins.

## Intent

Predict the real-space Hamiltonian and overlap blocks H(T), S(T) of an atomic-orbital DFT code from
atomic structure, with a linear, O(3)-equivariant ACE model that is fitted in closed form, carries a
calibrated uncertainty, and writes its predictions in a format DFT-injection codes read.

Why now: ACEhamiltonians.jl showed the model works but was not adopted, mainly because inference
was slow. ace-jax has the evaluation and fitting machinery to do the same physics fast, in Python.
**Inference speed is therefore a design constraint from the first commit, not a phase 3 topic.**

## Goals

1. `ace-jax-coupling` 0.3.0: L > 0 and both reflection parities from the compiled library, a
   backwards-compatible release (ABI v1's `etc_couple` unchanged). This comes first.
2. `ace_jax.ham`: on-site and off-site block models, operator assembly H(T), S(T) → H(k), an npz
   model file.
3. Data: a pure-Python DeepH reader and writer; a Quoll ingestion script writing a frozen npz cache.
4. Stage-A fitting: QR evidence per (species or species pair, L, parity), every shell pair a
   right-hand side; block covariance, jackknife shape and Mondrian conformal scales; POPS.
5. Validation on Quoll's SiC data and the ACEhamiltonians Si data; DeepH prediction output with an
   uncertainty twin.

## Non-goals (this branch)

- Observable (spectral) training, the degenerate-safe eigensolver rule: phase 2b.
- Dense bond layouts, GPU k-batches, a `jax.export` artifact, a C++ kernel: phase 3.
- Learned orbitals: phase 4.
- An `aj ham` CLI and user documentation pages. Python API only until 2a is done.
- Loading old ACEhamiltonians.jl models, or parity with its basis functions.
- A trim-compiled Quoll.

## Anti-goals

- One basis evaluation per shell pair. The cost of a structure must not grow with the number of
  shells except through the final contraction.
- Julia at fit or prediction time, or `juliacall` as a dependency.
- A wrong sign at the convention boundary that fits well: every convention is fixed by a rotation
  test on Quoll-converted data, never by inspection.
- Freezing today's energy-specific internals as the framework API.

## Constraints

- `ham` sits behind a `ham` extra (`h5py`), imported inside the functions that need it
  (`tests/test_core_deps.py`).
- An import-boundary test: `ham` imports only the framework modules of scoping §9; those import
  nothing from `ham` or the potential-only modules.
- The main test job stays Julia-free: a committed coupling cache for the ham bases and
  `ACEJAX_COUPLING_CACHE_ONLY=1`, a committed DeepH fixture under 4 MB.
- float64; the library never sets x64.
- No code derived from GPL sources; the coupling bundle stays GPL-free (`nullspace_solver=:dense`).
- Pushing to the ET fork, touching the ET pull request, and publishing any release are confirmed
  with the owner each time.

## Approaches

**Sharing across shell pairs.**
- *Per-shell-pair sub-models* (ACEhamiltonians): each (species pair, ℓ₁, ℓ₂) has its own basis.
  Flexible per block, but an atom or bond is evaluated once per shell pair. Rejected: it is the
  speed problem.
- *Shared features, per-shell-pair coefficients* (chosen): one A/AA per centre species (on-site)
  and per species pair (off-site); one feature set B^(L,p) per L and parity p from that AA; a shell
  pair holds only coefficients. All shell pairs of a species share one basis size; per-block degree
  caps can be added later as row masks.

**Off-site basis.**
- *A. Bond-environment ACE* (scoping §1, §4): environment atoms in a cylinder about the bond
  midpoint, pooled per bond, exactly one bond factor per product. Known accuracy. Cost: one
  pooling per bond.
- *B. Site features plus a bond factor*: reuse each atom's AA products from the on-site
  evaluation, couple with one R_n Y_l(r_ij) factor, for i and for j, and symmetrise. Cost: about
  one potential evaluation per structure plus a contraction per bond. Accuracy unknown: no terms
  joint in the two environments, spherical environments.
- *Chosen:* A first, behind a provider interface; B added in phase 2a and compared on accuracy and
  speed on the same data, before any phase 3 layout work.

**L > 0 couplings.**
- *Offline tables from ordinary Julia, committed as a cache*: works today (checked for L ≤ 2, both
  parities, on the pinned fork). Kept only as the fallback of scoping §11 risk 2.
- *Compiled library first* (chosen by the owner): fork trim fixes, ABI v2, release 0.3.0, then the
  model.

## Design

### 1. Coupling library 0.3.0

- **Fork** (`jameskermode/EquivariantTensors.jl`, new commits on `jk/trim-coupling`): forward
  `refl_sym` in `symmetrisation_matrix(Val(L), …)`; add L ∈ 1…L_MAX and both `refl_sym` values to
  `test/trim/entry.jl`; fix what `--trim=safe` reports.
- **ABI v2**, additive. `etc_abi_version()` returns 2. `etc_couple` keeps its signature and
  results bit for bit. New:
  - `etc_max_L()`: L_MAX, a compile-time constant, 6 (spdf orbitals).
  - `etc_couple_equivariant(L, parity, <the etc_couple inputs>, sizes, a2b_i, a2b_j, a2b_m, a2b_v,
    <the other etc_couple outputs>)`: A2B as (row, col, m, value) quadruplets, m ∈ 0…2L. `parity`
    is +1 or −1 = (−1)^Σl of the products, mapped to ET's `refl_sym` (`:sym` is Σl even); it is
    not "true/pseudo", which depends on L. L = 0, parity +1 must reproduce `etc_couple`.
  - New return code `IMAGINARY_COUPLING` for ET's purely-imaginary real-basis case, never a throw.
- **Python**: `ace_jax_coupling.couple_equivariant_raw`; the loader accepts ABI ≥ 2 for it.
  `basis/coupling.py::couple_equivariant(L, parity, mb, rnl, ylm)` → `Coupling` with `A2B` of
  shape (n_B, 2L+1, n_AA), aligned to the AA evaluation order by the existing `align_columns`;
  `couple_equivariant_cached` with L and parity in the key. `COUPLING_LIB_VERSION` → 0.3.0.
- **Parity**: `coupling/julia/reference/` extended to L > 0 and both parities against unpatched ET
  at the fork's base: index arrays identical, values within a few ulp, `subspace_residual` for
  degenerate blocks. The existing L = 0 parity must not move.
- **All L share one AA.** The model calls the library once per (L, parity) with the same
  `mb_spec`; the union of the returned AA columns is one evaluation order (checked by signature).

### 2. Framework refactor

- Split `EdgeSiteModel` so the A/AA product, layouts and `estimate_a_bytes` are usable without the
  scalar-energy E/F/V path; potentials keep their behaviour and goldens bit for bit.
- A generic row interface in `fit/`: a row generator yields (Φ, Y, w) per quantity, Y with any
  number of right-hand-side columns; `linear_qr_statistics` and the QR evidence take it. The E/F/V
  generator becomes one implementation. Multiple right-hand sides share R; each has its own c.
- These two are the only edits to potential code paths. Pipeline goldens are re-checked on
  lestrade and must not change.

### 3. `ace_jax.ham`

- `orbitals.py`: the orbital basis as data: per species, a list of shells (ℓ, and a label). Carried
  in the model file. No shell tables in code.
- `cg.py`: real Clebsch–Gordan stacks for (ℓ₁, ℓ₂, L) in ace-jax's Y_lm convention, computed in
  Python (closed form through the complex CG and the real transform), orthogonality tested.
- `features.py`: `site_features(model, graph)` → per atom, {(L, p): (n_B, 2L+1)};
  `bond_features(provider, graph)` → per bond, the same. Providers: `BondEnv` (A) and, in 2a,
  `SiteBond` (B). Both return features for the bond and its reverse so the readout can symmetrise.
- `graph.py`: bonds (i, j, T) within the orbital cutoff from `eval/nlist.py`; for `BondEnv`, the
  (bond, env atom) pairs with vectors relative to the midpoint and the cylindrical envelope.
  Padded to power-of-two buckets, as the calculators do.
- `readout.py`: block(ℓ₁, ℓ₂) = Σ_{L,p} CG_L · (C[ℓ₁ℓ₂, L, p] · B^(L,p)), one einsum per (L, p)
  over all shell pairs at once. Off-site: symmetrised with the reverse bond with sign
  (−1)^(ℓ₁+ℓ₂−L), so H_ji(−T) = H_ij(T)ᵀ exactly. On-site: only shell pairs s₁ ≤ s₂
  are fitted and (s₂, s₁) is the transpose; a same-shell block is symmetric, so its odd-L parts are
  left out of the feature set rather than fitted to zero.
  Per-block mean (ACEhamiltonians' `mean`) is an L = 0 constant column.
- `model.py`: `HamModel` (an `eqx.Module`): on-site model per species, off-site provider per
  species pair, coefficients, CG tables, orbital basis; `predict(atoms)` → blocks keyed
  (i, j, T); the same object serves H and S (two coefficient sets on one basis), so S costs only a
  second contraction.
- `operator.py`: blocks → H(T), S(T) → H(k), S(k) by Bloch sum; `bands(kpath)` through
  Cholesky + `eigh` (values only; no gradient claims before 2b).
- `build.py`: `build_ham_basis(spec)` on the pattern of `basis/model.py::build_basis`.
- `io.py`: npz schema, top-level key `ham_json`; `eval/io.py::load` dispatches on it.
- `calc.py`: an ASE calculator exposing `results["hamiltonian"]`, `["overlap"]` and their σ.

**Speed rules that hold from phase 1:** one A/AA per atom and per bond whatever the number of
shells; H and S from one feature evaluation; jit variants bounded by (species or pair) × bucket,
never by shell pair; radials through the existing spline/table forms; a benchmark
(`bench/ham/`) reporting ms per atom against ACEhamiltonians.jl on the same structure, run from
the first working model on.

### 4. Data

- `ham/deeph.py`: reader and writer for the DeepH layout (scoping §2 (a)), DeepH m-order and signs
  mapped to ace-jax's convention by a per-ℓ signed permutation table held in one place.
- `ham/julia/` (`Project.toml` + committed `Manifest.toml`, Julia 1.13, Quoll pinned at f26de28,
  **Configurations held at 0.17.6**: the release after it breaks Quoll's precompile): `ingest.jl <in> <out>`
  writes the frozen cache, npz blocks per (z₁, z₂, shell pair) in ace-jax's convention plus the
  Quoll commit and options.
- A path-gated `ham-ingest` CI job checks the Python reader against the script on a subset of
  Quoll's artifact. The convention test (rotate, convert with Quoll, check
  H(QR) = D(Q) H D(Q)ᵀ with ace-jax's D) runs there and, on the committed fixture, in the main job.
- SH anchor: SpheriCart internally, conversion at the IO boundary. Pseudo-tensor parts on by
  default.

### 5. Fitting and UQ (2a)

- Targets projected onto L with the CG stack: independent regressions per (species or pair, L, p),
  2L+1 rows per sample, all shell pairs as right-hand sides of one QR.
- QR evidence only (`lml_solver="qr"`); one σ per quantity; problems of equal width vmapped.
- UQ: BLR block covariance B_Lᵀ Σ B_L; delete-one-structure jackknife shape; Mondrian conformal
  scales with χ_{2L+1} reference, groups (species pair, bond-length band); POPS pointwise.
- Output: `hamiltonians_pred.h5` and an element-wise σ twin in DeepH layout.

### 6. Validation

- Structural (pure JAX): rotation and inversion equivariance of every block, exact bond reversal,
  Hermitian H(k), on-site symmetry.
- Coupling parity and the convention test, as above.
- Accuracy, fresh fits: block RMSE and band RMSE in a window on Quoll's SiC and the
  ACEhamiltonians Si data. Comparator: ACEhamiltonians.jl fitted on the same data (reference
  environment below), accuracy and inference time. No basis-level parity with it.
- Provider A against provider B: accuracy and ms per atom, same data, reported in
  `docs/dev/ham-results.md`.

### Reference environments (not in the repo)

`/storage/eng/essswb/projects/ace-jax/ham-ref/`: Quoll.jl @ f26de28 on Julia 1.13.1 (round trips
checked on SiC: DeepH exact, FHI-aims to 1.4e-14); ACEhamiltonians.jl @ 9ff1572 on Julia 1.10 with
ACE 0.12.22, ACEfit 0.1.4 and a one-line local patch (`linear_solve` → `ACEfit.solve`); its suite:
928 pass, 9 errors not yet read, all 588 basis checks pass. The Si paper data is in
`/storage/eng/essswb/ACEhamiltonians-data`.

## Order of work

1. Coupling: fork fixes under trim → ABI v2 → parity → local wheel; **stop for approval** → fork
   push, release 0.3.0, dependency bump.
2. Framework refactor and the import boundary.
3. DeepH IO, ingestion script, fixture, convention test.
4. On-site model, then `BondEnv`, assembly, structural tests, first speed benchmark.
5. Generic rows, stage-A fit, UQ, DeepH output.
6. `SiteBond` provider; accuracy and speed comparison; results document.

## Left to the builder

File and function names inside `ham`; bucket sizes; the fixture's content (smallest subset of SiC
that exercises s, p, d and two species); the benchmark harness; how the nine erroring
ACEhamiltonians tests are handled if the comparator needs those paths.

## Calls made here that the owner has not explicitly confirmed

- L_MAX = 6 in the library.
- `parity` = (−1)^Σl in the ABI, rather than exposing `refl_sym`.
- H and S as two coefficient sets on one basis.
- Same-shell on-site blocks: odd-L parts left out; (s₂, s₁) taken as the transpose of (s₁, s₂).
- The model work (steps 2–4) continues against a local library build while the 0.3.0 release
  waits for approval.
