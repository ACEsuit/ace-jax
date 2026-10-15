# Scoping: ACE Hamiltonians in ace-jax (a port of ACEhamiltonians.jl)

Status: research and scoping only. No code yet. Revision 3 (2026-10-15). Revision 2 took in the owner's decisions:
- old ACEhamiltonians models are dropped;
- the EquivariantTensors (ET) pin may move to the latest release;
- data comes from Quoll.jl;
- spectral and observable losses are in scope;
- consumers are Python/JAX tools and DFT-injection codes;
- UQ is by BLR and conformal prediction;
- a framework/application split is wanted.

**What revision 3 changes.** Revision 2 was written against ace-jax `main` @ 396fee0. Since then (81 commits, to f4324b7):
- **No Julia at runtime, and no juliacall/juliapkg.** Coupling tables come from `ace-jax-coupling`, ET's coupling construction compiled with `juliac --trim=safe` (Julia 1.13) and shipped as a platform wheel (`coupling/`, `docs/dev/coupling-etshim-spec.md`). It is a core dependency. Its C ABI v1 gives **L = 0 only**. Revision 2's "extend the juliacall shim, ~200 LOC" (§4) and "bump `juliapkg.json` to Julia 1.12" (§3) no longer apply. L>0 and pseudo-tensor couplings now mean an ABI v2 in the compiled library (§3), and in-process Quoll would reintroduce juliacall only for ingestion (§2).
- **ET has moved past the pin question.** The library builds from a fork rev (`jameskermode/EquivariantTensors.jl` `jk/trim-coupling` @ 1f24438, on ET `main` after v0.5.1). ET **v0.5.2** (2026-10-02) makes WignerD test-only, which removes the Quoll co-resolution conflict revision 2 waited for. The coupling-parity re-baselining revision 2 planned is done: parity is now per nnll block span (§3).
- **Renamed layout.** `construct/` is now `basis/` (`basis/spec.py`, `basis/coupling.py`, `basis/export.py`, `basis/model.py::build_basis`). Developer docs live in `docs/dev/`, where this file now sits.
- **Fitting.** The linear LML and posterior are QR by default (`QRStats`, batched merges, a host QR stream on the CPU), which is what a near-interpolating block fit needs (§5). Row generation (`fit/rows.py`) is still E/F/V-specific.
- **UQ.** `--uq ard` revision 2 serves a calibrated per-atom 3×3 force covariance: a delete-one-cluster jackknife shape (`fit/jackknife.py`) times Mondrian conformal scales (`fit/conformal.py`, `aj calibrate`). That is the template for equivariant block UQ (§7).
- **Evaluation performance.** The lean form (`eval/model.py::lean`: pruning, pair folding, l-blocked dense A), spline radials and radial tables in r (`eval/splinify.py`), skin neighbour lists (`calc/skin.py`) and dense row-blocked layouts exist now. Phase 3 reuses them (§10).
- Quoll.jl has not changed since 2026-07-26 (still 0.1.0, unregistered). ACEpotentials 0.10.2 still requires ET 0.4.3.
- **New §2.1:** a first probe of compiling Quoll with `juliac --trim`. It does not work with Quoll as it is (949 verifier errors; 147 without ASEconvert, a lower bound). The section gives the reasons and what would be needed.

Sources read:
- ACEhamiltonians.jl `main` @ 9ff1572;
- ACE.jl v0.12.22, the version ACEhamiltonians pins;
- Quoll.jl `main` @ f26de28 (2026-07-26);
- EquivariantTensors v0.4.3, v0.5.1, v0.5.2 (2026-10-02) and the trim fork @ 1f24438;
- ace-jax `main` @ f4324b7 (revision 2: @ 396fee0).

## 1. What ACEhamiltonians.jl does (reference for the physics only)

**Model.**
- Blocks are linear, O(3)-equivariant and matrix-valued: one on-site or off-site sub-model per species pair and shell pair (ℓ₁, ℓ₂).
- On-site block: `H_block = Σ_k c_k Re[C_ℓ₁ B_k C_ℓ₂†] + mean`, with `B_k = Σ U_k,AA AA` and U an ACE.jl `SphericalMatrix(ℓ₁,ℓ₂)` coupling (complex, converted to real by `_ctran`, `fitting.jl`).
- The parity filter admits every coupled L in |ℓ₁−ℓ₂|…ℓ₁+ℓ₂, including the pseudo-tensor parts with L + Σl odd. An example is the antisymmetric L=1 part of a pp block.

**On-site basis.** Ordinary site ACE.

**Off-site basis.** A bond-environment ACE (`basis.jl::_off_site_ace_basis_no_sym`, `states.jl::get_state`):
- The bond vector r_ij is one distinguished "particle".
- Environment atoms k come from atom i's neighbour list and are cut by `CylindricalBondEnvelope`, centred at the bond midpoint.
- Each product contains exactly one bond factor, and environment factors have capped n and l (`_filter_offsite_be`).
- Bond reversal H_ji(−T) = H_ij(T)ᵀ is not built in. Instead, `DUAL_BASIS_MODEL` fits a second model on transposed data and averages the two.

**Fitting.** Per sub-model LSQR with a fixed ridge, λ=1e-7 times `scaling(basis,2)`. The on-site overlap is fixed to the identity.

**Stack.** ACE.jl 0.12.22, JuLIP and Julia 1.6. Active 2022-07 to 2023-04, dormant since.

**Decision: dropped.** No exporter and no legacy Julia environment. Accuracy is proven by fresh fits on reference data (§6).

## 2. Data: Quoll.jl

**What it is.** Quoll.jl (maurergroup, MIT; about 176 commits since 2025-09; last activity 2026-07; one author does most of the work).
- It loads, converts and post-processes atomic-orbital operators such as H and S.
- It uses one internal canonical representation as a pivot between formats: FHI-aims (CSC real-space `rs_hamiltonian`/`rs_overlap`, `basis-indices.out`, `geometry.in`) and DeepH (the format used by MACE-H and DeepH-E3).
- It also provides:
  - spherical-harmonics convention tracking, `SHConvention` (a per-ℓ signed permutation; stored internally in the "wiki" convention);
  - sparsity patterns, `BlockRealSparsity` (atom pair → image list) and the new `SubBlockRealSparsity` (per shell pair);
  - core-orbital projection (`FC99V`, `LaikovCore`);
  - Fourier transform to a k-grid, spin, and symmetry-reduced k-grids through Spglib;
  - a TOML/MPI CLI.
- Tests: unit tests, Aqua checks, and MPI regression tests on a 221 MB artifact (FHI-aims water, SiC, gold; DeepH graphene and Bi₂Te₃).

**Maturity.** Pre-release (0.1.0) and not registered; install from git. No commits since 2026-07-26. It requires **Julia ≥ 1.12**. That no longer touches ace-jax's own Julia: ace-jax has no runtime Julia environment, and the coupling library builds separately on Julia 1.13 (§3). Quoll runs in its own Julia project.

**Three ways to consume it.** Revision 2 had (a) and (b). (c) is new, because ace-jax no longer hosts a Julia environment that (b) could share.

- **(a) File-based: the DeepH layout Quoll writes.**
  - The layout is Python-native:
    - `hamiltonians.h5` / `overlaps.h5`, one row-major dataset per atom pair, keyed `"[Rx, Ry, Rz, i, j]"` (1-based atoms);
    - `element.dat`, `lat.dat`, `site_positions.dat`, `orbital_types.dat` (ℓ per shell per atom), `R_list.dat`, `info.json`.
  - An `h5py` reader of about 150 LOC turns it into per-(z₁,z₂,shell pair) block arrays, the natural training layout.
  - Predictions are written back in the same layout (`hamiltonians_pred.h5`).
- **(b) In-process through juliacall.**
  - Call `load_operators` → `convert_operator(…, CanonicalBlockRealMetadata, …; out_shconv=…)`, then hand the per-species-pair block arrays across to numpy.
  - Predictions go back through `Quoll.wrap_data` and are written in any format Quoll supports.
  - Revision 2 planned this in one Julia 1.12 environment shared with ET. That environment no longer exists: juliacall and juliapkg were removed when the coupling library replaced them. (b) would bring juliacall back as a dependency used only for ingestion.
- **(c) A standalone Julia ingestion script** (new). A small Julia project in the repo, `ham/julia/` (a `Project.toml` + `Manifest.toml` pinning Quoll by commit, like `coupling/julia/` and `julia/`), run as `julia --project=ham/julia ingest.jl <inputs> <out>`.
  - It does exactly what (b) does on the Julia side: `load_operators`, `convert_operator(…; out_shconv=<ace-jax convention>)`, core projection, sparsity.
  - It writes the frozen training cache (npz per (z₁,z₂,shell pair) blocks in ace-jax's SH convention, plus the Quoll commit and conversion options) directly.
  - Python never starts Julia.

**Weighing them:**

| | (a) DeepH files | (b) juliacall + Quoll | (c) Julia ingestion script |
|---|---|---|---|
| Environments | Python only | juliacall back in the Python environment, plus a juliapkg-provisioned Julia 1.12 project with Quoll (MPI.jl and HDF5.jl JLLs). ET is no longer in it, so the WignerD conflict does not arise. | A separate Julia ≥ 1.12 project for one preprocessing step. Python stays Julia-free, which is the current rule (Julia only for building the coupling library and in parity CI). |
| Maturity exposure | A format (stable, shared with DeepH-E3/MACE-H) | An unregistered pre-release API, inside the Python process | The same API, but only in an offline step whose output is frozen |
| Conventions and units | ace-jax must re-implement DeepH's m-reordering and check units, i.e. duplicate `SHConvention` logic in Python | Quoll does it: `out_shconv` straight to ace-jax's convention, aims signs and core projection included | As (b) |
| Consumer fit | Consumers read files through Quoll anyway | The DFT-injection consumer (§8) already runs Quoll under juliacall, so an in-memory hand-off avoids a disk round trip | Files (npz in, DeepH out); the consumer reads DeepH with Quoll |
| Formats beyond DeepH | Only what someone has already converted to DeepH | Every Quoll input (FHI-aims directly, future ONETEP/psinc sources for §5's learned orbitals), plus core projection and k-grids | As (b) |
| Testing and CI cost | Small committed DeepH fixture; pure-pytest; no Julia in the main test job | A juliacall job in Python CI; precompiling about 200 packages | A path-gated Julia job (`setup-julia`, cached depot), like `julia-parity`. The test-data artifact is 221 MB, so fixtures must be subset. |

**Recommendation: a hybrid of (c) and (a).**
- Use **(c) for conversion and ingestion** from raw code output (FHI-aims now, other sources later), so conventions, units, core projection and sparsity are handled by Quoll alone.
  - Emit a **frozen training cache**: npz per-species-pair blocks in ace-jax's own SH convention, plus the Quoll commit.
- Use **(a) DeepH files as the stable exchange format**, for data from other people and for consumers that read files.
  - Keep a pure-Python DeepH reader and writer, validated against (c) in the path-gated Julia job.
- Fitting, evaluation and the main test job then stay Julia-free, and the unregistered API is touched only at ingestion.
- Keep (b) as a consumer-side option only: a caller that already hosts Quoll under juliacall can pass numpy block dicts in and out (§8). ace-jax does not depend on juliacall for it.
- Pin Quoll by commit in `ham/julia/Manifest.toml` until it is registered. Revision 2's wait for an ET release without the WignerD dependency is over (v0.5.2), and it no longer matters for route A because ET is not in that project.

**Conventions are pinned by Quoll.**
- ace-jax's real Y_lm (`eval/harmonics.py`, SpheriCart/P4ML, no Condon–Shortley phase) must be mapped to Quoll's wiki convention. That mapping is a per-ℓ `SHConvention`.
- DeepH reorders m (for example ℓ=1 → [3,1,2]). FHI-aims flips signs relative to wiki (for example ℓ=1 phases [1,1,−1]); Quoll handles that.
- A one-off test must establish the ace-jax ↔ wiki map. It is probably the identity up to signs. The test:
  - rotate a structure;
  - evaluate the model's equivariant features;
  - check H(QR) = D(Q) H D(Q)ᵀ on Quoll-converted data.
- Unit handling (eV/Å against atomic units) is also Quoll's; check it once.
- Requests to Quoll:
  - a documented canonical on-disk writer (there is none; DeepH is the de facto exchange);
  - registration;
  - a small Python reader shipped with it, or blessed by it.

### 2.1 Investigation: Quoll compiled with `juliac --trim` (option (d))

**Why.** A trim-compiled Quoll would do for ingestion what `ace-jax-coupling` does for couplings: a platform wheel exposing a C ABI (`ingest(fhi_dir | deeph_dir, out_shconv, …) → block buffers`), called by ctypes. Users could then turn FHI-aims or DeepH output into training data with no Julia install, and the path-gated Julia ingestion job would become an ordinary wheel test. Option (c) stays the plan until this is shown to work.

**Initial probe (2026-10-15), result: not feasible with Quoll as it is.**
- Setup: Quoll `main` @ f26de28, Julia 1.13.1, JuliaC (`--output-exe --trim=safe --experimental`, as `coupling/julia/build.jl`), on moriarty. The entry point is the narrowest useful path, DeepH → canonical → DeepH, which avoids FHI-aims and k-grids:
  ```julia
  using Quoll
  function quoll_roundtrip(indir::String, outdir::String)::Cint
      kind = Quoll.Hamiltonian(; source = :ref)
      H  = Quoll.load_operator(Quoll.Operator, Quoll.DeepHBlockNoSpinRealMetadata, indir, kind)
      Hc = Quoll.convert_operator(Quoll.KeyedOperator, Quoll.CanonicalBlockRealMetadata, H; hermitian = true)
      Hd = Quoll.convert_operator(Quoll.Operator, Quoll.DeepHBlockNoSpinRealMetadata, Hc; hermitian = false)
      Quoll.write_operators(Quoll.DeepHBlockNoSpinRealMetadata, outdir, [Hd])
      return Cint(0)
  end
  (@main)(args::Vector{String})::Cint = quoll_roundtrip(args[1], args[2])
  ```
- In ordinary Julia 1.13.1 the round trip works and is bit-identical on Quoll's SiC DeepH test data (2532 blocks, max |Δ| = 0). So Quoll runs on 1.13.
- **Compile 1, Quoll unchanged: 949 trim verifier errors** (34 s; 1.2 GB peak RSS). About 800 of them come from loading PythonCall and CondaPkg, whose `__init__` runs although the DeepH path never calls Python. Quoll loads them unconditionally through `using ASEconvert`, which it needs only to read FHI-aims `geometry.in` (`operators/fhiaims.jl:187`, `ase.io.read`). Building also installs a conda Python with ASE at precompile time.
- **Compile 2, ASEconvert removed** (`using` dropped, the FHI-aims geometry read stubbed): **147 errors.** Of these, 54+8+2 come from HDF5's and 8 from MPI's own code (init and generic `read`), 20 from untyped `readdlm`, and 44 from Quoll's load path at 16 sites:
  - the metadata types passed in are `UnionAll`s (`DeepHBlockNoSpinRealMetadata` is `RealMetadata{O,X,S} where …`), so `load_metadata`, `build_operator`, `convert_operator` and `write_operators` dispatch on abstract types;
  - values are lifted into types at run time: `statictuple(kind)` builds `Pair{Val{tag},Val{value}}` from the operator's tags (`core/components/operatorkind.jl:115`); `SHConvention{lmax}` and `make_static(order, Val(l))` come from vector lengths (`shconvention.jl:31-36`); the element type is read from the file, `V = typeof(read(first(io)))` (`deeph.jl:135`);
  - `@memoize` caches keep results in `IdDict{Any,Any}` (`load_atoms`, `BlockRealSparsity`, `BasisSetMetadata`);
  - atoms are built through Unitful `periodic_system` from untyped `readdlm` output (`deeph.jl:160-168`).
- **147 is a lower bound.** The verifier stops at an unresolved call, and the two `convert_operator` calls and `write_operators` in the entry are themselves unresolved. So the conversion internals (sparsity, SH-convention permutation, Hermitian folding), which are the reason to use Quoll, were never analysed.

**What it would take.** This is not a set of local trim fixes, as ET's L = 0 path was. Quoll's design carries metadata in types and decides those types from data, so trimming it means changing its API:
1. Move ASEconvert (and with it PythonCall/CondaPkg) and MPI into package extensions (weak dependencies), and add a pure-Julia `geometry.in` reader so FHI-aims input does not need ASE. This is small, useful to Quoll anyway (load time, no conda at precompile), and a reasonable upstream request.
2. A type-stable ingestion entry point for a closed set of cases (FHI-aims CSC or DeepH in, no spin, real; canonical out in a fixed SH convention): concrete metadata types, no `Val` lifting of data, no `IdDict{Any,Any}` caches, typed `readdlm`/HDF5 reads. Either a refactor of Quoll's load and convert path, or a separate trim-safe function set in Quoll that the general API calls into.
3. Trim-safe HDF5 (its `generic_read` and init are flagged here), or reading the HDF5 datasets through the HDF5 C library directly.
4. The C ABI, the wheel build and pruning (reusing `coupling/tools/`), and parity against ordinary Quoll on its regression data.

Item 1 takes days. Items 2–3 depend on Quoll's maintainers accepting the change and are estimated in weeks, with the conversion-path errors still unknown. **Recommendation:** keep (c) for phase 0. Make item 1 a request to Quoll now, since it costs Quoll little and removes Python from the ingestion environment. Repeat the probe when Quoll is registered or its API settles: the probe is the script above plus `JuliaC.main(["--output-exe", "quollprobe", "--project=.", "--trim=safe", "--experimental", "--bundle", "bundle", "probe.jl"])`; count `^Verifier error` in the log and group by the first non-Base frame.

## 3. EquivariantTensors: L>0 couplings in the compiled library

**What ET provides** (checked against v0.5.1 in revision 2; unchanged in v0.5.2):
- **L>0 couplings.**
  - `symmetrisation_matrix(L, mb_spec; …)` and `sparse_equivariant_tensors(LL=…)` give an `SVector{2L+1}`-valued A2B over one shared AA basis.
  - `O3.cgmatrix(L1,L2,L; basis=real)` and `TYVec2YMat(L1,L2)` map stacked L-vectors to a (2ℓ₁+1)×(2ℓ₂+1) block.
- **Pseudo-tensors (0.5.x).** `coupling_coeffs(L, ll, nn; refl_sym=:sym|:asym)` selects the parity of Σl independently of L. Verified:
  - ll=(1,1), L=1, `:sym` gives 1 function (an axial vector);
  - ll=(1,2), L=2, `:asym` gives 1;
  - ll=(1,1,1), L=0, `:asym` gives 1 (a pseudoscalar).
- **Real-SH couplings are orthonormalised** (0.5.x). A caveat, from ET's own commit history: real-basis couplings can come out purely imaginary, which `assemble_U` detects.

**What ace-jax has now.** `ace-jax-coupling` 0.2.0 (`coupling/julia/src/ETCouple.jl`, C ABI v1) exposes one entry point, `etc_couple`, for the **L = 0, real-basis** coupling only. It calls the fork's type-stable API, `symmetrisation_matrix(Val(0), mb; PI=true, basis=real, nullspace_solver=:dense)`. The fork (four commits on ET `main` after v0.5.1, proposed upstream as one PR) made **only the L = 0 path** trim-safe and tests only that in its `juliac --trim` CI. Two gaps follow:
- the `Val(L)` method of `symmetrisation_matrix` does not take or forward `refl_sym` (the `Integer` method forwards it through `kwargs...`; `coupling_coeffs(Val(L), …)` accepts it);
- `TVAL = SVector{2L+1,Float64}` for L>0 has never been compiled under `--trim=safe`.

**Work for L>0 and pseudo-tensors (route A).**
1. **Fork.** Forward `refl_sym` in `symmetrisation_matrix(Val(L), …)` (one line, worth upstreaming in the same PR). Add L ∈ {1,…,L_max} and `refl_sym` cases to `test/trim/entry.jl` and fix whatever trim reports. L is a compile-time constant in the `Val` API, so the library dispatches over a static table of L values, as it already does for correlation order (`MAX_STATIC_ORDER = 8`). L_max = 2·ℓ_max of the orbital basis: 4 for spd, 6 with f shells.
2. **ABI v2.** A new entry point, `etc_couple_equivariant(L, refl_sym, …)`, in the same caller-owned-buffer style. It returns A2B triplets with an extra m index (row, col, m, val), plus the real CG tables for each (ℓ₁,ℓ₂,L) the orbital basis needs (or compute those in Python, which is small and has a closed form). Keep `etc_couple` unchanged and bump `etc_abi_version`; the `ace_jax_coupling` loader (`_loader.py`) already checks it.
3. **Release.** A new `ace-jax-coupling` wheel on all four platforms through `coupling-wheels.yml` (Julia 1.13.1, prune, bundle check, clean-env test), then a core-dependency bump. The coupling cache key adds L and `refl_sym`, and `backend_id()` changes with the version, so old entries are never misread.
4. **Parity.** `coupling/python/tests` compares against unpatched upstream ET at the fork's base (index arrays identical, A2B within a few ulp); extend `coupling/julia/reference/` to L>0 and both parities. Span-based checks (`coupling.subspace_residual`) remain the test for degenerate blocks.

Size: about 1–1.5 weeks, most of it the trim fixes, whose extent is not known until the compile is tried. The risk is in step 1 (§11).

**Revision 2's pin-bump costs are settled.** The library already builds from ET `main` after v0.5.1: its parity against the ACEpotentials fixtures (which pin ET 0.4.3) is per nnll block span, with a different per-row normalisation (`docs/dev/coupling-etshim-spec.md`, "ET version vs the ACEpotentials fixtures"). The committed cache and `ACEJAX_COUPLING_CACHE_ONLY=1` replaced `juliapkg.json` hashing and `ACEJAX_NO_JULIA`. Nothing about ET versions remains to be decided for potentials.

**Julia versions now.** `coupling-wheels.yml` uses Julia 1.13.1 (exact; `build.jl` checks it against the Manifests). `julia-parity.yml` and `prior-parity.yml` stay on 1.11 for the ACEpotentials reference project. Quoll needs ≥ 1.12, so the ingestion project of §2 uses 1.13 too, to keep two Julia versions in CI rather than three.

## 4. Mapping onto ace-jax

| Piece | Verdict |
|---|---|
| Real Y_lm, A/AA product, `pool_a_sparse`/`pool_a_dense`, `aa_specs` layout, the recursive AA DAG (`eval/edge_model.py`, `eval/model.py`, `eval/aa_dag.py`) | **Reuse.** The on-site environment is the site environment. The AA layer is shared by all L, so only the A2B contraction is new. |
| Radials (`eval/radial.py::poly_recursion`, `basis/radial_ace1.jacobi_3term`, splines and radial tables in `eval/splinify.py`) | **Reuse.** ACE.jl's exact radials no longer matter, since old models are dropped. Learned radials (`fit/radial_learn.py`) are a later option for the block fit. |
| Species and bond/env category | **Reuse the channel fold** of `basis/spec.py::build_embedding_spec` (n = (n′−1)d+k). ET symmetrises only equal (n,l), so a bond factor with its own n is correctly never permuted with environment factors. |
| L>0 and pseudo-tensor couplings | **Compiled-library extension (§3).** Fork trim fixes for L>0 and `refl_sym`; ABI v2 `etc_couple_equivariant` returning per-L A2B triplets (row, col, m, val); a new `ace-jax-coupling` release; `basis/coupling.py::couple_equivariant` with L and parity in the cache key. Parity tests against unpatched ET. |
| Equivariant readout | **New.** `block = Σ_L CG_L · (c_Lᵀ B^L)`, one einsum. |
| Bond environments | **New.** Build (bond i→j in cell T) × (env k) triplets from the existing `eval/nlist.py` list, with env vectors relative to the bond midpoint and a cylindrical envelope. Evaluate the features on (i,j) and on the reversed bond and symmetrise with the CG transpose sign (−1)^(ℓ₁+ℓ₂−L). This makes H_ji(−T) = H_ij(T)ᵀ exact, with no dual model. |
| Operator assembly | **New.** Blocks → sparse H(T), S(T) keyed like DeepH → Bloch sums H(k) = Σ_T e^{ik·T} H(T). All in JAX, all differentiable. |
| Basis building | **Reuse the pattern of `basis/model.py::build_basis`** (spec → coupling cache → frozen zero-readout model) for a `build_ham_basis`. The per-shape coupling cache, `BasisUnavailable` and the committed cache for `ACEJAX_COUPLING_CACHE_ONLY=1` tests carry over. |
| Saved model | **Extend the npz + `meta_json` schema** (`basis/export.py::save_npz`, `eval/io.py::load`): per-(sub-model, L) A2B and coefficients, CG tables, orbital-basis definition. A new top-level key (as `gp_json` marks a GP) lets `load` dispatch. |
| Force/virial path (`EdgeSiteModel.energy_forces_virial`) | **Does not apply.** It assumes a scalar energy. ∂H/∂R is available by autodiff when needed. |
| Lean form, dense blocked layouts, skin lists (`eval/model.py::lean`, `calc/skin.py`) | **Reuse in phase 3.** `prune_columns` and spline/radial-table radials apply unchanged to the on-site basis; `block_dense` needs a per-L variant. The skin list applies to MD-style consumers that call the model every step. |

## 5. Fitting with spectra and observables in the loss

**Stage A: block loss (convex).**
- Targets are the block entries themselves: linear in the coefficients, with no Jacobians needed.
- The real CG stack is orthogonal, so projecting each target block onto L, y_L = CG_Lᵀ·H_block, decouples the problem into independent vector-valued regressions per (sub-model, L, parity), with 2L+1 rows per sample.
- `fit/solve.py` (streamed QR and LSQR) and the BLR evidence in `fit/objective.py` (σ, σ_c, prior Γ) apply once the row generator is generic. Today `fit/rows.py`/`stats.py` hard-wire E/F/V rows from site descriptors and edge Jacobians (and `fit/stats.py` the per-quantity E/F/V types).
- **Use the QR evidence path** (`lml_solver="qr"`, `stats.linear_qr_statistics` → `QRStats`, `objective.log_marginal_likelihood_qr` / `posterior_from_qr`). DFT blocks are noise-free up to SCF tolerance, so block fits will nearly interpolate, which is exactly the κ(G+Λ) ~ 1e16 regime where the Cholesky evidence is NaN or wrong. The batched merges and the CPU host QR stream apply unchanged; the per-quantity statistics become per (sub-model, L, parity), with one σ per quantity as now.
- Many small independent problems (one per sub-model × L × parity) suit a `vmap` over problems of equal width, bucketed like the edge buckets, rather than one jit per problem.
- Stage A gives initialisation, evidence-tuned regularisation, and a Gaussian posterior (§7).

**Stage B: observable loss (non-convex).**
- The loss is on eigenvalues, band energies, gaps or density matrices from the generalised problem H(k)c = εS(k)c. Plain LSQR is no longer enough.
- Minimise `L = w_blk·‖H − Ĥ‖² + w_spec·Σ_k w_k ℓ(ε(k), ε̂(k)) + …` over the coefficients with L-BFGS or Adam: `fit/radial_learn.lbfgs_loop` and optax (a core dependency). Warm-start from stage A. The block term can stay a QR-compressed quadratic (R_q, c_q), so only the spectral term needs per-step structure evaluations.
- Keep the block term as a regulariser. Spectra alone under-determine H, because any unitary rotation within the occupied space leaves them unchanged.
- **Generalised eigenproblem.** Reduce with `L = cholesky(S(k))` and `H̃ = L⁻¹HL⁻†`, then `eigh`. All of it is differentiable in JAX, complex Hermitian included. Checked in JAX 0.11.2 with float64 at an exact twofold degeneracy:
  - gradients of eigenvalue-only functions (sums over bands, band energy) are finite and correct for functions symmetric within the degenerate group;
  - a single degenerate eigenvalue's gradient is finite but depends on the basis choice, so it is meaningless;
  - **anything through eigenvectors, such as a density matrix, gives NaN**.
- **Required: a `custom_vjp` for spectral functions** F = V f(Λ) V†, including the density matrix and smeared occupations. Use Daleckii–Krein divided differences, (f(λᵢ)−f(λⱼ))/(λᵢ−λⱼ), which tend to f′(λ) as λⱼ → λᵢ. They are finite at degeneracies and need no eigenvector-derivative term.
- Losses on individual eigenvalues must be defined per degenerate group (group mean or sum), or through smooth occupation functions. Use Fermi–Dirac or Gaussian smearing, which Quoll also implements.
- **k-sums.** `vmap` over the k-points of a structure with weights. Symmetry-reduced grids come from Quoll/Spglib. Memory is n_k·n_orb² complex per structure.
- **Batching.** Bucket structures by padded orbital count, as `ACECalculator` pads edges to power-of-two buckets. `eigh` cost is O(n_orb³) per k.
- For large cells, restrict the spectral term to a few k-points or an energy window. Or use eigenvalue perturbation around the reference eigenvectors: ε̂ ≈ ε_ref + v_ref†(ΔH − ε_ref ΔS)v_ref, which avoids eigh entirely on most steps.
- **UQ in stage B.** The model is non-linear in its loss, so the BLR posterior no longer holds exactly. Use Laplace or Gauss-Newton around the optimum, or keep stage-A BLR and calibrate conformally on spectra (§7).

**Custom and learned orbitals** (for example NGWF-like orbitals).

Design commitments now:
- **Orbital basis as data, not code.** Sub-models are keyed by (species, shell index) from a basis definition carried in the model file. There are no hard-coded shell tables.
- **Differentiable contraction.**
  - A learned orbital set is a contraction of a larger reference basis χ: φ = Cχ, per species.
  - C is block-diagonal in ℓ and identical across m, which preserves equivariance by construction.
  - Then H_φ = CᵀH_χC and S_φ = CᵀS_χC. This is exactly Quoll's core projection, generalised and made differentiable.
  - The data layer should therefore keep the reference-basis blocks and apply the contraction in JAX.
- **Cheap VarPro.**
  - The ACE features depend only on geometry, not on C. For fixed C, the stage-A design matrix is unchanged and only the right-hand side CᵀH_χC changes.
  - The outer loop over C can reuse one QR factorisation and differentiate through the inner solve, as `fit/radial_learn.py` does for learned radials (its QR path is in production in `aj fit --learn-radial`; `fit/varpro.py` is a research driver exercised only by its tests).
  - The outer objective must be orbital-invariant, such as spectra in a window or model error normalised per orbital. A block loss alone rewards trivial orbitals.
- In-situ-optimised, environment-dependent orbitals such as ONETEP NGWFs need the operator in the underlying (psinc) basis. That is a data question for Quoll (a new source format), not a model question.

## 6. Validation without old models

1. **Pure JAX structural tests:**
   - rotation and inversion equivariance of each block;
   - exact bond-reversal symmetry;
   - Hermiticity of H(k);
   - autodiff against finite differences, including the degenerate spectral `custom_vjp`.
2. **Coupling parity:** the compiled library's ABI v2 against unpatched ET, both L>0 and pseudo-tensor (`coupling/python/tests`, index arrays identical, A2B within a few ulp), plus `subspace_residual` for degenerate blocks.
3. **Convention test** (§2): equivariance on rotated structures converted by Quoll.
4. **Accuracy on reference data, fresh fits:**
   - Quoll's regression data and small FHI-aims sets (water, SiC, gold);
   - ACEhamiltonians' own contrived Si HDF5 (160 kB), as data only.
   - Metrics: block RMSE, eigenvalue and band RMSE in a window, and SCF iterations saved by a downstream injection code.
   - Comparators: numbers published for ACEhamiltonians, and MACE-H or DeepH-E3 trained on the same Quoll-produced data.
5. **CI:**
   - a small committed DeepH-layout fixture (a few MB, under the 4 MB `check-added-large-files` hook limit) and a committed coupling cache for the ham bases, so the main suite runs with `ACEJAX_COUPLING_CACHE_ONLY=1` and needs no new library release to test the model code;
   - ham tests are whole files, so `.test_durations` must be refreshed when they land (the shards balance whole files);
   - a path-gated `ham-ingest` Julia job (§2 (c)): runs the Quoll ingestion script on a pinned subset of Quoll's artifact and checks the Python DeepH reader against it;
   - a path-gated `ham-accuracy` job on the frozen npz caches, with `ACEJAX_REQUIRE_FIXTURES=1`;
   - Julia elsewhere only in the existing `coupling-wheels` job (extended to L>0).

## 7. Uncertainty: BLR, POPS and conformal

**Stage-A BLR.** One posterior per (sub-model, L, parity): `objective.posterior_from_qr`, `log_marginal_likelihood_qr` and the Γ prior, with generic rows (§5).
- The predictive covariance of a block's L-part is B_Lᵀ Σ B_L. That is (2L+1)×(2L+1) and rotation-covariant.
- Element-wise σ follows through the CG map. The port can therefore give a *properly equivariant* per-block covariance.
- That is stronger than element-wise variances, which are not rotation-invariant and need heuristic averaging downstream.

**POPS** (`fit/pops.py`) carries over, applied per-observation and pointwise, for misspecification.

**Template from `--uq ard` (new since revision 2).** ace-jax now serves a calibrated per-atom 3×3 force covariance (`docs/dev/ard-force-uq.md`, `docs/dev/specs/2026-09-30-conformal-force-sigma-design.md`):
- the **shape** is a centred delete-one-cluster jackknife covariance (`fit/jackknife.py`), robust where the BLR posterior is over-confident under misspecification;
- the **scale** is per Mondrian group, configuration-weighted, from a stratified hold-out split (`fit/conformal.py`): an rms factor and a conformal quantile of Mahalanobis scores, with χ₃ as the Gaussian reference;
- `aj calibrate` re-pools a group's scores with target-regime data.

A force is an L=1 vector, so this carries over to each (2L+1)-vector block part almost unchanged: χ_{2L+1} replaces χ₃, the jackknife clusters are structures, and the Mondrian groups become (species pair, bond-length band) or (sub-model, L). `fit/conformal.py`'s grouping and split code is generic already; only `shell_features` is force-specific. The block-level calibration should come first, because it is what the DFT-injection consumer reads (§8).

**Conformal on observables.**
- Split-conformal calibration of the propagated eigenvalue σ: the score is Δε/σ_ε on held-out structures, and the calibration weight is its (1−α) quantile.
- Propagate element-wise with first-order perturbation theory, σ²_ε ≈ Σ |∂ε/∂H|² σ²_H. Or propagate exactly through the linearised spectral map, using autodiff in JAX.
- This fits naturally in an application module, and matches what DFT-injection consumers already do.

## 8. Consumers

**Python/JAX consumers.** An in-memory `HamModel.predict(atoms) → (H(T), S(T), σ)`, plus `h_k(k)`, `bands(kpath)` and `dos(...)`, all jit-able and differentiable.

**DFT injection.** A private DFT-injection prototype (example consumer) shows what injection needs:
- real-space H(T) (and S(T) if not taken from the code) with the uncertainty in a **Quoll-readable format**, with DeepH as the practical choice;
- block-wise sparsity with per-shell-pair cutoffs, so the consumer can mask to its code's own support;
- consistent units and SH convention, handled via Quoll;
- a per-element standard-deviation operator in the same layout.

The port meets this by writing `hamiltonians_pred.h5` and an uncertainty twin in DeepH layout, plus its metadata files. For callers that already host Quoll under juliacall, it can also hand over an in-memory numpy block dict, which the caller wraps with `Quoll.wrap_data` in its own Julia environment (§2 (b), consumer side). ace-jax itself never starts Julia, at prediction or at fit time.

**ASE.** `ACECalculator` and `GPCalculator` (`calc/`) are energy calculators; a Hamiltonian calculator exposing `results["hamiltonian"]`, `results["overlap"]` and their uncertainty is a new class in `ham`, reusing `eval/nlist.py` and the power-of-two edge buckets, not a subclass.

## 9. Framework and application boundary

**Proposal.** This mirrors the Julia stack (EquivariantTensors/Polynomials4ML as framework; ACEpotentials/ACEhamiltonians as applications).

- **Framework** (eventually its own distribution). In today's tree it is:
  - `eval/`: `harmonics.py`, `radial.py`, `splinify.py`, `nlist.py`, `aa_dag.py`, and the A/AA product and layouts in `edge_model.py`/`model.py` (to be separated from E/F/V), plus triplet graphs (new);
  - `basis/`: `spec.py`, `coupling.py` (and the `ace-jax-coupling` library), `prior.py`, `export.py`;
  - generic fitting primitives in `fit/`: `stats.py` (QR statistics), `objective.py`, `solve.py`, `hostcache.py`, `pops.py`, `conformal.py`, `jackknife.py`, `radial_learn.lbfgs_loop`, once their E/F/V assumptions are factored out;
  - the npz schema helpers (`eval/io.py`).
- **Applications:**
  - potentials: `ACEModel`/`PACEModel` site energies, E/F/V rows and the GP arm (`fit/kernels.py`, `inducing.py`, `ladder.py`), `calc/`, `export/` (LAMMPS, `.yace`), `fit/pipeline/`, `cli.py`, `runfile.py`, `tutorials/`;
  - `ace_jax.ham`: bond basis, equivariant readout, Quoll/DeepH IO, spectral losses, operator UQ, consumer IO.

**Recommendation: start in-tree, with an enforced boundary, and do not rename packages first.**
- The tree has grown since revision 2 (CLI, run files, tutorials, LAMMPS and `.yace` export, docs site, a user skill). Moving everything under `ace_jax.core`/`ace_jax.potentials` up front would churn every import, the docs and `skills/ace-jax/SKILL.md` for no gain to `ham`.
- Instead, add `ace_jax.ham` and enforce with an import-linter contract (or a test like `tests/test_core_deps.py`) that `ham` imports only the framework modules listed above, and that those modules import nothing from `ham` or the potential-only modules. Rename into `core` when the distribution is split.
- Split distributions later, once `core`'s API has settled.
- The first refactors are what make the boundary real:
  - move E/F/V out of `EdgeSiteModel`;
  - make `fit/rows.py`/`stats.py` generic over row generators.
  - They are far cheaper in one repo than across a versioned package boundary.
- A separate package from day one would freeze today's energy-specific internals as its API.
- Dependencies: DeepH IO needs `h5py`, which is not a core dependency. Ship `ham` behind a `ham` extra and import `h5py` inside the functions that need it (`tests/test_core_deps.py` enforces this).

## 10. Revised phased plan

The sizes are rough, for one developer who knows ace-jax.

| Phase | Content | Size |
|---|---|---|
| **0. Foundations** | L>0 and `refl_sym` couplings in the compiled library: fork trim fixes, ABI v2 `etc_couple_equivariant`, parity against unpatched ET, an `ace-jax-coupling` release (§3; 1–1.5 wk). The model work does not wait for the release: develop against a local build via `ACEJAX_COUPLING_LIB` and commit the coupling cache. Introduce the `ham` import boundary; move E/F/V out of `EdgeSiteModel`. Add the pure-Python DeepH reader and writer; the Quoll ingestion script (`ham/julia/`, git-pinned Quoll, Julia 1.13) writing a frozen npz training cache; a path-gated Julia job checking that the two agree; the ace-jax ↔ Quoll SH-convention test. (Revision 2's ET pin bump, Julia range change and coupling re-baselining are done.) | 3–4 wk |
| **1. Equivariant model and assembly** | On-site and bond models (triplet graph, cylindrical envelope, symmetrised bond basis, per-L readout). H(T)/S(T) → H(k). Structural test suite. npz schema. | 4–5 wk |
| **2a. Stage-A fitting and UQ** | Generic rows. Per-(sub-model, L) QR statistics and QR evidence, vmapped over equal-width problems. Block-covariance UQ, the jackknife shape and Mondrian conformal scales (§7), POPS. Accuracy benchmarks on Quoll data against MACE-H/DeepH-E3. DeepH-format prediction output for consumers. | 4–5 wk |
| **2b. Observable fitting** | Degeneracy-safe spectral `custom_vjp`. Generalised eigenproblem. k-batching and bucketing. Combined loss with L-BFGS/Adam warm-started from 2a. Conformal calibration on eigenvalues. | 4–6 wk |
| **3. Performance and deployment** | Dense (bond, K) layouts, reusing `lean` pruning, spline radials and radial tables, and the row-blocked dense evaluation. Batching sub-models by (ℓ₁,ℓ₂) to bound jit variants. Sparse operator output. GPU k-point batches. A `jax.export` artifact. | 3–4 wk |
| **(side) Trim-compiled Quoll** | Investigation only (§2.1): ask Quoll to make ASEconvert and MPI package extensions; repeat the probe when Quoll's API settles; if the conversion path can be made trim-safe, ship ingestion as a wheel, like `ace-jax-coupling`. Not on the critical path. | probe done; open |
| **4. Learned orbitals (research)** | Reference-basis data through Quoll. Differentiable contraction C. VarPro outer loop. Orbital-invariant objectives. | open |

**Total to a usable, UQ-carrying, observable-trained model: about 5–6 months.** Phases 0–2a, about 3 months, give a working block-fitted model with UQ that consumers can use.

## 11. Risks

1. **SH convention and units at the Quoll boundary.** A wrong sign fits well but breaks equivariance. Mitigated by the rotation test in phase 0.
2. **L>0 under `juliac --trim=safe`.** Only the L = 0 path has been made trim-safe. `SVector{2L+1}` values, a static table of L and the `refl_sym` branch may each hit trim failures, and the purely-imaginary real-coupling edge case must be reported through the C ABI rather than thrown. Mitigated by trying the compile first in phase 0 (before any model code depends on it); fallback: generate the L>0 tables offline with ordinary Julia + ET and commit them as a coupling cache, so `ham` works without a library release for fixed basis shapes.
3. **Quoll immaturity.** Pre-release, unregistered, one main author, unchanged since 2026-07, git-pinned in the ingestion project's Manifest.
   - Mitigated by confining the Quoll API to an offline ingestion step, with a frozen npz cache and DeepH files as the exchange format.
   - (Revision 2's co-resolution risk with ET is gone: ET v0.5.2 made WignerD test-only, and route A no longer puts ET and Quoll in one environment.)
4. **Spectral gradients.** Degeneracies; ill-conditioned S(k) in large or diffuse bases (Cholesky); cost of eigh at scale; and a non-convex loss that loses the closed-form evidence.
5. **Bond-environment edge cases.** Periodic self-images, atoms near the midpoint, long bonds; plus the number of jit variants across sub-models × L.
6. **Physics constraints.** S(k) is not guaranteed positive definite, and on-site S = I holds only for orthonormal-on-site bases.
7. **Ingestion keeps a Julia install.** Until a trim-compiled Quoll exists (§2.1, not feasible today), users who ingest raw FHI-aims output need Julia ≥ 1.12 and Quoll. Users who start from DeepH files or a released npz cache do not.

## 12. Open questions remaining

1. **SH-convention anchor.** Should ace-jax emit Quoll's wiki convention natively, or keep SpheriCart's and convert at the IO boundary? The latter is proposed.
2. **Quoll asks.** Is the owner willing to ask Quoll for registration and a canonical on-disk format? Or is DeepH layout the long-term exchange format? (The ET release that makes WignerD test-only is out: v0.5.2.)
3. **Julia in CI.** CI now has 1.13.1 (`coupling-wheels`) and 1.11 (`julia-parity`, `prior-parity`); the ingestion job is proposed on 1.13. Move the ACEpotentials jobs to 1.13 as well, if ACEpotentials supports it, or keep two versions?
4. **Pseudo-tensor inclusion by default.** On, for parity with ACEhamiltonians' completeness; or opt-in, since it adds basis functions?
5. **Stage-B loss priorities.** Which observables come first: eigenvalues in a window, band energy, density matrix, gaps?
6. **Split timing.** Is "in-tree with a lint-enforced boundary, no package rename until the split" acceptable?
7. **ET upstreaming.** Send the `refl_sym` forward and the L>0 trim fixes to ACEsuit together with the pending trim-coupling PR, or as a follow-up PR?
8. **CLI.** An `aj ham` subcommand group (fit, eval, calibrate, export to DeepH) with its own run-file schema, or Python API only until phase 2a?
9. **Quoll and trim.** Is a trim-compiled ingestion wheel worth asking Quoll's maintainers for (§2.1, items 1–2)? Item 1 alone (ASEconvert and MPI as extensions, a pure-Julia `geometry.in` reader) is the low-cost request.

## Appendix A. Route comparison: JAX port (A) against a new Julia ACEhamiltonians on ET (B)

Both routes have the same goal:
- on-site and off-site H/S blocks;
- data via Quoll;
- block fitting with BLR and conformal UQ;
- end-to-end training on spectra through the generalised eigenproblem and k-sums, degeneracies included;
- learned orbitals later;
- Python consumers (an ASE calculator exposing `results["hamiltonian"]` and its uncertainty), plus a private DFT-injection prototype (example consumer).

**Route A** is §§2–10: JAX, with ET only inside the compiled coupling library (basis construction) and Quoll only in an offline ingestion step (§2). **Route B** is a from-scratch Julia package on EquivariantTensors (ET), Polynomials4ML, Lux and ACEfit, using Quoll natively.

### A.1 What the Julia stack gives route B today

The versions checked are the ones on disk plus GitHub, 2026-09; rechecked 2026-10-15 for ET (v0.5.2), Quoll (unchanged) and ACEpotentials (still ET 0.4.3).

**Equivariant bases.**
- ET gives L>0 bases (`sparse_equivariant_tensors`, `SVector{2L+1}` A2B), real CG matrices and `TYVec2YMat`. Pseudo-tensors come from `refl_sym`, in 0.5.x.
- The bases evaluate natively as LuxCore layers (`src/embed/*`, `src/ace/sparse_ace_ka.jl`), with KernelAbstractions kernels for the sparse product and pooling.
- GPU work on ET is recent and active: a fused-scatter pullback and SelectLinL fixes (2026-03), GPU splines, and a "fix gpu grads" merge in 2026-06.
- Radials were split into `lib/ACEradials` (2026-06). **Neither ET nor ACEpotentials uses Reactant** (no references in either repo).

**ACEpotentials 0.10.2.**
- It has an ET/Lux backend ("ETACE", with LuxCUDA GPU benchmarks, 2026-01) and still requires ET 0.4.x.
- Activity since 2026-01 is maintenance: a 13× force-regression fix in 2026-06.
- There is no bond/matrix-valued model to build on; route B writes the bond basis, triplet graph and readout itself, exactly as route A does.

**Fitting and UQ.**
- ACEfit 0.3.1 has `bayesian_linear_regression` (evidence maximisation, ARD, an SVD variant) on dense in-memory design matrices on the CPU.
- There is no streamed or GPU solver and no conformal calibration anywhere in ACEsuit (code search: 0 hits). Route A reuses ace-jax's Mondrian conformal and jackknife code (`fit/conformal.py`, `fit/jackknife.py`, §7); route B would write it new.

**AD through the eigenproblem** (the crux for stage B):
- **ChainRules 1.73** has an `rrule` for `eigen(::Hermitian)`. It divides by (λⱼ−λᵢ) exactly as JAX does, so it is **not degeneracy-safe** for eigenvector paths.
  - Its Hermitian *matrix-function* rules (`exp`, `tanh`, …) use Daleckii–Krein and are degeneracy-safe, so a Fermi–Dirac density matrix written as ½(1−tanh(β(H̃−μ)/2)) works on the CPU.
  - There is no rule for the generalised `eigen(A, B)`. A Cholesky reduction is needed; its `rrule` exists.
  - The rules are typed on `StridedMatrix`, so a `CuArray` gets **no rule**.
- **Enzyme 0.13** has no eigensolver rules. "Add forward and reverse rules for eigen of a real symmetric matrix" is an open PR (#3683), so it falls back on ChainRules import.
- **Mooncake 0.5** has no eigen rule and relies on `@from_rrule`.
- **Reactant 0.2.289** has **no `eigen`/`syevd` lowering** at all (source search), so an eigensolver cannot sit inside a Reactant-compiled region.
- **DFTK** differentiates through self-consistent response (DFPT, forward mode). It has no reusable reverse-mode eigensolver rule.
- **GPU.** CUDA.jl 6.3 wraps `heevjBatched!`/`syevjBatched!` (batched Jacobi eigh, fine for k-batches of small matrices), with no AD rule. Route B would hand-write a degeneracy-safe `rrule` for the generalised Hermitian problem on CPU and GPU. This is the same mathematics as route A's `custom_vjp`, but on top of a less settled AD stack.
- **JAX, for comparison.** `eigh` is differentiable (complex Hermitian included), `vmap`-batched over k, and lowered to cuSOLVER on GPU inside one XLA program. Verified in §5: eigenvalue-only gradients are finite at degeneracies, and eigenvector paths need the same custom rule.

**Quoll.**
- Native in B: canonical `SpeciesPairDictionaryArray` blocks and `SHConvention` are used directly, with no conversion layer.
- ET v0.5.2 (2026-10-02) made WignerD test-only, which removes the StructArrays conflict that kept registered ET 0.5.1 from co-resolving with Quoll (expected from the dependency change; not re-resolved for this revision). B depends on both at **run time**.

**Python consumers in B.**
- They need a juliacall-backed ASE calculator. That is about 1–2 weeks: environment provisioning, array hand-off, GIL and threading, and MPI coexistence.
- Every Python process pays Julia start-up plus package load; ET and Quoll take about 12 s after precompilation, and first-call compilation comes on top.
- For the DFT-injection example consumer this is cheap, because it already runs Quoll under juliacall. For general Python/ASE/MLIP users it is a new dependency.

**Performance.**
- JAX pays XLA compile latency per shape. ace-jax already mitigates it with power-of-two bucketing and the persistent compile cache. It then runs fused forward and backward passes, and k-batched `eigh`, in one device program.
- Julia avoids XLA compile, but pays load and first-call compilation per process. Reverse passes run through Zygote plus hand-tuned KernelAbstractions kernels, with no whole-program fusion (Reactant cannot take the eigensolver).
- No like-for-like benchmark exists for this workload. The block model is small per structure; spectral training is dominated by O(n_orb³) `eigh` × n_k, where both routes call cuSOLVER. The difference is in batching and fusion around it, which favours JAX.

### A.2 Effort per phase (weeks, one developer)

| Phase | Route A (JAX, §10) | Route B (Julia on ET) |
|---|---|---|
| 0. Foundations | 3–4: L>0/`refl_sym` in the compiled library (ABI v2, release), `ham` import boundary and E/F/V refactor, DeepH IO and the Quoll ingestion script, SH-convention test | 2–3: package skeleton, ET L>0/`refl_sym` used directly, native Quoll, convention test (shared) |
| 1. Equivariant model and assembly | 4–5 | 4–5: bond triplet graph, envelope and readout as Lux layers. There is no bond model to reuse; learning the ET/Lux internals is on the critical path. |
| 2a. Block fit and UQ, consumer-ready | 4–5 (generic rows, streamed BLR, block covariance, conformal, DeepH output) | 3–4 (ACEfit BLR per sub-model; conformal new) + 1–2 (juliacall ASE calculator for Python consumers) |
| 2b. Observable training | 4–6 | 6–9: custom degeneracy-safe `rrule`s for the generalised Hermitian `eigen` on CPU **and** `CuArray`, a choice of AD system (Zygote now, Enzyme or Mooncake later), k-batching by hand, Zygote performance through the ACE layers |
| 3. Performance and deployment | 3–4 | 4–6: KernelAbstractions bond kernels, GPU AD robustness, calculator latency |
| 4. Learned orbitals | open | open |

| Milestone | Route A | Route B |
|---|---|---|
| (1) Block-fitted model with UQ, usable from Python | **11–14 wk** | **10–14 wk** |
| (2) End-to-end observable training | **15–20 wk** | **16–23 wk** |

Assumptions:
- Route B's estimates assume a developer already fluent in ET's Lux internals.
- Route A's include the `core` refactor that the planned framework split needs anyway.
- Without the split, A's phase 0 is about 1 week shorter.

### A.3 Assessment

**Faster to a usable result: a tie for milestone (1).**
- B saves the shim, the conversion code and part of fitting: ET evaluates L>0 natively, Quoll is native, and ACEfit BLR exists.
- It gives the time back on the Python calculator.

**Faster to (2), and with less variance: A.**
- Differentiable, batched and GPU-lowered `eigh` is mature in JAX.
- In Julia, every AD system lacks a degeneracy-safe or GPU rule for this exact operation, and Reactant cannot compile it.

**Better long term: A, on balance.** Criterion by criterion:
- **AD and GPU maturity for this workload: A.** The observable loss is the part that is hard to get right, and JAX's single-program model (vmap over k, cuSOLVER, jit) fits it.
- **Consumers and ecosystem: A.** The consumers are Python. The MLIP world, and ace-jax's users, are Python/JAX.
- **Reuse: A.** It reuses ace-jax's streamed stats, QR/LSQR, BLR evidence, POPS, VarPro/L-BFGS and layouts, which are working and tested code. B reuses ET evaluation, ACEfit and Quoll.
- **The owner's direction: A.** A is the planned framework/application split applied to Hamiltonians: `core` plays ET/Polynomials4ML and `ham` plays ACEhamiltonians.
- **Community fit: B.** ACEsuit's developers and maintainers work in Julia. B lands in their ecosystem, and ET improvements (bases, GPU kernels) arrive for free. A tracks ET only through the compiled coupling library, whose C ABI must grow with each ET feature it uses (L>0 and `refl_sym` now), and each growth is a trim-safety fix plus a four-platform wheel release.
- **Pre-release exposure: A's is smaller.** B depends at run time on unregistered Quoll (ET is now released, v0.5.2). A touches Quoll only in an offline ingestion step, and ET only through a pinned, compiled library.
- **The owner's direction since revision 2: A, more strongly.** ace-jax removed Julia from its runtime entirely (no juliacall; a `--trim` library on PyPI). Route B, or a juliacall-backed calculator, runs against that.
- **Maintainability: a split.**
  - B keeps one language for model and data.
  - A keeps one language for model, AD and consumers, and adds a two-language build only where it already has one: authoring, plus optional ingestion.

**Hybrid options.**
- **H1: Julia for construction and ingestion, JAX for everything differentiable.** Pure construction and ingestion work (ET couplings, Quoll conversion to a frozen cache) stays in Julia; evaluation, fitting, spectra, UQ and consumers are JAX. **This is route A as planned (§2).** Its extra cost is the Quoll ingestion script, the coupling-library ABI work (§3) and a path-gated Julia CI job; no Julia at fit or prediction time.
- **H2: basis and assembly in Julia, spectral losses in JAX (or the reverse).** Rejected.
  - Every training step would cross the language boundary with gradients: a `jax.pure_callback` plus `custom_vjp` calling Julia pullbacks through juliacall/DLPack.
  - That loses jit fusion, complicates GPU memory sharing, and makes two AD systems co-responsible for one gradient. It costs more than either pure route and is more fragile than both.

**Recommendation: route A with H1.**

What would change it:
1. The long-term owners or maintainers are ACEsuit Julia developers, rather than ace-jax's users. Community fit then outweighs AD maturity.
2. Reactant gains an `eigh` lowering and ACEsuit adopts Reactant end to end. That would give Julia a single-program, differentiable, GPU path comparable to XLA.
3. Consumers move to Julia, making the Python calculator moot.
4. Learned orbitals turn out to need Quoll's projections *inside* the differentiable loop in ways not reimplementable as a JAX contraction.
5. The ACEsuit plan for Hamiltonians is itself a new Julia package. In that case contributing there and keeping ace-jax to potentials avoids duplicate effort.

If (1) or (5) is true, B becomes the better long-term choice despite its slower and riskier path to observable training.
