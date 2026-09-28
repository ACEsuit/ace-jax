# Scoping: ACE Hamiltonians in ace-jax (a port of ACEhamiltonians.jl)

Status: research and scoping only. No code yet. Revision 2 takes in the owner's decisions:
- old ACEhamiltonians models are dropped;
- the EquivariantTensors (ET) pin may move to the latest release;
- data comes from Quoll.jl;
- spectral and observable losses are in scope;
- consumers are Python/JAX tools and DFT-injection codes;
- UQ is by BLR and conformal prediction;
- a framework/application split is wanted.

Sources read:
- ACEhamiltonians.jl `main` @ 9ff1572;
- ACE.jl v0.12.22, the version ACEhamiltonians pins;
- Quoll.jl `main` @ 2026-07-26;
- EquivariantTensors v0.4.3 and v0.5.1 (the latest registered release, 2026-09-11);
- ace-jax `main` @ 396fee0.

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

**Maturity.** Pre-release (0.1.0) and not registered; install from git. It requires **Julia ≥ 1.12**. That is not a blocker: ace-jax's `juliapkg.json` range `~1.10, ~1.11` was set only for Reactant, which the ET shim does not use, so ace-jax can move to Julia 1.12 (see §3 for what else the pin touches).

**Two ways to consume it.**

- **(a) File-based: the DeepH layout Quoll writes.**
  - The layout is Python-native:
    - `hamiltonians.h5` / `overlaps.h5`, one row-major dataset per atom pair, keyed `"[Rx, Ry, Rz, i, j]"` (1-based atoms);
    - `element.dat`, `lat.dat`, `site_positions.dat`, `orbital_types.dat` (ℓ per shell per atom), `R_list.dat`, `info.json`.
  - An `h5py` reader of about 150 LOC turns it into per-(z₁,z₂,shell pair) block arrays, the natural training layout.
  - Predictions are written back in the same layout (`hamiltonians_pred.h5`).
- **(b) In-process through juliacall, in the same Julia 1.12 environment as ET.**
  - Call `load_operators` → `convert_operator(…, CanonicalBlockRealMetadata, …; out_shconv=…)`, then hand the per-species-pair block arrays across to numpy.
  - Predictions go back through `Quoll.wrap_data` and are written in any format Quoll supports.

**Weighing the two:**

| | (a) DeepH files | (b) juliacall + Quoll |
|---|---|---|
| Environments | Python only at train and predict time; Julia only for coupling authoring (unchanged rule) | One Julia 1.12 environment with ET and Quoll. Verified: ET `main` + Quoll `main` resolve together (200 packages; `using` both in 12 s after precompile). **Registered ET 0.5.1 does not co-resolve with Quoll**: 0.5.1 hard-depends on WignerD 0.1.4, which caps StructArrays below the 0.7 Quoll needs. ET `main` has already made WignerD test-only, so the next ET release fixes this. |
| Maturity exposure | Depends on a format (stable, shared with DeepH-E3/MACE-H) | Depends on an unregistered pre-release API (git `rev` pin; `juliapkg.json` needs a `url`+`rev` entry, as the consumer below does). Also brings MPI.jl and HDF5.jl JLLs into the authoring environment. |
| Conventions and units | ace-jax must re-implement DeepH's m-reordering and check units, i.e. duplicate `SHConvention` logic in Python | Quoll does it: `out_shconv` straight to ace-jax's convention, aims signs and core projection included. One source of truth. |
| Consumer fit | Consumers read files through Quoll anyway | The DFT-injection consumer (§8) already runs Quoll under juliacall, so an in-memory hand-off avoids a disk round trip |
| Formats beyond DeepH | Only what someone has already converted to DeepH | Every Quoll input (FHI-aims directly, future ONETEP/psinc sources for §5's learned orbitals), plus core projection and k-grids |
| Testing and CI cost | Small committed DeepH fixture; pure-pytest; no Julia in the main test job | A Julia job, which `coupling-parity` already is: juliapkg provisioning (minutes; cacheable), precompiling about 200 packages, a git-pinned Quoll that can move under us. The test-data artifact is 221 MB, so fixtures must be subset. |

**Recommendation: a hybrid.**
- Use **(b) for conversion and ingestion** from raw code output (FHI-aims now, other sources later), so conventions, units, core projection and sparsity are handled by Quoll alone.
  - Run it as a preprocessing step in an optional `ace_jax.ham` Julia extra, alongside ET in one Julia 1.12 environment.
  - Emit a **frozen training cache**: npz per-species-pair blocks in ace-jax's own SH convention, plus the Quoll version.
- Use **(a) DeepH files as the stable exchange format**, for data from other people and for consumers that read files.
  - Keep a pure-Python DeepH reader and writer, validated against (b) in CI.
- Fitting, evaluation and the main test job then stay Julia-free, and the unregistered API is touched only at ingestion.
- Pin Quoll by commit in `juliapkg.json` until it is registered. Do (b) only once an ET release without the WignerD hard dependency is out; until then (a) alone.

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

## 3. EquivariantTensors at the latest release (v0.5.1)

**What v0.5.1 provides** (checked in Julia 1.11 against the registry build):
- **L>0 couplings.**
  - `symmetrisation_matrix(L, mb_spec; …)` and `sparse_equivariant_tensors(LL=…)` give an `SVector{2L+1}`-valued A2B over one shared AA basis.
  - `O3.cgmatrix(L1,L2,L; basis=real)` and `TYVec2YMat(L1,L2)` map stacked L-vectors to a (2ℓ₁+1)×(2ℓ₂+1) block.
  - All of these already exist in 0.4.3.
- **Pseudo-tensors, new in 0.5.x.** `coupling_coeffs(L, ll, nn; refl_sym=:sym|:asym)` selects the parity of Σl independently of L. Verified:
  - ll=(1,1), L=1, `:sym` gives 1 function (an axial vector);
  - ll=(1,2), L=2, `:asym` gives 1;
  - ll=(1,1,1), L=0, `:asym` gives 1 (a pseudoscalar).
- `symmetrisation_matrix` forwards `refl_sym` through kwargs. `sparse_equivariant_tensor(s)` does not, so the shim calls `symmetrisation_matrix` directly. Ideally ET adds a one-line forward.
- **Rewritten real-SH couplings.** They are now orthonormalised. A caveat, from ET's own commit history: real-basis couplings can come out purely imaginary, which `assemble_U` now detects.

**Cost of bumping the pin 0.4.3 → 0.5.1 for the existing authoring extension.**
- **Coupling values change.** Row norms of the same L=0 spec went from [1, 1, 1, 1.6, 1.71] (0.4.3) to [1, 1, 0.77, 1.04, 0.98] (0.5.1). L=2 went from 5–8.6 to about 1.
  - `tests/test_coupling_parity.py` checks multiplicity-1 blocks as permutation plus sign, so it will fail on scale.
  - Make that check scale-invariant, using the projector already used for degenerate blocks (`coupling.subspace_residual`).
  - Or regenerate `fixtures/coupling_ref_*.npz`. They cannot be regenerated from ACEpotentials, because ACEpotentials 0.10.x requires ET 0.4.x.
  - Either way, the "bit-for-bit against ACEpotentials" claim becomes "same span per nnll block".
- **Cache.** The cache is keyed by the `juliapkg.json` hash, so entries recompute automatically. The committed `fixtures/coupling_cache_embedding` used under `ACEJAX_NO_JULIA=1` must be regenerated.
- **Unaffected:**
  - saved `.npz` models, which store their A2B;
  - the `julia/` ACEpotentials export environment (a separate project);
  - the smoothness prior, which depends only on nnll.
- **Newly authored potentials** get a differently normalised B basis with the same span. Fitted predictions are unchanged. Γ-weighted priors shift slightly, which is worth re-running on one benchmark.
- **Julia version.** Moving `juliapkg.json` from `"~1.10, ~1.11"` to `"~1.12"` is safe for ET (and needed for Quoll). A grep of the repo shows what else touches 1.11:
  - `.github/workflows/coupling-parity.yml` runs `setup-julia` with `1.11` for the juliapkg-provisioned ET. It **must** change to `1.12`; otherwise juliapkg downloads its own Julia on every run.
  - `.github/workflows/julia-parity.yml` and `prior-parity.yml` also use `1.11`, but for the separate `julia/` ACEpotentials project (unpinned `Pkg.add`, no Manifest). ACEpotentials 0.10.2 declares `julia = "1.11, 1.12"`, so they may stay on 1.11 or move to 1.12. Either is fine; moving keeps a single Julia in CI.
  - `docs/smoothness-prior-port.md` mentions regenerating the oracle fixtures "with Julia 1.11". Informational only.
  - The committed coupling cache records the `juliapkg.json` hash, so any edit to the file (Julia range or ET pin) invalidates `fixtures/coupling_cache_embedding`. Bump the range and the ET pin in one commit, and regenerate that fixture once.
  - Nothing else in `src/`, `tests/` or `pyproject.toml` depends on the Julia version, and nothing in the repo references Reactant.
- **Co-installation with Quoll** needs the ET release after 0.5.1 (WignerD becomes test-only on ET `main`, §2).

## 4. Mapping onto ace-jax

| Piece | Verdict |
|---|---|
| Real Y_lm, A/AA product, `pool_a_sparse`/`pool_a_dense`, `aa_specs` layout (`eval/edge_model.py`, `eval/model.py`) | **Reuse.** The on-site environment is the site environment. |
| Radials (`poly_recursion`, `radial_ace1.jacobi_3term`, spline) | **Reuse.** ACE.jl's exact radials no longer matter, since old models are dropped. |
| Species and bond/env category | **Reuse the channel fold** of `construct/spec.py::build_embedding_spec` (n = (n′−1)d+k). ET symmetrises only equal (n,l), so a bond factor with its own n is correctly never permuted with environment factors. |
| L>0 and pseudo-tensor couplings | **Julia shim extension.** `couple_equivariant(mb, Rnl, Ylm, L, refl_sym)` returns per-L A2B triplets (row, col, m, val) plus real CG tables. Add L and parity to the cache key. About 200 LOC plus parity tests against ET itself. |
| Equivariant readout | **New.** `block = Σ_L CG_L · (c_Lᵀ B^L)`, one einsum. |
| Bond environments | **New.** Build (bond i→j in cell T) × (env k) triplets from the existing `eval/nlist.py` list, with env vectors relative to the bond midpoint and a cylindrical envelope. Evaluate the features on (i,j) and on the reversed bond and symmetrise with the CG transpose sign (−1)^(ℓ₁+ℓ₂−L). This makes H_ji(−T) = H_ij(T)ᵀ exact, with no dual model. |
| Operator assembly | **New.** Blocks → sparse H(T), S(T) keyed like DeepH → Bloch sums H(k) = Σ_T e^{ik·T} H(T). All in JAX, all differentiable. |
| Saved model | **Extend the npz + `meta_json` schema** (`construct/export.py`, `eval/io.py`): per-(sub-model, L) A2B and coefficients, CG tables, orbital-basis definition. |
| Force/virial path (`EdgeSiteModel.energy_forces_virial`) | **Does not apply.** It assumes a scalar energy. ∂H/∂R is available by autodiff when needed. |

## 5. Fitting with spectra and observables in the loss

**Stage A: block loss (convex).**
- Targets are the block entries themselves: linear in the coefficients, with no Jacobians needed.
- The real CG stack is orthogonal, so projecting each target block onto L, y_L = CG_Lᵀ·H_block, decouples the problem into independent vector-valued regressions per (sub-model, L, parity), with 2L+1 rows per sample.
- `fit/solve.py` (streamed QR and LSQR) and the BLR evidence in `fit/objective.py` (σ, σ_c, prior Γ) apply once the row generator is generic. Today `fit/rows.py`/`stats.py` hard-wire E/F/V rows from site descriptors and edge Jacobians.
- Stage A gives initialisation, evidence-tuned regularisation, and a Gaussian posterior (§7).

**Stage B: observable loss (non-convex).**
- The loss is on eigenvalues, band energies, gaps or density matrices from the generalised problem H(k)c = εS(k)c. Plain LSQR is no longer enough.
- Minimise `L = w_blk·‖H − Ĥ‖² + w_spec·Σ_k w_k ℓ(ε(k), ε̂(k)) + …` over the coefficients with L-BFGS or Adam: `fit/radial_learn.lbfgs_loop` and optax, which are already dependencies. Warm-start from stage A.
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
  - The outer loop over C can reuse one QR factorisation (as in `fit/varpro.py`, `fit/radial_learn.py`) and differentiate through the inner solve.
  - The outer objective must be orbital-invariant, such as spectra in a window or model error normalised per orbital. A block loss alone rewards trivial orbitals.
- In-situ-optimised, environment-dependent orbitals such as ONETEP NGWFs need the operator in the underlying (psinc) basis. That is a data question for Quoll (a new source format), not a model question.

## 6. Validation without old models

1. **Pure JAX structural tests:**
   - rotation and inversion equivariance of each block;
   - exact bond-reversal symmetry;
   - Hermiticity of H(k);
   - autodiff against finite differences, including the degenerate spectral `custom_vjp`.
2. **Coupling parity:** shim versus ET directly, both L>0 and pseudo-tensor, using `subspace_residual`.
3. **Convention test** (§2): equivariance on rotated structures converted by Quoll.
4. **Accuracy on reference data, fresh fits:**
   - Quoll's regression data and small FHI-aims sets (water, SiC, gold);
   - ACEhamiltonians' own contrived Si HDF5 (160 kB), as data only.
   - Metrics: block RMSE, eigenvalue and band RMSE in a window, and SCF iterations saved by a downstream injection code.
   - Comparators: numbers published for ACEhamiltonians, and MACE-H or DeepH-E3 trained on the same Quoll-produced data.
5. **CI:**
   - a small committed DeepH-layout fixture (a few MB, under the 4 MB hook limit) for unit and accuracy smoke tests;
   - a path-gated `ham-accuracy` job that downloads a pinned Quoll artifact;
   - no Julia needed at test time except the existing coupling-parity job.

## 7. Uncertainty: BLR, POPS and conformal

**Stage-A BLR.** One posterior per (sub-model, L, parity): `objective.posterior`, `log_marginal_likelihood` and the Γ prior, with generic rows.
- The predictive covariance of a block's L-part is B_Lᵀ Σ B_L. That is (2L+1)×(2L+1) and rotation-covariant.
- Element-wise σ follows through the CG map. The port can therefore give a *properly equivariant* per-block covariance.
- That is stronger than element-wise variances, which are not rotation-invariant and need heuristic averaging downstream.

**POPS** (`fit/pops.py`) carries over, applied per-observation and pointwise, for misspecification.

**Conformal.**
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

The port meets this by writing `hamiltonians_pred.h5` and an uncertainty twin in DeepH layout, plus its metadata files. For callers that already host Quoll under juliacall, it can also hand over an in-memory numpy block dict, or a Quoll operator built in the shared Julia 1.12 environment (§2 hybrid). Predicting never requires Julia.

## 9. Framework and application boundary

**Proposal.** This mirrors the Julia stack (EquivariantTensors/Polynomials4ML as framework; ACEpotentials/ACEhamiltonians as applications).

- **Framework** (`ace_jax.core`, eventually its own distribution):
  - harmonics, radials, the A/AA/A2B product and its layouts, neighbour and triplet graphs;
  - the ET shim and cache, spec builders, the smoothness prior;
  - generic fitting primitives: row-generator interface, streamed stats, QR/LSQR, BLR evidence, POPS, conformal calibration, VarPro/L-BFGS;
  - the npz schema helpers.
- **Applications:**
  - `ace_jax.potentials`: `ACEModel`/`PACEModel` site energies, E/F/V rows and the GP arm, calculators, LAMMPS export, the `aj fit` CLI;
  - `ace_jax.ham`: bond basis, equivariant readout, Quoll/DeepH IO, spectral losses, operator UQ, consumer IO.

**Recommendation: start in-tree, with an enforced boundary.**
- `ham` may import `core` but never `potentials`. `core` imports no application code. Lint this with a ruff/import-linter rule.
- Split distributions later, once `core`'s API has settled.
- The first refactors are what make the boundary real:
  - move E/F/V out of `EdgeSiteModel`;
  - make `fit/rows.py`/`stats.py` generic over row generators.
  - They are far cheaper in one repo than across a versioned package boundary.
- A separate package from day one would freeze today's energy-specific internals as its API.

## 10. Revised phased plan

The sizes are rough, for one developer who knows ace-jax.

| Phase | Content | Size |
|---|---|---|
| **0. Foundations** | In one commit: bump `juliapkg.json` to Julia `~1.12` and the latest ET; switch `coupling-parity.yml` to 1.12 (optionally also `julia-parity`/`prior-parity`); make coupling parity scale-invariant; regenerate the coupling cache fixture; re-run one potential benchmark. Add `couple_equivariant` (L, refl_sym, CG). Introduce the `core` boundary; move E/F/V out of `EdgeSiteModel`. Add the pure-Python DeepH reader and writer; the Quoll ingestion path (juliacall, git-pinned Quoll, once an ET release without WignerD is out) writing a frozen npz training cache; a CI check that the two paths agree; the ace-jax ↔ Quoll SH-convention test. | 3–4 wk |
| **1. Equivariant model and assembly** | On-site and bond models (triplet graph, cylindrical envelope, symmetrised bond basis, per-L readout). H(T)/S(T) → H(k). Structural test suite. npz schema. | 4–5 wk |
| **2a. Stage-A fitting and UQ** | Generic rows. Per-(sub-model, L) streamed QR and BLR evidence. Block-covariance UQ and POPS. Accuracy benchmarks on Quoll data against MACE-H/DeepH-E3. DeepH-format prediction output for consumers. | 4–5 wk |
| **2b. Observable fitting** | Degeneracy-safe spectral `custom_vjp`. Generalised eigenproblem. k-batching and bucketing. Combined loss with L-BFGS/Adam warm-started from 2a. Conformal calibration on eigenvalues. | 4–6 wk |
| **3. Performance and deployment** | Dense (bond, K) layouts. Batching sub-models by (ℓ₁,ℓ₂) to bound jit variants. Sparse operator output. GPU k-point batches. A `jax.export` artifact. | 3–4 wk |
| **4. Learned orbitals (research)** | Reference-basis data through Quoll. Differentiable contraction C. VarPro outer loop. Orbital-invariant objectives. | open |

**Total to a usable, UQ-carrying, observable-trained model: about 5–6 months.** Phases 0–2a, about 3 months, give a working block-fitted model with UQ that consumers can use.

## 11. Risks

1. **SH convention and units at the Quoll boundary.** A wrong sign fits well but breaks equivariance. Mitigated by the rotation test in phase 0.
2. **ET 0.5.1 coupling changes.** Renormalisation, the purely-imaginary real-coupling edge case, and no ACEpotentials-backed regeneration of the coupling fixtures.
3. **Quoll immaturity.** Pre-release, unregistered, one main author, git-pinned in juliapkg.
   - Mitigated by confining the Quoll API to ingestion, with a frozen npz cache and DeepH files as the exchange format.
   - Co-resolution with ET needs the next ET release (the WignerD conflict).
4. **Spectral gradients.** Degeneracies; ill-conditioned S(k) in large or diffuse bases (Cholesky); cost of eigh at scale; and a non-convex loss that loses the closed-form evidence.
5. **Bond-environment edge cases.** Periodic self-images, atoms near the midpoint, long bonds; plus the number of jit variants across sub-models × L.
6. **Physics constraints.** S(k) is not guaranteed positive definite, and on-site S = I holds only for orthonormal-on-site bases.

## 12. Open questions remaining

1. **SH-convention anchor.** Should ace-jax emit Quoll's wiki convention natively, or keep SpheriCart's and convert at the IO boundary? The latter is proposed.
2. **Quoll asks.** Is the owner willing to ask Quoll for registration and a canonical on-disk format? Or is DeepH layout the long-term exchange format? Also, should ET be asked to cut the release that makes WignerD test-only? Until then, in-process Quoll cannot co-install with a registered ET.
3. **Julia in CI.** Move `julia-parity`/`prior-parity` to 1.12 alongside `coupling-parity`, for a single Julia across CI? Or leave the ACEpotentials jobs on 1.11?
4. **Pseudo-tensor inclusion by default.** On, for parity with ACEhamiltonians' completeness; or opt-in, since it adds basis functions?
5. **Stage-B loss priorities.** Which observables come first: eigenvalues in a window, band energy, density matrix, gaps?
6. **Split timing.** Is "in-tree with a lint-enforced boundary, split later" acceptable?
