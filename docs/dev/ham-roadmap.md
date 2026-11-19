# Hamiltonians in ace-jax: roadmap

Status: proposal by the owner, 2026-11-18, stored here and merged with the running plan the same
day (the sections "Immediate queue" and "HInitASI: read from the source" and the notes marked
*merged* are additions; the rest is as received). Written against `feat/hamiltonians` at d7bbf5a
and `docs/dev/ham-results.md` as of that commit. Design background:
`specs/2026-11-16-hamiltonians-design.md`, `specs/2026-11-17-package-structure-design.md` and its
feature-provider addendum.

## Aim

1. Reach, then exceed, the feature set of ACEhamiltonians.jl, MACE-H and DeepH (DeepH-E3 and
   DeepH-pack 1.0) where it matters for our use.
2. Provide what two proposed projects need from the Hamiltonian model:
   - **Transport** (Neophytou, PI): calibrated Hamiltonians, unfolded spectral functions A(k,E)
     and linewidths Γ(k,E) with error bars for hundreds of disordered supercells of half-Heuslers,
     full-Heuslers and Kagome metals, handed to Boltzmann, Monte Carlo and NEGF solvers.
   - **Catalysis, "SPARK"** (Logsdail, lead): predicted electronic structure as the SCF starting
     point in FHI-aims, with an uncertainty rule that decides per calculation whether to use it,
     demonstrated on surface reactions (water-gas shift on Pt/TiO₂) and usable in QM/QM embedding.

The defect-qubit proposal (Ortner, lead) wants the same scale and spin work as the transport
project and is not planned separately here.

## Immediate queue (*merged*, 2026-11-18)

The work in flight when the roadmap arrived, in the order the owner set ("eigenvalue errors first
before evidence"), with the phase each item belongs to.

1. **Alloy eigenvalue errors** (Phase 2, conditioning). The four-element C/Si/Ge/Sn set gives
   3 to 6 eV eigenvalue RMSE for every model, also in the well-conditioned overlap subspace,
   while the blocks are at 25 to 29 meV. Diagnose before anything is built on multi-element data.
2. **Evidence for σ_c, ridge and noise** (Phase 1, second bullet). Brought forward: the alloy's
   prior optimum is narrow, and order-2 site features with `widths=(S, d, …)` are not usable
   without it.
3. **Refit with forces and stress** on aluminium (Phase 2, total energy): three fits running;
   all 100 FHI-aims force runs are usable.
4. **Gold cross-code table** (Phase 0, same-machine timings): ace-jax grid 54 of 56 points; on
   Sulis MACE-H with 1 to 8 structures running, 16 to 64 and all DeepH-E3 jobs pending; then
   predictions and one-CPU timings on the desktop.
5. **Re-time float32 GPU rows** (Phase 0): TF32 is fixed (`operators/precision.py`); the earlier
   float32 rows of the MACE-H comparison were taken with it on.

Decided 2026-11-18, in addition to the list at the end:

- **DeepH-pack 1.0 is not run.** Its wheel is obfuscated (PyArmor) and the owner does not trust
  hidden code. It stays in the scorecard by its published features and numbers only; the DeepH
  entry of the measured cross-code table is DeepH-E3.
- HInitASI's source can be read (the owner has access; MIT licence in the repository). Nothing of
  it is copied into this repository.

## Where the branch is

Done, with a result in `ham-results.md`:

- On-site and off-site block models for H and S with shared features: two-centre, site-coupled
  and bond-environment groups; exact bond reversal; operator channels.
- Closed-form fit on `fit.linear` (QR, multiple right-hand sides), on the device; smoothness prior.
- Collinear spin as channels (bcc Fe: 5 to 7 meV blocks, moment to 0.014 µ_B with a conditioned
  overlap).
- DeepH-layout reader and writer; FHI-aims ingestion through Quoll; core-orbital projection.
- Bloch sums and bands on the device in batches of k; device neighbour lists.
- Derivatives: spectrum with respect to the coefficients; eigenvalues, band energy, forces and
  stress with respect to positions (trace form).
- ASE calculator: band energy, free energy, forces, stress, eigenvalues, DeepH output.
- Total energy as band term plus a residual potential (aluminium, energies only: 3 to 3.5 meV per
  atom against 0.55 for the potential alone).
- Element-embedding (TrACE) site provider; species channels in the bond environment.
- Gold benchmark: 0.188 meV H MAE and 1.16 meV eigenvalue error from two training structures and
  a 74 s fit, against 0.316 / 2.96 (MACE-H) and 0.409 / 6.28 (DeepH-E3) as published; 8 to 11
  times faster than MACE-H on one CPU core at the accuracy of the 2022 paper's target.

*Merged* (results after d7bbf5a's parent, already in `ham-results.md`): every gold model is flat
from 32 to 152 training structures; the largest reaches 0.152 meV / 0.95 meV. Per-order embedding
widths exist (`build_embedding_site_basis(widths=)`); an embedding narrower than the number of
elements at order 1 loses an eV, so the only use of TrACE is from order 2. Float32 on a GPU is
correct after the TF32 fix and 4 to 7 times faster than float64 for E_band.

Not done (the "not yet" lines of the results, collected):

- No uncertainty of any kind on operators.
- No model file, no loader, no CLI, no user documentation.
- Hyperparameters by hand (ridge, σ_c); no evidence.
- Many species: two-centre and site-coupled groups and readouts are per species pair; the only
  multi-element accuracy number is poor (four-element alloy: 28.7 / 24.9 meV on-site / off-site).
- No spin-orbit coupling, no non-collinear spin; spin not in the calculator.
- Dense eigensolver only (60 to 64 % of a force call; N³).
- No DOS, no unfolding, no velocities, no transport or SCF hand-over.
- Open numerical items: threading of the end-to-end call; float32 GPU timings taken with TF32
  need re-timing; nearly dependent overlaps need a conditioned solve or a spectral fit.

## Parity scorecard

| Capability | ACEhamiltonians.jl | MACE-H | DeepH-pack 1.0 | ace-jax now | Phase |
|---|---|---|---|---|---|
| H and S, on-site and off-site, equivariant | yes | H (S from DFT) | H (S computed) | yes | – |
| Speed at matched accuracy | slow | reference | faster than DeepH-E3 | 8–11× MACE-H, CPU | – |
| Fit from a handful of structures | yes | no (hundreds, GPU-hours) | no | yes, seconds | – |
| Trained model file, CLI, docs | yes | yes | yes | no | 0 |
| ASE calculator | no | H + uncertainty | no | H, E_band, F, stress | 0 |
| Uncertainty | none in the code | Hermiticity score; propagation and gate in HInitASI | none | none | 1 |
| Many species | per-pair models | yes | yes; universal model | per pair, untested | 2 |
| Collinear spin | no | no | via xDeepH lineage | first fit | 3 |
| Spin-orbit coupling | no | inherited from DeepH-E3 | yes | no | 3 |
| Non-collinear magnetism | no | no | xDeepH | no | 3 (late) |
| DFT codes | FHI-aims | FHI-aims, OpenMX | FHI-aims, QE, SIESTA, ABACUS, OpenMX | FHI-aims (Quoll), DeepH layout | 2 |
| Hybrid-functional data | no | no | DeepH-hybrid | untested | 5C |
| Large cells (≥ 10⁴ atoms) | DFTB+ interface | Julia sparse tools | 4,096 atoms shown; PEXSI | dense only | 4 |
| Multi-GPU training / inference | distributed | no | yes | no | 4 |
| Bands, DOS, unfolding | bands | bands, DOS, entropy error | bands, DOS, unfolded bands | bands | 5T |
| Forces and stress | no | no | no | band term + residual | 2 |
| Position derivatives of H (electron-phonon) | no | no | DeepH-DFPT (separate) | autodiff of eigenvalues | 5T |
| Differentiable spectrum (observable training) | no | no | no | gradient exists | 2 |
| SCF injection | DFTB+ | FHI-aims via HInitASI | DFT restart via dock | DeepH output only | 5C |

The DeepH-pack column is from its publications; the package is not run here (see the decisions).

"Exceed" therefore means: calibrated uncertainty, fits in seconds from very little data, forces
and total energies, a differentiable spectrum, and the two project pipelines. It does not mean a
universal materials model (non-goal below).

## What the projects need that parity does not give

| Need | Transport | SPARK | Phase |
|---|---|---|---|
| Calibrated uncertainty on blocks, eigenvalues and derived quantities, Mondrian by environment | core | core (the gate) | 1 |
| Support diagnostic and per-configuration trust flag | D-AI3 | the gate | 1 |
| 3 to 5 elements with substitution and defects | core | Pt, Ti, O, C, H | 2 |
| Spin-orbit coupling (Bi, Sb, Sn, Te, Pt; Kagome bands) | core | useful | 3 |
| Collinear magnetism in production (Heuslers, Kagome metals, open-shell adsorbates) | core | core | 3 |
| 10³ to 10⁴ atoms: sparse assembly, window eigensolver or Chebyshev moments | core | no | 4 |
| LCAO unfolding, A(k,E), Γ(k,E) with intervals | D-AI2, D-AI3 | no | 5T |
| Velocities, transport hand-over files, NEGF blocks | hand-over | no | 5T |
| All-electron H handed to FHI-aims through HInitASI / ASI | no | core | 5C |
| Slabs with vacuum, adsorbates, tier-2 bases | no | core | 5C |
| SCF-iteration benchmark harness | no | core | 5C |
| Embedding: blocks of a subsystem in a frozen environment | no | QM/QM | 5C (late) |

## Phases

Phases 0 to 2 are common. 3 and 4 serve transport first. 5T and 5C are the project pipelines and
can run in parallel once 1 and 2 exist. Each phase ends with a section in `ham-results.md`.

### Phase 0. Land it

The branch is 117+ files and a day ahead of `main` in design. Nothing below should be built on an
unmerged branch.

- Model file: npz with `ham_json` (orbitals, channels, providers, coefficients, core projection,
  conditioning threshold); `load` dispatches on the key; calculator from a path.
- `aj ham fit | predict | bands`; one user page; the `ham` extra; `ace-jax-coupling` 0.3.0.
- Re-time every float32 GPU row with `highest` precision; close or document the threading item.
- Same-machine timings for the gold table once the MACE-H and DeepH-E3 ladder jobs return.
- Merge in three PRs: `basis`/coupling changes, `operators`, `ham` with benchmarks.

Exit: `pip install ace-jax[ham]`, a fitted gold model loaded from a file reproduces the table.

### Phase 1. Uncertainty for operators

The differentiator in both proposals, and the piece with no competitor. ace-jax's blocks are
Hermitian by construction, so MACE-H's Hermiticity score is identically zero here: the score has
to come from the linear model.

- Block posterior from the QR factors already returned by `stats=True`: covariance of each L part,
  B_Lᵀ Σ B_L, per channel.
- Evidence for σ_c, ridge and noise per (species pair, L, parity) group, replacing hand tuning
  (the four-element result shows a narrow optimum). *Merged*: second in the immediate queue.
- Shape by delete-one-structure jackknife; scale by Mondrian conformal with groups (species pair,
  bond-length band, L) and site groups (bulk-like, substituted, defective); χ_{2L+1} reference.
- POPS ensembles for misspecification.
- Propagation: first order, δe_n = c_n†(δH − e_n δS) c_n, for bands and band energy; coefficient
  draws through `spectrum_fn` for DOS and anything nonlinear. Check coverage on gold, aluminium
  and Fe eigenvalues.
- Support diagnostic on site and bond features (the potential's `uq.support`), giving the
  per-configuration flag.
- Calculator: `results["hamiltonian_uncertainty"]` (the key HInitASI reads; *merged*: it holds
  element **standard deviations**, which HInitASI squares itself), `["eigenvalues_std"]`; a twin
  in DeepH layout.
- Division of labour with HInitASI: for runs inside FHI-aims its propagation, calibration and
  gate are used as they are (Phase 5C). The propagation here is for ace-jax's own bands and for
  cells too large to put through FHI-aims (Phase 4, 5T), and is the exact reference for 5C step 5.

Exit: empirical coverage within tolerance for blocks and eigenvalues on three datasets, per
Mondrian group; a figure of bands with calibrated intervals.

### Phase 2. Chemistry and conditioning

- *Merged*, first: why the four-element alloy's eigenvalues are wrong by eV while its blocks are
  right to 25 meV (immediate queue, item 1).
- Species: element-embedded two-centre and site-coupled groups; order-2 site features with
  `widths=(S, d, …)` (order 1 at full width), chosen by the evidence of Phase 1.
- A multi-element benchmark that looks like the projects: one half-Heusler with substitution
  (transport) and one oxide-supported metal slab with adsorbates (SPARK). Both in FHI-aims.
- Larger orbital bases: tier 2, f shells throughout; check whether any target species brings g
  shells (features to L = 8; the coupling library's L_MAX is 6).
- Nearly dependent overlaps: a conditioned eigenproblem in the predictor (threshold in the model
  file), and the S^(−1/2)-weighted fit.
- Observable fine-tuning: a few steps on eigenvalues through `spectrum_fn`, starting from the
  closed-form fit, with the degenerate-safe rule of the design's phase 2b.
- Total energy: residual potential with forces and stress (data computed; fits running); a
  relaxation and a short MD as tests. Decide whether this is a product or stays a benchmark.
- Readers: OpenMX and SIESTA through the DeepH layout, so the DeepH-E3 2D-material sets
  (graphene, MoS₂, twisted bilayers, Bi₂Se₃) can be run.

Exit: the gold-quality result on a 3 to 5 element system, with uncertainty; the MACE-H paper's
other benchmarks in the cross-code table.

### Phase 3. Spin and relativity

- Collinear spin in production: both channels in one calculation, moments, the calculator, the
  model file, uncertainty per channel.
- Spin-orbit coupling: H = H₀ ⊗ 1 + Σ_a H_a ⊗ σ_a. Each H_a block couples the orbital parts with
  an l = 1 pseudovector, so it uses the integer-L features already built, with complex
  coefficients and time-reversal constraints. Needs FHI-aims SOC output through Quoll; validate
  on Bi₂Se₃ (the DeepH-E3 set) and on gold with SOC.
- Non-collinear magnetism (moment directions as inputs, xDeepH-style) only if a Kagome target
  needs it; otherwise recorded as out of scope.

Exit: SOC band structure of a heavy-element compound within the eigenvalue error of the
non-relativistic models; a ferromagnetic Heusler with both channels.

### Phase 4. Scale

The eigensolver is 60 % of a call at 64 atoms and grows as N³; the blocks are linear.

- Blocks straight into a sparse H(R), S(R) on the device, without the dense (atom, slot) copy.
- Energy-window eigenpairs by shift-invert (the transport proposal's stated route) for bands and
  unfolding near E_F; Chebyshev moments (kernel polynomial) for DOS and A(k,E) where eigenvectors
  are not needed, which also gives stochastic-trace uncertainty for free.
- Data-parallel fit and prediction across GPUs with the potential fit's shard-and-merge of QR
  factors; chunked prediction for cells that do not fit one device.
- A scaling table: 10², 10³, 10⁴ atoms, time and memory per stage, against DeepH-pack's published
  4,096-atom case.

Exit: DOS and window bands of a 10⁴-atom alloy supercell on one A100, with intervals.

### Phase 5T. Transport pipeline

Maps onto the proposal's D-AI1 (models and benchmark), D-AI2 (unfolding and calibration) and
D-AI3 (calibrated Γ database).

- LCAO spectral-weight unfolding from the model's eigenvectors and S; validate against
  plane-wave unfolding of DFT on small supercells.
- Linewidths Γ(k,E) from A(k,E). They are second order in the off-diagonal disorder, so a
  dedicated calibration set of DFT-unfolded widths; test whether ensemble averaging removes the
  broadening that model noise adds.
- Velocities: ∂H/∂k − e ∂S/∂k from the Bloch sums (exact, no finite differences).
- Hand-over: k- and energy-resolved scattering rates with error bars and trust flags in an
  agreed file format; principal-layer H and S blocks for NEGF; the transport distribution
  function for Seebeck and electronic thermal conductivity.
- Electron-phonon: g = ⟨ψ|∂H/∂R − e ∂S/∂R|ψ⟩ by autodiff of the blocks (position derivatives
  exist); phonons from the relaxing potential. This is the link to thermal transport, and the
  DeepH-DFPT parity item. Lattice thermal conductivity itself is the potential's job, not this
  branch's.
- Workflow: relax with the fine-tuned potential, predict, flag, unfold, export, for hundreds of
  configurations.

### Phase 5C. SCF acceleration for catalysis

Injection, uncertainty propagation, calibration and the gate already exist in HInitASI
(`vitartas/HInitASI`, docs at vitartas.github.io/HInitASI). It works with MACE-H. This phase
makes ace-jax a second source of predictions for it and adds nothing HInitASI already has.

What HInitASI does, from its documentation:

- Supplies the predicted H as the SCF starting point through an ASI Hamiltonian-init callback,
  per (k-point, spin): real-space blocks, Fourier-transformed on demand, scattered over BLACS.
- `HInitCalculator`: inject and let the SCF converge. `HInitOverwriteCalculator`: also propagate
  element variances of H to eigenvalue variances by first-order perturbation theory, reduce them
  to one number (`risk_function` over an `energy_window`), and skip the SCF when it is under
  `uncertainty_threshold`. The same variances go to bands, DOS and projected DOS.
- `calibrate_eigenvalue_uncertainty`: a weighted conformal quantile of (error / predicted σ) over
  reference configurations gives one `calibration_weight`.
- `calculate_eigenvalues`: inject, diagonalise once, stop.
- Conversion of the prediction to its canonical form, and the masking to FHI-aims' own real-space
  sparsity, are done in Julia through Quoll.jl; the predictor's output is passed on as it is.

What it needs from a predictor (`HamiltonianConfig(data_format, calc, kind_tags, …)`):

- an ASE calculator exposing `hamiltonian`, and `hamiltonian_uncertainty` for the overwrite path;
- output in a `data_format` Quoll reads, the same for H and its uncertainty;
- or no calculator at all: `HamiltonianIOCalculator(directory)` presents files already on disk.

#### HInitASI: read from the source (*merged*, at its commit b50919c, 2026-11-08)

What the source settles of the open questions below, and what it does not:

- **Uncertainty convention.** `hamiltonian_uncertainty` holds element **standard deviations**;
  `HInitOverwriteCalculator` squares them in place (`variance_from_sigma_inplace`) and then
  applies the preprocessing (default `subblock_averaging_dd`: mean over m, m' within each shell
  pair, on-site diagonal doubled). So Phase 1 writes σ, not σ².
- **File name.** In the DeepH layout the uncertainty is `hamiltonians_uncertainty.h5`, beside
  `hamiltonians.h5`, selected by the kind tag `source => pred`
  (`get_avail_filename` for `DeepHBlockNoSpinRealMetadata`). `write_results(…, uncertainty=True)`
  should write that name.
- **`data_format`** is a string Quoll resolves to a read format (`Quoll.get_readformat`), the
  same for H and its uncertainty; `kind_tags` build the `Quoll.OperatorKind`. In memory, a DeepH
  prediction crosses to Julia as a dictionary of JSON keys `"[R1, R2, R3, i, j]"` to arrays: the
  h5 file's own keys, so `results["hamiltonian"]` is the dictionary `ham.deeph.write` already
  builds. The exact strings MACE-H is run with are in the test data's metadata, which is not in
  the repository: still to ask.
- **Collinear spin: not through the DeepH route today.** ASI asks per (k-point, spin) and the
  propagation is indexed by it, but the only prediction format with a Python hand-over is
  `DeepHBlockNoSpinRealMetadata`. A two-channel prediction needs a format on the Quoll side.
  Confirms the open question; to raise with the author.
- **Core-projected models: nothing in the source.** No core handling anywhere; the injected
  matrix is masked to FHI-aims' basis-function radii and Fourier-transformed in the full basis.
  All-electron models are what works as written, which is what 5C step 1 uses.
- **One predicting process per node** (`h_comm` = node roots; a custom one is rejected); the
  calculator needs `write_metadata(output_dir)` and `write_results(output_dir, data, metadata,
  uncertainty)`, and other tasks never call `calculate`. Deferred model loading as planned.
- **Environment.** `asi4py` from `gitlab.com/vitartas/asi` branch `hinitasi`; `juliacall ≥
  0.9.20`; Julia `~1.12` with Quoll.jl at `main` (ace-jax's `ingest.jl` runs on 1.13.1 with its
  own pin: two Julia environments, not one); `hinitasi-setup` pins MPI.jl and HDF5.jl to the
  local libraries. FHI-aims must be built as a shared library with ASI; the build on `/storage`
  is an executable only.
- **Propagation methods.** `"diagonal"` (per-eigenvalue variance, independent elements) and
  `"full"` (covariance between eigenvalues, testing only). Neither takes covariance between
  Hamiltonian elements, so step 5 below stands as written.

Steps:

1. **Files first.** Write a prediction and an element-uncertainty twin in the DeepH layout with
   `HamiltonianCalculator.write_results`, and run HInitASI on them through
   `HamiltonianIOCalculator`. No interface code; it tests conventions, masking and the
   all-electron question on the aluminium model at once. *Merged*: needs FHI-aims rebuilt as an
   ASI shared library first.
2. **Element uncertainties from the linear model** (Phase 1's block posterior, expanded from L
   parts to elements). HInitASI's default preprocessing averages the variance within each
   angular-momentum subblock to make the eigenvalue uncertainty rotation-invariant; a variance
   that is constant over each L part already is, so check that the two agree.
3. **In-memory back end.** `HamiltonianCalculator` from a model path, deferred loading (HInitASI
   predicts on one task per node and the others hold no model), `write_metadata`, and the
   `hamiltonian_uncertainty` property. Custom prediction communicators are rejected by HInitASI,
   so one process predicts the whole cell: fine for SPARK's sizes.
4. **Coverage of the sparsity pattern.** With `mask_to_code_sparsity` the injected matrix is cut
   to FHI-aims' basis-function radii; ace-jax must predict every pair inside them, so the bond
   cutoff follows the largest pair of basis radii, not a tuned value.
5. **Correlated errors.** HInitASI propagates independent element variances and repairs the scale
   with one conformal weight. A linear model's errors are strongly correlated between blocks that
   share coefficients. Compare, on the same configurations, HInitASI's propagated σ with the exact
   first-order one, gᵀ Σ g with g the gradient of an eigenvalue with respect to the coefficients
   (`spectrum_fn`). If they differ in shape, not only in scale, offer the exact one as a
   propagated eigenvalue uncertainty HInitASI can take in place of its own.
6. **Benchmark**: SCF iterations and wall time against the default start, ace-jax and MACE-H
   through the same HInitASI path, from molecules to slabs with adsorbates to the water-gas-shift
   system; fraction of SCF cycles skipped at fixed coverage.

Open, to settle with the HInitASI author (*merged*: narrowed by the reading above):

- Core-projected models: FHI-aims needs H in its full basis, and HInitASI has no core handling.
  Are only all-electron models usable, or is rebuilding the core blocks wanted?
- The `data_format` and `kind_tags` strings MACE-H is run with.
- Collinear spin: a two-channel prediction format on the Quoll side.
- JAX alongside `juliacall`, `mpi4py` and FHI-aims in one process: threads, signal handlers, GPU.

Environment: Python ≥ 3.11, Julia 1.12 through `juliapkg`, one MPI shared by Python, Julia and
FHI-aims, parallel HDF5, Quoll.jl, FHI-aims as a shared library, and a fork of `asi4py`
(`hinitasi` branch) until its callbacks are merged upstream.

Later: slabs (vacuum and under-coordinated atoms in the training design; dipole corrections act
on H non-locally and stay outside the model); hybrid-functional data (longer-ranged blocks);
subsystem blocks for QM/QM embedding (EmbASI uses the same interface).

### Later, or not at all

- TT-ACE chain as a provider (addendum A4), once Phase 2 shows what order 2 needs.
- Learnt orbitals and gauge-aware targets (the defect-qubit proposal's method work).
- **Non-goals:** a universal materials model; training without data (DeepH-Zero); plane-wave
  Hamiltonians by projection; loading ACEhamiltonians.jl models; running DeepH-pack.

## Order and dependencies

```
0 land ──► 1 UQ ──► 2 chemistry ──┬─► 3 spin/SOC ──► 4 scale ──► 5T transport
                                  └─► 5C SCF (needs 1, 2; not 3, 4)
```

- 5C step 1 (files through `HamiltonianIOCalculator`) can start now, with the existing
  all-electron aluminium model; steps 2 to 5 need Phase 1's block posterior.
- 5T's unfolding and velocities can be built on gold and aluminium before Phase 3.
- For the proposals themselves, the two figures worth having are bands with calibrated intervals
  (Phase 1) and an SCF-iteration count from a predicted start on one small system (5C, step 1).
- *Merged*: the immediate queue runs ahead of Phase 0's merge, because its first two items
  decide what the model file must hold (a conditioning threshold, per-group hyperparameters).

## Risks

- **Multi-element accuracy.** The one result so far is poor, and every model is order 1. If
  order 2 with compressed widths does not close it, the fallback is per-pair models for the 3 to
  5 elements each project actually has, which is affordable at that count.
- **Linewidth calibration.** Γ may be dominated by model noise; a negative result still bounds
  what scattering models can claim, as the proposal says.
- **SCF injection may gain little** where the default guess is already good. The gate turns that
  into a reported result, but the benchmark hierarchy must include hard cases (metals, slabs,
  open shells).
- **SOC data path.** FHI-aims applies SOC after the SCF; what Quoll can read needs checking
  before Phase 3 is scheduled.
- **Ingestion.** Quoll stays (decided 2026-11-18): its author is porting it to Python, which
  will remove the Julia step. Its missing spin conversion is worked around in `ingest.jl`.
- **Branch size.** Phase 0 is the mitigation.
- *Merged*: **HInitASI and spin.** SPARK lists collinear magnetism as core, and the injection
  path has no spin prediction format yet.

## Decisions needed from the owner

1. Is the band-plus-residual total energy a product or a benchmark?
2. Is non-collinear magnetism in scope for the Kagome targets?
3. Which half-Heusler and which slab system become the Phase 2 benchmarks?

Decided 2026-11-18: SCF injection goes through HInitASI (a Hamiltonian, not a density matrix);
Quoll is kept for ingestion; DeepH-pack is not run; alloy eigenvalue errors before the evidence.
