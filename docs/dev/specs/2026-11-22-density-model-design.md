# Electron-density model: atom-centred expansion coefficients on the site features (design, 2026-11-22)

Status: spec, not implemented. To be implemented on a new branch `feat/density` from `origin/main`, by an agent that has not seen the prototype. Everything it needs is here or at a path given here.

## What and why

A model that predicts the electron density of a structure as coefficients of an atom-centred auxiliary basis,

    rho(r) = sum_i sum_{n, lam, m} c[i, n, lam, m] phi_{n lam}(|r - r_i|) Y_{lam m}(r - r_i),

each coefficient vector c[i, n, lam, :] a linear function of the atom's equivariant site features of (L, parity) = (lam, (-1)^lam). It is the same construction as the on-site Hamiltonian model (`operators/onsite.py`) with a vector per auxiliary shell in place of a block per shell pair, so it shares the site basis, the row streaming, the linear solves, the model file and the predictor layout.

Evidence, from a prototype outside the repository (`docs/dev/ham-results.md`, sections of 2026-11-22):

- Aluminium, 32-atom cells, 6 training structures, 394 parameters: the total energy after one Kohn-Sham solve from the predicted density is within 0.3 to 3.6 meV per atom of the converged one (free atoms: 48 to 92).
- Two structures of each new kind (vacancy, strain, slab, strong rattle) bring every held-out cell within 8.3 meV per atom; the H model given the same data did not adapt as well.
- The H that the DFT code builds from the predicted density, compared with the H model's own prediction, sizes the H model's band error within 0.64 to 1.27 without calibration, in and out of distribution. The density model is therefore also the second opinion of the Hamiltonian UQ.
- The loss matters: least squares in the metric of the auxiliary overlap gave a density error of 0.017 (test mean, 6 structures) where per-coefficient least squares gave 0.105, no better than predicting the mean coefficients (0.10 to 0.17).

Prototype (read it before starting; do not copy its hard-wired aluminium constants): `/storage/eng/essswb/projects/ace-jax/ham-ref/work/salted/scripts/{features,fit,sfit,predict}.py`; the model used in the results is `sfit.py`'s "S-metric" fit, `pred_o1/n6_r1e-06/model.npz`. A local mirror is in `~/projects/ace-jax-ham-ref/salted/`.

## Constraints

- Nothing derived from the DFT code's sources, and none of its input or species files, goes into the repository. The repository reads and writes a neutral format (below). The converter from the DFT code's text files stays outside (`~/projects/ace-jax-ham-ref`), and the tests use synthetic data only.
- Layers (`tests/test_import_layers.py`): the generic model is layer 3 in `operators`, which imports `eval`, `basis`, `fit.linear` only. Nothing density-specific (electron counts, file formats, command line) in `operators`.
- float64, as the block fit (`_need_x64`).
- Code and test style as CLAUDE.md. Conventional commits. No pull request is merged without the owner's word.

## Modules

### `operators/centred.py` (generic: equivariant per-atom coefficients)

Nothing here knows it is a density.

- **Auxiliary basis**: an `OrbitalBasis` (`operators/orbitals.py`): shells per species as a list of l. Reuse it as it is; `shell_slice(z, s)` gives the columns of shell s in an atom's vector.
- **`CentredModel(eqx.Module)`**: `basis: SiteBasis`, `coef: {(z, s): (n_channel, n_B(lam, p) [+ 1 for lam = 0])}`, `shells: OrbitalBasis` (static), `channels: tuple` (static; the density has one, "rho"; a spin density would have two). p = (-1)^lam. The lam = 0 part has a constant column, as the on-site L = 0 part: the coefficient of the isolated atom.
- **Prediction**: `predict(positions, cell, numbers, pbc=True) -> {channel: (n_atoms, n_aux_max) or list per atom}`; choose one layout and document it (a ragged list per species, `{channel: {z: (n_z atoms, n_aux(z))}}` plus the atom indices, matches `OnsitePredictor.dense`). Features from `onsite.site_graph` + `onsite._features` (do not write a second feature path). A compiled `CentredPredictor` like `OnsitePredictor`: neighbour list on the host, features and readout as compiled programs, the model's arrays passed per call.
- **m convention**: component m of shell lam is `eval.harmonics`' real Y_{lam, m}, index m + lam, the same as the features. The converter is responsible for bringing data to it.
- The site basis needs only the parities (-1)^lam up to the largest auxiliary lam: build it with `build_site_basis(elements, order, totaldegree, lmax_feat=lam_max, rcut=..., ...)` and check what parities it returns; if it builds both, the unused ones cost evaluation time, so add a way to ask for natural parity only if that is cheap, else leave it and say so in the results.

### `operators/fit.py`: `fit_centred`

Two losses. Both go through the existing statistics and solves; no new solver.

1. **`metric="plain"`** (per-coefficient least squares). Exactly `fit_onsite`'s structure: one design per (z, lam), columns the features of (lam, p) (+ constant for lam = 0), rows (atom, m), every shell of that lam and channel a right-hand side. Call `_fit(designs, n_structures, rows, nc, ridge, gamma=..., select=..., groups=...)` unchanged. This is the fallback when the data has no auxiliary overlap, and the cheap baseline.

2. **`metric="overlap"`** (default when every training structure has the overlap). Minimise sum over structures of (P b - c)^T S (P b - c): S the structure's auxiliary overlap (n_aux_total square, symmetric positive definite), P the map from all parameters to the structure's coefficient vector. S couples shells, lam and atoms, so there is **one design** whose columns are all the parameters of all species (prototype: `sfit.big_design`). Whitening makes it ordinary least squares: with S = L L^T, rows L^T P and targets L^T c. Feed these rows to `_fit` as a single design key with one right-hand side per channel, one chunk per structure, `groups` = the structure index. Then evidence, leave-structure-out selection, the closed-form jackknife (`_jackknife`) and `DesignStats` come for free.
   - P is block-sparse (an atom's rows touch only its species' columns of its lam): assemble L^T P by blocks, never P as a dense (n_aux_total, n_par) array times a dense S if that doubles memory needlessly; at the prototype's sizes (6,000 rows, 394 columns) anything works, but write it so that 10^4 rows by several thousand columns streams.
   - The Cholesky of S is O(n^3) per structure and S is O(n^2) memory. That is the method's scaling limit: record the measured time and memory for 32, 64 and 256 atoms in the results. A truncated or sparse S is out of scope.
   - If S is not positive definite to roundoff (near-linear dependence of the auxiliary basis), use the eigen-decomposition with eigenvalues clipped at 1e-12 of the largest, and report how many were clipped.

Common to both:

- **Prior**: `sigma_c=None | number | "evidence" | "loo" | "press"` with `SiteBasis.smoothness(lam, p)` for the feature columns and 0 for the constant columns, as `fit_onsite`. The prototype used a ridge of r times the diagonal of the Gram on the non-constant columns, r = 1e-6; that is `gamma` = r diag(G), which `_fit` accepts as a given prior, so the prototype can be reproduced for the acceptance test. Which default (evidence, press, or the prototype's ridge) is best is a measurement for the results, not decided here.
- `stats=True` returns the `DesignStats`, with `jack` when structures were kept apart. Note for the implementer: today `_fit` keeps per-structure factors only when `select` is "loo" or "press". Add a way to keep them at a fixed prior (for the jackknife without prior selection), in `_fit` itself, so that `fit_onsite` and `fit_offsite` get it too; the Hamiltonian UQ needs the same. A working monkey-patched version is `/storage/eng/essswb/projects/ace-jax/ham-ref/work/aluq/akjk.py`.

### `operators/io.py`

One more optional part in the model file, beside `onsite` and `offsite`: `centred_keys` (n, 2) (z, shell), `centred_ptr`, `centred_coef`, and in `meta_json` `centred: {shells: {"Z": [l, ...]}, channels: [...]}`. `save(path, onsite=None, offsite=None, centred=None, info=None)`; `OperatorFile` gains `centred` (default None, so existing callers and files are unchanged; `schema_version` stays 1 since old files load). A file may hold a site basis and a centred model only. A loaded model predicts the saved one's coefficients bit for bit (test).

### `ham/density.py` (application: the electron density)

The `ham` extra already carries what it needs (h5py is not required; npz is enough).

- **`DensityData`** (NamedTuple): `positions, cell, numbers, pbc, shells: OrbitalBasis, coefficients: (n_aux_total,)` in atom order then shell order then m, `overlap: (n_aux_total, n_aux_total) | None`, `charges: {z: (n_aux(z),)} | None` (the integral of each auxiliary function; zero for lam > 0), `nelectrons: float | None`.
- **Neutral format**: one `.npz` per structure with those names and a `meta_json` (`schema_version` 1, `kind` "density", the m convention stated). `read(path)`, `write(path, data)`, and `names(dir)` listing a directory of them. Document the schema in the module docstring, as `operators/io.py` does.
- **Electron count**: N = sum_i q_{z_i} . c_i. After prediction, `conserve(c, numbers, charges, nelectrons)` applies the smallest Euclidean change that makes it exact: c_i -= q_{z_i} dN / sum_j |q_{z_j}|^2. Return dN before the correction as well: it is a free out-of-distribution flag (0.01 to 0.08 e in distribution, up to 7.8 e on slabs) and belongs in every prediction's output.
  - Charges come from the data. If no structure has them, fit them from the training labels (sum over atoms of the lam = 0 coefficients against the electron count, least squares) and warn: the prototype found that fit ill-conditioned on six structures and replaced it by measured charges.
  - The model file records the charges and electrons per atom of each element under `info`.
- **`DensityPredictor` / `predict(model_file, atoms)`**: coefficients, corrected, with dN.
- **Metrics** (`density_metrics(ref, pred)`): the density error sqrt(dc^T S dc) and the same relative to sqrt(c^T S c) when the overlap is there; coefficient RMSE per lam relative to the spread of the labels about their mean; dN. A summary over structures with median and worst, as `ham.metrics.summary`.

### Command line

Mounted from `ham/cli.py` the way `aj ham` is (the parser lives with the application; `ace_jax.cli` mounts it):

- `aj density fit --data DIR --out model.npz [--validation DIR] [--metric overlap|plain] [--prior none|NUMBER|evidence|loo|press] [--order 1] [--site-rcut 7] [--site-radials 12] [--lmax L] [--ridge R] [--max-structures N]`. Defaults: order 1, rcut 7, 12 radials, total degree 40, wL 0.5 (the prototype's "o1"). Prints rows, parameters, training residual, and with `--validation` the metrics.
- `aj density predict --model model.npz --data DIR --out DIR`: one neutral file per structure with the corrected coefficients, and dN printed per structure.
- `aj eval --model model.npz --data DIR` on a density model prints the metrics (`aj eval` already branches on the model file's kind for operator models: branch on the presence of the centred part).

## Tests (synthetic data only; `tests/test_operators_centred.py`, `tests/test_ham_density.py`)

- Equivariance: rotate and invert a random structure; each shell's predicted vector transforms with the Wigner matrix of lam and the sign (-1)^lam (`tests/test_operators_bond.py::wigner` has the helper: move it to `conftest.py` and import it). Translation and atom permutation.
- Exact recovery: coefficients generated by a random `CentredModel` are refitted to roundoff by both metrics (two species, shells with lam up to 3, a random positive definite S for the overlap metric).
- `metric="overlap"` with S = identity equals `metric="plain"` to roundoff.
- With a random S and noisy targets, the overlap fit has the smaller S-weighted residual and the plain fit the smaller plain one (they are different minimisers).
- The closed-form jackknife deviations equal explicit refits without each structure (fixed prior), to 1e-8 relative.
- Electron count: exact after `conserve`; dN returned equals the direct sum; two species.
- File round trips: model file bit for bit; neutral data file; an old operator file without the centred part still loads.
- Command line smoke test: fit, predict, eval on four small synthetic structures.
- `tests/test_import_layers.py` passes with no new `ALLOWED` entry; `tests/test_core_deps.py` passes.

## Acceptance on real data (outside the repository; report in `docs/dev/ham-results.md`)

Data: `/storage/eng/essswb/projects/ace-jax/ham-ref/work/salted/data/s000` to `s039` (aluminium bulk; training s000 to s003, s007, s008; test s035 to s039) and `harsh/` (out of distribution). Each run directory has the DFT code's coefficient file and, where `ri_ovlp.npy` exists, the overlap; measured charges in `q_exact.npy`. Write the converter to the neutral format in `~/projects/ace-jax-ham-ref/` (not in ace-jax), from `scripts/fit.py`'s `split`/`join`/`smat`, which hold the file layout.

1. With the prototype's prior (r = 1e-6 times the Gram diagonal, constants unpenalised) and the overlap metric on the same six structures, the predicted coefficients of s035 to s039 before the count correction agree with `pred_o1/n6_r1e-06/s03*/ri_restart_coeffs.raw` to 1e-6 relative in the S norm. This pins the conventions (m order, constant columns, whitening).
2. Test density error no worse than the prototype's 0.0135 to 0.0236 per cell with the default prior chosen for the command.
3. Table: plain against overlap metric; prior none / evidence / press; order 1 against order 2; fit time and memory against cell size.
4. The jackknife of the density model against its true density error on the harsh cells, as was done for the H model (rank correlation, scale). This is new information, not a pass/fail gate.

The one-shot energies need the DFT code and are not part of this task; the owner's session reruns them from the predicted files.

## Out of scope, noted for later

- Forces or any derivative of the density with respect to positions.
- A sparse or truncated overlap for large cells; learning in a metric that needs no overlap.
- Element embedding (TrACE) site bases: `fit_centred` should take any `SiteBasis`, but only the categorical one is tested here.
- The Hamiltonian from the predicted density without the DFT code, and the projection-space consistency check with the H model (`docs/dev/ham-results.md`, "A consistency check without the DFT code"): both will sit on `DensityPredictor` and `HamiltonianPredictor`, so keep the predictor's output layout documented and stable.
- A spin density (two channels): the model and fit carry a channel axis from the start so that it is data, not a redesign.

## Documentation

- `docs/user/howto/` page for the density model (ASD-STE100, `--8<-- "float64.md"`, new terms into the Terms table of `concepts.md`), the CLI reference (generated), `skills/ace-jax/SKILL.md`, CLAUDE.md layout entries, CHANGELOG.
- Results in `docs/dev/ham-results.md` under a new dated section.
