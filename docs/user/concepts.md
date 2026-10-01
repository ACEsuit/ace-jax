# Concepts

## The ACE model

An ACE potential writes the energy as a sum of site energies,
E = Σᵢ Eᵢ, and each site energy is a linear combination of basis functions
of atom i's neighbour environment within a cutoff `rcut`:

Eᵢ = E0(zᵢ) + Σₖ cₖ(zᵢ) Bₖ(environment of i)

- The **B functions** are symmetric (rotation-, reflection- and
  permutation-invariant) polynomials of the neighbour positions. Each one
  couples up to `--order` neighbours (correlation order; body order = order
  + 1), built from radial functions times spherical harmonics. A **pair
  basis** of 2-body radial functions is added to them.
- `--max-degree` bounds the total degree of each B function, which is what
  sets the basis size. Each one-particle factor counts as n + wL·l (radial
  degree n, angular degree l; `--wL` defaults to 1.5, so angular degree
  costs more), following ACEpotentials' `TotalDegree`.
- The **coupling coefficients** that turn products of radial × spherical
  harmonic functions into invariant B functions are computed by the
  `ace-jax-coupling` library when a basis shape is first built, then cached.
- The fit regularises each basis function by the algebraic **smoothness
  prior** Γ (ACEpotentials' `algebraic_smoothness_prior`), which penalises
  high-degree functions more. `aj fit` rebuilds it from the basis when the
  basis does not carry it, and logs
  `gamma missing from <basis> -- rebuilt via basis.prior`; that message is
  expected.

The model maps edge vectors rᵢⱼ (not positions and a cell) to site energies,
and JAX differentiates that map once for forces and the virial. The same core
serves `aj eval`, ASE and LAMMPS.

Because E is linear in the coefficients c, fitting is a linear regression on
a design matrix whose rows are the energies, forces and virials of the
training configurations.

## E0: the reference energy

E0(z) is a fixed per-species energy: the site energy of an atom with no
neighbours, where the basis contributes nothing. The fit
targets E − Σᵢ E0(zᵢ), so E0 decides what the basis has to represent.

- A freshly built basis has E0 = 0, and `aj fit --e0 model` (the default)
  uses the E0 stored in the basis.
- `aj fit --e0 lsq` fits E0 by least squares on the training energies, so
  the basis only has to represent energy differences. Use it for a freshly
  built basis.
- An **isolated-atom** configuration (one atom, no neighbour within the
  cutoff) is predicted as E0 alone. With `--e0 lsq`, a species that has one
  in the training set takes its energy as E0 exactly, and the remaining
  species are fitted by least squares to the other configurations (see the
  [FAQ](faq.md#how-do-i-use-isolated-atom-energies-as-e0)).

## The radial basis

Each one-particle basis function is a radial function Rₙₗ(r) times a
spherical harmonic. The radial functions are built from a polynomial basis
Pq(x) in a transformed distance x, mixed by a weight array:

Rₙₗ(r) = envelope(x) · Σq W[zᵢ, zⱼ, n, q] · Pq(x)

- `--radial-mode` (on `aj fit` and `aj basis`) chooses the initial weights
  W. `onehot` is the linear-ACE (ACE1) convention: each radial is one
  polynomial times an indicator of one neighbour species, so for a single
  element Rₙ = Pₙ. It is the default. `glorot_normal` uses seeded random
  mixtures of the polynomials: a random starting point for learned radials,
  several times worse kept frozen. `aj fit` keeps W frozen and fits only the
  readout c.
- **Learned radials** optimise W itself by variable projection (VarPro):
  for any W the best readout c is a closed-form ridge solve, so the optimiser
  works on W alone and c is projected out exactly. A held-out gate then
  keeps either the learned or the initial radials, whichever predicts the
  held-out energies and forces better. See
  [Learn the radial basis](howto/learned-radials.md) and
  [Tutorial 2](tutorials/learned-radials.md).
- At deployment a learned radial is converted to a cubic spline (to 1e-10)
  so it evaluates as fast as a stock model; see
  [Learned radials at deployment](howto/learned-radials.md#deployment-spline-speed).

### Multi-element models

With several elements the default basis is **categorical**: each species
pair has its own radial functions, so the basis grows quickly with the
number of elements. `aj fit --basis-embedding <table.json>` (or
`aj basis --embedding <table.json>`) instead builds a **species-embedded**
basis, where the neighbour species enters through a frozen embedding vector
(for example taken from a MACE model) and `--d-max` caps the channel widths.
`identity` in place of a table uses a one-hot embedding.

## Fitting: Bayesian linear regression

`aj fit` treats the fit as Bayesian linear regression. Its hyperparameters θ
(the noise scales σ_E, σ_F, σ_V and the prior scale of the coefficients) are
chosen by maximising the log marginal likelihood (the *evidence*) plus a
weak log-normal hyperprior. That replaces hand-tuned energy, force and virial
weights and a hand-tuned regulariser.

- **Linear ACE** (`--m-per-species 0`). The features are the ACE basis B.
  The posterior mean gives the coefficients, written to `model.npz`, an
  ordinary ACE model file. The posterior covariance gives a predictive σ
  (`--uq blr`, the default).
- `--opt lbfgs` maximises the evidence with L-BFGS, `--opt adam` (the
  default, 500 steps) with Adam. L-BFGS is much faster on small data.
- The design rows are streamed into sufficient statistics, so memory scales
  with the number of basis functions squared, not with the number of
  configurations.

### Other fit arms

These are available from the same command and documented in the
[CLI reference](reference/cli.md). They are research options: the tutorials
do not use them.

- **POPS** (`--uq pops`, linear model only): pointwise-optimal parameter sets
  (Swinburne and Perez, [arXiv:2402.01810](https://arxiv.org/abs/2402.01810)),
  an estimate of misspecification uncertainty. It changes only the
  uncertainty; the mean is the Bayesian-regression mean.
- **ARD** (`--uq ard`, linear model only): an automatic-relevance-determination
  posterior with a per-atom force σ calibrated on a held-out part of the
  training set; writes `posterior.npz` next to `model.npz` and changes the
  mean as well.
- **Hybrid ACE + GP** (`--m-per-species M`, M > 0, the CLI default 500): a
  Gaussian-process correction on M inducing sites per species, fitted on the
  features [B | k_θ(B, B_M)]. The fitted model is `gp_model.npz`, loaded by
  `GPCalculator.from_file`. `--rungs` adds hyperparameter-posterior
  approximations (`laplace`, `pathfinder`, `vi`, `nuts`) beyond the default
  `map`; they cost far more than the MAP (the Laplace rung can take tens of
  minutes to compile).

## Reading the metrics

`aj fit` writes `metrics.csv` (test set) and, with `--ood`, `metrics_ood.csv`:
one row per rung and quantity, with energies in meV/atom, forces in eV/Å
and virials in eV.

| Column | Meaning |
|---|---|
| `rmse`, `mae` | errors of the mean prediction |
| `crps` | continuous ranked probability score of the predictive distribution (lower is better) |
| `coverage` | fraction of errors within ±1σ; about 0.68 when σ is calibrated |
| `rms_z` | root-mean-square of error / σ; about 1 when calibrated, above 1 when σ is too small |
| `rho` | rank correlation between \|error\| and σ: does σ point at the large errors? |
| `sigma_ratio`, `median_sigma` | spread and median of the predicted σ |

The linear model's σ comes from the posterior alone, which does not know
that the model is misspecified. On small datasets expect `coverage` well
below 0.68 and `rms_z` above 1: treat that σ as a ranking, not a calibrated
error bar.

## Model files

| File | Written by | Loaded by |
|---|---|---|
| `basis.npz` (any name) | `aj basis`, `ace_jax.basis.export.save_npz`, or an ACEpotentials export | `aj fit --model` |
| `model.npz` | `aj fit` (linear arm), the radial learner | `aj.load`, `ACECalculator`, `aj eval`, `export_lammps`, `aj fit --model` |
| `fit.yaml` | every `aj fit` | `aj fit --config` |
| `gp_model.npz` | `aj fit` (GP arm) | `GPCalculator.from_file`, `aj eval` |
| `posterior.npz` | `aj fit --uq ard` | `ACECalculator(..., posterior=)`, `aj eval --posterior` |
| `*.yace` | pacemaker, or `ace_jax.eval.write_yace` | `aj.load`, `ACECalculator`, `aj eval`, `export_lammps` |

An ACE `.npz` holds the basis definition, the splined or analytic radials, the
coupling table and the coefficients, with its metadata in a `meta_json` entry
(`schema_version: 1`). An unfitted basis and a fitted model have the same
format: a fitted model is the basis with its coefficients and E0 filled in.
`.yace` models evaluate and export, but `aj fit` needs an ACE `.npz`.
