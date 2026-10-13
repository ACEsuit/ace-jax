# Concepts

This page explains the model, the reference energy, the radial basis, the
fit and the metrics. [Terms](#terms) at the end of the page defines the
words that these pages use with a special meaning.

## The ACE model

An ACE potential writes the energy as a sum of site energies,
$E = \sum_i E_i$. Each site energy is a linear combination of basis
functions of the neighbour environment of atom $i$, in a cutoff `rcut`:

$$
E_i = E_0(z_i) + \sum_k c_k(z_i)\, B_k(\text{environment of } i)
$$

- The **B functions** are symmetric polynomials of the neighbour positions:
  they do not change under rotation, reflection or permutation. Each B
  function couples a maximum of `--order` neighbours (the correlation order;
  body order = order + 1). The B functions are products of radial functions
  and spherical harmonics. A **pair basis** of 2-body radial functions is
  added to them.
- `--max-degree` sets the maximum total degree of each B function. This
  sets the basis size. Each one-particle factor counts as $n + w_L\, l$,
  with radial degree $n$ and angular degree $l$. `--wL` is 1.5 by default,
  so angular degree costs more than radial degree. This follows
  `TotalDegree` in ACEpotentials.
- `--nmax-by-order` and `--lmax-by-order` set a maximum $n$ and $l$ for
  each correlation order, in a comma-separated list that starts at order 1.
  If the list is shorter than `--order`, its last value applies to the
  higher orders. They are the `nradmax_by_orders` and `lmax_by_orders`
  settings of pacemaker. With `--wL 1`, `--max-degree` then gives the power
  order of pacemaker.
- The **coupling coefficients** turn products of radial functions and
  spherical harmonics into invariant B functions. The `ace-jax-coupling`
  library calculates them at the first build of a basis specification.
  ace-jax then keeps them in a cache.
- The fit regularises each basis function with the algebraic **smoothness
  prior** $\Gamma$ (`algebraic_smoothness_prior` in ACEpotentials). This
  prior penalises high-degree functions more. If the basis does not contain
  the prior, `aj fit` builds it again from the basis, and logs
  `gamma missing from <basis> -- rebuilt via basis.prior`. This message is
  normal.

The model maps edge vectors $\mathbf{r}_{ij}$ (not positions and a cell) to
site energies. JAX differentiates that map once to get the forces and the
virial. `aj eval`, ASE and LAMMPS all use the same core.

$E$ is linear in the coefficients $c$. Thus the fit is a linear regression on
a design matrix. The rows of this matrix are the energies, forces and virials
of the training configurations.

## E0: the reference energy

$E_0(z)$ is a fixed energy for each species. It is the site energy of an
atom with no neighbours, where the basis gives zero. The fit targets
$E - \sum_i E_0(z_i)$. Thus $E_0$ sets what the basis must represent.

- A new basis has $E_0 = 0$. `aj fit --e0 model` (the default) uses the
  $E_0$ in the basis.
- `aj fit --e0 lsq` fits $E_0$ together with the model:
    - It adds one linear coefficient for each species. The prior of this
      coefficient has a width of 1 eV, centred on a least-squares fit of the
      training energies to the composition.
    - The basis then represents only energy differences. If the basis has a
      constant site energy that it cannot remove, $E_0$ absorbs it. This
      offset then does not become an energy error.
    - Use `--e0 lsq` for a new basis.
    - If the training set has no isolated atom, the fitted $E_0$ is a
      reference level, not the energy of a free atom. It changes together
      with the basis, and the prior has an effect on its value.
    - You can compare the log-evidence of these fits between bases fitted to
      the same data. Do not compare it with `--e0 prefit` fits.
- `aj fit --e0 prefit` is the older two-step method. It sets $E_0$ to the
  least-squares fit, and then fits the coefficients.

!!! note
    POPS fits use the `prefit` method, also with `--e0 lsq`. ARD fits
    $E_0$ together with the coefficients, as the default fit does.

An **isolated atom** is a configuration with one atom and no neighbour in
the cutoff. The model predicts $E_0$ alone for it. With `--e0 lsq` or
`prefit`, if the training set has an isolated atom of a species, $E_0$ of
that species is set to the energy of that atom. The fit calculates $E_0$ for
the other species (see the
[FAQ](faq.md#how-do-i-use-isolated-atom-energies-as-e0)).

## The radial basis

Each one-particle basis function is a radial function $R_{nl}(r)$ multiplied
by a spherical harmonic $Y_{lm}$. The radial functions are mixtures of a
polynomial basis $P_q(x)$ in a transformed distance $x$, with a weight array
$W$:

$$
R_{nl}(r) = f_\text{env}(x) \sum_q W_{z_i z_j n q}\, P_q(x)
$$

- `--radial-mode` (on `aj fit` and `aj basis`) sets the initial weights $W$.
    - `onehot` (the default) is the linear-ACE (ACE1) convention. Each radial
      function is one polynomial multiplied by an indicator of one neighbour
      species. Thus, for one element, $R_n = P_n$.
    - `glorot_normal` uses seeded random mixtures of the polynomials. It is a
      random start for learned radials. If you keep it frozen, the fit is
      several times worse.
- `aj fit` keeps $W$ frozen and fits only the coefficients $c$.
- **Learned radials** optimise $W$ by variable projection (VarPro). For each
  $W$, the best coefficients $c$ are the solution of a ridge regression in
  closed form. Thus the optimiser changes only $W$, and $c$ follows exactly.
  Then a validation gate compares the learned radials with the initial
  radials on the validation set. It keeps the radials that predict the
  energies and forces better. See
  [Learn the radial basis](howto/learned-radials.md) and
  [Tutorial 2](tutorials/learned-radials.md).
- For deployment, ace-jax converts a learned radial to a cubic spline (to
  1e-10), so that its evaluation is as fast as a stock model. See
  [Learned radials at deployment](howto/learned-radials.md#deployment-spline-speed).

### Multi-element models

With more than one element, the default basis is **categorical**. Each
species pair has its own radial functions, so the basis becomes large
quickly as you add elements.

`aj fit --basis-embedding <table.json>` (or `aj basis --embedding
<table.json>`) builds a **species-embedded** basis:

- The neighbour species enters through a frozen embedding vector, for
  example from a MACE model.
- `--d-max` sets the maximum channel width.
- `identity` in place of a table gives a one-hot embedding.

[Tutorial 3](tutorials/multi-element.md) compares the two bases on a
five-element alloy.

## Fitting: Bayesian linear regression

`aj fit` uses Bayesian linear regression. The fit selects its
hyperparameters $\theta$ by maximising the log marginal likelihood (the
*evidence*) plus a weak log-normal hyperprior. The hyperparameters are the
noise scales $\sigma_E, \sigma_F, \sigma_V$ and the prior scale of the
coefficients. Thus you do not set energy, force and virial weights or a
regulariser by hand.

- **Linear ACE** (`--m-per-species 0`). The features are the ACE basis B.
    - The posterior mean gives the coefficients. The fit writes them to
      `model.npz`, a usual ACE model file.
    - The posterior covariance gives a predictive $\sigma$ (`--uq blr`, the
      default).
- The default optimiser is bounded L-BFGS (`--opt lbfgs`).
    - For the linear model, a small number of Newton steps then refine the
      result to a stationary point (`--map-polish`).
    - If a fit stops away from a stationary point, it gives a warning.
      `--strict` makes this an error.
    - `--opt adam` uses Adam for `--map-steps` steps. Adam can stop far from
      the optimum.
- **Noise for each quantity, or one shared noise** (`--noise`).
    - The default is `auto`. If you give `--weights`, `auto` uses `shared`.
      If you do not give weights, `auto` uses `per-quantity`. `fit.yaml`
      records the mode that the fit used.
    - `per-quantity` learns $\sigma_E$, $\sigma_F$ and $\sigma_V$
      separately. Each $\sigma$ cancels the `--weights` of its quantity,
      so the E:F:V weights have no effect.
    - There are many more force rows than energy rows. Thus the evidence
      gives a large $\sigma_E$, and the energies get too little weight.
    - `--noise shared` learns one $\sigma$ for all weighted rows. This is
      the Bayesian linear regression (BLR) of ACEpotentials. The weights set
      the balance, and the evidence sets only the noise level and the prior.
    - Use `shared` when your weights must set the balance, for example the
      ACEpotentials weights E 30 / F 1 / V 1. On the Si_tiny example, the
      shared-noise fit gives the same coefficients as the BLR fit of
      ACEpotentials.
    - Without weights, the default weights set the balance to E:F:V 1:1:1.
      On a Cantor alloy dataset, `shared` at these weights increased the
      energy error by 79 %. Thus `auto` uses `per-quantity` there.
    - `auto` uses `per-quantity` with `--sigma-type`, with
      `--uq ard` (the joint mode fits its own noise) and with
      `--solver lstsq`. These do not support `shared`.
- The fit adds the design rows to sufficient statistics in batches. Thus
  the memory increases with the square of the number of basis functions,
  not with the number of configurations.

### Other fit arms

The same command gives these options. The
[CLI reference](reference/cli.md) documents them. They are research
options, and the tutorials do not use them.

- **POPS** (`--uq pops`, linear model only): pointwise-optimal parameter
  sets (Swinburne and Perez,
  [arXiv:2402.01810](https://arxiv.org/abs/2402.01810)). It estimates the
  misspecification uncertainty. It changes only the uncertainty. The mean is
  the mean of the Bayesian regression.
- **ARD** (`--uq ard`, linear model only): an automatic relevance
  determination posterior with a calibrated per-atom force uncertainty.
    - The uncertainty is a jackknife uncertainty shape multiplied by group
      scales. The fit calculates the scales on a validation set taken from
      the training set.
    - It gives a standard deviation, a 3×3 covariance and a conformal radius.
    - It writes `posterior.npz` next to `model.npz`.
    - It also changes the mean.
    - See [Per-atom force uncertainty](howto/force-uncertainty.md) and
      [the mathematics](concepts/force-uncertainty-maths.md).
- **Hybrid ACE + GP** (`--m-per-species M`, $M > 0$; the CLI default is
  500): a Gaussian-process correction on $M$ inducing sites for each
  species.
    - It is fitted on the features $[\,B \mid k_\theta(B, B_M)\,]$.
    - The fitted model is `gp_model.npz`. Load it with
      `GPCalculator.from_file`.
    - `--rungs` adds approximations of the hyperparameter posterior
      (`laplace`, `pathfinder`, `vi`, `nuts`) to the default `map`. They cost
      much more than the MAP. The Laplace rung can take tens of minutes to
      compile.

## Reading the metrics

`aj fit` writes `metrics.csv` (test set). With `--ood`, it also writes
`metrics_ood.csv`. These files have one row for each rung and quantity.
Their units are:

- energies: meV/atom;
- forces: eV/Å;
- virials: eV.

`aj fit` and `aj eval` also print an RMSE table, with one row for each
configuration type. The units of this table are meV/atom for energies and
virials, and eV/Å for forces.

| Column | Meaning |
|---|---|
| `rmse`, `mae` | errors of the mean prediction |
| `crps` | continuous ranked probability score of the predictive distribution (lower is better) |
| `coverage` | fraction of errors in $\pm 1\sigma$; approximately 0.68 when $\sigma$ is calibrated |
| `rms_z` | root-mean-square of error / $\sigma$; approximately 1 when calibrated, more than 1 when $\sigma$ is too small |
| `rho` | rank correlation between \|error\| and $\sigma$: does a large $\sigma$ show a large error? |
| `sigma_ratio`, `median_sigma` | spread and median of the predicted $\sigma$ |

The $\sigma$ of the linear model comes from the posterior alone. The
posterior does not know that the model is misspecified. Thus, on small
datasets, `coverage` is usually much less than 0.68 and `rms_z` is more
than 1. Use this $\sigma$ to rank errors, not as a calibrated error bar.

## Model files

| File | Written by | Loaded by |
|---|---|---|
| `basis.npz` (any name) | `aj basis`, `ace_jax.basis.export.save_npz`, or an ACEpotentials export | `aj fit --model` |
| `model.npz` | `aj fit` (linear arm), the radial learner | `aj.load`, `ACECalculator`, `aj eval`, `export_lammps`, `aj fit --model` |
| `fit.yaml` | every `aj fit` | `aj fit --config` |
| `gp_model.npz` | `aj fit` (GP arm) | `GPCalculator.from_file`, `aj eval` |
| `posterior.npz` | `aj fit --uq ard`, `aj calibrate` | `ACECalculator(..., posterior=)`, `aj eval --posterior`, `aj calibrate` |
| `*.yace` | pacemaker, or `ace_jax.eval.write_yace` | `aj.load`, `ACECalculator`, `aj eval`, `export_lammps` |

An ACE `.npz` contains:

- the basis definition;
- the splined or analytic radials;
- the coupling table;
- the coefficients;
- the metadata, in a `meta_json` entry (`schema_version: 1`).

An unfitted basis and a fitted model have the same format. A fitted model is
the basis with its coefficients and E0 added. You can evaluate and export
`.yace` models, but `aj fit` needs an ACE `.npz`.

## Terms

These pages use the words below with one meaning only.

| Term | Meaning |
|---|---|
| configuration | one labelled entry of a dataset: atoms, an optional cell, and labels |
| cell | the periodic simulation box of a configuration |
| structure | an arrangement of atoms that you evaluate with a model |
| frame | one entry of an extxyz file, or one snapshot of an MD trajectory |
| dataset | a set of configurations |
| validation set | configurations taken from the training set and not used to fit the coefficients. They score a choice, for example learned radials or the ARD scales |
| basis | an ACE model with zero coefficients, ready to fit |
| basis specification | the settings that define a basis: elements, `order`, `max_degree` and the other `BasisSpec` fields |
| coefficients | the linear weights $c$ of the B functions (the *readout*) |
| arm | the model type that a fit uses: the linear model (`--m-per-species 0`) or the hybrid ACE + GP model |
| evidence | the log marginal likelihood of the training data |
| rung | one approximation of the hyperparameter posterior: `map` (the default), `laplace`, `pathfinder`, `vi`, `nuts` |
| uncertainty shape | the per-atom 3×3 matrix $V$ of `--uq ard`, which shows where the model is not certain and in which direction |
