# Per-atom force uncertainty

`aj fit --uq ard` fits a linear ACE model and a **calibrated per-atom force
uncertainty**. For each atom of a new structure, the model gives:

- the standard deviation (`forces_std`) and the full 3×3 covariance
  (`forces_cov`) of the force error. Use them to propagate uncertainty.
- a **conformal radius** (`forces_q`). The force error is in this radius
  with a stated probability (0.9 by default), for atoms that are similar to
  the calibration atoms.
- the **group** of the atom (`forces_group`). The group tells you which
  calibration pool gave the scales.
- an optional **support flag** (`forces_support`). It marks the atoms that
  the calibration data does not support.

Use the calibrated force uncertainty when you need error bars, not only a
ranking. For example:

- to select the structures to label next;
- to find when a simulation leaves the training distribution;
- to put error bars on the forces at a defect.

Limits of use:

- Only forces are calibrated.
- `--uq ard` applies only to the linear model (`--m-per-species 0`). For the
  hybrid ACE + GP model, use `--uq ard-gp`
  ([From a GP fit](#from-a-gp-fit-uq-ard-gp)). `--uq ard-gp` is
  experimental.

[Force uncertainty: the mathematics](../concepts/force-uncertainty-maths.md)
gives the full method.

## Quick start

```bash
aj fit --order 3 --max-degree 10 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs --uq ard \
    --out fit_ard
```

The fit writes these files in addition to the usual files:

- `fit_ard/model.npz`: the model. Its coefficients are the **ARD posterior
  mean**, so `--uq ard` changes the mean as well as the uncertainty.
- `fit_ard/posterior.npz`: the posterior, the uncertainty shape and the
  scales of each group. Use this file only with the `model.npz` from the
  same fit.
- `fit_ard/ard.json`: a report. It contains the hyperparameters, the
  validation split (`"split"`), the transfer exponent and the group table
  (under `"groups"`).

### In Python

```python
import jax
jax.config.update("jax_enable_x64", True)   # required with posterior=

from ase.io import read
from ace_jax import ACECalculator

atoms = read("crack.xyz")
calc = ACECalculator("fit_ard/model.npz", posterior="fit_ard/posterior.npz")
atoms.calc = calc
F = atoms.get_forces()                              # eV/Å, (N, 3)
std = calc.get_property("forces_std", atoms)        # eV/Å, (N,)
q = calc.get_property("forces_q", atoms)            # eV/Å, (N,)
cov = calc.get_property("forces_cov", atoms)        # eV²/Å², (N, 3, 3)
group = calc.get_property("forces_group", atoms)    # int, (N,)
```

- The calculator calculates the uncertainty only when you ask for one of
  these properties. Energies and forces take the same time as without a
  posterior.
- The first request calculates `forces_std`, `forces_cov`, `forces_q`,
  `forces_q_mahal` and `forces_group` together. The calculator keeps them
  for that structure.
- `forces_std_every_call=True` adds `forces_std` to each calculation. This
  increases the time of each MD step.
- Give the model as the **file** from the same fit.
- Enable float64 before you make the calculator. If you do not, the
  calculator gives a `RuntimeError`.

### On the command line

```bash
aj eval --model fit_ard/model.npz --posterior fit_ard/posterior.npz \
    --data crack.xyz --per-atom crack_uq.xyz --support
```

`--per-atom` writes one extxyz frame for each input configuration, with
these per-atom arrays:

- `forces_pred`, `forces_std`, `forces_q` and `forces_group`;
- for an anisotropic posterior (the default), also `forces_cov` (9 columns,
  row-major) and `forces_q_mahal`;
- with `--support`, also `support_ok` and `support_q`.

To colour atoms by one of these arrays, load the file in OVITO or ASE. With
`--out`, `aj eval` also adds `ace_forces_std` to its predictions file.

## From a GP fit: `--uq ard-gp`

`--uq ard-gp` gives the same calibrated force uncertainty for the hybrid
ACE + GP model. The method is the same as `--uq ard`. The design has the
ACE columns and the GP columns $k(B, B_M)$, at the fitted GP
hyperparameters.

1. Fit the hybrid model with `--uq ard-gp`:

    ```bash
    aj fit --order 3 --max-degree 10 \
        --train train.xyz --test test.xyz \
        --e0 lsq --m-per-species 100 --opt lbfgs --uq ard-gp \
        --out fit_gp
    ```

2. Make the calculator from `gp_model.npz` and `posterior.npz` of the same
   fit:

    ```python
    import jax
    jax.config.update("jax_enable_x64", True)

    from ace_jax.calc.gp import GPCalculator

    calc = GPCalculator.from_file("fit_gp/gp_model.npz",
                                  posterior="fit_gp/posterior.npz")
    atoms.calc = calc
    std = calc.get_property("forces_std", atoms)        # eV/Å, (N,)
    ```

3. On the command line, give `--posterior` with the GP model:

    ```bash
    aj eval --model fit_gp/gp_model.npz --posterior fit_gp/posterior.npz \
        --data crack.xyz --per-atom crack_uq.xyz
    ```

The fit writes these files:

- `fit_gp/gp_model.npz`: the model. Its mean is the **ARD posterior mean**,
  as for `--uq ard`.
- `fit_gp/posterior.npz`: the posterior. It also contains the GP
  hyperparameters that the GP columns use.

The calculator gives `forces_std`, `forces_cov`, `forces_q`,
`forces_q_mahal` and `forces_group`, as for `--uq ard`.

On the Cantor benchmark, `--uq ard-gp` gives a slightly higher coverage at
crack tips than `--uq ard` (0.902 and 0.893). It ranks the atoms by error
slightly less well. It is experimental. Use `--uq ard` unless you need the
GP model.

These functions are for the linear model only:

- the support flag (`forces_support`, `aj eval --support`);
- `aj calibrate`.

A calculator refuses a posterior from the other model type. The error
message gives the correct calculator.

## What each quantity means

| Property | Shape | Units | Meaning |
|---|---|---|---|
| `forces_std` | (N,) | eV/Å | $\lambda_g\sqrt{\operatorname{tr}V}$: the rms length of the force-error **vector** $\lvert\Delta F\rvert$ (approximately, for `aniso`, where $\lambda_g$ is fitted on Mahalanobis scores). The std of one component is approximately `forces_std`/$\sqrt3$ |
| `forces_cov` | (N, 3, 3) | eV²/Å² | $\lambda_g^2 V$: the error covariance. Its trace is `forces_std`² |
| `forces_q` | (N,) | eV/Å | the conformal radius at `--ard-coverage`: $\lvert\Delta F\rvert \le$ `forces_q` with that probability |
| `forces_q_mahal` | (N,) | none | `aniso` only: the Mahalanobis radius $q_g$ of the ellipsoidal region |
| `forces_group` | (N,) | int | the calibration group of the atom (see [Groups](#groups)) |
| `forces_support` | dict | | `support_ok` (bool, N), `support_q` (N, score units), `n_eff` (per species) |

$V$ is the per-atom 3×3 **uncertainty shape**. It shows where the model is
not certain, and in which direction. $\lambda_g$ and $q_g$ are the two
**scales** of the group of the atom. The fit calculates them on validation
data. In one group, the rank order of atoms by `forces_std` or by `forces_q`
is the rank order by the uncertainty shape.

### `forces_std` or `forces_q`?

- **`forces_std` and `forces_cov`** are a Gaussian description. Their scale
  gives the standardised error a unit rms in each group. Use them to
  propagate uncertainty, in a likelihood, or as one number for each atom.
  They do not state a coverage.
- **`forces_q`** is a coverage statement. It does not assume a Gaussian.
  For an atom that is exchangeable with the calibration configurations of
  its group, $\lvert\Delta F\rvert \le$ `forces_q` with a probability of
  approximately `--ard-coverage` (default 0.9). Use it as an error bar, to
  flag atoms, or to select structures to label.

If you read `forces_std` as a Gaussian, the 90 % radius is approximately
$1.44\times$ `forces_std` (the $\chi_3$ quantile, 2.50, divided by
$\sqrt3$). The group ratio $r_g$ (see [Groups](#groups)) shows the
difference between this value and the actual radius. In groups with heavy
tails, `forces_q` is larger.

### Isotropic and anisotropic regions

At fit time, `--force-shape` sets the region that `forces_q` describes:

- **`aniso`** (the default): an ellipsoid,
  $\Delta F^\mathsf{T}(V+\epsilon I)^{-1}\Delta F\le q_g^2$, aligned with the
  directions in which the model is not certain. `forces_q_mahal` is $q_g$.
  `forces_q` is the **largest semi-axis** of the ellipsoid. Thus the sphere
  of radius `forces_q` contains the region, and $\lvert\Delta F\rvert\le$
  `forces_q` is true at least as frequently as the nominal coverage.
- **`iso`**: a sphere of radius `forces_q` $= q_g\sqrt{\operatorname{tr}V/3}$.

`aniso` is the default for these reasons:

- In the [validation](#validation), it was the only variant that met all
  coverage targets, including the crack tip, without calibration on target
  data. This was at the default validation fraction, `--ard-val-frac 0.2`.
  At 0.1, the isotropic variant also met all targets.
- At the crack tip, the isotropic region gave too little coverage (0.86,
  against a target of 0.88).

Both modes give `forces_cov`. In `iso` mode, the trace of `forces_cov` is
calibrated, but its orientation is the uncalibrated uncertainty shape.

## Groups

The fit calculates the scales separately for 8 **Mondrian groups** of atoms.
Thus a strained or under-coordinated atom is calibrated against similar
atoms, not against bulk atoms. For each atom:

- $z$ is its coordination in $r_1$, the first minimum of the training radial
  distribution function. $z^\star$ is the most frequent training
  coordination.
- $d$ is the distortion of its first shell: the std of the first-shell bond
  lengths divided by their mean.
- the group is $2\times\text{band}(d) + [z\ne z^\star]$. The band limits are
  at the 50th, 90th and 99th percentiles of $d$ over the training atoms.

Thus:

- Group 0 has bulk-like atoms: low distortion and normal coordination.
- Odd groups have atoms with an unusual coordination.
- Groups 6 and 7 have the most distorted 1 % of atoms, and atoms with fewer
  than two neighbours.

The fit sets these constants and keeps them in the posterior.
Recalibration never moves an atom to a different group.
`--ard-groups none` keeps only the coordination split (2 groups). If
percentiles are equal (for example, in perfect-lattice data), bands merge,
and there are fewer groups.

A group needs a minimum of `--ard-n-min` (default 20) calibration
configurations. A group with fewer configurations **borrows** scales, in
this order:

1. from the nearest band with the same coordination flag;
2. from the nearest band with the other coordination flag;
3. from all groups together.

A finite conformal radius needs a minimum of
$\lceil(1-\alpha)/\alpha\rceil$ configurations in a pool: 9 at coverage
0.9, 99 at 0.99. If this number is larger than `--ard-n-min`, it becomes the
minimum. If the pool of all groups is also too small, `q` is infinite,
`forces_q` is `inf`, and the fit logs a WARNING. To correct this, decrease
`--ard-coverage` or add configurations.

`aj calibrate` prints the **group table**. The table is also in
`posterior.npz` and in `ard.json` (under `"groups"`):

| Column | Meaning |
|---|---|
| `n_cfg` (`n_cfg_val`, `n_cfg_cal`) | calibration configurations in the group: from the validation set of the fit, and from `aj calibrate` sets |
| `n_atoms` | calibration atoms in the group |
| `lam_rms` | $\lambda_g$, the scale of `forces_std` |
| `q` | $q_g$, the conformal quantile of the scores |
| `r` | $q_g / (\lambda_g\,\chi_3^{-1}(1-\alpha))$: approximately 1 when `forces_std`, read as a Gaussian, gives the correct coverage; more than 1 for heavier tails |
| `merged` | `[g, g_src]` pairs: group `g` borrowed from `g_src` (`-1`: all groups together) |

Use `forces_group` to find the table row of each atom.

## Recalibrate on target data: `aj calibrate`

The fit calibrates on a validation set taken from the training set. Thus its
coverage applies to atoms that are similar to the training data. For a
regime that the training data does not cover well (for example crack tips,
interfaces or a new phase), recalibrate:

1. Label a small number of cells of that regime with the same reference
   method.
2. Run `aj calibrate` on these cells.
3. Use the new posterior with `aj eval`.

```bash
aj calibrate --model fit_ard/model.npz --posterior fit_ard/posterior.npz \
    --data crack_cells.xyz --out crack_posterior.npz
aj eval --model fit_ard/model.npz --posterior crack_posterior.npz \
    --data crack.xyz --per-atom crack_uq.xyz
```

!!! warning "Do not calibrate on training configurations"
    `aj calibrate` does not check for training configurations. Their errors
    are too small, so the scales also become too small.

- `aj calibrate` changes only the scales. The model, its mean, the
  uncertainty shape and the groups do not change. Thus recalibration takes
  minutes, but a refit takes hours.
- `--out` gives a new posterior file. The input posterior does not change.
  The coverage level is the level set at fit time.
- `--energy-key`, `--force-key` and `--virial-key` give the label names, as
  in `aj fit`. `aj calibrate` uses only the forces.
- `--ard-n-min` is a fit option. The posterior keeps its value, and
  `aj calibrate` uses that value.

The mode sets how the new scores and the stored validation scores combine
in each group:

| Mode | Pool of each group |
|---|---|
| default (per-group replace) | the new cells only, in groups where they have a minimum of max(the `--ard-n-min` of the fit, $\lceil(1-\alpha)/\alpha\rceil$) configurations; in other groups, the stored scores and the new scores together |
| `--append` | the stored scores and the new scores together, in all groups |
| `--replace` | the new cells only, in all groups (groups that become too small borrow, as in [Groups](#groups)) |

The default calibrates a group only on the target regime, and only when
there is sufficient target data for that group. Validation atoms are easier
than target atoms, so they make the quantile too small if they are mixed in.

!!! warning "A per-group-replace posterior applies only to its regime"
    In the validation, calibration on crack cells with the default
    per-group replace increased the crack-tip coverage to 0.90. This was the
    best result of all routes. But 54 crack configurations filled all 8
    groups, including the bulk-like groups. Thus they replaced the
    validation scores in all groups, and this posterior then gave **too
    little coverage in distribution** (0.87, against 0.90). `--append` kept
    the in-distribution coverage (0.895), but it changed the crack-tip
    coverage very little. The reason is that each group had hundreds of
    validation configurations, and the small number of new cells had little
    effect.

**Which posterior to use:**

- In general, use the `posterior.npz` from the fit.
- If you have labelled cells from a target regime, make a separate
  per-group-replace posterior. Use it only on structures of that regime.
- `--append` is safe in distribution. But if the new set is small compared
  with the validation set of the fit, it has little effect.

## The support flag

The coverage statement is correct only if the target atom is exchangeable
with the calibration atoms of its group. The support diagnostic does a
check of this from the structure alone. It does not need labels.

- For each species, a classifier on the site descriptors estimates how much
  more probable an atom is under the target data than under the calibration
  data. The diagnostic then calculates the conformal quantile again, with
  these weights.
- `support_ok` is `False` where no finite quantile is possible: the
  calibration data has too little weight near this atom to certify its
  coverage.
- `support_q` is the weighted quantile, in the units of the scores. Its pool
  is all calibration atoms of the species, in all groups. Thus you cannot
  compare it with the group value `q`.
- `n_eff` is the effective number of calibration atoms for each species.

Atoms with `support_ok = False` are candidates for labelling and for
`aj calibrate`. The flag is only a diagnostic: it does not change
`forces_std` or `forces_q`.

- `aj eval --per-atom ... --support` writes the flag.
- `calc.get_property("forces_support", atoms)` returns the dict.
- The fit builds the support reference. This takes a small amount of time.
  `--no-ard-support` skips it; the property then gives an error.

The classifier works in a whitened principal-component space of the site
descriptors. `--ard-support-features normalised` builds that space from the
unit-norm descriptor. It also adds the logarithm of the descriptor norm and
of the block norm of each body order, as separate channels. When an atom
loses its neighbours, its descriptor becomes smaller. The raw descriptor can
then stay inside the training data, but the log-norm channels do not. The
default is `raw`.

## Fit options

| Option | Default | Effect |
|---|---|---|
| `--force-shape` | `aniso` | `aniso` (ellipsoidal region, `forces_q_mahal`) or `iso` (spherical) |
| `--ard-coverage` | 0.9 | the nominal coverage $1-\alpha$ of `forces_q` |
| `--ard-groups` | `distortion` | 8 groups (distortion bands × coordination) or `none` (2) |
| `--ard-n-min` | 20 | the number of configurations a group needs so that it does not borrow |
| `--ard-val-frac` | 0.2 | the fraction of the training configurations kept for validation of the scales (stratified by group) |
| `--ard-transfer` | `exponent` | how the scale from the validation fit is applied to the model fitted on all data: `exponent` estimates the exponent $\beta$ for each fit, `sqrt` sets $\beta=\frac12$, `none` sets $\beta=0$ |
| `--ard-cluster-size` | 3 | the side of the spatial blocks that large training cells are divided into, in units of $r_\text{cut}$ (`inf`: whole configurations) |
| `--ard-press` | `exact` | `exact` leave-one-cluster-out correction, or the faster `block` approximation |
| `--ard-variance` | `sandwich` | the uncertainty shape: the jackknife (`sandwich`), or the posterior covariance (`kappa`) |
| `--ard-mode` | `joint` | evidence fit of the noise and prior scales together, or `sequential` (prior scales only; less memory) |
| `--no-ard-support` | | skip the support reference |
| `--ard-support-features` | `raw` | features of the support reference: `raw` descriptors, or `normalised` (unit-norm descriptor plus log-norm channels per body order) |
| `--batch-pack` | `auto` | batches configurations by an atom budget when a set has small and large cells |

With `--uq ard`, `--e0 lsq` fits E0 together with the coefficients, as the
default fit does:

- Each species has one E0 column, with the same fixed broad prior as the
  default fit. These columns are not in the ARD body-order groups.
- If the training set has an isolated atom of a species, the E0 of that
  species is fixed to its energy.
- `model.npz` contains the fitted E0.
- The E0 columns are zero on force rows, so they have no effect on the force
  uncertainties.

Posteriors written before this change load and work.

## Cost and memory

- **Fit.** In addition to the final fit, `--uq ard` does two validation
  evidence fits, on approximately 80 % and 64 % of the training
  configurations. The second fit estimates the transfer exponent;
  `--ard-transfer sqrt` or `none` skips it. The jackknife needs one pass
  over the training rows for each posterior. For training sets that have
  bulk cells and also cells of thousands of atoms, two methods limit the
  memory of the design rows of a large cell:
    - batches by atom budget (`--batch-pack auto`);
    - training statistics calculated in node chunks.
- **`posterior.npz`** contains the float32 $L\times L$ posterior factor
  (approximately 0.9 GB at $L$ = 15k basis functions). It also contains the
  shape factor, $L\times\min(K, L)$ for $K$ jackknife clusters (0.22 GB at
  $L$ = 15k, $K$ = 3.7k).
- **Evaluation.** The default method needs the force design rows of the
  full cell on the device: approximately $N\cdot3\cdot L\cdot 8$ bytes
  (7 GB for 100k atoms at $L$ = 3k), plus the posterior factors. The rows are
  built node by node and the uncertainty shape atom by atom, but all rows
  must fit in memory. On the Cantor benchmark (15k basis functions), a cell
  of 3.9k atoms needs approximately 6.4 GB on the GPU and 20 s on an
  A100-40GB. (`aj eval --no-deriv-dtc` is a different switch: it is for large
  cells with a GP model, not for `--uq ard`.)

  `--shape-path committee` (on `aj eval` and `aj calibrate`;
  `ACECalculator(..., shape_path="committee")`) calculates the same shape
  without the design rows. The shape is a sum of squared forces of a linear
  ACE model with $r$ coefficient vectors, one for each column of the shape
  factor. Thus the arrays are approximately $N\cdot3\cdot r\cdot 8$ bytes,
  not $N\cdot3\cdot L\cdot 8$. The values are the same to roundoff. Use it
  only for cells much larger than 3–4k atoms: at that size it is not faster
  and it needs more memory (approximately 13 GB).

  In Python, `shape_tau` and `shape_rank` truncate the shape factor to a
  lower rank. The scales are fitted for the full rank. Thus a truncated shape
  keeps the ranking of atoms (Spearman 0.99 at rank 200 of 3680), but its
  coverage is much too low (0.54, not 0.90). Use it only to rank atoms.

## Validation

Revision 2 was validated on a CrMnFeCoNi (Cantor) alloy, labelled by the
MACE-MH-1 foundation model. The test had 3680 training configurations, a
15k-function basis, and 34 large cells with cracks and dislocations that
were not in training. Results:

- The default (`aniso`, transfer exponent) met all coverage targets without
  target data. For example, the crack-tip coverage was 0.885, against a
  target of 0.88 or more.
- Without the transfer exponent, the validation scale was too small
  (crack-tip coverage 0.81). The error is mostly approximation error, so a
  scale fitted on 80 % of the data underestimates the error of the model
  fitted on all the data.
- Crack cells in training improved the crack tip (0.87–0.90), but not in all
  folds. The jackknife block size had no measurable effect.
- The rank correlation between `forces_std` and the actual error was
  0.29–0.38 on the large cells.

The [validation tables](https://github.com/ACEsuit/ace-jax/blob/main/docs/dev/ard-validation.md)
and the [acceptance report](https://github.com/ACEsuit/ace-jax/blob/main/bench/defect_uq/results/2026-10-03_rev2_acceptance.md)
give all results, with confidence intervals.

## Limits

- **Only forces are calibrated.** The calculator gives no energy or virial
  uncertainty. Under `--uq ard`, the energy and virial variances in the
  `aj fit` prediction files are the untempered posterior variances. They are
  not calibrated, and they can be too small on small training sets. On the
  small Si test fixtures, the test energy errors are approximately 1–3 times
  the predicted std (rms z-score ≈ 2.5). Before you use energy variances,
  examine the energy `rms_z` in `metrics.json`.
- **Coverage is marginal in each group**, for atoms that are exchangeable
  with the calibration configurations of the group. Only the support flag
  finds a shift that the groups do not resolve, for example a new phase or
  chemical order.
- **The uncertainty shape is a proxy.** It measures how much the prediction
  at an atom changes with the training clusters in the fit. The method uses
  it as a proxy for where the approximation error is large. It ranks errors,
  but it does not predict them.
- **Posterior files from before revision 2** (schemas 1 and 2) give only the
  scalar `forces_std`. The other properties give an error that tells you to
  refit with `--uq ard`.
