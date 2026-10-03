# Per-atom force uncertainty

`aj fit --uq ard` fits a linear ACE model together with a **calibrated
per-atom force uncertainty**. For every atom of a new structure it serves:

- a standard deviation (`forces_std`) and a full 3×3 covariance
  (`forces_cov`) of the force error, for propagating uncertainty;
- a **conformal radius** (`forces_q`): the force error is inside it with a
  stated probability, 0.9 by default, for atoms like the calibration atoms;
- the atom's **group** (`forces_group`), which says which calibration pool
  the scales came from;
- optionally a **support flag** (`forces_support`), which marks atoms the
  calibration data cannot vouch for.

Use it when you need error bars on forces that are calibrated rather than a
ranking: deciding which structures to label next, checking whether a
simulation leaves the training distribution, or attaching uncertainties to
forces at a defect. Only forces are calibrated. The fit is for the linear
model (`--m-per-species 0`); the hybrid ACE + GP arm has its own
uncertainty. The mathematics is written out in
[Force uncertainty: the mathematics](../concepts/force-uncertainty-maths.md).

## Quick start

```bash
aj fit --order 3 --max-degree 10 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs --uq ard \
    --out fit_ard
```

The fit writes, besides the usual files:

- `fit_ard/model.npz`: the model, with the **ARD posterior mean** as its
  coefficients (`--uq ard` changes the mean as well as the uncertainty);
- `fit_ard/posterior.npz`: the posterior, the uncertainty shape and the
  per-group scales; it belongs to this `model.npz` only;
- `fit_ard/ard.json`: a report (hyperparameters, the hold-out split, the
  transfer exponent, the per-group table under `"groups"`).

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

- The uncertainty is computed only when you ask for one of these properties;
  energies and forces cost the same as without a posterior. The first request
  computes `forces_std`, `forces_cov`, `forces_q`, `forces_q_mahal` and
  `forces_group` together and caches them for that structure.
  `forces_std_every_call=True` adds `forces_std` to every calculation, at a
  cost per MD step.
- The model must be given as the **file** written by the same fit, and float64
  must be enabled before the calculator is built (a `RuntimeError` says so).

### On the command line

```bash
aj eval --model fit_ard/model.npz --posterior fit_ard/posterior.npz \
    --data crack.xyz --per-atom crack_uq.xyz --support
```

`--per-atom` writes one extxyz frame per input configuration with the per-atom
arrays `forces_pred`, `forces_std`, `forces_q`, `forces_group`, and for an
anisotropic posterior (the default) `forces_cov` (9 columns, row-major) and
`forces_q_mahal`. `--support` adds `support_ok` and `support_q`. Load the file
in OVITO or ASE to colour atoms by any of them. With `--out`, `aj eval` also
adds `ace_forces_std` to its predictions file.

## What each quantity means

| Property | Shape | Units | Meaning |
|---|---|---|---|
| `forces_std` | (N,) | eV/Å | $\lambda_g\sqrt{\operatorname{tr}V}$: the rms of the force-error **vector** length $\lvert\Delta F\rvert$ (for `aniso`, where $\lambda_g$ is fitted on Mahalanobis scores, approximately); the per-component std is about `forces_std`/$\sqrt3$ |
| `forces_cov` | (N, 3, 3) | eV²/Å² | $\lambda_g^2 V$: the error covariance; its trace is `forces_std`² |
| `forces_q` | (N,) | eV/Å | the conformal radius at `--ard-coverage`: $\lvert\Delta F\rvert \le$ `forces_q` with that probability |
| `forces_q_mahal` | (N,) | none | aniso only: the Mahalanobis radius $q_g$ of the ellipsoidal region |
| `forces_group` | (N,) | int | the atom's calibration group (below) |
| `forces_support` | dict | | `support_ok` (bool, N), `support_q` (N, score units), `n_eff` (per species) |

Here $V$ is the per-atom 3×3 **shape**, which says where the model is unsure
and in which direction, and $\lambda_g$, $q_g$ are the two **scales** of the
atom's group, fitted on held-out data. Within a group, ranking atoms by
`forces_std` or by `forces_q` is ranking by the shape.

### `forces_std` or `forces_q`?

- **`forces_std` / `forces_cov`** are the Gaussian description, with the scale
  set so that the standardised error has unit rms in each group. Use them to
  propagate uncertainty, for a likelihood, or as a single number per atom.
  They make no coverage claim.
- **`forces_q`** is a coverage statement that does not assume a Gaussian:
  for an atom exchangeable with its group's calibration configurations,
  $\lvert\Delta F\rvert \le$ `forces_q` with probability about `--ard-coverage`
  (default 0.9). Use it as an error bar, to flag atoms, or to decide where to
  label.

Read as a Gaussian, `forces_std` implies a 90 % radius of about
$1.44\times$ `forces_std` (the $\chi_3$ quantile, 2.50, over $\sqrt3$). The
per-group ratio $r_g$ (below) says how far the actual radius is from that; on
heavy-tailed groups `forces_q` is larger.

### Isotropic and anisotropic regions

`--force-shape` (at fit time) chooses the region `forces_q` describes:

- **`aniso`** (the default): an ellipsoid,
  $\Delta F^\mathsf{T}(V+\epsilon I)^{-1}\Delta F\le q_g^2$, aligned with the
  directions in which the model is unsure. `forces_q_mahal` is $q_g$, and
  `forces_q` is the ellipsoid's **largest semi-axis**, so the sphere of radius
  `forces_q` contains the region and $\lvert\Delta F\rvert\le$ `forces_q` holds
  at least as often as the nominal coverage.
- **`iso`**: a sphere of radius `forces_q` $= q_g\sqrt{\operatorname{tr}V/3}$.

The default is `aniso` because, on the validation below, it is the only
variant without target-regime calibration that meets every coverage target
at the default hold-out fraction (`--ard-val-frac 0.2`; at 0.1 the
isotropic variant also met them),
the crack tip included; the isotropic region under-covers there (0.86
against 0.88). `forces_cov` is served in both modes; in iso mode its trace is
calibrated but its orientation is the uncalibrated shape.

## Groups

The scales are fitted separately for 8 **Mondrian groups** of atoms, so that
a strained or under-coordinated atom is calibrated against atoms like it
rather than against bulk. For each atom:

- $z$ is its coordination within $r_1$, the first minimum of the training
  radial distribution function, and $z^\star$ the most common training
  coordination;
- $d$ is the distortion of its first shell, the std over the mean of the
  first-shell bond lengths;
- the group is $2\times\text{band}(d) + [z\ne z^\star]$, with bands cut at the
  50th, 90th and 99th percentiles of $d$ over the training atoms.

So group 0 is low-distortion, normally coordinated (bulk-like), odd groups
have an unusual coordination, and groups 6 and 7 hold the most distorted 1 %
(and atoms with fewer than two neighbours). The constants are fixed at fit
time and stored with the posterior; recalibration never moves an atom to
another group. `--ard-groups none` keeps only the coordination split (2
groups). Tied percentiles (perfect-lattice data) merge bands, giving fewer
groups.

A group needs at least `--ard-n-min` (default 20) calibration
configurations; one with fewer **borrows** the scales of the nearest band
with the same coordination flag, then of the other flag, then of all groups
pooled. A finite conformal radius needs at least $\lceil(1-\alpha)/\alpha\rceil$
configurations in a pool (9 at coverage 0.9, 99 at 0.99), so the threshold
rises to that if it is larger. If even the pool of all groups is too small,
`q` is infinite, `forces_q` is `inf`, and the fit logs a WARNING: lower
`--ard-coverage` or add configurations.

The **per-group table** is printed by `aj calibrate`, stored in
`posterior.npz` and written to `ard.json` under `"groups"`:

| Column | Meaning |
|---|---|
| `n_cfg` (`n_cfg_val`, `n_cfg_cal`) | calibration configurations in the group: from the fit's hold-out and from `aj calibrate` sets |
| `n_atoms` | calibration atoms in the group |
| `lam_rms` | $\lambda_g$, the scale of `forces_std` |
| `q` | $q_g$, the conformal quantile of the scores |
| `r` | $q_g / (\lambda_g\,\chi_3^{-1}(1-\alpha))$: about 1 when reading `forces_std` as a Gaussian reproduces the coverage; above 1 for heavier tails |
| `merged` | `[g, g_src]` pairs: group `g` borrowed from `g_src` (`-1`: all groups pooled) |

`forces_group` joins each atom to its row.

## Recalibrate on target data: `aj calibrate`

The fit calibrates on a held-out part of the training set, so its coverage
holds for atoms like the training data. For a regime the training data does
not cover well (crack tips, interfaces, a new phase), label a few cells of
that kind with the same reference method and recalibrate:

```bash
aj calibrate --model fit_ard/model.npz --posterior fit_ard/posterior.npz \
    --data crack_cells.xyz --out crack_posterior.npz
aj eval --model fit_ard/model.npz --posterior crack_posterior.npz \
    --data crack.xyz --per-atom crack_uq.xyz
```

- Only the scales change: the model, its mean, the shape and the groups are
  untouched, so this takes minutes where a refit takes hours.
- The cells must **not** be training configurations (`aj calibrate` does not
  check): their errors would be too small, and so would the scales.
- `--out` is a new posterior file; the input posterior is not modified. The
  coverage level is the one chosen at fit time.
- `--energy-key`, `--force-key` and `--virial-key` name the labels as in
  `aj fit`; only forces are used.
- `--ard-n-min` is a fit option, stored in the posterior; `aj calibrate`
  uses the stored value.

How the new scores join the stored hold-out scores, per group:

| Mode | Pool of each group |
|---|---|
| default (per-group replace) | the new cells alone in groups where they have at least max(the fit's `--ard-n-min`, $\lceil(1-\alpha)/\alpha\rceil$) configurations; elsewhere the stored scores and the new ones together |
| `--append` | the stored scores and the new ones together, in every group |
| `--replace` | the new cells alone in every group (groups that end up too small borrow as above) |

The default calibrates a group against the target regime only when there is
enough target data for it, since mixing in easier hold-out atoms dilutes
the quantile.

!!! warning "A per-group-replace posterior is specific to its regime"
    In the validation below, calibrating on crack cells with the default
    per-group replace lifted crack-tip coverage to 0.90, the best of any route.
    But 54 crack configurations populate all 8 groups, the bulk-like ones too,
    so they replace the hold-out scores everywhere, and the same posterior
    then **under-covers in distribution** (0.87 against 0.90). `--append`
    keeps in-distribution coverage (0.895) but barely moves the tip, because
    the new cells are diluted among hundreds of hold-out configurations per
    group.

**Which posterior to serve.** Serve the fit's own `posterior.npz` in
general. With labelled cells from a target regime, build a separate
per-group-replace posterior and serve it only on structures of that regime.
`--append` is safe in distribution, but adds little when the new set is
small next to the fit's hold-out set.

## The support flag

The coverage statement needs the target atom to be exchangeable with its
group's calibration atoms. The support diagnostic checks this from the
structure alone, without labels:

- a classifier on each species' site descriptors estimates how much more
  likely an atom is under the target than under the calibration data, and
  the conformal quantile is recomputed with those weights;
- `support_ok` is `False` where no finite quantile is reachable: the
  calibration data has too little weight near this atom to certify its
  coverage. `support_q` is the weighted quantile, in the units of the
  scores; its pool is all calibration atoms of the species, across groups,
  so it is not comparable to the per-group `q`. `n_eff` is the effective
  number of calibration atoms per species.

Atoms with `support_ok = False` are candidates for labelling and for
`aj calibrate`. The flag is a diagnostic only: it never changes `forces_std`
or `forces_q`. `aj eval --per-atom ... --support` writes it;
`calc.get_property("forces_support", atoms)` returns the dict. The reference
is built at fit time and costs a little; `--no-ard-support` skips it (the
property then raises).

## Fit options

| Option | Default | Effect |
|---|---|---|
| `--force-shape` | `aniso` | `aniso` (ellipsoidal region, `forces_q_mahal`) or `iso` (spherical) |
| `--ard-coverage` | 0.9 | the nominal coverage $1-\alpha$ of `forces_q` |
| `--ard-groups` | `distortion` | 8 groups (distortion bands × coordination) or `none` (2) |
| `--ard-n-min` | 20 | configurations a group needs before it stops borrowing |
| `--ard-val-frac` | 0.2 | fraction of the training configurations held out to score the scales (stratified by group) |
| `--ard-transfer` | `exponent` | how the hold-out scale is carried to the model fitted on everything: `exponent` estimates the exponent $\beta$ per fit, `sqrt` fixes $\beta=\frac12$, `none` $\beta=0$ |
| `--ard-cluster-size` | 3 | side of the spatial blocks large training cells are split into, in units of $r_\text{cut}$ (`inf`: whole configurations) |
| `--ard-press` | `exact` | `exact` leave-one-cluster-out correction, or the cheaper `block` approximation |
| `--ard-variance` | `sandwich` | the shape: the jackknife (`sandwich`), or the posterior covariance (`kappa`) |
| `--ard-mode` | `joint` | evidence fit of noise and prior scales together, or `sequential` (prior scales only; lower memory) |
| `--no-ard-support` | | skip the support reference |
| `--batch-pack` | `auto` | size-aware batching: pack configurations by an atom budget when a set mixes small and large cells |

With `--uq ard`, `--e0 lsq` fits E0 jointly with the coefficients, as BLR
does: one E0 column per species with BLR's fixed broad prior, outside the
ARD body-order groups (a species with an isolated atom in training has its
E0 pinned). `model.npz` holds the fitted E0. The E0 columns are zero on
force rows, so force uncertainties are unaffected. Posteriors written before
this change still load and serve.

## Cost and memory

- **Fit.** Besides the served fit, `--uq ard` runs two hold-out evidence
  fits (on about 80 % and 64 % of the training configurations; the second
  estimates the transfer exponent, and `--ard-transfer sqrt` or `none` skips
  it) and one pass over the training rows per posterior for the jackknife.
  Training sets that mix bulk cells with cells of thousands of atoms are
  handled by size-aware batching (`--batch-pack auto`) and by node-chunked
  training statistics, which bound the memory of a large cell's design rows.
- **`posterior.npz`** stores the float32 $L\times L$ posterior factor (about
  0.9 GB at $L$ = 15k basis functions) and the shape factor,
  $L\times\min(K, L)$ for $K$ jackknife clusters (0.22 GB at $L$ = 15k,
  $K$ = 3.7k).
- **Evaluation.** The uncertainty needs the force design rows of the whole
  cell, about $N\cdot3\cdot L\cdot 8$ bytes on the device (7 GB for 100k atoms
  at $L$ = 3k), plus the posterior factors; the rows are built node by node
  and the shape atom by atom, but the rows themselves must fit. For cells of
  3–4k atoms at production basis sizes, `aj eval --posterior` and
  `aj calibrate` need an **A100-80GB-class GPU**; 40 GB runs out of memory.
  (`aj eval --no-deriv-dtc` is the corresponding switch for big cells with a
  GP model, not for `--uq ard`.)

## Validation

Revision 2 was validated on a CrMnFeCoNi (Cantor) alloy benchmark labelled by
the MACE-MH-1 foundation model: 3680 training and 920 test configurations
(bulk cells, vacancies, surfaces and stacking faults; a 15k-function basis),
and 34 large cells of 3–4k atoms with cracks and edge and screw dislocations,
none of them in training. Coverage of `forces_q` at nominal 0.90:

| Setting | In distribution (target 0.90 ± 0.01) | Crack (≥ 0.89) | Crack tip (≥ 0.88) | Edge, screw (≥ 0.90) |
|---|---|---|---|---|
| default (`aniso`, transfer exponent) | 0.898 | 0.903 | 0.885 | 0.937, 0.941 |
| `--force-shape iso` | 0.897 | 0.889 | 0.862 | 0.933, 0.937 |

`aj calibrate` on crack cells, leaving one crack realisation out (ranges over
the four folds; scored on the held-out crack pair and the dislocation cells,
and in distribution on a 300-configuration test sample):

| Posterior | In distribution | Crack | Crack tip | Edge, screw |
|---|---|---|---|---|
| `aniso`, uncalibrated (default) | 0.898 | 0.899–0.905 | 0.881–0.884 | 0.937, 0.941 |
| `aniso` + calibrate, per-group replace | **0.867–0.870** | 0.910–0.917 | 0.899–0.904 | 0.927–0.930, 0.932–0.935 |
| `aniso` + calibrate `--append` | 0.895 | 0.900–0.906 | 0.881–0.886 | 0.936, 0.941 |
| `iso` + calibrate `--append` | 0.895 | 0.895–0.903 | 0.869–0.878 | 0.934–0.935, 0.937–0.939 |

- The default meets every target without target data.
- Without the transfer exponent the hold-out scale is too small (crack tip
  0.81): the error is dominated by approximation error, so the scale fitted on
  80 % of the data underestimates the model fitted on all of it.
- Putting crack cells in training improves the tip (0.87–0.90) but not in
  every fold; the jackknife block size has no measurable effect.
- The rank correlation between `forces_std` and the actual error is 0.29–0.38
  on the large cells.

The full results, with confidence intervals and per-configuration-type
tables, are in the
[acceptance report](https://github.com/ACEsuit/ace-jax/blob/main/bench/defect_uq/results/2026-10-03_rev2_acceptance.md).

## Limits

- **Only forces are calibrated.** The calculator serves no energy (or
  virial) uncertainty. The energy and virial variances in `aj fit`
  prediction files under `--uq ard` are the untempered posterior variances,
  neither tempered nor calibrated, and they can be overconfident on small
  training sets: on the tiny Si test fixtures the test energy errors run at
  roughly 1–3 times the predicted std (rms z-score ≈ 2.5). Check the
  energy `rms_z` in `metrics.json` before relying on energy variances.
- **Coverage is marginal within a group**, for atoms exchangeable with its
  calibration configurations. A shift the groups do not resolve (a new
  phase, chemical order) is caught only by the support flag.
- **The shape is a proxy.** It measures how much the prediction at an atom
  depends on which training clusters were in the fit, and is used as a proxy
  for where the approximation error is large; it ranks errors, it does not
  predict them.
- **Posterior files from before revision 2** (schemas 1 and 2) serve only the
  scalar `forces_std`; the other properties raise and ask for a refit with
  `--uq ard`.
