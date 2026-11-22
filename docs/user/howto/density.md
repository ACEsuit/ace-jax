# Electron-density models

An electron-density model predicts the electron density of a structure. The
density is a sum of auxiliary functions on the atoms, and the model predicts
the coefficient of each function:

$$
\rho(\mathbf{r}) = \sum_i \sum_{s,m} c_{i s m}\, \phi_s(|\mathbf{r} - \mathbf{r}_i|)\, Y_{l_s m}(\mathbf{r} - \mathbf{r}_i)
$$

A DFT code can start from the predicted density. One Kohn-Sham solution from
this density then gives an energy and a Hamiltonian near the self-consistent
ones.

ace-jax fits the model to the density coefficients of DFT calculations. The fit
is a linear least-squares problem. It has no training loop and needs only a few
structures.

!!! warning
    Electron-density models are new. The command names, the options, the data
    file and the model file can change in the next releases.

--8<-- "float64.md"

## The data

The training data is a set of density files, one for each structure. A density
file is an `.npz` file that contains these arrays:

| Array | Contents |
|---|---|
| `positions`, `cell`, `numbers`, `pbc` | the structure. The cell vectors are the rows of `cell` |
| `coefficients` | the density coefficients of all atoms, as one vector |
| `overlap` | optional: the overlap matrix of the auxiliary functions of the structure |
| `charges_<Z>` | optional: the integral of each auxiliary function of the element `Z` |
| `radial_grid_<Z>`, `radial_values_<Z>` | optional: the radial functions of the auxiliary shells |
| `meta_json` | the auxiliary shells of each element, the number of electrons and the record of the calculation |

The coefficients have a fixed order. The atoms come in the order of the
structure. For each atom, the shells come in the order of the auxiliary basis.
For each shell of angular momentum $l$, the $2l+1$ components come in the
order $m = -l, \dots, l$ of the real spherical harmonics of ace-jax. The
overlap matrix has the same order.

!!! warning
    A DFT code has its own order and sign convention for the components of a
    shell. Convert the coefficients and the overlap to the order of ace-jax
    before you write the file. The fit cannot find a wrong convention.

ace-jax does not read the output of a DFT code. Write the density files from
your data in Python:

```python
import numpy as np
from ace_jax.ham import density as D
from ace_jax.operators.orbitals import OrbitalBasis

shells = OrbitalBasis.of({13: [0, 0, 1, 1, 2]})      # l of each auxiliary shell of aluminium
charges = {13: q}                                    # integral of each function, 0 for l > 0
data = D.DensityData(positions, cell, numbers, True, shells, coefficients,
                     overlap=S, charges=charges, nelectrons=416.0,
                     info={"fit_metric": "overlap", "kpts": [2, 2, 2], "self_consistent": True})
D.write("train/s000.npz", data)
```

`D.read` reads a density file. A density file with the name `density.npz` in
the DeepH directory of a structure is also a valid input. `D.read` and the
commands accept that directory.

The `info` dictionary is the record of the calculation. Write these keys when
you know them:

| Key | Meaning |
|---|---|
| `fit_metric` | how the DFT code calculates the coefficients from the density: `overlap`, `coulomb` or `unknown` |
| `kpts`, `kpt_offset`, `smearing`, `width` | the k-point grid and the occupation settings of the calculation |
| `self_consistent` | `true` for the density of a converged calculation |
| `hamiltonian` | the path of the Hamiltonian that the DFT code calculated from this density |

`aj density fit` gives a warning if the training structures have different
values of `fit_metric`, or different k-point and occupation settings.

## Fit a model

1. Put the density files of the training set in one directory and the density
   files of the validation set in a second directory.
2. Run the fit.
3. Read the errors on the validation set.

```bash
aj density fit --data train/ --validation validation/ --out density.npz
```

The command prints the number of rows, the number of parameters and the
training residual. Then it prints the errors of the model on the validation
set and writes the model file. The model file contains the basis, the
coefficients, the charges of the auxiliary functions and a record of the fit.

### The metric

`--metric` selects the quantity that the fit minimises.

| Metric | The fit minimises | Data that it needs |
|---|---|---|
| `overlap` | the integral of the squared error of the density, $\Delta c^T S\, \Delta c$ | the overlap matrix $S$ of each training structure |
| `plain` | the squared error of each coefficient | the coefficients only |

`overlap` is the default if each training structure has an overlap matrix.
Use it when you can. For aluminium, the density error with `plain` was six
times larger, and not smaller than the error of the mean coefficients.

!!! warning
    The `overlap` metric factorises the overlap matrix of each training
    structure. The time increases with the third power of the number of
    auxiliary functions in the structure, and the memory with the second
    power. Use small cells for the training set.

### The basis and the prior

| Option | Meaning | Default |
|---|---|---|
| `--order` | correlation order of the site features | 1 |
| `--site-rcut` | cutoff of the site features | 7.0 |
| `--site-radials` | number of radial functions, for each correlation order | 12, then 5 |
| `--lmax` | largest angular momentum of the one-particle functions, for each correlation order | the largest $l$ of the auxiliary basis, then 3 |
| `--scaled R` | a ridge of `R` times the diagonal of the Gram matrix, on each feature | 1e-6 |
| `--prior` | the smoothness prior: `none`, a number (its scale), or `evidence`, `loo`, `press` | `none` |
| `--ridge` | a ridge on each parameter | 1e-10 |

`--prior evidence` selects the scale of the smoothness prior with the evidence.
`--prior loo` and `--prior press` select it by leaving out one training
structure at a time. When you give `--prior`, the fit does not use `--scaled`.

## The electron count

The integral of the predicted density must be the number of electrons. The
model does not guarantee this. Each prediction is therefore corrected: the
coefficients change by the smallest quantity that makes the count exact.

The error of the count before the correction is `dN`. Each prediction gives
`dN`. It is a warning sign that costs nothing: `dN` is small for structures
that are similar to the training set, and large for structures that are
different from it.

The correction uses the charges of the auxiliary functions, that is, the
integral of each function. Put the charges in the density files (`charges_<Z>`).
If no training structure has charges, the fit calculates them from the electron
counts of the training set and gives a warning. Charges calculated from a few
structures are not accurate.

## Predict the density

```bash
aj density predict --model density.npz --data structures.xyz --out predicted/
```

`--data` is an extxyz file, or a directory of density files. The command writes
one density file for each structure in the output directory, with the corrected
coefficients. It prints `dN` for each structure and records it in the `info` of
the file.

In Python:

```python
from ase.io import read
from ace_jax.ham import density as D

predictor = D.DensityPredictor("density.npz")
pred = predictor.predict(read("structure.xyz"))
pred.coefficients        # corrected, atom then shell then m
pred.info["dN"]          # count error before the correction
```

ace-jax does not write the input files of a DFT code. Convert the predicted
coefficients to the format of your code.

## Score a model

```bash
aj eval --model density.npz --data validation/ --metrics-out metrics.json
```

`aj eval` prints these errors for each structure, and a summary with the
median structure and the worst structure:

| Quantity | Meaning |
|---|---|
| density error | $\sqrt{\Delta c^T S\, \Delta c}$, the L2 norm of the error of the density. It needs the overlap matrix of the reference structure |
| relative | the density error divided by $\sqrt{c^T S\, c}$ |
| coefficients | the RMSE of the coefficients divided by the spread of the reference coefficients. A value of 1 is the error of the mean coefficients |
| count error | `dN`, the error of the electron count before the correction |

The summary also gives the errors of the projections $w = S c$.

## Limits

- The model gives no forces and no derivative of the density.
- The model was tested with one element. A basis with an element embedding is
  not tested.
- The `overlap` metric is practical for training cells of up to a few hundred
  atoms.
