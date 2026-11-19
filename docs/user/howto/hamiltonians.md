# Hamiltonian models

A Hamiltonian model predicts the Hamiltonian matrix H and the overlap matrix S
of a structure in a basis of atomic orbitals. The bands, the density of states
and the band energy come from these matrices without a self-consistent
calculation.

ace-jax fits the model to the matrices of DFT calculations. The fit is a linear
least-squares problem. It has no training loop and needs only a few structures.

!!! warning
    Hamiltonian models are new. The command names, the options and the model
    file can change in the next releases.

--8<-- "float64.md"

## Install

1. Add `ham` to the extras that you install: `pip install "ace-jax[ham]"`.

The `ham` extra installs h5py, which reads and writes the DeepH file layout.

## The data

The training data is a set of structures with blocks. A block is the part of H
or S that connects the orbitals of one atom with the orbitals of a second atom.
An on-site block connects an atom with itself. An off-site block connects two
different atoms, or an atom and its periodic image.

`aj ham` reads a directory that contains one of these:

- subdirectories in the DeepH layout, one for each structure
  (`hamiltonians.h5`, `overlaps.h5` and the structure files);
- operator cache files (`*.npz`), one for each structure, written by
  `ace_jax.ham.cache.save`.

All structures of a dataset must have the same orbital basis. For FHI-aims
output, [Quoll.jl](https://github.com/maurergroup/Quoll.jl) writes the DeepH
layout.

A DeepH subdirectory with `up` and `dn` inside is a collinear-spin structure.
Its channels are `H_up`, `H_dn` and `S`.

## Fit a model

1. Put the training structures in one directory and the validation set in a
   second directory.
2. Run the fit.
3. Read the errors on the validation set.

```bash
aj ham fit --data train/ --validation validation/ --out gold.npz \
    --nelectrons 79:11 --kpts 4 4 4
```

The command prints the training residual, then the errors of the model on the
validation set, and writes the model file. The model file contains the basis,
the coefficients and a record of the fit.

`--nelectrons` gives the number of electrons that the bands contain. Give one
number for the cell, or `Z:n` pairs for the electrons of each atom of each
element. The default is the atomic number, which is correct for an
all-electron basis.

### The basis

The default basis is accurate for a metal with s, p and d orbitals. Change it
with these options:

| Option | Meaning |
|---|---|
| `--site-rcut`, `--site-radials` | cutoff and number of radial functions of the environment of each atom |
| `--bond-radials` | number of radial functions of the distance between the two atoms of a block |
| `--bond-rcut` | longest distance that has a block. The default is the longest distance in the data |
| `--coupled N L` | features that couple the bond with the environments of its two atoms |
| `--env R N L NB LB` | features of the atoms near the bond. This group gives most of the accuracy and most of the cost |
| `--no-env` | no features of the atoms near the bond. The model is fast and less accurate |

### The prior

`--prior` sets a smoothness prior on the coefficients.

- `none` (the default) fits with no prior. Use it when the data is from one
  material and the structures are similar.
- `loo` selects the prior scale by leaving one structure out at a time. Use it
  for datasets with many elements or few structures.
- `evidence` selects the prior scale by the Bayesian evidence.
- A number sets the scale directly.

!!! warning
    A fit with no prior can give an eigenvalue far from the bands when the
    overlap matrix is almost singular. The block errors do not show this
    problem. If the validation set shows a structure with a large band error,
    fit again with `--prior loo` or `--prior evidence`.

## Read the errors

`aj ham fit --validation` and `aj eval --model gold.npz --data validation/`
print the same tables.

- **Blocks**: the error of the matrix elements, for the on-site and the
  off-site blocks of each channel.
- **Bands at a common zero**: the eigenvalue error with the two spectra as
  they are.
- **Bands from own Fermi levels**: the eigenvalue error after each spectrum is
  moved to its own Fermi level. Most publications give this number.
- **Fermi level** and **band energy for each atom**: the errors of these two
  quantities.
- **Density of states**: the Wasserstein distance between the two densities
  of states. It is the mean distance that the states must move.

The eigenvalue errors include the states from `--window LO HI` relative to the
reference Fermi level. The summary names each structure with a band error
larger than 5 times the median.

Energies are in the unit of the data. For FHI-aims data the unit is eV, and the
tables show meV.

## Predict H and S

1. Give the model file and the structures: an extxyz file, or a directory of
   structures with blocks.

```bash
aj ham predict --model gold.npz --data structures.xyz --out predicted/
```

The command writes one subdirectory in the DeepH layout for each structure,
with `hamiltonians_pred.h5` and `overlaps_pred.h5`.

## Bands and density of states

1. Give the model file and one structure.

```bash
aj ham bands --model gold.npz --structure validation/ --index 0 \
    --nelectrons 79:11 --out bands.npz --plot bands.png
```

The command calculates the bands on the high-symmetry path of the cell and the
density of states on the `--kpts` grid. `--path` sets a different path, for
example `GXMG`.

If the structure has blocks, the command also calculates the bands of the
reference and prints the errors of the model. The figure then shows the two
sets of bands together. `--plot` needs matplotlib.

## Use the model from Python

`HamiltonianCalculator` is an ASE calculator.

```python
import jax
jax.config.update("jax_enable_x64", True)

from ace_jax.ham.calculator import HamiltonianCalculator

calc = HamiltonianCalculator("gold.npz", kpts=(4, 4, 4), nelectrons={79: 11}, width=0.1)
atoms.calc = calc
atoms.get_forces()
calc.get_eigenvalues(kpt=0)
calc.get_fermi_level()
H = calc.get_hamiltonian(atoms)      # {(Rx, Ry, Rz, i, j): block}
calc.write_results("predicted")      # DeepH layout
```

!!! warning
    The energy of this calculator is the band energy. It is not a total
    energy: it has no repulsive term. The forces and the stress are the
    derivatives of the band free energy only.

`threads=` sets the number of CPU threads for the off-site blocks.
