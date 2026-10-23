# Evaluate many structures

`BatchCalculator` gives the energy, the forces and the stress of many
structures in a small number of calls. Use it for screening, finite
displacements, committee or active-learning pools, and other lists of small
cells.

One structure in each call keeps a GPU or a multi-core CPU mostly idle when the
structure is small. `BatchCalculator` puts structures of similar size into one
batch, makes the neighbour lists of the batch in one step, and evaluates the
batch in one compiled call.

The results are the results of `ACECalculator` with the same options. The two
agree to roundoff.

--8<-- "float64.md"

## Evaluate a list of structures

1. Make the calculator from the model file.
2. Give the list of structures to `calculate`.
3. Read the results in the order of the input list.

```python
import jax
jax.config.update("jax_enable_x64", True)

from ase.build import bulk
from ace_jax.batch import BatchCalculator

structures = []
for k in range(1000):
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.rattle(0.05, seed=k)
    structures.append(atoms)

bc = BatchCalculator("fit/model.npz")
res = bc.calculate(structures)
res.energy        # eV, shape (n_structures,)
res.forces        # list of arrays, eV/Å, shape (n_atoms, 3) each
res.stress        # eV/Å³, shape (n_structures, 6), Voigt order
```

The structures can have different sizes, cells, periodicity and species. A
cell can be smaller than the cutoff. Positions do not have to be inside the
cell.

The stress row of a structure is NaN when the structure is not periodic in
all three directions. `res.virial` (eV, shape `(n_structures, 3, 3)`) gives
the virial of every structure.

To attach the results to copies of the structures, use `to_atoms`:

```python
labelled = res.to_atoms(structures)
labelled[0].get_potential_energy()
```

## Options

`BatchCalculator` accepts the evaluation options of `ACECalculator`, with
the same meaning:

- `dtype`, `lean`, `spline_tol`, `radial_table` and `aa_dag`.
- `energy_reference="E0"`. `res.e0_offset` then gives the constant for each
  structure.
- `ffi=True` or `ffi="auto"`: the C++ kernel on a CPU. On a CPU, use the
  kernel. The JAX model gets almost no speed-up from batches on a CPU.

Two options control the batches:

- `max_batch_atoms` sets the maximum number of atoms in one call. The
  default is 16384.
- `n_device` sets the largest structure that goes into a batch. Larger
  structures go through `ACECalculator`, one structure at a time. The default
  is 256 atoms, and 16 atoms for the JAX model on a CPU.

`bc.last_timing` gives the time of each step, the number of compilations and
the neighbour-list method.

!!! note "Compilation"
    Each size group compiles once, on its first batch. The first call is
    therefore slow. Later calls with structures of similar sizes do not
    compile again.

!!! warning "Small lists are slow"
    A list of one or two structures is slower than `ACECalculator`. Use
    `BatchCalculator` for tens of structures or more.

## Speed

The table gives the speed-up against `ACECalculator` with one structure in
each call. The model is linear ACE for Si (correlation order 4, total degree
12). Each call has 1024 structures (256 for 216 atoms on the CPU).

| Atoms in each structure | GPU, float32 | GPU, float64 | CPU with the C++ kernel, float64 |
|--:|--:|--:|--:|
| 2 | 46x | 54x | 52x |
| 8 | 51x | 28x | 19x |
| 64 | 7.4x | 4.8x | 2.4x |
| 216 | 2.2x | 1.5x | 1.5x |

The GPU is an NVIDIA RTX 4000 Ada. The CPU has 32 cores.

## `aj eval`

`aj eval` uses `BatchCalculator` when you do not give a posterior or a GP
model. You do not have to change the command.

## Limits

- `BatchCalculator` gives the energy, the forces, the stress and the virial.
  It does not give uncertainties. For per-atom force uncertainty, use
  `ACECalculator` with `posterior=` (see
  [Per-atom force uncertainty](force-uncertainty.md)).
- The forces are derivatives with respect to positions only. For
  derivatives with respect to the coefficients, use the JAX model directly.
- `BatchCalculator` does not run MD or geometry optimisation.
