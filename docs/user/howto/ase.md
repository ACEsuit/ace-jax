# Use a model as an ASE calculator

`ACECalculator` evaluates any ace-jax model file, an ACE `.npz` or a PACE
`.yace`, as an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator: energy,
forces and stress.

```python
import jax
jax.config.update("jax_enable_x64", True)   # evaluate in float64 (see below)

from ase.build import bulk
from ace_jax import ACECalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = ACECalculator("fit/model.npz")
E = atoms.get_potential_energy()      # eV
F = atoms.get_forces()                # eV/Å, shape (n_atoms, 3)
S = atoms.get_stress()                # eV/Å³, Voigt order
```

The cutoff, the species and the precision all come from the model file, so
the path is usually the only argument. Species in `atoms` that the model does
not know raise an error.

## Precision

The library never changes JAX's precision setting. Enable float64 before
anything imports JAX, with `jax.config.update("jax_enable_x64", True)` or
`JAX_ENABLE_X64=1`; otherwise the calculator runs in float32 (faster, with
float32 round-off in energies and forces).

## Molecular dynamics

The calculator is built for repeated calls on one structure:

- The default `skin=1.0` (Å) keeps a Verlet neighbour list for cutoff + skin
  and reuses it until an atom has moved more than skin / 2, or the cell,
  periodicity, species or atom count change. Each call is then one compiled
  step.
- Edge lists are padded to power-of-two sizes, so a growing neighbour count
  rarely triggers a recompile.
- `calc.last_timing` reports the time of the last call; its `rebuilds` entry
  counts the calls that built a neighbour list.

```python
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
atoms.calc = ACECalculator("fit/model.npz")
MaxwellBoltzmannDistribution(atoms, temperature_K=600)
dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=600, friction=0.02)
dyn.run(200)
print(atoms.calc.last_timing["rebuilds"])
```

For one-shot evaluation of many unrelated structures (a test set, a
screening loop), pass `skin=0`, which builds the list for the cutoff alone on
every call.

The first call on a new structure size compiles the model, which takes
seconds; later calls take milliseconds.

## Speed options

| Option | Default | Effect |
|---|---|---|
| `layout` | `"auto"` | `"dense"` (padded per-atom blocks, fastest on GPU) when the neighbour padding is efficient, else `"sparse"` (an edge list) |
| `lean` | `True` | evaluate the lean form of an ACE model: unused radial columns and harmonics dropped, pair weights folded in. Exact to round-off |
| `spline_tol` | `"auto"` | spline a *learned* radial at 1e-10 before evaluating (see [learned radials](learned-radials.md#deployment-spline-speed)); a float splines any analytic radial, `None` never splines |
| `edge_a_kind` | `"auto"` | the A-basis kernel, `"gather"` or `"matmul"`; `"auto"` times both once per edge bucket |
| `skin` | `1.0` | the Verlet skin in Å; `0` rebuilds every call |

`calc.model` is the model as loaded; `calc.eval_model` is the form actually
evaluated. Installing the `fast-neighbours` extra (matscipy-neighbours) gives
faster neighbour lists; ASE's list is the fallback.

## Site descriptors

The per-atom ACE descriptors (one row per atom) are available from the
calculator, or directly from a loaded model:

```python
D = atoms.calc.get_site_descriptors(atoms)

import ace_jax as aj
model, meta, _ = aj.load("fit/model.npz")
D = aj.site_descriptors(model, atoms.positions, atoms.numbers, atoms.cell.array,
                        atoms.pbc, meta=meta)
```

## Uncertainty

- A model fitted with `aj fit --uq ard` comes with `posterior.npz`.
  `ACECalculator("model.npz", posterior="posterior.npz")` then provides a
  per-atom `forces_std`, computed on request with
  `calc.get_property("forces_std", atoms)`.
- A hybrid ACE + GP fit writes `gp_model.npz`, which loads with
  `GPCalculator.from_file("gp_model.npz")` and adds `energy_std` and
  `forces_std` to `calc.results`.
