# Use a model as an ASE calculator

`ACECalculator` evaluates an ace-jax model file (an ACE `.npz` or a PACE
`.yace`) as an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator. It gives
the energy, the forces and the stress.

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

The cutoff, the species and the precision come from the model file. Thus,
usually, the path is the only argument. If `atoms` contains a species that
the model does not know, the calculator gives an error.

## Precision

--8<-- "float64.md"

If float64 is not enabled, the calculator uses float32. float32 is faster,
but it adds float32 round-off to energies and forces.

### Tight geometry optimisation of large cells

The energy includes the isolated-atom energy E0 of each atom (approximately
−160 eV for Si). This causes a problem for large cells:

- For a cell of 10^5 atoms, |E| is approximately 10^7 eV. The float64
  resolution is then approximately 10^-9 eV.
- An energy-based line search (for example, the Armijo test in the ASE
  `PreconLBFGS`) checks that the energy decreases. When the forces are less
  than approximately 1e-4 to 1e-5 eV/Å, it cannot resolve the decrease.
- The optimiser then becomes very slow.

For tight relaxations, use energies relative to the isolated atoms:

```python
calc = ACECalculator("model.npz", energy_reference="E0")
```

`atoms.get_potential_energy()` then gives the energy relative to the
isolated atoms. The forces and the stress do not change.
`calc.results["e0_offset"]` contains the constant that the calculator
subtracted. Thus the absolute energy is
`atoms.get_potential_energy() + calc.results["e0_offset"]`.

## Molecular dynamics

The calculator is designed for repeated calls on one structure:

- With the default `skin=1.0` (Å), the calculator keeps a Verlet neighbour
  list for the distance cutoff + skin. Thus each call is one compiled step.
  The calculator uses the list again until one of these occurs:
    - an atom moves more than skin / 2;
    - the cell, the periodicity, the species or the number of atoms changes.
- The edge lists are padded to sizes that are powers of two. Thus, when the
  number of neighbours increases, a recompilation is rare.
- `calc.last_timing` gives the time of the last call. Its `rebuilds` entry
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

To evaluate many unrelated structures one time each (for example a test set
or a screening loop), use `skin=0`. The calculator then builds the list for
the cutoff alone at each call.

The first call on a new structure size compiles the model. This takes
seconds. Later calls take milliseconds.

## Speed options

| Option | Default | Effect |
|---|---|---|
| `layout` | `"auto"` | `"dense"` (padded per-atom blocks, fastest on GPU) when the neighbour padding is efficient, else `"sparse"` (an edge list) |
| `lean` | `True` | evaluate the lean form of an ACE model: unused radial columns and harmonics removed, pair weights folded in. Exact to round-off |
| `spline_tol` | `"auto"` | change a *learned* radial to a spline at 1e-10 before evaluation (see [learned radials](learned-radials.md#deployment-spline-speed)); a float changes any analytic radial to a spline; `None` never makes splines |
| `edge_a_kind` | `"auto"` | the A-basis kernel, `"gather"` or `"matmul"`; `"auto"` measures the time of both, one time for each edge bucket |
| `skin` | `1.0` | the Verlet skin in Å; `0` builds the list again at each call |

`calc.model` is the model as loaded. `calc.eval_model` is the form that the
calculator evaluates. If you install the `fast-neighbours` extra
(matscipy-neighbours), the neighbour lists are faster. If not, the
calculator uses the ASE neighbour list.

## Site descriptors

You can get the per-atom ACE descriptors (one row for each atom) from the
calculator, or directly from a loaded model:

```python
D = atoms.calc.get_site_descriptors(atoms)

import ace_jax as aj
model, meta, _ = aj.load("fit/model.npz")
D = aj.site_descriptors(model, atoms.positions, atoms.numbers, atoms.cell.array,
                        atoms.pbc, meta=meta)
```

## Uncertainty

- A model fitted with `aj fit --uq ard` has a `posterior.npz` file.
  `ACECalculator("model.npz", posterior="posterior.npz")` then gives a
  calibrated per-atom `forces_std`, `forces_cov`, `forces_q` and
  `forces_group`. The calculator calculates them only when you ask, for
  example with `calc.get_property("forces_std", atoms)`. See
  [Per-atom force uncertainty](force-uncertainty.md).
- A hybrid ACE + GP fit writes `gp_model.npz`. Load it with
  `GPCalculator.from_file("gp_model.npz")`. This calculator adds
  `energy_std` and `forces_std` to `calc.results`.
