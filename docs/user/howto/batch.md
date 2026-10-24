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
  default is 262144 on a GPU and 16384 on a CPU. On a GPU, 262144 atoms of
  the Si model use approximately 1 GB in float64.
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

The table gives the speed-up of `calculate` against `ACECalculator` with one
structure in each call. The model is linear ACE for Si (correlation order 4, total degree
12). Each call has 1024 structures (256 for 216 atoms on the CPU).

| Atoms in each structure | RTX 4000 Ada, float32 | RTX 4000 Ada, float64 | A100, float32 | A100, float64 | CPU with the C++ kernel, float64 |
|--:|--:|--:|--:|--:|--:|
| 2 | 72x | 71x | 62x | 79x | 35x |
| 8 | 45x | 35x | 48x | 56x | 25x |
| 64 | 11x | 6.2x | 21x | 18x | 3.5x |
| 216 | 3.2x | 2.0x | 8.9x | 8.5x | 1.3x |

The GPUs are an NVIDIA RTX 4000 Ada and an NVIDIA A100 (40 GB). The A100 has
fast float64, so float64 is almost as fast as float32 on it. The CPU has 32 cores. The CPU values use
matscipy-neighbours with the batched neighbour-list build. Without it, the
speed-up for 64 atoms is approximately 2.4 times.

## Run MD or relax many structures with JAX-MD

ace-jax does not contain integrators or optimisers. It gives what
[JAX-MD](https://github.com/jax-md/jax-md) needs to run them on a batch with
`jax.vmap`:

- `bc.neighbour_lists(structures, skin)` puts all the structures into one
  padded batch and makes their neighbour lists.
- `nb.energy_fn(nl)` gives the energy of one structure, in the JAX-MD form.
- `ace_jax.batch` gives helpers for the padding atoms.

!!! note "Install"
    Install ace-jax with the `jaxmd` extra: `pip install "ace-jax[jaxmd]"`.

The procedure for NVE MD:

1. Make the neighbour lists and the energy function.
2. Make the JAX-MD integrator with `space.free()`.
3. Make the initial momenta with `batch.initial_momenta`.
4. Initialise the integrator for each structure with `jax.vmap`.
5. Run the steps in compiled chunks.
6. After each chunk, update the neighbour lists.

```python
import jax
jax.config.update("jax_enable_x64", True)
from jax_md import simulate, space
from ace_jax import batch
from ace_jax.batch import BatchCalculator

bc = BatchCalculator("fit/model.npz")
nb = bc.neighbour_lists(structures, skin=1.0)
nl = nb.allocate()
energy = nb.energy_fn(nl)

_, shift = space.free()
init, apply = simulate.nve(energy, shift, dt=1.0 / 10.18)    # 1 fs
mass = 28.0855

p0 = batch.initial_momenta(jax.random.PRNGKey(0), nb.live, kT=0.025, mass=mass,
                           dtype=nb.positions.dtype)
state = jax.vmap(lambda R, p, n: init(None, R, kT=0.0, mass=mass, momenta=p, neighbor=n))(
    nb.positions, p0, nl)

def chunk(state, n):
    return jax.lax.fori_loop(0, 50, lambda i, s: apply(s, neighbor=n), state)
step = jax.jit(jax.vmap(chunk))

for _ in range(100):
    state = step(state, nl)
    nl, rebuilt = nb.update(state.position, nl)

positions = nb.unpad(state.position)          # list of (n_atoms, 3) arrays
kT = batch.temperature(state.momentum, mass, nb.live)
```

`nb.energy_fn(nl)` sets the number of model slots for each atom from the
neighbours in the cutoff, plus 20%. If an atom gets more neighbours than
that, the energy and forces of its structure become NaN. Make the energy
function again with `nb.energy_fn(nb.allocate(state.position))`.

All structures go into one batch with the size of the largest structure.
The other structures get padding atoms. The cell of each structure can be
smaller than the cutoff.

The neighbour lists are made for the cutoff plus `skin`. `nb.update` makes
them again when an atom moves more than `skin / 2`. If a structure gets more
neighbours than the lists hold, `nb.update` makes larger lists. The compiled
step then compiles again.

### Update the lists inside the compiled loop

`nb.update` runs on the host between chunks. `nb.update_fn()` gives the same
update as a function that can run inside the compiled loop. Use it to check
the lists at every step:

1. Make the update function after `nb.allocate()`.
2. In the loop body, update the lists first.
3. Then apply one step to all structures with `jax.vmap`.
4. After the loop, read the overflow flags.

```python
update = nb.update_fn()

def body(i, carry):
    state, nl, overflow = carry
    nl, rebuilt, ov = update(state.position, nl)
    state = jax.vmap(lambda s, n: apply(s, neighbor=n))(state, nl)
    return state, nl, overflow | ov

run = jax.jit(lambda state, nl: jax.lax.fori_loop(
    0, 1000, body, (state, nl, jnp.zeros(len(structures), bool))))
new_state, nl, overflow = run(state, nl)
if overflow.any():
    nl = nb.allocate(state.position)      # larger lists; run again from `state`
```

The lists of the whole batch are made again in one call. On a CPU, the check
costs approximately 1 to 9% of the step time.

On a GPU, the check makes the host wait for the GPU at every step. On an idle
GPU this costs approximately 15 to 20% of the step time for small batches. On
a GPU that other programs also use, each wait is much longer.

`update_fn(branchless=True)` makes the lists at every step and keeps the new
lists only for the structures that moved. The loop then does not wait for the
host, but each step makes the lists. Use it on a shared GPU.

To reduce the cost on either device, check the lists every 10 steps:

```python
def chunk(state, nl):
    return jax.lax.fori_loop(0, 10, lambda i, s: jax.vmap(lambda x, n: apply(x, neighbor=n))(s, nl), state)

def body(i, carry):
    state, nl, overflow = carry
    nl, rebuilt, ov = update(state.position, nl)
    return chunk(state, nl), nl, overflow | ov
```

Use a skin that is large enough for 10 steps. An atom must not move more than
`skin / 2` between two checks.

!!! warning "Overflow inside the loop"
    Inside the loop, the size of the lists cannot change. If a structure gets
    more neighbours than the lists hold, its list is cut and its forces are
    wrong. `overflow` then shows the structure. Run the loop again from the
    last good state with larger lists.

This needs matscipy-neighbours with the batched build, or a GPU. On a CPU
without the batched build, the lists are made in JAX, which is slow for
structures of more than approximately 64 atoms.

For a geometry optimisation, use `minimize.fire_descent` in the same way.
Each structure keeps its own FIRE step size.

!!! warning "Padding atoms"
    JAX-MD gives momenta to all atoms, padding atoms included. This changes
    the kinetic energy and moves the centre of mass of the real atoms.

    - Give the initial momenta with `batch.initial_momenta`. Do not let
      `init` make them.
    - After Langevin steps, use `batch.zero_padding` before you read
      temperatures.
    - Calculate temperatures with `batch.temperature`, not with JAX-MD.
    - Do not use the Nose-Hoover thermostat with padding atoms. It counts
      the padding atoms as degrees of freedom. Give it structures of one
      size only.

!!! note "C++ kernel"
    On a CPU, use `BatchCalculator(..., ffi=True)`. `energy_fn` then uses the C++
    kernel, and `jax.vmap` sends the whole batch to it in one call. On 32 cores
    this MD is 37 to 48 times faster than with the JAX model, for 64 structures
    of 8 or 64 Si atoms. The kernel gives forces only: it does not give second
    derivatives.

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
- MD and geometry optimisation use JAX-MD (see above). The cell is fixed:
  NPT and cell relaxation are not available.
