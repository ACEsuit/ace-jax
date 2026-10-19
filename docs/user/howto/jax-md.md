# Run MD with JAX-MD

[JAX-MD](https://github.com/jax-md/jax-md) is a molecular-dynamics library
written in JAX. `ace_jax.jaxmd` gives an ace-jax model as JAX-MD energy and
force functions. The neighbour list, the model and the integrator then run on
the device in one compiled program.

The energy is the energy of `ACECalculator` with the same options. The two
agree to roundoff.

!!! note "Install"
    Install ace-jax with the `jaxmd` extra: `pip install "ace-jax[jaxmd]"`.
    For a GPU, add the `cuda` extra: `pip install "ace-jax[jaxmd,cuda]"`.

    The `jaxmd` extra installs jax-md and its dependencies. These are
    approximately 40 packages, for example flax and pymatgen.

--8<-- "float64.md"

## Run NVE

1. Make a JAX-MD space for the cell.
2. Make the potential and the neighbour list with the cutoff of the model.
3. Calculate the number of slots for each atom.
4. Make the integrator and run it in a compiled loop.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk
from jax_md import partition, simulate, space
from ace_jax import jaxmd

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(4)
box = jnp.asarray(atoms.cell.array.T)                  # JAX-MD: lattice vectors are columns
displacement, shift = space.periodic_general(box, fractional_coordinates=True)
R = jnp.asarray(atoms.get_scaled_positions())

rc = jaxmd.cutoff("fit/model.npz")
neighbor_fn = partition.neighbor_list(displacement, box, rc, dr_threshold=1.0,
                                      format=partition.Dense, fractional_coordinates=True)
nbrs = jaxmd.allocate(neighbor_fn, R)                   # not neighbor_fn.allocate
dt = 1.0 / 10.18                                         # 1 fs; time unit 10.18 fs (eV, A, amu)

def make(K):
    """The potential and a compiled chunk of 100 steps for K slots."""
    pot = jaxmd.potential("fit/model.npz", displacement, atoms.numbers, K=K)
    init, apply = simulate.nve(pot.force, shift, dt)

    @jax.jit
    def chunk(state, nbrs):
        def step(i, carry):
            state, nbrs = carry
            nbrs = nbrs.update(state.position)
            return apply(state, neighbor=nbrs), nbrs
        return jax.lax.fori_loop(0, 100, step, (state, nbrs))
    return pot, init, chunk

K = jaxmd.capacity(R, nbrs, displacement, rc + 1.0)     # slots for the cutoff + skin
pot, init, chunk = make(K)
state = init(jax.random.PRNGKey(0), R, mass=28.0855, kT=8.617e-5 * 300, neighbor=nbrs)

done, retries = 0, 0
while done < 10:
    new_state, new_nbrs = chunk(state, nbrs)
    if new_nbrs.did_buffer_overflow:                     # the list is too small
        nbrs = jaxmd.allocate(neighbor_fn, state.position)
    elif pot.overflow(new_state.position, new_nbrs) or not jnp.isfinite(new_state.position).all():
        K += 8                                           # an atom had more than K neighbours
        pot, _, chunk = make(K)
        state = state.set(force=pot.force(state.position, nbrs))   # the state keeps the last forces
    else:
        state, nbrs, done, retries = new_state, new_nbrs, done + 1, 0
        continue
    retries += 1                                         # run the same chunk again
    if retries > 3:
        raise RuntimeError("the chunk overflows again after 3 retries")
```

`jaxmd.allocate` replaces `neighbor_fn.allocate`. It avoids an error of
JAX-MD with JAX 0.11 when a new list goes into a compiled function.

The forces are the gradient of the energy. They are in Cartesian
coordinates, also with fractional positions. `quantity.stress` and the NPT
integrators of JAX-MD take `pot.energy`.

!!! warning "Stress sign"
    JAX-MD gives the stress with the opposite sign to ASE.

!!! warning "Minimum image"
    The JAX-MD neighbour list uses the minimum image. Every width of the cell
    must be more than 2 × (cutoff + skin). For a smaller cell, JAX-MD misses
    neighbours and does not give an error. To check the cell, use
    `jaxmd.min_image_ok(cell, rc + skin)`.

## Choose the slots

The model uses a fixed number of slots, `K`, for each atom. `K` sets the
cost of the model.

- `K` for the cutoff + skin (`capacity(..., rc + skin)`) is always
  sufficient between list builds.
- `K` for the cutoff (`capacity(..., rc)`) is approximately 1.5 times faster.
  Atoms can then get more neighbours than `K` during the run.

If an atom has more neighbours than `K`, the energy and forces are NaN. The
model never ignores a neighbour. After each chunk of steps, do these checks,
as in the example above:

- `pot.overflow(R, nbrs)` is true for an overflow at the last positions.
- Non-finite positions show an overflow earlier in the chunk.

Then make the potential again with a larger `K`, and run the chunk again.
The integrator state keeps the forces of the last step. Calculate them again
with the new potential before you run the chunk, as in the example.

## Memory

The model calculates the atoms in blocks of rows. Only one block at a time
is in the memory of the device. `potential` selects the block size as
`export_lammps` does for LAMMPS. Thus the memory is approximately the same
as for a LAMMPS bundle.

To set the block size, give `chunk=` (rows) to `potential`. A larger block
can be faster for a small model, but it uses more memory.

## Small cells, slabs and NPT (experimental)

`jaxmd.neighbor_list` makes a neighbour list with matscipy-neighbours on the
device. The list keeps the periodic image of each pair. Thus it works for
cells smaller than 2 × (cutoff + skin), triclinic cells, slabs and NPT.

!!! warning "Experimental"
    This list needs matscipy-neighbours built with its XLA FFI
    ([libAtoms/matscipy-neighbours#10](https://github.com/libAtoms/matscipy-neighbours/pull/10)).
    This build is not released yet. The API can change before the release.

1. Make the list with the box, the cutoff and the skin.
2. Give `None` as the displacement function of the potential.
3. Update the list in the loop, as above.

```python
nf = jaxmd.neighbor_list(box, rc, skin=1.0, fractional_coordinates=True)
nbrs = nf.allocate(R)
pot = jaxmd.potential("fit/model.npz", None, atoms.numbers,
                      K=jaxmd.capacity(R, nbrs, None, rc + 1.0))
```

On an NVIDIA GPU, the list builds again on the device without a step on the
host. On a CPU, or when a warning says that the planned build is not
available, use `jaxmd.run`. It examines the list once for each block of steps:

```python
state, nbrs, info = jax.jit(lambda s, n: jaxmd.run(apply, s, n, 1000, check_every=10))(state, nbrs)
```

For NPT, give the box of the barostat: `jaxmd.run(..., box_of=simulate.npt_box)`.

!!! warning "Slabs"
    `space.periodic_general` moves atoms back into the cell in all
    directions. For a slab, use sufficient vacuum so that no atom goes
    through the non-periodic boundary.
