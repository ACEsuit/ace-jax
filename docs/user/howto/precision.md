# Choose float64 or float32

ace-jax evaluates models in float64 or float32. float64 is the default and
the reference. float32 can be faster, but it adds round-off error to energies
and forces. This page tells you when float32 is faster, how to measure its
error for your model, and which tasks can use it.

Fitting and radial learning always use float64.

--8<-- "float64.md"

## When float32 is faster

The speed gain depends on the float64 throughput of the GPU.

| GPU | float32 speed-up (silicon, LAMMPS, 32768 atoms) |
|---|---|
| RTX A4500 (low float64 throughput) | 2.1 to 4.5 times |
| A100 (full float64 throughput) | 1.3 to 1.5 times |

Most workstation GPUs (RTX, L40) have low float64 throughput, as the RTX
A4500 does. The gain is largest for small models and smallest for large
models. On an A100, float64 is usually the better choice.

## Measure the float32 error of your model

The float32 error depends on the model more than on the code. Large
coefficients that cancel make it larger. Thus, measure it for each model.

1. Enable float64 (see above).
2. Make structures that represent your task. Include the most distorted
   structures that the task visits, for example hot MD, defect cores or
   transition states.
3. Compare float32 with float64 on each structure:

    ```python
    import ace_jax as aj
    from ace_jax.calc.precision import precision_check

    model, meta, _ = aj.load("model.npz")
    check = precision_check(model, meta, atoms)
    print(check["dE_meV_atom"], check["dF_rel"], check["dF_max_meV_A"])
    ```

4. Record the energy error per atom (`dE_meV_atom`), the relative force error
   (`dF_rel`) and the maximum force error (`dF_max_meV_A`).

`aj export --dtype float32 --structure start.xyz` does the same check on the
initial structure of a LAMMPS run. A warning occurs above 0.1 meV/atom or a
relative force error of 1e-4.

These are typical values for silicon models (linear ACE, order 4), from small
to large:

| error | small model (degree 12 to 14) | large model (degree 16 to 20) |
|---|---|---|
| maximum force error | 0.5 to 1.0 meV/Å | 0.07 to 0.35 meV/Å |
| relative force error | 2e-4 to 4e-4 | 2e-5 to 1e-4 |
| energy error | 0.12 to 0.21 meV/atom | 0.004 to 0.02 meV/atom |

The errors on strongly distorted structures can be 5 to 25 times larger than
these values.

## Tasks that can use float32

float32 is safe when the quantity that you need is much larger than the
float32 error. Tasks that use forces are more tolerant than tasks that compare
small energy differences.

| task | float32 |
|---|---|
| MD with a thermostat: averages, radial distribution functions, diffusion | safe |
| NVE MD at a timestep of 1 fs or less | safe |
| geometry optimisation to a maximum force of approximately 3e-3 eV/Å or more, with a force-only optimiser | safe |
| properties of relaxed structures: elastic constants, defect and surface energies | safe for well-conditioned models; measure first |
| geometry optimisation to a maximum force of 1e-3 eV/Å or less; tight NEB or dimer searches | caution |
| NVE MD at 2 fs over long times | caution |
| small energy differences between large cells | caution |
| optimisers with an energy line search (L-BFGS or CG with Armijo or Wolfe tests) | do not use |
| finite differences of energies, free energy perturbation, thermodynamic integration | do not use without a float64 check |

### Geometry optimisation

!!! warning "Energy line searches fail in float32"
    The float32 energy noise is larger than the energy change along a short
    line search step. The Armijo and Wolfe tests then fail. Use an optimiser
    that uses only forces, for example ASE `ODE12r`.

```python
import numpy as np
import ace_jax as aj
from ase.optimize import ODE12r

atoms.calc = aj.ACECalculator("model.npz", dtype=np.float32)
ODE12r(atoms).run(fmax=3e-3)
```

The optimiser cannot go below the float32 force error. Set `fmax` to
approximately 3 times the maximum force error of your model, or larger.

### Molecular dynamics

In NVE MD of silicon (4096 and 32768 atoms, 10 to 40 ps, LAMMPS), float32
did not increase the energy drift at timesteps of 0.5 fs and 1 fs. It added
noise of approximately 0.01 meV/atom to the total energy. This noise does not
increase with time. At 2 fs, float32 possibly drifts by up to approximately
1 meV/atom per ns. A thermostat removes a drift of this size.

!!! note "Large boxes in LAMMPS"
    A float32 LAMMPS bundle receives float32 atom positions. The round-off of
    the bond vectors thus increases with the box size. Measure the error in a
    box of the size that you use. The ASE calculator calculates bond vectors
    in float64, so this does not apply to it. With the kernel,
    [mixed precision](#mixed-precision-for-the-kernels) keeps float64
    positions.

## Mixed precision for the kernels

The [C++ kernel](ase.md#cpu-kernel) and the [CUDA kernel](ase.md#gpu-kernel)
for linear ACE have a mixed-precision mode. It is off by default. In this
mode, the kernel calculates the product basis in float32. It calculates the
bond vectors, the radial basis, the spherical harmonics and the sums over
neighbours in float64. Positions, energies and forces stay float64.

Mixed precision is faster only for large models, where the product basis
uses most of the time:

| model (silicon, linear ACE) | CPU, LAMMPS, 128 cores | A100, ASE calculator |
|---|---|---|
| o4d20 (5,476 basis functions) | 1.4 times | 1.4 times |
| o4d16 | 1.1 times | 1.1 to 1.2 times |
| o4d14 | 1.1 times | no gain |
| o4d12 and order 3 | no gain | no gain |

The errors are those of float32 for the product basis. For the silicon model
o4d20, the maximum force error is approximately 0.05 meV/Å, and the energy has
noise of approximately 0.1 meV in a cell of 200 atoms. For o4d16, the
maximum force error is approximately 0.2 meV/Å and the noise is 1 meV. The
errors are larger for smaller models, as they are in float32. The smallest
models also have an energy offset that is the same for each atom:
approximately 0.6 meV for o3d8 and 0.2 meV for o3d10 in bulk silicon. With a
force-only optimiser, mixed precision changed no property of o4d20 by more
than 0.013 of its tolerance in the [silicon benchmark](../benchmarks.md).

!!! warning "Energy line searches fail in mixed precision"
    The energy noise is the same type as in float32. Optimisers with an
    energy line search (L-BFGS or CG with Armijo or Wolfe tests) can fail
    near a minimum. Use an optimiser that uses only forces, for example ASE
    `ODE12r` or `FIRE`, or `min_style fire` in LAMMPS. The table in
    [Tasks that can use float32](#tasks-that-can-use-float32) applies to
    mixed precision also.

Use mixed precision in place of float32 in these conditions:

- On a CPU. The C++ kernel has no float32 mode.
- In LAMMPS on a GPU with a large box. A float32 bundle receives float32
  positions, and its force error increases with the distance from the
  origin (0.05 meV/Å at the origin, 0.8 meV/Å at 1000 Å for o4d20). A
  mixed-precision bundle receives float64 positions, and its error does
  not change.

On a GPU, a float32 calculation is faster than mixed precision and has
approximately the same force error in a small box.

To set mixed precision:

- ASE: `ACECalculator("model.npz", ffi=True, ffi_precision="mixed")`.
  `calc.ffi["precision"]` then gives `"mixed"`.
- Batches: `BatchCalculator(..., ffi=True, ffi_precision="mixed")`.
- LAMMPS: `aj export --ffi --ffi-precision mixed`, or
  `export_lammps(..., ffi=True, ffi_precision="mixed")`. The bundle records
  it as `ace_jax.ffi_precision`.

Mixed precision needs the kernel and float64. It uses the same compiled
kernel library as float64.

## Check a new task

1. Measure the float32 error of the model on structures of the task.
2. Compare the force error with the force tolerance of the task. Examples
   are `fmax`, the NEB tolerance and the thermal forces.
3. Multiply the energy error per atom by the number of atoms. Compare the
   result with the energy differences that you need.
4. Do a small number of calculations again in float64. For statistical
   results, for example free energies, also do a short float64 run.
5. Use float32 only if it is faster on your hardware.

## Set the precision

- ASE: `ACECalculator("model.npz", dtype=np.float32)`. See
  [ASE calculators](ase.md#precision).
- LAMMPS: `export_lammps(..., dtype="float32")` or
  `aj export --dtype float32`. See [LAMMPS export](lammps.md#learned-radials-and-precision).
- Mixed precision of the kernels: `ffi_precision="mixed"`. See
  [Mixed precision for the kernels](#mixed-precision-for-the-kernels).
