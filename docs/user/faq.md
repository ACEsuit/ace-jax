# FAQ and troubleshooting

## Installing

### `--rungs pathfinder` fails with an `ImportError` about blackjax

The pathfinder rung is the only part of `aj fit` that is not in the core
install. Install the `gp` extra: `pip install "ace-jax[gp]"`.

### Building a basis raises `BasisUnavailable`

The compiled coupling library builds new bases. pip installs it
automatically on Linux x86_64/aarch64, macOS arm64 and Windows x64. On other
platforms:

1. Build the basis on a supported machine (`aj basis ... --out basis.npz`).
2. Fit it with `aj fit --model basis.npz --r0 ...`.

To fit and evaluate an existing basis, you do not need the library.

### Warnings that float64 "is not available, and will be truncated to dtype float32"

JAX runs in float32. Enable float64 at the top of your script, before other
code uses JAX:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Alternatively, set `JAX_ENABLE_X64=1`. The `aj` command enables float64
itself.

## Fitting

### My fit is much worse than expected

Do these checks in this order:

1. **Label keys.** By default, `aj fit` reads `energy`, `forces` and
   `virial`. If your file uses different names, give them with
   `--energy-key`, `--force-key` and `--virial-key`. If no configuration
   has a key that you give, `aj fit` stops with an error that lists the keys
   in the file. (Virials are an exception: files without a periodic cell
   have no virials.)
2. **E0.** A new basis has E0 = 0. Use `--e0 lsq` to fit the reference
   energies with the model, or set them yourself (see the next question).
3. **A basis that is too small for the data.** If the basis cannot fit
   energies and forces together, the evidence explains the energies as
   noise. To find this, look for a large `log_sigma_E` in `theta_map.json`:
   approximately -1, against -3 to -5 for a good fit. To correct this,
   increase `--max-degree` (or `--order`). For example, on the silicon
   tutorial data:
    - `--max-degree 8` (54 functions) cannot separate the diamond and β-tin
      phases, and gives 239 meV/atom.
    - `--max-degree 10` (110 functions) gives 24 meV/atom.
   If the forces are good but the energies are bad, the evidence can give
   accuracy in the forces at the cost of the energies. Without `--weights`,
   the fit learns a separate σ_E (`--noise per-quantity`). Give `--weights`
   to set the balance: the default `--noise auto` then learns one σ for all
   rows (`shared`), as ACEpotentials does.
4. **The radial basis.** The default `--radial-mode onehot` uses the radial
   polynomials. `--radial-mode glorot_normal` mixes them with seeded random
   weights. If these weights stay frozen, the fit is several times worse. On
   the silicon tutorial data, the energy error is 470 against 24 meV/atom,
   and the force error is 0.20 against 0.10 eV/Å. Use `glorot_normal` only
   as a start for [learned radials](howto/learned-radials.md), not for a
   frozen fit.
5. **The optimiser.**
    - The default `--opt lbfgs` maximises the evidence. For the linear
      model, it then refines the result to a stationary point.
    - If one more Newton step would increase the evidence by more than
      10⁻³ nats, the fit gives the warning `MAP did not converge`.
      `--strict` makes this an error.
    - For a GP fit, try `--map-polish on` or `--map-restarts`.
    - `--opt adam` runs `--map-steps` steps and can stop far from the
      optimum. On GAP-18 silicon, 500 Adam steps stopped approximately 10⁷
      nats below the maximum of the log-posterior, and the force error
      doubled.

### How do I use isolated-atom energies as E0?

Keep the isolated atoms in the training set and fit with `--e0 lsq`.

- ace-jax identifies a configuration with one atom and no neighbour in the
  cutoff as an isolated atom.
- The E0 of that species is then set to the energy of the isolated atom.
- The fit calculates E0 for the species that have no isolated atom.
- The log shows the species with a fixed E0
  (`E0: isolated-atom energies for Z=14 ...`).

To set E0 yourself, set it on the basis before the fit. Then keep the
default `e0="model"`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
from ace_jax.basis.model import BasisSpec, build_basis
from ace_jax.basis.export import save_npz

b = build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",)))
b = b._replace(model=eqx.tree_at(lambda m: m.E0, b.model, jnp.array([-158.54496821])))  # eV, element order
save_npz("si_e0.npz", b)        # then: aj fit --model si_e0.npz --r0 2.4 ...
```

### The fit is slow, or it seems to stop

- `--opt adam` (500 steps by default) is slower than the default L-BFGS.
  Usually it stops before the optimum.
- `--rungs` other than `map` adds approximations of the hyperparameter
  posterior. These cost much more than the fit. The `laplace` rung
  differentiates the evidence two times, and its compilation can take tens
  of minutes. If the process shows no output after the last
  `lbfgs ... logpost` line, it is compiling. It has not stopped.
- The GP arm (`--m-per-species` > 0; the default is 500) costs much more
  than the linear model. For linear ACE, use `--m-per-species 0`.

### The fit runs out of memory

Memory increases with the square of the number of basis functions (plus the
inducing sites for the GP arm). To decrease the memory, do one of these:

- Decrease `--max-degree` or `--order`.
- For many elements, use `--basis-embedding`.
- Decrease `--configs-per-batch` (default 8).

### What does `gamma missing ... rebuilt via basis.prior` mean?

This message is normal. If the basis does not contain the smoothness prior,
the fit builds it again from the basis. The message tells you this.

### `coverage` is much less than 0.68 and `rms_z` much more than 1

The predictive $\sigma$ of the linear model comes from the Bayesian
posterior alone. The posterior does not include the error that occurs
because the model cannot fit the data exactly. Thus, on small datasets,
$\sigma$ is usually too small. See
[Reading the metrics](concepts.md#reading-the-metrics).

## Evaluating

### The first calculator call takes seconds

JAX compiles the model for each new padded structure size. Later calls with
the same padded size use the compiled program again, and take milliseconds.

### Energies are different in the last digits on different machines

This is expected. Fits and evaluations agree across platforms to round-off,
not bit for bit. See [Reproducible fits](howto/reproducibility.md).

## Data

### ace-jax does not find my labels, but they are in the file

ace-jax reads extxyz with the libAtoms `extxyz` parser. Thus each label
keeps the name that it was written with, also `energy` and `forces`. (The
ASE reader moves these two labels into a calculator. For this reason,
ace-jax does not use it for training data.)

- Give the label names as they are in the header of your file.
- If no configuration in the file has a name that you give, ace-jax stops
  with an error that lists the keys in the file.
- If the file does not use the default names (`energy`, `forces`,
  `virial`), ace-jax ignores them.
