# FAQ and troubleshooting

## Installing

### `--rungs pathfinder` fails with an `ImportError` about blackjax

The pathfinder rung is the one part of `aj fit` outside the core install:
`pip install "ace-jax[gp]"`.

### Building a basis raises `BasisUnavailable`

The compiled coupling library that builds new bases is installed
automatically on Linux x86_64/aarch64, macOS arm64 and Windows x64. On other
platforms, build the basis on a supported machine (`aj basis ... --out
basis.npz`) and fit it with `aj fit --model basis.npz --r0 ...`: fitting and
evaluating an existing basis never need the library.

### Warnings that float64 "is not available, and will be truncated to dtype float32"

JAX is running in float32. Call `jax.config.update("jax_enable_x64", True)`
at the top of your script, before anything else uses JAX, or set
`JAX_ENABLE_X64=1`. The `aj` command enables float64 itself.

## Fitting

### My fit is much worse than expected

Check, in order:

1. **Label keys.** `aj fit` reads `energy`, `forces` and `virial` unless told
   otherwise; pass `--energy-key`, `--force-key` and `--virial-key` with the
   names in your file. A key you name that no configuration has is an
   error that lists the keys the file does have. (Virials are the
   exception for files without a periodic cell, which have none.)
2. **E0.** A freshly built basis has E0 = 0. Use `--e0 lsq` to fit the
   reference energies to the training energies, or set them yourself (next
   question).
3. **A basis too small for the data.** When the basis cannot fit energies
   and forces together, the evidence explains the energies as noise: look
   for a large `log_sigma_E` in `theta_map.json` (around -1, against -3 to
   -5 for a good fit). On the silicon tutorial data, `--max-degree 8` (54
   functions) cannot separate the diamond and β-tin phases and gives
   239 meV/atom; `--max-degree 10` (110 functions) gives 24. Raise
   `--max-degree` (or `--order`).
4. **The radial basis.** The default `--radial-mode onehot` uses the radial
   polynomials themselves. `--radial-mode glorot_normal` mixes them with
   seeded random weights; kept frozen it fits several times worse (on the
   silicon tutorial data, 470 against 24 meV/atom in energy and 0.20
   against 0.10 eV/Å in forces). It is a starting point for
   [learned radials](howto/learned-radials.md), not for a frozen fit.
5. **The optimiser.** The default `--opt adam` runs 500 steps and can stop
   short on small data; `--opt lbfgs` converges in tens of evaluations.

### How do I use isolated-atom energies as E0?

Keep the isolated atoms in the training set and fit with `--e0 lsq`. A
single atom with no neighbour within the cutoff is recognised as isolated,
and its species takes its energy as E0 exactly; species without one are
fitted by least squares to the other configurations. The log names the
species it fixed (`E0: isolated-atom energies for Z=14 ...`).

To set E0 by hand instead, set it on the basis model before fitting and
keep the default `e0="model"`:

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

### The fit is slow, or seems to hang

- `--opt lbfgs` is much faster than the default Adam on small data.
- `--rungs` beyond `map` adds hyperparameter-posterior approximations, which
  cost far more than the fit itself. The `laplace` rung differentiates the
  evidence twice and can take tens of minutes just to compile: a silent
  process after the last `lbfgs ... logpost` line is compiling, not hung.
- The GP arm (`--m-per-species` > 0, which is the default of 500) is much
  more expensive than the linear model. Pass `--m-per-species 0` for linear
  ACE.

### The fit runs out of memory

Memory grows with the square of the number of basis functions (plus inducing
sites for the GP arm). Lower `--max-degree` or `--order`, use
`--basis-embedding` for many elements, or lower `--configs-per-batch`
(default 8) to reduce the per-batch working set.

### What does `gamma missing ... rebuilt via basis.prior` mean?

Nothing is wrong. The smoothness prior is rebuilt from the basis when the
basis does not store it; the message says so.

### `coverage` is far below 0.68 and `rms_z` far above 1

The linear model's predictive $\sigma$ comes from the Bayesian posterior alone,
which does not account for the model being unable to fit the data exactly.
On small datasets it is typically too small. See
[Reading the metrics](concepts.md#reading-the-metrics).

## Evaluating

### The first calculator call takes seconds

JAX compiles the model for each new padded structure size. Later calls with
the same padded size reuse the compiled program and take milliseconds.

### Energies differ in the last digits between machines

Expected: fits and evaluation agree across platforms to round-off, not bit for
bit. See [Reproducible fits](howto/reproducibility.md).

## Data

### My labels are not found, though they are in the file

ace-jax reads extxyz with the libAtoms `extxyz` parser, so every label keeps
the name it was written with, including `energy` and `forces`. (ASE's reader
moves those two into a calculator, which is why ace-jax does not use it for
training data.) Pass the names as they appear in your file's header. A name
that no configuration in the file has is an error listing the keys it does
have; the default names (`energy`, `forces`, `virial`) are simply absent when
the file does not use them.
