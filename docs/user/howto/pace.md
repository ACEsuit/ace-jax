# Load and write PACE `.yace` models

ace-jax reads pacemaker's `.yace` potential files and evaluates them in JAX,
through the same calculator, CLI and LAMMPS export as its own models. It
checks against the ML-PACE C++ code, python-ace and LAMMPS.

## Evaluate a `.yace`

```python
import jax
jax.config.update("jax_enable_x64", True)
from ace_jax import ACECalculator

atoms.calc = ACECalculator("model.yace")
atoms.get_potential_energy(); atoms.get_forces()
```

From the shell:

```bash
aj eval --model model.yace --data test.xyz --out predictions.xyz
```

## Supported features

| Feature | Supported |
|---|---|
| radial basis | ChebExpCos, ChebPow, ChebLinear, SBessel |
| embedding | FinnisSinclair, FinnisSinclairShiftedScaled |
| inner cutoff | `density`, `distance`, `zbl` |

A file using anything else raises an error at load time rather than
evaluating differently.

## Load, modify and write back

`ace_jax.load` returns a `PACEModel`, its metadata and the parsed file. The
model is a JAX pytree of arrays, so it can be modified with
[equinox](https://docs.kidger.site/equinox/) tools, and `write_yace` writes it
back:

```python
import ace_jax as aj
from ace_jax.eval import write_yace

pm, meta, spec = aj.load("model.yace")       # PACEModel, meta dict, parsed spec
write_yace(pm, spec, "copy.yace")            # numbers from pm, layout from spec
```

`write_yace` takes the numeric values from the model and everything else
(the function layout, the element names, the file structure) from `spec`, so the
basis layout is never regenerated. A model written back unmodified evaluates
identically to the original.

## Limits

- `.yace` models are for evaluation and export. `aj fit` and the radial
  learner need an ACE `.npz` model.
- `aj.load(path)` dispatches on the extension: `.yace` returns
  `(PACEModel, meta, spec)`, `.npz` returns `(ACEModel, meta, arrays)`.
- An SBessel radial with `nradbase >= 12` is evaluated in a matrix form
  (`PACEModel.sbessel_form == "matmul"`), chosen at load; values agree with
  the recurrence to round-off.
