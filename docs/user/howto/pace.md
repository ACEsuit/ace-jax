# Load and write PACE `.yace` models

ace-jax reads pacemaker `.yace` potential files and evaluates them in JAX.
It uses the same calculator, CLI and LAMMPS export as for its own models.
The results agree with the ML-PACE C++ code, python-ace and LAMMPS.

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

If a file uses a different feature, ace-jax gives an error when it loads
the file. It does not evaluate the file incorrectly.

## Load, modify and write back

`ace_jax.load` returns a `PACEModel`, its metadata and the parsed file. The
model is a JAX pytree of arrays. Thus you can change it with
[equinox](https://docs.kidger.site/equinox/) tools. `write_yace` writes it
to a file:

```python
import ace_jax as aj
from ace_jax.eval import write_yace

pm, meta, spec = aj.load("model.yace")       # PACEModel, meta dict, parsed spec
write_yace(pm, spec, "copy.yace")            # numbers from pm, layout from spec
```

`write_yace` takes the numeric values from the model. It takes all other
data (the function layout, the element names, the file structure) from
`spec`. Thus it never makes the basis layout again. If you write a model
without changes, it gives the same results as the original.

## Limits

- Use `.yace` models only for evaluation and export. `aj fit` and the
  radial learner need an ACE `.npz` model.
- The return value of `aj.load(path)` depends on the file extension:
  `.yace` gives `(PACEModel, meta, spec)`, `.npz` gives
  `(ACEModel, meta, arrays)`.
- For an SBessel radial with `nradbase >= 12`, ace-jax selects a matrix form
  at load time (`PACEModel.sbessel_form == "matmul"`). The values agree with
  the recurrence to round-off.
