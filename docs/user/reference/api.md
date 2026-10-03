# Python API

The public Python interface, grouped by task. Everything else under
`ace_jax` is internal and may change between releases. Fitting and radial
learning need float64: call `jax.config.update("jax_enable_x64", True)`
before anything else imports JAX.

```python
import ace_jax as aj
model, meta, arrays = aj.load("model.npz")      # an ACE model, its metadata, the raw arrays
```

## Loading and evaluating

::: ace_jax.load

::: ace_jax.calc.point.ACECalculator
    options:
      members: [splined, eval_model, model, skin]

::: ace_jax.calc.gp.GPCalculator
    options:
      members: [from_file]

::: ace_jax.site_descriptors

::: ace_jax.highest_precision

## Building a basis

::: ace_jax.basis.model.BasisSpec
    options:
      members: false

::: ace_jax.basis.model.build_basis

::: ace_jax.basis.model.Basis
    options:
      members: [eval_pair]

::: ace_jax.basis.model.basis_r0

::: ace_jax.basis.export.save_npz

::: ace_jax.basis.coupling.BasisUnavailable

## Fitting

The pipeline behind `aj fit`. `FitConfig`'s defaults are those of the research
driver and differ from the command line's in places (for example `arm="gp"`,
`e0="lsq"`, `opt="lbfgs"`); set the fields you rely on explicitly, and
`predict_stats="recompute"` to match the command line exactly.

::: ace_jax.fit.pipeline.FitConfig
    options:
      members: false

::: ace_jax.fit.pipeline.load_fit_data

::: ace_jax.fit.pipeline.fit

::: ace_jax.fit.pipeline.save_model

::: ace_jax.fit.pipeline.write_outputs

::: ace_jax.fit.data.load_configs

## Learned radials

See [Learn the radial basis](../howto/learned-radials.md) for how these fit
together.

::: ace_jax.fit.pipeline.problem.build_problem

::: ace_jax.fit.radial_model.to_analytic

::: ace_jax.fit.radial_learn.fit_radial

::: ace_jax.fit.radial_learn.learn_radial

::: ace_jax.fit.radial_learn.save_result

::: ace_jax.fit.radial_model.with_radial

::: ace_jax.basis.export.mark_radial_learned

## Deployment forms

::: ace_jax.eval.lean

::: ace_jax.eval.to_spline

## PACE `.yace` models

::: ace_jax.eval.load_yace

::: ace_jax.eval.write_yace

## LAMMPS export

::: ace_jax.export.lammps.export_lammps

::: ace_jax.export.lammps.neighbour_capacity

::: ace_jax.export.lammps.matrix_supported
