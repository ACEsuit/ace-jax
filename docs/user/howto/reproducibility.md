# Reproducible fits

An ace-jax fit is deterministic. The same inputs, settings and software on
the same platform give the same model, bit for bit. This page tells you what
to keep, so that you can repeat or audit a fit later.

## Keep the run file and the model

Each `aj fit` writes `fit.yaml` next to the model. The file contains all
options, with their final values, and a `provenance` block:

```yaml
provenance:
  ace_jax: 0.1.0
  coupling: {backend: ace-jax-coupling==0.2.0, ..., platform: Linux-x86_64}
  command: aj fit --order 3 --max-degree 10 ...
  created: '2026-10-01T16:57:00Z'
```

`aj fit --config fit.yaml --out again` repeats the run (see
[Run files](fit-yaml.md)). The run file gives the paths of its data files.
Thus keep the data with the run file, or record the checksums of the data
files.

The fitted `model.npz` is self-contained. It contains the basis, the
radials and the coefficients. Thus, to evaluate it later, you do not need
the run file or the data.

## Seeds

`--seed` (default 0) is the seed for two random operations:

- the random radial weights of a built basis (with
  `--radial-mode glorot_normal`);
- the train/test permutation of `--data ... --ntrain/--ntest`.

A fixed seed always gives the same basis and the same split.

## Pin the software

- Record the environment: `pip freeze > requirements-fit.txt`, or keep the
  `uv.lock` of a uv project. Different JAX and NumPy versions can change the
  last digits of a fit.
- ace-jax keeps the coupling coefficients in a cache, one entry for each
  basis specification. Each entry records the version of the coupling
  library. Thus, after a library upgrade, ace-jax calculates the
  coefficients again and does not use old entries. With
  `--no-coupling-cache`, ace-jax always calculates them again.

## Precision and platforms

- `aj` fits in float64. In Python, enable float64 before the fit
  (`jax.config.update("jax_enable_x64", True)`). The fit needs float64, and
  the radial learner gives an error without it.
- The coupling coefficients are bit-identical on Linux aarch64 and macOS
  arm64. On x86_64, they agree to a few units in the last place. Thus fits
  on different platforms agree to round-off, not bit for bit.
- The default `ACECalculator` evaluation (`lean=True`) is exact to
  round-off. At deployment, a learned radial becomes a spline (to 1e-10).
  This is an approximation. For the exact evaluation, use
  `spline_tol=None`.

## Sharing a basis

Save a basis to a file if you want to:

- fit more than one dataset with exactly the same basis;
- let a person fit on a platform that does not have the coupling library.

To do this, save the basis one time with `aj basis ... --out basis.npz`.
Then use `aj fit --model basis.npz`. Coupling cache entries are
self-describing files. You can also copy them into the cache directory of a
different machine.
