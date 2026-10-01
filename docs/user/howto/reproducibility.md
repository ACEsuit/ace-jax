# Reproducible fits

An ace-jax fit is deterministic: the same inputs, settings and software on the
same platform give the same model, bit for bit. This page lists what to keep
so that a fit can be repeated or audited later.

## Keep the run file and the model

Every `aj fit` writes `fit.yaml` next to the model, with every option
resolved and a `provenance` block:

```yaml
provenance:
  ace_jax: 0.1.0
  coupling: {backend: ace-jax-coupling==0.2.0, ..., platform: Linux-x86_64}
  command: aj fit --order 3 --max-degree 10 ...
  created: '2026-10-01T16:57:00Z'
```

`aj fit --config fit.yaml --out again` repeats the run (see
[Run files](fit-yaml.md)). The run file names its data files by path, so keep
the data with it, or record checksums of the files.

The fitted `model.npz` is self-contained: it holds the basis, the radials and
the coefficients, so evaluating it later needs neither the run file nor the
data.

## Seeds

`--seed` (default 0) seeds both the random radial weights of a built basis
(with `--radial-mode glorot_normal`) and the train/test permutation of
`--data ... --ntrain/--ntest`. Fixed seeds give the same basis and the same
split every time.

## Pin the software

- Record the environment: `pip freeze > requirements-fit.txt`, or keep the
  `uv.lock` of a uv project. JAX and NumPy versions can change the last digits
  of a fit.
- The coupling coefficients are cached per basis shape, stamped with the
  coupling library's version, so a library upgrade recomputes them rather than
  reusing stale entries. `--no-coupling-cache` always recomputes.

## Precision and platforms

- `aj` fits in float64. In Python, enable float64 before fitting
  (`jax.config.update("jax_enable_x64", True)`); the fit needs it, and the
  radial learner raises without it.
- Coupling coefficients are bit-identical on Linux aarch64 and macOS arm64;
  on x86_64 they agree to a few units in the last place. Fits on different
  platforms therefore agree to round-off, not bitwise.
- The default `ACECalculator` evaluation (`lean=True`) is exact to round-off.
  A learned radial is splined at deployment (to 1e-10), which is an
  approximation; pass `spline_tol=None` for the exact evaluation.

## Sharing a basis

To fit several datasets with exactly the same basis, or to let someone on a
platform without the coupling library fit, save it once with `aj basis ...
--out basis.npz` and use `aj fit --model basis.npz`. Coupling cache entries
are self-describing files and can also be copied into another machine's
cache directory.
