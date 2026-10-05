# Developer and research notes

Not part of the user documentation (that is `docs/user/`, built into the site).
These are the design specs, implementation plans, benchmark reports and research
results behind ace-jax, kept as the record that code comments and tests cite:

- `specs/`, `plans/`: designs and implementation plans, dated (historical: they
  describe the code as it was planned, and later changes do not update them).
- `benchmarks.md`: the performance numbers (figures in `figs/`, from
  `bench/scaling/`).
- `*-results.md`, `*-gap.md`, `perf-*.md`: benchmark and research reports
  (figures in `figures/`). Some cite prototypes that are archived at the git tag
  `archive/research-prototypes` (see `bench/ARCHIVED.md`).
- `coupling-etshim-spec.md`: the compiled coupling library.
- `ard-force-uq.md`: usage notes for `--uq ard` revision 2 (conformal per-atom force
  uncertainty, `aj calibrate`), the source for its user pages.
- `basis-build-internals.md`: how `build_model` builds a basis, how `save_npz`
  derives its metadata, the bridge parity test and the coupling cache.
- `ard-validation.md`: the validation tables of `--uq ard` revision 2, moved
  out of the user page on force uncertainty.
