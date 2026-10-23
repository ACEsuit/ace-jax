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
- `accuracy-cost-spec.md`, `accuracy-cost-plan.md`: the accuracy-matched speed
  benchmark on Si (`bench/accuracy_cost/`); `accuracy-cost-results.md`: its results
  and how to redraw the figures.
- `gpu-memory-results.md`: where the GPU memory of the force call goes (o4d20 Si, A100),
  the largest system that fits, and the measured headroom (`bench/gpu_memory/`).
- `jax-md-results.md`: `ace_jax.jaxmd` (JAX-MD) against LAMMPS + lammps-jax: throughput
  on the ACE ladder (A100, RTX 4000 Ada), row blocks, and JAX-MD's neighbour-list memory.
- `lammps-jax-cpu-port.md`: the plan for lammps-jax bundles on the CPU backend (#89).
- `python-domain-decomposition.md`: an assessment (deferred) of multi-device MD in Python with
  lammps-jax's `ghost_exchange`, without LAMMPS: fit, value per case, costs, and what to do first.
- `ard-force-uq.md`: usage notes for `--uq ard` revision 2 (conformal per-atom force
  uncertainty, `aj calibrate`), the source for its user pages.
- `basis-build-internals.md`: how `build_model` builds a basis, how `save_npz`
  derives its metadata, the bridge parity test and the coupling cache.
- `ard-validation.md`: the validation tables of `--uq ard` revision 2, moved
  out of the user page on force uncertainty.
