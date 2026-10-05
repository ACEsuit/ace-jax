# Performance

This page compares the speed of ace-jax with ML-PACE and MACE, on one CPU
and one GPU, for energies and forces.

Each figure shows the throughput against the system size. The throughput is
in **atom-steps per second** (atoms × evaluations per second; higher is
better).

- **Workload.** The workload is similar to MD. Before each timed evaluation
  of energy and forces, each atom moves a small distance, as between two
  MD steps. The compilation time is not included.
- **Models.** The models are **medium-size models with random weights**.
  Their cost is the same as the cost of a fitted model, but they do not
  predict correctly:
    - PACE `.yace` models of approximately 500 basis functions for each
      element (ace-jax and ML-PACE evaluate the same files);
    - linear ACE models of approximately 450 basis functions for each
      element;
    - MACE-MP-0b2 medium.
- **Standalone** (solid lines) is a direct call of the evaluator, outside an
  MD engine. For ace-jax and MACE, this is one ASE calculator call from
  Python, including the neighbour list.
- **LAMMPS** (dashed lines) is the time of one MD step in LAMMPS: ace-jax
  with lammps-jax ([LAMMPS export](howto/lammps.md)), ML-PACE as
  `pair_style pace`, MACE with Symmetrix.
- **Systems.** SiGe, a random alloy on diamond, and Cantor, the equiatomic
  CrMnFeCoNi alloy on fcc.

## CPU, float64

![Throughput against system size on CPU in float64, SiGe and Cantor panels: ACEpotentials.jl trim library and ML-PACE in LAMMPS are fastest, ace-jax standalone 6–10× below ML-PACE and level with ACEpotentials.jl direct, MACE slowest](assets/benchmarks/cpu_float64.png)

ace-jax runs in LAMMPS only on a GPU, because lammps-jax supports only
GPUs. Thus the CPU figure shows ace-jax standalone only.

## GPU, float64

![Throughput against system size on GPU in float64, SiGe and Cantor panels, for ace-jax PACE and linear ACE, ML-PACE and MACE, in LAMMPS and standalone](assets/benchmarks/gpu_float64.png)

## GPU, float32

![Throughput against system size on GPU in float32, SiGe and Cantor panels, for ace-jax PACE and linear ACE and MACE standalone](assets/benchmarks/gpu_float32.png)

## At 8,192 atoms

Atom-steps per second at 8,192 atoms. "—" shows that a code does not run
in that setting, or ran out of memory at a smaller size.

--8<-- "benchmarks-table.md"

## Caveats

--8<-- "benchmarks-notes.md"
- Random weights set the cost of a model, not its accuracy. Thus these
  numbers compare only speed. A fitted model of the same size has the same
  speed.
- ace-jax standalone uses its neighbour list again while the atoms stay in
  the skin. MACE standalone builds the list again at each call.
- The throughput depends on the model size, the cutoff and the number of
  neighbours. The [full benchmarks](https://github.com/ACEsuit/ace-jax/blob/main/docs/dev/benchmarks.md)
  include small and large models, peak memory and the largest system that
  fits in memory.

## More

- [Full benchmark results](https://github.com/ACEsuit/ace-jax/blob/main/docs/dev/benchmarks.md):
  every model size and host, memory, precision, parity checks and versions.
- [How to rerun the benchmarks](https://github.com/ACEsuit/ace-jax/blob/main/bench/scaling/README.md).
