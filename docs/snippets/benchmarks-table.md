| evaluator | mode | CPU, float64: SiGe | CPU, float64: Cantor | GPU, float64: SiGe | GPU, float64: Cantor | GPU, float32: SiGe | GPU, float32: Cantor |
|---|---|--:|--:|--:|--:|--:|--:|
| ace-jax (PACE model) | standalone | 28k | 44k | 892k | 961k | 932k | 1.40M |
| ace-jax (PACE model) | LAMMPS | — | — | 1.09M | 1.33M | 1.79M | 2.04M |
| ace-jax (linear ACE) | standalone | 69k | 60k | 1.47M | 1.48M | 2.04M | 2.03M |
| ace-jax (linear ACE) | LAMMPS | — | — | 2.47M | 1.49M | 3.90M | 2.45M |
| ML-PACE | LAMMPS | 423k | 567k | 2.91M | 3.30M | — | — |
| MACE | standalone | 336 | — | 24k | 19k | 26k | 20k |
| MACE | LAMMPS | 4k | 3k | 60k | 48k | — | — |
| ACEpotentials.jl (linear ACE, direct) | standalone | 95k | 63k | — | — | — | — |
| ACEpotentials.jl (linear ACE, trim library in LAMMPS) | LAMMPS | 1.08M | 944k | — | — | — | — |
