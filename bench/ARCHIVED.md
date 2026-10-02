# Archived research code

Prototypes and spikes that no longer run against the library, removed before the
0.1.0 release. They are kept, unchanged, at the git tag `archive/research-prototypes`
(`git checkout archive/research-prototypes -- <path>`), as the evidence behind the
reports in `docs/dev/` that cite them:

| Path | What it was | Cited by |
|---|---|---|
| `bench/perf/variants.py`, `lammps_variant.py`, `profile_acejax.py` | `FastPACE` prototypes and their profiling harness, written against PACE methods the pool-first rewrite removed | `docs/dev/pace-performance-gap.md`, `perf-optimisation-plan.md` |
| `bench/pace_modal/` | the Modal driver and results of the `.yace` PACE parity and layout study (stable-branch LAMMPS image) | `docs/dev/pace-yace-results.md` |
| `bench/pace_dense_proto.py`, `bench/pace_profile.py` | the dense-A prototype and force-pass profile of `PACEModel` | `docs/dev/pace-yace-results.md` |
| `bench/pops_profile.py` | a POPS stage timer | — |
| `spike/` | five POPS regime experiments | — |

The results these produced (`bench/perf/results/`) stay on main: library docstrings
and tests cite them.
