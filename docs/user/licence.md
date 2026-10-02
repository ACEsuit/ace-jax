# Licence and citation

## ace-jax

ace-jax is released under the
[MIT licence](https://github.com/ACEsuit/ace-jax/blob/main/LICENSE).

## ace-jax-coupling

ace-jax depends on [`ace-jax-coupling`](https://pypi.org/project/ace-jax-coupling/),
the compiled library that computes the symmetry-adapted coupling coefficients
when a basis is built. Its own code and the library it compiles,
[EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl),
are MIT-licensed.

The binary wheels are compiled with Julia's ahead-of-time compiler and also
bundle the Julia runtime and the shared libraries it loads, each under its
own licence: MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, LGPL-2.1-or-later and
LGPL-3.0-or-later (each LGPL library is a separate, replaceable shared library),
and the GCC runtime under GPL-3.0-or-later with the GCC Runtime Library
Exception. No GPL-only code is bundled. Every component, its version and its
licence are listed in
[`THIRD_PARTY_NOTICES.md`](https://github.com/ACEsuit/ace-jax/blob/main/coupling/python/THIRD_PARTY_NOTICES.md),
and the summary is in the wheel's
[`LICENSE`](https://github.com/ACEsuit/ace-jax/blob/main/coupling/python/LICENSE).

## Tutorial data

- `si_tiny_train.xyz` is part of the ace-jax test fixtures.
- The CrMnFeCoNi subsets under `docs/user/tutorials/data/cantor/` are
  labelled with the MACE-MH-1 foundation model. The data are MIT-licensed;
  the MACE-MH-1 weights, which are not included, are under the Academic
  Software License. Provenance: the `README.md` next to them.

## Credits

ace-jax builds on the ACE ecosystem of [ACEsuit](https://github.com/ACEsuit):
its models follow [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl),
its fits reproduce ACEfit, and its coupling coefficients come from
EquivariantTensors.jl. The PACE evaluator follows
[pacemaker](https://github.com/ICAMS/python-ace) and ML-PACE. LAMMPS
deployment goes through [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).

## Citing

If you use ace-jax in published work, please cite the atomic cluster
expansion and its ACEpotentials implementation, whose models and fits ace-jax
reproduces:

- R. Drautz, "Atomic cluster expansion for accurate and transferable
  interatomic potentials", *Phys. Rev. B* **99**, 014104 (2019).
- W. C. Witt et al., "ACEpotentials.jl: A Julia implementation of the
  atomic cluster expansion", *J. Chem. Phys.* **159**, 164101 (2023).

If you use POPS uncertainty (`--uq pops`), also cite T. D. Swinburne and
D. Perez, [arXiv:2402.01810](https://arxiv.org/abs/2402.01810).

A citation for ace-jax itself will be added here when one is available.
