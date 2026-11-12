# Licence and citation

## ace-jax

ace-jax has the
[MIT licence](https://github.com/ACEsuit/ace-jax/blob/main/LICENSE).

## ace-jax-coupling

ace-jax needs [`ace-jax-coupling`](https://pypi.org/project/ace-jax-coupling/).
This compiled library calculates the symmetry-adapted coupling coefficients
when ace-jax builds a basis. Its code, and the library that it compiles
([EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl)),
have the MIT licence.

The Julia ahead-of-time compiler compiles the binary wheels. The wheels
also contain the Julia runtime and the shared libraries that it loads. Each
of these has its own licence:

- MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib;
- LGPL-2.1-or-later and LGPL-3.0-or-later. Each LGPL library is a separate
  shared library that you can replace.
- the GCC runtime: GPL-3.0-or-later with the GCC Runtime Library Exception.

The wheels contain no GPL-only code. All components, with their versions and
licences, are listed in
[`THIRD_PARTY_NOTICES.md`](https://github.com/ACEsuit/ace-jax/blob/main/coupling/python/THIRD_PARTY_NOTICES.md),
and the summary is in the wheel's
[`LICENSE`](https://github.com/ACEsuit/ace-jax/blob/main/coupling/python/LICENSE).

## Tutorial data

- `si_tiny_train.xyz` is part of the ace-jax test fixtures.
- The CrMnFeCoNi subsets in `docs/user/tutorials/data/cantor/` have labels
  from the MACE-MH-1 foundation model. The data have the MIT licence. The
  MACE-MH-1 weights are not included; they have the Academic Software
  License. The `README.md` in the same directory gives the provenance.

## Credits

ace-jax uses the ACE software of [ACEsuit](https://github.com/ACEsuit):

- its models follow [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl);
- its fits give the same results as ACEfit;
- its coupling coefficients come from EquivariantTensors.jl.

The PACE evaluator follows [pacemaker](https://github.com/ICAMS/python-ace)
and ML-PACE. LAMMPS deployment uses [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).

## Citing

If you use ace-jax in published work, cite the atomic cluster expansion
and its ACEpotentials implementation. ace-jax uses the same models and fits:

- R. Drautz, "Atomic cluster expansion for accurate and transferable
  interatomic potentials", *Phys. Rev. B* **99**, 014104 (2019),
  [doi:10.1103/PhysRevB.99.014104](https://doi.org/10.1103/PhysRevB.99.014104).
- G. Dusson et al., "Atomic cluster expansion: Completeness, efficiency and
  stability", *J. Comput. Phys.* **454**, 110946 (2022),
  [doi:10.1016/j.jcp.2022.110946](https://doi.org/10.1016/j.jcp.2022.110946).
  This paper gives the construction of the basis.
- W. C. Witt et al., "ACEpotentials.jl: A Julia implementation of the
  atomic cluster expansion", *J. Chem. Phys.* **159**, 164101 (2023),
  [doi:10.1063/5.0158783](https://doi.org/10.1063/5.0158783).

Cite these papers also, if you use the related functions:

- The calibrated force uncertainty (`--uq ard`, `aj calibrate`):
  J. R. Kermode, "Jackknife-conformal uncertainty for linear ACE
  potentials: calibrated forces and silicon properties compared with
  Bayesian and POPS baselines" (2026),
  [arXiv:2611.08678](https://arxiv.org/abs/2611.08678).
- POPS uncertainty (`--uq pops`): T. D. Swinburne and D. Perez, "Parameter
  uncertainties for imperfect surrogate models in the low-noise regime",
  *Mach. Learn.: Sci. Technol.* **6**, 015008 (2025),
  [doi:10.1088/2632-2153/ad9fce](https://doi.org/10.1088/2632-2153/ad9fce).
- A species-embedded basis (`--basis-embedding`): J. P. Darby et al.,
  "Tensor-reduced atomic density representations", *Phys. Rev. Lett.*
  **131**, 028001 (2023),
  [doi:10.1103/PhysRevLett.131.028001](https://doi.org/10.1103/PhysRevLett.131.028001).
- PACE `.yace` models: Y. Lysogorskiy et al., "Performant implementation of
  the atomic cluster expansion (PACE) and application to copper and
  silicon", *npj Comput. Mater.* **7**, 97 (2021),
  [doi:10.1038/s41524-021-00559-9](https://doi.org/10.1038/s41524-021-00559-9).
- PACE `.yace` models that pacemaker fitted: A. Bochkarev et al., "Efficient
  parametrization of the atomic cluster expansion", *Phys. Rev. Materials*
  **6**, 013804 (2022),
  [doi:10.1103/PhysRevMaterials.6.013804](https://doi.org/10.1103/PhysRevMaterials.6.013804).

The file
[`CITATION.cff`](https://github.com/ACEsuit/ace-jax/blob/main/CITATION.cff)
gives these references in a format that reference managers can read. On
GitHub, the "Cite this repository" button reads this file.

When a citation for ace-jax is available, it will be on this page.

### Also consider citing

ace-jax uses other software and models. Cite those that your work uses:

- Labels from a MACE foundation model: I. Batatia et al., "A foundation
  model for atomistic materials chemistry", *J. Chem. Phys.* **163**, 184110
  (2025), [doi:10.1063/5.0297006](https://doi.org/10.1063/5.0297006).
- LAMMPS (`aj export` bundles): A. P. Thompson et al., *Comput. Phys.
  Commun.* **271**, 108171 (2022),
  [doi:10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).
- ASE (the calculators): A. H. Larsen et al., *J. Phys.: Condens. Matter*
  **29**, 273002 (2017),
  [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).
- JAX-MD (`ace_jax.jaxmd`): S. S. Schoenholz and E. D. Cubuk, "JAX, M.D.:
  A framework for differentiable physics", *Adv. Neural Inf. Process. Syst.*
  **33** (2020), [arXiv:1912.04232](https://arxiv.org/abs/1912.04232).
- JAX: J. Bradbury et al., "JAX: composable transformations of Python+NumPy
  programs" (2018), [github.com/jax-ml/jax](https://github.com/jax-ml/jax).
