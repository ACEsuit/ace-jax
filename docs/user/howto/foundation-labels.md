# Fit labels from a foundation model

You can use a foundation model such as
[MACE-MPA-0](https://github.com/ACEsuit/mace-foundations) in place of DFT.
Label your structures with it, and fit ACE to these labels. Tutorials 4, 5
and 8 use this method. This page shows each part.

## Fit labelled `Atoms` directly

`load_fit_data` accepts lists of `ase.Atoms` and also file paths. Thus
labels that you calculate in Python do not need a file:

```python
import jax
jax.config.update("jax_enable_x64", True)   # fitting needs float64

from ace_jax.basis.model import BasisSpec
from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, save_model

cfg = FitConfig(model=BasisSpec(order=3, max_degree=10, elements=("Si",)), arm="linear",
                m_per_species=0, e0="lsq", opt="lbfgs", r0=None, rungs=("map",),
                energy_key="energy", force_key="forces", stress_key="stress").validate()
data = load_fit_data(cfg, train=train_atoms, test=test_atoms)
res = fit(cfg, data)
save_model(res, "fit")
```

ace-jax looks for each label in this order:

1. under its key in `atoms.info` (energy, virial, stress) or `atoms.arrays`
   (forces);
2. in the `energy`, `forces` and `stress` results of the attached
   calculator. An example is the `SinglePointCalculator` that the ASE extxyz
   reader attaches.

If the results of the calculator are for a different structure, ace-jax
gives an error. This occurs when you attach one calculator object to more
than one structure: the calculator keeps only the results of the last
structure.

## Stress or virial

ace-jax fits the virial. Calculators and DFT codes usually give the
stress. Thus give the stress key, and the reader converts the stress: for a
periodic cell, virial = −stress × volume.

- If a configuration has a virial label, ace-jax uses that label.
- A non-periodic configuration has no virial.

- Python: `FitConfig(stress_key="stress")`.
- Command line: `aj fit ... --stress-key stress`, and the same on `aj eval`.

The stress can be a 3 × 3 matrix, a flat 9-vector or a Voigt 6-vector
(xx, yy, zz, yz, xz, xy, the ASE order), in eV/Å³.

## Labelling with a foundation model

All ASE calculators can label structures. For MACE, install it in the same
environment. If you do not have a GPU, use the CPU version of torch:

```bash
pip install mace-torch --extra-index-url https://download.pytorch.org/whl/cpu
```

```python
from ase.calculators.singlepoint import SinglePointCalculator
from mace.calculators import mace_mp

calc = mace_mp(model="medium-mpa-0", default_dtype="float64", device="cpu")
labelled = []
for a in structures:
    a = a.copy()
    a.calc = calc
    results = dict(energy=a.get_potential_energy(), forces=a.get_forces(), stress=a.get_stress())
    a.calc = SinglePointCalculator(a, **results)    # each structure keeps its own labels
    labelled.append(a)
```

MACE-MPA-0 and MACE-MP-0b3 have the MIT licence. Before you distribute
labels from a different model, check the licence of that model.

## The tutorials' labels

The tutorials include their labels. Thus, with their default settings,
they do not need a labeller.

`ace_jax.tutorials.labels.label(structures, model="mpa-0", cache=...)` looks
for each structure in a cache file that is part of the tutorials. The key
is the content of the structure: numbers, cell, periodicity and wrapped
positions, to 10⁻⁶. It runs MACE only for a structure that is not in the
cache. This module is for the tutorials. It is not a stable API.

The cache files are in `docs/user/tutorials/data/school/` in the
repository. `make_labels.py` in that directory makes them again. Its
docstring gives the commands.

## Least squares and the evidence

Two options help you compare fits with different basis sizes, as
tutorial 5 does:

- `res.map.log_evidence` is the log marginal likelihood of the training
  data at the fitted hyperparameters. You can compare it between bases
  fitted to the same data. A larger value shows that the data gives more
  support to the basis.
- `FitConfig(solver="lstsq")` (`aj fit --solver lstsq`) is a weighted least
  squares fit. It has no prior, no evidence and no uncertainty (the
  predicted variances are zero). Its weights come from `weights=`
  (`--weights`). Use it only for teaching: a large basis overfits.
