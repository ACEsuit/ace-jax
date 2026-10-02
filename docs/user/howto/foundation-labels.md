# Fit labels from a foundation model

A foundation model such as [MACE-MPA-0](https://github.com/ACEsuit/mace-foundations)
can stand in for DFT: label your structures with it, and fit ACE to those
labels. Tutorials 3, 4 and 7 work this way. This page shows the pieces.

## Fit labelled `Atoms` directly

`load_fit_data` takes lists of `ase.Atoms` as well as file paths, so labels
computed in Python need no file:

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

Each label is looked up under its key in `atoms.info` (energy, virial,
stress) or `atoms.arrays` (forces), and, when it is not there, in the
`energy`, `forces` and `stress` results of the attached calculator, such as
the `SinglePointCalculator` that ASE's own extxyz reader attaches. A
calculator whose results belong to a different structure (one calculator
object attached to several structures keeps only the last one's results) is
an error, not a label.

## Stress or virial

ace-jax fits the virial. Calculators and DFT codes usually report the
stress, so give its key and the reader converts it: for a periodic cell,
virial = −stress × volume. A configuration that has a virial label keeps it;
a non-periodic one gets no virial.

- Python: `FitConfig(stress_key="stress")`.
- Command line: `aj fit ... --stress-key stress`, and the same on `aj eval`.

The stress may be stored as a 3 × 3 matrix, a flat 9-vector or a Voigt
6-vector (xx, yy, zz, yz, xz, xy, ASE's order), in eV/Å³.

## Labelling with a foundation model

Any ASE calculator can label structures. For MACE, install it into the same
environment, with CPU torch unless you have a GPU:

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

MACE-MPA-0 and MACE-MP-0b3 are MIT-licensed. Check the licence of any other
model before you redistribute labels made with it.

## The tutorials' labels

The tutorials ship their labels, so their default settings need no labeller.
`ace_jax.tutorials.labels.label(structures, model="mpa-0", cache=...)` looks
each structure up in a shipped cache file, keyed on its content (numbers,
cell, periodicity and wrapped positions, to 10⁻⁶), and runs MACE only for a
structure the cache does not hold. This module supports the tutorials and is
not a stable API.

The cache files are under `docs/user/tutorials/data/school/` in the
repository, and `make_labels.py` there regenerates them; its docstring has
the commands.

## Least squares and the evidence

Two options help to compare fits across basis sizes, as tutorial 4 does:

- `res.map.log_evidence` is the log marginal likelihood of the training data
  at the fitted hyperparameters. It is comparable across bases fitted to the
  same data; a larger value is a better-supported basis.
- `FitConfig(solver="lstsq")` (`aj fit --solver lstsq`) is plain weighted
  least squares, with no prior, no evidence and no uncertainty (predicted
  variances are zero). Its weights come from `weights=` (`--weights`).
  It is meant for teaching: a large basis overfits.
