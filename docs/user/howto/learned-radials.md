# Learn the radial basis

A built basis keeps its radial functions frozen: `aj fit` only fits the
linear readout. The radial learner optimises the radial mixing weights too,
by variable projection (VarPro), and a held-out gate keeps the result only
if it predicts better than the starting radials. The output is an ordinary
`model.npz`, so everything downstream (`aj fit --model`, `ACECalculator`,
LAMMPS export) works unchanged. [Concepts](../concepts.md#the-radial-basis)
explains the method and [Tutorial 2](../tutorials/learned-radials.md) runs it
end to end.

Learning the radials needs the `gp` extra and float64. It costs one streamed
pass over the fit split per L-BFGS step, so it is far more expensive than a
linear fit: about two minutes for 26 two-atom silicon cells on a CPU, and
tens of minutes on a GPU for production datasets of a few hundred cells.

## In Python

```python
import jax
jax.config.update("jax_enable_x64", True)

from ace_jax.basis.export import save_npz
from ace_jax.basis.model import BasisSpec, build_basis
from ace_jax.fit.pipeline import FitConfig, load_fit_data
from ace_jax.fit.pipeline.problem import build_problem
from ace_jax.fit.radial_learn import fit_radial, save_result
from ace_jax.fit.radial_model import rnl_degrees, to_analytic

keys = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
save_npz("basis.npz", build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",))))

# the linear problem on the fit split, with the validation split as its test set
cfg = FitConfig(model="basis.npz", arm="linear", m_per_species=0, e0="lsq", opt="lbfgs",
                r0=2.35, rungs=("map",), **keys).validate()
d = load_fit_data(cfg, train="fit.xyz", test="val.xyz")
model, _ = to_analytic(d.model, 12)                      # widen to n_q = 12 polynomials
prob = build_problem(cfg, d).prob._replace(model=model)

W, info = fit_radial(prob, d.ds_train, d.ds_test, model.rnl_Wnlq,
                     lam_grid=(0.0, 1e-2),               # relative roughness weights to try
                     rough_weights=1.0 / (1.0 + rnl_degrees(d.meta)) ** 2,
                     steps=40, reprofile_every=20, log=print)
print(info["selected"], info["scores"])                  # the gate's choice and scores
save_result("learned", W, info, src_npz="basis.npz", model=model)   # learned/model.npz
```

Then refit the readout on the whole training set and use the model as any
other:

```bash
aj fit --model learned/model.npz --train train.xyz --test test.xyz $K \
    --e0 lsq --m-per-species 0 --opt lbfgs --r0 2.35 --out fit_learned
```

### Options

| Argument | Meaning |
|---|---|
| `to_analytic(model, n_q)` | the polynomial span of each radial. 12 is a modest widening that optimises well; 30 is ill-conditioned and learns only small high-frequency changes |
| `steps`, `reprofile_every` | the L-BFGS step budget, and how often the evidence hyperparameters are re-fitted at the current radials |
| `lam_grid` | relative weights of a roughness penalty on the radials; each value is a gate candidate (`learned_lam=<value>`) |
| `spec_grid`, `gap_grid` | optional relative weights of a spectral prior and a data-gap prior on the change of the radials, also gated |

`info["selected"]` is `"init"` when no learned candidate beats the starting
radials on the validation split; the saved model then has the starting
radials, a readout fitted for them, and is not marked as learned.

## With the research driver

The repository's `bench/learn_radial/run.py` (not part of the installed
package) runs the same steps on one extxyz file, with a seeded fit/validation
split:

```bash
python bench/learn_radial/run.py --model basis.npz --data train.xyz --out learned \
    --energy-key dft_energy --force-key dft_force --virial-key dft_virial \
    --ntrain 26 --nval 13 --r0 2.35 --n-q 12 --steps 40 --lam-grid 0,1e-2
```

It writes `learned/model.npz`, the radial weights, the readout and a
`summary.json` with the gate scores. Note that it uses the E0 stored in the
basis file rather than fitting it.

## Deployment: spline speed

A learned radial is a polynomial mixture, slower to evaluate than the cubic
splines of a stock model. The learner marks its output as learned
(`radial_learned` in the file's metadata), and with the default
`spline_tol="auto"`, `ACECalculator` and `export_lammps` then convert the
learned radials to cubic splines at a relative tolerance of 1e-10 before
evaluating.

```python
from ace_jax import ACECalculator

calc = ACECalculator("fit_learned/model.npz")
calc.splined        # {'spline_tol': 1e-10, 'radials': ['rnl'], 'n_intervals': {'rnl': ...}}
```

- The splined model agrees with the exact one to about the tolerance, not to
  round-off: energies to about 1e-9 relative, forces to about 2e-8 of the
  largest force on the benchmark models.
- `spline_tol=None` never splines; a float (e.g. `1e-10`) splines any
  analytic radial, learned or not.
- Bases built by ace-jax are analytic but not learned, so by default they
  stay exact.
- The conversion is cached on the radial's content, so replacing only the
  readout (`calc.model = ...`) does not redo it.
- A learned-radial file written before the metadata flag existed loads as
  not learned; mark it with
  `ace_jax.basis.export.mark_radial_learned("model.npz")`, or pass a float
  `spline_tol`.
