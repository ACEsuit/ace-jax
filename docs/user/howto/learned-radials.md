# Learn the radial basis

In a built basis, the radial functions are frozen: `aj fit` fits only the
linear coefficients. The radial learner also optimises the radial mixing
weights, by variable projection (VarPro). A validation gate then keeps the
learned radials only if they predict better than the initial radials.

The output is a usual `model.npz` file. Thus all later steps
(`aj fit --model`, `ACECalculator`, LAMMPS export) work without changes.
[Concepts](../concepts.md#the-radial-basis) explains the method.
[Tutorial 2](../tutorials/learned-radials.md) shows all the steps.

Radial learning needs float64. The `aj` command enables float64 itself.

Radial learning costs much more than a linear fit. Each L-BFGS step does one
pass over the fit split, in batches. For example:

- 26 two-atom silicon cells: approximately 2 minutes on a CPU;
- production datasets of a few hundred cells: tens of minutes on a GPU.

## On the command line

```bash
aj fit --order 3 --max-degree 10 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --learn-radial \
    --out fit
```

`--learn-radial` does these steps:

1. It keeps a seeded validation set of the training configurations
   (`--radial-val-frac`, default 0.2).
2. It learns the radials on the remaining configurations, and the
   validation set gates the result.
3. It fits the coefficients on the **full** training set, as usual.

`--learn-radial` also works with `--model`, and with all final arms and UQ
options. Its options are:

- `--radial-n-q 12`: the number of polynomials for each radial;
- `--radial-steps 40`: the number of L-BFGS steps for each roughness weight;
- `--radial-lam-grid 0,1e-2`: the roughness weights. The gate selects from
  these weights and the initial radials.

The fit writes these files:

- `fit/radial_info.json`: the selection of the gate, and the validation
  score of each candidate.
- `fit/model.npz`: the learned radials. The file has a "learned" mark, so
  ace-jax changes the radials to splines at deployment (see
  [below](#deployment-spline-speed)). If the gate keeps the initial radials,
  the file has no mark.
- `fit/fit.yaml`: all settings of the run, including the radial options.

`--learn-radial` does not support species-embedded bases
(`--basis-embedding`) at this time.

### Radial basis

`--radial-basis` sets the functions that make the tensor radials:

- `poly` (the default): polynomials of a transformed distance, with an
  envelope, as in ACEpotentials.
- `sbessel`: the simplified spherical Bessel functions of pacemaker (PACE),
  as functions of the distance. They are zero at the cutoff and after it.

`--radial-basis` is a basis option. It does not need `--learn-radial`. With
`--learn-radial`, the learner changes the coefficients of the functions in
the same way for the two bases.

!!! note
    ace-jax does not change `sbessel` radials to splines at deployment. They
    stay exact, learned or not.

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

save_npz("basis.npz", build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",))))

# the linear problem on the fit split, with the validation split as its test set
cfg = FitConfig(model="basis.npz", arm="linear", m_per_species=0, e0="lsq", opt="lbfgs",
                r0=2.35, rungs=("map",)).validate()
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

Then fit the coefficients again on the full training set, and use the model
as usual:

```bash
aj fit --model learned/model.npz --r0 2.35 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --out fit_learned
```

### Options

| Argument | Meaning |
|---|---|
| `to_analytic(model, n_q)` | the polynomial span of each radial. 12 is a small increase that optimises well. 30 is ill-conditioned and learns only small high-frequency changes |
| `steps`, `reprofile_every` | the maximum number of L-BFGS steps, and the interval at which the evidence hyperparameters are fitted again at the current radials |
| `lam_grid` | relative weights of a roughness penalty on the radials. Each value is a gate candidate (`learned_lam=<value>`) |
| `spec_grid`, `gap_grid` | optional relative weights of a spectral prior and a data-gap prior on the change of the radials. The gate also selects from these |

If no learned candidate is better than the initial radials on the
validation split, `info["selected"]` is `"init"`. The saved model then has
the initial radials and coefficients fitted for them, and it has no
"learned" mark.

## With the research driver

`bench/learn_radial/run.py` in the repository does the same steps on one
extxyz file, with a seeded fit/validation split. It is not part of the
installed package.

```bash
python bench/learn_radial/run.py --model basis.npz --data train.xyz --out learned \
    --ntrain 26 --nval 13 --r0 2.35 --n-q 12 --steps 40 --lam-grid 0,1e-2
```

It writes `learned/model.npz`, the radial weights, the coefficients and a
`summary.json` file with the gate scores.

!!! note
    The research driver uses the E0 in the basis file. It does not fit E0.

## Deployment: spline speed

A learned radial is a polynomial mixture. Its evaluation is slower than the
cubic splines of a stock model. Thus:

1. The learner marks its output as learned (`radial_learned` in the
   metadata of the file).
2. With the default `spline_tol="auto"`, `ACECalculator` and
   `export_lammps` change the learned radials to cubic splines before
   evaluation, at a relative tolerance of 1e-10.

```python
from ace_jax import ACECalculator

calc = ACECalculator("fit_learned/model.npz")
calc.splined        # {'spline_tol': 1e-10, 'radials': ['rnl'], 'n_intervals': {'rnl': ...}}
```

- The splined model agrees with the exact model to approximately the
  tolerance, not to round-off. On the benchmark models, energies agree to
  approximately 1e-9 (relative), and forces to approximately 2e-8 of the
  largest force.
- `spline_tol=None` never makes splines. A float (for example `1e-10`)
  makes splines of all analytic radials, learned or not.
- Bases built by ace-jax are analytic but not learned. Thus, by default,
  they stay exact.
- The calculator keeps the splines in a cache, keyed on the content of the
  radials. Thus, if you replace only the coefficients (`calc.model = ...`),
  it does not make the splines again.
- A learned-radial file written before the "learned" mark existed loads as
  not learned. To correct this, mark it with
  `ace_jax.basis.export.mark_radial_learned("model.npz")`, or give a float
  `spline_tol`.
