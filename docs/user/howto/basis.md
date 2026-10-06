# Build a basis

A basis is an ACE model with all coefficients set to zero. It is ready to fit.
You can build a basis in three ways:

```bash
aj fit --order 3 --max-degree 10 --train train.xyz --out fit/     # build it inside the fit
aj basis --elements Si --order 3 --max-degree 10 --out si.npz    # or save it on its own
aj fit --config fit/fit.yaml --out fit2/                         # the run file has a basis: block
```

All three use the same builder, so the same settings give the same basis.

## In Python

```python
from ace_jax.basis.model import BasisSpec, build_basis
from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data

spec = BasisSpec(order=3, max_degree=10)             # elements=None: the species in the data
b = build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",)), seed=0)
b.meta["n_B"], b.model.a2b_shape                      # inspect; modify with b._replace(model=...)
cfg = FitConfig(model=b, arm="linear", m_per_species=0, r0=None)   # a Basis, a BasisSpec or a path
res = fit(cfg, load_fit_data(cfg, train="train.xyz"))
```

`FitConfig(model=...)` accepts three types of value:

- the path of a saved basis file;
- a `Basis`, from `build_basis`;
- a `BasisSpec`. `load_fit_data` builds the basis for the species in the
  training, test and out-of-distribution data.

Notes:

- If you give `elements` and the data contains a species that is not in
  `elements`, the fit stops with an error.
- With `r0=None`, the fit uses the mean bond length of the basis as the
  centre of the hyperprior.
- A basis built from a `BasisSpec` and the same basis loaded from a file give
  identical fits, bit for bit.
- The coupling coefficients come from EquivariantTensors.jl, precompiled in
  the `ace-jax-coupling` wheel. All other parts of the basis are built in
  Python. The [maintainer notes](https://github.com/ACEsuit/ace-jax/blob/main/docs/dev/basis-build-internals.md)
  describe each build step.

## Species-embedded bases

`aj basis --embedding <table.json | identity> [--d-max N]` (or
`aj fit --basis-embedding ...`) builds a species-embedded basis.
[Multi-element models](../concepts.md#multi-element-models) explains the
method. To select the table and `d_max`:

- `identity` is the one-hot element table, with one channel for each
  element. It cannot compress the elements. A `d_max` that is less than the
  number of elements causes an error, because a one-hot table has no
  preferred direction to compress onto. To compress, use a table of element
  properties.
- ace-jax scales each row of the table to unit length. Thus, with `d_max=1`,
  each element is +1 or −1, and the basis has a maximum of two element
  groups. If two elements get the same row, ace-jax gives a warning: the
  basis cannot tell these elements apart.
