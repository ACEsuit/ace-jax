# Fit a multi-element model

Nothing changes on the command line for several elements: `aj fit` builds
the basis for every species it finds in the training data. This page fits the
five-component CrMnFeCoNi (Cantor) alloy from 40 small bulk cells labelled by
the MACE-MH-1 foundation model, then tests it in and out of distribution. It
takes about four minutes on a CPU.

## The data

```bash
B=https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor
for f in train test vacancy compressed; do curl -LO $B/cantor_$f.xyz; done
K="--energy-key mace_energy --force-key mace_force --virial-key mace_virial"
```

| File | Configs | Contents |
|---|---|---|
| `cantor_train.xyz` | 40 | bulk 32-atom fcc cells, strained and rattled |
| `cantor_test.xyz` | 30 | the same distribution: in-distribution test |
| `cantor_vacancy.xyz` | 30 | 47-atom cells with one vacancy, unrelaxed: out of distribution |
| `cantor_compressed.xyz` | 30 | compressed bulk: out of distribution |

The [data README](https://github.com/ACEsuit/ace-jax/blob/main/docs/user/tutorials/data/cantor/README.md)
records how the subsets were made.

## Fit

```bash
aj fit --order 2 --max-degree 5 \
    --train cantor_train.xyz --test cantor_test.xyz --ood cantor_vacancy.xyz $K \
    --e0 lsq --m-per-species 0 --opt lbfgs --out fit
```

```text
basis Cr,Mn,Fe,Co,Ni order 2 max-degree 5 -> 206 B functions (elements from the data)
r0 2.500 A (mean bond length of the basis; pass --r0 to override)
...
test map {'E': {'rmse': 25.5964, ...}, 'F': {'rmse': 0.175, ...}, 'V': {'rmse': 10.8308, ...}}
ood map {'E': {'rmse': 30.316, ...}, 'F': {'rmse': 0.1982, ...}, 'V': {'rmse': 10.5128, ...}}
```

- With five elements the default **categorical** basis gives every species
  pair its own radial functions, so it grows quickly: 206 many-body functions
  per species here, 1055 coefficients in all, from a small order and degree.
  `--basis-embedding <table.json>` builds a species-embedded basis instead,
  where species enter through a frozen element embedding and `--d-max` caps
  the channel widths.
- `--ood` adds a second test set, scored into `metrics_ood.csv`. The vacancy
  cells are only slightly harder than the bulk test set: a vacancy changes
  local environments that the bulk data already samples.

## Out of distribution: compression

```bash
aj eval --model fit/model.npz --data cantor_compressed.xyz $K --forces
```

```text
E RMSE 179.236 meV/atom  (30.0 configs)
F RMSE 0.9296 eV/A  (3456.0 components)
```

The errors are several times larger than in distribution: compressed cells
put neighbours at distances the training set never contains, and a linear
model extrapolates there without warning. The remedy is data: add compressed
cells to the training set.
