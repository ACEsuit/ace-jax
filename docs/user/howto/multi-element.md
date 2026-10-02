# Fit a multi-element model

Nothing changes on the command line for several elements: `aj fit` builds
the basis for every species it finds in the training data. This page fits the
five-component CrMnFeCoNi (Cantor) alloy from 40 small bulk cells labelled by
the MACE-MH-1 foundation model, then tests it in and out of distribution. It
takes about four minutes on a CPU.
[Tutorial 3](../tutorials/multi-element.md) fits the same data in a notebook
and compares the default categorical basis with species-embedding bases.

## The data

```bash
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_train.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_test.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_vacancy.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_compressed.xyz
```

These files store their labels as `mace_energy`, `mace_force` and
`mace_virial`, so every command below names them with `--energy-key`,
`--force-key` and `--virial-key`.

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
    --train cantor_train.xyz --test cantor_test.xyz --ood cantor_vacancy.xyz \
    --energy-key mace_energy --force-key mace_force --virial-key mace_virial \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --out fit
```

```text
basis Cr,Mn,Fe,Co,Ni order 2 max-degree 5 -> 206 B functions (elements from the data)
r0 2.500 A (mean bond length of the basis; pass --r0 to override)
...
RMSE, test (map)
-----------------------------------------------------------------
config type  configs  atoms  E (meV/atom)  F (eV/Å)  V (meV/atom)
-----------------------------------------------------------------
all               30   1200         25.60    0.1750        258.42
RMSE, ood (map)
-----------------------------------------------------------------
config type  configs  atoms  E (meV/atom)  F (eV/Å)  V (meV/atom)
-----------------------------------------------------------------
all               30   1122         30.32    0.1982        281.66
```

The files carry no `config_type`, so each table has a single row.

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
aj eval --model fit/model.npz --data cantor_compressed.xyz \
    --energy-key mace_energy --force-key mace_force --virial-key mace_virial
```

```text
RMSE, cantor_compressed.xyz vs fit/model.npz
-----------------------------------------------------------------
config type  configs  atoms  E (meV/atom)  F (eV/Å)  V (meV/atom)
-----------------------------------------------------------------
all               30   1152        179.24    0.9296       1097.39
```

The errors are several times larger than in distribution: compressed cells
put neighbours at distances the training set never contains, and a linear
model extrapolates there without warning. The remedy is data: add compressed
cells to the training set.
