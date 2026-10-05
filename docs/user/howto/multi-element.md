# Fit a multi-element model

For more than one element, the command line does not change: `aj fit`
builds the basis for all species in the training data.

On this page, you fit the five-component CrMnFeCoNi (Cantor) alloy. The
training data is 40 small bulk cells, labelled by the MACE-MH-1 foundation
model. Then you test the model in and out of distribution. The procedure
takes approximately 4 minutes on a CPU.
[Tutorial 3](../tutorials/multi-element.md) fits the same data in a notebook.
It also compares the default categorical basis with species-embedding
bases.

## The data

```bash
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_train.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_test.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_vacancy.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/cantor_compressed.xyz
```

The labels in these files are `mace_energy`, `mace_force` and
`mace_virial`. Thus each command below gives these names with
`--energy-key`, `--force-key` and `--virial-key`.

| File | Configs | Contents |
|---|---|---|
| `cantor_train.xyz` | 40 | bulk 32-atom fcc cells, strained and rattled |
| `cantor_test.xyz` | 30 | the same distribution: in-distribution test |
| `cantor_vacancy.xyz` | 30 | 47-atom cells with one vacancy, unrelaxed: out of distribution |
| `cantor_compressed.xyz` | 30 | compressed bulk: out of distribution |

The [data README](https://github.com/ACEsuit/ace-jax/blob/main/docs/user/tutorials/data/cantor/README.md)
tells how the subsets were made.

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

The files have no `config_type`, so each table has one row.

- With five elements, the default **categorical** basis gives each species
  pair its own radial functions. Thus the basis becomes large quickly. Here,
  a small order and degree give 206 many-body functions for each species,
  and 1055 coefficients in total.
- `--basis-embedding <table.json>` builds a species-embedded basis. In this
  basis, the species enter through a frozen element embedding, and
  `--d-max` sets the maximum channel width.
- `--ood` adds a second test set. The fit writes its scores to
  `metrics_ood.csv`. The errors on the vacancy cells are only a little
  larger than on the bulk test set: a vacancy changes local environments
  that the bulk data already contains.

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

The errors are several times larger than in distribution. In compressed
cells, the neighbour distances are not in the training set. A linear model
extrapolates to these distances without a warning. To correct this, add
compressed cells to the training set.
