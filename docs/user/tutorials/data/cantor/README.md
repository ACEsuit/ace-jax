# CrMnFeCoNi (Cantor alloy) example data

Four small extxyz files used by the multi-element how-to (`docs/user/howto/multi-element.md`).
Labels are stored as `mace_energy` (eV), `mace_force` (eV/Å) and `mace_virial` (eV).

| File | Configs | Atoms each | Contents |
|---|---|---|---|
| `cantor_train.xyz` | 40 | 32 | bulk random CrMnFeCoNi fcc solid solution, strained and rattled: training set |
| `cantor_test.xyz` | 30 | 32 | the same distribution, disjoint from the training set: in-distribution test |
| `cantor_vacancy.xyz` | 30 | 47 | one atom removed from a 48-atom bulk cell, unrelaxed: out of distribution |
| `cantor_compressed.xyz` | 30 | 32 | compressed bulk cells: out of distribution |

Total size about 670 KB.

## Provenance

- **Labels:** the MACE-MH-1 foundation model, head `matpes_r2scan`
  (r²SCAN-level), evaluated with the torch MACE implementation. The model
  reproduces the stored labels of the parent sets to 5e-5 eV/atom, 7e-4 eV/Å
  and 7e-3 eV (virial).
- **Source files** (in the maintainers' data cache `acegp-data/cantor/`):
  - `cantor1k_b_mh1.xyz`: 1000 bulk configurations (32 atoms, lattice
    constant 3.59 Å with ±strain, shear and rattle) from the bounded bulk
    generator;
  - `cantor_vac_mh1.xyz`: 200 single-vacancy cells, one atom removed at a
    uniformly random site (unrelaxed) from held-out parents of the
    4000-configuration set (info keys `ood_type=vacancy`, `parent`,
    `vac_site`, `vac_Z`);
  - `cantor_ood_mh1.xyz`: 200 compressed bulk cells (`ood_type=compress`).
- **Selection:** `make_subsets.py` (standard library only) copies whole
  frames verbatim, labels untouched, chosen by a seeded permutation
  (`random.Random(0)`) of each source file: the training and test sets are
  disjoint slices of the same permutation of `cantor1k_b_mh1.xyz`. It
  regenerates these files byte for byte:

    ```bash
    python make_subsets.py --src /path/to/acegp-data/cantor
    ```

  It prints the first 16 hex digits of each output's SHA-256:
  `cantor_train.xyz` 6909bdff19f1f06a, `cantor_test.xyz` a87858e436391bcf,
  `cantor_vacancy.xyz` 1e09fddf34741af4, `cantor_compressed.xyz` f87cb59861fe90cb.

## Terms

These labels are outputs of MACE-MH-1. Check the MACE-MH-1 model licence
before redistributing them or using them beyond research and teaching.
<!-- TODO(maintainers): confirm the MACE-MH-1 licence terms for redistributing model outputs, and state them here. -->
