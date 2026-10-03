# Labels for the MLIP-school-derived tutorials

These files hold the precomputed labels for the tutorials adapted from the
[MLIP School 2026](https://mlipschool.uk/) workshop
notebooks (E1, E1x, E2, E3, C and D). With them, a tutorial's default settings run without
a labeller. `ace_jax.tutorials.labels.label()` serves them, keyed on structure
content, and labels a structure live only when it is not here.

| File | Structures | Labeller |
|---|---|---|
| `e1/labels-mpa-0.xyz` | E1's slider grid (6 strains × 5 rattles × 10 cubic 8-atom Si cells) and the 64/63-atom vacancy pair | MACE-MPA-0 |
| `e1x/reference.xyz` | the 250 rattled-bulk Si frames of the school's E1x reference set, relabelled | MACE-MPA-0 |
| `c/labels-mpa-0.xyz`, `c/labels-mp-0b3.xyz` | C's bulk and 6-layer (111) slab, plus the surface recipe: 8 rattled bulk cells and 4-layer (100), (110) and (111) slabs | MACE-MPA-0 and MACE-MP-0b3 |
| `e2/labels-mpa-0.xyz` | tutorial 6: the slab grid (4 to 12 layers × 6 to 12 Å vacuum, three faces), the bulk reference, and the displaced repair slabs | MACE-MPA-0 |
| `e2/relaxed-mpa-0.xyz` | tutorial 6: every grid slab relaxed by the labeller (BFGS, fmax 0.03 eV/Å), its energy, keyed by the unrelaxed slab (`from_key`) | MACE-MPA-0 |
| `e3/pools.xyz`, `e3/labels-mpa-0.xyz` | tutorial 7: the MD pools of the canonical campaign (seed 0; per `driver` and `round`), and labels for every frame | MACE-MPA-0 |
| `d/labels-mpa-0.xyz` | tutorial 9: the GaAs demo (isolated atoms, bulk and (100) slab targets, the slider grid of training cells, the repair slabs) | MACE-MPA-0 |

## Models and licence

- **MACE-MPA-0** (`medium-mpa-0`) and **MACE-MP-0b3** (`medium-0b3`) come from
  [ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations). Both
  are MIT-licensed, so these labels and this data are MIT-licensed too.
  - They were evaluated with mace-torch 0.3.16, in float64 on CPU.
  - Both are trained on Materials Project data (PBE+U). The +U correction does
    not apply to Si.
  - Cite: I. Batatia et al., *A foundation model for atomistic materials
    chemistry*, arXiv:2401.00096.
- The E1x structures are those of the MLIP-school-2026 reference set. Their
  original labels are dropped.

## Conventions

- Info `energy` is in eV.
- Arrays `forces` are in eV/Å.
- Info `virial` is the 3×3 virial in eV, `virial = −stress · V`, for periodic
  cells only.
- Info `label_model` is `mpa-0` or `mp-0b3`.
- Info `structure_key` is the content key the cache is looked up by.
- Info `config_type` is kept from the structure.
- Floats are written at extxyz's 8 decimals.

## Regenerating

The structures come from `ace_jax.tutorials.structures`, so the notebooks
rebuild exactly these. To regenerate the labels, use an environment with
mace-torch, as described in the docstring of `make_labels.py`:

```bash
PYTHONPATH=src <mace-env>/bin/python docs/user/tutorials/data/school/make_labels.py a0
PYTHONPATH=src <mace-env>/bin/python docs/user/tutorials/data/school/make_labels.py all \
    --e1x-source <school sources>/notebooks/reference/e1x-bulk-reference.xyz
```

`all` also writes the `e2`, `e3` and `d` files; each has its own mode. `e2`
relaxes 60 slabs with MACE (about an hour on a CPU); `e3` runs the curation
campaign of `ace_jax.tutorials.curation` with live labels (about 15 minutes) and
needs ace-jax's fitting in the same environment.

`a0` prints MACE-MPA-0's diamond-Si lattice constant, 5.4672 Å. That is
`structures.E1_A0`, the lattice constant the vacancy pair is built at.
