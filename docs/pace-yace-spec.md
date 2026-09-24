# PACE `.yace` import/export — design spec

Status: design agreed 2026-09-24; implementation not started.

## Goal

Load a pacemaker `.yace` (PACE `ACECTildeBasisSet`) potential into ace-jax for
**fast JAX inference and lammps-jax deployment**, and write compatible models
back out as `.yace`.

Success means:

- energy / forces / virial agree with python-ace (pyace) and LAMMPS
  `pair_style pace` to within the C++ evaluator's own radial-spline error
  (measured once and pinned; expected ~1e-8 eV/atom in f64);
- the model exports to lammps-jax as stock StableHLO (no FFI custom calls);
- `load_yace -> write_yace` round-trips to an equal parsed YAML tree, and a
  model modified in JAX exports to a `.yace` that pyace evaluates identically;
- inference speed competitive with LAMMPS `pace` (CPU) and `pace/kk` (GPU),
  benchmarked on moriarty.

## Non-goals

- GRACE (including GRACE-1L-FS): different schema; separate spec once a real
  file has been inspected. It may map onto the factorised radial.
- Exporting ACEpotentials-native models to `.yace` (ACE.jl spline radial).
  Export covers only models whose radial is a PACE family.
- B-basis methods on imported models (`site_basis`, `site_descriptors`,
  `edge_jacobian`), and fitting. They raise a clear error on a `PACEModel`.
- Per-bond differing `radbasename`: `NotImplementedError` until a real model
  needs it.
- The pacemaker B-basis `potential.yaml` format.

## Architecture

```
model.yace ──parse──► PACESpec (numpy/dicts, 1:1 with the YAML)
                          │ build (offline, numpy)
                          ▼
   PACEModel(eqx.Module)
     leaves:  crad, radparams, core, ctilde_complex, fs_params, rho_core_cut, E0
     static:  radbasename, embedding kind, ndensity, index tables, T (sparse)
                          │
     rij → g_k(r), R_nl(r) (analytic, per bond) ; Ylm (real, existing)
         → edge A (narrow) → pool by (node, zj) → A (n_nodes, NZ·n_A_local)
         → AA (aa_specs) → ρ_p = AA · ctilde_real[:, zi, p]
         → E_i = E0 + F_zi(ρ)·switch(ρ_core) + core/ZBL
                          │
          ├─ ACECalculator / eval api (unchanged)
          ├─ lammps-jax export (unchanged)
          └─ write_yace(model)
```

`PACEModel` is a **sibling** of `ACEModel`, not more optional fields on it.
It reuses the harmonics, pooling and neighbour-list helpers as plain
functions. `aj.load(path)` dispatches on the `.yace` extension, so the CLI
`eval`/`predict` paths need no change.

New and changed files:

| file | content |
|---|---|
| `eval/pace_io.py` | `load_yace`, `write_yace`; PyYAML with the C loader |
| `eval/pace_build.py` | complex→real map T, A/AA index construction, padding |
| `eval/pace_model.py` | `PACEModel` |
| `eval/radial.py` | `cheb_exp_cos`, `cheb_pow`, `simplified_bessel`, `fexp` |
| `eval/io.py` / `__init__.py` | extension dispatch in `load` |
| `pyproject.toml` | add `pyyaml` |
| `pace_ref/` | reference env recipe + `make_fixtures.py` |
| `.github/workflows/pace-parity.yml` | path-gated parity job |

## Representation

```python
class PACEModel(eqx.Module):
    # trainable leaves (what write_yace serialises)
    crad            # (NZ, NZ, nradmax, lmax+1, nradbase), zero-padded per bond
    radparams       # (NZ, NZ, k): lambda, rcut, dcut, rcut_in, dcut_in, ...
    core            # (NZ, NZ, 2): prehc, lambdahc
    ctilde_complex  # (n_terms, ndensity): yace ms_comb coefficients, flat
    fs_params       # (NZ, 2*ndensity)
    rho_core_cut    # (NZ, 2)
    E0              # (NZ,)
    # int leaves
    a_nl, a_lm              # local A entry -> (radial column, Ylm column)
    aa_specs                # per order, into flattened (NZ * n_A_local) A
    T_rows, T_cols, T_vals  # ctilde_real[(a, z0)] += v * ctilde_complex[t]
    # static
    radbasename, embedding_kind, ndensity, lmax, nradmax, nradbase, elements,
    yace_meta               # non-numeric YAML fields, verbatim, for export
```

### Neighbour-species channel via pooling

PACE's A is indexed (μj, n, l, m); ace-jax carries species only through the
radial. Edge features stay narrow: each edge emits `[g_k | R_nl] ⊙ Ylm`
entries for its own (zi, zj). Pooling uses segment id `node*NZ + zj`, giving
A of shape (n_nodes, NZ, n_A_local), reshaped to (n_nodes, NZ·n_A_local).
There is no NZ-times widening of per-edge work. The dense (n, K) layout uses
a one-hot contraction over K.

### One shared set of AA products

Products are keyed by (μj, n, l, m) tuples; the central species μ0 enters only
through the radial R^{μ0 μj}. The AA set is the union over centres, and
`ctilde_real` is (n_AA, NZ, ndensity), the same shape convention as
`ACEModel`'s folded readout. **Assumption to measure:** the union is close to
one element's set, not NZ times as large.

### Complex → real map T

PACE forms products of complex A_lm; ace-jax uses real (SpheriCart) Y_lm.
At build time:

1. write each complex A_lm as at most 2 real A_lm′ (fixed per-l unitary map
   between PACE's Y_lm convention and SpheriCart's);
2. expand each `ms_comb` product, keep the real part;
3. merge terms by sorted real-index tuple.

This yields `aa_specs` and a sparse real T. The forward pass computes
`ctilde_real = segment_sum(T_vals * ctilde_complex[T_cols], T_rows)`. That is
negligible per call, keeps gradients on the complex leaf, and means export
never needs an inverse map.

**To verify from source:** PACE's Y_lm normalisation and phase, and whether
the `.yace` lists negative m explicitly or relies on conjugate symmetry.

Rank-1 functions use the radial base g_k directly (not R_nl); they are
order-1 AA entries pointing at the g_k columns.

Bonds with differing `nradmax`/`lmax`/`nradbase` are zero-padded.

## Radials, embedding, core terms

All formulas below are **ported line by line from ML-PACE's C++**
(`ace_radial.cpp`, `ace_evaluator.cpp`), not re-derived; the source is the
authority for every constant.

- **Radial base g_k(r)**: `cheb_exp_cos`, `cheb_pow`, `simplified_bessel`,
  each with PACE's cosine cutoff over [rcut − dcut, rcut]. SBessel roots are
  computed once at load (static table).
- **R_nl(r)** = Σ_k crad[zi, zj, n, l, k] g_k(r): an einsum against
  `crad[zi, zj]` gathered per edge. The per-edge gather is O(S²) in memory
  traffic; a per-bond-type contraction is the lever if benchmarks show it
  matters at many elements.
- **Neighbour-list cutoff** is the max bond `rcut`; smaller bonds are zeroed by
  their own cutoff function.
- **Embedding F**: `FinnisSinclair` / `FinnisSinclairShiftedScaled`, per
  element, summed over densities via `fs_params`. The power law is **PACE's
  `Fexp` ported exactly**, including its small-|ρ| linear blend. That keeps
  value parity everywhere and gives a finite gradient at ρ = 0, which padded
  and neighbourless nodes in lammps-jax buffers hit. The `fit/` smoothed
  signed sqrt (ρ(ρ²+1e-6)^(−1/4)) is **not** reused: its values differ at
  small ρ.
- **Inner regimes**: the loader reads which one the `.yace` declares:
  1. legacy density-based: hard-core repulsion from `prehc`/`lambdahc`,
     feeding ρ_core; the embedding energy is switched off as ρ_core crosses
     `rho_core_cutoff` ± `drho_core_cutoff`;
  2. distance-based: a switch on [rcut_in − dcut_in, rcut_in] multiplied into
     the radials, with ZBL below. Reuse `fit/zbl.py` (lifted into `eval/`) only
     if its form matches PACE's exactly; otherwise port PACE's.
- **Site energy**: E_i = E0[zi] + F_zi(ρ)·switch(ρ_core) + core repulsion.
  Forces and virial by autodiff through the existing edge-vector wrapper.
- **Precision**: nothing touches `jax.config`; `highest_precision` applies as
  for `ACEModel`.

## Writer

`write_yace(model, path)` accepts only a `PACEModel`.

- Numeric fields come from the live leaves: `crad` (un-padded per bond),
  `radparams`, `core`, `ctilde_complex` (split back per function), `fs_params`,
  `rho_core_cut`, `E0`.
- Everything structural (element order, `ns`/`ls`/`mus`/`ms_combs`,
  `radbasename`, embedding and cutoff types, unknown keys) comes verbatim from
  `yace_meta`. The function layout is never regenerated.
- Floats are written round-trip exact. `load_yace` keeps f64 leaves even when
  evaluating in f32; dtype is an evaluation property.
- Pruning zero-ctilde functions is the only structural edit contemplated, and
  is deferred until needed.

## Testing

### Reference environment

Mirrors julia-parity: references are generated out-of-band and committed as
fixtures; the everyday suite stays pip-only.

- **python-ace**: no PyPI wheels; build from source
  (`pip install git+https://github.com/ICAMS/python-ace`, needs CMake + C++17)
  in its own env, not ace-jax's.
- **LAMMPS, local (Mac)**: `~/gits/lammps` has the ML-PACE source but
  `build/` is configured with `PKG_ML-PACE=OFF`. Reconfigure with
  `-D PKG_ML-PACE=ON` and rebuild (configure downloads `lammps-user-pace`).
  CPU only.
- **LAMMPS, moriarty**: already built with ML-PACE; use it for GPU
  validation (`pace/kk` if Kokkos is present) and speed comparisons.
- `pace_ref/make_fixtures.py` writes `fixtures/pace_ref_*.npz` (E, F, virial
  per structure) alongside each `.yace`.
- `pace-parity` CI job (path-gated, cached python-ace build) regenerates and
  compares.

### Fixture models

Random-coefficient models from pyace's basis-config tools, converted with
`pace_yaml2yace`:

| fixture | covers |
|---|---|
| 1 element, ChebExpCos, ndensity 1, linear F | simplest path |
| 1 element, ChebPow, ndensity 2, FS √ | Fexp, several densities |
| 3 elements, SBessel, FSShiftedScaled, mixed `nradmax` | species channels, padding, SBessel |
| 2 elements, legacy hard-core + ρ_core switch | inner regime 1 |
| 2 elements, distance inner cutoff + ZBL | inner regime 2 |
| one real published `.yace` | realistic size; perf; AA-union measurement |

### Test layers

1. **Unit**: g_k per family against the C++ (pyace radial functions if
   exposed, else a direct numpy port); T map (complex and real products equal
   on random A); `fexp` values and gradients incl. ρ = 0; ρ_core switch.
2. **Model**: E/F/virial against pyace fixtures at the pinned tolerance.
3. **Consistency** (reusing `test_efv` patterns): finite-difference forces
   and virial; sparse vs dense layout; padded neighbourless node and isolated
   atom give finite results.
4. **Round trip**: parsed-tree equality after load→write; scale
   `ctilde_complex` by 2, export, pyace energy change equals JAX's prediction;
   LAMMPS `pair_style pace` reads the exported file.
5. **Deployment** (manual, not CI): lammps-jax export vs `pair_style pace` on
   a snapshot and a short NVE run; speed benchmark on moriarty vs `pace` (CPU)
   and `pace/kk` (GPU).

## Build order

1. Parser, `PACESpec`, T map, with unit tests.
2. Radial families and `fexp`, with unit tests.
3. `PACEModel` + readout, parity on the simplest fixture.
4. Remaining fixtures (densities, species, inner regimes).
5. Writer and round-trip tests.
6. Deployment check and benchmarks on moriarty.

## Compatibility with open PRs (checked 2026-09-24)

No merge prerequisite; branch from `main`. PRs #2–#4 touch `fit/*`,
`construct/*` and `cli.py`; #5 touches `fit/*`, `uv.lock` and appends two
methods to `ACEModel`. None touch the `eval/` files changed here. Expect a
trivial `uv.lock` conflict with #5 (PyYAML); keep changes out of `cli.py`.
