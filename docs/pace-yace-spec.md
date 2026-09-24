# PACE `.yace` import/export — design spec

Status: design agreed 2026-09-24; checked against ML-PACE source
(`ICAMS/lammps-user-pace` @ `99aa6e6`) the same day; implementation not started.

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
- Exporting ACEpotentials-native models to `.yace`. ML-PACE does read
  `ACE.jl.*` tabulated radials (`ace_c_basis.cpp`), so this is a feasible
  follow-up; it is not part of this spec, and importing such files is
  rejected too.
  Export covers only models whose radial is a PACE family.
- B-basis methods on imported models (`site_basis`, `site_descriptors`,
  `edge_jacobian`), and fitting. They raise a clear error on a `PACEModel`.
- Per-bond differing `radbasename`: `NotImplementedError` until a real model
  needs it. (The C++ does allow it.)
- Per-bond differing `inner_cutoff_type`: rejected. The C++ keeps one global
  value (last bond wins), so such files are ill-defined.
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
| `eval/pace_radial.py` | `cheb_exp_cos`, `cheb_pow`, `cheb_linear`, `simplified_bessel`, `cutoff_func_poly`, `radcore`, `pace_zbl`, `fexp`, `fexp_shifted_scaled` |
| `eval/io.py` / `__init__.py` | extension dispatch in `load` |
| `pyproject.toml` | add `pyyaml` |
| `pace_ref/` | reference env recipe + `make_fixtures.py` |
| `.github/workflows/pace-parity.yml` | path-gated parity job |

## Representation

```python
class PACEModel(eqx.Module):
    # trainable leaves (what write_yace serialises)
    crad            # (NZ, NZ, nradmax, lmax+1, nradbase), zero-padded per bond
    radparams       # (NZ, NZ, 5): lambda, rcut, dcut, rcut_in, dcut_in
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
    radbasename, inner_cutoff_type, npoti (per element), ndensity, lmax,
    nradmax, nradbase, n_a_local, elements

# returned alongside the model, not stored on it: a dict is not hashable, so
# it cannot be an Equinox static field without breaking jit
@dataclass
class PACESpec:
    tree: dict              # the parsed YAML, verbatim (structural fields for export)
    element_names: list
    functions: list         # per function: (element, index, term_start, num_ms_combs, ndensity)
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

1. write each complex A_lm as at most 2 real A_lm′, using the convention
   map given under "R_nl and the A basis" below (verified against the C++);
2. expand each `ms_comb` product, keep the real part;
3. merge terms by sorted real-index tuple.

This yields `aa_specs` and a sparse real T. The forward pass computes
`ctilde_real = segment_sum(T_vals * ctilde_complex[T_cols], T_rows)`. That is
negligible per call, keeps gradients on the complex leaf, and means export
never needs an inverse map.

Rank-1 functions use the radial base g_k directly (not R_nl), times
Y00 = 1; they are order-1 AA entries pointing at the g_k columns.

Bonds with differing `nradmax`/`lmax`/`nradbase` are zero-padded.

## Radials, embedding, core terms

Every formula here was checked against `ICAMS/lammps-user-pace` @ `99aa6e6`
(`ML-PACE/ace-evaluator/`: `ace_radial.cpp`, `ace_abstract_basis.cpp`,
`ace_evaluator.cpp`, `ace_spherical_cart.cpp`, `ace_c_basis.cpp`) and is
ported from it line by line. The line references below are to that commit.

### Bond parameters (YAML → leaves)

Per bond `[μi, μj]`: `radbasename`, `radparameters` (λ = element 0),
`radcoefficients` (→ `crad[n][l][k]`), `nradmax`, `lmax`, `nradbasemax`,
`rcut`, `dcut`, `rcut_in`, `dcut_in`, `prehc`, `lambdahc`,
`inner_cutoff_type`. The C++ allocates `crad` as
(NZ, NZ, max nradmax, max lmax + 1, max nradbasemax) and zero-fills. Our
padding matches that exactly.

`radbasename` values starting `ACE.jl` select a different, tabulated radial
path in ML-PACE. That's out of scope here (see Non-goals), and a model that
mixes ACE.jl and PACE radials is rejected by the C++ itself.

### Radial base g_k(r), `ace_radial.cpp:241–445`

g ≡ 0 for r ≤ cut_in − dcut_in or r ≥ rcut. Otherwise, by family:

| family | x | g | own cutoff |
|---|---|---|---|
| `ChebExpCos` | 1 − 2(e^{−λr/rc} − e^{−λ})/(1 − e^{−λ}) | g₀ = 1; g_{n−1} = ½ − ½T_{n−1}(x), n = 2..K | × ½(1 + cos πr/rc), then × ½(1 + cos π(r − rc + dcut)/dcut) on r > rc − dcut |
| `ChebPow` | 2(1 − (1 − r/rc)^λ) − 1 | g_{n−1} = ½ − ½T_n(x), n = 1..K | none (vanishes at rc by construction; `dcut` unused) |
| `ChebLinear` | 1 − r/rc | g_{n−1} = ½ − ½T_n(x) | none |
| `SBessel` | — | closed-form sinc sum f_n (`simplified_bessel_aux`) plus a 3-term orthogonalising recursion; **no roots needed** | none (zero for r ≥ rc) |

For `distance` and `zbl`, g is then multiplied by 1 − P(r; cut_in, dcut_in),
where P is `cutoff_func_poly` (a quintic from 1 at r_in − δ to 0 at r_in).
**Port the quirk at `ace_radial.cpp:246`:** for `zbl`, `cut_in = (dcut_in == 0)`,
so with dcut_in ≠ 0 the inner cutoff becomes P(r; 0, dcut_in) = 0 for r > 0
and g is left unmodified. The ACE-to-ZBL switch then happens only in the
energy assembly (below). A `zbl` fixture pins this.

### R_nl and the A basis, `ace_evaluator.cpp:315–380`

- R_nl(r) = Σ_k crad[zi, zj, n, l, k]·g_k(r). This is an einsum against
  `crad[zi, zj]` gathered per edge. That gather is O(S²) in memory traffic;
  a per-bond-type contraction is the lever if benchmarks show it matters at
  many elements.
- Rank 1: A¹[μj, k] = Σ_j g_k(r_j)·Y00, with **Y00 = 1**.
- Rank > 1: A[μj, n, l, m] = Σ_j R_nl(r_j)·Y_lm(r̂_j) for m ≥ 0, then
  A[l, −m] = (−1)^m·conj A[l, m].
- **Harmonics convention** (`ace_spherical_cart.cpp`, `ace_spherical_cart.h:51`):
  complex, Condon–Shortley phase, Y00 = 1, Y10 = √3·ẑ, Y11 = −√(3/2)(x̂ + iŷ).
  That is √(4π) times the standard complex Y_lm. In terms of SpheriCart's
  real L2-normalised Y^R (no Condon–Shortley phase), for m > 0:
  Y^PACE_{l,m} = √(4π)·(−1)^m (Y^R_{l,m} + i·Y^R_{l,−m})/√2,
  Y^PACE_{l,−m} = (−1)^m·conj(Y^PACE_{l,m}) = √(4π)·(Y^R_{l,m} − i·Y^R_{l,−m})/√2,
  and Y^PACE_{l,0} = √(4π)·Y^R_{l,0}. The T-map unit test confirms these
  numerically against a numpy port of the C++ recursion.
- Each function contributes ρ_p += Re(Π_t A[μ_t, n_t, l_t, m_t] · c̃_p)
  **over its `ms_combs` exactly as listed**. Whether a file stores full or
  half m-sets needs no special handling.
- Neighbours with r ≥ rcut(μi, μj) are skipped per bond. The neighbour-list
  cutoff is the max bond `rcut`.

### Embedding, `ace_abstract_basis.cpp:37–122`

`FS_parameters` holds [w₀, m₀, w₁, m₁, …] per element; E_F = Σ_p w_p·F(ρ_p; m_p).

- `FinnisSinclair` → `Fexp`, with w = 10⁶, λ = w^{1−m}:
  F = sign(x)·((1 − g)|x|^m + λ·g·|x|), g = exp(−(w|x|)³), with g := 0 when
  (w|x|)³ > 30; and F = λx for |x| ≤ 10⁻¹⁰. Ported with safe `where`s so the
  unused branch never produces a NaN gradient (padded and neighbourless
  nodes sit at x = 0). The `fit/` smoothed signed sqrt is **not** reused.
- `FinnisSinclairShiftedScaled` → `FexpShiftedScaled`: F = x if |m − 1| < 10⁻¹⁰;
  otherwise with a = |x|, e = e^{−a}, ν = 1/m: F = sign(x)·((ν^{ν/(1−ν)}·e + a)^m − ν^{1/(1−ν)}·e).

### Inner regimes and site energy, `ace_radial.cpp:615–700`, `ace_evaluator.cpp:497–536`

`inner_cutoff_type` is one global value in the C++ (the last bond read wins).
The loader **rejects** files whose bonds disagree, rather than copying that
behaviour. Missing ⇒ `density` (the backward-compatible default).

ρ_core = Σ_j c_r(r_j), where c_r depends on the regime:

| regime | c_r(r) | site energy |
|---|---|---|
| `density` | \|pre\|·e^{−\|λhc\|r²}/r · ½(1 + cos πr/rc) (0 if λhc·r² ≥ 50) | E0 + E_F·f + ρ_core, f = P(ρ_core; `rho_core_cutoff`, `drho_core_cutoff`) (`inner_cutoff`) |
| `distance` | the `density` c_r × P(r; r_in, δ_in) | same as `density` |
| `zbl` | pre·(K/2)·Zi·Zj·(φ(r/a)/r + S(r)) × P(r; rc, dcut); a = 0.4685/(Zi^0.23 + Zj^0.23), K = 14.399645351950543, φ is the 4-exponential ZBL screening, S is the C²-matching shift over [rc − dcut, rc] | E0 + E_F·f + ρ_core·(1 − f), with f = P(dcut_in − d_min; dcut_in, dcut_in), d_min = min_j (r_j − (cut_in − dcut_in)) and f = 1 with no neighbours |

The `zbl` switch depends on the **nearest neighbour's distance** per atom,
not on a sum. In JAX that is a per-node `segment_min` over edges (padded
edges masked to +∞). It's continuous and differentiable except where two
neighbours tie. `fit/zbl.py` is not reused: PACE's ZBL has its own shift
S(r) and cutoff, and needs Zi, Zj from the element names.

Forces and virial come from autodiff through the existing edge-vector
wrapper. Nothing touches `jax.config`; `highest_precision` applies as for
`ACEModel`.

## Writer

`write_yace(model, spec, path)` accepts only a `PACEModel`.

- Numeric fields come from the live leaves: `crad` (un-padded per bond),
  `radparams`, `core`, `ctilde_complex` (split back per function), `fs_params`,
  `rho_core_cut`, `E0`.
- Everything structural (element order, `ns`/`ls`/`mus`/`ms_combs`,
  `radbasename`, embedding and cutoff types, unknown keys) comes verbatim from
  `PACESpec.tree`. The function layout is never regenerated. The signature is
  therefore `write_yace(model, spec, path)`, and `load_yace` returns
  `(model, meta, spec)`, the same triple shape as `eval.io.load`.
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
| 1 element, ChebLinear | the fourth radial family |
| 3 elements, SBessel, FSShiftedScaled, mixed `nradmax` | species channels, padding, SBessel |
| 2 elements, `density` (hard-core + ρ_core switch) | regime 1 |
| 2 elements, `distance` | regime 2 |
| 2 elements, `zbl`, dcut_in ≠ 0, a close-contact structure | regime 3, nearest-neighbour switch, the `cut_in` quirk |
| one real published `.yace` | realistic size; perf; AA-union measurement |

### Test layers

1. **Unit**: g_k per family, c_r per regime and `cutoff_func_poly` against
   the C++ (pyace radial functions if exposed, else a numpy port); the
   harmonics convention map and T map (complex and real products equal on
   random A); `fexp`/`fexp_shifted_scaled` values and gradients, including
   x = 0; the ρ_core and nearest-neighbour switches.
2. **Model**: E/F/virial against pyace fixtures. The C++ evaluates splined
   g, R_nl and c_r on a grid of spacing `deltaSplineBins` (default 0.001 Å,
   stored in the `.yace`), while ace-jax is analytic. So each fixture is
   generated twice: once with `deltaSplineBins` reduced (e.g. 1e-5) for a
   tight parity check of the formulas, and once at the shipped value to
   measure and pin the real-world gap.
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

## Sub-project 2 (deferred): exporting ace-jax / ACEpotentials models to ML-PACE

Not part of this spec; recorded here so the findings aren't lost. Start only
after PACE import/export lands **and** the upstream C++ change below has been
agreed with ICAMS (coordinated by James Kermode).

### History and current state (checked 2026-09-24)

- ACEpotentials v0.6 (ACE1) `export2lammps` (now `src/outdated/export.jl`)
  wrote `radbasename: "ACE.jl"` bonds holding per-bond cubic-Hermite tables
  (`splinenodalvals`, `splinenodalderivs`, `nbins`, `rcut`; R_nl independent
  of l), a linear embedding (`FinnisSinclairShiftedScaled` with m = 1,
  `rho_core_cutoff` 1e5), ctildes c/(4π)^{ν/2}, and the pair potential as a
  separate LAMMPS `pair_style table` file.
- That format is read only by wcwitt's fork (`wcwitt/lammps-user-pace`,
  `f92fcdb` "change ships_radial to acejl_radial with splines", 2023-04; now 14
  commits behind upstream). Upstream sends every `ACE.jl*` radbasename to the
  older SHIPs reader (polynomial recursion coefficients), so spline-format
  files fail on stock ML-PACE. That is the break.
- ACEpotentials ≥ 0.8 has no export (`fit_model.jl:173`: "automatic lammps
  export currently not supported").
- `pace/kk` (GPU) only accepts `ACERadialFunctions`
  (`KOKKOS/pair_pace_kokkos.cpp:252`); SHIPs and the fork's radial are both
  CPU-only.
- LAMMPS pins the PACE library via `PACELIB_URL`/`PACELIB_MD5`
  (`cmake/Modules/Packages/ML-PACE.cmake`; currently `v.2023.11.25.fix2`), so a
  patched library can be built without waiting for a LAMMPS release.

### Findings

1. **Projecting into native PACE radials is lossy.** A spike fitting the
   `si_fitted` fixture's R_nl into ChebExpCos/ChebPow reached only ~6e-4
   worst-case relative error at K = 80 (median ~2e-5), concentrated at the
   outer cutoff, with slow algebraic convergence. Exact export needs a
   tabulated radial.
2. **Species expansion for the PACE format.** `ACEModel` pools all neighbours
   into one A (species only via the radial); PACE needs per-neighbour-species
   A. An order-ν product expands into up to NZ^ν PACE products (C(NZ+ν−1, ν)
   multisets): fine at 1–3 elements, heavy at 5 elements / order 4.
3. **Real → complex with real ctildes** (needed for the PACE format) is
   lossless only if the imaginary parts cancel, which O(3)-invariant bases
   (Σl even) guarantee. It must be tested, not assumed.
4. **Embedded-species models fit ML-PACE's GRACE-FS format natively**
   (`ML-PACE/ace/grace_fs_evaluator.cpp`, in the library since
   `v.2024.9.11`). GRACE-FS has A(n,l,m) = Σ_j Z[μj, n]·R_nl(r)·Y^R_lm(r̂)
   with **one species-independent radial**, a chemical embedding Z
   (nelements × nradmax), **real** harmonics, and per-centre-element functions
   with `gen_cgs` × per-density `coeff`, plus the FS embedding. ace-jax's
   factorised radial (frozen embedding, uniform cutoffs:
   R_i = P_{n′(i)}(r)·emb[zj, k(i)]) maps onto it channel by channel
   (Z[:, i] = emb[:, k(i)], R_i ← P_{n′(i)}), with **no species expansion and
   no complex conversion**, only a per-(l,m) normalisation/sign map between
   SpheriCart and PACE's real Y. Its blockers:
   - GRACE-FS's radial base is SBessel-only; the same tabulated-radial patch
     is needed there (`GRACEFSRadialFunction` already holds
     `splines_gk`/`splines_rnl`).
   - GRACE-FS has no pair term. ace-jax's pair basis (full (zi, zj)
     dependence) can be folded into rank-1 functions using extra channels per
     (μi, μj, q) with one-hot Z columns, used only by the matching centre
     element; alternatively a LAMMPS `pair_style table` file.
   - Which LAMMPS release ships `pair_style grace/fs`, and whether a Kokkos
     variant exists, must be checked. The local `~/gits/lammps` (Nov 2025,
     library `v.2023.11.25.fix2`) has neither.

   Non-factorised `ACEModel`s (per-pair radial tables) cannot use GRACE-FS,
   because its radial does not depend on (μi, μj). They need the PACE format
   with species expansion.

### Proposed route (to be designed in its own spec)

- **Upstream C++ (coordinate with ICAMS):** a tabulated radial type, under a
  new `radbasename` distinct from `ACE.jl*`, implemented *inside*
  `ACERadialFunctions` (and `GRACEFSRadialFunction`) by loading nodal values
  and derivatives straight into the existing `splines_gk`/`splines_rnl`. Being
  the same class and the same spline structures, it keeps `pace/kk` working
  without Kokkos changes.
- **ace-jax exporters:**
  - factorised/embedded models → GRACE-FS YAML (preferred: compact, real
    harmonics);
  - other `ACEModel`s → PACE `.yace` with species expansion and real→complex;
  - both tabulate radials from JAX with exact autodiff derivatives.
- **Optional:** a writer for wcwitt's fork format (same tables), only if the
  fork still has users.
