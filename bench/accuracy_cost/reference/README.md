# Reference: the ACEpotentials.jl Si fit that matched GAP-18

`ACEworkflows_Silicon_2023_fit_model.jl` is a verbatim copy of
`ACEsuit/ACEworkflows` → `Silicon_2023/fit_model.jl`. It was fetched on 2026-10-05
at commit `155fcb8a6f1228143a3c0a0a184377aba47e5486`:
<https://github.com/ACEsuit/ACEworkflows/blob/155fcb8a6f1228143a3c0a0a184377aba47e5486/Silicon_2023/fit_model.jl>.

The upstream repository has no licence file. The copy is kept as a reference for
this benchmark, and its settings are summarised below.

It fits the Si potential reported in the ACEpotentials.jl paper on the same GAP-18
training set (`gp_iter6_sparse9k.xml.xyz`, Bartók et al., PRX 8, 041048 (2018)).
The upstream `Project.toml` pins ACEpotentials 0.6.x and ACE1x 0.1.8 (Julia 1.9).

## Settings, and how they compare with this benchmark

| setting | ACEworkflows Silicon_2023 | accuracy-cost benchmark |
|---|---|---|
| data | the whole GAP-18 set (train only, no split) | 90/10 stratified split (seed 0) |
| E0 | -158.54496821 | same |
| basis | `acemodel(order=4, totaldegree=20)`; radial `transform=(:agnesi,2,4)`, `envelope=(:x,2,3)`; pair `pair_transform=(:agnesi,1,4)`, `pair_envelope=(:r,3)` (ACE1 style) | ladder up to order 4, max degree 16 (`sweep.yaml`); ace-jax's `ace_model`-style radials |
| cutoff | the `acemodel` default for Si (printed at runtime, not set) | 5.0 Å for every family |
| solver | `ACEfit.BLR(factorization=:svd)`: one shared noise variance, evidence-fitted | ace-jax MAP (#63); `--noise shared` (#67) is the BLR equivalent |
| smoothness prior | `smoothness = 5` | ace-jax default (p = 4) |
| weights `default` | E 30, F 1, V 1 | E 30, F 1, V 1 |
| `liq` | E 10, F 0.66, V 0.25 | E 10, F 0.667, V 0.25 |
| `amorph`, `sp` | E 3, F 0.5, V **0.1** | E 3, F 0.5, V **0.125** |
| `bc8`, `vacancy`, `interstitial` | E **50**, F 0.5, V 0.1 (hand-tuned) | default (E 30, F 1, V 1) |
| `dia` | E **60**, F 2, V 2 (hand-tuned) | default |
| virials | fitted | fitted (ace-jax, GAP, MACE); PACE E+F only |
| deployment | `.yace` plus a pair-potential table: LAMMPS `pair_style hybrid/overlay pace table spline 5401` | — |

What this means for the benchmark:
- Our `liq`, `amorph` and `sp` weights, derived from GAP-18's `config_type_sigma` and
  rescaled onto ACEpotentials' default, reproduce the Silicon_2023 values. The one
  difference is V 0.125 against 0.1 for `amorph`/`sp`.
- Silicon_2023 also hand-tunes `bc8`, `vacancy`, `interstitial` and `dia`, which upweights
  energies on the defect and ground-state types. Its accuracy partly comes from that
  tuning, which GAP-18's own sigmas do not contain.
- Its basis (order 4, totaldegree 20) is larger than our ladder's top (o4d16), and it
  uses a larger default cutoff. Keep both in mind when quoting it as "ACE matches or
  beats GAP-18".
- p = 5 smoothness against ace-jax's default p = 4.
