# ARD force uncertainty: validation of revision 2

Moved from the user page [Per-atom force uncertainty](../user/howto/force-uncertainty.md)
in the ASD-STE100 review of the user documentation.

Revision 2 was validated on a CrMnFeCoNi (Cantor) alloy benchmark labelled by
the MACE-MH-1 foundation model: 3680 training and 920 test configurations
(bulk cells, vacancies, surfaces and stacking faults; a 15k-function basis),
and 34 large cells of 3–4k atoms with cracks and edge and screw dislocations,
none of them in training. Coverage of `forces_q` at nominal 0.90:

| Setting | In distribution (target 0.90 ± 0.01) | Crack (≥ 0.89) | Crack tip (≥ 0.88) | Edge, screw (≥ 0.90) |
|---|---|---|---|---|
| default (`aniso`, transfer exponent) | 0.898 | 0.903 | 0.885 | 0.937, 0.941 |
| `--force-shape iso` | 0.897 | 0.889 | 0.862 | 0.933, 0.937 |

`aj calibrate` on crack cells, leaving one crack realisation out (ranges over
the four folds; scored on the held-out crack pair and the dislocation cells,
and in distribution on a 300-configuration test sample):

| Posterior | In distribution | Crack | Crack tip | Edge, screw |
|---|---|---|---|---|
| `aniso`, uncalibrated (default) | 0.898 | 0.899–0.905 | 0.881–0.884 | 0.937, 0.941 |
| `aniso` + calibrate, per-group replace | **0.867–0.870** | 0.910–0.917 | 0.899–0.904 | 0.927–0.930, 0.932–0.935 |
| `aniso` + calibrate `--append` | 0.895 | 0.900–0.906 | 0.881–0.886 | 0.936, 0.941 |
| `iso` + calibrate `--append` | 0.895 | 0.895–0.903 | 0.869–0.878 | 0.934–0.935, 0.937–0.939 |

- The default meets every target without target data.
- Without the transfer exponent the hold-out scale is too small (crack tip
  0.81): the error is dominated by approximation error, so the scale fitted on
  80 % of the data underestimates the model fitted on all of it.
- Putting crack cells in training improves the tip (0.87–0.90) but not in
  every fold; the jackknife block size has no measurable effect.
- The rank correlation between `forces_std` and the actual error is 0.29–0.38
  on the large cells.

The full results, with confidence intervals and per-configuration-type
tables, are in the
[acceptance report](https://github.com/ACEsuit/ace-jax/blob/main/bench/defect_uq/results/2026-10-03_rev2_acceptance.md).

