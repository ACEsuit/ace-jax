# Acceptance: `--uq ard-gp` (Phase 3 of the GP-discrepancy spec), bench365 / v3

**Status:** 2026-10-07. Branch `feat/gp-sandwich` at 2d10079.

**Runs.** Modal B200 for the fits, A100-80GB for `big_errors`. Arms from `modal/ard_arms.py::GP_ARMS`:
- the `gp_conv` settings: PCA-16 density map, host-cache LML, 100 inducing sites per species;
- L-BFGS started from `gp_conv`'s converged θ, one start;
- aniso force shape, joint E0.

**Data.** bench365 train/test (3680/920 configurations) and all 34 v3 big cells.

**Scoring.**
- Big-cell coverage and ρ: `scoring/validate_shape.py`, coverage of the served region, cell-weighted.
- Paired differences against the linear default `ard_default` (rev2 §7): a 1000-resample bootstrap over cells. Both runs are scored on the same atoms, so the intervals on the differences are tight.
- In-distribution coverage: the rev2 300-configuration test sample (`scoring/d4_score.py::id300`, served through `GPCalculator(posterior=)`).

## Acceptance rule (spec, Phase 3)

> GP arm + sandwich (and + `dtc`) must **match or beat the linear-arm rev2 result** on crack-tip coverage
> and on Spearman ρ on large cells. If neither shape does, document the result and keep the option
> experimental (no default change).

## Results

| arm | ID (cfg-weighted) | all | crack | **tip** | edge | screw | ρ crack / edge / screw / tip |
|---|---|---|---|---|---|---|---|
| linear `ard_default` (rev2 §7) | 0.898 [0.891, 0.904] | 0.911 | 0.908 | 0.893 [0.887, 0.899] | 0.937 | 0.942 | 0.370 / 0.296 / 0.318 / 0.316 |
| GP `ard_gp_sw` (sandwich) | **0.900** [0.893, 0.907] | 0.916 | 0.912 | **0.902** [0.896, 0.907] | 0.937 | 0.941 | 0.362 / 0.286 / 0.312 / 0.309 |
| GP `ard_gp_dtc` | n/a (see note) | 0.844 | 0.838 | **0.827** | 0.878 | 0.888 | 0.282 / 0.136 / 0.163 / 0.230 |

Paired differences against the linear default (90 % cell bootstrap; atom-weighted coverage):

| arm | family | Δρ | Δcoverage |
|---|---|---|---|
| `ard_gp_sw` | crack | −0.0082 [−0.0095, −0.0071] | +0.0048 [+0.0041, +0.0055] |
| | tip | −0.0073 [−0.0107, −0.0035] | **+0.0086 [+0.0071, +0.0102]** |
| | edge | −0.0093 [−0.0134, −0.0054] | −0.0006 [−0.0012, +0.0000] |
| | screw | −0.0056 [−0.0071, −0.0040] | −0.0009 [−0.0019, +0.0000] |
| `ard_gp_dtc` | crack | −0.088 [−0.093, −0.083] | −0.069 [−0.071, −0.067] |
| | tip | −0.087 [−0.094, −0.080] | −0.066 [−0.070, −0.062] |
| | edge | −0.159 [−0.161, −0.158] | −0.059 [−0.064, −0.055] |
| | screw | −0.154 [−0.166, −0.145] | −0.054 [−0.055, −0.054] |

Other quantities:
- **Test RMSE:** both GP arms give E 2.96 meV/atom, F 0.0686 eV/Å, V 0.728. The linear default gives 3.03 / 0.0687 / 0.734.
- **Big-cell force RMSE:** `ard_gp_sw` 0.184 eV/Å (tip 0.204), against linear 0.186 (tip 0.208).
- **Fitted ARD scales (both arms):** a_GP = −0.04, so the evidence keeps the MAP's GP prior almost exactly. a_2body and a_3body are on their floor, as in rev2.
- **Fit time (B200):** `ard_gp_sw` 27 min (MAP 2.4 min, ARD stage 21 min). `ard_gp_dtc` 86 min (ARD stage 49 min).
- **`big_errors` (A100-80GB) per v3 file:** about 25 min for sandwich, about 50/30 min for `dtc`.
- **Cost:** about $25 of Modal credit on 2026-10-06, plus the `dtc` `big_errors` after midnight.

## Verdict

- **`ard_gp_sw` passes on crack-tip coverage but not on ranking.**
  - Tip coverage rises from 0.893 to 0.902 (+0.009, interval excludes 0). Whole-crack coverage rises by +0.005.
  - In-distribution coverage is 0.900 (target 0.90 ± 0.01), and dislocations are unchanged.
  - Spearman ρ is lower in every family by 0.006–0.009, and every interval excludes 0.
  - The ρ loss is small (2–3 % relative) but real. It does not "match or beat", so under the spec's rule
    the GP arm **stays experimental**.
- **`ard_gp_dtc` fails clearly.**
  - Under-coverage everywhere off distribution: tip 0.827, all 0.844.
  - Ranking drops by 0.09–0.16.
  - This is what D3 predicted. The derivative-DTC term is near-constant across environments, and the κ
    (posterior) shape it is added to does not grow away from the data the way the jackknife shape does. So
    the in-distribution calibration does not transfer.
  - Recommend removing `--ard-variance dtc`, or keeping it only as a documented negative result.
  - **Known defect (dtc only).** Its in-distribution pass (300 small cells, `GPCalculator(posterior=)` on CPU,
    mnf148) crashed after 50+ configurations with `JaxRuntimeError: Failed to materialize symbols`.
    - Where: in `dtc_shape` → `rows.batch_rows_parts` → `_rows_scan`.
    - Cause: `dtc_shape` builds the rows eagerly, not jitted, so every new cell shape compiles fresh XLA code
      until the CPU JIT's symbol space runs out. The final review flagged this eager path as a deferred minor.
    - The GPU `big_errors` runs (102 configurations) were not affected.
    - Its ID coverage was not obtained. It is not needed for the verdict: the big-cell failure already decides
      it.
- **No default changes.** `--uq ard` (linear) stays the recommended calibrated force UQ.
  `--uq ard-gp` (sandwich) is a working experimental option. It gives slightly better tip coverage and
  slightly lower ρ.

## Files

- Runs: `acegp-data/results/2026-10-06-gp-discrepancy-d4/bench365_ard_gp_{sw,dtc}/` (and `bench365_ard_default_pol/` for the baseline).
- Spawn log: `/storage/eng/essswb/projects/ace-jax/runs/gp_discrepancy/modal/spawned.txt`.
