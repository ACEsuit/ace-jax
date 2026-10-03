# Acceptance: `--uq ard` revision 2 (PRESS jackknife shape + Mondrian conformal scales), bench365 / v3

**Status:** 2026-10-03. Branch feat/conformal-uq (PR #34), head 18e11ae.
- Fits ran on Modal B200s; `big_errors` and calibrate ran on A100-80GB.
- Data: bench365 train/test (3680/920 configs), plus the v3 big cells:
  - big3_mh1: r0-1 cracks, edge and screw dislocations;
  - big3_cracks_r{2-3,4-5,6-7,8-9}: 18 configs each, 3276 atoms each.
- Scoring: `bench/defect_uq/scoring/validate_shape.py`; tables are in `validate_*.md` in this directory.
- Coverage is of the served `forces_q` region at nominal 0.90 (aniso: the Mahalanobis radius `forces_q_mahal`).
- 90 % CIs come from a bootstrap over whole cells (cell = (file, cfg // 3)).
- The tip is the atoms with r_core ≤ 10 Å; fixed boundary atoms are excluded.

## Targets (spec)

| target | value |
|---|---|
| in-distribution held-out coverage | 0.90 ± 0.01 |
| crack, whole cell | ≥ 0.89 |
| crack tip | ≥ 0.88 |
| edge / screw | ≥ 0.90 |

## 1. Transfer of the hold-out scale (the finding that shaped the defaults)

Scores from the hold-out posterior P_fit (N_fit = 2944 of N = 3680 configs) under-scale the served posterior P:
- in-distribution force rms-z was 1.087;
- big-cell coverage was 0.850, tip 0.808.

The f sweep gives λ ∝ N_fit^0.37, so the predicted bias is (3680/2944)^0.37 = 1.086, which matches.

The fix (`ard_transfer = exponent`, now the default): a nested second subset fit estimates β per fit, clipped to [0, ½], and the scores are scaled by (N/N_fit)^β.

| arm | β | factor | ID rms-z before → after |
|---|---|---|---|
| f = 0.2, iso | 0.383 | 1.089 | 1.087 → 0.998 |
| f = 0.2, aniso | 0.381 | 1.089 | 1.068 → 0.981 |
| f = 0.1 | 0.379 | 1.041 | 1.053 → 1.011 |
| f = 0.3 | 0.361 | 1.137 | 1.149 → 1.010 |

## 2. In-distribution coverage (bench365 test, 300-config random sample)

| posterior | cfg-weighted | atom-weighted | bulk | vac | sf | surf100 | surf111 |
|---|---|---|---|---|---|---|---|
| iso + transfer | **0.897** | 0.902 | 0.891 | 0.899 | 0.968 | 0.957 | 0.969 |
| **aniso + transfer (default)** | **0.898** | 0.903 | 0.892 | 0.901 | 0.970 | 0.954 | 0.965 |
| iso, calibrated on cracks (hold r2-3) | 0.870 | 0.877 | 0.862 | 0.880 | 0.956 | 0.942 | 0.974 |
| iso, calibrated on cracks (hold r8-9) | 0.872 | 0.878 | 0.864 | 0.881 | 0.956 | 0.942 | 0.974 |

- **PASS** for both uncalibrated posteriors.
- Calibrating on crack cells with the default per-group replace **under-covers in distribution**. The 54 crack configs populate and replace all 8 groups, including the bulk-like ones, so a calibrated posterior is specific to its regime.

## 3. Big cells, no target data in training or calibration

These are uncalibrated; they score all 34 v3 cells (`validate_shape_tx.md`).

| arm | all | crack (≥ 0.89) | tip (≥ 0.88) | edge (≥ 0.90) | screw (≥ 0.90) |
|---|---|---|---|---|---|
| legacy (#18 shape, own-cluster-out scores, G = 2) | 0.909 | 0.904 | 0.886 | 0.941 | 0.943 |
| iso, no transfer | 0.850 | 0.844 | 0.808 | 0.896 | 0.904 |
| iso + transfer | 0.895 | 0.889 | **0.862 ✗** | 0.933 | 0.937 |
| **aniso + transfer (default)** | **0.907** | **0.903 ✓** | **0.885 ✓** | **0.937 ✓** | **0.941 ✓** |
| f = 0.1 + transfer | 0.908 | 0.905 | 0.896 | 0.929 | 0.935 |
| f = 0.3 + transfer | 0.887 | 0.881 | 0.851 | 0.928 | 0.932 |

- The **default (aniso + transfer) passes every target.**
- The iso variant misses at the tip.
- A residual dependence on f remains on the out-of-distribution cracks (tip 0.896 at f = 0.1 against 0.851 at f = 0.3), although the in-distribution scale is f-independent.

Spearman ρ(σ, |e|) is 0.38 (crack), 0.29 (edge) and 0.31 (screw), the same across arms.

## 4. Crack cells in training (fold runs, iso + transfer)

Each fold trains on 3 crack realisation pairs and is scored on the held-out pair plus big3_mh1 (`fold_summary.txt`).

| tip coverage | ℓ = 2 r_cut | ℓ = 3 | ℓ = 4 | ℓ = 6 | ℓ = ∞ |
|---|---|---|---|---|---|
| fold r2-3 | 0.891 | 0.891 | 0.893 | 0.896 | 0.896 |
| fold r4-5 | 0.889 | 0.889 | 0.886 | 0.887 | 0.887 |
| fold r8-9 | 0.878 | 0.877 | 0.877 | 0.874 | 0.874 |

- Whole crack is 0.894–0.911, edge 0.928–0.937, screw 0.931–0.939.
- **Block size ℓ has no effect** (≤ 0.005, inside the CIs). The default stays at 3 r_cut.
- Training on crack cells lifts tip coverage from 0.86 to 0.874–0.896, but fold r8-9 misses 0.88.
- Fits with 3.3k-atom cells in training needed two library changes:
  - size-aware batching, `build_dataset(pack=auto)`;
  - node-chunked training statistics.

## 5. Calibration on target data (`aj calibrate`, leave one realisation out, per-group replace)

Each fold calibrates on the other 3 crack pairs and is scored on the held-out pair plus big3_mh1 (`validate_cal_tx.md`).

| hold-out | all | crack | tip | edge | screw |
|---|---|---|---|---|---|
| r2-3 | 0.915 | 0.909 | 0.898 | 0.928 | 0.933 |
| r4-5 | 0.919 | 0.914 | 0.901 | 0.930 | 0.935 |
| r6-7 | 0.920 | 0.916 | 0.897 | 0.930 | 0.935 |
| r8-9 | 0.920 | 0.916 | 0.903 | 0.930 | 0.935 |

- **PASS** on every big-cell target in every fold. This is the best tip coverage of any route, and it costs minutes against a refit of hours.
- The cost is the in-distribution under-coverage in §2 (0.87). Serve the calibrated posterior only on cells like U.

## 6. `aj calibrate` modes on the aniso default (and iso `--append`), leave one realisation out

Big-cell scores use the held-out pair plus big3_mh1. In-distribution scores use a 300-config bench365 test sample. Each cell gives the range over the 4 folds.

| posterior | ID cfg-weighted | ID bulk | tip | crack | edge / screw |
|---|---|---|---|---|---|
| aniso, uncalibrated (default) | 0.898 | 0.892 | 0.881–0.884 | 0.899–0.905 | 0.937 / 0.941 |
| aniso + calibrate, per-group replace | **0.867–0.870 ✗** | 0.858–0.862 | **0.899–0.904** | 0.910–0.917 | 0.927–0.930 / 0.932–0.935 |
| aniso + calibrate `--append` | **0.895** | 0.888–0.889 | 0.881–0.886 | 0.900–0.906 | 0.936 / 0.941 |
| iso + calibrate `--append` | 0.895 | 0.888–0.889 | 0.869–0.878 | 0.895–0.903 | 0.934–0.935 / 0.937–0.939 |

- **`--append` is safe in distribution (0.895) but does not help on the cracks here.** The 54 crack configs are diluted among hundreds of T_val configs per group, and every configuration counts once.
- **Per-group replace is the only calibration that lifts the tip, to 0.90.** It makes a regime-specific posterior: in-distribution coverage drops to 0.87.
- **Recommendation:** serve the default posterior generally. With labelled target-regime cells, build a separate per-group-replace posterior and use it only for that regime.

## 7. Rerun with the converged evidence fits and joint E0 (2026-10-03, head 80e24d8)

| run | ID cfg-weighted | crack | tip | edge / screw | ID force rms-z |
|---|---|---|---|---|---|
| aniso + transfer (recorded, L-BFGS only) | 0.898 | 0.903 | 0.885 | 0.937 / 0.941 | 0.981 |
| + projected-Newton polish | 0.898 | 0.903 | 0.885 | 0.937 / 0.941 | 0.981 |
| **+ polish + joint E0 (library defaults, `ard_default`)** | **0.898** | **0.908** | **0.893** | **0.937 / 0.942** | 0.980 |

- **Polish.** Every evidence fit converges, to within 10× its measured gradient roundoff, after 1 Newton step and 2 Hessians. That includes the fits where L-BFGS stopped ABNORMAL. The acceptance numbers are unchanged.
- **Joint E0.** Tip and crack coverage rise slightly. Test energy RMSE falls from 3.054 to 3.028.
- **Wall time.** About 95 min becomes about 61 min, from the per-call compile caching.

**Conditioning.**
- cond(S) at the evidence endpoints is 9.8e13 (P_fit), 9.6e13 (P_fit2) and 1.0e14 (full fit), against `ard_cond_max` = 1e14.
- a₂ (and a₃ in the full fit) sit on `a_floor`, so the floor binds.
- The leak of the h0-based floor is about 1 % on bench365. It can reach e⁶ only on tiny sets whose σ falls far.
- **Decision: leave `a_floor`/`ard_cond_max` as is.** The fit logs cond(S) and warns above `ard_cond_max`.

## Verdict

- **The default (aniso + transfer exponent) passes every target**, both in distribution and on the uncalibrated v3 big cells.
- **`aj calibrate` on labelled target-regime cells** (per-group replace) gives the highest target coverage under both shapes, but the calibrated posterior is regime-specific (ID 0.87). `--append` keeps ID coverage at 0.895 but does not lift the tip (§6).
- **iso + transfer** passes in distribution but misses at the crack tip (0.862) and is borderline on the whole crack (0.889).
- **Crack cells in training** improve the tip but don't reach 0.88 in every fold. Block size is not a lever.

## Engineering notes from the run

- The modal workspace caps concurrency at 10 GPUs.
- `modal volume get` intermittently corrupts ~1 GB downloads. Verify npz files (zip `testzip`), or compute on Modal against the volume.
- `aj calibrate` / `big_errors` on 3–4k-atom cells need A100-80GB. 40 GB OOMs in the XLA autotune of the rows pass.
