# Per-atom uncertainty on defect combinations (Cantor CrMnFeCoNi)

Which per-atom uncertainty score tells a user **where not to trust** an ACE model when simple
defects that were in the training set combine in ways that were not, the situation in big-cell
fracture and plasticity runs? This directory holds the benchmark, the fits and the scoring. It is
research code: it records how the results below were produced, and it is not a supported API.

Cantor is a proxy system. Labels come from **MACE-MH-1** (torch, head `matpes_r2scan`: the head
that reproduces the cantor4k labels to 5e-5 eV/atom). Set `MACE_MODEL` to the model file and
`ACEGP_DATA` to the data root (default `~/acegp-data`).

## Benchmark (`gen/`)

All cells are equiatomic with random occupancy and centred on **MACE's own Cantor lattice
constant**, a0 = 3.6502 Å.
- MACE's constants (`gen/big2.py`, `cantor_properties`) are C11/C12/C44 = 128/101/94 GPa and
  γ(111) = 2.2 J/m² unrelaxed.
- The original cantor4k set used a0 = 3.59 Å, which put its median pressure at **+7.3 GPa**.
- The re-centred set sits at +0.6 GPa.

| set | contents | split |
|---|---|---|
| `make_bulk.py` | 4000 bulk cells, the cantor4k recipe at a0 = 3.6502 (±3 % strain, 0.02 shear, 0.02–0.10 Å rattle) | 3200 train / 800 test |
| `make_defects.py`, train families | vacancy, (100) and (111) slabs, intrinsic stacking fault (pure Ni check: 80 mJ/m² unrelaxed) | 120 train / 30 test each |
| `make_defects.py`, OOD families | NN divacancy, vacancy in the surface layer, vacancy at the stacking fault, slab under 4–8 % tension | 100 each, held out |
| `big2.py` + `modal/modal_big2.py` | (111)[1-10] crack cylinders (R = 60 Å, 3236 atoms) at K/K_G = 1.1/1.2/1.3 with K_G from MACE's C_ij and γ; edge (R = 50 Å) and screw (R = 60 Å) a/2⟨110⟩ dislocations built dissociated; relaxed with the boundary fixed, plus 0.05 and 0.10 Å rattles | 30 configs, held out |
| `big3.py` + `modal/modal_big3.py` (v3, use this) | the same cells made thick along the line (period > 2 r_cut = 12.5 Å: screw 5 × 2.58 Å, edge 3 × 4.47 Å, crack 3 × 5.16 Å) with species drawn after the repeat, so the chemistry is random along the line; R = 29–34 Å for ~4k atoms | 30 configs, held out |

All six cracks stay open and lattice-trapped, with the tip within ±4 Å of the seed. MACE-MH-1 in
float64 fails on single cells above ~7k atoms (A100 out of memory, B200 illegal memory access), so
the cylinders were sized to 3–4k atoms.

**v2 is chemically ordered along the line; use v3.** The v2 cells are one period thick along the
line (2.58 / 4.47 / 5.16 Å) under the model's r_cut = 6.25 Å. So every atom sees its own periodic
images (in the screw cells as nearest neighbours) and sits in a single-species column. No
random-alloy training config looks like that. σ correctly flags those environments as novel, but
the errors barely grow, so v2 over-covers: screw rms-z 0.61, edge 0.79. `big3.py` fixes it.

## Fits (`modal/`)

`fit_bench.py` drives the `ace_jax.fit.pipeline` API on B200s (`modal_bench365.py`), using the
production basis `cantor_embed_d16_deg10` (L = 15 035).
- **POPS** (linear): 82 min.
- **GP** (PCA-128, host-cached LML, 4 L-BFGS starts): 5.2 h.

Test errors are E 3.1 meV/atom and F 0.069 eV/Å for both.

Big cells are not predicted through the pipeline. One >2.8k-atom cell's dense edge Jacobian
exceeds 2³¹ elements, which breaks the int32-indexed GEMM autotuning. Their per-atom errors come
from the saved linear model via `ACECalculator` (`big_errors`).

## Scoring (`scoring/`)

- `descriptors.py`: per-site descriptors for train, test, OOD and big sets.
- `score_atoms.py`: per-atom scores against |ΔF| per atom.
  - **Hyperparameters are chosen only on the in-distribution test split.**
  - Every OOD family and the big cells are held out.
- **Metrics:**
  - Spearman ρ(score, |ΔF|);
  - AUROC of OOD atoms against test atoms;
  - recall of the worst 5 % of atoms within the top 10 % of scores;
  - crack/dislocation core (< 10 Å) against far interior (> 25 Å).
- `bayes_bodyorder.py`:
  - Bayesian linear regression posteriors under four priors: the fitted Γ, isotropic, and ARD per
    body order (on Γ, or isotropic), each fitted by evidence;
  - nested 2/3/4-body truncations and Bayesian model averaging over them;
  - per-atom posterior σ of forces and of site energies.

## Results (2026-09-28)

AUROC is OOD atoms against test atoms; ρ is Spearman ρ(score, |ΔF|).

| score | ρ test | AUROC: divac / vac_surf / vac_sf / strained surf | ρ: divac / vac_surf / vac_sf / strained surf | ρ: crack / edge / screw |
|---|---|---|---|---|
| POPS σ_F (pipeline) | 0.32 | 0.68 / 0.53 / 0.70 / **0.27** | 0.26 / 0.02 / 0.22 / 0.08 | – |
| GP σ_F (pipeline) | 0.21 | 0.58 / 0.58 / 0.59 / **0.31** | 0.16 / 0.05 / 0.19 / 0.12 | – |
| **BLR posterior σ_F, ARD per body order** | 0.28 | **0.75 / 0.90 / 0.85 / 0.88** | 0.19 / 0.27 / 0.21 / 0.20 | (needs design rows) |
| descriptor leverage (ridge 1e-8) | 0.26 | 0.69 / 0.84 / 0.81 / 0.87 | 0.22 / 0.32 / 0.24 / 0.23 | 0.32 / 0.20 / 0.20 |
| kNN, whitened centred pair channels | 0.29 | 0.68 / 0.67 / 0.77 / 0.88 | 0.24 / 0.22 / 0.32 / 0.07 | **0.39 / 0.31 / 0.30** |
| body-order difference \|F₄ − F₃\| | 0.21 | 0.64 / 0.80 / 0.65 / 0.78 | 0.15 / 0.25 / 0.20 / 0.16 | – |
| T²+Q (PCA 128), full descriptor | 0.18 | 0.68 / 0.83 / 0.75 / 0.67 | 0.10 / 0.21 / 0.10 / 0.15 | 0.10 / 0.04 / 0.06 |
| DADApy k*NN density (whitened PCA 32, ID ≈ 21) | 0.08 | 0.57 / 0.55 / 0.69 / 0.35 | ≈ 0 | ≈ 0 |
| site-energy posterior σ (any prior) | −0.11 | 0.47 / 0.30 / 0.47 / 0.12 | < 0 | < 0 |

### Findings

1. **The pipeline's σ fails where fracture needs it.** POPS's hypercube σ and the GP's σ are
   *more* confident on strained surfaces than on the test set (AUROC 0.27 and 0.31). POPS also has
   the best in-distribution ρ (0.32), so choosing an estimator in-distribution picks the wrong one.
2. **The Bayesian posterior predictive σ of each atom's force matches or beats descriptor
   leverage** on every held-out combination family.
   - This is the Bayesian counterpart of the D-optimality extrapolation grade: γ = max|c_j| is an
     L∞ bound on the square root of a flat-prior BLR variance against the active set.
   - ARD per body order, fitted by evidence (+4865 nats over the fitted Γ prior), leaves the
     2-body terms nearly unregularised and shrinks the 4-body terms most.
3. **Per-atom UQ must be defined on observables.** Site energies are not identifiable from energy
   and force data. Their posterior σ tracks that gauge freedom and is anti-correlated with error.
   Leverage-style scores work only because they treat site energies as observed.
4. **Evidence-weighted averaging over body-order truncations collapses** onto the full model (a
   64 000-nat gap to 3-body), so its between-model term is zero. The raw body-order difference is
   weaker than the posterior σ.
5. **Sampling density on the full-descriptor manifold does not track error**, even though that
   manifold is low-dimensional (2NN intrinsic dimension 13–21). A density in the whitened pair
   (2-body) subspace does, and it is the best score so far on cracks and dislocations.
6. **The epistemic ARD σ has a constant scale.** It is ~7× too small, but its rms-z is 6.2–7.0 in
   every family, in distribution and held out. So one temperature fitted in-distribution (a
   tempered or generalised posterior) should transfer. POPS's σ is calibrated in-distribution
   (rms-z 0.6–0.7) but drifts OOD (strained surfaces 1.09).

### Calibration (`scoring/calibrate.py`)

Each atom's force error is modelled as N(0, (s²/3) I₃). Scale parameters are fitted by NLL on half
of the in-distribution test configs only; everything else is evaluation.

| model | rms-z: test bulk / test defect / divac / vac_surf / vac_sf / strained surf | 90 % coverage: same order |
|---|---|---|
| POPS as fitted | 0.61 / 0.70 / 0.67 / 0.97 / 0.68 / 1.09 | 0.99 / 0.98 / 0.99 / 0.89 / 0.98 / 0.85 |
| POPS rescaled (×0.63) | 0.97 / 1.11 / 1.06 / 1.53 / 1.07 / 1.72 | 0.90 / 0.84 / 0.86 / 0.66 / 0.85 / 0.54 |
| **tempered ARD posterior, s = 6.85 σ_ARD** | **1.01 / 0.94 / 1.01 / 1.01 / 0.91 / 1.02** | **0.88 / 0.92 / 0.88 / 0.88 / 0.92 / 0.88** |
| tempered fitted-Γ posterior, s = 6.69 σ_BLR | 1.01 / 0.94 / 1.01 / 1.03 / 0.91 / 1.05 | 0.88 / 0.92 / 0.88 / 0.87 / 0.92 / 0.87 |
| POPS + tempered ARD | 0.99 / 1.03 / 1.05 / 1.25 / 1.01 / 1.31 | 0.89 / 0.88 / 0.87 / 0.77 / 0.88 / 0.72 |

- **A single temperature fitted in-distribution calibrates the Bayesian posterior σ_F on every
  held-out defect combination**, including strained surfaces: rms-z 0.91–1.02 and 90 % coverage
  0.87–0.92.
- This is a tempered, or generalised, posterior: the likelihood's weight is scaled by 1/κ², and κ
  absorbs the model misspecification.
- POPS cannot be rescaled into calibration, because its σ does not grow off-distribution.
- Adding the POPS term to the tempered ARD posterior degrades OOD calibration.

### Joint type-II ML vs ARD after MAP (`scoring/bayes_joint.py`)

The joint maximisation over (σ_E, σ_F, σ_V, a₂, a₃, a₄) was run on cached statistics.
- ARD-after-MAP reproduces +4865.4 nats over Γ-only. Its value was checked against an independent
  NumPy float64 evaluation to 0.013 nats.
- The joint fit adds ≈ +2.4 nats and moves log σ_F by 0.0023 (0.2 %).
- The hyperparameter Laplace std is 0.001–0.11.

So maximisation suffices here, and the sequential and joint fits are equivalent at this data size.

Numerics that matter:
- evaluate in the prior-scaled system (cond 10¹⁷ → 10¹³);
- gradient-scale the objective for L-BFGS-B;
- floor the 2-body prior precision (cond ≈ 4·10¹⁷ at the ARD optimum; evaluation noise of ±0.3
  nats).

Next: node-chunked design rows (force σ on the big cells), then the tempered ARD posterior σ_F as a
per-atom output of ace-jax fits and calculators.

## Acceptance of the ace-jax implementation (`--uq ard`, PR #18; 2026-09-28/29)

Fits: `modal/fit_bench.py` arms `ard`, `ard_<tag>` and `ard_c<k>` (the last sets `ard_cond_max = 10**k`).
These arms, and `ard_ABblk` below, are pinned to `ard_force_shape="iso"`: the pre-2026-10-03 results are iso, while the library default is now aniso.
Every ARD arm of `fit_bench.py` (and `modal/sandwich_spike.py`) is also pinned to `e0="prefit"`: all ARD runs before 2026-10-03, including the revision-2 acceptance tables, used the pre-fit E0, while the library now fits E0 jointly under `e0="lsq"` (as BLR).
Big cells: `modal_bench365.py::big_errors`. Scoring:
- `scoring/eval_ard.py`: rms-z, cov90, NLL, ρ and AUROC per family, with 95 % confidence intervals from
  a block bootstrap over configurations;
- `scoring/rank_local.py`: ranking within a cell, against the ceiling a perfect σ could reach;
- `scoring/sandwich_eval.py`: scoring for the sandwich spike, `modal/sandwich_spike.py`.

**The κ-tempered posterior**, with κ refitted for the full posterior (runs `ard_v2`, `ard_c15`, `ard_c16`):
- Calibrated with no test tuning. The test-refit factor is 0.984, so the pipeline's own κ was within
  2 % of the test set's choice.
- rms-z 0.90–1.01 on every held-out family.
- `ard_cond_max` from 1e14 to 1e16 changes neither calibration nor ranking, although the evidence
  rises by 835 nats. Ranking within cells is weak: ρ 0.15–0.26.

**Why ranking needs more than a posterior.** The posterior σ is epistemic, but the local errors in the
big cells are misspecification. They track novelty in the 2-body environment: whitened pair-kNN
gives ρ 0.39/0.31/0.30 against a ceiling of 0.41–0.49.

**The configuration-clustered sandwich** (the default, `ard_sw2`; λ fitted on the train hold-out,
leaving each atom's own cluster out; test-refit factor 0.968):

| | rms-z | cov90 | ρ, sandwich / κ | AUROC |
|---|---|---|---|---|
| test + held-out families | 0.86–0.98 | 0.90–0.95 | 0.30–0.37 / 0.19–0.26 | 0.74–0.91 |
| crack (core ≤ 10 Å) | 0.90 (0.97) | 0.93 | 0.35 / 0.24 | 0.91 |
| edge | 0.79 | 0.96 | 0.26 / 0.15 | 0.89 |
| screw | 0.61 | 0.99 | 0.26 / 0.18 | 0.99 |

- In the big cells, the top 10 % of atoms by σ hold 57–79 % of the atoms in the top 1 % of errors.
- A likelihood fit of a·ARD + b·sandwich puts a = 0.
- Leaving the own cluster out changes λ by +2.7 % at 3,680 configurations, against ×1.9 on a
  30-configuration fixture.
- *(Withdrawn.)* "Screw dislocations are over-covered by about 1.6×" was a v2 artefact (see
  Benchmark). The table above uses the v2 big cells.

**The same posterior on the v3 big cells** (`ard_sw2`, no refit; `scoring/eval_big.py`; the CIs
come from a block bootstrap over cells, with only 2 cells per dislocation family):

| | rms-z, v2 → v3 [95% CI] | cov90 | core ≤ 10 Å | ρ, v2 → v3 |
|---|---|---|---|---|
| crack (6 cells) | 0.90 → 1.01 [1.00, 1.02] | 0.89 | 1.08 [1.05, 1.10] | 0.35 → 0.41 |
| edge (2 cells) | 0.79 → 0.91 [0.91, 0.92] | 0.93 | 0.96 | 0.26 → 0.30 |
| screw (2 cells) | 0.61 → 0.90 [0.88, 0.91] | 0.93 | 0.98 | 0.26 → 0.32 |

- Dislocations are calibrated and slightly conservative.
- Crack tips are slightly overconfident: rms-z 1.08 and cov90 0.85 within 10 Å of the tip, across
  all 6 cells.

## Validation of the jackknife shape and conformal scales (schema 3; run 2026-10-01..03)

`--uq ard` now serves `forces_std` (lam_g x an exact centred jackknife shape), `forces_cov`, `forces_q`
(conformal radius), `forces_group` and `forces_support`. `aj calibrate` re-scales the per-group scales on a
labelled set. This programme measures them on the v3 big cells and ablates the ingredients. The acceptance
targets are in `docs/dev/specs/2026-09-30-conformal-force-sigma-design.md`.

**Results: [`results/2026-10-03_rev2_acceptance.md`](results/2026-10-03_rev2_acceptance.md).** The default
(aniso shape + transfer exponent) passes every target:
- in distribution: 0.898;
- crack: 0.903;
- tip: 0.885;
- edge / screw: 0.937 / 0.941.

`aj calibrate` on labelled crack cells gives the best tip coverage (0.897–0.903), but the calibrated posterior
under-covers in distribution (0.87), so serve it only on cells like its calibration set. Block size ℓ has no
effect. The tables, run names and commands follow.

**Code**
- `modal/ard_arms.py`: `ARD_ARMS`, the arm name -> `FitConfig` override table (`fit_bench.py` takes the name as its arm).

  | arm | what it ablates |
  |---|---|
  | `ard_legacy` | #18 uncentred sandwich, own-cluster-out scores, 2 groups (see note) |
  | `ard_A` | #18 shape with scores from the fit split, distortion groups |
  | `ard_AB` | centred PRESS shape, whole-configuration clusters (ell = inf) |
  | `ard_ABblk` | PRESS, ell = 3 r_cut, **iso** (explicitly `ard_force_shape="iso"`, since the library default became aniso on 2026-10-03; keeps the recorded ablation reproducible) |
  | `ard_aniso` | Mahalanobis region (`forces_q_mahal`); now equal to the library defaults |
  | `ard_ell{2,4,6}` | ell sweep (3 = `ard_ABblk`, inf = `ard_AB`) |
  | `ard_f{1,3}` | `ard_val_frac` 0.1 / 0.3 (0.2 = `ard_ABblk`) |
  Note: `ard_legacy` reproduces the 30-Sep *shape* and scores, but it still serves the revision-2
  scales: per-group (G = 2, [z = z*]) configuration-weighted `lam_rms` / `q`, not the 30-Sep single
  atom-weighted scalar lambda. The scalar is still computed and stored as `ard.json` `"lam"`;
  `validate_shape.py` prints it as the `lam (scalar)` column beside `lambda_rms` for that comparison.
- `fit_bench.py --train-extra A.xyz,B.xyz` appends those configurations to `train.xyz` (written to
  `<out>/train_plus_extra.xyz`); the Modal `launch` takes `--train-extra` and `--tag` (output dir suffix).
- `modal_bench365.py::big_errors` now saves `sd`, `forces_q`, `forces_group`, `forces_cov` (schema-3 posteriors)
  and the error vector `dF` into `big<tag>_err.npz`. `modal/served_arrays.py` holds the concatenation helper.
- `scoring/validate_shape.py --runs DIR... --out report.md`: leverage summary, lambda_rms, rho per family, coverage
  per conformal group / family / tip band (cell- and atom-weighted, 90 % bootstrap over whole cells, B = 1000),
  region volume (raw and at nominal coverage), f-sweep slope, ell-sweep table. Old runs (no `forces_q`) report n/a.
- Tests: `uv run pytest bench/defect_uq/tests -q` (outside the repo's `testpaths`, so run it explicitly).

**Commands** (from `bench/defect_uq/modal`, with `ACEGP_DATA`, `ACEJAX_SRC` set as for the existing runs;
`/out` is the `acegp-prod-out` volume; the crack files must first be on the volume as `/out/defects/big3_*`,
check with `modal volume ls acegp-prod-out defects`)

```bash
modal deploy modal_bench365.py

# 1. ablation arms on bench365 (5 fits)
modal run modal_bench365.py::launch --arms ard_legacy,ard_A,ard_AB,ard_ABblk,ard_aniso

# 2. ell sweep: ell in {2,3,4,6,inf} = ard_ell2, ard_ABblk, ard_ell4, ard_ell6, ard_AB. First without tip data
#    (calibration only: the runs of step 1 + ard_ell2/4/6), then with crack realisations NOT held out in T.
#    A fold holds out one crack FILE (a realisation pair: r2-3, r4-5, r6-7 or r8-9) and trains on the other three
#    via --train-extra; tag _fold_<pair>. Start with 3 folds (r2-3, r4-5, r8-9); extend to all 4 if the
#    spread between folds is > 1 point.
for ell in ard_ell2 ard_ABblk ard_ell4 ard_ell6 ard_AB; do
  modal run modal_bench365.py::launch --arms $ell --tag _fold_r2-3 \
      --train-extra /out/defects/big3_cracks_r4-5.xyz,/out/defects/big3_cracks_r6-7.xyz,/out/defects/big3_cracks_r8-9.xyz
done
# (repeat with --tag _fold_r4-5 and the extras r2-3,r6-7,r8-9, and so on for each fold)
# --train-extra checks every frame for mace_energy and mace_force (error naming file, frame, key), allows a
# missing mace_virial (no virial row, logged), and logs the counts of frames, virial-less frames and fixed
# boundary atoms. Fixed atoms stay in the fit with their MACE force labels (the loader has one force weight per
# configuration, no per-atom mask); see modal/train_extra.py.

# 3. f sweep (0.2 = ard_ABblk)
modal run modal_bench365.py::launch --arms ard_f1,ard_f3

# per-atom errors + served arrays on the v3 cells (one call per file; the tag names the file).
# Non-fold runs: all five files.
for run in bench365_ard_legacy bench365_ard_A bench365_ard_AB bench365_ard_ABblk bench365_ard_aniso \
           bench365_ard_ell2 bench365_ard_ell4 bench365_ard_ell6 bench365_ard_f1 bench365_ard_f3; do
  modal run modal_bench365.py::launch_big --run $run --xyz /out/defects/big3_mh1.xyz --tag 3
  for p in 2-3 4-5 6-7 8-9; do
    modal run modal_bench365.py::launch_big --run $run --xyz /out/defects/big3_cracks_r$p.xyz --tag 3x_r$p
  done
done
# Fold runs: only the held-out pair's file and the main file (r0-1 cracks + edge/screw, never in T); the
# training cracks must not be scored. Example for fold r2-3:
for ell in ard_ell2 ard_ABblk ard_ell4 ard_ell6 ard_AB; do
  run=bench365_${ell}_fold_r2-3
  modal run modal_bench365.py::launch_big --run $run --xyz /out/defects/big3_mh1.xyz --tag 3
  modal run modal_bench365.py::launch_big --run $run --xyz /out/defects/big3_cracks_r2-3.xyz --tag 3x_r2-3
done

# 4. leave-one-realisation-out aj calibrate on the v3 crack cells, default arm (local, CPU is enough
#    for the labels; GPU for the descriptor pass): for each held-out realisation r
#      aj calibrate --model $RUN/model.npz --posterior $RUN/posterior.npz --data <v3 cracks of the others, +ID test> \
#          --force-key mace_force --energy-key mace_energy --replace --out $RUN/posterior_cal_r$r.npz
#    then big_errors against posterior_cal_r$r.npz on realisation r and score with validate_shape.py.

# report (after fetching the runs: modal volume get acegp-prod-out <run> results/2026-10-xx/).
# Non-fold arms, and the ell sweep with tip data (fold runs restricted to the files they did not train on
# with DIR:PATTERN; patterns are fnmatch on the err-file basenames, comma-separated):
R=results/2026-10-xx
uv run python ../scoring/validate_shape.py --out validate_shape.md --runs \
    $R/bench365_ard_legacy $R/bench365_ard_A $R/bench365_ard_AB $R/bench365_ard_ABblk $R/bench365_ard_aniso
uv run python ../scoring/validate_shape.py --out validate_ell_tip_r2-3.md --runs \
    $R/bench365_ard_ell2_fold_r2-3:big3_err.npz,big3x_r2-3_err.npz $R/bench365_ard_ABblk_fold_r2-3:big3_err.npz,big3x_r2-3_err.npz \
    $R/bench365_ard_ell4_fold_r2-3:big3_err.npz,big3x_r2-3_err.npz $R/bench365_ard_ell6_fold_r2-3:big3_err.npz,big3x_r2-3_err.npz \
    $R/bench365_ard_AB_fold_r2-3:big3_err.npz,big3x_r2-3_err.npz
```

Step 4 driver: `modal/modal_calibrate.py` (own app `acegp-calibrate`; `modal_bench365.py` untouched). Per
held-out pair it runs `aj calibrate` (per-group by default; `--mode append|replace`) on the other three
`big3_cracks_r*.xyz` files and writes `/out/<run>_cal[_<mode>]_r<pair>/{model.npz,posterior.npz,calibrate.log,
big3x_r<pair>_err.npz,big3_err.npz}`:

```
cd bench/defect_uq/modal && modal deploy modal_calibrate.py
modal run modal_calibrate.py::launch --run bench365_ard_ABblk      # 4 calls: holds 2-3,4-5,6-7,8-9
for p in 2-3 4-5 6-7 8-9; do modal volume get acegp-prod-out bench365_ard_ABblk_cal_r$p $R/; done
uv run python ../scoring/validate_shape.py --out validate_loro.md --runs $R/bench365_ard_ABblk_cal_r2-3 ...
```

**Estimated cost.** 25 fits: 5 ablation + 3 ell (no tip data) + 15 ell with tip data (5 ell x 3 folds) +
2 f. A bench365 ARD fit was 19-31 min of solver time (`ard.json` `seconds`, 2026-09-28) plus loading and
the new PRESS/support stages, taken as ~0.75 B200-h: about 19 B200-h. `big_errors` on the 5 v3 files per run
(A100-80GB, ~0.2 h per run, 25 runs): ~5 A100-h. Calibration is minutes per fold.

**Acceptance** (record results here and in `results/2026-10-xx/ACCEPTANCE.md`; a miss is reported as a miss
with the per-group table): in-distribution held-out coverage 0.90 +- 0.01; crack whole cell >= 0.89, tip >= 0.88,
edge/screw >= 0.90 (calibrate, leave-one-realisation-out).

## Comparison with CALM (arXiv:2609.40060; 2026-10-05)

Bond scans, normalised support features, novelty groups, a precision audit and the shape as an r-output
linear ACE: [`results/2026-10-05_calm_comparison.md`](results/2026-10-05_calm_comparison.md) (plan:
`docs/dev/force-uq-vs-calm.md`). Scripts: `scoring/bond_scan.py`, `scoring/support_rebuild.py`,
`scoring/novelty_groups.py`, `scoring/precision_audit.py`, `scoring/shape_eval.py` (+ `modal/modal_shape_eval.py`),
`scoring/calm_metrics.py`.
