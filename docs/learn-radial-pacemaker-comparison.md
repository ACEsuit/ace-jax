# Learned radials vs pacemaker: RMSE, weighting and physical properties

This note compares our learned radials with pacemaker (python-ace + TensorPotential), which learns radials jointly with the readout. It records what does and does not move the accuracy. Physical properties decide the comparison, not RMSE. Setup and earlier results are in `docs/learn-radial-results.md`. Raw numbers are under `docs/figures/learn-radial/comparison/`.

**Summary.** At equal basis size, our VarPro-learned radials give:

- the best energies of any model tested;
- forces that sit on pacemaker's energy–force trade-off curve;
- much more faithful elastic constants than every pacemaker variant (2.3–2.4% against 5.9–7.7% mean deviation from the MACE teacher).

Every linear model, ours included, misses SiGe vacancy formation energies by 0.2–1.8 eV. The pacemaker model with the sqrt(ρ) (Finnis–Sinclair) embedding gets them within 0.14 eV. That is the case for adding an optimised density term to our model, next.

## Setup

- **Data and splits.** The seed-0 splits used throughout: SiGe 200/100 train/val, Cantor (CrMnFeCoNi) 150/100. Labels are MACE-MH-1, head `matpes_r2scan`. That head reproduces the stored labels to 0.00 meV/atom (SiGe) and 0.04 meV/atom (Cantor).
- **Our model.** ACEpotentials basis at a fixed size: SiGe 1428 functions, 714 per element; Cantor 1950, 390 per element.
  - Analytic radials at n_q = 12, learned by VarPro L-BFGS with θ re-profiled.
  - Final readout from the linear posterior at the MAP θ.
  - `bench/learn_radial/run.py`, on Modal A100s via `modal_run.py`.
- **pacemaker.** The same number of functions per element, SBessel radials with `nradbase` 12, learnable `crad`, joint BFGS. Inputs are in `bench/learn_radial/pace/`.
  - Energies are corrected by per-element references fitted on the training split. That is used for pacemaker only, and doesn't change per-atom RMSEs.
  - The linear variant uses `ndensity 1`, `fs_parameters [1,1]`. The sqrt(ρ) variant uses `ndensity 2`, `fs_parameters [1,1,1,0.5]`.
  - Environment: TF 2.16.2, TensorPotential @1e44b25, python-ace @66c35ea. Runs were on lestrade and on Modal (`pace_modal.py`).
  - Evaluated with `pace/eval_pace.py`, which uses pyace on the same validation configs with the same definitions as `rmse.py`.

## RMSE (validation, E meV/atom / F meV/Å)

| | SiGe | Cantor |
|---|---|---|
| ours, initial radials | 2.08 / 53.7 | 10.29 / 152.0 |
| ours, learned, 40 steps | 0.54 / 40.2 | 6.94 / 129.6 |
| ours, learned, 400 steps | 0.50 / 34.6 | 5.68 / 108.8 |
| ours, learned, 800 steps | 0.50 / 33.4 | 5.32 / 104.8 |
| ours, learned, 1600 steps | **0.49** / 32.9 | — |
| pacemaker linear, κ=0.02 (1000 it) | 0.85 / 32.2 | — |
| pacemaker linear, κ=0.1 (1000 it) | 0.97 / 31.1 | — |
| pacemaker linear, κ=0.3 (2000 it) | 1.07 / 28.7 | 7.32 / **88.7** |
| pacemaker sqrt(ρ), κ=0.3 (2000 it) | 0.94 / **26.5** | — |

- **More steps help, with diminishing returns.** Forces keep improving with the step count: SiGe 40 → 400 → 1600 steps gives 40.2 → 34.6 → 32.9 meV/Å. Forty steps captures about 65% of the attainable force gain and all of the energy gain.
- **Pacemaker's force advantage is mostly its weighting.** At κ = 0.02, its energy-weighted end, its forces match ours at 1600 steps (32.2 against 32.9) while its energies are about 1.7× worse.
- **sqrt(ρ) adds about 8% on forces and 12% on energies** over linear pacemaker.

## What does not move our force accuracy

Each of these was measured on SiGe, and on Cantor where noted. Data is in `comparison/weighting/`.

- **Energy/force weighting of the final fit, at fixed radials.** Sweeping σ_E by ×0.01…×1000 moves forces by at most 3 meV/Å. With about 38k force components against 200 energies, forces already set the readout (`tradeoff.py`; SiGe and Cantor).
- **Force-heavy weighting during radial learning** (MACE-style stage 1). σ_E ×10 or ×100 in the radial objective only, with MAP weights for the final fit: at most 2% on forces at 40 steps, nothing at 200, and worse energies on Cantor (`--learn-sigma-e-mult`; SiGe and Cantor).
- **The shape of the smoothness prior.** Replacing Γ with Γ^α for α = 0…1, with the σ's re-MAP'd: under 2% on forces (`tradeoff.py --gamma-powers`).

  Side finding: a longer, warm-started θ-MAP improves the *initial-radial* baseline from 2.08/53.7 to 1.45/50.8. The default 300-step MAP isn't fully converged, but the learned-radial numbers barely change.

## Physical properties (deviation from the MACE teacher)

Computed with `bench/learn_radial/qoi.py` and tabulated with `qoi_compare.py`. Every model goes through identical ASE protocols:

- a full relaxation;
- a 7-point relaxed-ion EOS fitted with Birch–Murnaghan;
- ±0.5% strain–stress relaxed-ion cubic elastic constants;
- relaxed vacancy formation energy, E(N−1) − (N−1)/N · E(N).

**SiGe.** Structures are diamond Si, diamond Ge and ordered SiGe. The mean abs. deviation covers a₀, B, C11, C12 and C44 for all three structures. Vacancy columns are E_vac − E_vac(MACE) in eV. The MACE values are Si 3.65 eV and Ge 2.55 eV.

| Model | mean abs. deviation (15 quantities) | Si vacancy (eV) | Ge vacancy (eV) |
|---|---|---|---|
| ours, initial radials | 12.1% | −1.52 | −0.93 |
| **ours, learned, 40 steps** | **2.4%** | −1.29 | −0.69 |
| **ours, learned, 1600 steps** | **2.3%** | −1.79 | −1.35 |
| pacemaker linear κ=0.02 | 5.9% | −0.74 | −0.52 |
| pacemaker linear κ=0.1 | 6.7% | −0.74 | −0.44 |
| pacemaker linear κ=0.3 | 7.7% | −1.08 | −0.19 |
| pacemaker sqrt(ρ) κ=0.3 | 5.9% | **+0.03** | **+0.14** |

Every pacemaker variant overestimates C11 by 9–19% and underestimates C12, so it overstates the shear stiffness C′. Our learned radials keep every elastic constant within ±7%.

**Cantor.** Three random 108-atom fcc draws. B and C44 are shown per draw, in % deviation. The vacancy column is the MAE over 2 sites per species. C′ ≈ 10 GPa sits near the tetragonal instability, so percentages on C′, C11 and C12 are not meaningful. MACE's own draw 2 relaxed to a distorted cell.

| Model | B per draw | C44 per draw | vacancy MAE (eV) |
|---|---|---|---|
| ours, initial radials | +27 / +16 / +11 | −20 / +16 / −1 | 0.28 |
| ours, learned, 40 steps | +13 / +14 / +32 | −20 / −16 / +8 | 0.58 |
| ours, learned, 800 steps | **+6 / +2 / −8** | **−3 / −2 / −20** | 0.41 |
| pacemaker linear κ=0.3 | −33 / −35 / −43 | −10 / −12 / +6 | 1.14 |

## Conclusions

1. **Learned radials are worth having, and a short budget is enough for most of the benefit.** Forty VarPro steps cut the SiGe elastic-property error from 12% to 2.4%. They keep the linear model's advantages: a convex final fit, Bayesian UQ, and fast refits.
2. **Lower force RMSE doesn't mean better physics.** Pacemaker has the best forces, yet it is consistently worse on elastic constants, and its Cantor bulk modulus is off by about 35%. Model choices should be judged on quantities of interest.
3. **Vacancies need a nonlinear embedding.** No linear model reproduces the vacancy formation energies, while the sqrt(ρ) embedding does. Next step: joint VarPro over the radials W and a density η spanning the full ACE basis, as pacemaker's second density does, with the sqrt(ρ) embedding. The existing pair-only density term had no effect on SiGe. The goal is to keep our elastic accuracy and reach sqrt(ρ)-level vacancy energies.
4. **Loose ends:**
   - converge the θ-MAP by default, or add a convergence check;
   - vacancy and defect configurations are absent from training, so these results are extrapolation for every model;
   - a single teacher, and three Cantor draws.
