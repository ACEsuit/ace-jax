# Radials + sqrt(density) by VarPro: results

This note tests the joint radial and density learning of `docs/specs/2026-09-28-radial-density-varpro-design.md` against the success criteria in that spec. The data, splits, MACE teacher and QoI protocol are the same as in `docs/learn-radial-pacemaker-comparison.md`. Raw numbers are under `docs/figures/learn-radial/density/`, and `qoi_table.py` there regenerates the QoI tables.

## Summary

**Learned sqrt(ρ) features in a linear readout do not improve the vacancy energies.** The fitted readout of the density term stays close to zero, with d ≈ 10⁻³, whether or not vacancies are in the training data. The term then contributes about 1 meV per atom, and under 1 meV across a vacancy.

The reason is structural. With a full-span density ρ = η·X, sqrt(ρ₀+δ) ≈ const + δ/(2√ρ₀) + O(δ²), and the linear part δ already lies in the span of the linear ACE basis. The new column adds only the curvature, which the evidence-tuned ridge treats as a weak, redundant direction and shrinks. Pacemaker's sqrt(ρ) model gets vacancies right (+0.03/+0.14 eV) because its embedding has a fixed form with unit weights, so it is forced to use that curvature.

**Vacancy data is what moves the vacancy energy.** With 80 SiGe vacancy cells in training, the SiGe vacancy errors change as follows:

| SiGe vacancy error (eV) | Without vacancy data | With vacancy data |
|---|---|---|
| Si, 40 steps | about −1.5 | −0.4 to −0.5 |
| Si, 200 steps | about −1.5 | −0.66 to −0.82 |
| Ge | −0.6 to −1.1 | −0.11 to −0.38 |

For Cantor, 60 vacancy cells give a smaller change: the vacancy MAE goes from 0.41–0.48 to 0.34–0.47 eV. On both systems, more radial learning improves bulk accuracy at the vacancy's expense, because the objective has 150–200 bulk configurations against 60–80 defect ones.

**Success criteria from the spec:**

1. **SiGe vacancies within 0.3 eV of MACE: fails.** Ge meets it with vacancy data. Si does not: its best error is −0.43 eV, with a density at 40 steps.
2. **Elastic mean deviation ≤ 3%: holds.** It is 1.4–1.8% at 200 steps and 1.4–2.5% at 40 steps with vacancy data.
3. **RMSE no worse than radials-only: holds on SiGe.** Energies are within ±5% and forces up to 6% worse. On Cantor, the density costs 15% in energy and 5% in forces at 200 steps.
4. **Gate soundness: holds.**

**Optimiser.** The joint L-BFGS with the block preconditioner (approach A) fails on Cantor. The preconditioner scale r_η goes to 3×10⁻³, and the line search breaks down after 11 accepted steps. Alternating blocks (approach B, the spec's fallback) complete their full budget on both systems. Approach A works on SiGe.

## Setup

- **Runs.** `bench/learn_radial/modal_run.py` on A100-80GB, with n_q 12, lam_rough 0.1, `--reprofile-every 50` (the Modal default) and lam_eta 0.
- **Candidates.** Each run learns two candidates from the same start, split and step budget: radials-only, and radials + density. The held-out gate picks between them. Both candidates are evaluated below, whichever the gate picks.
- **RMSEs.** Measured on the saved model itself (`rmse_npz.py`, on the unchanged bulk validation split): energies in meV/atom, forces in meV/Å.
- **QoIs.** `qoi.py` against MACE-MH-1 (Si vacancy 3.65 eV, Ge 2.55 eV). SiGe QoIs were run on a Mac CPU and Cantor QoIs on lestrade.
- **Vacancy data, the second round.** It is added to the training split only (`run.py --extra-train`):
  - SiGe: `~/acegp-data/sige/sige_vac_mh1.xyz`, 80 cells of 63 atoms, a quarter pure Si and a quarter pure Ge, labelled with MACE-MH-1. The generator is `gen/make_sige_vac.py`.
  - Cantor: `~/acegp-data/cantor/cantor_vac_train_mh1.xyz`, the first 60 training vacancy cells of the defect benchmark.

  This round also uses `--tol 0`, so both candidates run their full step budget.
- **Run-to-run variation.** Radials-only candidates that should be identical differ by ≈1–2% across runs, from GPU nondeterminism. They are quoted as ranges.
- **Baseline.** Re-profiling θ every 50 steps makes these radials-only numbers differ from PR #14's (0.77 against 0.54 meV/atom at 40 steps). The comparison inside each run is like for like.

## SiGe

Vacancy columns are E_vac − E_vac(MACE) in eV. The elastic column is the mean |%| deviation over a₀, B, C11, C12 and C44 for Si, Ge and SiGe.

| Data | Budget | Model | E (meV/atom) | F (meV/Å) | Elastic \|%\| | Si vacancy (eV) | Ge vacancy (eV) |
|---|---|---|---|---|---|---|---|
| bulk | 40 | radials only | 0.774 | 44.54 | 4.0 | −1.52 | −0.84 |
| bulk | 40 | full, P=1 | 0.772 | 44.71 | 4.3 | −1.49 | −0.62 |
| bulk | 40 | full, P=2 | 0.801 | 45.07 | 3.9 | −1.51 | −0.62 |
| bulk | 40 | pair, P=1 | 0.784 | 44.76 | 4.6 | −1.56 | −0.59 |
| bulk | 200 | radials only | 0.536–0.541 | 36.12–36.26 | 1.3–1.6 | −1.48 to −1.57 | −1.11 to −1.14 |
| bulk | 200 | full, P=1 | 0.521 | 36.97 | 1.8 | −1.47 | −0.98 |
| bulk | 200 | full, P=2 | 0.525 | 37.41 | 1.7 | −1.46 | −0.94 |
| bulk | 200 | pair, P=1 | 0.536 | 36.79 | 1.4 | −1.48 | −1.04 |
| + vacancies | 40 | radials only | 0.817 | 45.65 | 2.5 | −0.51 | −0.38 |
| + vacancies | 40 | full, P=1 | 0.800 | 45.63 | 2.3 | −0.43 | −0.33 |
| + vacancies | 40 | full, P=2 | 0.748 | 46.87 | 1.4 | −0.49 | −0.27 |
| + vacancies | 200 | radials only | 0.621–0.639 | 36.30–36.35 | 1.7 | −0.81 | −0.11 to −0.15 |
| + vacancies | 200 | full, P=1 | 0.652 | 38.34 | 1.5 | −0.66 | −0.15 |
| + vacancies | 200 | full, P=2 | 0.629 | 38.12 | 1.6 | −0.82 | −0.20 |

For reference, pacemaker sqrt(ρ) (bulk data) scores Si +0.03, Ge +0.14 and 5.9% elastic.

The density readouts for Si are d ≈ −3×10⁻⁴ (bulk data, 200 steps, P=1) and −1.2×10⁻³ (with vacancies). In bulk, ρ = −1.215, and it moves to about −1.0 next to a vacancy, so the density does detect the defect. The term still contributes under 2 meV per atom and changes by under 0.3 meV across the vacancy, against a site-energy spread of 0.5–1.0 eV. The density runs' small vacancy differences therefore come from their different radial trajectories, not from the density term.

## Cantor

B and C44 are % deviations from MACE for the three random draws. The vacancy column is the MAE over 2 sites per species.

| Data, mode | Budget | Model | E (meV/atom) | F (meV/Å) | B % per draw | C44 % per draw | Vacancy MAE (eV) |
|---|---|---|---|---|---|---|---|
| bulk, joint | 40 | radials only | 7.46 | 148.7 | +10 / +13 / +1 | +2 / +8 / +24 | 0.48 |
| bulk, joint | 40 | full, P=1 | 7.86 | 150.4 | +11 / +13 / +7 | +3 / +9 / −10 | 0.44 |
| bulk, joint | 40 | full, P=2 | 10.47 | 152.6 | +16 / +9 / −1 | +8 / +11 / +29 | 0.29 |
| bulk, joint | 40 | pair, P=1 | 7.61 | 149.1 | +9 / +13 / +7 | +1 / +6 / −10 | 0.47 |
| bulk, joint | 200 | radials only | 6.21–6.30 | 120.0 | +13 to +16 | −7 to −11 | 0.41–0.42 |
| bulk, joint | 200 | full, P=1 (stopped at 100) | 7.58 | 145.2 | +11 / +12 / +10 | +2 / +5 / −11 | 0.45 |
| bulk, joint | 200 | full, P=2 (stopped at 50) | 10.56 | 152.7 | +16 / +8 / −1 | +8 / +10 / +29 | 0.29 |
| bulk, joint | 200 | pair, P=1 | 7.36 | 146.5 | +10 / +13 / +9 | +2 / +5 / −10 | 0.46 |
| + vacancies, joint | 40 | radials only | 7.12 | 146.6 | +1 / 0 / −13 | +1 / +2 / −23 | 0.34 |
| + vacancies, joint | 200 | radials only | 5.95–6.09 | 119.8–120.1 | 0 / 0 / −12 to −13 | −12 / −11 / +1 | 0.45–0.46 |
| + vacancies, joint | 200 | full, P=1 (line search failed at 11 steps) | 8.74 | 149.7 | +11 / +2 / −7 | +9 / +10 / +26 | 0.28 |
| + vacancies, alternating | 40 | full, P=1 | 7.14 | 150.1 | +4 / −1 / −12 | +3 / +4 / +21 | 0.37 |
| + vacancies, alternating | 40 | full, P=2 | 7.19 | 150.2 | +4 / −1 / −12 | +3 / +4 / +21 | 0.38 |
| + vacancies, alternating | 200 | radials only | 5.94–5.97 | 119.7–119.8 | 0 / 0 / −12 | −12 / −12 / +1 to +2 | 0.47 |
| + vacancies, alternating | 200 | full, P=1 | 6.82 | 125.7 | +1 / 0 / −12 | −11 / −10 / +4 | 0.44 |
| + vacancies, alternating | 200 | full, P=2 | 6.84 | 124.6 | +1 / −1 / −13 | −11 / −10 / +2 | 0.45 |

For reference, pacemaker linear κ=0.3 (bulk data) scores a vacancy MAE of 1.14 eV, with B off by −33 to −43%.

Once optimised properly (alternating mode, full budget), the density leaves the Cantor properties essentially unchanged: vacancy MAE 0.44–0.45 against 0.47 eV, and B and C44 within a few %. It costs about 15% in energy RMSE, partly because the radials get only half the steps. The low vacancy MAEs of the stalled joint runs (0.28–0.29 eV) come with near-initial radials and much worse bulk errors, so they are not a density effect.

## Optimiser

**Joint mode (approach A).** The preconditioner scale r_η = RMS(∂f/∂H)/RMS(∂f/∂V) behaved very differently on the two systems:

| System and density | r_η |
|---|---|
| SiGe, full P=1 | 8 to 280 |
| Cantor, full P=1 | 1e-3 to 3e-3 |
| Cantor, full P=2 | 0.5 to 5.5 |

On Cantor the tiny r_η gives huge density steps. In the strongly nonlinear density coordinates the line search then fails (after 11 accepted steps), or the relative-decrease test stops the run early.

**Alternating mode (approach B).** It completes the full budget on Cantor.

**Cost.** A joint 50-step round took 145 s on SiGe (A100), against 101–128 s for radials alone: +15–40%, including the extra gradient for the preconditioner.

## Conclusions

1. **A linear readout over a learned sqrt(ρ) is not an FS embedding in practice.** The density column is nearly collinear with the linear ACE basis, the ridge suppresses it, and it neither helps nor hurts the physical properties. The rest of the linear-model advantages are unaffected: convex final fit, UQ and speed. So keeping the machinery is cheap, but on these data it does not buy vacancy accuracy.
2. **Training data sets the defect energies.** Adding vacancy cells cut the SiGe Si vacancy error by about 1 eV and brought Ge inside the 0.3 eV target, with no model change. The remaining Si error grows with longer radial learning, which suggests up-weighting the defect configurations, or adding more of them. That is cheap to test with the existing per-type noise machinery.
3. **Options for the density itself:**
   - A fixed-form embedding, as pacemaker does, would make the curvature count. That is nonlinear in the readout, which is the spec's fallback and gives up convexity.
   - Alternatively, a density span that excludes the linear basis's own columns, so the linear part of sqrt(ρ) is not already representable.

   Both are separate decisions.
4. **Optimiser.** If the density is kept, alternating mode should be the default, or the joint preconditioner should cap r_η ≤ 1 (never amplify density steps).
