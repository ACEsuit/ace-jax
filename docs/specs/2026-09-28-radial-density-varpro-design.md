# Joint radial + density learning by VarPro — design

Sub-project 1 of 2. Sub-project 2 (TRACE-style species embedding) gets its own spec and PR. It reuses the outer loop built here.

## Motivation

`docs/learn-radial-pacemaker-comparison.md` (PR #14) showed the following:

- VarPro-learned radials give the best elastic constants of any model tested. SiGe comes out at 2.3–2.4% mean deviation from the MACE teacher, against 5.9–7.7% for pacemaker.
- Every *linear* model misses SiGe vacancy formation energies by 0.2–1.8 eV.
- The only model that gets vacancies right is pacemaker with a sqrt(ρ) (Finnis–Sinclair) embedding, at +0.03 and +0.14 eV.

The goal is to add an optimised sqrt(density) term. It must keep what makes the linear model worth having:

- a convex final fit;
- Bayesian, POPS and GP UQ;
- fast refits;
- a short L-BFGS budget.

The cap on the step count is decided once the joint problem runs (the 40 vs 200 comparison below).

## Earlier findings this builds on

These are from `~/.julia/dev/ACEpotentials/docs/findings/`, the FS, VarPro, embedding and embed-VarOpt spikes.

- **A fixed sqrt(ρ_tot) feature helps.** On Cantor it gave −5% force RMSE.
- **VarPro-learned pair densities help more.** η learned by VarPro over the pair basis gave −10 to −12% forces on Cantor, and the gain transfers to held-out data. On SiGe the existing pair-only term (`fit/density.py`) had no measurable effect. That is why this design spans the full basis by default.
- **Learned species embeddings gain little.**
  - The gain is marginal once a baseline is present.
  - A naive joint [θ; E] BFGS is ill-conditioned. It needs a profiled nest.
  - The rank of a frozen embedding must lie in [S, C(S+ν−1, ν)], or the basis becomes linearly dependent.

  Together with TRACE (Darby et al.), this supports *random* species weights by default. It is the reason the embedding comes second and has learning as an ablation only.

## Decisions (agreed in brainstorming)

1. **Density first, species embedding second.** Separate specs and PRs, with a shared outer loop.
2. **The density span is selectable. The full basis is the default.** The span is set by a mask m over basis columns: `full` (every column) or `pair` (the pair block only, which reproduces the existing `density_rows_pair`). Pair-only is an ablation.
3. **Optimiser: approach A.** A joint L-BFGS over the stacked outer variables [V; η], with a small diagonal preconditioner to handle the scale mismatch. **Approach B** is the fallback: alternating blocks, each a short L-BFGS on V then on η.
4. **Keep a linear readout.** The sqrt(density) features (P ≥ 1 of them) enter as extra linear columns, with η learned in the outer loop and then frozen. The fallback, if success criterion 1 fails, is a truly nonlinear correction. That fallback is a separate decision and is out of scope here.

## Model

The site energy of atom i with species z is

```
E_i = c_z · X_i  +  Σ_{p=1..P}  d_{p,z} · F(ρ_{p,i})
ρ_{p,i} = η_{p,z} · (m ⊙ X_i)
F(ρ) = ssqrt(ρ) = ρ (ρ² + ε)^(-1/4)
```

The symbols are as follows:

- X_i is the site's ACE descriptor, which depends on the radials W.
- The mask m ∈ {0,1}^D selects the density span. η has shape (P, NZ, D) and is zero off the mask.
- ssqrt is the smooth signed sqrt already in `fit/density.py` (`_ssqrt`).

At fixed (W, η) the model is linear in [c | d]. Its length is `len_basis + NZ·P`. The final fit is therefore the same convex ridge/Bayesian solve, and the MAP-θ, POPS and GP machinery apply unchanged to the widened design.

**P = 0 is exactly today's model.** Every code path must reduce to it bit-for-bit.

### Evaluation: `FSModel`

`eval/fs_model.py` defines `FSModel(base, eta, d, mask, eps)`. It wraps an `ACEModel`:

- `site_energies` and `site_energies_dense` compute the base site energy, plus the density term from the base's site basis (`site_basis` / `site_basis_dense`).
- Energy, forces and virial come from `EdgeSiteModel.energy_forces_virial*`, the same `value_and_grad` over `rij`. There is no bespoke force code.
- `FSModel` subclasses `EdgeSiteModel`, delegating layout and neighbour attributes to `base`. `ACECalculator`, `PaddedACECalculator`, `qoi.py` and `md_*` therefore take it unchanged.

### File format

These are new optional npz keys. With them absent, the model loads as today.

| key | shape | meaning |
|---|---|---|
| `fs_eta` | (P, NZ, D) | frozen density weights |
| `fs_d` | (P, NZ) | readout coefficients for the density columns |
| `fs_mask` | (D,) bool | span mask |
| `meta["fs"]` | `{"P": int, "F": "ssqrt", "eps": float}` | |

- `aj.load` wraps the model in `FSModel` when `fs_eta` is present.
- `construct.export.patch_radial_npz(src, dst, model, readout=None, fs=None)` writes the keys, where `fs` is `(eta, d, mask, eps)`.
- `schema_version` stays 1, because the keys are additive.

## Fitting

### Statistics

`fit/density.py` gains `linear_density_statistics(model, eta, mask, cfg, ds)`. For each batch it:

1. reuses `linear_rows` for the basis rows (E/F/V of X, and the edge Jacobian J);
2. appends NZ·P density columns per quantity, from a new `density_rows_masked(eta, mask, cfg, batch, X, J)` that takes the mask and P ≥ 1. The existing `density_rows(eta, cfg, batch, X, J)`, which is pair-only with P = 1 and used by `tests/test_gp_density.py`, keeps its signature and becomes a thin wrapper over it;
3. accumulates the same `LinearStats` (G_q, b_q, yy_q, counts) as `linear_statistics`, over `L' = len_basis + NZ·P` columns.

- `density_rows` and `density_rows_pair` become the `mask = pair block`, P = 1 special case. Their existing tests must still pass unchanged.
- Density rows use the analytic chain rule. The force row of a column is `F'(ρ)·(η·∂X/∂r)`, so the cost is one extra contraction of J per p. There is no second autodiff pass.

### Prior on the density readout

The readout d gets a ridge weight of γ_d = the geometric mean of Γ (the ACE smoothness prior). The prior vector is `gamma' = [Γ ; γ_d·1_{NZ·P}]`. It is shared by the ridge, the MAP θ, and the final Bayesian solve.

**Prerequisite fix.** Several places size the linear block from `cfg.len_basis` rather than from the prior:

- `fit/objective.py:prior_precision`;
- `fit/solve.py` (3 sites);
- `fit/predict.py:293`;
- `fit/hostcache.py`.

These must size from `prob.gamma.shape[0]` (or an explicit `L`), so a widened design works. This is a separate commit, and a pure refactor with identical results at P = 0.

### Outer variables and gauges

The outer vector is `x = [vec(V)/s_W ; vec(H)/s_η]`:

- V is the radial parameter of `learn_radial`, after gauge normalisation (unchanged).
- H is the raw density weight, and `η = normalise_ρ(H)` rescales each (p, z) so that the training ρ_{p,i} has unit mean square.

That removes the scale gauge between η and d, which d absorbs. It also keeps ε meaningful. The sign gauge is left alone: ssqrt is odd, so d absorbs it.

The objective is the VarPro projected residual `projected_residual_from_stats` on the widened statistics, with θ re-profiled by MAP every `reprofile_every` steps as in `learn_radial`. The gradient is autodiff through the streamed statistics. By the envelope theorem that is the Kaufman gradient, exactly as for radials alone.

**Priors:**

- the existing radial priors (roughness, spectral, gap);
- `λ_η Σ_{p,z} ‖Γ_m ⊙ η_{p,z}‖²`, where Γ_m is Γ restricted to the mask. It penalises density weight on rough, high-degree functions.

Both are scaled relative to r0, as the radial priors are.

### Preconditioner

The diagonal preconditioner scales are `s_W` and `s_η`, one per block. Each is set to the RMS of that block's gradient at the start of a round, and reset at every θ re-profile, because L-BFGS memory is reset there anyway.

The step function stays a single compiled `_lbfgs_step`: the scales are traced arguments, not statics. The objective is invariant to the scales, which the tests below check.

### Modes

`learn_radial_density(prob, ds, W0, H0, *, mask, P, mode="joint"|"alternating", steps, lam_eta, ...)` lives in `fit/radial_density.py`. It returns the same kind of result as `learn_radial`, plus η.

- `alternating` runs k steps on V (η fixed), then k steps on H (V fixed), sharing the step budget. The re-profile cadence is unchanged.

### Initialisation

- **p = 1:** equal weights on the pair block for every species (the total pair density). This is the established FS starting point. It is zero elsewhere, even under the `full` mask, so the full span starts at the pair solution.
- **p ≥ 2:** the p = 1 initialisation plus a seeded small random perturbation over the mask. Without it the P columns would be identical and the design singular.

H0 is then normalised as above.

### Gate

`fit_radial` extends to a candidate set scored by `holdout_score` on the widened readout:

- `init`;
- `radials_only` (today's learned result);
- `radials+density_lam_eta=<l>`, one per value in `lam_eta_grid`.

The existing gate picks the best candidate, with ties going to the earlier (simpler) one. Density is therefore only kept when it helps on held-out data.

## Driver and benchmarks

- `bench/learn_radial/run.py` gains:
  - `--density {none,pair,full}` (default `none`, so today's behaviour is unchanged);
  - `--P` (default 1);
  - `--density-mode {joint,alternating}`;
  - `--lam-eta` (a grid).

  With density, it writes the fs keys via `patch_radial_npz`.
- `rmse.py`, `qoi.py`, `qoi_compare.py` and `modal_run.py` are reused unchanged, apart from passing the new flags through. `FSModel` makes the npz transparent to the calculators.
- The benchmark matrix covers SiGe and Cantor, on Modal A100s, at 40 and 200 steps each:
  - radials-only (baseline, from PR #14);
  - full P=1;
  - full P=2;
  - pair P=1.

  Results are RMSE and QoI against MACE, written to `docs/learn-radial-density-results.md`. The step count needed for the joint problem is decided from these runs.
- moriarty stays free for the user's benchmarks. Use lestrade and Modal.

## Tests

In `tests/test_radial_density.py` and `tests/test_fs_model.py`, float64 throughout:

1. **Density rows by finite differences.** The E, F and V rows of the density columns match FD, for the full mask, the pair mask and P = 2.
2. **P = 0 matches today bit-for-bit.** The statistics, `projected_residual` and `FSModel` all equal the current path.
3. **Streamed statistics match an in-memory design.** The widened `LinearStats` equal the dense design built from explicit rows.
4. **The objective matches the QR reference.** The projected residual on the widened stats equals the residual from a dense QR least-squares solve.
5. **The gradient matches finite differences** over [V; η], in both modes.
6. **The gauges and the preconditioner leave the objective alone.**
   - The objective is unchanged under scaling of H per (p, z).
   - It is unchanged under a flip of the sign of η_p.
   - It is unchanged by the preconditioner scales.
7. **Single compile.** One `_lbfgs_step` compilation across rounds and re-profiles, as in the existing radial test.
8. **The `prior_precision` / `len_basis` generalisation.** A widened γ gives a correctly sized Λ. At P = 0 the results are identical.
9. **`FSModel` round trip.**
   - Export with `patch_radial_npz`, then `aj.load`: E, F and V match the in-memory `FSModel`.
   - The FSModel forces match an FD of its energy.
   - `ACECalculator` and `PaddedACECalculator` agree.
10. **Smoke test.** `run.py --density full --P 1 --steps 3` on the small fixture writes a loadable npz.
11. **The gate works on synthetic data.**
    - On data generated with a sqrt(ρ) term, the gate picks the density candidate.
    - On purely linear data, it picks `radials_only` or `init`.

## Success criteria

1. **SiGe vacancies:** the Si and Ge vacancy formation energies are within 0.3 eV of MACE. Today they are off by 0.7–1.8 eV.
2. **Elastic properties:** the SiGe mean absolute deviation stays at or below 3%. Today it is 2.4%.
3. **RMSE:** validation E and F are no worse than radials-only at the same step count.
4. **The gate:** it keeps density only when the held-out score improves, and the synthetic test passes.

If criterion 1 fails while 2–4 hold, the nonlinear-correction fallback becomes a separate decision, brought back to the user. It does not become an automatic extension of this work.

## Out of scope

- Species embedding (sub-project 2).
- A nonlinear readout.
- Using the density features in the GP arm or the inducing kernel.
- `aj fit` CLI flags. This is bench-driver only until the benchmarks justify it.
