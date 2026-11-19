# YE3T and polynomial MLPs in the Si accuracy-cost benchmark (design)

Status: draft, 2026-11-12. Owner: James Kermode.

## Goal and scope

Add three entries to the Si accuracy-cost benchmark (`docs/dev/accuracy-cost-spec.md`, `bench/accuracy_cost/`):

1. **YE3T-symmetric, an evaluator.** Goff and Thompson's (Sandia) λ = (N) YE3T kernels evaluating the *existing*
   linear ACE ladder. Same models and same numbers, only a different code. Timing only.
2. **YE3T-mix, a model family.** Linear models with mixed-permutation intermediates and a fully symmetric scalar
   output (λ = (N), L = 0), fitted with the authors' own tooling under their published protocol (arXiv:2609.31895,
   Sec. IV C). Scored and timed like every other family.
3. **polymlp, a model family.** Seko's polynomial MLPs (pypolymlp, BSD-3), fitted with pypolymlp and timed on the
   CPU only through Seko's LAMMPS package.

This is a **first pass**. We do the fits under a documented protocol and then share it with the YE3T authors (and
optionally Seko), who may open a PR to change the protocol or the fits. Everything must be reproducible from the
repo plus `$ACEJAX_AC_ROOT`, and written so an outside author can follow and change it.

Out of scope: GPU timings for polymlp (it has no GPU path) or for YE3T (unless the authors' code has one: record
what exists, don't build one); reimplementing either method in ace-jax; DAKOTA-style hyperparameter optimisation
(see Fairness); any change to existing families, rows or figures beyond adding the new families.

## Fixed rules (same as every family)

- Data: the GAP-18 split (`DATA/train.xyz`, `DATA/test.xyz`, split hash in `DATA/split.json`). **The test set is
  never used for any choice**: not for regularisation, hyperparameters or model selection. Where a tool selects
  a regulariser by validation, give it a validation split carved from *train* (seeded, recorded) or its own CV
  on train.
- Cutoff: `RCUT` = 5.0 Å for both radial and angular parts.
- Energies relative to `E0_SI` where the tool needs it (NEP precedent, `fit_nep.py`); the calculator adds it back.
- Weights: GAP-18 per-config-type force sigmas, as for the other families (`weights.py`). If a tool can't take
  per-structure weights, map them to the nearest thing it supports (e.g. per-dataset weights grouped by
  `config_type`) and document the mapping. If it can't do that either, fit unweighted and flag it.
- Virials: fit them where the tool supports it, as the other families do.
- Scoring: `evaluate.py` (test force/energy RMSE) and `bench/si_qoi/score.py` (21 QoIs), through an ASE
  calculator in float64.
- Timing: `timing.py` in LAMMPS on Sulis, float64, with the existing CPU-node protocol (128 MPI ranks × 1 thread,
  4,096 atoms, same repeats and idle checks), plus a gate (`gate_model`) checking LAMMPS energies and forces
  against the reference evaluator before any timing row is accepted.
- Provenance: record code commit hashes, versions, build flags and module lists in a `VERSIONS-*` file and in each
  run's `status.json`. Record licences in `data.md` next to the pacemaker note.

## Phase 0: locate and build the codes

- **YE3T.** Find the authors' code. The paper says its evaluators agree with PACE to machine precision, with
  examples in the LAMMPS repository, and that the tagged basis and the Schur–Weyl kernels are in the Supplemental
  Material. Identify which repo or branch, which pair style, what model file format, and which fitting tool
  (probably FitSNAP). **If the code isn't publicly available, stop here and write a short note saying what is
  missing. Don't reimplement it.**
- **pypolymlp.** Install from conda-forge or PyPI into its own venv (`envs/polymlp/`, pinned). Find Seko's LAMMPS
  interface (`lammps-polymlp-package` or whatever the pypolymlp docs currently point to) and build it against the
  benchmark's LAMMPS version, with an env script modelled on `bench/scaling/envs/sulis-nep.sh`.
- Build both LAMMPS variants on Sulis with GCC/12.3.0 and OpenMPI/4.1.5, as for the existing CPU codes. Add
  `envs/sulis-cpu-ye3t.{json,sbatch}` and `envs/sulis-cpu-polymlp.{json,sbatch}` following
  `sulis-cpu-nep.{json,sbatch}`.

## Phase 1: YE3T-symmetric (evaluator)

1. Export the linear ACE ladder (o4d12, o4d14, o4d16, o4d20, plus o3d12 as a small check) into YE3T's evaluator
   format. Start from the `ace_yace` / `write_yace` path (`ace_to_yace.py`, tabulated radials) if YE3T reads
   .yace. Otherwise write a converter, `ace_to_ye3t.py`.
2. **Gate first.** Energies and forces must match ace-jax and ML-PACE on the gate cells to the existing parity
   tolerances (`parity.TOL`). If they don't, it isn't the same model: stop and report, and don't time it.
3. Timing rows, new code key `ye3t-sym`:
   - the standard CPU node, 128 × 1, 4,096 atoms;
   - **also one rank × one thread** for ye3t-sym, ML-PACE (stock and patched) and the ace-jax C++ kernel. This
     reproduces the paper's single-thread setting and tests its 2–5× (product evaluator) and ~2× (recursive
     evaluator) claims against our evaluators. Use results file `sulis-cpu-1x1-ye3t.jsonl`.
   - A100 rows only if the authors' code has a GPU path; otherwise say "CPU only" in the table.
4. Plot ye3t-sym as another linear ACE implementation (same marker family as the other linear ACE codes), with no
   new accuracy points: it shares ace's metrics rows.

## Phase 2: YE3T-mix (model family)

Families (separate rows, so the input-basis question stays visible):

- `ye3t_mix_A`: mixed-permutation intermediates, density-projected inputs (u = A).
- `ye3t_mix_T`: the same with the tagged basis (u = T), if the code supports it.
- `ye3t_sym_fit`: λ = (N) only, fitted with **the same tool, protocol and descriptor list** as `ye3t_mix_A`. This
  is the matched-span control.

Protocol (the authors' published one, as the default):

- Ridge regression with penalty 1e-8; ranks 1–8 with nmax = 6 and lmax = 4 as the anchor size; a ladder of 5–7
  sizes chosen so the descriptor counts span roughly the `ace` ladder's n_B (54 to 5,456). Record the exact
  (η, l) descriptor lists per size.
- Cutoff fixed at 5.0 Å. Use the authors' default radial decay unless they give a Si-specific value. No DAKOTA
  (see Fairness).
- Add `fit_ye3t.py`, modelled on `fit_nep.py`: write the tool's inputs from the split, run the fit, and register
  the model in `status.json`. Add a calculator in `evaluate.load_calc`, and sweep entries in `sweep.yaml` under
  the three families.

Optional Phase 2b (only if the tool can export the design matrix or the descriptors): refit `ye3t_mix_A` and
`ye3t_sym_fit` with ace-jax's smoothness prior, as families `ye3t_mix_A_sp` and `ye3t_sym_fit_sp`. This separates
the effect of the basis from the effect of the regulariser. Skip it, and note why, if the descriptors can't be
exported.

Analysis to write up (in `docs/dev/accuracy-cost-results.md`, a new section):

- `ye3t_mix_A` against `ye3t_sym_fit` at matched descriptor lists. The hypothesis is that they reach the same
  accuracy, because products of commuting A's lie only in λ = (N). Report any difference, and whether it goes
  away under the smoothness prior.
- `ye3t_mix_T` against both: is any gain due to the tagged feature family?
- The "ACE saturates" claim: overlay the `ace` and `ace_wl` ladders (to d24), which already go beyond the paper's
  rank-8 / nmax-6 / lmax-4 truncation.
- Force RMSE **and** QoI score for every point; the paper reports force RMSE only.

## Phase 3: polymlp (model family, CPU only)

- Family `polymlp`. Fit with pypolymlp on the train split.
- Ladder: follow Seko's systematic grid (J. Appl. Phys. 133, 011101, 2023, and the pypolymlp docs) with the cutoff
  fixed at 5.0 Å. Vary model type, polynomial order (max_p), the order and maximum l of the group-theoretical
  invariants, and the number of Gaussian radial functions, to get 6–10 sizes from a few hundred to ≥5k
  coefficients. Record each size's full input and its coefficient count.
- Regularisation: pypolymlp's own choice of alpha, but on a validation split taken from train (seeded, recorded),
  never test.
- Calculator: pypolymlp's ASE/Python interface for `evaluate.py` and the QoIs. Gate LAMMPS against it.
- Timing: CPU node 128 × 1, code key `polymlp`, results file `sulis-cpu-128x1-polymlp.jsonl`. No GPU row. Mark
  it "CPU only (no GPU code)" in the table, as for GAP.
- Effective size: also record the **numerical rank** of the training design matrix (SVD with a relative cutoff
  of 1e-10), next to the coefficient count. Products of invariants overlap with higher-order invariants, so the
  coefficient count overstates the effective size. If pypolymlp doesn't expose the design matrix, note that and
  skip it.
- Reference only: if Seko's Polynomial MLP repository has published Si potentials at a 5 Å cutoff, add them as a
  separate `polymlp_repo` family. They aren't trained on our split, so mark them as such or leave them off the
  Pareto plot.

## Fairness and what is deliberately not done

- No method gets per-system hyperparameter optimisation in this pass. The existing families use fixed protocols,
  and the YE3T paper's DAKOTA tuning of cutoff and decay is replaced by fixed values. Say this explicitly in the
  README so the authors can propose a PR adding tuned fits. If they do, tune the `ace` ladder under the same
  budget too.
- Every new family goes through the identical scorer, gate and timing harness. Don't use special-case timing code.
- Report mixed precision explicitly if a fast path has no float64 mode (it shouldn't for these codes).

## Deliverables

- Code: `fit_ye3t.py`, `fit_polymlp.py`, converters, `evaluate.load_calc` entries, `timing.py` code entries,
  `sweep.yaml` entries, env scripts, and gate support for the new pair styles.
- Results: rows in `results/metrics.jsonl`, the QoI results, and the timing `.jsonl` files above.
- Figures: the new families in `bench/si_qoi/plot_cost.py`, `plot_cost_web.py` and `qoi_heatmap.py`. Keep the
  existing colours, and add markers for YE3T and polymlp.
- Docs: a results section in `docs/dev/accuracy-cost-results.md`; provenance and licences in `data.md`; and
  `bench/accuracy_cost/ye3t/README.md` and `bench/accuracy_cost/polymlp/README.md`. Each README should be short and
  aimed at the method's authors: the protocol, the exact commands, where the inputs live, and the line *"PRs that
  change the protocol or the fits are welcome."*
- Don't update the user-facing Performance page (`docs/user/benchmarks.md`) until James has reviewed the results
  and the authors have had a chance to respond.

## Tests

- Unit: converters round-trip (ace → YE3T format → energies match ace-jax on a few structures); calculators load
  and return E/F/S of the right shape; sweep entries resolve.
- Gates: the LAMMPS-vs-reference parity checks above, run in the sbatch scripts before timing.
- A smoke run on one small size per family under `ACEJAX_AC_RUNS` / `ACEJAX_AC_RESULTS` overrides before the full
  ladders.

## Stop-and-report points

- The YE3T code or the tagged-basis definition isn't available.
- The YE3T-symmetric gate fails against ace-jax or ML-PACE.
- polymlp's LAMMPS package won't build against the benchmark LAMMPS.
- A tool can only select hyperparameters using the test set.

In each case write what was tried into the family README and the session summary, and continue with the other
families.
