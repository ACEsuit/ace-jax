# YE3T and polymlp in the Si accuracy-cost benchmark: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add YE3T-symmetric (evaluator), YE3T-mix (family) and polymlp (family) to `bench/accuracy_cost`.

**Architecture:** Each new code plugs into the existing drivers: a `fit_*.py` per family, a calculator in
`evaluate.load_calc`, a pair style in `scaling/run_lammps._pair`, a reference evaluator and tolerance in
`scaling/parity`, a `PLAN` entry in `timing.py`, env scripts for Sulis. No special-case timing code.

**Tech stack:** ye3t / ye3t-methods / ye3t-lammps (Goff & Thompson), pypolymlp / lammps-polymlp-package (Seko),
each in its own venv; LAMMPS on Sulis (GCC/12.3.0, OpenMPI/4.1.5).

**Spec:** `docs/dev/specs/2026-11-12-ye3t-polymlp-benchmark-design.md`

## Phase 0 findings (2026-11-12) that change the spec's assumptions

Sources are cloned under `$ACEJAX_AC_ROOT/ext/` (commits in `data.md`).

| Spec assumption | What the code does | Decision |
|---|---|---|
| YE3T code may be private | Public: `ye3t` f8ff96d (BSD-3), `ye3t-methods` 6b85c54 (BSD-3), `ye3t-lammps` 06100f9 (GPL-2.0, `pair_style ye3t`) | Proceed |
| Fitting tool is probably FitSNAP | `ye3t_methods.LinearModel.fit` (ridge; ASE calculator; `export_lammps`) | `fit_ye3t.py` drives it |
| ye3t-sym can read the ladder's tabulated `.yace` | `pair_style ye3t` reads `.yace`, but only `ChebExpCos` radials, no halved `ms_combs`, exact key sets | New timing-only family `ace_cheb`: the ladder's radials projected on ChebExpCos, written in the full conjugation-closed form. ye3t, stock ML-PACE (product and recursive) and ace-jax all read the same file, so the gate is same-file parity; the projection error against the source `.npz` is reported, not gated |
| The paper's protocol fixes weights and virials | Global E/F/S weights only; no per-structure weight | Fit unweighted (E 1, F 1), flagged; virials fitted where the source supports a stress target |
| "mix with u = A" and "u = T" are code options | `tensor_product.kind` `density` / `tagged`; `intermediates.young_kappa` `all_valid` / `symmetric_only` | `ye3t_mix_A`: density + all_valid. `ye3t_sym_fit`: density + symmetric_only. `ye3t_mix_T`: tagged |
| polymlp times at 128 MPI ranks × 1 | `pair_style polymlp` has no MPI: per-atom arrays are indexed by tag, sized by local count; README says OpenMP only | Stop-and-report for the 128 × 1 row. Time 1 rank × 1 thread (comparable with the 1 × 1 file) and 1 rank × 128 OpenMP threads, both labelled; never plotted as a 128 × 1 row |
| pypolymlp picks alpha itself | argmin of its weighted RMSE on the "test" datasets it is given | Pass a seeded 10 % validation split of train as "test"; the benchmark test set never enters pypolymlp |
| pypolymlp per-structure weights | One scalar per dataset; built-in row heuristics | One dataset per config type (and stress availability), weight = sigma_F,default / sigma_F |
| pypolymlp ASE stress | Returned as +virial, eV/cell, order xx yy zz xy yz zx | Wrapper converts to ASE's convention and adds E0 back |

## Global constraints

- Test set never used for any choice. Cutoff 5.0 Å. Energies relative to `E0_SI`, added back by the calculator.
- Scoring through `evaluate.py` and `si_qoi/score.py` in float64; timing through `timing.py` with a parity gate.
- Provenance in `VERSIONS-*`, `status.json`, `data.md`. No change to existing families, rows or figures.
- `docs/user/benchmarks.md` is not touched.
- No GPL code copied into `src/` or `bench/` (ye3t-lammps is built, not vendored).

## Review focus

- A `.yace` written for ye3t that stock ML-PACE reads differently (full vs half `ms_combs` double counting).
- polymlp energies off by `N * E0_SI`, or stress with the wrong sign or Voigt order, in the QoIs.
- A validation split that leaks test frames, or changes between runs (seed and indices recorded).
- A polymlp timing row mistaken for a 128-rank row in `make_cost.py` (ranks and threads are in the row).
- ye3t fits compared at unequal descriptor lists (sym_fit must use mix_A's catalogue with only the kappa filter changed).

## Tasks

### Task 1: harness wiring (no external code needed)

**Files:** `bench/scaling/run_lammps.py`, `bench/scaling/parity.py`, `bench/scaling/sweep.py`,
`bench/accuracy_cost/timing.py`, `bench/accuracy_cost/evaluate.py`, `bench/accuracy_cost/sweep.yaml`,
`tests/test_bench_accuracy_cost.py`.

- [ ] Tests first: `_pair("ye3t", "m.yace", ["Si"], "cpu") == "pair_style ye3t\npair_coeff * * m.yace Si\n"`;
  `_pair("polymlp", "polymlp.yaml", ["Si"], "cpu") == "pair_style polymlp\npair_coeff * * polymlp.yaml Si\n"`;
  `lammps_supported("ye3t-sym", "gpu")` and `("polymlp", "gpu")` are False; `parity.TOL` has `ye3t-sym`,
  `ye3t` and `polymlp`; `timing.PLAN` has `ace_cheb`, `ye3t_mix_A`, `ye3t_mix_T`, `ye3t_sym_fit`, `polymlp`;
  the sweep-families test lists the new families.
- [ ] Implement; `ACEJAX_AC_CPU_RANKS=1` with `polymlp` sets `OMP_NUM_THREADS` from `ACEJAX_AC_CPU_THREADS`.

### Task 2: `ace_cheb` converter (`bench/accuracy_cost/ace_to_ye3t.py`)

- [ ] Test: convert a small fixture ACE model; ace-jax `PACEModel` on the file matches the source model to the
  projection error the converter reports; every monomial's conjugate partner is present; keys are exactly
  ye3t's.
- [ ] `convert(path, lam, K, dcut) -> (tree, info)`: reuse `ace_to_yace.products`; least-squares ChebExpCos
  projection of `R_nl r^l` and of the rank-1 function; full `ms_combs`; `info` has the projection errors.
- [ ] `register(size)` writes `runs/ace_cheb/<size>/model.yace` and `status.json` (`source` = the `.npz`).

### Task 3: polymlp (`fit_polymlp.py`, `envs/polymlp/`, `polymlp/README.md`)

- [ ] Venv `ac-polymlp` (pinned), `envs/polymlp/pyproject.toml`.
- [ ] Tests (skip without pypolymlp): dataset grouping by config type and stress; validation split seeded and
  disjoint; calculator returns E (with E0), F (N, 3), stress in ASE convention (finite-difference check).
- [ ] `fit(size)`: params from the sweep entry, datasets, `fit_polymlp(params, train, val)`, `polymlp.yaml`,
  `n_params`, numerical rank (updating QR over batches, SVD of R, cutoff 1e-10), `status.json`.
- [ ] Ladder: count coefficients with `n_features` over Seko's grid at 5.0 Å; pick 6–10 sizes.
- [ ] Fit (mnf148), score, QoIs.
- [ ] Sulis: `scaling/envs/sulis-polymlp.sh`, `envs/sulis-cpu-polymlp.{json,sbatch}`; gate; time.

### Task 4: YE3T (`fit_ye3t.py`, `envs/ye3t/`, `ye3t/README.md`)

- [ ] Venv `ac-ye3t` (torch CPU, ye3t, ye3t-methods at the pinned commits).
- [ ] Smoke: the authors' quickstart density fit runs; our extxyz reads; stress target works or is refused.
- [ ] `fit(family, size)`: config from the sweep entry (ranks, nmax, lmax, kappa policy, tensor product),
  ridge 1e-8, cutoff 5.0, the authors' default decay; `model.write`, `export_lammps`, descriptor labels saved.
- [ ] Ladder of 5–7 sizes spanning n_B 54–5,456 by descriptor count; fit, score, QoIs.
- [ ] Sulis: `scaling/envs/sulis-ye3t.sh` (LAMMPS stable_22Jul2025_update4 + ML-YE3T + ML-PACE),
  `envs/sulis-cpu-ye3t.{json,sbatch}`; gates; 128 × 1 and 1 × 1 rows.
- [ ] Phase 2b if `basis.create(atoms, derivatives=True)` gives the design rows for the density source.

### Task 5: figures and docs

- [ ] `plot_cost.py`, `plot_cost_web.py`, `qoi_heatmap.py`: new families and markers.
- [ ] `docs/dev/accuracy-cost-results.md` section; `data.md` provenance and licences; the two READMEs.
