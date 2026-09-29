# Learned radials at deployment: splining the analytic branch

**Scope:** by default (`spline_tol="auto"`), `lean`, `ACECalculator` and `export_lammps` spline only a *learned* analytic tensor radial, at 1e-10.

- **Learned radials:** the model has `radial_learned` set, which `radial_learn` writes into meta_json. The splined result agrees with the full model to up to ~1e-9 in energy (relative) and ~2.3e-8 of max|F| in forces, not to roundoff.
- **Other analytic models:** Julia `ace_model` exports and Python-authored models are `radial_kind="analytic"` too, but not learned, so they stay exact by default. A float `spline_tol`, e.g. `1e-10`, opts them in; `None` never splines.
- **Pair radial:** never learned, so `"auto"` keeps an analytic pair radial exact.
- **Old files:** learned-radial files written before the flag load as not learned. Mark one with `ace_jax.construct.export.mark_radial_learned(path)`, or pass `spline_tol=1e-10`.

The policy lives in `splinify.spline_plan` alone.

A learned tensor radial (`fit/radial_learn.py`) lives on the **analytic** branch
of `ACEModel._radial_one`:

    R_n(r) = env(x) * sum_q Wnlq[zi, zj, n, q] P_q(x),   x = T_{zi,zj}(r)

Per edge this runs the n_q-term polynomial recursion and then contracts it with
the (n_rnl x n_q) block `Wnlq[zi, zj]`. The stock (Julia-exported) models are on
the **spline** branch instead: a 4-row gather from a cubic B-spline table and a
cubic. The lean evaluation form of PR #16 (`eval/model.py::lean`) also leans on
the spline branch. Its biggest lever, the species-compact l-blocks
(`block_dense`, `_rnl_owner`), reads the spline table's zero pattern, so
analytic models do not get it.

This note measures what that costs, and what `to_spline` recovers. `to_spline`
is the deploy-time conversion of an analytic radial to the spline branch, and
`lean` now applies it.

## 1. Timing: analytic against spline (A100, 8192 atoms, float64)

**Setup:**
- **Models:** the benchmark models `bench/scaling/models/ace_{SiGe,Cantor}_{medium,large}.npz`.
- **Analytic variants:** each model is converted by `to_analytic(model, n_q)`, and `rnl_Wnlq` is perturbed on its active rows only (20% of the row rms, Gaussian) to mimic learning.
  - Learning keeps the zero rows exactly zero: `radial_model.normalise` freezes them.
  - `_filled` also fills the zero rows, so no per-species pattern is left.
- **Timed call:** the dense E/F/V model call, jitted, median of 10 per round, median over 3 interleaved rounds, all in one container.
- **Source:** `bench/perf/learned_radial_bench.py` via `bench/perf/modal_gap.py::learned`. Results are in `bench/perf/results/learned_radial_8192.json`, device `NVIDIA A100-SXM4-80GB`.

**Columns:** ms per E/F/V call, with the ratio to the stock spline model's full call in brackets.
- **full:** the model as given.
- **lean_exact:** `lean(m, spline_tol=None)`. It keeps the analytic radial, so it is exact but not species-compact.
- **lean:** `lean(m)`. It applies `to_spline` first, then the spline lean form. This run used tol 1e-8; the default is now 1e-10, whose speed is in section 3.

| model | radial | full | lean_exact | lean (to_spline) |
|---|---|---|---|---|
| SiGe_medium | spline (stock) | 3.83 (1.00) | 2.80 (0.73) | = lean_exact |
| | analytic n_q=8 | 4.77 (1.25) | 3.04 (0.79) | 2.75 (0.72) |
| | analytic n_q=12 | 5.39 (1.41) | 3.43 (0.89) | 2.71 (0.71) |
| | analytic n_q=16 | 6.39 (1.67) | 4.31 (1.12) | 2.79 (0.73) |
| | analytic n_q=20 | 6.83 (1.78) | 4.75 (1.24) | 2.66 (0.69) |
| | n_q=12, filled | 5.10 (1.33) | 3.37 (0.88) | 2.71 (0.71) |
| Cantor_medium | spline (stock) | 6.67 (1.00) | 3.23 (0.48) | = lean_exact |
| | analytic n_q=8 | 7.90 (1.18) | 5.39 (0.81) | 3.34 (0.50) |
| | analytic n_q=12 | 11.23 (1.68) | 6.19 (0.93) | 3.29 (0.49) |
| | analytic n_q=16 | 9.85 (1.48) | 8.33 (1.25) | 3.93 (0.59) |
| | analytic n_q=20 | 13.42 (2.01) | 8.97 (1.34) | 3.44 (0.51) |
| | n_q=12, filled | 11.28 (1.69) | 6.74 (1.01) | 3.96 (0.59) |
| SiGe_large | spline (stock) | 6.94 (1.00) | 4.58 (0.66) | = lean_exact |
| | analytic n_q=12 | 9.99 (1.44) | 5.78 (0.83) | 4.36 (0.63) |
| Cantor_large | spline (stock) | 4.25 (1.00) | 3.04 (0.72) | = lean_exact |
| | analytic n_q=12 | 5.81 (1.37) | 4.26 (1.00) | 3.14 (0.74) |

**Cost ratio analytic/spline (full model):**

| n_q | SiGe_medium | Cantor_medium |
|---|---|---|
| 8 | 1.25x | 1.18x |
| 12 | 1.41x | 1.68x |
| 16 | 1.67x | 1.48x |
| 20 | 1.78x | 2.01x |

The large models at n_q=12 are 1.44x (SiGe) and 1.37x (Cantor). The cost grows
with n_q: the recursion, and a per-edge gather of the (n_rnl x n_q) Wnlq block.
The Cantor n_q=12/16 inversion is one sample of XLA's fusion choices. Rounds agree to ~2%.

**How much of #16's lean gain analytic models lose.** On the stock spline
model, lean is 1.37x (SiGe_medium) and 2.07x (Cantor_medium) faster than full.
Without `to_spline`, the lean form of an analytic model:
- runs prune and pairfold, and l-blocks the dense A, but not species-compact;
- is 1.1-1.7x (SiGe_medium) and 1.7-2.8x (Cantor_medium) slower than the spline lean form, over n_q 8-20;
- at n_q=12 Cantor_medium, is 6.19 ms against 3.23 ms: it keeps less than half of the gain.

**Local CPU sanity check.** It ran on the local Mac CPU, before `to_spline`
existed (`bench/perf/results/learned_radial_cpu_pre_*.json`). It shows the same
shape:
- analytic/spline full: 1.09-1.50x (SiGe_medium) and 1.2-2.5x (Cantor_medium);
- Cantor_medium n_q=20 swapped (12.9x): its per-edge (E, n_rnl, n_q) Wnlq gather is ~6 GB at 8192 atoms;
- lean_exact kept 0.71-1.01x of the spline full cost, against 0.32x for the spline lean form (Cantor).

## 2. `to_spline`: the conversion

`ace_jax.eval.to_spline(model, n_intervals=None, tol=1e-10, deriv_tol=None, return_info=False) -> (model, max_rel_err)`
lives in `src/ace_jax/eval/splinify.py`. `ace_jax.fit.radial_model` re-exports
it, as the inverse of `to_analytic`.

**What it does:**
- **Tabulates** the envelope-free S(x) = sum_q W P_q(x) at the knots of a uniform grid on [-1, 1]: (x0, h, n) = (-1, 2/n_int, n_int + 1).
- **Interpolates** with the cubic B-spline `spline_eval` reads. The builder is `construct.radial_ace1.cubic_bspline_coefs`, the one behind the authored and Julia `splinify` tables. It is now a banded O(n) solve with an optional `end_d2`.
- **End condition:** the polynomial's exact second derivative at the end knots, where Interpolations.jl's `Line(OnGrid())` puts y'' = 0. That keeps the error O(h^4) up to the ends, where y'' = 0 would leave an O(h^2) boundary layer. The layer matters for the pair radial, whose r-envelope does not vanish at x = 1.
- **Error:** max over species pairs and the columns A reads of max|spline - exact| / max|exact|, on 10 points per interval. For R_nl it includes the envelope; the pair radial is envelope-free. The derivative error is measured too (section 2b).
- **Interval choice:** doubles from 32 until error <= tol. The last doubling can overshoot by up to 16x, so it also tries buckets from the one the h^4 rate predicts. It returns the smallest bucket (section 2b) that meets tol.
- **Pair radial:** converted too when analytic (`pair_Wnlq`), on its own grid.
- **Already-spline radials:** a spline or `spline_factorised` R_nl, and a spline pair radial, are kept as they are.
- **Result:** a normal full model. `require_full` passes, the basis methods work, and the table is a live, trainable leaf. `require_full` is enforced on the input.

**Accuracy against intervals** (Cantor_medium, perturbed analytic n_q=12;
`bench/perf/results/learned_radial_to_spline_accuracy.json`). The error falls
16x per doubling (O(h^4)):

| intervals | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| max rel err | 7.3e-2 | 1.6e-3 | 7.0e-5 | 4.3e-6 | 2.7e-7 | 1.7e-8 | 1.1e-9 | 6.7e-11 | 4.2e-12 |

**Intervals chosen by the tol loop:**

| n_q | tol 1e-6 SiGe / Cantor | tol 1e-8 SiGe / Cantor | tol 1e-10 SiGe / Cantor |
|---|---|---|---|
| 8 | 168 / 128 | 512 / 441 | 1678 / 1393 |
| 12 | 215 / 203 | 682 / 644 | 2048 / 2036 |
| 16 | 297 / 320 | 941 / 1014 | 2978 / 3208 |
| 20 | 383 / 357 | 1217 / 1137 | 3847 / 3596 |
| 30 | 512 / 601 | 1700 / 1894 | 5385 / 6002 |

At the default tol 1e-10 the tables have 1260-3850 intervals (n_q 8-20),
against the stock 100 knots. At 1e-8 they have 400-1900. The conversion takes
0.1-2 s on a laptop and 3-6 s on the Modal host. It is cached, so it runs once
per distinct radial, not once per `lean` call (section 5).

**Energies and forces,** lean(to_spline) against the full analytic model.
Forces see the spline's derivative error, which is O(h^3), so they are ~100x
looser than `tol`, which controls R_nl values.

| tol | intervals (n_q 8-20) | \|dE\|/\|E\| | max\|dF\| / max\|F\| |
|---|---|---|---|
| 1e-8 | 398-1217 | 1e-10 to 3.3e-8 | 5e-8 to 6.7e-7 |
| **1e-10** (default) | 1258-3847 | 1e-13 to 6.6e-10 | **4.1e-9 to 2.3e-8** |

- **Sources:** 1024 atoms on the local CPU for all four benchmark models, n_q 8/12/16/20 and the filled n_q=12 variant (`bench/perf/results/learned_radial_spline_tol_parity_1024.json`).
- **A100 check:** 8192 atoms at 1e-10 agrees (`learned_radial_8192_tol1e-10.json`): dE 7e-13 to 2.2e-10, dF/F 4.2e-9 to 2.2e-8.
- **Force target:** 1e-10 meets ~1e-8 of the largest force on 16 of the 20 local cases. The other 4 reach 1.4-2.3e-8:
  - SiGe_medium n_q 16 and filled;
  - SiGe_large n_q 20 and filled.

  On the A100, 3 cases reach 1.0-2.2e-8.

These rows predate two refinements in section 2b: interval bucketing, and
measuring only the columns A reads. The re-measured table is there.

### 2b. Review refinements: bucketing, derivative error, read columns

Re-measured at 1024 atoms on the local CPU at tol 1e-10, for all four models
and 20 cases (`bench/perf/results/learned_radial_spline_tol_parity_1024_bucketed.json`).

**Measured columns.** The error is now measured on the R_nl columns the A basis
reads (`np.unique(aspec_r)`) only. Columns that `prune_columns` drops no longer
drive the grid, and the splining still happens before the pruning.

**Bucketing.** The interval count is the smallest `splinify.BUCKETS` entry,
round(2^(k/4)) for 32..65536, that meets tol:
- At most 19% more intervals than needed (20% with integer rounding at small n), and on these 20 cases 4-19%, mean 11.5%. The minimal count was found by bisection.
- Buckets are quarter-octave, so two radials needing similar grids share the table shape and the static `rnl_grid`: a radial swap reuses the compiled step instead of retracing. `tests/test_to_spline.py::test_radial_swap_does_not_retrace` counts the traces.
- Why not multiples of 256: that costs 20% at 1258 intervals, and gives ~10 buckets over the 1258-3847 range 1e-10 needs, against 7.
- `spline_intervals=N` on `lean`, `ACECalculator` and `export_lammps` pins the grid.

**The derivative error.** `tol` bounds values. The error in d/dx(env S), which is
what forces see, is O(h^3) and is now measured on the same grid:
- `to_spline(..., return_info=True)` reports it as `max_rel_deriv_err`;
- `deriv_tol` also gates on it.

| | range over the 20 cases |
|---|---|
| intervals chosen | 1218-4096 |
| value error | 4.1e-11 to 8.5e-11 |
| derivative error | 1.8e-8 to 4.0e-8 |
| \|dE\|/\|E\| | 1e-13 to 9.4e-10 |
| max\|dF\| / max\|F\| | 2.5e-9 to 1.9e-8 |

So at the 1e-10 default, the energies agree to up to ~1e-9 relative. The forces
agree to up to ~2.3e-8 of max|F|, counting the earlier measurements, and the
derivative error bounds them from above.

**Tolerance handling:**
- `tol` < 1e-14 (`TOL_FLOOR`) or <= 0 is rejected.
- A float32 model's tol is floored at 10 eps(float32) = 1.2e-6.
- The check grid is processed in 16k-point blocks, which bounds its memory.

**Provenance.** `ACECalculator.splined`, `last_timing["spline_tol"]` and the
bundle's `ace_jax.spline_tol` / `ace_jax.spline_intervals` are read off what
`lean` returned (`eval.model.splining`), looking through a wrapper's `.base`:
- an unfolded analytic model, which `lean` returns as given, records nothing;
- a wrapper whose base was splined records it, and `ace_jax.lean` is True.

  The tolerance was not tightened past 1e-10 for these. Each 10x in forces costs about 1.8x in intervals (h^3).

## 3. Recovered speed

**At the default tol 1e-10** (A100 80GB PCIe, one container,
`learned_radial_8192_tol1e-10.json`), with the full call's ratio to the stock
spline model in brackets:

| model | spline lean | analytic lean (to_spline), n_q 8 / 12 / 16 / 20 | filled n_q=12 |
|---|---|---|---|
| SiGe_medium | 2.10 ms (0.68) | 2.09 / 2.08 / 2.07 / 2.09 ms (0.68) | 2.13 ms (0.69) |
| Cantor_medium | 2.07 ms (0.39) | 2.13 / 2.13 / 2.15 / 2.18 ms (0.40-0.41) | 2.89 ms (0.55) |
| SiGe_large | 4.11 ms (0.64) | n_q 12: 4.09 ms (0.63) | |
| Cantor_large | 2.61 ms (0.67) | n_q 12: 2.62 ms (0.67) | |

The 3x larger tables cost nothing. Every analytic lean runs at the spline lean
form's speed, to within 5.4% (the worst is Cantor_medium n_q=20), and is
1.93-5.8x faster than the full analytic model: 1.93x for SiGe_medium n_q=8,
2.06x for Cantor_large n_q=12, and 5.81x for Cantor_medium n_q=20. The filled
radial is the exception (+40%), since it cannot be compacted.

These timings predate the interval bucketing of section 2b, which makes tables
up to 19% larger. The gather reads 4 rows per edge whatever the table size, and
the 3x step from 1e-8 to 1e-10 above made no measurable difference, so I did
not re-time; the change is expected to be within noise.

**At tol 1e-8** (the first run, A100-SXM4):
- it runs at 0.69-0.74x of the stock spline full call (SiGe) and 0.49-0.59x (Cantor). The spline lean form's own speed is 0.73x and 0.48x, and the analytic lean matches it to within 5%, except Cantor n_q=16 and the filled variant (+22%);
- against the analytic full model it is 1.7-2.6x faster (SiGe_medium) and 2.4-3.9x faster (Cantor_medium);
- against lean_exact it is 1.1-1.8x and 1.6-2.6x faster.

The finer tables do not slow the gather: 4 rows per edge are read whatever
ncoef is. Cantor n_q=16 and the filled variant (0.59x) are the exceptions:
- the filled one cannot be species-compacted, so it keeps the full l-blocks;
- n_q=16 (1014 intervals) is one sample. The other n_q, with 441-1137 intervals, sit at 0.49-0.51x.

**The spline gather itself.** The first A100 run used an explicit per-edge row
gather (`coefs[zi, zj, idx]`) in `_radial_one`. That form was up to 12% slower than
the fused `vmap(spline_eval)(x, coefs[zi, zj])` of #16 (the `:oldgather` rows).
A slice gather and a flat-row gather were then measured at 1-6% slower
(`learned_radial_8192_gather_{slice,flat}.json`).

The code now keeps #16's expression inside an always-jitted
`radial.spline_eval_pairs`. Compiled on CPU, it has the same fusion structure as
#16's inline form: the same multiset of 2627 HLO instructions once names and
source metadata are stripped. The results are bitwise equal. That is not a
text-level identity of the HLO: instruction names and order differ.

So the `:lean` column above slightly overstates the final cost. The final code's
cost is the `:oldgather` row: a12:lean is 2.51 ms (SiGe_medium) and 3.08 ms
(Cantor_medium), the stock spline lean's 2.51 and 3.25.

Why keep it always jitted: the fused expression, run eagerly (e.g. an
un-jitted `model.radial`), materialises the (E, ncoef, n_rnl) per-edge table.
On a 2000-interval table and 20k edges that is 32 GB. A local analysis script
hit exactly this.

## 4. Does compaction apply to real learned radials?

Yes, when the learning started from an ACE1-style radial. Learning keeps the
zero rows exactly zero (`normalise` and `row_active`), and a zero row
tabulates to exact zeros, so `_rnl_owner` sees the per-z_j pattern:
- `tests/test_to_spline.py::test_real_learned_radial_keeps_the_pattern` runs `learn_radial` for 5 L-BFGS steps on `sige_nofit` and si_tiny data. The splined table has the same owner per column as the original spline, and `lean` is compact.
- All 24 real learned `rnl_Wnlq.npy` from the radial-density study (`.worktrees/radial-density/runs/density_vac*`) keep the pattern: Cantor (5, 5, 37, 12) with 185/925 active rows, and SiGe (2, 2, 77, 12) with 154/308.

It does not apply to a radial with every z_j block filled. The pattern check is
done on the data, not assumed. Examples:
- a Python-authored model with `radial_mode="glorot_normal"` (the `build_model` default);
- the `_filled` rows above.

Those still get the spline gather: 0.59-0.71x instead of 0.49-0.71x.

## 5. API

- **`lean(model, spline_tol="auto")`:** applies `to_spline` to a learned analytic R_nl (at 1e-10), then prune, pairfold and block as before. A float splines every analytic radial; `spline_tol=None` never splines.
- **`ACECalculator(..., spline_tol=1e-10)` and `export_lammps(..., spline_tol=1e-10)`:** pass the tolerance to `lean`. The bundle records it as `ace_jax.spline_tol`, or None when nothing was splined; that key is not one lammps-jax reads.
- **The cache.** `to_spline` keeps a content-keyed LRU of 8 converted tables (`splinify.CACHE_SIZE`, `clear_cache()`).
  - **Key:** a sha256 of each radial's Wnlq, its polynomial recursion (A, B, C), the R_nl envelope (the error check reads it; the pair radial has none), n_intervals and tol. It never depends on object identity.
  - **Hits:** a model whose radials are unchanged hits the cache, e.g. after a `calc.model` swap of ctilde, WB, Wpair or E0.
  - **Misses:** any radial edit misses, down to a 1e-12 relative change of Wnlq, and so does a changed tol or envelope. The tests count the fits.
  - **Swap cost:** a readout-only `calc.model` swap of an analytic model (local CPU, `learned_radial_swap_cost.json`) dropped from 72 / 792 / 80 / 348 ms to 2.0 / 9.5 / 2.2 / 6.1 ms (SiGe_medium / Cantor_medium / SiGe_large / Cantor_large).
  - **Not cached:** `prune_columns`, `fold_pair` and `block_dense` cost 0.1-4.4 ms of host time each on these models.
- **`lean_keep_basis(model, spline_tol="auto")`:** only the basis-preserving transforms, `to_spline` and `prune_columns`, never `fold_pair` or `block_dense`. After it, `site_basis`, `site_basis_dense`, `_readout` (WB, Wpair) and the descriptors still work:
  - B matches to roundoff: bitwise on Cantor_small, <= 6e-16 on sige_nofit, where XLA vectorises the pruned-width spline contraction differently;
  - Apair is bitwise.
- **Wrapper models:** `lean(w)` for a model with `.base` and `with_base(new_base)`, e.g. `FSModel(base, ...)`, returns `w.with_base(lean_keep_basis(w.base))`.
- **The learned flag:** `ACEModel.radial_learned` is a static field.
  - It is set by `fit.radial_model.with_radial`. `to_analytic` clears it, since a projected table is not learned, and `widen_radial` keeps it.
  - It is stored as meta_json `radial_learned` by `patch_radial_npz` (hence `radial_learn.save_result`, unless the gate kept `init`), `save_npz` / `eval_pair` and `mark_radial_learned`.
  - It is read by `load`, defaulting to False.
- **Unchanged:** `load` still returns the full model, and fitting keeps the analytic model.

## 6. Accuracy contract and UQ

The splined lean form agrees with the full analytic model to about `tol`, not
to roundoff:
- energies to ~tol relative (at 1e-10: 1e-13 to 7e-10);
- forces to ~100 tol relative to the largest force (at 1e-10: 4e-9 to 2.3e-8).

The exact transforms (prune, pairfold, block) still agree to 1e-12
(`tests/test_lean.py` runs them with `spline_tol=None`).

**For UQ,** take σ (posterior variances, the GP or Laplace ladder) from the
full, unsplined model: it is the model the posterior was built on, and the
basis methods need it anyway. The lean mean then differs from the full mean by
at most ~tol, well below any σ.

Some Julia-parity tests run analytic Julia exports through `ACECalculator` with the defaults: `test_efv`, `test_descriptors`, `test_perf_parity` and `test_python_authoring`. Those exports are not learned, so `"auto"` keeps them analytic and the tests are exact again. `test_to_spline.py` checks that `si_ace_model` and `si_s69` give bitwise the same result as `spline_tol=None`.

Splined at 1e-10, they would miss their references:
- `test_efv` and `test_descriptors`: dE 1.5e-9 against 1e-10;
- `test_perf_parity`: dE 8.8e-9 against 1e-12 of |E| in float64, and dF 2.9e-5 against a 2.5e-5 bound in float32.
