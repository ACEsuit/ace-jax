# wL ladder: angular weight vs accuracy and cost (Si accuracy-cost benchmark)

Status: designed, **not launched**. Run with `run.sh` (this directory).

## Motivation

`aj fit --wL W` sets the TotalDegree level of a one-particle function to
`n + W*l` (Si, one species), and a product basis function is kept when the
sum of its factors' levels is `<= max_degree`. The default is `W = 1.5`, and
every existing linear-ACE point (`sweep.yaml`, family `ace`) uses it. A larger
`W` makes angular degree more expensive: at a fixed basis size `n_B` the basis
trades high-`l` angular channels for more radial channels (`n`), a larger
`max_degree` and so a longer pair basis. Prior experience (user): raising
`wL` can improve angle-dependent interactions while keeping cost from
growing too fast. This ladder tests that at matched `n_B`, and separately at
matched cost.

## What the basis sizes show (computed, `build_model([14], order, md, wL=W, rcut=5.0)`, CPU)

The key fact, which the fit times and the existing timings make important:
**at a fixed `n_B` the product-basis size `n_AA` (and `n_A`) falls steeply
with `wL`**. Inference cost follows `n_AA` far more than `n_B` (A100,
spline-deployed, µs/atom-step: o4d14 `n_AA`=2956: 0.67; o4d16 8638: 1.21;
o4d20 64187: 5.66). Tier T4 (`n_B` ~1.3-1.6k) spans `n_AA` 4.7k (wL 2.5) to
16.7k (wL 1.0), a 3.6x range. At the top end, `o4d21_wL2` has `n_B` 3817 with
`n_AA` 25k, against the baseline `o4d20` 5456 / 64k.

Other observations:

- **Max l collapses at high wL.** The 3-body (order-2 product) terms need
  `l1 = l2`, so their max l is `floor((md/2 - 1)/wL)`. wL 2.5 keeps l <= 2
  everywhere up to `n_B` 662 (o4d16), and wL 2.0 up to `n_B` 342 (o4d13).
  For diamond Si, where the tetrahedral angle is picked out by l = 3
  (`P_3(-1/3)` is large), this is the most likely failure mode of high wL.
- **Coarse steps at low wL.** At wL 1.0 each `max_degree` step multiplies
  `n_B` by ~1.55-1.6 (252, 398, 637, 990, 1546, then 3603 at md 16); at wL
  2.0-2.5 the step is ~1.3. Matched-size points at wL 1.0 are therefore
  further from the tier target.
- **Confound: `max_degree` sets the pair basis too.** `pair_maxn =
  max_degree`, so at matched `n_B` a higher-wL point has a few more pair
  radials (9-21) and a higher radial `n` (`nmax = max_degree`). `n_params =
  n_B + max_degree`. This is small in count but is part of what "trading
  angular for radial" means here.
- **Order 5 at a fixed `max_degree` only appends a 5-body layer**
  (`aa_lens` of o5dX equals o4dX plus one block), so the o4/o5 matched-`n_B`
  pairs differ in both order and `max_degree`.

Full probe output (54 points, wL 1.0-2.5, orders 3-5) is reproduced in the
appendix.

## The ladder (23 core fits + 2 optional)

Tiers match the existing wL 1.5 points: T1 ~110 (o3d10), T2 ~340 (o4d12),
T3 ~710 (o4d14), T4 ~1445 (o4d16), T5 a large point below o4d20. The O5
rows are order-5 points matched in `n_B` to an order-4 point at the same wL
(pairs listed under H4). wL 1.5 order 4 is the existing ladder
(`runs/accuracy_cost/ace/<size>`), not refitted.

Columns: `n_B` from the coupling library; `n_par = n_B + max_degree`;
`l max` overall and per product order (2-body/3-body/4-body[/5-body] terms,
i.e. aa orders 2..order); `n max (l>=1)` is the largest radial index on an
angular channel; `cpb` is `--configs-per-batch` (memory only); est. fit time
in minutes on lestrade's RTX 4000 Ada.

| tier | tag | wL | order | md | n_B | n_par | n_A | n_AA | l max | l max by order | n max | n max (l>=1) | cpb | est. min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | o3d9_wL1 | 1 | 3 | 9 | 111 | 120 | 61 | 334 | 3 | 3/3 | 9 | 6 | - | 2 |
| T1 | o3d11_wL2 | 2 | 3 | 11 | 120 | 131 | 39 | 207 | 2 | 2/2 | 11 | 6 | - | 2 |
| T2 | o4d13_wL2 | 2 | 4 | 13 | 342 | 355 | 57 | 748 | 2 | 2/2/2 | 13 | 8 | - | 4 |
| T2 | o4d13_wL1.75 | 1.75 | 4 | 13 | 373 | 386 | 69 | 971 | 3 | 3/2/2 | 13 | 8 | - | 5 |
| T2 | o4d14_wL2.5 | 2.5 | 4 | 14 | 373 | 387 | 53 | 681 | 2 | 2/2/2 | 14 | 8 | - | 5 |
| T2 | o4d11_wL1 | 1 | 4 | 11 | 398 | 409 | 111 | 2014 | 4 | 4/4/3 | 11 | 8 | - | 6 |
| T2 | o4d12_wL1.25 | 1.25 | 4 | 12 | 399 | 411 | 96 | 1516 | 4 | 4/3/3 | 12 | 8 | - | 6 |
| T3 | o4d13_wL1.25 | 1.25 | 4 | 13 | 598 | 611 | 121 | 2797 | 4 | 4/4/3 | 13 | 9 | - | 15 |
| O5 | o5d14_wL2 | 2 | 5 | 14 | 626 | 640 | 73 | 1511 | 3 | 3/2/2/2 | 14 | 9 | - | 21 |
| T3 | o4d12_wL1 | 1 | 4 | 12 | 637 | 649 | 147 | 4192 | 5 | 5/4/4 | 12 | 9 | - | 17 |
| O5 | o5d13_wL1.5 | 1.5 | 5 | 13 | 638 | 651 | 91 | 2150 | 3 | 3/3/3/2 | 13 | 9 | - | 22 |
| T3 | o4d15_wL2 | 2 | 4 | 15 | 649 | 664 | 89 | 1838 | 3 | 3/3/2 | 15 | 10 | - | 17 |
| T3 | o4d16_wL2.5 | 2.5 | 4 | 16 | 662 | 678 | 71 | 1504 | 2 | 2/2/2 | 16 | 10 | - | 18 |
| O5 | o5d15_wL2.5 | 2.5 | 5 | 15 | 672 | 687 | 62 | 1320 | 2 | 2/2/2/2 | 15 | 9 | - | 24 |
| T3 | o4d15_wL1.75 | 1.75 | 4 | 15 | 735 | 750 | 101 | 2567 | 3 | 3/3/3 | 15 | 10 | - | 22 |
| T4 | o4d15_wL1.25 | 1.25 | 4 | 15 | 1317 | 1332 | 182 | 9654 | 5 | 5/4/4 | 15 | 11 | 2 | 41 |
| T4 | o4d17_wL1.75 | 1.75 | 4 | 17 | 1405 | 1422 | 151 | 6631 | 4 | 4/4/3 | 17 | 12 | 2 | 44 |
| T4 | o4d19_wL2.5 | 2.5 | 4 | 19 | 1492 | 1511 | 119 | 4690 | 3 | 3/3/3 | 19 | 13 | 2 | 45 |
| T4 | o4d14_wL1 | 1 | 4 | 14 | 1546 | 1560 | 232 | 16717 | 6 | 6/5/5 | 14 | 11 | 2 | 45 |
| T4 | o4d18_wL2 | 2 | 4 | 18 | 1624 | 1642 | 146 | 7037 | 4 | 4/3/3 | 18 | 13 | 2 | 46 |
| O5 | o5d18_wL2.5 | 2.5 | 5 | 18 | 1693 | 1711 | 103 | 4636 | 3 | 3/3/2/2 | 18 | 12 | 2 | 60 |
| O5 | o5d17_wL2 | 2 | 5 | 17 | 1764 | 1781 | 121 | 6484 | 3 | 3/3/3/3 | 17 | 12 | 2 | 60 |
| T5 | o4d21_wL2 | 2 | 4 | 21 | 3817 | 3838 | 221 | 24988 | 4 | 4/4/4 | 21 | 16 | 1 | 52 |
| opt | o4d20_wL1.75 | 1.75 | 4 | 20 | 3557 | 3577 | 237 | 27208 | 5 | 5/4/4 | 20 | 15 | 1 | 52 |
| opt | o4d16_wL1 | 1 | 4 | 16 | 3603 | 3619 | 345 | 61178 | 7 | 7/6/6 | 16 | 13 | 1 | 52 |

Baseline (wL 1.5, existing fits) for reference:

| size | n_B | n_A | n_AA | l max by order | n max (l>=1) | fit (lestrade) |
|---|---|---|---|---|---|---|
| o3d8 | 54 | 25 | 84 | 2/1 | 4 | 84 s |
| o3d10 | 110 | 43 | 230 | 2/2 | 6 | 105 s |
| o3d12 | 211 | 75 | 596 | 3/3 | 8 | 170 s |
| o4d12 | 338 | 75 | 975 | 3/3/2 | 8 | 259 s |
| o4d14 | 710 | 116 | 2956 | 4/3/3 | 10 | 1252 s |
| o4d16 | 1445 | 166 | 8638 | 4/4/4 | 12 | 8126 s (cpb 2; contended) |
| o4d20 | 5456 | 323 | 64187 | 6/5/5 | 16 | 3199 s (cpb 1) |

### Fit-time estimate

Log-log interpolation in `n_B` between the measured anchors (54: 1.4 min,
110: 1.75, 211: 2.8, 338: 4.3, 710: 21, 1445: 45 (taken as typical, not the
contended 135), 5456: 55), times 1.3 for order 5. **Core total ~580 min,
about 10 GPU-hours** (realistic range 7-15 h; T4/O5/T5 are 80% of it). The
two optional points add ~2 h. The time is not a function of `n_B` alone:
the jump from o4d12 (4 min) to o4d14 (21 min) and the flat o4d16 -> o4d20
show the batch size and the box load matter as much. The wL 1.0 points
carry 2-4x the baseline's `n_AA` at the same `n_B`, so expect them to run
slow and to need the most GPU memory (`o4d14_wL1`: n_AA 16.7k, about
o4d17-at-1.5 size; drop it to cpb 1 if it runs out of memory).

### Recommended subset if time is short (~12 fits, ~5 h)

T3 and T4 at wL 1.0, 2.0, 2.5 (6 fits: the extremes and the most likely
winner), plus the O5 pairs at wL 2.0 (`o5d14_wL2`, `o5d17_wL2`) and the
l<=2 control `o4d13_wL2` (T2), plus `o4d15_wL1.75`, `o4d17_wL1.75` and
`o4d21_wL2` (T5). That keeps the matched-`n_B` comparison at two sizes, the
max-l collapse test, the order-5 test and one large point that can be set
against o4d20 at a third of its `n_AA`.

## Hypotheses and where they should show

- **H1 (cost).** At matched `n_B`, higher wL is cheaper at inference
  (`n_AA` 2-3.6x lower at T4). Measure: µs/atom-step with
  `accuracy_cost.timing` (lestrade GPU/CPU, Sulis A100), against `n_AA`.
  Expect the accuracy-vs-cost Pareto front to shift toward wL 1.75-2.0 even
  if accuracy at matched `n_B` is unchanged.
- **H2 (accuracy at matched `n_B`).** Test force RMSE is flat or slightly
  better for wL 1.75-2.0 than 1.5 (more radial channels per angular
  channel, plus the higher-`n` pair), and worse at wL 1.0 at large `n_B` for
  the same reason. The angle-dependent QoIs move the most: C44,
  dE_bc8 / dE_st12 / dE_beta-Sn / dE_sh, gamma_us glide and shuffle,
  phonon RMS, and the three interstitials (the current ACE weak spot: -10%
  to -24% at o4d16/o4d20). The user's prior: these improve with wL at a
  fixed cost.
- **H3 (an l = 3 floor).** Points whose 3-body l max is 2 (`o3d11_wL2`,
  `o4d13_wL2`, `o4d14_wL2.5`, `o4d16_wL2.5`, `o5d15_wL2.5`) are worse on
  C44, bc8, st12, the interstitials and gamma_us than the same tier at
  wL <= 1.75, even when force RMSE is close. `o4d19_wL2.5` (l max 3) should
  recover. If it does, the useful range of wL is bounded by keeping l >= 3
  (for Si, wL <= ~(md/2 - 1)/3).
- **H4 (order 5 at higher wL).** Higher wL makes order 5 affordable. Matched
  pairs (same wL, `n_B` within ~10%, order 5 vs order 4):
  `o5d13_wL1.5` (638) vs `ace/o4d14` (710); `o5d14_wL2` (626) vs
  `o4d15_wL2` (649); `o5d15_wL2.5` (672) vs `o4d16_wL2.5` (662);
  `o5d17_wL2` (1764) vs `o4d18_wL2` (1624); `o5d18_wL2.5` (1693) vs
  `o4d19_wL2.5` (1492). Expect gains on the defects (vacancy,
  interstitials), the high-energy phases (hcp, fcc, bcc) and decohesion,
  which have many-neighbour environments, and little change in elastic
  constants.

## Analysis plan

Inputs per fit: `metrics.csv` (test E/F/V RMSE, from `--test`),
`status.json` (wall time), `model.npz`; QoIs via
`python bench/si_qoi/compare.py MODEL ...` and `score.py MODEL ... --json`.
Baseline wL 1.5 QoIs exist for o4d16 and o4d20 only; run them for o3d10,
o4d12 and o4d14 too so every tier has its baseline.

1. Test force RMSE (and energy, virial) vs `n_B`, one line per wL (1.5 from
   the existing ladder), log x. Order-5 points as open markers at their wL.
2. The same vs `n_AA` (cost proxy), then vs measured µs/atom-step once
   timed. This is the plot that tests the user's claim.
3. QoI score (score.py geometric mean and fraction within tolerance) vs
   `n_B` and vs cost, one line per wL.
4. Per-QoI heatmap at T3 and T4: rows wL (1.0 ... 2.5), columns the 21
   scored QoIs grouped as angle-dependent (C44, bc8, st12, beta-Sn, sh,
   hex-diamond, gamma_us x2, phonons, 3 interstitials) vs other (B, C11,
   C12, hcp, fcc, bcc, vacancy, surface, decohesion); cell = normalised
   error minus the wL 1.5 baseline at the same tier (signed, diverging
   colour map).
5. Order-5 vs order-4 paired deltas (H4): one bar per QoI per pair.
6. l max annotation: mark the l<=2 points on plots 1-3 (H3).

## Follow-ups (not done here)

- To time these with `accuracy_cost.timing` and plot them with `plot.py`,
  add a `ace_wl` family to `sweep.yaml` and pass `--wL` in
  `fit_ace.command` (`*(["--wL", str(e["wL"])] if "wL" in e else [])`).
  `run.sh` writes the same run-directory layout (`runs/accuracy_cost/ace_wl/<tag>`,
  `status.json`, `model.npz`) so that family can adopt the fits. The
  `n_params` field is not written by `run.sh`; it is `n_B + max_degree`.
- If H2/H3 hold, try wL 1.75-2.0 with `--learn-radial` (family `ace_lr`):
  the learned radials make the extra radial channels cheaper still.

## Appendix: all probed points

`wL order md: n_B n_A n_AA l-max-by-order n-max(l>=1)`

```
1.0 3 9: 111 61 334 3/3 6          1.75 3 10: 92 35 155 2/2 5
1.0 4 10: 252 86 987 4/3/3 7       1.75 4 12: 263 53 589 2/2/2 7
1.0 4 11: 398 111 2014 4/4/3 8     1.75 4 13: 373 69 971 3/2/2 8
1.0 4 12: 637 147 4192 5/4/4 9     1.75 4 14: 525 85 1574 3/3/2 9
1.0 4 13: 990 183 8371 5/5/4 10    1.75 4 15: 735 101 2567 3/3/3 10
1.0 4 14: 1546 232 16717 6/5/5 11  1.75 4 16: 1019 126 4127 4/3/3 11
1.0 4 16: 3603 345 61178 7/6/6 13  1.75 4 17: 1405 151 6631 4/4/3 12
1.25 3 10: 121 55 305 3/2 6        1.75 4 20: 3557 237 27208 5/4/4 15
1.25 4 11: 264 71 821 3/3/2 7      1.75 5 14: 694 85 2036 3/3/2/2 9
1.25 4 12: 399 96 1516 4/3/3 8     1.75 5 15: 1007 101 3442 3/3/3/2 10
1.25 4 13: 598 121 2797 4/4/3 9    2.0 3 11: 120 39 207 2/2 6
1.25 4 14: 892 146 5257 4/4/4 10   2.0 4 12: 247 48 475 2/2/2 7
1.25 4 15: 1317 182 9654 5/4/4 11  2.0 4 13: 342 57 748 2/2/2 8
1.25 4 16: 1929 218 17295 5/5/4 12 2.0 4 15: 649 89 1838 3/3/2 10
1.25 4 18: 4021 316 52379 6/6/5 14 2.0 4 16: 890 105 2904 3/3/3 11
1.5 5 13: 638 91 2150 3/3/3/2 9    2.0 4 17: 1202 121 4526 3/3/3 12
1.5 5 14: 959 116 3898 4/3/3/3 10  2.0 4 18: 1624 146 7037 4/3/3 13
2.5 3 11: 104 26 148 1/1 5         2.0 4 21: 3817 221 24988 4/4/4 16
2.5 3 12: 136 35 208 2/1 6         2.0 5 14: 626 73 1511 3/2/2/2 9
2.5 4 14: 373 53 681 2/2/2 8       2.0 5 15: 887 89 2447 3/3/2/2 10
2.5 4 15: 498 62 1014 2/2/2 9      2.0 5 16: 1255 105 3984 3/3/3/2 11
2.5 4 16: 662 71 1504 2/2/2 10     2.0 5 17: 1764 121 6484 3/3/3/3 12
2.5 4 17: 872 87 2204 3/2/2 11     2.5 5 15: 672 62 1320 2/2/2/2 9
2.5 4 18: 1144 103 3212 3/3/2 12   2.5 5 16: 919 71 2016 2/2/2/2 10
2.5 4 19: 1492 119 4690 3/3/3 13   2.5 5 17: 1250 87 3065 3/2/2/2 11
                                   2.5 5 18: 1693 103 4636 3/3/2/2 12
```

The coupling shapes for all of these are now in `~/.cache/ace-jax/coupling`,
so the fits will not call the coupling library again.
