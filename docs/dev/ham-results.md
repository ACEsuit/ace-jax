# ACE Hamiltonians: results

Results of `ace_jax.ham` as they are measured. Design: `specs/2026-11-16-hamiltonians-design.md`.

## Conventions on real data (2026-11-16)

- **Orbital order.** Quoll's DeepH output orders a shell as `deeph.DEEPH_ORDER` (positions in m = −l..l), with no sign change relative to ace-jax's real harmonics. Quoll's canonical ("wiki") convention is ace-jax's.
- **How it was fixed.** An overlap is a two-centre integral, so each L part of an overlap shell block is a multiple of Y_L of the bond direction for l₁+l₂+L even, and zero for odd. On Quoll's SiC regression data (s, p, d on C; s, p, d, f on Si) the largest departure is 7.6e-4 over all 2532 blocks (1.5e-3 bound in the test on the 68-block fixture). That is the data's own integration error: the on-site overlap departs from the identity by 2e-3. A wrong order for any single l gives more than 0.05.
- **Units.** eV and angstrom in Quoll's output (the Si 1s on-site energy is −1791.7 eV).
- **Ingestion.** FHI-aims output through `ham/julia/ingest.jl` and `deeph.read` equals Quoll's reference DeepH conversion to 1.4e-14 (H) and 4.4e-16 (S).

## On-site model on the SiC fixture (2026-11-16)

- Basis: order 2, total degree 7, wL = 1, rcut 4 Å, features to L = 6: 280 A functions, 2780 AA products, 564 features over 12 (L, parity).
- The fitted model returns the on-site H and S blocks of all four atoms to 7.8e-9 eV and 3.1e-12. With two equivalent atoms per species this is interpolation: it checks the path on real data, not the accuracy.
- A basis with wL = 1.5 and total degree 5 misses up to 3.7 eV in d–f parts: its only (3, −1) feature is a product of l = 1 and l = 2 functions, which vanish at a tetrahedral site. An orbital basis with f shells needs single-particle functions to l = 3 at least, and features to L = 6.

## On-site timing, first numbers (2026-11-16)

`bench/ham/onsite_time.py 4`: 512-atom rattled SiC, 28 neighbours, the basis above, orbital basis of the fixture (30 orbitals on Si, 23 on C; 175 coefficient vectors for two channels), float64, lestrade (i9-14900K), JAX on the CPU.

| per atom | 32 threads | 1 thread |
|---|---|---|
| neighbour list (host) | 22 µs | 23 µs |
| A and AA (once, for all L) | 30–39 µs | 43 µs |
| A, AA and the features of all 12 (L, parity) | 51 µs | 54 µs |
| on-site H and S, reference readout (a numpy loop over shell pairs) | 209 µs | 148 µs |
| on-site H and S readout alone (compiled), given the features | 5 µs | 10 µs |
| on-site H and S, `OnsitePredictor`, end to end with the neighbour list | 190–225 µs | 93 µs |

- The features of every symmetry cost about 25% on top of the A/AA evaluation an ACE potential already does.
- The compiled readout of all 175 shell-pair parts and both channels costs 5 to 10 µs per atom: one contraction and one sparse scatter per (L, parity).
- **Open.** On one thread the end-to-end time is the sum of its stages (23 + 54 + 10, measured 93 µs). On 32 threads it is 190 to 225 µs, although the stages timed alone take 22, 42 and 5 µs. Fusing features and readout into one compiled program, or `OMP_WAIT_POLICY=passive`, did not change it. Until this is understood, one thread per process is the faster way to run the on-site path, and the off-site model should be timed both ways from the start.

- The comparison with ACEhamiltonians.jl waits for the off-site model, where its cost is.

## On-site speed against ACEhamiltonians.jl (2026-11-16)

Same structure and orbital basis for both codes: 512-atom diamond silicon rattled by 0.05 Å, cutoff 4 Å (15.9 neighbours), the Si basis of Quoll's SiC data (10 shells, s to f, 30 orbitals, 55 shell pairs), correlation order 2, random coefficients, float64, one thread, lestrade (i9-14900K).

- ACEhamiltonians.jl @ 9ff1572 (Julia 1.10, ACE.jl 0.12.22): `bench/ham/aceh_onsite_time.jl`. One sub-model per shell pair; each distinct (l₁, l₂) timed on 32 atoms with the states already built, weighted by its number of shell pairs.
- ace-jax: `bench/ham/onsite_compare.py`. One A/AA evaluation, features of every (L, parity), compiled readout; the end-to-end time includes the neighbour list.

| | size | on-site H, per atom |
|---|---|---|
| ACEhamiltonians.jl, degree 6 | 3163 basis functions over 55 sub-models | 4547 µs |
| ACEhamiltonians.jl, degree 8 | 6821 | 8439 µs |
| ace-jax, total degree 7 (wL = 1) | 2066 coefficients on 134 shared features | 26 µs (neighbour list 13, features 4, readout 5) |
| ace-jax, total degree 9 | 4710 coefficients on 374 shared features | 44 µs (16, 19, 7) |

- **About 100 to 190 times faster** at a comparable number of fitted coefficients (26 against 4547 µs; 44 against 8439 µs). Adding the overlap channel costs ace-jax 11 to 13 µs more (39 and 55 µs for H and S); ACEhamiltonians.jl needs a second model.
- Where it comes from: ACEhamiltonians.jl evaluates 55 separate ACE bases per atom, at 30 to 760 µs each. Here the per-atom work that depends on the environment (A, AA, features) is done once, and half of the 26 µs is the neighbour list.
- **What this does not show.** The two bases are not the same functions (different radials, and a shared feature set against per-block bases), so equal coefficient counts are not equal accuracy. An accuracy-matched comparison needs fits of both codes on the same data, which is planned for the stage-A fitting work. The ACEhamiltonians.jl environment is its last release on current dependencies, not a tuned build.
- On 32 threads ace-jax's stage sums are the same or lower (21 and 49 µs), but the end-to-end time at total degree 9 is 192 µs: the open item above again. At total degree 7 it is 25 µs.

## Accuracy-matched on-site comparison with ACEhamiltonians.jl (2026-11-16)

**Data and protocol** (the ACEhamiltonians paper's own on-site benchmark, arXiv:2111.13736): aluminium, FHI-aims, 729-atom supercells from 500 K MD, 14 orbitals per atom (3 s, 2 p, 1 d shells; 21 shell pairs). Train on the odd-numbered atoms of BCC `SK-supercell-001` and FCC `SK-supercell-002` (730 atoms), the paper's selection; test on every third atom of FCC `SK-supercell-001` and BCC `SK-supercell-002` (486 atoms). Environment cutoff 9 Å (176 neighbours), the paper's value. H blocks divided by 27 (the files hold a 3×3×3 k-sum). Error: RMSE over the elements of the shell pairs s₁ ≤ s₂. Time: one thread, lestrade.

- ace-jax: `bench/ham/onsite_accuracy.py` (reader `bench/ham/aitb.py`; FHI-aims' phases converted with Quoll's table and checked by the overlap's two-centre form, violation 1e-3 on a scale of 27). Least squares, ridge 1e-10.
- ACEhamiltonians.jl @ 9ff1572: `bench/ham/aceh_onsite_accuracy.jl`, same atoms, its own assembly, mean, scaling and LSQR solver (λ = 1e-7), one sub-model per shell pair.

| model | coefficients | test RMSE (meV) | worst block type | model time per atom |
|---|---|---|---|---|
| ACEhamiltonians.jl, order 2, degree 3 | 241 | 9.88 | dd 21.6 | 1198 µs |
| ACEhamiltonians.jl, order 2, degree 4 | 446 | 1.10 | dd 1.99 | 1736 µs |
| ACEhamiltonians.jl, order 2, degree 6 | 1106 | 0.53 | ss 0.98 | 3491 µs |
| ACEhamiltonians.jl, order 1, degree 12 | 322 | 2.52 | dd 4.22 | 10135 µs |
| ace-jax, order 1, 6 radials, l ≤ 4 | 178 | 4.51 | dd 8.11 | 56 µs |
| ace-jax, order 1, 8 radials | 234 | 1.13 | dd 1.94 | 68 µs |
| ace-jax, order 1, 12 radials | 346 | 0.49 | ss 0.97 | 96 µs |
| ace-jax, order 2, (10, 4) radials, l ≤ 4 | 2276 | 0.56 | ss 1.03 | 112 µs |
| ace-jax, order 2, total degree 10, wL = 1 (no caps) | 2210 | 2.93 | dd 7.59 | 143 µs |

Model time is features plus readout for ace-jax and `predict` on prebuilt states for ACEhamiltonians.jl; neither includes the neighbour search (ace-jax: 11.5 µs per atom with matscipy-neighbours, 355 µs with the ASE fallback that the earlier tables used).

- **At matched accuracy ace-jax is 25 times faster at 1.1 meV (68 against 1736 µs) and 36 times faster at 0.5 meV (96 against 3491 µs)**, with fewer coefficients (234 against 446; 346 against 1106). With the neighbour search included on ace-jax's side only, 22 and 33 times.
- Both codes reach the same floor, 0.97 meV in the s–s blocks, whatever the basis.
- **The basis has to be shaped for the problem.** A plain total-degree basis is poor here: 2.9 meV with 2210 coefficients. The L part of an on-site block is, to leading order, linear in the neighbour density with its own radial profile, so accuracy follows the number of radial functions each l ≤ 4 has at correlation order 1; order-2 terms change nothing on this data (1.13 meV with and without them at 8 radials). `build_site_basis(nmax_by_order=, lmax_by_order=)` gives that shape. Total degree charges high l with too few radials (5 features for the L = 4 part of d–d at total degree 6, and 20 meV left of a 33 meV signal).
- ACEhamiltonians.jl needs order 2 (order 1 stalls at 2.5 meV); ace-jax does not. Not understood: it may be its radial basis or its mean handling.
- One dataset, one element, 500 K crystals. Whether order 2 matters for defects, surfaces or liquids is open.
- On 32 threads ace-jax's feature time at this neighbour count is 253 µs against 68 µs on one thread: the open threading item again.

## Inference time of MACE-H, and where ace-jax's on-site model stands (2026-11-16)

**Not yet like for like.** MACE-H (maurergroup/MACE-H @ 3d6248b, arXiv:2508.15108) predicts every block of the Hamiltonian in one forward pass, on-site and off-site. ace-jax has only the on-site model so far. The MACE-H numbers below are the target the off-site model has to be compared with; the ace-jax numbers are the on-site part alone.

**Input.** Quoll's SiC data tiled to a 3×3×3 supercell (108 atoms) and truncated at 7.2 Å, MACE-H's example cutoff: 159 blocks per atom (`bench/ham/tile_deeph.py`). Orbital basis s to f on Si, s to d on C (30 and 23 orbitals).

**MACE-H** (`bench/ham/maceh_time.py`, its own environment: torch 2.4, e3nn 0.6): the network of its example config (3 blocks, correlation 3, hidden 64x0e+64x1o+64x2e, l_max 4, 1.48 M parameters), untrained weights, float32, forward pass plus conversion to blocks as its `eval` does.

| MACE-H | per structure | per atom | per block |
|---|---|---|---|
| CPU, 1 thread (i9-14900K) | 6772 ms | 62.7 ms | 394 µs |
| GPU (RTX 4000 Ada) | 400 ms | 3.7 ms | 23 µs |
| GPU, the 4-atom cell | 49 ms | 12.3 ms | 78 µs |

A 32-thread CPU run did not finish in 48 minutes (one core busy); it was stopped and is not reported.

**ace-jax, on-site H only** (`bench/ham/onsite_time_deeph.py`): the same 108 atoms, environment cutoff 7.2 Å (158 neighbours), order 1 with l ≤ 6, random coefficients, float64, CPU, one thread.

| ace-jax on-site | per atom, polynomial radials | per atom, tabulated radials |
|---|---|---|
| 8 radials per species (112 features, 2375 coefficients) | 558 µs | 247 µs |
| 12 radials (168 features, 3543 coefficients) | 1220 µs | 347 µs |

- With two species and l ≤ 6 the per-neighbour radial stage dominated (543 of 558 µs); the radial tables ace-jax already has for potentials (`eval.model.with_radial_table`, relative error 1.4e-10 here) cut the feature time from 543 to 227 µs. They should be the default for the Hamiltonian carrier.
- What remains is forming and pooling 784 A functions over 158 neighbours per atom; the readout is 2 µs.
- For orientation only: one on-site block costs ace-jax about 0.25 ms on one CPU thread; MACE-H's average cost per block is 0.39 ms on one CPU thread and 0.023 ms on the GPU, for all blocks at once. The off-site model decides the comparison: it has to deliver about 158 off-site blocks per atom, and if each needs its own environment pooling at this cost it will not beat MACE-H's GPU time. That is the argument for sharing the per-atom features across bonds (design option B) and for a GPU run of ace-jax.
- MACE-H's accuracy is not part of this: the weights are untrained, and its model size here is its example's, not one matched to an error.

## Off-site blocks from shared site features: first accuracy (2026-11-16)

The off-site model (`ham/bond.py`, `ham/offsite.py`): for a bond i→j, two-centre functions φ_n(r) Y_L(r̂) plus the bond harmonics coupled with the per-atom features of i and of j, which the on-site model already computes. No environment is pooled per bond. Bond reversal H_ji(−R) = H_ij(R)ᵀ is exact by construction.

**Data.** The aluminium supercells of the ACEhamiltonians paper, all pairs within 9 Å (168 bonds per atom). Train on 20,000 bonds drawn from BCC `SK-supercell-001` and FCC `SK-supercell-002`, test on 10,000 bonds from FCC `SK-supercell-001` and BCC `SK-supercell-002`. Site basis: order 1, 8 radials, l ≤ 4, cutoff 9 Å. 12 two-centre radials. `bench/ham/offsite_accuracy.py`. RMSE over the elements of the canonical blocks; H in meV, S × 1e3.

| coupled features (radials n_cpl, bond harmonics l ≤ l_cpl) | coefficients per channel | H test (train) | S test | H, d–d | H, bonds < 3.5 Å |
|---|---|---|---|---|---|
| none: two-centre only | 336 | 2.40 (2.44) | 0.0077 | 4.10 | 7.15 |
| 2, l ≤ 2 | 4128 | 1.69 (1.71) | 0.0075 | 3.16 | 3.36 |
| 4, l ≤ 2 | 7920 | 1.51 (1.52) | 0.0073 | 2.80 | 3.09 |
| 4, l ≤ 4 | 13392 | 0.92 (0.94) | 0.0069 | 1.79 | 1.52 |

- The overlap is two-centre and is fitted to 7e-6 by the two-centre functions alone.
- The shared site features carry real environment dependence of H: 2.40 → 0.92 meV overall, and 7.2 → 1.5 meV for nearest-neighbour bonds. Train and test errors agree, so the limit is the basis, not the amount of data.
- The error is still falling with l_cpl and n_cpl; where it stops, and how that compares with ACEhamiltonians.jl's bond-environment model on the same bonds, is not yet measured. The on-site errors on this data were 0.5 to 1.1 meV for comparison.
- Largest single least-squares problem: 652 columns; the fit streams bonds in chunks and takes 37 s.

## Full Hamiltonian inference time against MACE-H (2026-11-16)

Now like for like in what is predicted: every on-site and off-site block of H for the 108-atom SiC input above (159 blocks per atom, cutoff 7.2 Å, s to f orbitals), with untrained weights in both codes. **Not matched in accuracy**: neither model is fitted to this system, and the two model sizes are not known to give the same error.

ace-jax (`bench/ham/offsite_time_deeph.py`): on-site model plus the shared-feature off-site model, compiled (`OnsitePredictor`, `OffsitePredictor`). Site basis order 1, 8 radials per species, l ≤ 6, tabulated radials; 12 two-centre radials. The off-site program contracts the coefficients with the site features once per atom, then does one matrix product per atom over its bonds.

| | per structure | per atom | per block |
|---|---|---|---|
| MACE-H, CPU, 1 thread, float32 | 6772 ms | 62.7 ms | 394 µs |
| ace-jax, CPU, 1 thread, float64, coupled features (2 radials, l ≤ 2) | 1141 ms | 10.6 ms | 67 µs |
| ace-jax, CPU, 1 thread, float64, coupled (4, l ≤ 4) | 1647 ms | 15.3 ms | 96 µs |
| MACE-H, GPU (RTX 4000 Ada), float32 | 400 ms | 3.7 ms | 23 µs |
| ace-jax, GPU, float32, coupled (2, l ≤ 2) | 113–176 ms (5 runs) | 1.05–1.63 ms | 6.6–10.2 µs |
| ace-jax, GPU, float32, coupled (4, l ≤ 2) | 164–168 ms (2 runs) | 1.52–1.56 ms | 9.5–9.8 µs |
| ace-jax, GPU, float32, coupled (4, l ≤ 4) | 148 ms, 831 ms, and one run over 500 s | 1.37 ms, 7.7 ms, – | 8.6 µs, 48 µs, – |

- **CPU, one thread: 4 to 6 times faster than MACE-H.** GPU: 2.3 to 3.5 times faster for the two smaller coupled models.
- **The largest coupled model is erratic on this GPU.** Three runs: 1.37 ms per atom (2.7 times faster than MACE-H), 7.7 ms, and one that did not finish in 500 s. It holds 15 GB of the card's 20 GB (the per-atom coefficient tensors for all atoms at once), so memory pressure is the suspected cause, not confirmed; processing the atoms in chunks is the fix to try.
- On the GPU the time is nearly the same for the two smaller models (113–176 ms) while the on-site program takes 2.5 ms: most of the off-site time is on the host (building the bond list with a Python dictionary for the reverse bonds, and copying 62 MB of blocks back). That is the next thing to remove.
- float64 on this GPU did not finish (its float64 rate is 1/64 of float32); float64 GPU numbers need an A100.
- Three consecutive GPU runs gave 10 to 24 ms per atom while a stray float64 run of mine was still on the GPU; they are excluded.
- Cost by model part: the off-site matrix product does about 100 multiplications per predicted number at (4, l ≤ 4) and 18 at (2, l ≤ 2); the accuracy table above shows what the larger one buys on aluminium (0.92 against 1.69 meV).
