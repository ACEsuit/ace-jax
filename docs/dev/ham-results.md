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

**Data and protocol** (the ACEhamiltonians paper's own on-site benchmark, arXiv:2111.13736): aluminium, FHI-aims, 729-atom supercells from 500 K MD, 14 orbitals per atom (3 s, 2 p, 1 d shells; 21 shell pairs). Train on the odd-numbered atoms of BCC `SK-supercell-001` and FCC `SK-supercell-002` (730 atoms), the paper's selection; test on every third atom of FCC `SK-supercell-001` and BCC `SK-supercell-002` (486 atoms). Environment cutoff 9 Å (176 neighbours), the paper's value. H blocks divided by 27 (the files hold a 3×3×3 k-sum). Error: RMSE over the elements of the shell pairs s₁ ≤ s₂, in meV (the files are in Hartree; an earlier version of this table printed mHa as meV). Time: one thread, lestrade.

- ace-jax: `bench/ham/onsite_accuracy.py` (reader `bench/ham/aitb.py`; FHI-aims' phases converted with Quoll's table and checked by the overlap's two-centre form, violation 1e-3 on a scale of 27). Least squares, ridge 1e-10.
- ACEhamiltonians.jl @ 9ff1572: `bench/ham/aceh_onsite_accuracy.jl`, same atoms, its own assembly, mean, scaling and LSQR solver (λ = 1e-7), one sub-model per shell pair.

| model | coefficients | test RMSE (meV) | worst block type | model time per atom |
|---|---|---|---|---|
| ACEhamiltonians.jl, order 2, degree 3 | 241 | 268.8 | dd 587.8 | 1198 µs |
| ACEhamiltonians.jl, order 2, degree 4 | 446 | 29.9 | dd 54.2 | 1736 µs |
| ACEhamiltonians.jl, order 2, degree 6 | 1106 | 14.4 | ss 26.7 | 3491 µs |
| ACEhamiltonians.jl, order 1, degree 12 | 322 | 68.6 | dd 114.8 | 10135 µs |
| ace-jax, order 1, 6 radials, l ≤ 4 | 178 | 122.7 | dd 220.7 | 56 µs |
| ace-jax, order 1, 8 radials | 234 | 30.7 | dd 52.8 | 68 µs |
| ace-jax, order 1, 12 radials | 346 | 13.3 | ss 26.4 | 96 µs |
| ace-jax, order 2, (10, 4) radials, l ≤ 4 | 2276 | 15.2 | ss 28.0 | 112 µs |
| ace-jax, order 2, total degree 10, wL = 1 (no caps) | 2210 | 79.7 | dd 206.5 | 143 µs |

Model time is features plus readout for ace-jax and `predict` on prebuilt states for ACEhamiltonians.jl; neither includes the neighbour search (ace-jax: 11.5 µs per atom with matscipy-neighbours, 355 µs with the ASE fallback that the earlier tables used).

- **At matched accuracy ace-jax is 25 times faster at 30 meV (68 against 1736 µs) and 36 times faster at 14 meV (96 against 3491 µs)**, with fewer coefficients (234 against 446; 346 against 1106). With the neighbour search included on ace-jax's side only, 22 and 33 times.
- Both codes reach the same floor, 26 meV in the s–s blocks, whatever the basis.
- **The basis has to be shaped for the problem.** A plain total-degree basis is poor here: 79 meV with 2210 coefficients. The L part of an on-site block is, to leading order, linear in the neighbour density with its own radial profile, so accuracy follows the number of radial functions each l ≤ 4 has at correlation order 1; order-2 terms change nothing on this data (31 meV with and without them at 8 radials). `build_site_basis(nmax_by_order=, lmax_by_order=)` gives that shape. Total degree charges high l with too few radials (5 features for the L = 4 part of d–d at total degree 6, and 20 meV left of a 33 meV signal).
- ACEhamiltonians.jl needs order 2 (order 1 stalls at 68 meV); ace-jax does not. Not understood: it may be its radial basis or its mean handling.
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

**Data.** The aluminium supercells of the ACEhamiltonians paper, all pairs within 9 Å (168 bonds per atom). Train on 20,000 bonds drawn from BCC `SK-supercell-001` and FCC `SK-supercell-002`, test on 10,000 bonds from FCC `SK-supercell-001` and BCC `SK-supercell-002`. Site basis: order 1, 8 radials, l ≤ 4, cutoff 9 Å. 12 two-centre radials. `bench/ham/offsite_accuracy.py`. RMSE over the elements of the canonical blocks; H in meV (converted from the files' Hartree; an earlier version printed mHa as meV), S × 1e3.

| coupled features (radials n_cpl, bond harmonics l ≤ l_cpl) | coefficients per channel | H test (train) | S test | H, d–d | H, bonds < 3.5 Å |
|---|---|---|---|---|---|
| none: two-centre only | 336 | 65.3 (66.4) | 0.0077 | 111.6 | 194.6 |
| 2, l ≤ 2 | 4128 | 46.0 (46.5) | 0.0075 | 86.0 | 91.4 |
| 4, l ≤ 2 | 7920 | 41.1 (41.4) | 0.0073 | 76.2 | 84.1 |
| 4, l ≤ 4 | 13392 | 25.0 (25.6) | 0.0069 | 48.7 | 41.4 |

- The overlap is two-centre and is fitted to 7e-6 by the two-centre functions alone.
- The shared site features carry real environment dependence of H: 65 → 25 meV overall, and 196 → 41 meV for nearest-neighbour bonds. Train and test errors agree, so the limit is the basis, not the amount of data.
- The error is still falling with l_cpl and n_cpl; where it stops, and how that compares with ACEhamiltonians.jl's bond-environment model on the same bonds, is not yet measured. The on-site errors on this data were 14 to 30 meV for comparison.
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

## Profile of the full-Hamiltonian prediction on the GPU (2026-11-17)

`bench/ham/offsite_profile.py`: the 108-atom SiC input (158 bonds per atom, s to f orbitals), the small coupled model (2 radials, l ≤ 2; 12 two-centre radials), H only, float32, RTX 4000 Ada.

| stage | where | before | after |
|---|---|---|---|
| site neighbour list | host | 1.0 ms | 1.0 ms |
| site features | device | 2.3 ms | 1.3 ms |
| bond list and reverse-bond index | host | 29.6 ms (a Python dictionary) | 4.8 ms (two sorts) |
| dense (atom, slot) layout | host | 0.2 ms | 0.4 ms |
| off-site blocks | device | 21.5 ms | 20.0 ms (now with the transposed, non-canonical parts) |
| copy the blocks to the host | – | about 190 ms (hidden in the end-to-end time) | 88.5 ms for 192 MB |
| select bonds per species pair, transposes | host | 111.3 ms | 16.9 ms (selection only) |
| end to end | | 354 ms | 123 ms |

- **The model is 21 ms on the device: 0.20 ms per atom, 1.2 µs per block.** With the host-side lists, 27.5 ms (0.25 ms per atom), against MACE-H's 400 ms (3.7 ms per atom) on the same GPU: about 15 times faster if the blocks stay on the device.
- **Two thirds of the remaining end-to-end time is copying the blocks back to the host** (192 MB of float32). A consumer that works on the device (Bloch sums, eigenvalues) does not pay it: `OffsitePredictor.device` returns the blocks there. MACE-H's 400 ms includes its `.cpu()` copy of the same blocks.
- Changes made from the profile: the reverse-bond index by sorting rather than a dictionary, and the non-canonical blocks (transposes of the reverse bond) formed on the device.
- Still wasteful: each ordered species pair is computed on every (atom, slot) and masked afterwards, so a two-species system does about twice the necessary work.

## Positions to eigenvalues on the device (2026-11-17)

`ham/operator.py::HamiltonianPredictor` keeps the predicted blocks on the device, sums them into H(k) and S(k), and solves H(k) c = e S(k) c there (Cholesky of S, then a Hermitian eigensolver). Against a host assembly of the reference predictions and SciPy, the Bloch matrices agree to 1e-11 and the eigenvalues to 1e-9 (`tests/test_ham_operator.py`).

`bench/ham/bands_time.py`, the 108-atom SiC input (2862 orbitals, 158 bonds per atom), H and S, small coupled model, float32, RTX 4000 Ada, one k-point:

| stage | time | share |
|---|---|---|
| blocks of H and S on the device, with the host-side lists | 55.7 ms | 22 % |
| Bloch sum H(k) | 3.2 ms | 1 % |
| Bloch sum S(k) | 3.2 ms | 1 % |
| eigenvalues of a 2862 × 2862 Hermitian matrix, copied to the host | 186.3 ms | 75 % |
| positions to eigenvalues | 248 ms | 2.3 ms per atom |

- The 88 ms copy of the blocks to the host is gone; the only download is the eigenvalues.
- The eigensolver is now three quarters of the time, and it grows as the cube of the orbital count while everything else is linear in the atom count.
- Timed on H(k) alone: with random coefficients the overlap is not positive definite, so the generalised solve (one more Cholesky factor and two triangular solves) is not in this number.
- float32 throughout. Core levels near −1800 eV carry about 0.1 eV of float32 rounding in the eigenvalues; float64 on this card is 1/64 the speed, so an accurate GPU spectrum needs an A100 or a core/valence split.

## Neighbour lists on the device: no host-to-device transfer beyond the inputs (2026-11-17)

`HamiltonianPredictor.prepare_device` builds the site and bond lists on the GPU with matscipy-neighbours' CUDA backend (`eval.nlist.dense_graph(device="cuda")`) and derives everything else there: the image index of each bond, the reverse-bond slot (two sorts), the species masks and the index lists of the Bloch sum.

- **Check.** `tests/test_ham_operator.py::test_device_path_uploads_nothing_but_its_inputs` runs positions → blocks → H(k), S(k) → eigenvalues under `jax.transfer_guard_host_to_device("disallow")` after the compiling call. It passes; the host-list path (`prepare`) raises under the same guard. The uploads that remain are explicit and are the inputs: positions, cell and species (and the neighbour-list library takes the positions itself). One scalar comes down per list (the largest neighbour count, to size the slots).
- **Same result.** H(k) and S(k) from the device lists equal the host-list path's to 1e-10, and every bond finds its reverse.
- **Build.** The PyPI wheels of matscipy-neighbours are CPU-only. On lestrade: a CUDA 12.9 toolchain from conda-forge (`micromamba create -p /storage/eng/essswb/venvs/cuda-build -c conda-forge cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-cccl cuda-version=12.9`), then, with its `bin` on `PATH` and `CUDA_HOME` set, `uv pip install --reinstall --no-binary matscipy-neighbours -C cmake.define.ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=89 matscipy-neighbours==1.1.0`. A later `uv sync` puts the CPU wheel back.
- **Time.** No gain at this size: blocks of H and S take 58–60 ms with device lists against 56–67 ms with host lists (108-atom SiC, float32); positions to eigenvalues 251–254 ms either way, three quarters of it the eigensolver. The first version was slower (127 ms) because the slot count doubled from 64 to 256 where 158 are occupied; the lists are now trimmed to the occupied slots in steps of 16. What the device lists buy is a pipeline with nothing large on the host, which matters when the structure changes every call (MD) and for larger systems, not a faster single call here.
- Two things had to be made explicit to pass the guard: the padding vector in `dense_graph` (now an explicit upload) and the trimming slice (compiled, because an eager slice uploads its bounds).

## Eigenvalue error on a held-out aluminium supercell (2026-11-17)

What matters for use is the spectrum, not the blocks. `bench/ham/eigen_accuracy.py`: fit on-site and off-site H and S on BCC `SK-supercell-001` and FCC `SK-supercell-002`, predict the full matrices of FCC `SK-supercell-001` (729 atoms, 10206 orbitals, the Γ point of the supercell), solve H c = e S c and compare with the reference spectrum state by state. Site basis: order 1, 12 radials, l ≤ 4; bond cutoff 9 Å; 20,000 training bonds. Run on mnf148 (the job needs about 12 GB).

| | block RMSE of H inside the cutoff | valence window (E_F − 15 … E_F + 3 eV, 1554 states): MAE / RMSE / max | core (3645 states): MAE / max | all occupied (4739): MAE |
|---|---|---|---|---|
| reference truncated at 9 Å (the floor for this cutoff) | 0 | 0.8 / 1.1 / 3.1 meV | 0.0 / 0.0 meV | 0.12 meV |
| two-centre features only | 55.8 meV | 3954 / 4406 / 43,920 meV | 5.4 / 87 meV | 861 meV |
| with coupled features (4 radials, l ≤ 4) | 20.5 meV | 143 / 188 / 521 meV | 5.5 / 87 meV | 24.9 meV |

- **The model is not accurate enough in the quantity that matters.** A 20 meV block error becomes a 143 meV mean error in the valence eigenvalues, seven times larger; the two-centre model's 56 meV becomes 4 eV.
- The cutoff is not the cause: truncating the reference itself at 9 Å costs 0.8 meV.
- The overlap is not the cause: with the reference S in place of the predicted one the numbers do not change (S is fitted to 5e-6).
- Block errors do not average out in the spectrum. The overlap matrix of this basis has eigenvalues down to 0.03, so the generalised problem amplifies errors of H along nearly dependent orbital combinations, and the errors of the ~180 bonds of an atom are correlated, not independent. A block RMSE is the wrong target for these two reasons.
- Core levels are far better (5 meV): they depend on a few on-site elements.
- ACEhamiltonians.jl's off-site model on the same bonds reaches block errors of 58, 33 and 18 meV at order 1, degrees 4, 6 and 8 (4.1, 14.8 and 39.7 ms per bond), and 51 and 32 meV at order 2, degrees 4 and 6 (11 and 230 ms per bond). Its eigenvalue error is not measured here; at 18–33 meV of block error there is no reason to expect it to be better than the table above.

## Against the 2022 ACEhamiltonians paper: band error on FCC and BCC aluminium (2026-11-17)

**The target** (arXiv:2111.13736, npj Comput. Mater. 2022). Its optimised model (on-site H order 2, degree 9, cutoff 10 Å; off-site H order 1, bond cutoff 10 Å, environment cutoff 5 Å, degrees 9 to 14 per sub-block; overlap order 0, degree 16) gives band structures "with a RMSE of less than 0.4 eV for both phases"; the on-site d–d sub-block has an RMSE of about 10 meV, its largest; the off-site overlap an RMSE of 1e-4. The band error is the paper's `band_error` (its `test/plots.jl`): the RMS over k and over bands 2 to 14 of e·f(e) − e_ref·f(e_ref), f the Fermi function at the DFT Fermi level with σ = 0.086 eV, normalised by all 14 bands; the 1s band is left out. Unoccupied bands therefore count for nothing, and occupied bands count by their full value.

**Same test here** (`bench/ham/band_accuracy.py`): train on the 500 K MD supercells as above; test on the equilibrium FCC and BCC crystals of the paper's `reference_data` (not in the training data, and at lattice constants 4.00 and 3.24 Å, about 1.3 % below the MD cells'), primitive-cell bands along the paper's k-paths from one atom's row of the 9×9×9 supercell matrices.

| ace-jax model (bond cutoff 9 Å, 20,000 training bonds) | FCC, paper's band error | BCC, paper's band error | valence bands to E_F + 10 eV, plain RMSE (FCC / BCC) |
|---|---|---|---|
| reference truncated at 9 Å (floor) | 0.0015 eV | 0.0065 eV | 4.6 / 12.6 meV |
| coupled features (4 radials, l ≤ 4) | **0.219 eV** | **0.711 eV** | 959 / 973 meV |
| paper's optimised model | < 0.4 eV | < 0.4 eV | not given |

- **FCC is inside the paper's target, BCC is not** (0.71 against 0.4 eV).
- The plain RMSE of the valence and low conduction bands is about 1 eV with errors up to 4.8 eV: the unoccupied bands are poor. The paper's metric does not see them; its Fig. 5 claims agreement to 10 eV above the Fermi level, which this model does not have.
- The reference files here are not multiplied by 27, unlike the MD files (their on-site overlap is 1); the script reads the factor from the data.

**What drives the spectrum error** (first-order attribution on the 729-atom MD supercell, `eigen_accuracy.py … diagnose`, valence window, exact MAE 143 meV):

| part of ΔH | block RMSE | first-order eigenvalue MAE |
|---|---|---|
| on-site blocks | 10.5 meV | 5.0 meV |
| off-site, bonds < 3.5 Å | 28.3 meV | 5.8 meV |
| off-site, 3.5–5.5 Å | 43.4 meV | 48.3 meV |
| off-site, 5.5–9 Å | 8.0 meV | 20.5 meV |
| pairs beyond the cutoff | 0.06 meV | 0.9 meV |
| p–d blocks | 10.8 meV | 43.2 meV |
| d–d blocks | 23.0 meV | 28.2 meV |
| p–p blocks | 2.8 meV | 22.1 meV |

- The second and third neighbour shells (3.5–5.5 Å) carry most of the eigenvalue error, and the long-range blocks matter far more than their small block error suggests (8 meV of block error, 20 meV in the eigenvalues): many bonds, coherent errors.
- In the orthonormalised basis ΔH has an RMS element of 47 meV against 10 meV in the orbital basis, and an RMS diagonal of 590 meV. The near-dependence of the basis (overlap eigenvalues down to 0.03) amplifies the error, as suspected.
- First order explains only part: its MAE is 59 meV against the exact 143 meV, and 139 meV of the exact error is a rigid downward shift of the window. The error is not small on the scale of the level spacing of a 10206-orbital supercell.

**Model variants on the paper's metric** (same test; bond cutoff 9 Å and 20,000 training bonds unless said):

| variant | FCC | BCC |
|---|---|---|
| two-centre features only | 2.41 eV | 1.79 eV |
| coupled, 4 radials, l ≤ 4 (the model above) | 0.219 eV | 0.711 eV |
| coupled, 6 radials, l ≤ 4 | 0.235 eV | 0.587 eV |
| coupled, 4 radials; bond cutoff 10 Å, 16 two-centre radials | 0.539 eV | 0.999 eV |
| coupled, 4 radials; 60,000 training bonds | 0.216 eV | 0.801 eV |

No variant reaches 0.4 eV on BCC. More coupled radials help a little, more data does not, and a longer cutoff makes it worse.

**Where the band error comes from** (reference blocks swapped in for one part at a time, reference S):

| H used | FCC: paper's error / plain RMSE to E_F + 10 eV | BCC |
|---|---|---|
| predicted on-site, reference off-site | 0.007 eV / 11 meV | 0.097 eV / 55 meV |
| reference on-site, predicted off-site | 0.219 eV / 965 meV | 0.672 eV / 971 meV |
| predicted only for bonds < 3.5 Å | 0.012 eV / 66 meV | 0.162 eV / 240 meV |
| predicted only for bonds 3.5–5.5 Å | 0.171 eV / 801 meV | 0.233 eV / 872 meV |
| predicted only for bonds > 5.5 Å | 0.157 eV / 303 meV | 0.125 eV / 404 meV |

- **The on-site model is good enough**: with the reference off-site blocks the bands are right to 11 meV (FCC) and 55 meV (BCC).
- **The whole problem is the off-site model, at medium and long range.** Nearest-neighbour bonds are fine in FCC; the second and third shells (3.5–5.5 Å) and the long bonds each cost about as much as the paper's whole budget.
- This is the regime the shared-feature design is weakest in by construction: for a bond longer than a few ångström the environment that matters lies between and around the two atoms, and features centred on each atom separately, summed, do not describe it. ACEhamiltonians' bond-centred cylinder does.
