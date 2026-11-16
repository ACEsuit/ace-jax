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
| neighbour list | 21 µs | 23 µs |
| A and AA (once, for all L) | 37 µs | 43 µs |
| A, AA and the features of all 12 (L, parity) | 43 µs | 55 µs |
| on-site H and S blocks of every shell pair | 224 µs | 148 µs |

- The features of every symmetry cost 15 to 25% on top of the A/AA evaluation an ACE potential already does.
- The rest of the last row is the readout, which is still a Python loop over shell pairs in numpy (and is slower with more threads). It has to become one jitted contraction per (L, parity) over all shell pairs and channels.
- The comparison with ACEhamiltonians.jl waits for the off-site model, where its cost is.
