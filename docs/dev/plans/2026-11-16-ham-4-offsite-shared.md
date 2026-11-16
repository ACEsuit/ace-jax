# Hamiltonians 4: off-site blocks from shared site features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Off-site blocks H_ij(R), S_ij(R) predicted from the per-atom features the on-site model already computes plus a function of the bond vector, at a cost per bond of a small contraction, with exact bond-reversal symmetry.

**Architecture:** For a directed bond i→j with vector r, the features of symmetry (L, p) are (a) the two-centre functions φ_n(r) Y_L(r̂), and (b) the bond harmonics coupled with a site feature of i or of j, [φ_n(r) Y_l(r̂) ⊗ B^(λ,π)]^L with the real CG tensors. A shell pair reads the features of parity p = (−1)^(l₁+l₂), one coefficient vector per (species pair, shell pair, L, channel), as on-site. Only canonical blocks are predicted ((z_i, s₁) ≤ (z_j, s₂)); the others are the transpose of the reverse bond's block; same-species same-shell blocks use features symmetrised under i ↔ j.

**Tech Stack:** numpy first (reference implementation and fit), then one jitted program; `ace_jax.ham` only.

**Spec:** `docs/dev/specs/2026-11-16-hamiltonians-design.md` §3 (provider B), reordered before provider A by the owner on 2026-11-16 after the MACE-H timing (`docs/dev/ham-results.md`).

## Global Constraints

- Site features are computed once per atom and shared by the on-site model and every bond.
- H_ji(−R) = H_ij(R)ᵀ exactly, by construction, for every channel.
- Parity of a block is (−1)^(l₁+l₂); features of the other parity are never used for it.
- The accuracy of this model is unproven: every accuracy statement comes from a fit on real data with a held-out test set, compared with ACEhamiltonians.jl's off-site model on the same bonds.
- Timings on one CPU thread and on the GPU, per atom and per block, on the SiC input MACE-H was timed on.

## Review Focus

1. A bond and its reverse in different periodic images (R ≠ 0, i = j included): reversal symmetry must hold for self-image bonds too.
2. A bond longer than the site cutoff but inside the bond cutoff: features must not assume j is a neighbour in i's environment.
3. Same-species same-shell blocks: must be symmetric under the swap with the transpose, including the pseudo-tensor (odd L) parts.
4. A species pair absent from the training data: predicted as zero with a report entry, not an error.
5. Memory: the design matrix of all bonds of a 729-atom cell does not fit; the fit must stream.

---

### Task 1: bond features

**Files:** Create `src/ace_jax/ham/bond.py`, `tests/test_ham_bond.py`.

**Interfaces:**
- `bond.BondBasis(site: SiteBasis, rcut_bond: float, n_bond: int, n_cpl: int, l_cpl: int)`; `bond.radial(r, rcut, n) -> (..., n)` (polynomials in r times a cutoff envelope that vanishes with two derivatives at rcut).
- `bond.bond_graph(basis, positions, cell, numbers, pbc=True) -> BondGraph(i, j, R (n, 3 int), rij (n, 3), reverse (n,))`: every directed pair within `rcut_bond`, i = j images included, on-site excluded; `reverse[k]` is the index of (j, i, −R).
- `bond.features(basis, F, graph, sel) -> {(L, p): {"bond": (n, k0, 2L+1), "i": (n, k1, 2L+1), "j": (n, k1, 2L+1)}}` for the bonds `sel`, `F` the site features; `bond.symmetric(group_i, group_j, L)` for same-shell blocks.

- [ ] Tests: features rotate with the Wigner matrices and take the factor p under inversion; `features` of the reverse bond are related by i ↔ j and r → −r as the symmetry rule needs; `reverse` is an involution and covers self-image bonds.
- [ ] Implement. Commit `feat(ham): bond features from shared site features`.

### Task 2: off-site model and streamed fit

**Files:** Create `src/ace_jax/ham/offsite.py`; extend `src/ace_jax/ham/fit.py` (`fit_offsite`); `tests/test_ham_offsite.py`.

**Interfaces:**
- `offsite.OffsiteModel(bond: BondBasis, coef: {(zi, zj, s1, s2, L): (n_channel, n_feat)}, orbitals, channels)`; `predict(positions, cell, numbers) -> {channel: {(Rx, Ry, Rz, i, j): block}}`.
- `fit.fit_offsite(bond_basis, datasets, channels, ridge, max_bonds=None, chunk=...) -> (model, report)`: QR accumulated over chunks of bonds; `max_bonds` subsamples per species pair.

- [ ] Tests: random-coefficient model: blocks rotate and invert with the orbitals; reverse bond is the transpose for every bond and channel; SiC fixture: fit runs, the overlap is reproduced to a recorded tolerance.
- [ ] Implement. Commit `feat(ham): off-site model from shared features, streamed block fit`.

### Task 3: accuracy on the aluminium data

- [ ] `bench/ham/offsite_accuracy.py`: off-site H and S of the paper's Al supercells (bonds from the dense Γ matrices within the orbital range), train and test bonds from different supercells; error against bond length and by block type; convergence in `n_bond`, `n_cpl`, `l_cpl` and the site basis.
- [ ] `bench/ham/aceh_offsite_accuracy.jl`: ACEhamiltonians.jl's off-site sub-models on the same bonds.
- [ ] Record in `docs/dev/ham-results.md`, including what the shared-feature model cannot reach.

### Task 4: compiled prediction and timing

- [ ] One jitted program for site features, bond features and readout; equality with the reference implementation.
- [ ] Time on the 108-atom SiC input (CPU one thread; GPU) per atom and per block, beside MACE-H's numbers.
