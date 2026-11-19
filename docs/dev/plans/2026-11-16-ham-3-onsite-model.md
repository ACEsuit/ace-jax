# Hamiltonians 3: on-site model, block fit and first speed number Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On-site blocks of H and S predicted from one A/AA evaluation per atom, fitted by least squares on the SiC fixture, with the structural tests and a first timing.

**Architecture:** A carrier `ACEModel` from `build_model` supplies radials, harmonics and the A/AA product; its `aspec`/`aa_specs` are replaced by the union of an `EquivariantSet`. Features B^(L,p) come from sparse `A2B3` contractions of the one AA vector. A shell pair (s1 ≤ s2) of a species reads the features of parity p = (−1)^(l1+l2) for each L, through its own coefficient vectors; the block is `cg.assemble` of the parts.

**Tech Stack:** JAX, equinox, numpy; no Julia (committed coupling cache).

**Spec:** `docs/dev/specs/2026-11-16-hamiltonians-design.md` §3, §5, §6.

## Global Constraints

- One A/AA evaluation per atom whatever the number of shells and channels.
- Parity of a shell-pair block is (−1)^(l1+l2); same-shell blocks use even L only; (s2, s1) is the transpose of (s1, s2).
- float64; the library never sets x64.
- Tests run with `ACEJAX_COUPLING_CACHE_ONLY=1` on a committed cache (`fixtures/ham/coupling_cache`).
- `ham` imports only framework modules (`tests/test_ham_boundary.py`).

## Review Focus

1. A species in the structure with no orbital basis or no model: clear `KeyError`, not a shape error.
2. An isolated atom (no neighbours): features are finite and the block is the constant (L = 0) part only.
3. A shell pair whose L has no feature of the needed parity (empty coupling): that part is predicted as zero, and the fit reports it rather than failing.
4. Rotated and inverted structures: blocks transform with the Wigner matrices of the shells (the equivariance test), for every shell pair including pseudo-tensor parts.
5. Channel order: H and S coefficients must not be swapped between fit and predict (test fits two different targets).

---

### Task 1: specs for all L, and the feature evaluator

**Files:** Modify `src/ace_jax/basis/spec.py` (`build_spec(..., admit=None)`); Create `src/ace_jax/ham/features.py`, `tests/test_ham_features.py`, `fixtures/ham/coupling_cache/`.

**Interfaces:**
- `build_spec(..., admit=None)`: `admit(bb) -> bool` replaces the L = 0 admissibility test (`rpe_admissible` and `_couples_to_zero`) when given; default unchanged.
- `features.SiteBasis` (eqx.Module): `carrier: ACEModel`, `lps: tuple[(L, p)]`, per-(L, p) `rows, ms, cols, vals` arrays and `n_B`; `features.build_site_basis(elements, order, totaldegree, lmax_feat, *, rcut, wL=1.5, coupling_cache_dir=None, **radial) -> SiteBasis`, with `lps` = all (L, ±1) for L ≤ `lmax_feat` that are non-empty.
- `SiteBasis.features(rij, zi, zj, segment_ids, n_nodes, mask=None) -> {(L, p): (n_nodes, n_B, 2L+1)}`.

- [ ] Tests: features rotate with `wigner(L)` and take the factor p under inversion, on a random 2-species cluster; an isolated atom gives finite features; `build_spec` default output unchanged (existing tests).
- [ ] Implement; generate the cache with the development library; run with `ACEJAX_COUPLING_CACHE_ONLY=1`.
- [ ] Commit `feat(ham): site features for all L from one AA`.

### Task 2: readout, on-site model and least-squares block fit

**Files:** Create `src/ace_jax/ham/onsite.py`, `src/ace_jax/ham/fit.py`, `tests/test_ham_onsite.py`.

**Interfaces:**
- `onsite.parts_of(l1, l2, same_shell) -> list[(L, p)]`.
- `onsite.OnsiteModel` (eqx.Module): `basis: SiteBasis`, `orbitals: OrbitalBasis`, `channels: tuple[str]`, `coef: {(z, s1, s2, L): (n_channel, n_B(L, p) + [L == 0])}` (the extra column is the constant); `predict(graph) -> (n_atoms, n_channel, n_orb_max, n_orb_max)` and `blocks(data-like) -> {channel: {(0, 0, 0, i, i): block}}`.
- `fit.fit_onsite(basis, datasets, channels=("H", "S"), ridge=1e-8) -> OnsiteModel`: per (z, s1 ≤ s2, L) one least-squares problem with all channels as right-hand sides (QR of the stacked (n_atoms·(2L+1), n_B) design; the ridge only as an appended row block).

- [ ] Tests on the SiC fixture: fit reproduces the on-site blocks of both channels to the residual the test records; rotating the structure rotates the predicted blocks (D ⊗ D); predicted on-site blocks are symmetric; H and S are not swapped; transposed shell pair is the transpose.
- [ ] Implement.
- [ ] Commit `feat(ham): on-site model and block fit`.

### Task 3: first timing

**Files:** Create `bench/ham/onsite_time.py`.

- [ ] Time `predict` per atom (jit, after warm-up) on a 512-atom SiC supercell on one CPU thread and all threads; print features-only and features-plus-readout times.
- [ ] Record the numbers in `docs/dev/ham-results.md` with the machine and the basis size. The comparison with ACEhamiltonians.jl waits for the off-site model (plan 4), where its cost is.
