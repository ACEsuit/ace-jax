# Package structure, addendum: feature providers for TrACE and TT-ACE (design)

Status: design, awaiting approval (2026-11-17). Addendum to
`docs/dev/specs/2026-11-17-package-structure-design.md` ("the main spec"), which is being
implemented as written. This document changes none of its steps 1 to 5. It refines §4 and
step 6, and adds one small step that can land at any time.

Reviewed against `feat/hamiltonians` at 6d5bbeb and `jameskermode/ace-jax-tt` at 9fff531
(`bench/tt_ace/`).

## Intent

Let an operator model (Hamiltonian, overlap, later others) take its equivariant features from any
ACE architecture, not only from a categorical linear ACE: first linear ACE with an element
embedding (TrACE), then the TT-ACE chain.

## Findings

- The operator readout needs only equivariant features per atom and per bond,
  {(L, parity): (n, n_features, 2L+1)}. Where they come from is not its business.
- `ham/features.py::SiteBasis` hard-wires one source: the sparse A/AA product of `build_model`
  (categorical species) and the `ace-jax-coupling` A2B tables. The main spec §4 keeps that
  coupling by building `operators` on `eval.product` directly.
- TrACE is already in the package as a potential (`basis/model.py::build_embedding_model`,
  `aj fit --basis-embedding`). `build_site_basis` does not take it.
- TT-ACE (`bench/tt_ace/model.py` in the fork) already carries every channel (L, s), L ≤ `Lbond`,
  s = proper or pseudo, at every chain position p. The energy reads only (0, +).
- TT-ACE's fit is ALS: with the other blocks fixed, each block solve is a ridge least squares
  streamed through `fit.stats.device_qr_stream` (`bench/tt_ace/als.py`).
- `ham/cg.py` and `bench/tt_ace/cg.py` both build real Clebsch–Gordan tables against
  `eval/harmonics.py`, by different methods (Racah's formula and the real transform; a null space
  of the equivariance constraint).

## Goals

1. A feature-provider protocol that `operators` is written against.
2. TrACE as a provider, with the closed-form fit and the UQ unchanged.
3. A defined route for TT-ACE as a provider, without moving TT-ACE into the package now.
4. One implementation of the real Clebsch–Gordan tables.

## Non-goals

- Moving TT-ACE from the fork into `ace_jax`.
- A bond-environment (`BondEnv`) provider on a TT chain.
- Uncertainty in the TT cores.
- Any change to the main spec's stream A (steps 1 to 5) or to its frozen files.

## Design

### A1. The protocol

A site feature provider has:

- `features(rij, zi, zj, segment_ids, n_nodes, mask=None) -> {(L, p): (n_nodes, n(L, p), 2L+1)}`
- `lps`, `n(L, p)`, `rcut`, `species_index(numbers)`

This is `SiteBasis`'s public surface today; the protocol is that surface with the carrier model
and the COO coupling arrays kept private to one implementation. Parity p is (−1)^Σl, as in the
Hamiltonian design. The bond providers (`BondEnv`, `SiteBond`) take a site provider, not a
`SiteBasis`.

The protocol (a `typing.Protocol`, no base class) lives in layer 0 of the main spec's table, next
to `eval.product`. `operators` imports the protocol and never a concrete model.

### A2. Clebsch–Gordan tables

`basis/cg.py` in layer 0, from `ham/cg.py` (closed form, tested against Wigner matrices taken
from the harmonics). The fork's `bench/tt_ace/cg.py` is replaced by it when TT-ACE next merges
`main`; its tables must agree up to the sign convention, which a test fixes before the swap.

### A3. Providers

| Provider | Source | Readout fit | Work |
|---|---|---|---|
| Linear ACE, categorical | `eval.product` + `basis.coupling` | closed form, full UQ | the main spec's step 6 (drop the carrier) |
| Linear ACE, element embedding (TrACE) | the same product on `build_embedding_model`'s embedded radials | closed form, full UQ | `build_site_basis` takes the embedding; the coupling call is unchanged |
| TT-ACE chain | `h(p)_(L,s)` at every position p | closed form and full UQ with the cores frozen; ALS for joint training | A4 |

### A4. What TT-ACE needs

1. Expose every channel: `site_scalars` generalised to all (L, s), with `Lbond` at least twice
   the largest orbital l. Map s to the parity used here: p = s (−1)^L.
2. Split the chain (a feature provider) from its energy readout (a potential), as the main spec
   §3 does for `EdgeSiteModel`, so that operators use the chain without importing a potential.
3. Off-site blocks through `SiteBond` (site features times one bond factor).
4. Joint training of the cores on E/F/V and operator blocks: one more right-hand-side family per
   ALS block solve, through the generic row interface of the main spec's step 4.
5. UQ is conditional on the cores: jackknife, conformal and POPS apply to the linear readout.

When TT-ACE moves into the package it is a layer-3 potential (`ace_jax.tt`) whose chain module
sits with the providers.

## Order of work

Numbered against the main spec.

- **A2 (`basis/cg.py`)**: on `feat/hamiltonians` now, as a move of `ham/cg.py`. It touches no
  file stream A edits.
- **A1 (protocol)** and **TrACE**: inside the main spec's step 6, which already rewrites
  `ham/features.py`. Step 6 builds `operators` on the protocol instead of on `eval.product`.
- **A4**: in the `ace-jax-tt` fork, after step 6 and after the generic rows of step 4 are on
  `main`. Items 1 to 3 need only the protocol; item 4 needs step 4.

## Validation

- TrACE provider: rotation and inversion equivariance of every block; with an identity
  embedding, features equal the categorical provider's to roundoff.
- TT provider: rotation and inversion equivariance of every exposed channel; the (0, +) channel
  reproduces `site_scalars`.
- Accuracy and ms per atom of each provider on the Hamiltonian design's SiC and Si data, in
  `docs/dev/ham-results.md`, beside the `BondEnv` against `SiteBond` comparison planned there.

## Calls made here that the owner has not confirmed

- The protocol is taken from `SiteBasis`'s current surface; it is not tested against a TT chain.
- `ham/cg.py` is the copy that is kept.
- Whether TT site features are accurate enough for off-site blocks through `SiteBond` is
  unmeasured; if they are not, a TT chain over the bond environment is new work outside this
  addendum.
- TT-ACE stays in the fork until after step 6.

## Done in stream A after this addendum (2026-11-17, step "3b")

Agreed with the owner after the addendum was written; the text above is as received.

- `eval/features.py` (layer 0): the `SiteFeatures` protocol of A1, and `pooled_a` / `edge_a_rows`,
  the one-particle basis of a linear ACE from a radial and index arrays, no model.
- `eval/radial.py::TensorRadial` (`tensor_radial`, `radial_one`, `rnl_basis_values`): the tensor
  radial taken out of `ACEModel`, which now calls these. `TensorRadial.of(model)` copies the radial
  of a full model, categorical or element-embedding ("spline_factorised"), so a provider keeps
  the radial without keeping the model. Tested bit for bit against `ACEModel` for the analytic,
  spline, factorised and spherical-Bessel radials.
- `tests/test_import_layers.py` lists `basis.cg` as a framework module before it exists (A2).

Still open for step 6: building the radial itself. `TensorRadial.of` needs a model, and the only
builders (`build_model`, `build_embedding_model`) are in `basis/model.py`, an application module
that is frozen until `feat/hamiltonians` merges. Step 6 either factors the radial construction
out of it into layer 0, or has a front end build the provider and hand it to `operators`.
