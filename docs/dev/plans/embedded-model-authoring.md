# Embedded-model authoring in Python (PR #4 follow-up)

**Goal.** `build_embedding_model(...)` authors the frozen-element-embedding ACE model
(ACEpotentials `ace_embedding_model`, default `ace1_compat=true`) entirely in
Python, so an embedded linear fit (lossless or `d_max`) no longer needs the Julia
exporter.  The coupling stays on the existing ET shim.

**Reference.** ACEpotentials `src/models/embeddings.jl` (`embedding_rows`,
`_pca_reduce`, `_generic_frame`, `embedding_widths`, `ace_embedding_model`) on the
`pr/element-embeddings` / `gp/hybrid-ace-gp` branches; `acejax/julia/export_model.jl`
(`embedding` kind, factorised spline export).

**Parity strategy.**
- Unit parity against Julia oracle fixtures (`julia/embedding_reference.jl` ->
  `fixtures/embedding_ref_*.npz`, committed; tests skip if missing, as the
  smoothness/coupling parity tests do).
- End-to-end: a Julia-exported embedded model (small shapes) loaded by
  `eval.io.load` and the Python-built one must span the SAME descriptor space on
  random configs (subspace test) -- immune to basis ordering and PCA signs --
  and have identical shapes (n_B, n_rnl, widths, n_pair).

## Stages (one commit each, test first)

1. **Embedding reduction** (`construct/embedding.py`): `read_embedding` (JSON
   `Z`/`emb`), `embedding_rows(emb, Z, zlist, d, reduction, normalise)`,
   `_pca_reduce`, `_generic_frame` (xorshift64* bit-exact, QR), `embedding_widths`.
   Oracle: rows for identity and MH-1 tables at d < rank, = rank, > rank; widths.
2. **ace1-compat radial pieces** (`construct/radial_ace1.py`): Jacobi(alpha, beta)
   basis values as Polynomials4ML `jacobi_basis`; cubic B-spline coefficients as
   Interpolations `cubic_spline_interpolation` (Line(OnGrid)) on 100 nodes of
   [-1, 1]; ace1 transforms (Agnesi (2,4) tensor, (1,3) pair), envelopes (poly2sx
   tensor, ace1_poly1sr pair).  Oracle: an exported ace1/embedding npz's
   `rnl_spline_coefs_single` / `pair_spline_coefs` and transform/envelope params.
3. **Embedded spec**: `:ace1` folded block rule (fold with NZ, keep z' = 1, unfold),
   `rspec` n = (n'-1) d + k, channel-diagonal AA spec with per-order widths,
   sorted by order; coupling via `couple_cached`.  Oracle: Julia rspec / AA_spec
   sets.
4. **Pair basis + assembly + export**: ace1 pair basis (Legendre, one-hot,
   splined), uniform cutoffs (mean bond length, rcut = 2.5 r0 unless given),
   ACEModel with `spline_factorised` radial, spherical Y, meta["embedding"];
   smoothness prior on the UNFOLDED radial degree (ChannelLevel); `save_npz`
   round-trip.  End-to-end subspace parity vs the Julia export.
5. **GP reuse**: `inducing.principal_frame` delegates to `_pca_reduce`'s SVD step
   (cross-PR; done on the GP branch after this merges).

## Global constraints
- Pure numpy for construction (JAX only in eval), float64.
- Existing `build_model` behaviour and its tests unchanged.
- No new required dependencies.
