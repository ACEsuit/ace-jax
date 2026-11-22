"""Hamiltonian and overlap models: what is specific to H and S on top of `ace_jax.operators`.
The DeepH-layout reader and writer and the npz cache (`deeph`, `cache`), and the Bloch sums and
bands of predicted blocks (`operator`), and the electron density as coefficients of
atom-centred auxiliary functions (`density`, on `operators.centred`; no h5py).  Ingestion from electronic-structure codes is the Julia
script `ham/julia/ingest.jl` (Quoll.jl).

Design: docs/dev/specs/2026-11-16-hamiltonians-design.md and, for the package layout,
docs/dev/specs/2026-11-17-package-structure-design.md.  h5py (the `ham` extra) is imported
inside the functions that read or write files."""
