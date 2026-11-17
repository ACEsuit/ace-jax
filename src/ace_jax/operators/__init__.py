"""Equivariant one- and two-centre block models of atomic-orbital operators: everything that
does not know which operator it is.  Site features, bond features, the on-site and off-site
models with their compiled predictors, the orbital bases and the block fit; operator channels
are a leading coefficient axis.

`ace_jax.ham` is the Hamiltonian and overlap on top (file formats, Bloch sums, bands); another
operator is a sibling of it.  Layering: docs/dev/specs/2026-11-17-package-structure-design.md,
section 4 (`tests/test_import_layers.py`)."""
