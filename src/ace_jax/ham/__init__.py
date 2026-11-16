"""ACE models of atomic-orbital operators (Hamiltonian, overlap).

Design: docs/dev/specs/2026-11-16-hamiltonians-design.md.  This package imports
only the framework parts of ace-jax (eval, basis, the generic parts of fit);
tests/test_ham_boundary.py holds that boundary.  h5py (the `ham` extra) is
imported inside the functions that read or write files."""
