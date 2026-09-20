"""Julia-free unit tests for the degenerate-block subspace invariant used by the
coupling parity tests (tests/test_coupling_parity.py).

At order >= 4 a nnll block of multiplicity m has a symmetrisation basis defined
only up to a within-block ORTHOGONAL rotation, so exact A2B parity between two
basis constructions becomes a SUBSPACE match.  `subspace_residual` compares the
orthogonal projectors onto the two blocks' row spaces.  These tests prove that
invariant is (a) exactly rotation/permutation/sign invariant and (b) genuinely
necessary -- a naive entrywise comparison fails on a rotated block.  They run in
the core pip suite (numpy only; no ET/Julia)."""
import numpy as np
import pytest

from ace_jax.construct.coupling import subspace_residual


def _random_orthogonal(m, rng):
    return np.linalg.qr(rng.standard_normal((m, m)))[0]


@pytest.mark.parametrize("m,k", [(2, 12), (2, 25), (3, 20), (4, 40)])
def test_subspace_residual_rotation_invariant(m, k):
    """A block rotated within its row space by any orthogonal O has ~0 residual,
    while the raw entries change a lot -- so the projector is the right invariant."""
    rng = np.random.default_rng(0xACE + m * 100 + k)
    M = rng.standard_normal((m, k))
    for _ in range(20):
        O = _random_orthogonal(m, rng)
        rotated = O @ M
        assert subspace_residual(M, rotated) < 1e-12
        # the rotation is non-trivial: raw entries really do differ
        assert np.abs(M - rotated).max() > 1e-3


def test_subspace_residual_permutation_and_sign_invariant():
    rng = np.random.default_rng(7)
    M = rng.standard_normal((3, 15))
    perm = rng.permutation(3)
    signs = rng.choice([-1.0, 1.0], size=3)[:, None]
    assert subspace_residual(M, signs * M[perm]) < 1e-12


def test_subspace_residual_detects_a_genuinely_different_block():
    """Guard against false positives: an unrelated row space must NOT read as 0."""
    rng = np.random.default_rng(11)
    M = rng.standard_normal((2, 12))
    other = rng.standard_normal((2, 12))
    assert subspace_residual(M, other) > 1e-2


def test_naive_comparison_fails_where_projector_succeeds():
    """The degeneracy is real: a rotated block breaks entrywise (perm+sign)
    comparison but passes the subspace invariant."""
    rng = np.random.default_rng(3)
    M = rng.standard_normal((2, 10))
    O = _random_orthogonal(2, rng)
    rotated = O @ M
    naive = min(np.abs(M - rotated).max(), np.abs(M + rotated).max())  # perm+sign, m=1 style
    assert naive > 1e-2                      # naive comparison would (wrongly) fail
    assert subspace_residual(M, rotated) < 1e-12  # subspace invariant holds
