"""Parity of the Python smoothness-prior port (construct/prior.py) against the
Julia/ACEpotentials oracle fixtures (julia/smoothness_reference.jl) -- pip-only,
no Julia needed (the CI prior-parity job regenerates the fixtures).

Fixtures span the same four shapes as the coupling references (orders 2-4,
including the degenerate-block SiGe wL=0.5 shape).  Each stores the prior
diagonal ``gamma`` together with the FULL per-column nnll basis the Julia
functional consumed, so the comparison isolates the FORMULA (the port is fed
the oracle's own inputs, like the coupling test's "oracle" level).

Bit-for-bit is asserted: every prior term is an integer over 2^4 in float64 and
partial sums stay below 2^53, so Julia and Python must agree exactly (see
prior.py's docstring).  A tolerance here would hide a convention drift.
"""
import collections

import numpy as np
import pytest

from conftest import FIXTURE_DIR, REQUIRE

from ace_jax.construct.prior import P, WL, WN, smoothness_prior, unflatten_nnll

# Same shapes as fixtures/coupling_ref_*.npz (order 2/3 multiplicity-1 blocks,
# order 4 SiGe wL=0.5 with degenerate nnll blocks).
FIXTURES = [
    "smoothness_ref_SiGe_o2d6.npz",     # order 2, 2 species
    "smoothness_ref_CrMnFe_o2d5.npz",   # order 2, 3 species
    "smoothness_ref_CrMnFe_o3d6.npz",   # order 3, 3 species
    "smoothness_ref_SiGe_o4d5w05.npz",  # order 4, 2 species
]


def _load(name):
    p = FIXTURE_DIR / name
    if not p.exists():
        if REQUIRE:
            raise FileNotFoundError(
                f"ACEJAX_REQUIRE_FIXTURES is set but {p} is missing -- "
                "julia/smoothness_reference.jl did not produce it")
        pytest.skip(f"missing fixture {p.name}; see julia/smoothness_reference.jl")
    return np.load(p)


@pytest.mark.parametrize("name", FIXTURES)
def test_matches_oracle_bit_for_bit(name):
    z = _load(name)
    nnll = unflatten_nnll(z["nnll_flat"], z["nnll_len"])
    assert len(nnll) == int(z["n_basis"])
    gamma = smoothness_prior(nnll, p=float(z["prior_p"]), wl=float(z["prior_wl"]),
                             wn=float(z["prior_wn"]))
    ref = np.asarray(z["gamma"], np.float64)
    assert np.array_equal(gamma, ref), f"max |dgamma| = {np.abs(gamma - ref).max()}"
    assert np.all(ref > 0)


def test_defaults_match_oracle():
    """The exporter calls the prior with no kwargs; the port's defaults must be
    exactly the Julia defaults, or every fixture comparison above is moot."""
    z = _load(FIXTURES[0])
    assert (P, WL, WN) == (float(z["prior_p"]), float(z["prior_wl"]), float(z["prior_wn"]))
    # ...and the documented constants really are smoothness_priors.jl:125's.
    assert (P, WL, WN) == (4.0, 2.0 / 3.0, 1.0)


@pytest.mark.parametrize("name", FIXTURES)
def test_layout_species_replication(name):
    """Column layout contract, read off the oracle: per kind (tensor/pair), NZ
    CONSECUTIVE IDENTICAL species blocks -- _nnll_basis assigns the same
    species-independent spec to every species' index range.  The (n,l) patterns
    may repeat WITHIN a block (order-4 coupling paths sharing an nnll), so
    block identity is compared column-by-column, not as a set."""
    z = _load(name)
    nnll = unflatten_nnll(z["nnll_flat"], z["nnll_len"])
    nz, kind = int(z["n_species"]), np.asarray(z["nnll_kind"])
    assert len(nnll) == int(z["n_basis"])
    for k, n_expect in ((0, int(z["n_tensor"])), (1, int(z["n_pair"]))):
        idx = np.flatnonzero(kind == k)
        assert len(idx) == n_expect * nz
        blocks = idx.reshape(nz, n_expect)
        first = [tuple(nnll[i]) for i in blocks[0]]
        for s in range(1, nz):
            assert [tuple(nnll[i]) for i in blocks[s]] == first, \
                f"species block {s} differs for kind={k}"


@pytest.mark.parametrize("name", FIXTURES)
def test_reference_constant_across_identical_nnll(name):
    """The prior is a per-column functional of nnll: columns with IDENTICAL
    bodies must carry IDENTICAL gamma -- this is what makes per-column
    signature alignment (and the production wire-in's rebuilt nnll) valid."""
    z = _load(name)
    nnll = unflatten_nnll(z["nnll_flat"], z["nnll_len"])
    nz, kind = int(z["n_species"]), np.asarray(z["nnll_kind"])
    ref = np.asarray(z["gamma"], np.float64)
    by_pattern = collections.defaultdict(list)
    for i, bb in enumerate(nnll):
        by_pattern[(int(kind[i]), tuple(bb))].append(i)
    for key, cols in by_pattern.items():
        assert len(cols) % nz == 0, f"{key} appears {len(cols)}x, not a multiple of {nz}"
        assert np.unique(ref[cols]).size == 1, f"gamma varies within {key}"


@pytest.mark.parametrize("name", FIXTURES)
def test_port_agrees_with_reference_per_column(name):
    """Column-wise (not vector) agreement: catches an off-by-one column shift
    that a whole-vector comparison of a constant-ish gamma could mask."""
    z = _load(name)
    nnll = unflatten_nnll(z["nnll_flat"], z["nnll_len"])
    gamma = smoothness_prior(nnll)
    ref = np.asarray(z["gamma"], np.float64)
    worst = int(np.argmax(np.abs(gamma - ref)))
    assert gamma[worst] == ref[worst]
