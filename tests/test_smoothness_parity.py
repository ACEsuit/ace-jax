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
import json

import numpy as np
import pytest

from conftest import FIXTURE_DIR, REQUIRE

from ace_jax.construct.prior import (
    P, WL, WN, gamma_from_model, model_nnll, prior_diagonal, smoothness_prior,
    unflatten_nnll)

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
def test_oracle_block_order_is_tensor_then_pair(name):
    """Pin the GLOBAL block order the port hard-codes in model_nnll:
    ``[tensor x NZ | pair x NZ]``.  test_layout_species_replication reshapes by
    kind and so would also pass for an interleaved [t, p, t, p] layout; this is
    the contract that aligns the rebuilt gamma with fit/rows.py:_place."""
    z = _load(name)
    nz, kind = int(z["n_species"]), list(np.asarray(z["nnll_kind"]))
    assert kind == [0] * (int(z["n_tensor"]) * nz) + [1] * (int(z["n_pair"]) * nz)


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


# ---------------------------------------------------------------------------
# Production wire-in (Phase 4): rebuild gamma from an exported model npz when
# the exporter did not store it.  The tensor bodies come from the exporter's
# own spec dump meta["nnll"] (get_nnll_spec(m.tensor), n_B entries shared
# across species) tiled per species, pair bodies are the (n, 0) singletons.
# The ship fixtures (real exports) are the targets; si_fitted stores the
# Julia-computed gamma, closing the loop end-to-end bit-for-bit.
MODELS = [
    "si_fitted.npz",     # NZ=1, spline radial, HAS exported gamma -> bit-for-bit ref
    "si_ace_model.npz",  # NZ=1, analytic radial, dense A2B, no gamma
    "si_1429.npz",       # NZ=1, big basis (n_B=1429, order 3), no gamma
    "sige_nofit.npz",    # NZ=2, no gamma -- the actual refit fallback target
]


def _load_model(name):
    p = FIXTURE_DIR / name
    if not p.exists():
        if REQUIRE:
            raise FileNotFoundError(f"ACEJAX_REQUIRE_FIXTURES is set but {p} is missing")
        pytest.skip(f"missing fixture {p.name}")
    z = np.load(p)
    return z, json.loads(bytes(z["meta_json"]).decode())


@pytest.mark.parametrize("name", MODELS)
def test_model_nnll_length_matches_exporter(name):
    """The rebuilt full basis has exactly the exporter's own ``len_basis``
    (Julia's length_basis(m)) columns -- an independent source, unlike
    NZ * (n_B + n_pair) which model_nnll itself computes -- and a corrupted
    len_basis is refused rather than silently misaligning gamma."""
    z, meta = _load_model(name)
    assert len(model_nnll(meta)) == int(meta["len_basis"])
    with pytest.raises(ValueError, match="len_basis"):
        model_nnll(dict(meta, len_basis=int(meta["len_basis"]) + 1))


def test_model_nnll_refuses_export_without_nnll():
    """An export older than the nnll meta dump cannot rebuild gamma: fail with
    a message naming the missing key and the fix, not a bare KeyError."""
    z, meta = _load_model("si_ace_model.npz")
    meta.pop("nnll")
    with pytest.raises(ValueError, match="nnll.*re-export"):
        model_nnll(meta)


@pytest.mark.parametrize("name", MODELS)
def test_gamma_from_model_structure(name):
    """Full-basis length, positivity, and the species-block replication
    contract (NZ consecutive identical blocks per kind) on rebuilt gamma."""
    z, meta = _load_model(name)
    n_B, n_pair, NZ = int(meta["n_B"]), int(meta["n_pair"]), len(meta["elements"])
    gamma = gamma_from_model(meta)
    assert gamma.dtype == np.float64 and np.all(gamma > 0)
    tensor = gamma[:NZ * n_B].reshape(NZ, n_B)
    pair = gamma[NZ * n_B:].reshape(NZ, n_pair)
    assert np.array_equal(tensor, np.repeat(tensor[:1], NZ, axis=0))
    assert np.array_equal(pair, np.repeat(pair[:1], NZ, axis=0))
    # pair columns are the (n, 0) singletons: (1/wn)^4 = 1, 16, 81, ...
    assert np.array_equal(pair[0], np.arange(1, n_pair + 1, dtype=np.float64) ** 4)


def test_gamma_from_model_matches_exported():
    """On the one ship fixture that stores gamma, the rebuild must agree
    bit-for-bit -- the fallback never silently overrides an exported value,
    but it must reproduce it when the export lacks one."""
    z, meta = _load_model("si_fitted.npz")
    assert np.array_equal(gamma_from_model(meta), np.asarray(z["gamma"], np.float64))


def test_prior_diagonal_prefers_exported_gamma(capsys):
    """The single wire-in helper cli.py / bench run.py use: an exported gamma
    is authoritative and used verbatim, silently."""
    z, meta = _load_model("si_fitted.npz")
    gamma = prior_diagonal(z, meta, "si_fitted.npz")
    assert np.array_equal(gamma, np.asarray(z["gamma"], np.float64))
    assert capsys.readouterr().out == ""


def test_prior_diagonal_rebuilds_when_missing(capsys):
    """Without an exported gamma the helper rebuilds it from meta and says so,
    naming the file, so a refit log shows which prior was used."""
    z, meta = _load_model("sige_nofit.npz")
    assert "gamma" not in z.files
    gamma = prior_diagonal(z, meta, "sige_nofit.npz")
    assert np.array_equal(gamma, gamma_from_model(meta))
    out = capsys.readouterr().out
    assert "sige_nofit.npz" in out and "gamma" in out


def test_gamma_from_model_folds_embedded_radial_index():
    """Embedded models (export_model.jl KIND=embedding) carry a channel k folded
    into the tensor radial index, n = (n'-1)*d_max + k.  Julia's
    smoothness_prior (smoothness_priors.jl:110-117) unfolds the TENSOR block to
    n' = (n-1) div d_max + 1 before applying the functional; the pair block is
    categorical and is left alone.  The port must do the same."""
    z, meta = _load_model("sige_nofit.npz")
    n_B, n_pair, NZ = int(meta["n_B"]), int(meta["n_pair"]), len(meta["elements"])
    d = 3
    meta = dict(meta, embedding=json.dumps({"d_max": d, "widths": [d, d], "block_rule": "x"}))
    gamma = gamma_from_model(meta)
    assert gamma.shape == (NZ * (n_B + n_pair),)
    # Independent evaluation of the Julia formula on the unfolded bodies.
    expect_tensor = np.array([
        sum((l / WL) ** P + (((n - 1) // d + 1) / WN) ** P for n, l in bb)
        for bb in meta["nnll"]], np.float64)
    assert np.array_equal(gamma[:NZ * n_B].reshape(NZ, n_B),
                          np.repeat(expect_tensor[None], NZ, axis=0))
    assert np.array_equal(gamma[NZ * n_B:].reshape(NZ, n_pair),
                          np.repeat(np.arange(1, n_pair + 1, dtype=np.float64)[None] ** 4,
                                    NZ, axis=0))
    # ...and the fold actually changed something (some tensor n > d_max).
    assert not np.array_equal(gamma, gamma_from_model(dict(meta, embedding="")))
