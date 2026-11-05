"""Embedded basis spec (ace_embedding_model): the :ace1 folded block rule, the
channel-widened radial spec n = (n'-1) d + k, and the channel-diagonal many-body
bodies with per-order widths -- against Julia-exported embedded models."""
import json
import pathlib

import numpy as np
import pytest

from ace_jax.basis import radial_ace1 as ra
from ace_jax.basis.spec import build_embedding_spec, build_spec

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
CASES = ["emb_ref_SiGe_o2d6", "emb_ref_CrMnFe_o3d5_dmax4"]


def _load(name):
    p = FIX / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"missing {p.name}; see julia/export_model.jl (embedding kind)")
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.eval import load
    model, meta, z = load(p)
    return meta, z


def _spec(meta):
    emb = json.loads(meta["embedding"])
    S = len(meta["elements"])
    return build_embedding_spec(S, meta["order"], meta["totaldegree"], emb["widths"], wL=1.5,
                                maxl=6, block_rule=emb["block_rule"])


@pytest.mark.parametrize("name", CASES)
def test_many_body_bodies_match_julia(name):
    meta, _ = _load(name)
    sp = _spec(meta)
    got = {tuple(sorted(map(tuple, b))) for b in sp.mb}
    ref = {tuple(sorted(tuple(c) for c in b)) for b in meta["nnll"]}
    assert got == ref
    assert [len(b) for b in sp.mb] == sorted(len(b) for b in sp.mb)       # grouped by order


@pytest.mark.parametrize("name", CASES)
def test_radial_spec_and_factorised_index_maps_match_julia(name):
    meta, z = _load(name)
    sp = _spec(meta)
    assert len(sp.rspec) == meta["n_rnl"]
    assert np.array_equal(sp.nidx, z["rnl_emb_nidx"]) and np.array_equal(sp.kidx, z["rnl_emb_kidx"])
    lmax = max(l for l, _ in sp.Ylm)
    assert lmax >= meta["lmax"] and lmax >= max(l for b in sp.mb for _, l in b)    # covers every body
    # the factorised table's columns are the (n', l) blocks in r1 order:
    # column j is the spline of Q_{n'_j - 1}
    T = ra.jacobi_spline_table(max(n for n, _ in sp.r1), 4.0, 4.0)
    ref = z["rnl_spline_coefs_single"]
    got = T[:, [n - 1 for n, _ in sp.r1]]
    assert np.allclose(got, ref, atol=1e-12 * np.abs(ref).max())


def test_widths_give_channel_diagonal_bodies():
    sp = build_embedding_spec(2, 2, 4, [2, 3], wL=1.5)
    d = 3
    for b in sp.mb:
        ks = {(n - 1) % d for n, _ in b}
        assert len(ks) == 1                                               # one channel per body
        assert next(iter(ks)) < [2, 3][len(b) - 1]                        # within its order's width


# (S, order, totaldegree, maxl, block_rule): ace1 folding at several S and orders, an l cap, and the
# symmetric rule (NZf = 1, where keep only applies maxl)
PREFILTER_CASES = [(4, 3, 8, None, "ace1"), (5, 4, 7, None, "ace1"), (3, 4, 8, 3, "ace1"),
                   (8, 4, 6, None, "ace1"), (4, 3, 8, 2, "symmetric")]


@pytest.mark.parametrize("S, order, md, maxl, rule", PREFILTER_CASES)
def test_build_spec_keep_matches_filtering_afterwards(S, order, md, maxl, rule):
    """keep prunes the search to the accepted elements: the same bodies in the same order as
    enumerating everything and filtering (build_embedding_spec's path before keep existed)."""
    NZf = S if rule == "ace1" else 1
    ok = lambda n, l: (n - 1) % NZf == 0 and (maxl is None or l <= maxl)  # noqa: E731
    mb_all, Rnl_all, Y_all = build_spec(NZf, order, md)
    mb, Rnl, Y = build_spec(NZf, order, md, keep=ok)
    assert mb == [b for b in mb_all if all(ok(n, l) for n, l in b)]
    assert len(mb) > 0 and max(len(b) for b in mb) == order
    assert Rnl == Rnl_all and Y == Y_all                                  # keep touches the bodies only


@pytest.mark.parametrize("S, order, md, maxl, rule", PREFILTER_CASES)
def test_embedding_spec_unchanged_by_prefilter(S, order, md, maxl, rule):
    """build_embedding_spec end to end against the post-filter reference."""
    widths = [2] * order
    sp = build_embedding_spec(S, order, md, widths, maxl=maxl, block_rule=rule)
    NZf = S if rule == "ace1" else 1
    ok = lambda n, l: (n - 1) % NZf == 0 and (maxl is None or l <= maxl)  # noqa: E731
    unfold = lambda n: (n - 1) // NZf + 1                                 # noqa: E731
    mb_f, _, _ = build_spec(NZf, order, md)
    mb1 = [[(unfold(n), l) for n, l in b] for b in mb_f if all(ok(n, l) for n, l in b)]
    ref = [[((n1 - 1) * 2 + k, l) for n1, l in b] for b in mb1 for k in (1, 2)]
    ref.sort(key=len)
    assert sp.mb == ref
