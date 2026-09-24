"""Embedded basis spec (ace_embedding_model): the :ace1 folded block rule, the
channel-widened radial spec n = (n'-1) d + k, and the channel-diagonal many-body
bodies with per-order widths -- against Julia-exported embedded models."""
import json
import pathlib

import numpy as np
import pytest

from ace_jax.construct import radial_ace1 as ra
from ace_jax.construct.spec import build_embedding_spec

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
