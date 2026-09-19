"""Pin the npz orientation and the per-stage probe values.

The probe block exists so a mismatch localises to a stage (transform, envelope,
spline, Ylm) instead of only surfacing at the end in the site energy.
"""
import json
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)   # test-local, not library-level
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.eval.radial import (agnesi_normalized, env_ace1_poly1sr, env_poly2sx,
                           spline_eval)

@pytest.fixture
def loaded(npz):
    return load(npz)


def test_orientation(loaded):
    model, meta, z = loaded
    NZ = len(meta["elements"])
    assert model.WB.shape == (meta["n_B"], NZ)
    assert model.Wpair.shape == (meta["n_pair"], NZ)
    assert model.A2B.shape == (meta["n_B"], meta["n_AA"])
    # the radial branch is per-basis, so check whichever one this model populated
    if model.radial_kind == "spline_factorised":
        # stored once, not per species pair: (ncoef, n1) + (NZ, d)
        assert model.rnl_coefs_single.shape[0] == meta["rnl_spline"]["ncoef"]
        assert model.rnl_emb_nidx.shape == (meta["n_rnl"],)
    elif meta["radial_kind"] == "spline":
        assert model.rnl_coefs.shape == (NZ, NZ, meta["rnl_spline"]["ncoef"], meta["n_rnl"])
    else:
        n_q = model.polys_A.shape[0]
        assert model.rnl_Wnlq.shape == (NZ, NZ, meta["n_rnl"], n_q)
        assert model.polys_B.shape == (n_q,) and model.polys_C.shape == (n_q,)
    if meta["pair_radial_kind"] == "spline":
        assert model.pair_coefs.shape == (NZ, NZ, meta["pair_spline"]["ncoef"], meta["n_pair"])
    assert z["probe_Rnl"].shape == (len(z["probe_r"]), meta["n_rnl"])
    assert z["test_pos"].shape[0] == 3 and z["test_edge_rij"].shape[0] == 3


def test_probe_transform(loaded):
    model, meta, z = loaded
    r = jnp.asarray(z["probe_r"])
    x = agnesi_normalized(r, model.rnl_transform[0, 0])
    assert np.max(np.abs(np.asarray(x) - z["probe_x"])) < 1e-14


def test_branch_flags_are_consistent(loaded):
    """The flags actually select what they claim; a silently wrong branch here
    would still produce plausible numbers."""
    model, meta, z = loaded
    assert meta["radial_kind"] in ("spline", "analytic")
    # a frozen-embedding export stores the radial FACTORISED -- one
    # species-independent spline table plus the (NZ, d) embedding -- so the
    # loader reports a third kind that the metadata does not name
    if model.radial_kind == "spline_factorised":
        assert meta["radial_kind"] == "spline"
        assert model.rnl_coefs_single.ndim == 2
        assert model.rnl_embedding.shape[0] == len(meta["elements"])
        assert (model.rnl_emb_nidx.shape == model.rnl_emb_kidx.shape)
        return
    assert meta["pair_radial_kind"] in ("spline", "analytic")
    assert model.radial_kind == meta["radial_kind"]
    assert model.pair_radial_kind == meta["pair_radial_kind"]
    assert model.ysolid == (meta["ybasis_kind"] == "real_solidharmonics")
    assert model.pair_envelope_kind == meta["pair_envelope_kind"]


def test_probe_envelope(loaded):
    model, meta, z = loaded
    env = env_poly2sx(jnp.asarray(z["probe_x"]), model.rnl_envelope[0, 0])
    assert np.max(np.abs(np.asarray(env) - z["probe_env"])) < 1e-14


def test_probe_rnl(loaded):
    model, meta, z = loaded
    n = len(z["probe_r"])
    zi = jnp.zeros(n, jnp.int32)
    rij = jnp.asarray(z["probe_rij"].T)
    Rnl, Rpair = model.radial(rij, zi, zi)
    e_rnl = np.max(np.abs(np.asarray(Rnl) - z["probe_Rnl"]))
    e_pair = np.max(np.abs(np.asarray(Rpair) - z["probe_Rpair"]))
    assert e_rnl < 1e-11, f"Rnl {e_rnl:.2e}"
    assert e_pair < 1e-11, f"Rpair {e_pair:.2e}"


def test_probe_ylm(loaded):
    model, meta, z = loaded
    Y = model.angular(jnp.asarray(z["probe_rij"].T))
    assert np.max(np.abs(np.asarray(Y) - z["probe_Ylm"])) < 1e-11
