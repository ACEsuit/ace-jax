"""radial_init: the n_q_factor knob and the from_table projector."""
import math

import numpy as np
import pytest

from ace_jax.construct.radial_init import tensor_radial_init
from ace_jax.construct.spec import build_spec


def _rnl():
    _, Rnl, _ = build_spec(1, 3, 10, 1.5)
    return Rnl, max(n for n, _ in Rnl)


def test_tensor_radial_init_default_n_q_unchanged():
    Rnl, maxn = _rnl()
    d = tensor_radial_init([14], Rnl, rcut=5.5)
    assert d["rnl_Wnlq"].shape[-1] == math.ceil(1.5 * maxn) == len(d["polys_A"])


def test_tensor_radial_init_n_q_factor_widens_and_keeps_onehot():
    Rnl, maxn = _rnl()
    d1 = tensor_radial_init([14], Rnl, rcut=5.5, mode="onehot")
    d3 = tensor_radial_init([14], Rnl, rcut=5.5, mode="onehot", n_q_factor=3.0)
    n1 = d1["rnl_Wnlq"].shape[-1]
    assert d3["rnl_Wnlq"].shape[-1] == math.ceil(3.0 * maxn) == len(d3["polys_A"])
    np.testing.assert_array_equal(d3["rnl_Wnlq"][..., :n1], d1["rnl_Wnlq"])
    assert not d3["rnl_Wnlq"][..., n1:].any()
    np.testing.assert_allclose(d3["polys_A"][:n1], d1["polys_A"], rtol=0, atol=0)


def test_build_model_n_q_factor(tmp_path):
    from test_python_authoring import _primed_cache
    from ace_jax.construct.model import build_model
    _, maxn = _rnl()
    auth = build_model([14], 3, 10, coupling_cache_dir=_primed_cache(tmp_path),
                       radial_mode="onehot", n_q_factor=3.0)
    assert auth.model.rnl_Wnlq.shape[-1] == math.ceil(3.0 * maxn)
    assert auth.model.polys_A.shape == (math.ceil(3.0 * maxn),)
    assert auth.meta["authoring"]["n_q_factor"] == 3.0
