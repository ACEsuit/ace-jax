"""radial_init: the n_q_factor knob and the from_table projector."""
import math

import numpy as np
import pytest

from ace_jax.basis.radial_init import tensor_radial_init
from ace_jax.basis.spec import build_spec


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
    from test_basis_build import _primed_cache
    from ace_jax.basis.model import build_model
    _, maxn = _rnl()
    auth = build_model([14], 3, 10, coupling_cache_dir=_primed_cache(tmp_path),
                       radial_mode="onehot", n_q_factor=3.0)
    assert auth.model.rnl_Wnlq.shape[-1] == math.ceil(3.0 * maxn)
    assert auth.model.polys_A.shape == (math.ceil(3.0 * maxn),)
    assert auth.meta["basis"]["n_q_factor"] == 3.0


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, float("nan")])
def test_n_q_factor_below_one_raises(bad):
    from ace_jax.basis.model import build_model
    Rnl, _ = _rnl()
    with pytest.raises(ValueError, match="n_q_factor"):
        tensor_radial_init([14], Rnl, rcut=5.5, n_q_factor=bad)
    with pytest.raises(ValueError, match="n_q_factor"):   # before any coupling work
        build_model([14], 3, 10, coupling_cache=False, n_q_factor=bad)


from conftest import FIXTURE_DIR


def _analytic_fixture():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.eval import load
    p = FIXTURE_DIR / "si_ace_model.npz"
    if not p.exists():
        pytest.skip("missing si_ace_model.npz")
    m, meta, _ = load(p)
    return m


def _sample(m, x):
    """env(x) P(x) W^T per pair, numpy, (NZ, NZ, n_x, n_rnl)."""
    from ace_jax.basis.radial_init import envelope2sx_eval, poly_eval
    W = np.asarray(m.rnl_Wnlq)
    P = poly_eval(x, np.asarray(m.polys_A), np.asarray(m.polys_B), np.asarray(m.polys_C))
    NZ = W.shape[0]
    env = np.asarray(m.rnl_envelope)
    return np.stack([np.stack([envelope2sx_eval(None, x, env[i, j])[:, None] * P @ W[i, j].T
                               for j in range(NZ)]) for i in range(NZ)])


def test_from_table_roundtrip_recovers_Wnlq():
    from ace_jax.basis.radial_init import from_table
    m = _analytic_fixture()
    x = np.linspace(-1, 1, 801)
    R = _sample(m, x)
    polys = tuple(np.asarray(a) for a in (m.polys_A, m.polys_B, m.polys_C))
    W, rel = from_table(x, R, np.asarray(m.rnl_envelope), polys)
    Wt = np.asarray(m.rnl_Wnlq)
    assert np.abs(W - Wt).max() < 1e-10 * np.abs(Wt).max()
    assert rel.max() < 1e-10


def test_from_table_zero_radial_has_zero_residual():
    from ace_jax.basis.radial_init import from_table
    m = _analytic_fixture()
    x = np.linspace(-1, 1, 201)
    R = _sample(m, x)
    R[..., 0] = 0.0
    polys = tuple(np.asarray(a) for a in (m.polys_A, m.polys_B, m.polys_C))
    W, rel = from_table(x, R, np.asarray(m.rnl_envelope), polys)
    assert np.all(W[..., 0, :] == 0.0) and np.all(rel[..., 0] == 0.0)
