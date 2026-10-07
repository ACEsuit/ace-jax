import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_basis_build import _primed_cache


def test_build_basis_equals_build_model(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec, build_basis, build_model
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    a = build_basis(BasisSpec(order=3, max_degree=10, elements=(14,), coupling_cache_dir=cache), seed=0)
    b = build_model([14], 3, 10, coupling_cache_dir=cache, seed=0)
    assert np.array_equal(np.asarray(a.model.a2b_matrix()), np.asarray(b.model.a2b_matrix()))
    assert np.array_equal(np.asarray(a.model.rnl_Wnlq), np.asarray(b.model.rnl_Wnlq))
    assert a.meta == b.meta


def test_build_basis_requires_elements():
    from ace_jax.basis.model import BasisSpec, build_basis
    with pytest.raises(ValueError, match="elements"):
        build_basis(BasisSpec(order=2, max_degree=4))


def test_basis_r0_is_mean_of_meta_r0():
    from ace_jax.basis.model import basis_r0
    assert basis_r0({"basis": {"r0": [[2.0, 3.0], [3.0, 4.0]]}}) == 3.0
    assert basis_r0({"basis": {"r0": 2.5}}) == 2.5
    assert basis_r0({}) is None


def test_symbols_and_numbers_are_equivalent(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec, build_basis
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    a = build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",), coupling_cache_dir=cache))
    assert list(a.meta["elements"]) == [14]


def test_default_radial_mode_is_onehot():
    """Frozen bases are what aj fit/aj basis build: onehot (R_n = P_n) fits several times
    better kept frozen than seeded glorot mixtures (24 vs 470 meV/atom on the Si tutorial),
    and does not depend on a random draw."""
    import inspect
    from ace_jax.basis.model import BasisSpec, build_model
    from ace_jax.cli import _parse
    assert BasisSpec(order=2, max_degree=6).radial_mode == "onehot"
    assert inspect.signature(build_model).parameters["radial_mode"].default == "onehot"
    assert _parse(["basis", "--elements", "Si", "--order", "2", "--max-degree", "6", "--out", "x.npz"]).radial_mode == "onehot"


CAPS = ((15, 6, 4, 3, 2, 2), (0, 4, 3, 2, 1, 0))       # pacemaker's nradmax_by_orders, lmax_by_orders


def test_order_caps_select_capped_bodies_and_their_radials():
    """Every kept body of correlation order nu respects nmax[nu], lmax[nu]; the radial spec is
    exactly the (n, l) the bodies use; dropping the caps gives the uncapped selection back; and
    a body excluded by the caps is admitted without them (the caps did select)."""
    from ace_jax.basis.spec import build_spec
    for NZ in (1, 2):
        mb, Rnl, Ylm = build_spec(NZ, 6, 8, 1.0, caps=CAPS)
        assert mb and all(len(b) <= 6 for b in mb)
        for bb in mb:
            nu = len(bb)
            assert all((n - 1) // NZ + 1 <= CAPS[0][nu - 1] and l <= CAPS[1][nu - 1] for n, l in bb)
        assert set(Rnl) == {tuple(b) for bb in mb for b in bb}
        assert max(l for l, _ in Ylm) == max(l for _, l in Rnl)
        free, _, _ = build_spec(NZ, 6, 8, 1.0)
        assert {tuple(map(tuple, b)) for b in mb} < {tuple(map(tuple, b)) for b in free}
    assert build_spec(1, 4, 12, 1.5, caps=None) == build_spec(1, 4, 12, 1.5)


def test_order_caps_short_lists_repeat_and_none_is_uncapped():
    from ace_jax.basis.spec import _caps, build_spec
    assert _caps(((6, 4), None), 4) == ([6, 4, 4, 4], [10 ** 9] * 4)
    mb, _, _ = build_spec(1, 3, 8, 1.0, caps=((8,), (0,)))
    assert all(l == 0 for bb in mb for _, l in bb)


def test_order_caps_reach_the_cli_basis_spec():
    from ace_jax.cli import _basis_spec, _parser
    a = _parser().parse_args(["basis", "--elements", "Si", "--order", "6", "--max-degree", "10", "--wL", "1",
                              "--nmax-by-order", "15,6,4,3,2,2", "--lmax-by-order", "0,4,3,2,1,0", "--out", "x.npz"])
    s = _basis_spec(a, embedding=None)
    assert s.nmax_by_order == CAPS[0] and s.lmax_by_order == CAPS[1]


def test_capped_model_records_caps_and_rebuilds_its_radial_degrees():
    """A capped build records its caps in meta['basis']; rnl_degrees rebuilds the same radial
    spec from meta (the learned-radial roughness weights depend on it)."""
    from conftest import require_coupling_lib
    require_coupling_lib()
    from ace_jax.basis.model import build_model
    from ace_jax.fit.radial_model import rnl_degrees
    b = build_model([14], 6, 8, wL=1.0, nmax_by_order=CAPS[0], lmax_by_order=CAPS[1])
    assert b.meta["basis"]["nmax_by_order"] == list(CAPS[0])
    assert len(rnl_degrees(b.meta)) == b.meta["n_rnl"] == len(b.Rnl_spec)
