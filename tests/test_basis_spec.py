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
