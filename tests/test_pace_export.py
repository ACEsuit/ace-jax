import pathlib
import equinox as eqx
import jax
import numpy as np
import pytest
from conftest import pace_fixture

jax.config.update("jax_enable_x64", True)
from ase import Atoms

from ace_jax.eval import ACECalculator, load
from ace_jax.eval.pace_io import load_tree, write_yace

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
NAMES = ["si_chebpow_fs", "gesi_sbessel", "sige_zbl"]


def _need(name):
    p = FIX / f"{name}.yace"
    pace_fixture(p)
    return p


@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_tree_equal(name, tmp_path):
    p = _need(name)
    model, _, spec = load(str(p))
    out = tmp_path / "rt.yace"
    write_yace(model, spec, out)
    assert load_tree(out) == load_tree(p)


@pytest.mark.parametrize("name", NAMES)
def test_modified_model_roundtrips(name, tmp_path):
    p = _need(name)
    model, meta, spec = load(str(p))
    m2 = eqx.tree_at(lambda m: m.ctilde_complex, model, 2.0 * model.ctilde_complex)
    m2 = eqx.tree_at(lambda m: m.crad, m2, 0.9 * m2.crad)
    out = tmp_path / "mod.yace"
    write_yace(m2, spec, out)
    ref = np.load(FIX / f"{name}_ref.npz")
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
    at.calc = ACECalculator(m2, meta=meta)
    E_mem = at.get_potential_energy()
    at.calc = ACECalculator(str(out))
    assert at.get_potential_energy() == pytest.approx(E_mem, abs=1e-12)


def test_modified_model_matches_pyace(tmp_path):
    import os
    # under ACEJAX_REQUIRE_FIXTURES (the pace venv in CI) a missing pyace is a failure
    pyace = (__import__("pyace") if os.environ.get("ACEJAX_REQUIRE_FIXTURES")
             else pytest.importorskip("pyace"))
    p = _need("sige_zbl")
    model, meta, spec = load(str(p))
    m2 = eqx.tree_at(lambda m: m.ctilde_complex, model, 1.5 * model.ctilde_complex)
    out = tmp_path / "mod.yace"
    write_yace(m2, spec, out)
    ref = np.load(FIX / "sige_zbl_ref.npz")
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
    at.calc = pyace.PyACECalculator(str(out))
    E_cpp = at.get_potential_energy()
    at.calc = ACECalculator(m2, meta=meta)
    assert at.get_potential_energy() == pytest.approx(E_cpp, abs=1e-6 * len(at))


def test_padded_crad_entries_survive_export(tmp_path):
    """gesi_sbessel: Ge-Ge nradbasemax 6 < global 8, yet g_6, g_7 are live on
    that bond, so edited padded crad entries must reach the file."""
    p = _need("gesi_sbessel")
    model, meta, spec = load(str(p))
    m2 = eqx.tree_at(lambda m: m.crad, model, model.crad.at[0, 0, :, :, 6:].add(0.01))
    out = tmp_path / "pad.yace"
    write_yace(m2, spec, out)
    ref = np.load(FIX / "gesi_sbessel_ref.npz")
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
    at.calc = ACECalculator(m2, meta=meta)
    E_mem = at.get_potential_energy()
    at.calc = ACECalculator(str(out))
    assert at.get_potential_energy() == pytest.approx(E_mem, abs=1e-10)


def test_write_yace_warns_when_leaves_are_not_float64(tmp_path):
    import jax.numpy as jnp
    p = _need("si_chebpow_fs")
    model, _, spec = load(str(p), dtype=jnp.float32)
    with pytest.warns(UserWarning, match="float32"):
        write_yace(model, spec, tmp_path / "f32.yace")
