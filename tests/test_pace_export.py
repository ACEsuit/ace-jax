import pathlib
import equinox as eqx
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase import Atoms

from ace_jax.eval import ACECalculator, load
from ace_jax.eval.pace_io import load_tree, write_yace

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
NAMES = ["si_chebpow_fs", "gesi_sbessel", "sige_zbl"]


def _need(name):
    p = FIX / f"{name}.yace"
    if not p.exists():
        pytest.skip("fixtures not generated")
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
    pyace = pytest.importorskip("pyace")
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
