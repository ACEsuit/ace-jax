"""eval.fs_model.FSModel: frozen sqrt-density term on a linear ACE model; npz round trip."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from conftest import FIXTURE_DIR
from ace_jax.eval import load

MODEL = FIXTURE_DIR / "si_ace_model.npz"
XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not (MODEL.exists() and XYZ.exists()), reason="missing fixtures")


def _fs(model, meta, P=2, seed=0):
    rng = np.random.default_rng(seed)
    NZ, D = len(meta["elements"]), meta["n_B"] + meta["n_pair"]
    mask = np.ones(D)
    return (1e-2 * rng.standard_normal((P, NZ, D)), rng.standard_normal((P, NZ)), mask, 1e-6)


def _atoms():
    # ace_jax.fit.xyz.read_extxyz returns a plain Frame NamedTuple, not an ASE
    # Atoms object; ase.io.read is fine here since only positions/cell/numbers
    # are needed, not the stored labels (see CLAUDE.md's extxyz pitfall).
    # Frame 0 of this fixture is a single isolated Si atom (no neighbours, so
    # X_i == 0 and the density term is identically zero -- degenerate for the
    # "density term is live" check, and too small for the two-atom forces
    # check below); frame 1 is the first Si2 dimer.
    import ase.io
    return ase.io.read(XYZ, 1)


def test_plain_npz_loads_plain_model():
    from ace_jax.eval.fs_model import FSModel
    m, _, _ = load(MODEL)
    assert not isinstance(m, FSModel)


def test_npz_roundtrip_and_calculator(tmp_path):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval.fs_model import FSModel
    base, meta, _ = load(MODEL)
    eta, d, mask, eps = _fs(base, meta)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=(eta, d, mask, eps))
    m, meta2, z = load(tmp_path / "fs.npz")
    assert isinstance(m, FSModel) and meta2["fs"] == {"P": 2, "F": "ssqrt", "eps": eps}
    ref = FSModel(base, jnp.asarray(eta), jnp.asarray(d), jnp.asarray(mask), eps)
    at = _atoms()
    at.calc = ACECalculator(tmp_path / "fs.npz")
    e_fs = at.get_potential_energy()
    at.calc = ACECalculator(ref, meta)
    np.testing.assert_allclose(at.get_potential_energy(), e_fs, rtol=1e-12)
    at.calc = ACECalculator(MODEL)
    assert abs(at.get_potential_energy() - e_fs) > 1e-6          # the density term is live
    import sys
    from conftest import ROOT
    sys.path.insert(0, str(ROOT / "bench/learn_radial/md"))
    from padded_calc import PaddedACECalculator
    at.calc = PaddedACECalculator(tmp_path / "fs.npz")
    np.testing.assert_allclose(at.get_potential_energy(), e_fs, rtol=1e-12)


def test_fsmodel_forces_are_energy_derivative(tmp_path):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.construct.export import patch_radial_npz
    base, meta, _ = load(MODEL)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=_fs(base, meta))
    at = _atoms()
    at.calc = ACECalculator(tmp_path / "fs.npz")
    F = at.get_forces()
    h = 1e-5
    for i, a in ((0, 0), (1, 2)):
        p = at.get_positions()
        p[i, a] += h; at.set_positions(p); ep = at.get_potential_energy()
        p[i, a] -= 2 * h; at.set_positions(p); em = at.get_potential_energy()
        p[i, a] += h; at.set_positions(p)
        np.testing.assert_allclose(F[i, a], -(ep - em) / (2 * h), atol=1e-6)


def test_edge_kind_switch_keeps_density(tmp_path):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval.edge_model import with_edge_a_kind
    from ace_jax.eval.fs_model import FSModel
    base, meta, _ = load(MODEL)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=_fs(base, meta))
    m, _, _ = load(tmp_path / "fs.npz")
    mm = with_edge_a_kind(m, "matmul")
    assert isinstance(mm, FSModel) and mm.edge_a_kind == "matmul"
    np.testing.assert_array_equal(np.asarray(mm.eta), np.asarray(m.eta))


def test_patch_without_fs_drops_stale_density(tmp_path):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval.fs_model import FSModel
    base, meta, _ = load(MODEL)
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base, fs=_fs(base, meta))
    patch_radial_npz(tmp_path / "fs.npz", tmp_path / "plain.npz", base)
    m, meta2, z = load(tmp_path / "plain.npz")
    assert not isinstance(m, FSModel) and "fs" not in meta2
    assert not any(k.startswith("fs_") for k in z.files)
