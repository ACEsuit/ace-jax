"""Collinear spin: H_up and H_dn are operator channels on one geometry, basis and set of
features.  Fixture: bcc Fe (2 atoms) from FHI-aims 261002, each spin channel converted by Quoll
(`ham/julia/ingest.jl fhi-aims-spin`); `bench/ham/spin_check.py` held the full calculation to
FHI-aims' own eigenvalues (5e-6 eV, its print precision) and electron count."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib
import shutil

import numpy as np
import pytest
from conftest import require_optional

from ace_jax.ham import cache, deeph
from ace_jax.ham.bond import BondBasis
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.fit import fit_offsite, fit_onsite
from ace_jax.ham.operator import HamiltonianPredictor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE, FIX = ROOT / "fixtures" / "ham" / "coupling_cache", ROOT / "fixtures" / "ham" / "fe_spin_deeph"
CH = ("H_up", "H_dn", "S")


@pytest.fixture(scope="module")
def data():
    require_optional("h5py")
    return deeph.read_spin(FIX)


def test_read_spin(data):
    assert tuple(data.blocks) == CH and list(data.numbers) == [26, 26]
    assert data.basis.ls(26) == (0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3) and data.basis.n_orb(26) == 31
    keys = set(data.blocks["S"])
    assert keys == set(data.blocks["H_up"]) == set(data.blocks["H_dn"]) and len(keys) == 30
    for name in CH:                                    # every bond's reverse is its transpose
        for (Rx, Ry, Rz, i, j), b in data.blocks[name].items():
            assert np.allclose(data.blocks[name][(-Rx, -Ry, -Rz, j, i)], b.T, atol=1e-10)
    on = (0, 0, 0, 0, 0)
    assert np.allclose(np.diag(data.blocks["S"][on]), 1.0, atol=1e-3)       # light integration grids
    # the exchange splitting: the two channels differ, most in the 3d shell (the majority lower)
    diff = np.diag(data.blocks["H_dn"][on] - data.blocks["H_up"][on])
    d3 = data.basis.shell_slice(26, 8)
    assert diff[d3].min() > 1.0 and np.abs(diff[d3]).mean() > np.abs(diff[:5]).mean()


def test_read_spin_refuses_two_structures(data, tmp_path):
    shutil.copytree(FIX, tmp_path / "x")
    pos = np.loadtxt(tmp_path / "x" / "dn" / "site_positions.dat")
    np.savetxt(tmp_path / "x" / "dn" / "site_positions.dat", pos + 0.01)
    with pytest.raises(ValueError, match="different structures"):
        deeph.read_spin(tmp_path / "x")


def test_cache_keeps_the_spin_channels(data, tmp_path):
    cache.save(tmp_path / "fe.npz", data)
    got, meta = cache.load(tmp_path / "fe.npz")
    assert meta["channels"] == list(CH)
    for name in CH:
        assert all(np.array_equal(v, got.blocks[name][k]) for k, v in data.blocks[name].items())


def test_fit_and_bands_per_spin_channel(data):
    """One design matrix for the three channels.  The on-site blocks of the two atoms are
    interpolated (it checks the path, not the model's accuracy); the bands of a channel are the
    eigenvalues of that channel's Bloch matrix, and the two channels' differ."""
    import scipy.linalg
    site = build_site_basis([26], order=1, totaldegree=40, lmax_feat=6, rcut=4.0, wL=0.5,
                            nmax_by_order=(6,), lmax_by_order=(6,), coupling_cache_dir=str(CACHE))
    on, _ = fit_onsite(site, [data], channels=CH)
    assert on.channels == CH
    pred = on.predict(data.positions, data.cell, data.numbers)
    for name, tol in (("H_up", 5e-3), ("H_dn", 5e-3), ("S", 2e-5)):          # H: 1s at -7000 eV, ridge 1e-10
        assert max(np.abs(pred[name][k] - data.blocks[name][k]).max() for k in pred[name]) < tol, name
    bond = BondBasis(site=site, rcut_bond=3.0, n_bond=6, n_cpl=2, l_cpl=2)
    off, _ = fit_offsite(bond, [data], channels=CH)
    hp = HamiltonianPredictor(on, off)
    prep = hp.prepare(data.positions, data.cell, data.numbers)
    k = np.array([0.1, 0.25, -0.2])
    S = np.asarray(hp.hk(prep, k, "S"))
    assert np.abs(S - S.conj().T).max() < 1e-10
    # the fixture keeps blocks to 3 angstrom only, where S(k) is not positive definite: the
    # eigenvalues are checked without the overlap
    e = {}
    for name in ("H_up", "H_dn"):
        H = np.asarray(hp.hk(prep, k, name))
        assert np.abs(H - H.conj().T).max() < 1e-8
        e[name] = np.asarray(hp.bands(prep, [k], overlap=None, channel=name))[0]
        assert np.allclose(e[name], scipy.linalg.eigvalsh(H), atol=1e-6)
    assert np.abs(e["H_up"] - e["H_dn"]).max() > 0.5
