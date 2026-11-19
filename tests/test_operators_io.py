"""Operator models on disk (`operators.io`): the blocks of a loaded model are the saved model's."""
import jax

jax.config.update("jax_enable_x64", True)

import json
import pathlib

import numpy as np
import pytest

from ace_jax.basis import cg
from ace_jax.operators import bond, io
from ace_jax.operators.features import build_embedding_site_basis, build_site_basis
from ace_jax.operators.offsite import OffsiteModel, canonical_pairs
from ace_jax.operators.onsite import OnsiteModel, parts_of, shell_pairs
from ace_jax.operators.orbitals import OrbitalBasis

CACHE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "coupling_cache"
ORB = OrbitalBasis.of({14: (0, 1, 1, 2), 6: (0, 1)})
CH = ("H", "S")


def models(bb, seed=0):
    rng = np.random.default_rng(seed)
    on, off = {}, {}
    for z in ORB.species:
        for s1, s2, l1, l2 in shell_pairs(ORB, z):
            for L, p in parts_of(l1, l2, s1 == s2):
                if bb.site.n(L, p) + (L == 0):
                    on[(z, s1, s2, L)] = rng.normal(size=(2, bb.site.n(L, p) + (L == 0)))
        for zj in ORB.species:
            for s1, s2, l1, l2, same in canonical_pairs(ORB, z, zj):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                for L in cg.Ls(l1, l2):
                    k0, k1 = bond.n_features(bb, L, p)
                    n = k0 + (k1 if same else 2 * k1) + bond.n_env_features(bb, L, p, same=same)
                    if n:
                        off[(z, zj, s1, s2, L)] = rng.normal(size=(2, n))
    return OnsiteModel(basis=bb.site, coef=on, orbitals=ORB, channels=CH), OffsiteModel(bond=bb, coef=off, orbitals=ORB, channels=CH)


def structure():
    rng = np.random.default_rng(0)
    cell = 4.6 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    return rng.uniform(size=(4, 3)) @ cell, cell, np.array([14, 6, 14, 6])


def site(kind):
    if kind == "embedding":
        return build_embedding_site_basis([14, 6, 32], 1, 6, 3, "identity", rcut=4.0, wL=0.5, coupling_cache_dir=str(CACHE))
    sb = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    return sb.with_table(512) if kind == "table" else sb


@pytest.mark.parametrize("kind", ["tensor", "table", "embedding"])
def test_round_trip_predicts_the_same_blocks(tmp_path, kind):
    """Every radial kind, with the bond environment (an embedding of the environment species for
    the element-embedding basis): the loaded models give the saved models' blocks bit for bit,
    and the file carries the schema's meta."""
    sb = site(kind)
    emb = tuple(tuple(float(v) for v in r) for r in np.asarray(sb.radial.rnl_embedding)) if kind == "embedding" else None
    bb = bond.BondBasis(site=sb, rcut_bond=5.0, n_bond=4, n_cpl=2, l_cpl=2, r_env=4.5, n_env=3, l_env=2, n_ebond=2, l_ebond=2, env_emb=emb)
    on, off = models(bb)
    path = tmp_path / "model.npz"
    io.save(path, on, off, info={"fit": "random", "n_train": 0})
    f = io.load(path)
    assert f.meta["schema_version"] == 1 and f.meta["kind"] == "operator" and f.meta["info"] == {"fit": "random", "n_train": 0}
    assert f.onsite.orbitals == ORB and f.offsite.channels == CH and f.offsite.bond.site is f.onsite.basis
    assert f.onsite.basis.gamma == sb.gamma and f.offsite.bond._replace(site=None) == bb._replace(site=None)
    pos, cell, numbers = structure()
    for a, b in ((on, f.onsite), (off, f.offsite)):
        pa, pb = a.predict(pos, cell, numbers), b.predict(pos, cell, numbers)
        assert all(pa[c].keys() == pb[c].keys() and all(np.array_equal(pa[c][k], pb[c][k]) for k in pa[c]) for c in CH)
    z = np.load(path)
    assert json.loads(bytes(z["meta_json"]).decode())["site"]["radial"]["kind"] == ("table" if kind == "table" else "tensor")


def test_one_model_and_refusals(tmp_path):
    """A file with only one of the two models; two models of different site bases are refused,
    and so is a file that is not an operator model."""
    sb = site("tensor")
    bb = bond.BondBasis(site=sb, rcut_bond=5.0, n_bond=4)
    on, off = models(bb)
    io.save(tmp_path / "on.npz", onsite=on)
    io.save(tmp_path / "off.npz", offsite=off)
    a, b = io.load(tmp_path / "on.npz"), io.load(tmp_path / "off.npz")
    assert a.offsite is None and b.onsite is None and a.meta["bond"] is None and sorted(b.offsite.coef) == sorted(off.coef)
    with pytest.raises(ValueError, match="differ"):
        io.save(tmp_path / "x.npz", OnsiteModel(basis=sb.with_table(64), coef=on.coef, orbitals=ORB, channels=CH), off)
    with pytest.raises(ValueError, match="give"):
        io.save(tmp_path / "x.npz")
    np.savez(tmp_path / "p.npz", meta_json=np.frombuffer(json.dumps({"schema_version": 1}).encode(), np.uint8))
    with pytest.raises(ValueError, match="not an operator model"):
        io.load(tmp_path / "p.npz")
