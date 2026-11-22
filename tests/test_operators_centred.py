"""Centred model (`operators.centred`): equivariance of the coefficients, the two fits, the
closed-form jackknife and the model file.  Synthetic data only."""
import jax

jax.config.update("jax_enable_x64", True)

import dataclasses
import pathlib

import equinox as eqx
import numpy as np
import pytest
from conftest import wigner

from ace_jax.operators import centred as C
from ace_jax.operators import io
from ace_jax.operators.data import CentredData, OperatorData
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import centred_jackknife, fit_centred, fit_onsite, overlap_root
from ace_jax.operators.onsite import OnsiteModel, parts_of, shell_pairs
from ace_jax.operators.orbitals import OrbitalBasis

CACHE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "coupling_cache"
AUX = OrbitalBasis.of({14: (0, 0, 1, 2, 3), 6: (0, 1, 1, 2)})
CH = ("rho", "spin")


@pytest.fixture(scope="module")
def basis():
    return build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))


def random_model(basis, seed=0, channels=CH, shells=AUX):
    rng = np.random.default_rng(seed)
    coef = {(z, s): rng.normal(size=(len(channels), C.n_columns(basis, l))) for z in shells.species for s, l in enumerate(shells.ls(z))}
    return C.CentredModel(basis=basis, coef=coef, shells=shells, channels=tuple(channels))


def structure(seed=0, n=5):
    rng = np.random.default_rng(seed)
    cell = 4.4 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    numbers = np.array([14, 6, 14, 6, 6, 14, 14, 6])[:n]
    return rng.uniform(size=(n, 3)) @ cell, cell, numbers


def spd(n, seed):
    """A random symmetric positive definite matrix with unit diagonal (an overlap)."""
    A = np.random.default_rng(seed).normal(size=(n, n))
    S = A @ A.T / n + 0.5 * np.eye(n)
    d = np.sqrt(np.diag(S))
    return S / d[:, None] / d[None]


def dataset(model, n_structures, noise=0.0, seed=0, n=8):
    """`CentredData`s with the coefficients of `model` (plus noise) and a random overlap each."""
    rng = np.random.default_rng(100 + seed)
    out, Ss = [], []
    for k in range(n_structures):
        pos, cell, numbers = structure(10 + seed + k, n)
        c = {ch: v + noise * rng.normal(size=v.shape) for ch, v in model.predict(pos, cell, numbers).items()}
        out.append(CentredData(cell, pos, numbers, model.shells, c))
        Ss.append(spd(len(next(iter(c.values()))), 50 + k))
    return out, Ss


def test_layout_and_predictor(basis):
    """The flat layout is atoms, then shells, then m; the compiled predictor gives the model's
    coefficients; a shell without coefficients is zero."""
    m = random_model(basis)
    pos, cell, numbers = structure()
    off = C.offsets(AUX, numbers)
    assert list(np.diff(off)) == [AUX.n_orb(z) for z in numbers]
    p = m.predict(pos, cell, numbers)
    q = C.CentredPredictor(m)(pos, cell, numbers)
    assert set(p) == set(CH) and all(p[c].shape == (off[-1],) for c in CH)
    assert all(np.allclose(p[c], q[c], rtol=1e-11, atol=1e-11) for c in CH)
    F = m.site_features(pos, cell, numbers)
    i, s = 2, 3                                              # a Si atom, its l = 2 shell
    sl = AUX.shell_slice(14, s)
    assert np.allclose(p["spin"][off[i] + sl.start:off[i] + sl.stop], m.coef[(14, s)][1] @ F[(2, 1)][i])
    assert np.allclose(p["rho"][off[i]], m.coef[(14, 0)][0, :-1] @ F[(0, 1)][i, :, 0] + m.coef[(14, 0)][0, -1])    # the constant column is last
    less = eqx.tree_at(lambda mm: mm.coef, m, {k: v for k, v in m.coef.items() if k != (14, s)})
    for pred in (less.predict, C.CentredPredictor(less)):
        assert not np.any(pred(pos, cell, numbers)["rho"][off[i] + sl.start:off[i] + sl.stop])
    with pytest.raises(KeyError):
        m.predict(pos, cell, np.where(numbers == 6, 32, numbers))


@pytest.mark.parametrize("improper", [False, True])
def test_equivariance(basis, improper):
    """Each shell's vector transforms with the Wigner matrix of its l; under the inversion it
    takes the sign (-1)^l (the improper Q's matrix has it)."""
    from scipy.spatial.transform import Rotation
    m = random_model(basis)
    pos, cell, numbers = structure(3)
    Q = Rotation.from_rotvec([0.4, -0.7, 0.3]).as_matrix() * (-1.0 if improper else 1.0)
    p0, p1 = m.predict(pos, cell, numbers), m.predict(pos @ Q.T, cell @ Q.T, numbers)
    off = C.offsets(AUX, numbers)
    for i, z in enumerate(numbers):
        for s, l in enumerate(AUX.ls(z)):
            sl = slice(off[i] + AUX.shell_slice(z, s).start, off[i] + AUX.shell_slice(z, s).stop)
            D = wigner(l, Q) if l else np.ones((1, 1))
            for c in CH:
                assert np.allclose(p1[c][sl], D @ p0[c][sl], atol=1e-9), (i, s, l, c)
    if improper:
        assert np.allclose(wigner(3, -np.eye(3)), -np.eye(7)) and np.allclose(wigner(2, -np.eye(3)), np.eye(5))


def test_translation_and_permutation(basis):
    m = random_model(basis)
    pos, cell, numbers = structure(4)
    p0 = m.predict(pos, cell, numbers)["rho"]
    assert np.allclose(m.predict(pos + np.array([0.3, -1.1, 2.0]), cell, numbers)["rho"], p0, atol=1e-10)
    perm = np.array([3, 0, 4, 1, 2])
    p1 = m.predict(pos[perm], cell, numbers[perm])["rho"]
    off0, off1 = C.offsets(AUX, numbers), C.offsets(AUX, numbers[perm])
    for new, old in enumerate(perm):
        assert np.allclose(p1[off1[new]:off1[new + 1]], p0[off0[old]:off0[old + 1]], atol=1e-10)


@pytest.fixture(scope="module")
def exact(basis):
    m = random_model(basis)
    return m, *dataset(m, 14)


@pytest.mark.parametrize("metric", ["plain", "overlap"])
def test_exact_recovery(basis, exact, metric):
    """Coefficients a random model generated are refitted to roundoff by both metrics: the
    training residual vanishes and a new structure is predicted."""
    m, data, Ss = exact
    fit, report = fit_centred(basis, data, channels=CH, metric=metric, overlaps=Ss, ridge=1e-14)
    assert set(fit.coef) == set(m.coef) and fit.channels == CH and fit.shells == AUX
    if metric == "overlap":
        assert report[("clipped",)] == (0,) * len(data)
    assert all(np.all(r[2] < 1e-6) for k, r in report.items() if k != ("clipped",))      # the report's residual is yy - c'c: sqrt(roundoff)
    pos, cell, numbers = structure(99, 8)
    a, b = m.predict(pos, cell, numbers), fit.predict(pos, cell, numbers)
    assert all(np.abs(a[c] - b[c]).max() < 1e-10 * np.abs(a[c]).max() for c in CH)


def test_overlap_metric_with_identity_is_plain(basis, exact):
    """S = 1: the joint design is the plain fit's designs side by side."""
    data, _ = dataset(random_model(basis), 14, noise=0.05)
    eye = [np.eye(len(d.coefficients["rho"])) for d in data]
    kw = dict(channels=CH, ridge=1e-8)
    a, ra = fit_centred(basis, data, metric="plain", **kw)
    b, rb = fit_centred(basis, data, overlaps=eye, **kw)                     # the default metric with overlaps
    assert rb[("overlap",)][1] == C.columns(basis, AUX)[1]
    assert all(np.allclose(a.coef[k], b.coef[k], rtol=1e-7, atol=1e-9) for k in a.coef)
    n = np.array([r[0] for k, r in ra.items()], float)
    rms = np.array([r[2] for k, r in ra.items()])
    assert np.allclose(np.sqrt((n[:, None] * rms ** 2).sum(0) / rb[("overlap",)][0]), rb[("overlap",)][2], rtol=1e-8)
    with pytest.raises(ValueError, match="overlap"):
        fit_centred(basis, data, metric="overlap", **kw)
    with pytest.raises(ValueError, match="not both"):
        fit_centred(basis, data, sigma_c=1.0, scaled=1e-6, **kw)


def residuals(model, data, Ss):
    """(plain, S-weighted) sums of squared residuals of `model` on the data."""
    out = np.zeros(2)
    for d, S in zip(data, Ss):
        p = model.predict(d.positions, d.cell, d.numbers)
        for c in model.channels:
            r = p[c] - d.coefficients[c]
            out += r @ r, r @ S @ r
    return out


def test_metrics_are_different_minimisers(basis):
    """Noisy targets and a random S: each fit has the smaller residual in its own metric."""
    data, Ss = dataset(random_model(basis, 1), 14, noise=0.3, seed=1)
    kw = dict(channels=CH, ridge=1e-10)
    plain, _ = fit_centred(basis, data, metric="plain", **kw)
    over, rep = fit_centred(basis, data, metric="overlap", overlaps=Ss, chunk=37, **kw)       # rows in several chunks per structure
    (pp, ps), (op, os_) = residuals(plain, data, Ss), residuals(over, data, Ss)
    assert pp < 0.999 * op and os_ < 0.999 * ps
    assert np.isclose(np.sqrt(os_ / rep[("overlap",)][0]), np.sqrt((rep[("overlap",)][2] ** 2).sum()), rtol=1e-8)
    lazy, _ = fit_centred(basis, data, overlaps=[lambda S=S: S for S in Ss], **kw)             # an overlap read when its rows are made
    assert all(np.allclose(lazy.coef[k], over.coef[k], rtol=1e-8, atol=1e-10) for k in over.coef)


def test_overlap_root():
    """W^T W = S; a singular S goes through the clipped eigen-decomposition and says so."""
    S = spd(30, 0)
    W, clipped = overlap_root(S)
    assert clipped == 0 and np.allclose(W.T @ W, S, atol=1e-12) and np.allclose(W, np.triu(W))
    V = np.random.default_rng(0).normal(size=(30, 27))
    S1 = V @ V.T                                             # rank 27: three dependent functions
    S1[0, 0] -= 1e-9                                         # and not positive semi-definite to roundoff
    for overwrite in (False, True):                          # in place: S comes back from the triangle LAPACK left
        W, clipped = overlap_root(S1.copy(), overwrite=overwrite)
        assert clipped == 3 and np.allclose(W.T @ W, S1, atol=1e-8 * np.abs(S1).max())
    Sf = np.asfortranarray(S.copy())
    W, clipped = overlap_root(Sf, overwrite=True)
    assert clipped == 0 and np.allclose(W.T @ W, S, atol=1e-12) and np.allclose(W, np.triu(W)) and np.shares_memory(W, Sf)
    with pytest.raises(ValueError, match="square"):
        overlap_root(np.zeros((3, 4)))


@pytest.mark.parametrize("metric,prior", [("overlap", {}), ("overlap", {"scaled": 1e-6}), ("plain", {"sigma_c": 3.0}), ("overlap", {"sigma_c": "evidence"})])
def test_jackknife_is_the_refits(basis, metric, prior):
    """The closed-form deviations are the coefficients of explicit refits without each structure,
    at the prior of the full fit (a prior scaled to the Gram diagonal: the full fit's values)."""
    data, Ss = dataset(random_model(basis, 2), 12, noise=0.2, seed=2)
    kw = dict(channels=CH, metric=metric, ridge=1e-6)
    model, _, st = fit_centred(basis, data, overlaps=Ss, stats=True, jackknife=True, **kw, **prior)
    jk = centred_jackknife(model, st)
    assert len(jk) == len(data) and all(s.jack.shape[2] == len(data) for s in st.values())
    if prior.get("sigma_c") == "evidence":                   # every right-hand side its own prior: one refit is not the reference
        assert all(s.hyper is not None and np.isfinite(s.jack).all() for s in st.values())
        return
    for g in (0, 5, 11):
        keep = [k for k in range(len(data)) if k != g]
        if "scaled" in prior:                                # the same prior precisions, not r times the smaller Gram's diagonal
            ref = fit_fixed(basis, [data[k] for k in keep], [Ss[k] for k in keep], st[("overlap",)].gamma, kw)
        else:
            ref, _ = fit_centred(basis, [data[k] for k in keep], overlaps=[Ss[k] for k in keep], **kw, **prior)
        for key in model.coef:
            d_ref, d_jk = ref.coef[key] - model.coef[key], jk[g][key] - model.coef[key]
            assert np.abs(d_jk - d_ref).max() < 1e-8 * np.abs(d_ref).max(), (g, key)

def fit_fixed(basis, data, Ss, gamma, kw):
    """The overlap fit with given prior precisions (through `scaled`: a function of the Gram's diagonal that ignores it)."""
    from ace_jax.operators import fit as F
    orig = F._fit
    try:
        F._fit = lambda *a, **k: orig(*a, **{**k, "gamma": {("overlap",): gamma}})
        return fit_centred(basis, data, overlaps=Ss, scaled=1.0, **kw)[0]
    finally:
        F._fit = orig


def test_onsite_jackknife_at_a_fixed_prior(basis):
    """`fit_onsite(jackknife=True)` without prior selection: the deviations are the refits."""
    orb = OrbitalBasis.of({14: (0, 1), 6: (0, 1)})
    rng = np.random.default_rng(0)
    coef = {(z, s1, s2, L): rng.normal(size=(1, basis.n(L, p) + (L == 0))) for z in orb.species for s1, s2, l1, l2 in shell_pairs(orb, z)
            for L, p in parts_of(l1, l2, s1 == s2)}
    gen = OnsiteModel(basis=basis, coef=coef, orbitals=orb, channels=("H",))
    data = []
    for k in range(10):
        pos, cell, numbers = structure(40 + k, 8)
        blocks = {key: v + 0.1 * (lambda e: e + e.T)(rng.normal(size=v.shape)) for key, v in gen.predict(pos, cell, numbers)["H"].items()}
        data.append(OperatorData(cell, pos, numbers, orb, {"H": blocks}))
    kw = dict(channels=("H",), ridge=1e-6)
    model, _, st = fit_onsite(basis, data, stats=True, jackknife=True, **kw)
    assert all(s.jack is not None and s.hyper is None and s.ridge == 1e-6 for s in st.values())
    ref, _ = fit_onsite(basis, data[1:], **kw)
    for s in st.values():
        for i, key in enumerate(s.keys):
            d_ref = ref.coef[key] - model.coef[key]
            assert np.abs(s.jack[i, :, 0] / np.sqrt(9 / 10) - d_ref[0]).max() < 1e-8 * np.abs(d_ref).max(), key
    _, _, plain = fit_onsite(basis, data, stats=True, **kw)
    assert all(s.jack is None for s in plain.values())


def gaussians(shells, seed=0):
    """Radial tables of analytic Gaussians r^l exp(-a r^2), one per shell."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.0, 6.0, 40)
    return {z: (r, np.stack([r ** l * np.exp(-rng.uniform(0.3, 2.0) * r ** 2) for l in ls])) for z, ls in shells.shells}


def test_bases_compare_by_shells():
    """Two bases with the same shells and different radial tables are equal and hash equal; `of`
    checks the tables' shape."""
    a, b, c = OrbitalBasis.of(dict(AUX.shells)), OrbitalBasis.of(dict(AUX.shells), gaussians(AUX, 1)), OrbitalBasis.of(dict(AUX.shells), gaussians(AUX, 2))
    assert a == b == c == AUX and not (b != c) and hash(a) == hash(b) == hash(c) and len({a, b, c}) == 1
    assert a.radials is None and b.radials[14][1].shape == (5, 40) and not np.array_equal(b.radials[6][1], c.radials[6][1])
    assert a != OrbitalBasis.of({14: (0, 0, 1, 2, 3)}) and a != dict(AUX.shells) and b.n_orb(6) == 12
    with pytest.raises(ValueError, match="radials"):
        OrbitalBasis.of({14: (0, 1)}, {14: (np.arange(4.0), np.zeros((3, 4)))})


@pytest.mark.parametrize("radials", [False, True])
def test_file_round_trip(basis, tmp_path, radials):
    """A file with a site basis and a centred model only: the loaded model predicts the saved
    one's coefficients bit for bit, and the radial functions come back when there are any."""
    shells = OrbitalBasis.of(dict(AUX.shells), gaussians(AUX)) if radials else AUX
    m = random_model(basis, shells=shells)
    path = tmp_path / "model.npz"
    io.save(path, centred=m, info={"fit": "random"})
    f = io.load(path)
    assert f.onsite is None and f.offsite is None and f.meta["kind"] == "operator" and f.meta["schema_version"] == 1
    assert f.meta["centred"] == {"shells": {"6": [0, 1, 1, 2], "14": [0, 0, 1, 2, 3]}, "channels": list(CH)} and f.meta["info"] == {"fit": "random"}
    assert f.centred.shells == AUX and f.centred.channels == CH and set(f.centred.coef) == set(m.coef)
    if radials:
        assert all(np.array_equal(f.centred.shells.radials[z][i], shells.radials[z][i]) for z in (6, 14) for i in (0, 1))
    else:
        assert f.centred.shells.radials is None and not any("radial" in k for k in np.load(path).files)
    pos, cell, numbers = structure(5)
    a, b = m.predict(pos, cell, numbers), f.centred.predict(pos, cell, numbers)
    assert all(np.array_equal(a[c], b[c]) for c in CH)
    a, b = C.CentredPredictor(m)(pos, cell, numbers), C.CentredPredictor(f.centred)(pos, cell, numbers)
    assert all(np.array_equal(a[c], b[c]) for c in CH)


def test_file_with_block_models(basis, tmp_path):
    """A centred model beside an on-site model, whose orbital basis carries radial functions;
    a file without the centred part (what every earlier version wrote) loads with `centred`
    None; another site basis is refused."""
    orb = OrbitalBasis.of({14: (0, 1), 6: (0, 1)}, gaussians(OrbitalBasis.of({14: (0, 1), 6: (0, 1)})))
    rng = np.random.default_rng(0)
    on = OnsiteModel(basis=basis, orbitals=orb, channels=("H", "S"), coef={
        (z, s1, s2, L): rng.normal(size=(2, basis.n(L, p) + (L == 0))) for z in orb.species for s1, s2, l1, l2 in shell_pairs(orb, z)
        for L, p in parts_of(l1, l2, s1 == s2)})
    m = random_model(basis, channels=("rho",))
    io.save(tmp_path / "both.npz", onsite=on, centred=m)
    f = io.load(tmp_path / "both.npz")
    assert f.onsite.channels == ("H", "S") and f.centred.channels == ("rho",) and f.centred.basis is f.onsite.basis
    assert np.array_equal(f.onsite.orbitals.radials[14][1], orb.radials[14][1]) and f.centred.shells.radials is None
    pos, cell, numbers = structure(6)
    assert np.array_equal(f.centred.predict(pos, cell, numbers)["rho"], m.predict(pos, cell, numbers)["rho"])
    io.save(tmp_path / "old.npz", onsite=dataclasses.replace(on, orbitals=OrbitalBasis.of({14: (0, 1), 6: (0, 1)})))
    z = np.load(tmp_path / "old.npz")
    assert not any(k.startswith("centred") or "radial_" in k for k in z.files)
    old = io.load(tmp_path / "old.npz")
    assert old.centred is None and old.onsite.orbitals.radials is None and len(old) == 4 and old[:3] == (old.onsite, old.offsite, old.meta)
    other = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=3.5, coupling_cache_dir=str(CACHE))
    with pytest.raises(ValueError, match="another site basis"):
        io.save(tmp_path / "bad.npz", onsite=on, centred=random_model(other, channels=("rho",)))
    with pytest.raises(ValueError, match="give"):
        io.save(tmp_path / "none.npz")
