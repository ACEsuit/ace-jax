"""The compiled design rows of the off-site fit (`operators.design`) against the NumPy reference
(`bond.features`, `bond.design`), and the fit on them against the reference fit."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import numpy as np
import pytest
from conftest import require_optional

from ace_jax.ham import deeph
from ace_jax.operators import bond
from ace_jax.operators.design import bond_designs, qr_merge
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_offsite
from ace_jax.operators.onsite import _features, site_graph

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE, FIX = ROOT / "fixtures" / "ham" / "coupling_cache", ROOT / "fixtures" / "ham" / "sic_deeph"


@pytest.fixture(scope="module")
def basis():
    site = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    return bond.BondBasis(site=site, rcut_bond=4.2, n_bond=3, n_cpl=2, l_cpl=2, r_env=4.5, n_env=3, l_env=2, n_ebond=2, l_ebond=2)


def structure(seed=0):
    rng = np.random.default_rng(seed)
    cell = 4.6 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    return rng.uniform(size=(4, 3)) @ cell, cell, np.array([14, 6, 14, 6])


@pytest.mark.parametrize("env", [True, False])
def test_compiled_designs_are_the_reference(basis, env):
    """Every symmetry, both the tied (equal shells) and the untied form, with padding bonds."""
    b = basis if env else basis._replace(n_env=0)
    pos, cell, numbers = structure()
    F = _features(b.site, *site_graph(b.site, pos, cell, numbers))
    g = bond.bond_graph(b, pos, cell, numbers)
    lists = bond.env_lists(b, pos, cell, numbers=numbers)
    sel = np.arange(0, len(g.i), 3)
    lps = [(L, p) for L in range(5) for p in (1, -1)]
    ref = bond.features(b, {k: np.asarray(v) for k, v in F.items()}, g, sel, lps, env=lists)
    keys = [(L, p, same) for L, p in lps for same in (False, True)]
    pad = np.concatenate([sel, np.full(5, sel[0])])
    mask = np.concatenate([np.ones(len(sel)), np.zeros(5)])
    got = bond_designs(b, keys)(F, g.rij[pad], g.i[pad], g.j[pad], mask, lists)
    seen = 0
    for (L, p, same), A in zip(keys, got):
        want = np.swapaxes(bond.design(ref[(L, p)], L, same), 1, 2).reshape(len(sel) * (2 * L + 1), -1)
        A = np.asarray(A)
        assert A.shape == ((len(sel) + 5) * (2 * L + 1), want.shape[1])
        assert not A[len(want):].any()                                   # padding: zero rows
        if want.size:
            seen += 1
            assert np.abs(A[:len(want)] - want).max() < 1e-12 * max(1.0, np.abs(want).max()), (L, p, same)
    assert seen >= 12


def test_qr_merge_is_the_factor_of_the_stacked_rows():
    rng = np.random.default_rng(0)
    A1, A2, Y1, Y2 = rng.normal(size=(40, 6)), rng.normal(size=(25, 6)), rng.normal(size=(40, 2)), rng.normal(size=(25, 2))
    R = qr_merge(qr_merge(np.zeros((8, 8)), A1, Y1), A2, Y2)
    M = np.concatenate([np.concatenate([A1, Y1], 1), np.concatenate([A2, Y2], 1)])
    assert np.allclose(np.asarray(R).T @ np.asarray(R), M.T @ M, atol=1e-10)
    assert np.allclose(np.tril(np.asarray(R), -1), 0.0)


def test_compiled_fit_is_the_reference_fit(basis):
    """SiC fixture, H and S, chunks smaller than the bond count: the same predictions and
    statistics from the compiled rows merged on the device as from the NumPy rows merged on the host."""
    require_optional("h5py")
    d = deeph.read(FIX)
    b = basis._replace(rcut_bond=3.3, r_env=3.5)
    a, ra, sa = fit_offsite(b, [d], ridge=1e-8, chunk=16, stats=True)
    r, rr, sr = fit_offsite(b, [d], ridge=1e-8, chunk=16, stats=True, compiled=False)
    assert set(a.coef) == set(r.coef) and set(sa) == set(sr)
    for key in sa:
        ga, gr = np.asarray(sa[key].factor.G), np.asarray(sr[key].factor.G)
        assert np.allclose(ga, gr, atol=1e-9 * np.abs(gr).max() + 1e-18) and float(sa[key].factor.n) == float(sr[key].factor.n)
        assert np.allclose(sa[key].factor.b, sr[key].factor.b, atol=1e-9 * max(1.0, np.abs(np.asarray(sr[key].factor.b)).max()))
        assert np.allclose(sa[key].factor.yy, sr[key].factor.yy)
    pa, pr = a.predict(d.positions, d.cell, d.numbers), r.predict(d.positions, d.cell, d.numbers)
    for c in ("H", "S"):
        assert max(np.abs(pa[c][k] - pr[c][k]).max() for k in pr[c]) < 1e-6 * max(np.abs(v).max() for v in pr[c].values())
    assert all(ra[k][:2] == rr[k][:2] for k in rr)


def test_smoothness_prior_lines_up_with_the_design_columns(basis):
    """`bond.smoothness` has one gamma per column of `bond.design`, for the tied and untied forms;
    it is the potential fit's functional on the factors of each feature (two-centre: bond radial
    and L; coupled: plus the site feature's gamma; environment: plus its radial and lam, the same
    in every species channel), and a fit with it shrinks the rough columns' coefficients."""
    from ace_jax.basis.prior import P, WL
    pos, cell, numbers = structure()
    b = basis
    F = {k: np.asarray(v) for k, v in _features(b.site, *site_graph(b.site, pos, cell, numbers)).items()}
    g = bond.bond_graph(b, pos, cell, numbers)
    lps = [(L, p) for L in range(5) for p in (1, -1)]
    f = bond.features(b, F, g, np.arange(6), lps, env=bond.env_lists(b, pos, cell, numbers=numbers))
    for L, p in lps:
        for same in (False, True):
            gam = bond.smoothness(b, L, p, same)
            assert gam.shape == (bond.design(f[(L, p)], L, same).shape[1],) and (gam > 0).all()
    assert np.allclose(bond.smoothness(b, 2, 1, False)[:b.n_bond], [(2 / WL) ** P + n ** P for n in (1, 2, 3)])
    two, cpl = bond.layout(b, 2, 1)
    l, lam, pi, nB = cpl[0]
    first = bond.smoothness(b, 2, 1, False)[b.n_bond:b.n_bond + nB]                 # bond radial 1 with the first group
    assert np.allclose(first, (l / WL) ** P + 1 + b.site.smoothness(lam, pi))
    env = bond.smoothness(b, 2, 1, False)[-bond.n_env_features(b, 2, 1):].reshape(-1, b.n_ebond, 2, b.n_env)
    assert np.allclose(env[:, :, 0], env[:, :, 1])                                 # the two species channels
    assert (np.diff(env, axis=3) > 0).all() and (np.diff(env, axis=1) > 0).all()   # rougher with either radial


def test_fit_with_the_smoothness_prior(basis):
    require_optional("h5py")
    d = deeph.read(FIX)
    b = basis._replace(rcut_bond=3.3, r_env=3.5)
    plain, _ = fit_offsite(b, [d], channels=("H",), ridge=1e-8)
    smooth, _ = fit_offsite(b, [d], channels=("H",), ridge=1e-8, sigma_c=1e2)
    ref, _ = fit_offsite(b, [d], channels=("H",), ridge=1e-8, sigma_c=1e2, compiled=False)
    weak, _ = fit_offsite(b, [d], channels=("H",), ridge=1e-8, sigma_c=1e12)
    norm = lambda m: np.sqrt(sum((np.asarray(c) ** 2).sum() for c in m.coef.values()))
    assert norm(smooth) < 0.8 * norm(plain)                                        # the prior shrinks the coefficients
    pw, pp = weak.predict(d.positions, d.cell, d.numbers)["H"], plain.predict(d.positions, d.cell, d.numbers)["H"]
    assert max(np.abs(pw[k] - pp[k]).max() for k in pp) < 1e-5                      # and vanishes as sigma_c grows
    ps, pr = smooth.predict(d.positions, d.cell, d.numbers)["H"], ref.predict(d.positions, d.cell, d.numbers)["H"]
    assert max(np.abs(ps[k] - pr[k]).max() for k in pr) < 1e-7
