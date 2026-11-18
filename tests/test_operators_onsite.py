"""On-site model: equivariance and symmetry of the blocks, and the block fit on real data."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import numpy as np
import pytest
from conftest import require_optional

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.basis import cg
from ace_jax.ham import deeph
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_onsite
from ace_jax.operators.onsite import OnsiteModel, parts_of, shell_pairs
from ace_jax.operators.orbitals import OrbitalBasis

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE, FIX = ROOT / "fixtures" / "ham" / "coupling_cache", ROOT / "fixtures" / "ham" / "sic_deeph"
ORB = OrbitalBasis.of({14: (0, 0, 1, 1, 2), 6: (0, 1, 2)})


@pytest.fixture(scope="module")
def basis():
    return build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))


def random_model(basis, seed=0):
    rng = np.random.default_rng(seed)
    coef = {}
    for z in ORB.species:
        for s1, s2, l1, l2 in shell_pairs(ORB, z):
            for L, p in parts_of(l1, l2, s1 == s2):
                n = basis.n(L, p) + (L == 0)
                if n:
                    coef[(z, s1, s2, L)] = rng.normal(size=(2, n))
    return OnsiteModel(basis=basis, coef=coef, orbitals=ORB, channels=("H", "S"))


def structure(seed=0):
    rng = np.random.default_rng(seed)
    cell = 4.4 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    numbers = np.array([14, 6, 14, 6, 6])
    return rng.uniform(size=(5, 3)) @ cell, cell, numbers


def wigner(L, Q):
    pts = np.random.default_rng(1).normal(size=(80, 3))
    sl = slice(L * L, (L + 1) ** 2)
    Y0 = np.asarray(real_spherical_harmonics(pts, max(L, 1)))[:, sl]
    Y1 = np.asarray(real_spherical_harmonics(pts @ Q.T, max(L, 1)))[:, sl]
    return np.linalg.lstsq(Y0, Y1, rcond=None)[0].T


def atom_rotation(z, Q):
    """Block-diagonal D over the shells of species z: orbitals rotate as their harmonics."""
    import scipy.linalg
    return scipy.linalg.block_diag(*[wigner(l, Q) if l else np.ones((1, 1)) for l in ORB.ls(z)])


def test_parts():
    assert parts_of(1, 1, True) == [(0, 1), (2, 1)]                       # same shell: symmetric, even L
    assert parts_of(1, 1, False) == [(0, 1), (1, 1), (2, 1)]              # (1, +1): the axial, antisymmetric part
    assert parts_of(0, 1, False) == [(1, -1)] and parts_of(1, 2, False) == [(1, -1), (2, -1), (3, -1)]
    assert [(a, b) for a, b, _, _ in shell_pairs(ORB, 6)] == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_blocks_are_symmetric_and_shaped(basis):
    pos, cell, numbers = structure()
    out = random_model(basis).predict(pos, cell, numbers)
    assert set(out) == {"H", "S"} and set(out["H"]) == {(0, 0, 0, i, i) for i in range(5)}
    for name in out:
        for (_, _, _, i, _), B in out[name].items():
            n = ORB.n_orb(numbers[i])
            assert B.shape == (n, n) and np.allclose(B, B.T, atol=1e-13) and np.abs(B).max() > 1e-3
    assert not np.allclose(out["H"][(0, 0, 0, 0, 0)], out["S"][(0, 0, 0, 0, 0)])


@pytest.mark.parametrize("improper", [False, True])
def test_blocks_rotate_with_the_orbitals(basis, improper):
    """H(Q R) = D(Q) H(R) D(Q)^T for a rotation, and with the orbital parities for an inversion;
    random coefficients, so every part (pseudo-tensor parts included) is exercised."""
    pos, cell, numbers = structure()
    m = random_model(basis)
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    a = m.predict(pos, cell, numbers)
    b = m.predict(pos @ Q.T * (-1 if improper else 1), cell @ Q.T * (-1 if improper else 1), numbers)
    for name in a:
        for key, B in a[name].items():
            z = int(numbers[key[3]])
            D = atom_rotation(z, Q)
            if improper:
                D = D * np.concatenate([[(-1) ** l] * (2 * l + 1) for l in ORB.ls(z)])[:, None]
            assert np.abs(b[name][key] - D @ B @ D.T).max() < 1e-9 * np.abs(B).max(), (name, key)


def test_isolated_atom_is_the_constant_part(basis):
    """No neighbour within the cutoff: only the L = 0 constants remain, so each
    same-l block is a multiple of the identity and the rest vanish."""
    m = random_model(basis)
    B = m.predict(np.zeros((1, 3)), 50 * np.eye(3), np.array([6]))["H"][(0, 0, 0, 0, 0)]
    for s1, s2, l1, l2 in shell_pairs(ORB, 6):
        blk = B[ORB.shell_slice(6, s1), ORB.shell_slice(6, s2)]
        if l1 == l2:
            T0 = cg.real_cg(l1, l2, 0)[:, :, 0]                      # +-identity / sqrt(2l+1)
            assert np.allclose(blk, m.coef[(6, s1, s2, 0)][0, -1] * T0, atol=1e-13)
            assert np.allclose(np.abs(T0), np.eye(2 * l1 + 1) / np.sqrt(2 * l1 + 1), atol=1e-14)
        else:
            assert not blk.any()


def test_species_without_orbitals(basis):
    pos, cell, numbers = structure()
    with pytest.raises(KeyError, match="14"):
        OnsiteModel(basis=basis, coef={}, orbitals=OrbitalBasis.of({6: (0,)}), channels=("H",)).predict(
            pos, cell, numbers)


@pytest.fixture(scope="module")
def basis_spdf():
    """l up to 6 as single-particle functions (wL = 1), so every part of an spdf block has features."""
    return build_site_basis([14, 6], order=2, totaldegree=7, lmax_feat=6, rcut=4.0, wL=1.0,
                            coupling_cache_dir=str(CACHE))


def test_fit_reproduces_real_onsite_blocks(basis_spdf):
    """SiC fixture (4 atoms, s, p, d and f shells): the fitted model returns the on-site H and S
    blocks.  With two equivalent atoms per species this is interpolation, so it checks the
    whole path on real data (conventions, design against readout, channels, transposed shell
    pairs, the pseudo-tensor parts), not the model's accuracy."""
    require_optional("h5py")
    d = deeph.read(FIX)
    model, report = fit_onsite(basis_spdf, [d])
    assert model.channels == ("H", "S") and model.orbitals == d.basis
    assert all(cols > 0 for _, cols, _ in report.values())
    pred = model.predict(d.positions, d.cell, d.numbers)
    for name, tol in (("H", 1e-6), ("S", 1e-9)):
        err = max(np.abs(pred[name][k] - d.blocks[name][k]).max() for k in pred[name])
        assert err < tol, (name, err)
    assert pred["H"][(0, 0, 0, 0, 0)][0, 0] < -1700 and abs(pred["S"][(0, 0, 0, 0, 0)][0, 0] - 1) < 1e-3


def test_parts_without_features_are_reported(basis):
    """The small basis has no L >= 4 feature: those parts get no coefficients, are predicted
    as zero, and the report gives the size of what is missed.  (Some parts it does have
    features for are still missed here: its only (3, -1) feature is a product of l = 1 and
    l = 2 functions, which vanish at a tetrahedral site.)"""
    require_optional("h5py")
    d = deeph.read(FIX)
    model, report = fit_onsite(basis, [d])
    empty = [k for k, (_, cols, _) in report.items() if cols == 0]
    assert empty and all(k not in model.coef for k in empty)
    assert {k for k in report if k[3] >= 4} <= set(empty)                      # and some pseudo-tensor parts
    assert max(float(report[k][2].max()) for k in empty) > 1.0                   # eV: a real omission
    assert np.isfinite(model.predict(d.positions, d.cell, d.numbers)["H"][(0, 0, 0, 0, 0)]).all()


def test_compiled_readout_is_the_reference_readout(basis):
    """`OnsitePredictor` (one jitted program, dense tensors) against `OnsiteModel.predict`
    (a loop over shell pairs), for a model with every part, and for an isolated atom."""
    from ace_jax.operators.onsite import OnsitePredictor
    m = random_model(basis)
    fast = OnsitePredictor(m)
    for pos, cell, numbers in (structure(), structure(3), (np.zeros((1, 3)), 50 * np.eye(3), np.array([6]))):
        a, b = m.predict(pos, cell, numbers), fast(pos, cell, numbers)
        assert set(a["H"]) == set(b["H"])
        for name in a:
            for k, v in a[name].items():
                assert np.abs(b[name][k] - v).max() < 1e-11 * max(1.0, np.abs(v).max()), (name, k)
    atoms, blocks = fast.dense(*structure())
    assert [list(x) for x in atoms] == [[1, 3, 4], [0, 2]] and blocks[0].shape == (3, 2, ORB.n_orb(6), ORB.n_orb(6))


def test_fit_to_selected_atoms(basis_spdf):
    """`atoms=` restricts the fit to some atoms of each dataset, for every species (a loop
    variable once shadowed the argument, so only the first species was fitted)."""
    require_optional("h5py")
    d = deeph.read(FIX)
    model, report = fit_onsite(basis_spdf, [d], atoms=[[0, 1]])
    assert {k[0] for k in report} == {6, 14} and report[(14, 0, 0, 0)][0] == 1 and report[(6, 0, 0, 0)][0] == 1
    pred = model.predict(d.positions, d.cell, d.numbers)
    # atoms 2 and 3 were not fitted but are equivalent to 0 and 1
    assert max(np.abs(pred["H"][k] - d.blocks["H"][k]).max() for k in pred["H"]) < 1e-6


def test_fit_statistics_are_generic_factors(basis_spdf):
    """`stats=True`: one `fit.linear.stats.Factor` per design (species, L, parity), with a
    right-hand side per (shell pair, channel).  The posterior mean of the generic diagonal-prior
    model from it is the fitted coefficient, its evidence is finite, and two copies of the data
    give twice the Gram."""
    import jax.numpy as jnp

    from ace_jax.fit.linear import objective as lo
    require_optional("h5py")
    d = deeph.read(FIX)
    model, report, st = fit_onsite(basis_spdf, [d], stats=True, ridge=1e-8)
    _, _, st2 = fit_onsite(basis_spdf, [d, d], stats=True, ridge=1e-8)
    assert st and all(key[0] in (6, 14) for key in st)
    for key, (fac, keys, _) in st.items():
        k = fac.R.shape[0]
        assert fac.c.shape == (k, 2 * len(keys)) and fac.yy.shape == (2 * len(keys),)
        assert all(report[ck][:2] == (int(fac.n), k) for ck in keys)
        mu, _ = lo.posterior_qr(jnp.full(k, 1e-8), [fac], [0.0])
        for q, ck in enumerate(keys):
            assert np.allclose(np.asarray(mu)[:, 2 * q:2 * q + 2].T, model.coef[ck], rtol=1e-9, atol=1e-9)
        assert np.isfinite(float(lo.log_evidence_qr(jnp.full(k, 1e-8), [fac], [0.0])))
        scale = np.abs(np.asarray(fac.G)).max()
        assert np.allclose(st2[key].factor.G, 2 * fac.G, atol=1e-9 * scale) and float(st2[key].factor.n) == 2 * float(fac.n)


def test_evidence_fit_is_the_generic_evidence():
    """`_evidence`: for every right-hand side the SVD evidence at its optimum equals the QR form's
    (`fit.linear.objective`), the optimum is stationary, the mean is the QR posterior's at the
    scales used, and it finds the noise and prior scales the data were drawn with.  One column
    has no smoothness (gamma 0): it keeps the ridge."""
    import jax.numpy as jnp

    from ace_jax.fit.linear import objective as lo
    from ace_jax.fit.linear import stats as ls
    from ace_jax.operators.fit import _evidence
    rng = np.random.default_rng(0)
    n, k, nc, ridge = 2000, 40, 2, 1e-10
    gamma = np.concatenate([np.linspace(1.0, 6.0, k - 1) ** 2, [0.0]])
    true = {0: (0.05, 30.0), 1: (2e-4, 0.5)}                               # channel: (sigma, sigma_c)
    A = rng.normal(size=(n, k))
    Y = np.zeros((n, 3 * nc))
    for q in range(3 * nc):
        sig, sc = true[q % nc]
        c = np.concatenate([rng.normal(size=k - 1) * sc / gamma[:-1], [0.3]])
        Y[:, q] = A @ c + sig * rng.normal(size=n)
    Q, R = np.linalg.qr(A)
    fac = ls.Factor(jnp.asarray(R), jnp.asarray(Q.T @ Y), jnp.asarray((Y ** 2).sum(0)), jnp.asarray(float(n)), jnp.zeros(()))
    mean, hyper = _evidence(fac, gamma, ridge, nc)
    for q, (sig, sc, value, used) in enumerate(hyper):
        f = ls.Factor(fac.R, fac.c[:, q], fac.yy[q], fac.n, fac.logw)
        lin = lambda lc, s2=1.0: jnp.concatenate([jnp.asarray(gamma[:-1]) ** 2 * jnp.exp(-2.0 * lc), jnp.asarray([ridge / s2])])
        ev = lambda x, lin=lin, f=f: lo.log_evidence_qr(lin(x[1]), [f], [x[0]])
        x = jnp.log(jnp.asarray([sig, sc]))
        assert abs(float(ev(x)) - value) < 1e-6 * abs(value)
        assert np.abs(np.asarray(jax.grad(ev)(x))).max() < 1e-3 * abs(value)
        assert np.allclose(mean[:, q], np.asarray(lo.posterior_qr(lin(jnp.log(used), sig ** 2), [f], [jnp.log(sig)])[0]), atol=1e-8)
        assert abs(np.log10(used / sc)) <= 0.03
        assert 0.9 < sig / true[q % nc][0] < 1.1 and 0.5 < sc / true[q % nc][1] < 2.0


def test_fit_with_the_evidence(basis_spdf):
    """sigma_c="evidence" on real blocks: scales per design and channel in the statistics, a
    model no worse on the fitted structure than a prior fixed far from them."""
    require_optional("h5py")
    d = deeph.read(FIX)
    model, report, st = fit_onsite(basis_spdf, [d], sigma_c="evidence", stats=True)
    assert st and all(s.hyper.shape == (2 * len(s.keys), 4) and np.isfinite(s.hyper).all() and (s.hyper[:, :2] > 0).all() for s in st.values())
    tight, _ = fit_onsite(basis_spdf, [d], sigma_c=1e-3)
    err = lambda m: np.sqrt(np.mean(np.concatenate([(m.predict(d.positions, d.cell, d.numbers)[c][k] - v).ravel()
                                                    for c in ("H", "S") for k, v in d.blocks[c].items() if k[:3] == (0, 0, 0) and k[3] == k[4]]) ** 2))
    assert err(model) < err(tight)
