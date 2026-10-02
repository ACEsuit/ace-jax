"""Equivariance of the served ARD force uncertainty under a global orthogonal Q (rotation or reflection) of
positions AND cell, and under atom permutation.

V(x)_ab = (R^T u_a).(R^T u_b), u_a = D^-1 phi_a(x)^T; R, D act in coefficient space and the ACE basis is O(3)-invariant,
so the force rows phi_a transform as vectors: forces_cov(Qx) = Q forces_cov(x) Q^T, while forces_std / forces_q /
forces_q_mahal / forces_group / support are invariant and the forces themselves rotate."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"

# The posterior's R is stored float32, but it is the SAME R for x and Qx, so it cancels exactly in exact arithmetic; what
# is left is float64 roundoff of the descriptors (~1e-13) amplified by the conditioning of the solve, so cov is held to a
# deliberately loose rtol 1e-5 (float32 headroom) with atol scaled to max|cov|.  Forces/invariants never meet R: tight.
COV_RTOL = 1e-5
TIGHT = 1e-8


def _rand_q(seed, proper):
    rng = np.random.default_rng(seed)
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(Rr))
    if (np.linalg.det(Q) > 0) != proper:
        Q[:, 0] *= -1
    assert np.isclose(np.linalg.det(Q), 1.0 if proper else -1.0)
    return Q


def _cell():
    from ase.io import read
    at = max(read(XYZ, ":8"), key=len)                # the largest fixture cell (multi-atom, periodic)
    assert len(at) > 1 and at.pbc.all()
    at.rattle(0.05, seed=7)
    return at


def _transform(at, Q, perm=None):
    b = at.copy()
    b.set_cell(np.asarray(at.cell) @ Q.T, scale_atoms=False)
    b.set_positions(at.get_positions() @ Q.T)
    return b if perm is None else b[perm]


def _props(calc, at, support):
    aniso = calc.posterior.force_shape == "aniso"
    a = at.copy(); a.calc = calc
    out = {"F": a.get_forces()}
    for k in ("forces_std", "forces_cov", "forces_q", "forces_group") + (("forces_q_mahal",) if aniso else ()):
        out[k] = np.asarray(calc.get_property(k, a))
    if support:
        s = calc.get_property("forces_support", a)
        out["support_ok"], out["support_q"] = np.asarray(s["support_ok"]), np.asarray(s["support_q"])
    return out


def _check(o0, o1, Q, perm, dev, rtol_cov=COV_RTOL):
    """o1 = props(transformed structure) vs o0 = props(original); records max deviations in dev."""
    p = np.arange(len(o0["F"])) if perm is None else perm
    cov0, cov1 = o0["forces_cov"][p], o1["forces_cov"]
    cmax = np.abs(o0["forces_cov"]).max()
    assert cmax > 0
    pred = np.einsum("ab,nbc,dc->nad", Q, cov0, Q)                      # Q cov Q^T per atom
    dev["cov"] = max(dev.get("cov", 0), np.abs(cov1 - pred).max() / cmax)
    np.testing.assert_allclose(cov1, pred, rtol=rtol_cov, atol=rtol_cov * cmax)
    # forces are exactly F(x) rotated (vectors; positions/cell transform with Q)
    Fpred = o0["F"][p] @ Q.T
    dev["F"] = max(dev.get("F", 0), np.abs(o1["F"] - Fpred).max() / np.abs(o0["F"]).max())
    np.testing.assert_allclose(o1["F"], Fpred, rtol=TIGHT, atol=TIGHT * np.abs(o0["F"]).max())
    for k in ("forces_std", "forces_q"):
        a, b = o0[k][p], o1[k]
        assert np.array_equal(np.isinf(a), np.isinf(b)) and np.array_equal(np.isnan(a), np.isnan(b))
        f = np.isfinite(a)
        dev[k] = max(dev.get(k, 0), np.abs(a[f] - b[f]).max() / np.abs(a[f]).max())
        np.testing.assert_allclose(b[f], a[f], rtol=rtol_cov, atol=rtol_cov * np.abs(a[f]).max())
    for k in ("forces_q_mahal", "forces_group"):
        if k in o0:
            np.testing.assert_array_equal(o1[k], o0[k][p])
    if "support_ok" in o0:
        assert np.array_equal(o1["support_ok"], o0["support_ok"][p])
        a, b = o0["support_q"][p], o1["support_q"]
        assert np.array_equal(np.isinf(a), np.isinf(b))
        f = np.isfinite(a)
        if f.any():
            np.testing.assert_allclose(b[f], a[f], rtol=1e-6, atol=1e-6 * np.abs(a[f]).max())


@pytest.fixture(scope="module", params=["iso", "aniso"])
def fit_dir(request, fitted, aniso_fit, tmp_path_factory):
    """(model path, posterior path).  The tiny fixture fit has too few calibration configs for a finite conformal q
    (forces_q would be inf, making its check vacuous), so q is overwritten with finite values in a copy of the posterior."""
    from ace_jax.fit.ard import ARDPosterior
    d = fitted if request.param == "iso" else aniso_fit
    post = ARDPosterior.load(d / "posterior.npz")
    tab = dict(post.group_table)
    tab["q"] = [1.5 + 0.25 * i for i in range(len(tab["q"]))]
    out = tmp_path_factory.mktemp("finite_q") / "posterior.npz"
    post._replace(group_table=tab).save(out)
    assert np.isfinite(ARDPosterior.load(out).group_table["q"]).all()
    return d / "model.npz", out


@pytest.mark.parametrize("lean", [False, True])
@pytest.mark.parametrize("proper", [True, False], ids=["rotation", "reflection"])
def test_served_uncertainty_is_equivariant(fit_dir, lean, proper):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    model, pp = fit_dir
    post = ARDPosterior.load(pp)
    calc = ACECalculator(str(model), posterior=str(pp), lean=lean)
    at = _cell()
    Q = _rand_q(11 if proper else 12, proper)
    o0 = _props(calc, at, post.support is not None)
    o1 = _props(calc, _transform(at, Q), post.support is not None)
    dev = {}
    _check(o0, o1, Q, None, dev)
    print(f"[{post.force_shape} lean={lean} proper={proper}] max rel dev: {dev}")
    # negative control: V is genuinely anisotropic / orientation-dependent, so un-rotated cov must differ
    cmax = np.abs(o0["forces_cov"]).max()
    assert np.abs(o1["forces_cov"] - o0["forces_cov"]).max() > 1e-3 * cmax
    wrong = np.einsum("ab,nbc,dc->nad", Q.T, o0["forces_cov"], Q.T)     # Q^T cov Q: the wrong action
    assert np.abs(o1["forces_cov"] - wrong).max() > 1e-3 * cmax


def test_served_uncertainty_permutation_covariant(fit_dir):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    model, pp = fit_dir
    post = ARDPosterior.load(pp)
    calc = ACECalculator(str(model), posterior=str(pp))
    at = _cell()
    perm = np.random.default_rng(3).permutation(len(at))
    assert not np.array_equal(perm, np.arange(len(at)))
    o0 = _props(calc, at, post.support is not None)
    o1 = _props(calc, _transform(at, np.eye(3), perm), post.support is not None)
    dev = {}
    _check(o0, o1, np.eye(3), perm, dev)
    print(f"[{post.force_shape} permutation] max rel dev: {dev}")


def test_aniso_conformal_score_is_invariant(aniso_fit):
    """s = sqrt(e^T M^-1 e) with M = V + eps tr V/3 I is unchanged when the label error e rotates with the structure."""
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior, conformal_scores
    post = ARDPosterior.load(aniso_fit / "posterior.npz")
    assert post.force_shape == "aniso"
    calc = ACECalculator(str(aniso_fit / "model.npz"), posterior=str(aniso_fit / "posterior.npz"))
    at = _cell()
    rng = np.random.default_rng(5)
    e = rng.normal(size=(len(at), 3)) * 0.1
    for proper in (True, False):
        Q = _rand_q(21 if proper else 22, proper)
        V0 = np.asarray(_props(calc, at, False)["forces_cov"])
        V1 = np.asarray(_props(calc, _transform(at, Q), False)["forces_cov"])
        s0 = conformal_scores(e, V0, "aniso", post.eps)
        s1 = conformal_scores(e @ Q.T, V1, "aniso", post.eps)
        print(f"[score proper={proper}] max rel dev {np.abs(s1 - s0).max() / s0.max():.2e}")
        np.testing.assert_allclose(s1, s0, rtol=COV_RTOL, atol=COV_RTOL * s0.max())
        # negative control: NOT rotating e with the structure changes the score
        assert np.abs(conformal_scores(e, V1, "aniso", post.eps) - s0).max() > 1e-3 * s0.max()
