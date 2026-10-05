"""Normalised support features (--ard-support-features normalised; docs/dev/force-uq-vs-calm.md, Step 2).

phi_hat = phi/(|phi| + eps) with log|phi| and log|phi_b| (per body order b) channels that bypass the PCA:
an atom losing its neighbours has phi -> 0, inside a raw cloud that reaches 0 but far out in log|phi|."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.fit.support import (  # noqa: E402
    _proj,
    build_support,
    extend_support,
    fit_pca,
    flatten_support,
    n_pass,
    support_body,
    support_check,
    support_features,
    unflatten_support,
)

BODY = np.array([2, 2, 3, 3, 3, 4, 4, 2])          # e.g. 6 B columns (nu = 1, 2, 3) + 2 pair columns


def test_support_body_from_meta():
    meta = {"nnll": [[(1, 0)], [(1, 0), (2, 0)], [(1, 0), (1, 0), (1, 0)]], "n_pair": 2}
    assert support_body(meta).tolist() == [2, 3, 4, 2, 2]


def test_raw_is_identity_and_normalised_layout():
    X = np.random.default_rng(0).normal(size=(5, 8))
    assert support_features(X, None) is not None and np.array_equal(support_features(X, None), X)
    assert np.array_equal(support_features(X, {"kind": "raw", "body": BODY}), X)
    feat = {"kind": "normalised", "body": BODY}
    F = support_features(X, feat)
    assert F.shape == (5, 8 + n_pass(feat)) and n_pass(feat) == 4
    np.testing.assert_allclose(np.linalg.norm(F[:, :8], axis=1), 1.0)
    np.testing.assert_allclose(F[:, 8], np.log(np.linalg.norm(X, axis=1) + 1e-12))
    for j, b in enumerate((2, 3, 4)):
        np.testing.assert_allclose(F[:, 9 + j], np.log(np.linalg.norm(X[:, BODY == b], axis=1) + 1e-12))
    G = support_features(7.0 * X, feat)                             # direction part is scale invariant
    np.testing.assert_allclose(G[:, :8], F[:, :8], atol=1e-14)
    np.testing.assert_allclose(G[:, 8:], F[:, 8:] + np.log(7.0), atol=1e-12)


def test_pass_channels_bypass_pca():
    rng = np.random.default_rng(1)
    F = np.c_[rng.normal(size=(400, 30)) @ rng.normal(size=(30, 30)), rng.normal(3.0, 2.0, size=(400, 2))]
    mu, sd, W = fit_pca({0: F}, cap=5, n_pass=2)[0]
    assert W.shape == (32, 5 + 2)
    P = _proj({0: (mu, sd, W)}, 0, F)
    np.testing.assert_allclose(P[:, -2:], (F[:, -2:] - F[:, -2:].mean(0)) / F[:, -2:].std(0), rtol=1e-9)
    np.testing.assert_allclose(np.cov(P[:, :5].T, bias=True), np.eye(5), atol=1e-8)   # whitened PCA part


def _cloud(rng, n, scale_sd=0.6):
    """Training sites: directions near one mean direction, log-normal norms (compressed tail large)."""
    d = np.ones(8) + 0.3 * rng.normal(size=(n, 8))
    return d / np.linalg.norm(d, axis=1)[:, None] * np.exp(rng.normal(0.0, scale_sd, n))[:, None]


@pytest.mark.parametrize("kind", ["raw", "normalised"])
def test_in_distribution_not_flagged(kind):
    rng = np.random.default_rng(2)
    feat = {"kind": kind, "body": BODY}
    X = _cloud(rng, 1200)
    pca = fit_pca({0: support_features(X, feat)}, n_pass=n_pass(feat))
    ref = build_support(pca, X, np.zeros(1200, int), rng.random(1200), np.arange(1200) // 4, 50000, 0, features=feat)
    out = support_check(ref, _cloud(rng, 200), np.zeros(200, int), alpha=0.1)
    assert out["support_ok"].mean() > 0.9


def test_stretched_sites_flagged_by_normalised_features():
    """phi -> 0 along the training direction (an atom losing neighbours): normalised flags it."""
    rng = np.random.default_rng(3)
    X = _cloud(rng, 1200)
    feat = {"kind": "normalised", "body": BODY}
    pca = fit_pca({0: support_features(X, feat)}, n_pass=n_pass(feat))
    ref = build_support(pca, X, np.zeros(1200, int), rng.random(1200), np.arange(1200) // 4, 50000, 0, features=feat)
    Xt = _cloud(rng, 200) * 1e-3
    assert not support_check(ref, Xt, np.zeros(200, int), alpha=0.1)["support_ok"].any()


def test_features_round_trip_and_old_reference_is_raw(tmp_path):
    rng = np.random.default_rng(4)
    X, Z = _cloud(rng, 300), np.repeat([0, 1], 150)
    feat = {"kind": "normalised", "body": BODY}
    F = support_features(X, feat)
    pca = fit_pca({0: F[Z == 0], 1: F[Z == 1]}, n_pass=n_pass(feat))
    ref = build_support(pca, X, Z, rng.random(300), np.arange(300) // 3, 1000, 0, grp=np.zeros(300, int),
                        features=feat)
    flat = flatten_support(ref)
    np.savez(tmp_path / "s.npz", **flat)
    z = np.load(tmp_path / "s.npz")                              # as ARDPosterior.load reads it (no pickle)
    back = unflatten_support({k[8:]: z[k] for k in z.files})
    assert back["features"]["kind"] == "normalised" and np.array_equal(back["features"]["body"], BODY)
    assert sorted(k for k in back if isinstance(k, int)) == [0, 1]
    a = support_check(ref, X[:20], Z[:20], 0.1)["support_q"]
    b = support_check(back, X[:20], Z[:20], 0.1)["support_q"]
    np.testing.assert_allclose(a, b, rtol=1e-5)
    old = unflatten_support({k[8:]: v for k, v in flat.items() if not k.startswith("support_feat_")})
    assert "features" not in old                                 # pre-Step-2 posteriors: raw descriptors


def test_extend_support_keeps_features():
    rng = np.random.default_rng(5)
    X, Z = _cloud(rng, 200), np.zeros(200, int)
    feat = {"kind": "normalised", "body": BODY}
    ref = build_support(fit_pca({0: support_features(X, feat)}, n_pass=n_pass(feat)), X, Z, rng.random(200),
                        np.arange(200) // 4, 1000, 0, grp=np.zeros(200, int), features=feat)
    new = extend_support(ref, {}, X[:40], Z[:40], rng.random(40), np.arange(40) // 4, 1000, 0, np.zeros(40, int),
                         log=lambda *_: None)
    assert new["features"] is ref["features"] and new[0]["Xc"].shape[1] == ref[0]["Xc"].shape[1]
    np.testing.assert_allclose(new[0]["Xc"][-40:], _proj(ref["pca"], 0, support_features(X[:40], feat)))
