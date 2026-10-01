import numpy as np
import pytest
from ase.build import bulk, fcc111


def _wq(s, w, alpha):
    """Brute-force pooled-CDF quantile: smallest t with (sum_{s_i<=t} w_i) / (W + 1) >= 1 - alpha."""
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / (w.sum() + 1.0)
    k = np.searchsorted(cw, 1 - alpha - 1e-15)
    return np.inf if k >= len(s) else s[o][k]


def test_shell_features_fcc_strained_slab(one_config_batch):
    from ace_jax.fit.conformal import shell_features
    a0 = 3.65
    r1 = 0.5 * (a0 / np.sqrt(2) + a0)
    z, d = shell_features(one_config_batch(bulk("Ni", "fcc", a=a0, cubic=True).repeat(3)), r1)
    assert np.all(z[:108] == 12) and np.allclose(d[:108], 0.0, atol=1e-5)  # rij is float32
    st = bulk("Ni", "fcc", a=a0, cubic=True).repeat(3)
    st.set_cell(st.cell.array @ np.diag([1.04, 1, 1]), scale_atoms=True)
    z2, d2 = shell_features(one_config_batch(st), r1)
    assert np.all(z2[:108] == 12) and np.all(d2[:108] > 0.005)
    sl = fcc111("Ni", size=(3, 3, 6), a=a0, vacuum=8.0)
    sl.pbc = True
    z3, _ = shell_features(one_config_batch(sl), r1)
    assert z3[:len(sl)].min() == 9 and z3[:len(sl)].max() == 12


def test_shell_reference_fcc_bcc_and_fallback():
    from ace_jax.fit.conformal import shell_reference
    from ase.neighborlist import neighbor_list
    # bcc: the first minimum is the clean gap between the 8 and 6 neighbour shells (2.49 | 2.87 A)
    for at, lo, hi in ((bulk("Ni", "fcc", a=3.65, cubic=True).repeat(4), 2.6, 3.6),
                       (bulk("Fe", "bcc", a=2.87, cubic=True).repeat(5), 2.5, 2.87)):
        at.rattle(0.03, seed=1)
        assert lo < shell_reference(neighbor_list("d", at, 6.0)) < hi
    flat = np.full(1000, 2.5) + np.random.default_rng(0).normal(0, 1e-3, 1000)
    assert shell_reference(flat) == pytest.approx(1.25 * 2.5, rel=0.02)


def test_assign_groups_extremes():
    from ace_jax.fit.conformal import assign_groups, n_groups
    edges = np.array([0.01, 0.02, 0.05])
    g = assign_groups(np.array([12, 12, 11, 3, 1]), np.array([0.0, 0.03, 0.9, np.nan, np.nan]), 12, edges)
    assert g.tolist() == [0, 4, 7, 7, 7] and n_groups(edges) == 8
    assert assign_groups(np.array([12, 3]), np.array([0.5, np.nan]), 12, np.array([])).tolist() == [0, 1]
    assert n_groups(np.array([])) == 2


def test_stratified_split_guarantees():
    from ace_jax.fit.conformal import config_strata, stratified_split
    assert config_strata([np.array([0, 4, 1]), np.array([], int), np.array([7])]).tolist() == [4, 0, 7]
    strata = np.r_[np.zeros(100, int), np.full(10, 6), np.full(2, 7), [5]]  # a lone config in stratum 5
    fit, val = stratified_split(strata, 0.2, seed=0)
    assert len(np.intersect1d(fit, val)) == 0 and len(fit) + len(val) == len(strata)
    for s in (0, 6, 7):
        m = np.flatnonzero(strata == s)
        assert np.isin(m, val).any() and np.isin(m, fit).any(), s
    assert abs(np.isin(np.flatnonzero(strata == 0), val).sum() - 20) <= 1
    f2, v2 = stratified_split(strata, 0.2, seed=0)
    assert np.array_equal(v2, val)                                              # deterministic in seed


def test_config_weighted_scales_against_brute_force():
    from ace_jax.fit.conformal import chi3_ppf, group_scales
    rng = np.random.default_rng(3)
    n_cfg = 60
    sizes = rng.integers(1, 40, n_cfg)                                          # very unequal atom counts
    cfg = np.repeat(np.arange(n_cfg), sizes)
    s = rng.chisquare(3, len(cfg)) ** 0.5 * (1 + 0.5 * (cfg % 3))
    g = np.zeros(len(cfg), int)
    t = group_scales(s, g, cfg, G=2, alpha=0.1, n_min=20)
    w = 1.0 / np.bincount(cfg)[cfg]
    assert t.lam_rms[0] == pytest.approx(np.sqrt(np.sum(w * s ** 2) / (3 * n_cfg))) and t.n_cfg[0] == n_cfg
    assert t.q[0] == pytest.approx(_wq(s, w, 0.1))
    assert t.r[0] == pytest.approx(t.q[0] / (t.lam_rms[0] * chi3_ppf(0.9)))
    assert chi3_ppf(0.9) == pytest.approx(2.50028, abs=1e-4)


def test_scales_merge_small_groups():
    from ace_jax.fit.conformal import GroupTable, group_scales
    rng = np.random.default_rng(4)
    cfg = np.r_[np.arange(400) // 10, 1000 + np.arange(30) // 10]               # 40 cfgs in g=0, 3 in g=2
    g = np.r_[np.zeros(400, int), np.full(30, 2)]
    src = np.r_[np.zeros(400, int), np.ones(30, int)]
    s = rng.chisquare(3, 430) ** 0.5
    t = group_scales(s, g, cfg, G=8, alpha=0.1, n_min=20, src=src)
    assert [2, 0] in t.merged and t.q[2] == t.q[0]                              # band 1 flag 0 -> band 0
    assert np.isfinite(t.q).all() and np.isfinite(t.lam_rms).all()             # empty groups served too
    assert t.n_cfg_val[0] == 40 and t.n_cfg_cal[2] == 3 and t.n_min == 20
    back = GroupTable.from_dict(t.to_dict())
    assert np.array_equal(back.q, t.q) and back.merged == t.merged and back.n_min == 20


def test_support_flags_out_of_support_with_config_masses():
    from ace_jax.fit.support import build_support, fit_pca, support_check
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 10))
    Z = np.zeros(4000, int)
    cfg = np.arange(4000) // 20
    s = rng.chisquare(3, 4000) ** 0.5
    ref = build_support(fit_pca({0: X}), X, Z, s, cfg, max_atoms=50000, seed=0)
    Xin, Xout = rng.normal(size=(400, 10)), rng.normal(size=(400, 10)) + 10.0
    a = support_check(ref, np.r_[Xin, Xout], np.zeros(800, int), 0.1)
    assert a["support_ok"][:400].mean() > 0.95 and a["support_ok"][400:].mean() < 0.05
    assert np.isinf(a["support_q"][400:][~a["support_ok"][400:]]).all()
    b = support_check(ref, Xin, np.zeros(400, int), 0.1)
    assert b["n_eff"][0] > 0.4 * 200 and np.isfinite(b["support_q"]).all()      # over 200 config masses


def test_support_pca_cap_and_variance():
    from ace_jax.fit.support import fit_pca
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3)) @ rng.normal(size=(3, 100)) + 1e-6 * rng.normal(size=(500, 100))
    mu, sd, W = fit_pca({0: X})[0]
    assert W.shape[1] == 3
    mu, sd, W = fit_pca({0: rng.normal(size=(2000, 200))})[0]
    assert W.shape[1] == 64


def test_support_flatten_roundtrip_and_absent_species():
    from ace_jax.fit.support import build_support, fit_pca, flatten_support, support_check, unflatten_support
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 6))
    Z = np.arange(600) % 2
    ref = build_support(fit_pca({0: X[Z == 0], 1: X[Z == 1]}), X, Z, rng.random(600), np.arange(600) // 10, 1000, 0)
    flat = flatten_support(ref)
    assert "support_pca_0_mu" in flat and "support_1_Xc" in flat and flat["support_0_g"].dtype.kind == "i"
    back = unflatten_support({k[8:]: v for k, v in flat.items()})
    Xt, Zt = rng.normal(size=(30, 6)), np.r_[np.zeros(15, int), np.ones(10, int), np.full(5, 7)]
    a, b = support_check(ref, Xt, Zt, 0.1), support_check(back, Xt, Zt, 0.1)
    assert not b["support_ok"][25:].any() and np.isinf(b["support_q"][25:]).all() and b["n_eff"][7] == 0
    assert (a["support_ok"] == b["support_ok"]).mean() > 0.9


def test_support_logistic_separable_no_nan_and_fails_closed():
    import warnings
    from ace_jax.fit.support import _logistic, build_support, fit_pca, support_check
    rng = np.random.default_rng(3)
    X = np.r_[rng.normal(size=(50, 3)) - 20, rng.normal(size=(50, 3)) + 20]
    y = np.r_[np.zeros(50), np.ones(50)]
    beta, conv = _logistic(X, y, np.ones(100), 1e-12, iters=60)
    assert np.isfinite(beta).all() and not conv
    # a reference whose classifier cannot converge must fail closed with a warning
    Xr = rng.normal(size=(300, 4))
    ref = build_support(fit_pca({0: Xr}), Xr, np.zeros(300, int), rng.random(300), np.arange(300) // 5, 1000, 0)
    import ace_jax.fit.support as sp
    orig = sp._logistic
    sp._logistic = lambda *a, **k: (orig(*a, **k)[0], False)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = support_check(ref, rng.normal(size=(20, 4)), np.zeros(20, int), 0.1)
    finally:
        sp._logistic = orig
    assert not r["support_ok"].any() and np.isinf(r["support_q"]).all() and r["n_eff"][0] == 0
    assert any("did not converge" in str(x.message) for x in w)


def test_support_zero_mass_gives_neff_zero():
    from ace_jax.fit.support import build_support, fit_pca, support_check
    rng = np.random.default_rng(4)
    Xr = rng.normal(size=(200, 3))
    ref = build_support(fit_pca({0: Xr}), Xr, np.zeros(200, int), rng.random(200), np.arange(200) // 5, 1000, 0)
    ref[0]["m"] = np.zeros(200)
    r = support_check(ref, rng.normal(size=(10, 3)), np.zeros(10, int), 0.1)
    assert r["n_eff"][0] == 0.0 and not r["support_ok"].any() and np.isfinite(r["n_eff"][0])


def test_support_weighted_quantile_matches_direct_formula(monkeypatch):
    import ace_jax.fit.support as sp
    rng = np.random.default_rng(5)
    n = 40
    s = rng.random(n)
    m = rng.random(n) + 0.1
    lc, lt = rng.normal(size=n), np.array([0.3, -1.0])
    ref = {"pca": {0: (np.zeros(2), np.ones(2), np.eye(2))},
           0: {"Xc": np.zeros((n, 2)), "s": s, "m": m, "g": np.arange(n)}}
    monkeypatch.setattr(sp, "_ratio", lambda *a, **k: (lc, lt, True))
    alpha = 0.2
    out = sp.support_check(ref, np.zeros((2, 2)), np.zeros(2, int), alpha)
    for j in range(2):
        w = m * np.exp(lc)
        pt = np.exp(lt[j])
        tot = w.sum() + pt
        o = np.argsort(s)
        cum = np.cumsum(w[o]) / tot
        k = np.flatnonzero(cum >= 1 - alpha)
        if len(k):
            assert out["support_ok"][j] and out["support_q"][j] == s[o][k[0]]
        else:
            assert not out["support_ok"][j] and np.isinf(out["support_q"][j])


def test_support_check_time_large_reference():
    import time
    from ace_jax.fit.support import build_support, fit_pca, support_check
    rng = np.random.default_rng(6)
    X = rng.normal(size=(50000, 40))
    ref = build_support(fit_pca({0: X}), X, np.zeros(50000, int), rng.random(50000), np.arange(50000) // 25, 50000, 0)
    t = time.time()
    support_check(ref, rng.normal(size=(1000, 40)), np.zeros(1000, int), 0.1)
    print("SUPPORT_CHECK_SECONDS", time.time() - t)
    assert time.time() - t < 60
