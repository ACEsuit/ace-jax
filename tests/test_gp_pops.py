import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ace_jax.fit.pops import (
    pops_corrections,
    pops_posterior,
    pops_var,
    whiten,
)


def test_whiten_scales_rows_by_w_over_sigma():
    phi = jnp.ones((3, 2)); r = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([2.0, 2.0, 2.0]); sigma = 4.0
    pt, rt = whiten(phi, r, w, sigma)
    assert jnp.allclose(pt, 0.5 * phi) and jnp.allclose(rt, 0.5 * r)


@pytest.fixture
def pops_synth():
    """A MISSPECIFIED linear synthetic: fit a straight line (features [1, x])
    to strongly quadratic data, so the model cannot fit exactly and the
    residual is systematic. Everything is treated as already-whitened
    (homoscedastic scale 1), so ``Sigma0 = A^{-1}`` with A the ridge
    precision. Returns ``(Sigma0, phi_t, r_t, c, phi_star, epi_var)``.
    """
    N = 60
    x = jnp.linspace(-1.0, 1.0, N)
    # strong quadratic target that a line cannot represent
    y = 2.0 + 0.5 * x + 3.0 * x ** 2
    phi_t = jnp.stack([jnp.ones_like(x), x], axis=1)  # (N, 2)

    lam = 1.0e-3
    A = phi_t.T @ phi_t + lam * jnp.eye(2)
    Sigma0 = jnp.linalg.inv(A)            # epistemic weight covariance A^{-1}
    c = Sigma0 @ (phi_t.T @ y)            # inner-MAP line fit
    r_t = y - phi_t @ c                   # systematic misspecification residual

    x_star = jnp.linspace(-1.0, 1.0, 20)
    phi_star = jnp.stack([jnp.ones_like(x_star), x_star], axis=1)
    epi_var = jnp.sum((phi_star @ Sigma0) * phi_star, axis=1)
    return Sigma0, phi_t, r_t, c, phi_star, epi_var


def test_pops_var_exceeds_epistemic_under_misspecification(pops_synth):
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    post = pops_posterior(d, c, form="samples")
    v = pops_var(phi_star, post)
    assert jnp.all(v >= 0.5 * epi_var)            # misspecification inflates
    assert v.mean() > epi_var.mean()


def test_pops_corrections_fit_their_own_point(pops_synth):
    # the Newton step for point i must fit point i exactly: phi_i . (c + d_i) = y_i
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    fitted = jnp.sum(phi_t * d, axis=1)           # phi_i . d_i
    assert jnp.allclose(fitted, r_t, atol=1e-8)


def test_pops_leverage_percentile_keeps_top_fraction(pops_synth):
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d_all = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    d_top = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=50.0)
    assert d_all.shape[0] == phi_t.shape[0]
    assert d_top.shape[0] <= d_all.shape[0]
    assert d_top.shape[0] >= 1


def test_pops_hypercube_cov_inflates(pops_synth):
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    post_cov = pops_posterior(d, c, form="hypercube")
    assert post_cov["cov"].shape == (phi_t.shape[1], phi_t.shape[1])
    v_cov = pops_var(phi_star, post_cov)
    assert jnp.all(v_cov >= 0.0)
    # hypercube misspecification variance also exceeds epistemic on average
    assert v_cov.mean() > epi_var.mean()


# ---------------------------------------------------------------------------
# Task 11 -- SiGe per-quantity POPS calibration gate (the Cantor spike, inverted)
# ---------------------------------------------------------------------------
# This is a VALIDATION/measurement test, not a unit test.  It reproduces the
# linear-arm (M = 0) fit of bench/acegp_cantor/run.py on REAL SiGe data
# (MACE-MH-1 labels) and MEASURES whether the *native* POPS predictive -- the
# weight-space misspecification variance, with NO post-hoc scalar -- calibrates
# BOTH energy and forces at once.
#
# Pipeline (identical to run.py --arm linear and test_gp_objective's Problem
# construction): load model -> split (r0=2.35, ntrain=200, test-start=250,
# ntest=100, seed=0, configs_per_batch=4) -> per-species E0 by lstsq on train
# energies -> build_dataset -> select_inducing(..., M=0, ...) -> Problem(
# KernelSpec "cosine"+bump, gamma from the export, default_prior(r0)) -> make_lml
# -> run_map (MAP over the 10 hypers incl. sigma_E/F/V) -> predict_fixed(...).
#
# FRAMING (task-11-brief): the RMSE-equality and finite-sigma checks are HARD
# asserts (POPS never changes the mean).  The strict calibration bar (POPS
# rms-z in [0.7,1.5], cov@90 in [0.85,0.95] on E and F, CRPS_POPS <=
# CRPS_BLR+scalar) is MEASURED and PRINTED; if it holds it is asserted, if it
# does NOT it is recorded via pytest.xfail -- never forced to pass, never
# weakened into a tautology.
import os
import pathlib

import numpy as np
from scipy.stats import norm

_SIGE_DIR = pathlib.Path(os.path.expanduser("~/acegp-data/sige"))
_SIGE_MODEL = _SIGE_DIR / "sige_embed_d16.npz"
_SIGE_XYZ = _SIGE_DIR / "sige_mh1.xyz"

# split / config matching the Cantor spike (run.py defaults for this study)
_R0, _NTRAIN, _TEST_START, _NTEST, _SEED, _BATCH = 2.35, 200, 250, 100, 0, 4
_MAP_STEPS = 300
_Z90 = float(norm.ppf(0.95))          # 90% two-sided interval half-width in sigmas

_sige = pytest.mark.skipif(
    not (_SIGE_MODEL.exists() and _SIGE_XYZ.exists()),
    reason="missing SiGe data (~/acegp-data/sige/{sige_embed_d16.npz,sige_mh1.xyz})")


def _sige_metrics(y, mu, sigma):
    """rms-z, coverage@90, mean Gaussian CRPS and RMSE over sigma>0 rows."""
    from ace_jax.fit.metrics import crps_gaussian, rms_z, rmse
    y, mu, sigma = (np.asarray(a, float).reshape(-1) for a in (y, mu, sigma))
    keep = np.isfinite(sigma) & (sigma > 0)
    yk, muk, sk = y[keep], mu[keep], sigma[keep]
    return {
        "rms_z": rms_z(yk, muk, sk),
        "cov90": float(np.mean(np.abs(yk - muk) <= _Z90 * sk)),
        "crps": float(np.mean(crps_gaussian(yk, muk, sk))),
        "rmse": rmse(y, mu),                          # RMSE over ALL rows (mean only)
        "n": int(keep.sum()),
    }


def _blr_scalar_heldout(y, mu, sigma):
    """Post-hoc global sigma-scale: fit s = sqrt(<z^2>) on the FIRST half of the
    rows, apply it to the SECOND half, report metrics there.  Returns
    (scaled_metrics, eval_half_slice, s)."""
    y, mu, sigma = (np.asarray(a, float).reshape(-1) for a in (y, mu, sigma))
    n = y.shape[0]
    a_idx, b_idx = slice(0, n // 2), slice(n // 2, n)
    za = (y[a_idx] - mu[a_idx]) / sigma[a_idx]
    s = float(np.sqrt(np.mean(za ** 2)))
    return _sige_metrics(y[b_idx], mu[b_idx], s * sigma[b_idx]), b_idx, s


@pytest.fixture(scope="module")
def sige_fit():
    """Full linear-arm MAP fit + POPS/BLR predictives on SiGe.  ~3-4 min of real
    f64 CPU compute; module-scoped so it runs once."""
    import equinox as eqx
    jax.config.update("jax_enable_x64", True)

    from ace_jax.eval import highest_precision, load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import default_prior, to_array
    from ace_jax.fit.inducing import (GPConfig, descriptor_scale,
                                       select_inducing, site_features)
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.ladder import run_map
    from ace_jax.fit.objective import Problem, make_lml
    from ace_jax.fit.predict import predict_fixed

    model, meta, z = load(_SIGE_MODEL)
    keys = dict(energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial")
    configs = load_configs(str(_SIGE_XYZ), **keys)
    rng = np.random.default_rng(_SEED)
    perm = rng.permutation(len(configs))
    train = [configs[i] for i in perm[:_NTRAIN]]
    test = [configs[i] for i in perm[_TEST_START:_TEST_START + _NTEST]]

    # per-species E0 by least squares on the training energies (no isolated atoms)
    els = [int(e) for e in meta["elements"]]
    counts = np.array([[np.sum(c.numbers == e) for e in els] for c in train], float)
    E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in train]), rcond=None)
    model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))

    ds_train = build_dataset(train, meta, E0, _BATCH)
    ds_test = build_dataset(test, meta, E0, _BATCH)
    cfg = GPConfig(r0=_R0, rcut=float(meta["rcut"]), n_B=meta["n_B"],
                   n_pair=meta["n_pair"], NZ=len(els), C=_BATCH)

    with highest_precision():
        X, S = site_features(model, cfg, ds_train)
        scale = descriptor_scale(X, ds_train.node_mask)
        ind = select_inducing(X, S, ds_train.node_z, ds_train.node_mask, 0, scale)  # M = 0
        prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                       jnp.asarray(z["gamma"]), default_prior(_R0))
        lik = make_lml(prob, ds_train, cache_linear=True)
        jax.block_until_ready(lik(to_array(prob.prior.mu)))
        theta_map = run_map(lik, prob.prior, steps=_MAP_STEPS, lr=0.02, seed=_SEED)
        # All predictives reuse the SAME warm fit (theta_map); only the POSTERIOR
        # form / aleatoric flag differ.  Two POPS posterior forms:
        #   samples   -- CENTERED committee variance, drops (phi.mean(delta))^2
        #   hypercube -- UNCENTERED 2nd moment about the origin, KEEPS that term
        # (the coordinator's crux: systematic misspecification lives in the mean
        # of the delta corrections, which only the uncentered form retains).
        def _pops(form, alea):
            return predict_fixed(theta_map, prob, ds_train, ds_test, uq="pops",
                                 pops_form=form, aleatoric=alea)
        pred = {
            ("samples", False): _pops("samples", False),      # run.py default "native POPS"
            ("samples", True): _pops("samples", True),
            ("hypercube", False): _pops("hypercube", False),
            ("hypercube", True): _pops("hypercube", True),
        }
        pred_blr = predict_fixed(theta_map, prob, ds_train, ds_test, uq="blr")

    nat = np.array([len(c.numbers) for c in test])
    E = np.array([c.energy for c in test])
    F = np.concatenate([c.forces for c in test]).reshape(-1)
    sig = {q: float(np.exp(getattr(theta_map, f"log_sigma_{q}"))) for q in "EFV"}
    return dict(theta_map=theta_map, sig=sig, nat=nat, E=E, F=F,
                E0={k: float(v) for k, v in zip(els, E0)},
                pred=pred, blr=pred_blr, ntest=len(test))


def _sige_qty(fit, pred, quantity):
    """(y, mu, sigma) for 'E' (meV/atom) or 'F' (eV/A)."""
    if quantity == "E":
        nat = fit["nat"]
        y = 1e3 * fit["E"] / nat
        mu = 1e3 * np.asarray(pred.E_mean) / nat
        sigma = 1e3 * np.sqrt(np.asarray(pred.E_var)) / nat
    else:
        y = fit["F"]
        mu = np.asarray(pred.F_mean).reshape(-1)
        sigma = np.sqrt(np.asarray(pred.F_var)).reshape(-1)
    return y, mu, sigma


# The four POPS predictives (posterior form x aleatoric), plus a short label.
# The misspecification-only rows (aleatoric False) are the candidate "native
# POPS" forms; "samples,False" is the run.py --uq pops default.
_POPS_ROWS = [
    (("samples", False), "POPS s"),
    (("samples", True), "POPS s+alea"),
    (("hypercube", False), "POPS hc"),
    (("hypercube", True), "POPS hc+alea"),
]
# candidate native forms (no post-hoc scalar, misspecification-only)
_NATIVE_FORMS = [(("samples", False), "POPS s"), (("hypercube", False), "POPS hc")]


@pytest.mark.slow
@_sige
def test_sige_pops_per_quantity_calibration_gate(sige_fit):
    from ace_jax.fit.metrics import rmse
    fit = sige_fit
    default_pops = fit["pred"][("samples", False)]

    # ---- HARD ASSERTS (must hold regardless of the scientific outcome) --------
    # Every POPS sigma is finite and the right shape (E: one per config; F: one
    # per force component); POPS never changes the mean, so its RMSE == BLR RMSE.
    assert default_pops.E_var.shape == (fit["ntest"],)
    assert default_pops.F_var.reshape(-1).shape == fit["F"].shape
    for key, _ in _POPS_ROWS:
        for q in ("E", "F"):
            yp, mp, s_pops = _sige_qty(fit, fit["pred"][key], q)
            assert np.all(np.isfinite(s_pops)), f"POPS{key} {q} sigma non-finite"
            assert np.all(s_pops >= 0.0), f"POPS{key} {q} sigma negative"
            yb, mb, _ = _sige_qty(fit, fit["blr"], q)
            r_pops, r_blr = rmse(yp, mp), rmse(yb, mb)
            assert abs(r_pops - r_blr) <= 0.02 * r_blr, (
                f"{key} {q}: POPS RMSE {r_pops:.5g} != BLR RMSE {r_blr:.5g} (>2%); "
                f"POPS must not change the mean")

    # ---- MEASURE the full 6-way table (samples/hypercube x alea off/on) -------
    print(f"\n=== SiGe POPS per-quantity calibration gate "
          f"(ntrain={_NTRAIN} ntest={fit['ntest']} seed={_SEED}) ===")
    print("E0 (eV):", {k: round(v, 4) for k, v in fit["E0"].items()},
          "  MAP sigma:", {k: round(v, 5) for k, v in fit["sig"].items()})
    print(f"{'q':>2} {'method':<13} {'rms_z':>9} {'cov@90':>8} "
          f"{'CRPS':>11} {'RMSE':>11}   n")

    table = {}                                     # (quantity, method) -> metrics
    b_half = {}                                    # quantity -> eval half slice
    for q in ("E", "F"):
        for key, label in _POPS_ROWS:
            y, mu, s = _sige_qty(fit, fit["pred"][key], q)
            m = _sige_metrics(y, mu, s)
            table[(q, label)] = m
            print(f"{q:>2} {label:<13} {m['rms_z']:9.3f} {m['cov90']:8.3f} "
                  f"{m['crps']:11.4g} {m['rmse']:11.4g}   {m['n']}")
        y, mu, s = _sige_qty(fit, fit["blr"], q)
        m = _sige_metrics(y, mu, s)
        table[(q, "BLR")] = m
        print(f"{q:>2} {'BLR':<13} {m['rms_z']:9.3f} {m['cov90']:8.3f} "
              f"{m['crps']:11.4g} {m['rmse']:11.4g}   {m['n']}")
        m_scalar, b_idx, s_scale = _blr_scalar_heldout(y, mu, s)
        table[(q, "BLR+scalar")] = m_scalar
        b_half[q] = b_idx
        print(f"{q:>2} {'BLR+scalar':<13} {m_scalar['rms_z']:9.3f} "
              f"{m_scalar['cov90']:8.3f} {m_scalar['crps']:11.4g} "
              f"{m_scalar['rmse']:11.4g}   {m_scalar['n']}  (s={s_scale:.3f}, held-out half)")

    # ---- THE STRICT CALIBRATION BAR: measure, then assert-or-xfail ------------
    # A candidate native form (misspecification-only, no scalar) passes iff, on
    # BOTH E and F: rms-z in [0.7,1.5], cov@90 in [0.85,0.95], and CRPS (on the
    # BLR+scalar evaluation half) <= CRPS_BLR+scalar.  The gate is green iff ANY
    # native form passes; otherwise xfail with the measured numbers.
    def _bar(key, label):
        ok, detail = True, {}
        for q in ("F", "E"):
            m = table[(q, label)]
            yp, mup, sp = _sige_qty(fit, fit["pred"][key], q)
            b = b_half[q]
            cp = _sige_metrics(yp[b], mup[b], sp[b])["crps"]
            cs = table[(q, "BLR+scalar")]["crps"]
            checks = {"rms_z": 0.7 <= m["rms_z"] <= 1.5,
                      "cov90": 0.85 <= m["cov90"] <= 0.95,
                      "crps<=blr+scalar": cp <= cs}
            detail[q] = (m["rms_z"], m["cov90"], cp, cs, checks)
            ok = ok and all(checks.values())
        return ok, detail

    any_native_ok, summaries = False, []
    for key, label in _NATIVE_FORMS:
        ok, detail = _bar(key, label)
        any_native_ok = any_native_ok or ok
        for q in ("F", "E"):
            rz, c90, cp, cs, checks = detail[q]
            print(f"[bar {label} {q}] rms_z={rz:.3f} cov@90={c90:.3f} "
                  f"CRPS(half)={cp:.4g} CRPS_BLR+scalar={cs:.4g} -> {checks}")
        mF, mE = table[("F", label)], table[("E", label)]
        summaries.append(
            f"{label}: F rms-z={mF['rms_z']:.3f} cov@90={mF['cov90']:.3f}, "
            f"E rms-z={mE['rms_z']:.3f} cov@90={mE['cov90']:.3f}"
            f"{'  [PASS]' if ok else ''}")

    if any_native_ok:
        # green gate: SOME native POPS form calibrates both quantities, no scalar
        passing = [(k, l) for k, l in _NATIVE_FORMS if _bar(k, l)[0]]
        key, label = passing[0]
        for q in ("F", "E"):
            m = table[(q, label)]
            assert 0.7 <= m["rms_z"] <= 1.5
            assert 0.85 <= m["cov90"] <= 0.95
            yp, mup, sp = _sige_qty(fit, fit["pred"][key], q)
            b = b_half[q]
            assert (_sige_metrics(yp[b], mup[b], sp[b])["crps"]
                    <= table[(q, "BLR+scalar")]["crps"])
    else:
        # Do NOT force a pass and do NOT weaken the bar into a tautology: record
        # the measured numbers and mark the strict bar xfail so the file stays
        # runnable.
        pytest.xfail("no native POPS form meets the SiGe calibration bar -- "
                     + "; ".join(summaries))
