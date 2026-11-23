"""`ham.metrics`: each metric answers its own question (a rigid shift, a Fermi-level error and a
spurious state are told apart)."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from ace_jax.ham import metrics as M


def spectra():
    rng = np.random.default_rng(0)
    e = np.sort(np.concatenate([rng.uniform(-9.0, -1.0, size=(12, 20)), rng.uniform(1.0, 6.0, size=(12, 14))], axis=1), axis=1)
    return e, 40.0, 8                               # twenty filled bands below a gap


def test_identical_and_shifted_spectra():
    e, nel, na = spectra()
    kw = dict(width=0.05, window=(-12.0, 3.0))
    m = M.spectrum_metrics(e, e, nel, na, **kw)
    assert all(abs(m[k]) < 1e-12 for k in ("band_rmse", "band_max", "aligned_rmse", "fermi", "ee", "ee_abs", "band_energy", "dos"))
    assert -1.0 < m["fermi_ref"] < 1.0
    s = M.spectrum_metrics(e, e + 0.3, nel, na, **kw)             # a rigid shift: the bands are right from their own Fermi level
    assert abs(s["band_rmse"] - 0.3) < 1e-9 and abs(s["fermi"] - 0.3) < 1e-6 and s["aligned_rmse"] < 1e-6 and s["ee"] < 1e-6
    assert abs(s["ee_abs"] - 0.3) < 1e-9 and abs(s["band_energy"] - 0.3 * nel / na) < 1e-6 and 0.25 < s["dos"] < 0.3 + 1e-9


def test_a_spurious_state_is_seen_where_it_matters():
    """One state too many deep in the valence band: every band above it is off by index and the
    Fermi level falls to the valence band, while the density of states moves by one state over
    the energies above it (a fraction of an eV per state); an error of the conduction bands alone leaves the Fermi level and the
    band energy (nearly) alone."""
    e, nel, na = spectra()
    kw = dict(width=0.05, window=(-12.0, 3.0))
    bad = np.sort(np.concatenate([e, np.full((len(e), 1), -5.0)], axis=1), axis=1)[:, :e.shape[1]]
    m = M.spectrum_metrics(e, bad, nel, na, **kw)
    assert m["fermi"] < -1.0 and m["band_max"] > 1.0 and 0.1 < m["dos"] < 0.5
    c = e.copy(); c[:, 20:] += 0.2
    m = M.spectrum_metrics(e, c, nel, na, **kw)
    assert abs(m["band_energy"]) < 1e-6 and 0.05 < m["band_rmse"] < 0.2 and abs(m["fermi"]) < 0.11 and 0.0 < m["dos"] < 0.2


def test_summary_and_blocks():
    e, nel, na = spectra()
    per = [M.spectrum_metrics(e, e + d, nel, na, width=0.05) for d in (0.01, 0.01, 0.01, 0.2)]
    s = M.summary(per)
    assert s["n"] == 4 and s["outliers"] == [3] and abs(s["band_rmse_median"] - 0.01) < 1e-9 and abs(s["band_rmse_worst"] - 0.2) < 1e-9
    assert len(M.format_summary(s)) == 4
    ref = {(0, 0, 0, 0, 0): np.eye(2), (1, 0, 0, 0, 1): np.ones((2, 3))}
    pred = {k: v + 0.5 for k, v in ref.items()}
    b = M.block_metrics(pred, ref, select=lambda k: k[:3] != (0, 0, 0))
    assert b == dict(mae=0.5, rmse=0.5, max=0.5, n=6) and M.block_metrics(pred, ref)["n"] == 10


def test_dos_distance_is_the_wasserstein_distance():
    """With every state inside the window the distance is SciPy's 1-D Wasserstein distance of the
    two k-weighted spectra; a state moving across the window's edge adds only how far it moved
    inside."""
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(1)
    e, nel, na = spectra()
    p = np.sort(e + 0.1 * rng.normal(size=e.shape), axis=1)
    wk = rng.uniform(0.5, 1.5, size=len(e)); wk /= wk.sum()
    m = M.spectrum_metrics(e, p, nel, na, weights=wk, width=0.05, window=(-30.0, 30.0))
    ref = wasserstein_distance(e.ravel(), p.ravel(), np.repeat(wk, e.shape[1]), np.repeat(wk, e.shape[1]))
    assert abs(m["dos"] - ref) < 1e-12 and m["dos"] <= m["band_mae"] + 1e-12
    one = e.copy(); one[:, -1] += 1e-3                              # the top state, 1 meV up
    hi = e[:, -1].min() + 5e-4                                      # ... across an edge placed between
    mu = M.spectrum_metrics(e, e, nel, na, width=0.05)["fermi_ref"]
    m = M.spectrum_metrics(e, one, nel, na, width=0.05, window=(-30.0, hi - mu))
    assert 0.0 <= m["dos"] < 1e-3 / e.shape[1]


def test_overlap_threshold_drops_near_null_directions():
    """`operator._host_eig(cut=)` and its device form: the generalised problem in the subspace of S above the
    threshold, the directions left out +inf at the end; a threshold below the spectrum of S changes nothing; and a
    small error in H that moves a state of the full problem by electron-volts moves none of the subspace's."""
    from ace_jax.ham import operator as O
    g = np.random.default_rng(0); n = 12
    Q = np.linalg.qr(g.normal(size=(n, n)) + 1j * g.normal(size=(n, n)))[0]
    s = np.r_[1e-6, np.linspace(0.3, 2.0, n - 1)]
    S = (Q * s) @ Q.conj().T
    A = g.normal(size=(n, n)) + 1j * g.normal(size=(n, n)); H = A + A.conj().T
    full = scipy.linalg.eigh(H, S, eigvals_only=True)
    assert np.allclose(O._host_eig(H[None], S[None], cut=1e-9)[0], full, rtol=1e-7, atol=1e-7 * abs(full).max())
    X = Q[:, 1:] / np.sqrt(s[1:])
    want = np.linalg.eigvalsh(X.conj().T @ H @ X)
    e = O._host_eig(H[None], S[None], cut=1e-3)[0]
    assert np.isposinf(e[-1]) and np.allclose(e[:-1], want, atol=1e-10)
    dev = np.asarray(O._eigvals_cut_k_dev(jnp.asarray(H[None]), jnp.asarray(S[None]), 1e-3))[0]
    assert np.isposinf(dev[-1]) and np.allclose(dev[:-1], want, atol=1e-9)
    v = Q[:, 0]; dH = 1e-3 * np.outer(v, v.conj())                  # 1e-3 along the direction of norm 1e-6: 1e3 in its energy
    assert abs(scipy.linalg.eigh(H + dH, S, eigvals_only=True) - full).max() > 1.0
    assert abs(O._host_eig((H + dH)[None], S[None], cut=1e-3)[0][:-1] - want).max() < 1e-10
    with pytest.raises(ValueError, match="eigenvectors"):
        O._host_eig(H[None], S[None], vectors=True, cut=1e-3)


def test_metrics_ignore_directions_left_out():
    """+inf columns (an overlap threshold) hold no electrons and no error: the metrics are those without them."""
    e, nel, na = spectra()
    m = M.spectrum_metrics(e, e + 0.02, nel, na, width=0.05)
    pad = lambda x: np.concatenate([x, np.full((len(x), 3), np.inf)], 1)
    q = M.spectrum_metrics(pad(e), pad(e + 0.02), nel, na, width=0.05)
    assert all(np.isclose(m[k], q[k], rtol=1e-5, atol=1e-5) for k in m)       # the Fermi level in a gap is defined to the bisection's roundoff
