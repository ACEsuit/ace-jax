"""`ham.metrics`: each metric answers its own question (a rigid shift, a Fermi-level error and a
spurious state are told apart)."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

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
