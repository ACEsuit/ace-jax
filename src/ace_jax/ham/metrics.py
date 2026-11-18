"""Accuracy of a predicted electronic structure against a reference: one definition for `aj eval`
and the benchmarks, whatever code made the prediction.

One number does not do: an eigenvalue error measured from each spectrum's own Fermi level mixes
the error of the bands, the error of the Fermi level (one misplaced state moves it by an eV and
then every eigenvalue with it) and states that should not be there at all, and matching bands by
index breaks when one is.  `spectrum_metrics` therefore gives, for one structure:

    band_rmse, band_mae, band_max   e~ - e by band index at a COMMON energy zero (both spectra as
                                    they are), over the reference states in the window
    aligned_rmse                    the same with each spectrum from its own Fermi level (the
                                    number most papers quote; band error and Fermi error mixed)
    fermi                           mu~ - mu, the Fermi levels at the same electron count
    ee, ee_abs                      the MACE-H paper's smeared eigenvalue error EE(T): |e~ - e|
                                    weighted by the reference occupations, from each spectrum's
                                    own Fermi level, and at a common zero
    band_energy                     E~_band - E_band per atom, each with its own occupations
    dos                             the Wasserstein-1 distance of the two densities of states
                                    at a common zero, an energy: int |N~(E) - N(E)| dE over the
                                    window, N the k-averaged number of states below E (all of
                                    them, so a state that crosses an edge of the window counts
                                    by how far it moved), per reference state in the window.
                                    The mean distance states have to move; no band index, no
                                    broadening
                                    (a spurious state shows here and in band_max, which it
                                    makes large for every band above it)

Energies in the unit of the eigenvalues.  `summary` pools a set of structures; `block_metrics`
is the error of the matrix elements."""
import jax.numpy as jnp
import numpy as np

from .operator import smeared_occupations


OUTLIER = 5.0


def block_metrics(pred, ref, select=None):
    """{mae, rmse, max, n} of the elements of the blocks of `ref` (a dictionary of blocks) that
    `select(key)` keeps (default all); a block `pred` lacks is an error."""
    d = [np.ravel(pred[k] - v) for k, v in ref.items() if select is None or select(k)]
    d = np.concatenate(d) if d else np.zeros(0)
    if not len(d):
        return dict(mae=float("nan"), rmse=float("nan"), max=float("nan"), n=0)
    return dict(mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())), max=float(np.abs(d).max()), n=len(d))


def _occupy(e, wk, nelec, width, smearing, degeneracy):
    f, mu, _ = smeared_occupations(jnp.asarray(e), jnp.asarray(wk), nelec, width, smearing, degeneracy)
    return np.asarray(f), float(mu)


def spectrum_metrics(e_ref, e_pred, nelectrons, n_atoms, weights=None, width=0.1, smearing="fermi-dirac", degeneracy=2.0,
                     window=(-12.0, 5.0)):
    """The metrics above for one structure.  e_ref, e_pred: (n_k, n) eigenvalues, ascending at
    each k-point, of the same k-points; nelectrons: per cell; weights: of the k-points (default
    equal); width, smearing: of the occupations (`smeared_occupations`); window: (lo, hi) from
    the reference Fermi level."""
    e0, e1 = np.asarray(e_ref, float), np.asarray(e_pred, float)
    wk = np.full(len(e0), 1.0 / len(e0)) if weights is None else np.asarray(weights, float)
    (f0, mu0), (f1, mu1) = (_occupy(e, wk, nelectrons, width, smearing, degeneracy) for e in (e0, e1))
    lo, hi = mu0 + window[0], mu0 + window[1]
    w = (e0 > lo) & (e0 < hi)
    ww = (wk[:, None] * w)
    d, da = e1 - e0, (e1 - mu1) - (e0 - mu0)
    avg = lambda x: float((ww * x).sum() / ww.sum()) if ww.sum() else float("nan")
    occ = wk[:, None] * f0
    x = np.concatenate([e0.ravel(), e1.ravel()])        # N~ - N is a step function: +w_k at a predicted state, -w_k at a reference one
    step = np.concatenate([-np.repeat(wk, e0.shape[1]), np.repeat(wk, e1.shape[1])])
    o = np.argsort(x, kind="stable")
    x, diff = x[o], np.cumsum(step[o])
    w1 = float((np.abs(diff[:-1]) * np.clip(np.diff(np.clip(x, lo, hi)), 0.0, None)).sum())
    return dict(
        band_rmse=float(np.sqrt(avg(d ** 2))), band_mae=avg(np.abs(d)), band_max=float(np.abs(d[w]).max()) if w.any() else float("nan"),
        aligned_rmse=float(np.sqrt(avg(da ** 2))), fermi=mu1 - mu0,
        ee=float((np.abs(da) * occ).sum() / occ.sum()), ee_abs=float((np.abs(d) * occ).sum() / occ.sum()),
        band_energy=float(((wk[:, None] * f1 * e1).sum() - (wk[:, None] * f0 * e0).sum()) / n_atoms),
        dos=w1 / float(ww.sum()) if ww.sum() else float("nan"),
        n_states=int(w.sum()), fermi_ref=mu0)


def summary(per):
    """A set of structures' `spectrum_metrics` as one record: for the band error the root mean
    square, the median and the worst structure (a pooled number hides one bad structure, or is
    made by it); the Fermi level and band energy as root mean square and largest; the density of
    states as mean and largest; and `outliers`, the indices of the structures whose band error is
    more than OUTLIER times the median."""
    a = lambda k: np.asarray([m[k] for m in per], float)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    return dict(
        n=len(per), band_rmse=rms(a("band_rmse")), band_rmse_median=float(np.median(a("band_rmse"))), band_rmse_worst=float(a("band_rmse").max()),
        band_max=float(a("band_max").max()), aligned_rmse=rms(a("aligned_rmse")), aligned_rmse_median=float(np.median(a("aligned_rmse"))),
        fermi_rmse=rms(a("fermi")), fermi_max=float(np.abs(a("fermi")).max()), ee=float(a("ee").mean()), ee_abs=float(a("ee_abs").mean()),
        band_energy_rmse=rms(a("band_energy")), band_energy_max=float(np.abs(a("band_energy")).max()),
        dos=float(a("dos").mean()), dos_worst=float(a("dos").max()),
        outliers=[int(i) for i in np.flatnonzero(a("band_rmse") > OUTLIER * np.median(a("band_rmse")))])


def format_summary(s, scale=1e3, unit="meV"):
    """Lines of text for a `summary` (energies times `scale`, named `unit`)."""
    v = lambda k: scale * s[k]
    return [
        f"bands at a common zero: RMSE {v('band_rmse'):.2f} {unit} (median structure {v('band_rmse_median'):.2f}, worst {v('band_rmse_worst'):.2f}; "
        f"largest state error {v('band_max'):.1f})",
        f"bands from own Fermi levels: RMSE {v('aligned_rmse'):.2f} {unit} (median structure {v('aligned_rmse_median'):.2f}); "
        f"occupied-weighted EE {v('ee'):.2f} (common zero {v('ee_abs'):.2f})",
        f"Fermi level: RMSE {v('fermi_rmse'):.2f} {unit}, largest {v('fermi_max'):.1f}; band energy per atom: RMSE {v('band_energy_rmse'):.2f}, "
        f"largest {v('band_energy_max'):.2f}",
        f"density of states (Wasserstein distance): {v('dos'):.2f} {unit} on average, {v('dos_worst'):.2f} in the worst structure; "
        f"{len(s['outliers'])} of {s['n']} structures above {OUTLIER:g} times the median band error" + (f" ({s['outliers']})" if s["outliers"] else "")]
