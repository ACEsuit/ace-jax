"""Calibrated intervals for the eigenvalues of a Hamiltonian model, by split conformal prediction.

A calibration set (structures with reference blocks that the fit did not use) gives, for each
level, the half-width h such that |e_pred - e_ref| <= h held for that fraction of its
eigenvalues; both spectra are taken from their own Fermi levels.  The width is a constant within
an energy band (Mondrian groups: bands of the predicted eigenvalue from the predicted Fermi
level, each with its own quantile), and not a function of the structure: on bulk gold neither the
coefficient posterior, the delete-one-structure jackknife nor the disagreement of models that see
different things ranked eigenvalue errors better than a constant did, while one width for every
energy left the coverage of the bands 0.2 apart (docs/dev/ham-results.md, 2026-11-21).

What the numbers mean.  `pointwise`: of the eigenvalues of structures drawn like the calibration
set, the stated fraction lies within the half-width, on average over structures; the eigenvalues
of one structure share most of their error, so the fraction within a single structure varies
(0.5 to 0.9 at a nominal 0.68 on gold).  `joint`: with the stated probability EVERY eigenvalue
of a structure's window lies within the half-width (the largest error of each calibration
structure, the quantile over structures; this is the one with the exact finite-sample
guarantee, for exchangeable structures).  Neither says anything about structures unlike the
calibration set."""
import numpy as np

LEVELS = (0.68, 0.9, 0.95)


def _quantile(scores, level):
    """The split-conformal quantile: the ceil((n + 1) level)-th smallest of n scores (inf beyond n)."""
    s = np.sort(np.asarray(scores, float))
    k = int(np.ceil((len(s) + 1) * level))
    return float(s[k - 1]) if 1 <= k <= len(s) else float("inf")


def _band(edges, e):
    return np.clip(np.searchsorted(np.asarray(edges, float), e, side="right") - 1, 0, len(edges) - 2)


def calibrate(pairs, window=(-12.0, 5.0), levels=LEVELS, n_bands=6):
    """The calibration record (a JSON-serialisable dictionary) from `pairs`: per structure
    (e_ref, e_pred), eigenvalues (n_k, n) ascending at each k-point, each from its own Fermi level.
    Scored: the eigenvalues whose prediction lies in `window`.  Energy bands: `n_bands` with equal
    numbers of calibration eigenvalues.  `pointwise[b][l]`: half-width of band b at levels[l];
    `one[l]`: without bands; `joint[l]`: for all of a structure's window at once (inf when the
    structures are too few for the level: n >= level / (1 - level))."""
    err, pred = [], []
    for e0, e1 in pairs:
        e0, e1 = np.asarray(e0, float), np.asarray(e1, float)
        w = (e1 > window[0]) & (e1 < window[1])
        err.append(np.abs(e1 - e0)[w]); pred.append(e1[w])
    if not err or not sum(len(x) for x in err):
        raise ValueError("calibrate: no eigenvalue in the window")
    a, p = np.concatenate(err), np.concatenate(pred)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bands + 1)); edges[0], edges[-1] = window
    b = _band(edges, p)
    return dict(kind="split conformal, constant width per energy band", levels=[float(v) for v in levels], window=[float(v) for v in window],
                edges=[float(v) for v in edges], pointwise=[[_quantile(a[b == i], v) for v in levels] for i in range(n_bands)],
                one=[_quantile(a, v) for v in levels], joint=[_quantile([x.max() for x in err if len(x)], v) for v in levels],
                n_structures=len(err), n_eigenvalues=int(len(a)), rmse=float(np.sqrt(np.mean(a ** 2))))


def _level(cal, level):
    i = int(np.argmin(np.abs(np.asarray(cal["levels"]) - level)))
    if abs(cal["levels"][i] - level) > 1e-9:
        raise ValueError(f"level {level}: the calibration has {cal['levels']}")
    return i


def half_width(cal, e_pred, level=0.9, joint=False):
    """Half-widths, in the shape of `e_pred` (eigenvalues from their own Fermi level), at `level`:
    of each eigenvalue's energy band, or with `joint` the one width that holds for the whole window
    at once.  NaN outside the calibration's window."""
    e, i = np.asarray(e_pred, float), _level(cal, level)
    h = np.full(e.shape, cal["joint"][i]) if joint else np.asarray(cal["pointwise"], float)[_band(cal["edges"], e), i]
    return np.where((e > cal["window"][0]) & (e < cal["window"][1]), h, np.nan)


def coverage(cal, pairs):
    """How the calibration does on other structures with references: per level
    (level, fraction of eigenvalues inside, smallest and largest fraction of a structure,
    fraction of structures with every eigenvalue inside the joint width, mean pointwise half-width)."""
    out = []
    for lv in cal["levels"]:
        hit, joint, hw = [], [], []
        for e0, e1 in pairs:
            d, h, hj = np.abs(np.asarray(e1, float) - np.asarray(e0, float)), half_width(cal, e1, lv), half_width(cal, e1, lv, joint=True)
            w = np.isfinite(h)
            if w.any():
                hit.append(d[w] <= h[w]); joint.append(bool(np.all(d[w] <= hj[w]))); hw.append(h[w])
        per = [float(x.mean()) for x in hit]
        out.append(dict(level=lv, pointwise=float(np.concatenate(hit).mean()), structure_min=min(per), structure_max=max(per),
                        joint=float(np.mean(joint)), half_width=float(np.concatenate(hw).mean())))
    return out


def format_calibration(cal, unit=""):
    lines = [f"calibration: {cal['n_structures']} structures, {cal['n_eigenvalues']} eigenvalues from {cal['window'][0]:g} to {cal['window'][1]:g} of the Fermi level; "
             f"half-widths in 1e-3 of the data's unit{unit}",
             "  energy band        " + "".join(f"{v:>10.2f}" for v in cal["levels"])]
    for i, row in enumerate(cal["pointwise"]):
        lines.append(f"  {cal['edges'][i]:7.2f} to {cal['edges'][i + 1]:6.2f}  " + "".join(f"{1e3 * v:10.3f}" for v in row))
    lines.append("  every energy       " + "".join(f"{1e3 * v:10.3f}" for v in cal["one"]))
    lines.append("  whole window (joint)" + "".join(f"{1e3 * v:9.3f} " for v in cal["joint"]))
    return lines


def format_coverage(cov):
    return ["coverage of the model's calibrated intervals on these structures (level: eigenvalues inside; range over structures; structures wholly inside the joint width):"] + [
        f"  {c['level']:.2f}: {c['pointwise']:.3f}; {c['structure_min']:.2f} to {c['structure_max']:.2f}; {c['joint']:.2f}" for c in cov]
