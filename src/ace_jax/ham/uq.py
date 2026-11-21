"""Calibrated intervals for the eigenvalues of a Hamiltonian model, and for its matrix elements
(`BlockErrors`, at the end), by split conformal prediction.

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


# -- matrix elements ---------------------------------------------------------------------------------
class BlockErrors:
    """|predicted - reference| of the blocks of structures, kept per channel and ordered species
    pair, for `calibrate` (half-widths of the matrix elements) and `coverage`.

    Groups (Mondrian): the channel, the species of the two atoms, on-site or one of `n_dist`
    distance bands (equal numbers of calibration blocks), and the pair of shells an element
    couples.  On bulk gold the half-width at 0.9 falls from 1.2 meV for the nearest bonds to
    0.07 meV for the longest and is 7.7 meV on-site, and at one distance differs fourfold
    between shell pairs: one width per distance band left the coverage of the shell pairs
    between 0.54 and 1.00 at 0.9, distance bands times shell pairs between 0.88 and 0.90
    (docs/dev/ham-results.md, 2026-11-21).  `whole`: a width for the largest element error of a
    block, per distance band.  Unlike the eigenvalues, the elements of one structure do not share
    their error: the coverage of single structures stays within about 0.02 of the level."""

    def __init__(self, orbitals):
        self.orbitals, self.err, self.n = orbitals, {}, 0

    def add(self, channel, numbers, positions, cell, ref, pred):
        """One structure's blocks of `channel`: dictionaries {(Rx, Ry, Rz, i, j): block}; scored where both have the block."""
        numbers, positions, cell = np.asarray(numbers), np.asarray(positions, float), np.asarray(cell, float)
        new = {}
        for k, v in ref.items():
            if k in pred:
                on = k[:3] == (0, 0, 0) and k[3] == k[4]
                r = 0.0 if on else float(np.linalg.norm(positions[k[4]] - positions[k[3]] + np.asarray(k[:3], float) @ cell))
                e, rr = new.setdefault((channel, int(numbers[k[3]]), int(numbers[k[4]])), ([], []))
                e.append(np.abs(np.asarray(pred[k]) - np.asarray(v)).astype(np.float32)); rr.append(-1.0 if on else r)
        for g, (e, rr) in new.items():
            self.err.setdefault(g, []).append((np.stack(e), np.asarray(rr), self.n))
        self.n += 1

    def _pair(self, g):
        e = np.concatenate([x[0] for x in self.err[g]]); r = np.concatenate([x[1] for x in self.err[g]])
        return e, r, np.concatenate([np.full(len(x[1]), x[2]) for x in self.err[g]])

    def _shells(self, g):
        o = self.orbitals
        return [o.shell_slice(g[1], s) for s in range(len(o.ls(g[1])))], [o.shell_slice(g[2], s) for s in range(len(o.ls(g[2])))]

    def calibrate(self, levels=LEVELS, n_dist=6):
        """The record: per channel and species pair "z1-z2", `edges` of the distance bands (band 0:
        on-site, r < 0), and per band `element[s1][s2][l]` (half-width of an element between
        shells s1 and s2 at levels[l]), `pooled[l]` (every shell pair), `whole[l]` (the largest
        element error of a block) and `n` (calibration blocks).  inf: too few scores for the level."""
        out = dict(kind="split conformal, constant width per species pair, distance band and shell pair", levels=[float(v) for v in levels], n_structures=self.n, channels={})
        for g in sorted(self.err):
            e, r, _ = self._pair(g)
            edges = np.r_[0.0, np.quantile(r[r >= 0], np.linspace(0.0, 1.0, n_dist + 1))[1:-1], np.inf] if (r >= 0).any() else np.array([0.0, np.inf])
            band = np.where(r < 0, 0, 1 + _band(edges, r)); s1, s2 = self._shells(g); rows = []
            for b in range(len(edges)):
                x = e[band == b]
                rows.append(dict(n=int(len(x)), whole=[_quantile(x.max((1, 2)), v) if len(x) else float("inf") for v in levels],
                                 pooled=[_quantile(x.ravel(), v) for v in levels],
                                 element=[[[_quantile(x[:, a, c].ravel(), v) for v in levels] for c in s2] for a in s1]))
            out["channels"].setdefault(g[0], {})[f"{g[1]}-{g[2]}"] = dict(edges=[float(v) for v in edges], bands=rows)
        return out

    def coverage(self, cal):
        """Per channel and level: (level, fraction of elements inside their half-width, smallest and
        largest fraction of a structure, fraction of blocks wholly inside the whole-block width)."""
        out = {}
        for g in sorted(self.err):
            rec = cal["channels"].get(g[0], {}).get(f"{g[1]}-{g[2]}")
            if rec is None:
                continue
            e, r, s = self._pair(g); band = np.where(r < 0, 0, 1 + _band(rec["edges"], r)); s1, s2 = self._shells(g)
            for li in range(len(cal["levels"])):
                h = np.full((len(rec["bands"]),) + e.shape[1:], np.inf); w = np.array([b["whole"][li] for b in rec["bands"]])
                for b, row in enumerate(rec["bands"]):
                    for i, a in enumerate(s1):
                        for j, c in enumerate(s2):
                            h[b, a, c] = row["element"][i][j][li]
                hit = e <= h[band]
                t = out.setdefault(g[0], [dict(hit=0, n=0, per={}, whole=0, blocks=0) for _ in cal["levels"]])[li]
                t["hit"] += int(hit.sum()); t["n"] += hit.size; t["whole"] += int((e.max((1, 2)) <= w[band]).sum()); t["blocks"] += len(e)
                for q in np.unique(s):
                    p = t["per"].setdefault(int(q), [0, 0]); p[0] += int(hit[s == q].sum()); p[1] += hit[s == q].size
        return {c: [dict(level=lv, elements=t["hit"] / t["n"], structure_min=min(a / b for a, b in t["per"].values()),
                         structure_max=max(a / b for a, b in t["per"].values()), whole=t["whole"] / t["blocks"]) for lv, t in zip(cal["levels"], v)] for c, v in out.items()}


def block_half_widths(cal, orbitals, numbers, positions, cell, keys, channel="H", level=0.9, whole=False):
    """{key: half-widths} for the blocks `keys` of a structure, from a `BlockErrors.calibrate` record:
    an array in the block's shape (each element's shell pair and distance band), or with `whole`
    one number that bounds the block's largest element error.  inf where the calibration has no
    such species pair or too few blocks."""
    numbers, positions, cell, i = np.asarray(numbers), np.asarray(positions, float), np.asarray(cell, float), _level(cal, level)
    tab, out = {}, {}
    for k in keys:
        z = (int(numbers[k[3]]), int(numbers[k[4]])); rec = cal["channels"].get(channel, {}).get(f"{z[0]}-{z[1]}")
        shape = (orbitals.n_orb(z[0]), orbitals.n_orb(z[1]))
        if rec is None:
            out[k] = float("inf") if whole else np.full(shape, np.inf); continue
        on = k[:3] == (0, 0, 0) and k[3] == k[4]
        b = 0 if on else 1 + int(_band(rec["edges"], np.linalg.norm(positions[k[4]] - positions[k[3]] + np.asarray(k[:3], float) @ cell)))
        if whole:
            out[k] = rec["bands"][b]["whole"][i]; continue
        if (z, b) not in tab:
            h = np.empty(shape)
            for a in range(len(orbitals.ls(z[0]))):
                for c in range(len(orbitals.ls(z[1]))):
                    h[orbitals.shell_slice(z[0], a), orbitals.shell_slice(z[1], c)] = rec["bands"][b]["element"][a][c][i]
            tab[z, b] = h
        out[k] = tab[z, b]
    return out


def format_block_calibration(cal):
    lines = [f"block calibration: {cal['n_structures']} structures; half-widths in 1e-3 of the data's unit at " + ", ".join(f"{v:g}" for v in cal["levels"])
             + ": of an element (all shell pairs; the file has one per shell pair), then of a whole block"]
    for c, pairs in cal["channels"].items():
        for z, rec in pairs.items():
            for b, row in enumerate(rec["bands"]):
                if row["n"]:
                    where = "on-site" if b == 0 else f"{rec['edges'][b - 1]:5.2f} to {rec['edges'][b]:5.2f}"
                    lines.append(f"  {c:5s} {z:7s} {where:15s} {row['n']:8d} blocks  " + "".join(f"{1e3 * v:9.4f}" for v in row["pooled"]) + "   " + "".join(f"{1e3 * v:9.4f}" for v in row["whole"]))
    return lines


def format_block_coverage(cov):
    lines = ["coverage of the calibrated block intervals (channel, level: elements inside; range over structures; blocks wholly inside the whole-block width):"]
    for c, rows in cov.items():
        lines += [f"  {c:5s} {r['level']:.2f}: {r['elements']:.3f}; {r['structure_min']:.3f} to {r['structure_max']:.3f}; {r['whole']:.3f}" for r in rows]
    return lines
