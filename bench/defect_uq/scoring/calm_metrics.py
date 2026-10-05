"""CALM-style ranking metrics (arXiv:2609.40060; plan: docs/dev/force-uq-vs-calm.md, Step 7) on served runs.

Reads <run>/big3*_err.npz (modal_bench365.big_errors: per-atom dF, sd = forces_std, forces_q, family,
r_core, fixed) and reports per family, free atoms only:
- Spearman rho(score, |dF|) per atom;
- ROC-AUC of the score for |dF| > t, t swept over 0.05-5 eV/A (n_pos given; blank below 10 positives);
- contamination P(|dF| > 0.5 eV/A | confident) when the lowest-score fraction k of atoms is kept.
The correlations are not comparable to the paper's numbers: models, data and error scales all differ.

    python calm_metrics.py --out calm_metrics.md RUN_DIR [RUN_DIR ...]
"""
import argparse
import glob
import pathlib

import numpy as np
from scipy.stats import rankdata, spearmanr

THRESH = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
KEEP = (0.5, 0.8, 0.9, 0.95, 0.99)
CONTAM = 0.5
SCORES = ("sd", "forces_q")


def auroc(score, pos):
    """Mann-Whitney ROC-AUC with tied scores counted half; NaN without both classes."""
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def contamination(score, err, keep, t=CONTAM):
    """P(err > t) among the round(keep * n) atoms of lowest score (stable order)."""
    o = np.argsort(score, kind="stable")[:max(1, int(round(keep * len(score))))]
    return float(np.mean(np.asarray(err)[o] > t))


def load(run):
    files = sorted(glob.glob(str(pathlib.Path(run) / "big3*_err.npz")))
    if not files:
        raise FileNotFoundError(f"no big3*_err.npz in {run}")
    parts = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        free = ~z["fixed"].astype(bool)
        d = {"err": np.linalg.norm(z["dF"], axis=1) if "dF" in z.files else z["err"], "family": z["family"].astype(str),
             "r_core": z["r_core"]}
        d.update({k: z[k] for k in SCORES if k in z.files})
        parts.append({k: np.asarray(v)[free] for k, v in d.items()})
    keys = set.intersection(*(set(p) for p in parts))
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def subsets(d):
    for fam in sorted(set(d["family"])):
        m = d["family"] == fam
        yield fam, m
        if fam == "crack":
            yield "crack tip (<= 10 A)", m & (d["r_core"] <= 10)
    yield "all big cells", np.ones(len(d["err"]), bool)


def report(runs):
    out = []
    for run in runs:
        d = load(run)
        name = pathlib.Path(run).name
        sc = [k for k in SCORES if k in d]
        out += [f"### {name}", "", "| subset | atoms | score | rho | " + " | ".join(f"AUC@{t:g}" for t in THRESH)
                + " | " + " | ".join(f"P(>{CONTAM:g}) keep {k:g}" for k in KEEP) + " |",
                "|---|---|---|---|" + "---|" * (len(THRESH) + len(KEEP))]
        for lab, m in subsets(d):
            e = d["err"][m]
            npos = [int(np.sum(e > t)) for t in THRESH]
            out.append(f"| {lab} | {int(m.sum())} | n_pos | | " + " | ".join(map(str, npos)) + " | "
                       + " | ".join("" for _ in KEEP) + " |")
            for k in sc:
                s = d[k][m]
                auc = [auroc(s, e > t) if n >= 10 and n <= len(e) - 10 else float("nan") for t, n in zip(THRESH, npos)]
                out.append(f"| | | {k} | {spearmanr(s, e)[0]:.3f} | "
                           + " | ".join("" if np.isnan(a) else f"{a:.3f}" for a in auc) + " | "
                           + " | ".join(f"{contamination(s, e, kp):.4f}" for kp in KEEP) + " |")
            out.append("| | | (no ranking) | | " + " | ".join("" for _ in THRESH) + " | "
                       + " | ".join(f"{np.mean(e > CONTAM):.4f}" for _ in KEEP) + " |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--out")
    a = p.parse_args(argv)
    md = report(a.runs)
    if a.out:
        pathlib.Path(a.out).write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
