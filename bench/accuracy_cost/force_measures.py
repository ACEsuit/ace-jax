"""Force-error measures beside the RMSE, on the test set: results/force_measures.jsonl, one row per (family, size).

    python -m accuracy_cost.force_measures FAMILY [SIZE ...]        (from bench/, in the family's environment)

Why (A. Seko, 2026-11-17): pypolymlp weights a force component above 1 eV/A by 1/|F| (mlp_dev/core/utils_weights.py),
so a plain RMSE is dominated by the components its fit holds least important.  The RMSE stays the benchmark's
measure and the target's; these columns show what it hides.  LARGE = 1 eV/A is pypolymlp's threshold.

  mae_F, median_F            mean and median |dF| over components
  rmse_F_small, rmse_F_large RMSE over components with |F_DFT| <= LARGE and > LARGE
  frac_large                 the fraction of components above LARGE (a property of the test set)
  share_large                their share of the summed squared error
  rel_F_large                mean |dF| / |F_DFT| over them
  wrmse_F                    RMSE of min(1, 1/|F_DFT|) dF: pypolymlp's force weighting applied to every model
"""
import json
import sys

import numpy as np

from . import common, evaluate
from .common import DATA, RESULTS

LARGE = 1.0   # eV/A


def measures(dF, F):
    """The measures above from force errors `dF` and reference forces `F` (any shape; components pooled)."""
    dF, a = np.ravel(dF), np.abs(np.ravel(F))
    big = a > LARGE
    rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if x.size else None
    return {"rmse_F": rms(dF), "mae_F": float(np.mean(np.abs(dF))), "median_F": float(np.median(np.abs(dF))),
            "rmse_F_small": rms(dF[~big]), "rmse_F_large": rms(dF[big]), "frac_large": float(np.mean(big)),
            "share_large": float(np.sum(dF[big] ** 2) / np.sum(dF ** 2)) if np.any(dF) else None,
            "rel_F_large": float(np.mean(np.abs(dF[big]) / a[big])) if big.any() else None,
            "wrmse_F": rms(np.minimum(1.0, 1.0 / np.maximum(a, 1e-12)) * dF)}


def score(family, size, atoms=None, device="cpu"):
    from .split import read_atoms
    st = common.read_status(common.run_dir(family, size))
    if not st or st.get("status") != "ok":
        raise RuntimeError(f"{family}/{size}: no successful fit")
    want = json.loads((DATA / "split.json").read_text())["hash"]
    if st.get("split_hash") != want:
        raise evaluate.SplitMismatch(f"{family}/{size} was fitted on split {st.get('split_hash')}, test is {want}")
    atoms = atoms if atoms is not None else read_atoms(DATA / "test.xyz")
    rows = evaluate.errors(atoms, evaluate.load_calc(family, st["model"], device))
    row = {"family": family, "size": size, "split_hash": want,
           **measures(np.concatenate([r["dF"].ravel() for r in rows]),
                      np.concatenate([a.arrays["forces"].ravel() for a in atoms]))}
    common.upsert_jsonl(RESULTS / "force_measures.jsonl", row, ("family", "size"))
    return row


if __name__ == "__main__":
    from .split import read_atoms
    fam = sys.argv[1]                                # default sizes: the family's scored rows (metrics.jsonl)
    sizes = sys.argv[2:] or [r["size"] for r in map(json.loads, open(RESULTS / "metrics.jsonl")) if r["family"] == fam]
    test = read_atoms(DATA / "test.xyz")
    for s in sizes:
        try:
            r = score(fam, s, test)
            print(f"{fam}/{s}: rmse {r['rmse_F']:.4f}  mae {r['mae_F']:.4f}  small {r['rmse_F_small']:.4f}  "
                  f"large {r['rmse_F_large']:.4f}  share {r['share_large']:.2f}", flush=True)
        except Exception as ex:                      # a family's environment or a model may be missing here
            print(f"{fam}/{s}: skipped ({ex!r:.200})", flush=True)
