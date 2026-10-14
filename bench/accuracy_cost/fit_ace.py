"""Linear ACE and ACE+GP through `aj fit`: the CLI's defaults, plus the shared
cutoff, weights (E, F and V sigmas) and E0 (--e0 lsq takes the isolated-atom
energy, spec)."""
import sys

import numpy as np

from . import common
from .common import DATA, RCUT

MODEL = {"ace": "model.npz", "ace_lr": "model.npz", "ace_gp": "gp_model.npz"}


def command(family, e, out):
    return ["aj", "fit", "--train", str(DATA / "train.xyz"), "--test", str(DATA / "test.xyz"), "--order", str(e["order"]),
            "--max-degree", str(e["max_degree"]), "--rcut", str(RCUT), "--virial-key", "virial", "--weights",
            str(DATA / "weights_acejax.json"), "--e0", "lsq", "--rungs", "map",
            "--m-per-species", str(e.get("m", 0) if family == "ace_gp" else 0), "--out", str(out),
            *(["--learn-radial"] if family == "ace_lr" else []),
            *(["--noise", "shared"] if family in ("ace", "ace_lr") else []),  # one noise scale (ACEfit BLR style): the weights set the balance, #64
            *(["--configs-per-batch", str(e["configs_per_batch"])] if "configs_per_batch" in e else [])]  # memory only: streamed sums


def n_params(family, model):
    z = np.load(model)
    if family in ("ace", "ace_lr"):
        return int(z["WB"].size + z["Wpair"].size)
    keys = [k for k in z.files if k.endswith("WB") or k.endswith("Wpair")]
    if not keys:
        raise ValueError(f"{model}: no ace/WB or ace/Wpair keys in {z.files}")
    return int(sum(z[k].size for k in keys))                     # + inducing count, added by the caller


def fit(family, size):
    from .split import ensure_split
    from .weights import write_all
    h = ensure_split(); write_all(DATA)
    e = common.entry(family, size)
    d = common.run_dir(family, size)
    st = common.fit_once(family, size, MODEL[family], command(family, e, d), split_hash=h)
    if st["status"] == "ok" and st.get("n_params") is None:
        st["n_params"] = n_params(family, d / MODEL[family]) + (e.get("m", 0) if family == "ace_gp" else 0)
        common.write_status(d, **{k: v for k, v in st.items() if k != "host"})
    return st


if __name__ == "__main__":
    fam, which = sys.argv[1], sys.argv[2]
    sizes = [e["size"] for e in common.load_sweep()[fam]] if which == "all" else [which]
    for s in sizes:
        print(s, fit(fam, s)["status"])
