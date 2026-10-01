"""Per-atom posterior-served arrays for the big-cell error files (used by modal_bench365.big_errors)."""
import numpy as np

# calculator property -> key in the *_err.npz ("forces_std" keeps its historical name "sd")
KEYS = {"forces_std": "sd", "forces_q": "forces_q", "forces_group": "forces_group", "forces_cov": "forces_cov"}


def collect(calc, atoms_iter, props=tuple(KEYS)):
    """Evaluate `props` on every Atoms (calc attached by the caller or not: properties are read through
    calc.get_property) and concatenate per atom.  forces_group is stored int16, forces_cov float32 (N,3,3)."""
    acc = {p: [] for p in props}
    for a in atoms_iter:
        for p in props:
            acc[p].append(np.asarray(calc.get_property(p, a)))
    out = {}
    for p, v in acc.items():
        if not v:
            continue
        x = np.concatenate(v)
        out[KEYS[p]] = x.astype(np.int16) if p == "forces_group" else x.astype(np.float32) if p == "forces_cov" else x
    return out
