"""The E/F/V RMSE table, per config type, that `aj fit` (per split) and `aj eval` print.

Units follow the rest of the pipeline's metrics: energy and virial in meV/atom (the
virial is per configuration, divided by its atom count), forces in eV/A per component.
A configuration without a label (NaN) drops out of that column only; a type with no
label of a kind prints "-"."""
import numpy as np

UNSET = "(none)"                      # configs without a config_type


def _rmse(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def rmse_by_type(types, nat, E, Em, F, Fm, V, Vm):
    """{type: {n_cfg, n_atoms, E, F, V}} for each config type (sorted) and "all".
    types, nat: per config; E, Em: (n_cfg,) eV; F, Fm: (n_atoms, 3) eV/A; V, Vm:
    (n_cfg, 6) eV (Voigt)."""
    types = np.array([UNSET if t is None else str(t) for t in types])
    nat = np.asarray(nat)
    atom_types = np.repeat(types, nat)
    dE = 1e3 * (np.asarray(Em) - np.asarray(E)) / nat
    dF = np.asarray(Fm) - np.asarray(F)
    dV = 1e3 * (np.asarray(Vm) - np.asarray(V)) / nat[:, None]
    out = {}
    for t in sorted(set(types)) + ["all"]:
        c = np.ones(len(types), bool) if t == "all" else types == t
        a = np.ones(len(atom_types), bool) if t == "all" else atom_types == t
        out[t] = {"n_cfg": int(c.sum()), "n_atoms": int(nat[c].sum()),
                  "E": _rmse(dE[c]), "F": _rmse(dF[a].reshape(-1)), "V": _rmse(dV[c].reshape(-1))}
    return out


def format_rmse_table(stats, title):
    """An aligned plain-text table of rmse_by_type's output."""
    head = ("config type", "configs", "atoms", "E (meV/atom)", "F (eV/Å)", "V (meV/atom)")
    fmt = lambda x, p: "-" if not np.isfinite(x) else f"{x:.{p}f}"
    rows = [(t, str(s["n_cfg"]), str(s["n_atoms"]), fmt(s["E"], 2), fmt(s["F"], 4), fmt(s["V"], 2))
            for t, s in stats.items()]
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    line = lambda r: "  ".join(r[0].ljust(w[0]) if i == 0 else r[i].rjust(w[i]) for i in range(len(r)))
    rule = "-" * len(line(head))
    body = [line(r) for r in rows]
    if len(body) == 2:                    # one config type: its row would repeat "all"
        return "\n".join([f"RMSE, {title}", rule, line(head), rule, body[-1]])
    return "\n".join([f"RMSE, {title}", rule, line(head), rule, *body[:-1], rule, body[-1]])
