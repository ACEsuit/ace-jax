"""How much a model loses evaluated in float32: energy, force and stress differences against
float64 on one structure, through `ACECalculator` (lean, as deployed).

What sets the loss is the model, not the evaluator: a readout whose terms cancel (sum_k |c_k B_k|
hundreds to thousands of times |E_i|, as in near-interpolating fits with large coefficients)
turns float32 rounding of B into 1e-4..1e-3 relative force errors, where a well-conditioned
model of the same basis stays near 1e-5 (docs/dev/fp32-results.md).  So the check is run per
model, on the structure it will be used for; `export_lammps(precision_structure=)` records it."""
import warnings

import numpy as np

PRECISION_WARN = {"dE_meV_atom": 0.1, "dF_rel": 1e-4}
"""Thresholds above which `warn_precision` warns: 0.1 meV/atom of energy (the scale of small
energy differences, and the NVE drift floor it sets) and 1e-4 of relative force error."""


def precision_check(model, meta, atoms, dtype="float32", **calc_kw):
    """`dtype` evaluation against float64 on `atoms`: {"dtype", "n_atoms", "dE_meV_atom",
    "dF_rel" (|dF| / |F|, Frobenius), "dF_max_meV_A", "dS_rel" (None without a cell)}.
    Needs x64 for the reference.  calc_kw go to both `ACECalculator`s."""
    import jax
    if not jax.config.jax_enable_x64:
        raise RuntimeError("precision_check needs the float64 reference: enable jax x64 first")
    from .point import ACECalculator
    out = {}
    for key, dt in (("ref", None), ("low", np.dtype(dtype))):
        at = atoms.copy()
        at.calc = ACECalculator(model, meta, dtype=dt, **calc_kw)
        S = at.get_stress() if at.cell.volume > 0 and all(at.pbc) else None
        out[key] = (at.get_potential_energy(), at.get_forces(), S)
    (E, F, S), (e, f, s) = out["ref"], out["low"]
    n = len(atoms)
    return {"dtype": str(np.dtype(dtype)), "n_atoms": n,
            "dE_meV_atom": 1e3 * abs(e - E) / n,
            "dF_rel": float(np.linalg.norm(f - F) / max(np.linalg.norm(F), 1e-300)),
            "dF_max_meV_A": 1e3 * float(np.abs(f - F).max()),
            "dS_rel": None if S is None else float(np.linalg.norm(s - S) / max(np.linalg.norm(S), 1e-300))}


def warn_precision(check, what="the model"):
    """Warn when `check` (a `precision_check` result) exceeds PRECISION_WARN; returns the
    exceeded keys."""
    over = [k for k, lim in PRECISION_WARN.items() if check[k] > lim]
    if over:
        warnings.warn(f"{what} loses accuracy in {check['dtype']}: energy {check['dE_meV_atom']:.3g} meV/atom, "
                      f"relative force error {check['dF_rel']:.2g} against float64 on {check['n_atoms']} atoms "
                      f"(thresholds {PRECISION_WARN}); the readout's terms likely cancel (large coefficients). "
                      "Use float64, or a better-regularised fit", stacklevel=3)
    return over
