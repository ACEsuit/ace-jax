"""A linear ACEModel `.npz` as a Symmetrix JSON model, for `pair_style symmetrix/ace` in LAMMPS.

The conversion is Symmetrix's own (`symmetrix.extract_ace_data.extract_acejax_data`): ace-jax carries no copy of
the format, so the file is whatever the installed Symmetrix reads. Why export at all: Symmetrix's Kokkos kernels
are 1.4 to 2.4 times faster than ace-jax's CUDA kernel on an A100, and its pair style has none of the
JAX -> lammps-jax -> FFI path's fixed cost per step (docs/dev/specs/2026-11-19-coupled-product-basis.md).
The converter decides what it takes (today: one element, analytic radials, real solid harmonics) and raises
ValueError otherwise; this module passes that on."""
import json
import pathlib


class SymmetrixUnavailable(ImportError):
    """Symmetrix is not installed, or its version has no ace-jax converter."""


def converter():
    """Symmetrix's `extract_acejax_data`, or `SymmetrixUnavailable`."""
    try:
        from symmetrix.extract_ace_data import extract_acejax_data
    except ImportError as e:
        raise SymmetrixUnavailable(
            "a Symmetrix export needs Symmetrix with its ACE converter (symmetrix.extract_ace_data."
            f"extract_acejax_data): {e}") from e
    return extract_acejax_data


def export_symmetrix(model, out, *, coupled=True, num_bins=None, coupled_options=None):
    """Write the ace-jax model file `model` (.npz) as the Symmetrix JSON `out`; returns a summary dict.

    coupled: include Symmetrix's coupled form of the product basis (its fast GPU evaluator); coupled_options go to
    its `coupled_form` (e.g. ilp_seconds). num_bins: the radial tables' intervals (None: Symmetrix's default)."""
    model, out = pathlib.Path(model).expanduser(), pathlib.Path(out).expanduser()
    if model.suffix != ".npz":
        raise ValueError(f"{model}: a Symmetrix export takes an ace-jax .npz model")
    kw = {"coupled": coupled, "coupled_options": coupled_options}
    if num_bins is not None:
        kw["num_bins"] = int(num_bins)
    data = converter()(str(model), **kw)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(out)
    sp = data.get("species", [])
    return {"elements": list(data.get("elements", [])), "lmax": data.get("lmax"),
            "monomials": [len(s.get("monomials", ())) for s in sp],
            "coupled": [s.get("coupled") is not None for s in sp],
            "spline_error": data.get("spline_error"), "path": str(out)}
