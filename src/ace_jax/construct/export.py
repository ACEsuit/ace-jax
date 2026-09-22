"""Disposable npz bridge writer for authored models.

Packages an `Authoring` bundle into exactly the npz schema `eval.io.load`
reads (the same one `julia/export_model.jl` writes):

    A2B_{rows,cols,vals,shape}   sparse coupling triplets (one nonzero/column)
    aspec_r, aspec_y             0-based A-basis indices
    aa_spec_{1..K}               per-order A-index tuples, (n_v, order) int32
    elements, rcuts, pair_rcuts  informational structure
    rnl_transform (NZ,NZ,7), rnl_envelope (NZ,NZ,5), rnl_Wnlq, polys_A/B/C
    pair_transform (NZ,NZ,7), pair_envelope (NZ,NZ,2), pair_Wnlq, pair_polys_*
    WB (n_B, NZ), Wpair (n_pair, NZ), E0 (NZ,)
    meta_json                    schema_version = 1

This file is deliberately throwaway: once Python-authored models are the
source of truth, the export/import round-trip disappears and with it this
module.  The round-trip test (`tests/test_python_authoring.py::test_bridge_*`)
pins the writer against the committed Julia fixture, so any schema drift on
either side is caught.
"""

import json

import numpy as np


def save_npz(path, auth):
    """Write `auth` (an `Authoring`) to `path` in the eval-io schema.

    The meta written here is derived from the model being saved, not from the
    authoring defaults: any tree patched onto a different branch (e.g. the
    fixture-injection round trip) must save as what it now is, or the loader
    silently routes branch arrays to placeholders.
    """
    model, meta = auth.model, auth.meta
    meta = json.loads(json.dumps(meta))            # deep copy; don't mutate auth.meta
    NZ = len(meta["elements"])
    meta["radial_kind"] = model.radial_kind
    meta["pair_radial_kind"] = model.pair_radial_kind
    meta["pair_envelope_kind"] = model.pair_envelope_kind
    meta["ybasis_kind"] = ("real_solidharmonics" if model.ysolid
                           else "real_sphericalharmonics")
    out = {
        "A2B_rows": np.asarray(model.a2b_rows, np.int32),
        "A2B_cols": np.asarray(model.a2b_cols, np.int32),
        "A2B_vals": np.asarray(model.a2b_vals, np.float64),
        "A2B_shape": np.asarray(model.A2B.shape, np.int64),
        "aspec_r": np.asarray(model.aspec_r, np.int32),
        "aspec_y": np.asarray(model.aspec_y, np.int32),
        "elements": np.asarray(meta["elements"], np.int64),
        "rcuts": np.full(NZ, meta["rcut"]),
        "pair_rcuts": np.full((NZ, NZ), meta["rcut"]),
        "rnl_transform": np.asarray(model.rnl_transform, np.float64),
        "rnl_envelope": np.asarray(model.rnl_envelope, np.float64),
        "pair_transform": np.asarray(model.pair_transform, np.float64),
        "pair_envelope": np.asarray(model.pair_envelope, np.float64),
        "WB": np.asarray(model.WB, np.float64),
        "Wpair": np.asarray(model.Wpair, np.float64),
        "E0": np.asarray(model.E0, np.float64),
    }
    if meta["radial_kind"] == "analytic":
        out["rnl_Wnlq"] = np.asarray(model.rnl_Wnlq, np.float64)
        out["polys_A"] = np.asarray(model.polys_A, np.float64)
        out["polys_B"] = np.asarray(model.polys_B, np.float64)
        out["polys_C"] = np.asarray(model.polys_C, np.float64)
    else:
        out["rnl_spline_coefs"] = np.asarray(model.rnl_coefs, np.float64)
        meta["rnl_spline"] = {"x0": model.rnl_grid[0], "h": model.rnl_grid[1],
                              "n": model.rnl_grid[2],
                              "ncoef": model.rnl_coefs.shape[2]}
    if meta["pair_radial_kind"] == "analytic":
        out["pair_Wnlq"] = np.asarray(model.pair_Wnlq, np.float64)
        out["pair_polys_A"] = np.asarray(model.pair_polys_A, np.float64)
        out["pair_polys_B"] = np.asarray(model.pair_polys_B, np.float64)
        out["pair_polys_C"] = np.asarray(model.pair_polys_C, np.float64)
    else:
        out["pair_spline_coefs"] = np.asarray(model.pair_coefs, np.float64)
        meta["pair_spline"] = {"x0": model.pair_grid[0], "h": model.pair_grid[1],
                               "n": model.pair_grid[2],
                               "ncoef": model.pair_coefs.shape[2]}
    for k, spec in enumerate(model.aa_specs):
        out[f"aa_spec_{k+1}"] = np.asarray(spec, np.int32)
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(path, **out)
