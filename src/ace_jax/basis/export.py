"""Disposable npz bridge writer for authored models.

Packages a `Basis` bundle into exactly the npz schema `eval.io.load`
reads (the same one `julia/export_model.jl` writes):

    A2B_{rows,cols,vals,shape}   sparse coupling triplets (one nonzero/column)
    aspec_r, aspec_y             0-based A-basis indices
    aa_spec_{1..K}               per-order A-index tuples, (n_v, order) int32
    elements, rcuts, pair_rcuts  informational structure
    rnl_transform (NZ,NZ,7), rnl_envelope (NZ,NZ,5), and per radial_kind:
        analytic            rnl_Wnlq, polys_A/B/C
        spline              rnl_spline_coefs (NZ,NZ,ncoef,n_rnl)
        spline_factorised   rnl_spline_coefs_single (ncoef,n1), rnl_embedding
                            (NZ,d), rnl_emb_nidx/kidx (n_rnl,)
    pair_transform (NZ,NZ,7), pair_envelope (NZ,NZ,2), pair_Wnlq, pair_polys_*
    WB (n_B, NZ), Wpair (n_pair, NZ), E0 (NZ,)
    meta_json                    schema_version = 1

This file is the compatibility path only: Python callers evaluate the
in-memory tree directly (`Basis.eval_pair`, no npz round-trip); the file
serves ACEfit-fitted interchange and shell hand-off (`ace-jax basis
--out`).  The round-trip test (`tests/test_basis_build.py::test_bridge_*`)
pins the writer against the committed Julia fixture, so any schema drift on
either side is caught.
"""

import json

import numpy as np


def save_npz(path, auth):
    """Write `auth` (a `Basis`) to `path` in the eval-io schema.

    The meta written here is derived from the model being saved, not from the
    authoring defaults: any tree patched onto a different branch (e.g. the
    fixture-injection round trip) must save as what it now is, or the loader
    silently routes branch arrays to placeholders.  The derivation and the
    structural checks live in `Basis.eval_pair` -- the file path and the
    in-memory hand-off share one source of truth.
    """
    model, meta = auth.eval_pair()
    model.require_full("save_npz")
    NZ = len(meta["elements"])
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
    elif meta["radial_kind"] == "spline_factorised":
        # the exporter's separable layout; the loader keys the branch on the
        # presence of rnl_spline_coefs_single
        out["rnl_spline_coefs_single"] = np.asarray(model.rnl_coefs_single, np.float64)
        out["rnl_embedding"] = np.asarray(model.rnl_embedding, np.float64)
        out["rnl_emb_nidx"] = np.asarray(model.rnl_emb_nidx, np.int32)
        out["rnl_emb_kidx"] = np.asarray(model.rnl_emb_kidx, np.int32)
        meta["rnl_spline"] = {"x0": model.rnl_grid[0], "h": model.rnl_grid[1],
                              "n": model.rnl_grid[2],
                              "ncoef": int(model.rnl_coefs_single.shape[0])}
    elif meta["radial_kind"] == "spline":
        out["rnl_spline_coefs"] = np.asarray(model.rnl_coefs, np.float64)
        meta["rnl_spline"] = {"x0": model.rnl_grid[0], "h": model.rnl_grid[1],
                              "n": model.rnl_grid[2],
                              "ncoef": model.rnl_coefs.shape[2]}
    else:
        raise NotImplementedError(f"save_npz: radial_kind {meta['radial_kind']!r}")
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
    if getattr(auth, "gamma", None) is not None and auth.meta.get("basis", {}).get("with_gamma", True):
        out["gamma"] = np.asarray(auth.gamma, np.float64)    # the prior it was built with: no rebuild on load
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(path, **out)


def readout_to_npz(c, n_B, n_pair, NZ):
    """Split a length-(n_B + n_pair) * NZ linear readout, species-blocked as
    ACEModel.site_descriptors / fit.rows._place (all NZ B-blocks first, then
    all NZ pair blocks), into the npz arrays WB (n_B, NZ), Wpair (n_pair, NZ):
    e_i = B_i . WB[:, z_i] + Apair_i . Wpair[:, z_i] + E0[z_i]."""
    c = np.asarray(c, np.float64)
    if c.shape != ((n_B + n_pair) * NZ,):
        raise ValueError(f"readout shape {c.shape} != ((n_B + n_pair) * NZ,) = ({(n_B + n_pair) * NZ},)")
    WB = c[:NZ * n_B].reshape(NZ, n_B).T
    Wpair = c[NZ * n_B:].reshape(NZ, n_pair).T
    return np.ascontiguousarray(WB), np.ascontiguousarray(Wpair)


def patch_radial_npz(src, dst, model, readout=None):
    """Copy the npz at `src` to `dst` with the tensor radial replaced by
    `model`'s analytic one (rnl_Wnlq + polys_A/B/C; any rnl spline arrays are
    dropped and meta radial_kind/rnl_spline updated).  `readout`: the linear
    readout fitted FOR these radials (length (n_B + n_pair) * NZ, species-
    blocked; see readout_to_npz), written into WB/Wpair.  readout=None copies
    WB/Wpair verbatim -- correct only when the radials are unchanged (e.g. an
    exact branch conversion), stale otherwise.  Coupling, pair basis and E0
    are always copied verbatim (a readout with joint-E0 columns appended has them
    dropped: they shift the fit's pre-fit E0, which this file does not hold).  This is how a learned radial is written back
    from a model that was loaded rather than authored (save_npz needs an
    Basis)."""
    model.require_full("patch_radial_npz")
    if model.radial_kind != "analytic":
        raise ValueError(f"patch_radial_npz: model radial_kind {model.radial_kind!r} is not analytic")
    with np.load(src, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    meta = json.loads(bytes(out["meta_json"]).decode())
    for k in ("rnl_spline_coefs", "rnl_spline_coefs_single", "rnl_embedding",
              "rnl_emb_nidx", "rnl_emb_kidx"):
        out.pop(k, None)
    out["rnl_Wnlq"] = np.asarray(model.rnl_Wnlq, np.float64)
    out["polys_A"] = np.asarray(model.polys_A, np.float64)
    out["polys_B"] = np.asarray(model.polys_B, np.float64)
    out["polys_C"] = np.asarray(model.polys_C, np.float64)
    if readout is not None:
        (n_B, NZ), n_pair = out["WB"].shape, out["Wpair"].shape[0]
        readout = np.asarray(readout)
        if readout.size == (n_B + n_pair) * NZ + NZ:     # + joint-E0 columns (an e0='lsq' problem):
            readout = readout[:(n_B + n_pair) * NZ]      # shifts of that fit's pre-fit E0, not of this file's
        out["WB"], out["Wpair"] = readout_to_npz(readout, n_B, n_pair, NZ)
    meta["radial_kind"] = "analytic"
    meta["rnl_spline"] = None
    meta["radial_learned"] = bool(model.radial_learned)
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(dst, **out)


def mark_radial_learned(src, dst=None, learned=True):
    """Set meta_json "radial_learned" in the npz at `src` (in place, or into
    `dst`), leaving every array as is.  For learned-radial files written before
    the flag existed, which load as not learned, so `lean`'s default keeps
    their radial analytic: `mark_radial_learned("model.npz")`.  (Or pass
    spline_tol=1e-10 to ACECalculator / export_lammps / lean instead.)"""
    with np.load(src, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    meta = json.loads(bytes(out["meta_json"]).decode())
    meta["radial_learned"] = bool(learned)
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(src if dst is None else dst, **out)
