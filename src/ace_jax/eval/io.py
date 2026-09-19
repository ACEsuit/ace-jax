"""Load an npz exported by stage1/export_model.jl.

npz rather than JSON: Julia writes matrices column-major, so 2-D arrays
round-trip through JSON transposed.  NPZ.jl records Fortran order and numpy
restores the declared shape, so array orientation here is exactly as written on
the Julia side -- `tests/test_roundtrip.py` pins that rather than trusting it.

Two model families are supported, and they differ in more than one place:

              ace1_model              ace_model
  rbasis      spline                  analytic (Wnlq + 3-term recursion)
  pairbasis   spline                  spline          <- both splined
  Ylm         spherical               solid
  pair env    ACE1_PolyEnvelope1sR    PolyEnvelope1sR <- different formulas

so the branches are per-basis, not per-model.
"""

import json

import jax.numpy as jnp
import numpy as np

from .model import ACEModel


def _a2b(z, dtype):
    """A2B, from sparse triplets when present.

    It is extremely sparse -- one nonzero per column -- so at a few thousand
    basis functions the dense form is hundreds of MB while the triplets are a
    fraction of one.  It is densified here because the contraction is a single
    matmul and dense is faster at these sizes; if it ever stops fitting, this is
    the place to switch to `jax.experimental.sparse.BCOO`.
    """
    if "A2B" in z.files:
        return jnp.asarray(z["A2B"], dtype=dtype)      # older exports
    shape = tuple(int(v) for v in z["A2B_shape"])
    dense = np.zeros(shape, dtype=np.float64)
    dense[z["A2B_rows"], z["A2B_cols"]] = z["A2B_vals"]
    return jnp.asarray(dense, dtype=dtype)


def _a2b_triplets(z, n_B, n_AA):
    if "A2B" in z.files:
        d = np.asarray(z["A2B"])
        r, c = np.nonzero(d)
        return r.astype(np.int32), c.astype(np.int32), d[r, c]
    return (np.asarray(z["A2B_rows"], np.int32), np.asarray(z["A2B_cols"], np.int32),
            np.asarray(z["A2B_vals"]))


def load(path, dtype=jnp.float64, a2b_sparse=False, edge_a_kind="gather", fold=True):
    """Load a model.  Caller controls dtype; nothing here touches jax.config, so
    f64 requires the caller to have enabled x64 first.

    `a2b_sparse` selects a gather/segment-sum contraction instead of a dense
    matmul against A2B.  A2B is ~0.07% occupied at 1429 basis functions, so the
    dense form does far more arithmetic than needed at large basis; it is
    retained as the default because it is faster at small basis.

    `edge_a_kind` selects how the A-basis product is formed: "gather" (default,
    today's behaviour) or "matmul", an algebraically identical one-hot form whose
    adjoint is a matmul rather than a scatter.  Which is faster depends on the
    backend, the dtype and the edge-buffer length -- see `calibrate_edge_a`,
    and do not guess from the platform.

    `fold` (default True) folds the linear readout through A2B (see
    `fold_readout`); pass False to keep the B-materialising path, e.g. to
    measure the fold by difference.  Descriptors are unaffected either way.
    """
    if edge_a_kind not in ("gather", "matmul"):
        raise ValueError(f'edge_a_kind must be "gather" or "matmul", got {edge_a_kind!r}')
    z = np.load(path)
    meta = json.loads(bytes(z["meta_json"]).decode())
    if meta["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version {meta['schema_version']}")

    def A(k, default=None):
        if k not in z.files:
            if default is None:
                raise KeyError(f"{k} missing from {path}")
            return jnp.zeros(default, dtype)
        return jnp.asarray(z[k], dtype=dtype)

    kind = meta["radial_kind"]
    # the exporter writes the factorised radial when a frozen embedding with
    # uniform cutoffs makes the table separable; it falls back silently otherwise
    factorised = "rnl_spline_coefs_single" in z.files
    if factorised:
        kind = "spline_factorised"
    pkind = meta.get("pair_radial_kind", "spline")
    for k in (kind, pkind):
        if k not in ("spline", "analytic", "spline_factorised"):
            raise NotImplementedError(f"unknown radial_kind {k!r}")

    _tr = _a2b_triplets(z, meta["n_B"], meta["n_AA"])
    rs = meta.get("rnl_spline") or {"x0": 0.0, "h": 1.0, "n": 2}
    ps_ = meta.get("pair_spline") or {"x0": 0.0, "h": 1.0, "n": 2}
    n_orders = len(meta["aa_lens"])
    z0 = (1, 1, 1, 1)          # placeholder shape for the unused branch

    _ar = jnp.asarray(z["aspec_r"], jnp.int32)
    _ay = jnp.asarray(z["aspec_y"], jnp.int32)
    # Widths of the two embeddings the A-basis gathers from.  Taken from the
    # coefficient arrays rather than from max(aspec)+1, which would under-count
    # whenever the last radial function happens to be unused by any A entry.
    n_rnl = (int(z["rnl_emb_nidx"].shape[0]) if factorised
             else int(z["rnl_spline_coefs"].shape[-1]) if kind == "spline"
             else int(z["rnl_Wnlq"].shape[-2]))
    n_ylm = (int(meta["lmax"]) + 1) ** 2
    if int(_ar.max()) >= n_rnl or int(_ay.max()) >= n_ylm:
        raise ValueError(f"aspec out of range: max r {int(_ar.max())} vs n_rnl {n_rnl}, "
                         f"max y {int(_ay.max())} vs n_ylm {n_ylm}")

    def _sel(idx, width, dt):
        """One-hot (width, n_A) selector such that X @ _sel(idx, ...) == X[:, idx]."""
        return jnp.zeros((width, idx.shape[0]), dt).at[idx, jnp.arange(idx.shape[0])].set(1)
    model = ACEModel(
        rnl_coefs=A("rnl_spline_coefs", z0),
        pair_coefs=A("pair_spline_coefs", z0),
        rnl_Wnlq=A("rnl_Wnlq", z0),
        pair_Wnlq=A("pair_Wnlq", z0),
        polys_A=A("polys_A", (2,)), polys_B=A("polys_B", (2,)), polys_C=A("polys_C", (2,)),
        pair_polys_A=A("pair_polys_A", (2,)), pair_polys_B=A("pair_polys_B", (2,)),
        pair_polys_C=A("pair_polys_C", (2,)),
        rnl_transform=A("rnl_transform"),
        pair_transform=A("pair_transform"),
        rnl_envelope=A("rnl_envelope"),
        pair_envelope=A("pair_envelope"),
        A2B=_a2b(z, dtype),
        a2b_rows=jnp.asarray(_tr[0]), a2b_cols=jnp.asarray(_tr[1]),
        a2b_vals=jnp.asarray(_tr[2], dtype=dtype),
        a2b_sparse=bool(a2b_sparse),
        WB=A("WB"), Wpair=A("Wpair"), E0=A("E0"),
        aspec_r=_ar,
        aspec_y=_ay,
        edge_a_kind=edge_a_kind,
        rnl_coefs_single=(A("rnl_spline_coefs_single") if factorised else None),
        rnl_embedding=(A("rnl_embedding") if factorised else None),
        rnl_emb_nidx=(jnp.asarray(z["rnl_emb_nidx"], jnp.int32) if factorised else None),
        rnl_emb_kidx=(jnp.asarray(z["rnl_emb_kidx"], jnp.int32) if factorised else None),
        a_sel_r=_sel(_ar, n_rnl, dtype) if edge_a_kind == "matmul" else None,
        a_sel_y=_sel(_ay, n_ylm, dtype) if edge_a_kind == "matmul" else None,
        aa_specs=tuple(jnp.asarray(z[f"aa_spec_{k+1}"], jnp.int32) for k in range(n_orders)),
        lmax=int(meta["lmax"]),
        ysolid=(meta["ybasis_kind"] == "real_solidharmonics"),
        radial_kind=kind,
        pair_radial_kind=pkind,
        pair_envelope_kind=meta["pair_envelope_kind"],
        rnl_grid=(float(rs["x0"]), float(rs["h"]), int(rs["n"])),
        pair_grid=(float(ps_["x0"]), float(ps_["h"]), int(ps_["n"])),
        elements=tuple(int(e) for e in meta["elements"]),
    )
    if fold:
        from .model import fold_readout
        model = fold_readout(model)
    return model, meta, z
