"""Python-side ACEModel authoring.

`build_model(elements, order, totaldegree, ...)` constructs a frozen
`ACEModel` entirely in Python: the basis spec comes from `build_spec`, the
coupling from the ET shim (`couple`, the only Julia piece), and all radial
coefficients from seeded `radial_init` initialisers.  The result evaluates
energies/descriptors immediately and can be packaged with `export.save_npz`.

dtype convention: arrays are built in float64; evaluating in float64 requires
the caller to have enabled x64 (`jax.config.update("jax_enable_x64", True)`) —
nothing here touches jax.config, same contract as `eval.io.load`.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..eval.model import ACEModel, fold_readout
from . import radial_init as ri
from .coupling import couple
from .prior import smoothness_prior
from .spec import build_spec


class Authoring(NamedTuple):
    """An authored model plus everything needed to reproduce or package it."""

    model: ACEModel
    meta: dict                      # the writer's meta_json payload
    nnll_spec: tuple                # per-B-row body dump, ET's get_nnll_spec
    Rnl_spec: tuple
    Ylm_spec: tuple
    aa_sig: tuple                   # per-A2B-column (n,l,m) signatures
    aspec: tuple                    # 0-based (Rnl_idx, Ylm_idx) per A function
    aa_specs: tuple                 # per-order (n_v, order) 0-based A indices
    nnll: tuple                     # derived body lists (== ET dump)
    gamma: np.ndarray               # (len_basis,) algebraic smoothness prior

    def eval_pair(self):
        """(model, meta) ready for direct evaluation -- the in-memory hand-off.

        The four branch-selector meta keys are derived from the tree being
        handed over, exactly as `export.save_npz` does for the file path: the
        loader-side consumers (`eval.io.load`, and anything that reads meta to
        route branches) must never see authoring defaults that a patched tree
        has outgrown -- JAX clamps out-of-range branch indices silently, so a
        mismatch fails numerically, not loudly.  The meta is deep-copied; the
        caller may mutate it freely.  Structural checks guard the derivable
        keys (`_resolve` consumers read `rcut`/`elements`/`n_*` from meta).

        `ACECalculator(*auth.eval_pair())` and `site_descriptors(auth.model,
        ..., meta=meta)` evaluate the in-memory tree directly -- no npz
        round-trip; `save_npz` remains for ACEfit-fitted interchange and
        shell hand-off only."""
        import json
        model, meta = self.model, json.loads(json.dumps(self.meta))
        meta["radial_kind"] = model.radial_kind
        meta["pair_radial_kind"] = model.pair_radial_kind
        meta["pair_envelope_kind"] = model.pair_envelope_kind
        meta["ybasis_kind"] = ("real_solidharmonics" if model.ysolid
                               else "real_sphericalharmonics")
        checks = (
            ("elements vs WB", len(meta["elements"]), model.WB.shape[1]),
            ("n_B vs A2B", meta["n_B"], model.A2B.shape[0]),
            ("n_AA vs A2B", meta["n_AA"], model.A2B.shape[1]),
            ("aa_lens vs aa_specs", meta["aa_lens"], [int(g.shape[0]) for g in model.aa_specs]),
            ("lmax", meta["lmax"], int(model.lmax)),
        )
        for what, m, t in checks:
            if m != t:
                raise ValueError(f"eval_pair: meta/tree mismatch in {what}: {m!r} vs {t!r}")
        return model, meta


def nnll_from_coupling(A2B, aa_sig, tol=1e-12):
    """Per-B-row body lists from the block-diagonal coupling: each row's first
    nonzero column names the body; its sorted (n,l,m) signature gives the
    (n,l) channels WITH multiplicity.  Replicates the exporter's
    `get_nnll_spec(m.tensor)` (EquivariantTensors sparse_ace_basis.jl:279)."""
    A2B = np.asarray(A2B)
    nnll = []
    for r in range(A2B.shape[0]):
        nz = np.flatnonzero(np.abs(A2B[r]) > tol)
        if nz.size == 0:
            raise ValueError(f"A2B row {r} has no support; not block-diagonal in nnll")
        sig = aa_sig[int(nz[0])]
        nnll.append(tuple((int(n), int(l)) for n, l, _ in sig))
    return tuple(nnll)


def build_model(elements, order, totaldegree, *, wL=1.5, rcut=5.5, r0=None,
                rin=0.0, radial_mode="glorot_normal", pair_mode="onehot",
                seed=0, with_gamma=True, edge_a_kind="gather"):
    """Author a frozen `ace_model`-family model in memory.

    elements: atomic numbers or symbols; order: correlation order (body
    order - 1); totaldegree: TotalDegree(NZ, 1/wL) level bound.  Radial
    coefficients are frozen at the seeded initial values (no fitting here).

    Returns an `Authoring`.  Requires the `authoring` extra (Julia coupling
    shim); evaluate in float64 with x64 enabled."""
    zs = ri.resolve_elements(elements)
    NZ = len(zs)
    mb, Rnl, Ylm = build_spec(NZ, order, totaldegree, wL)
    cpl = couple(mb, Rnl, Ylm)

    nnll = nnll_from_coupling(cpl.A2B, cpl.aa_sig)
    # within a row, channel order differs (signature-sorted vs original body
    # order); compare rows as multisets
    norm = lambda rows: sorted(sorted(bb) for bb in rows)
    if norm(nnll) != norm(cpl.nnll_spec):
        raise AssertionError("nnll derivation disagrees with the ET dump")

    tinit = ri.tensor_radial_init(zs, Rnl, rcut=rcut, r0=r0, rin=rin,
                                  mode=radial_mode, seed=seed)
    pair_maxn = totaldegree
    pinit = ri.pair_radial_init(zs, pair_maxn, rcut=rcut, r0=r0, rin=rin,
                                mode=pair_mode, seed=seed)

    n_B, n_AA = cpl.A2B.shape
    n_A = len(cpl.aspec)
    lmax = max(l for l, _ in Ylm)
    n_ylm = len(Ylm)
    n_pair = pair_maxn
    n_rnl = len(Rnl)
    len_basis = (n_B + n_pair) * NZ

    # frozen readout: zeros (the exporter's acefit!-fresh init)
    WB = np.zeros((n_B, NZ))
    Wpair = np.zeros((n_pair, NZ))
    E0 = np.zeros(NZ)

    # sparse coupling triplets: one nonzero per column
    rr, cc = np.nonzero(cpl.A2B)
    order_ = np.argsort(cc)
    a2b_rows = rr[order_].astype(np.int32)
    a2b_cols = cc[order_].astype(np.int32)
    a2b_vals = cpl.A2B[rr[order_], cc[order_]]

    ar = np.array([a[0] for a in cpl.aspec], np.int32)
    ay = np.array([a[1] for a in cpl.aspec], np.int32)

    z0 = (1, 1, 1, 1)  # placeholder shape for the unused spline branch
    model = ACEModel(
        rnl_coefs=jnp.zeros(z0, jnp.float64), pair_coefs=jnp.zeros(z0, jnp.float64),
        rnl_Wnlq=jnp.asarray(tinit["rnl_Wnlq"]),
        pair_Wnlq=jnp.asarray(pinit["pair_Wnlq"]),
        polys_A=jnp.asarray(tinit["polys_A"]), polys_B=jnp.asarray(tinit["polys_B"]),
        polys_C=jnp.asarray(tinit["polys_C"]),
        pair_polys_A=jnp.asarray(pinit["pair_polys_A"]),
        pair_polys_B=jnp.asarray(pinit["pair_polys_B"]),
        pair_polys_C=jnp.asarray(pinit["pair_polys_C"]),
        rnl_transform=jnp.asarray(tinit["rnl_transform"]),
        pair_transform=jnp.asarray(pinit["pair_transform"]),
        rnl_envelope=jnp.asarray(tinit["rnl_envelope"]),
        pair_envelope=jnp.asarray(pinit["pair_envelope"]),
        A2B=jnp.asarray(cpl.A2B),
        a2b_rows=jnp.asarray(a2b_rows), a2b_cols=jnp.asarray(a2b_cols),
        a2b_vals=jnp.asarray(a2b_vals, jnp.float64),
        a2b_sparse=False,
        WB=jnp.asarray(WB), Wpair=jnp.asarray(Wpair), E0=jnp.asarray(E0),
        aspec_r=jnp.asarray(ar), aspec_y=jnp.asarray(ay),
        aa_specs=tuple(jnp.asarray(g, jnp.int32) for g in cpl.aa_specs),
        lmax=int(lmax), ysolid=True,
        radial_kind="analytic", pair_radial_kind="analytic",
        pair_envelope_kind="poly1sr",
        rnl_grid=(0.0, 1.0, 2), pair_grid=(0.0, 1.0, 2),
        elements=tuple(int(e) for e in zs),
        edge_a_kind=edge_a_kind,
    )
    model = fold_readout(model)

    tensor_nnll = list(nnll)
    pair_nnll = [[(n, 0)] for n in range(1, n_pair + 1)]
    gamma = (smoothness_prior(tensor_nnll * NZ + pair_nnll * NZ)
             if with_gamma else np.zeros(len_basis))

    meta = {
        "schema_version": 1,
        "source": "ace-jax python authoring (build_model)",
        "embedding": "",
        "elements": zs,
        "order": int(order), "totaldegree": int(totaldegree),
        "radial_kind": "analytic", "pair_radial_kind": "analytic",
        "transform_kind": "agnesi_normalized",
        "envelope_kind": "poly2sx",
        "pair_envelope_kind": "poly1sr",
        "ybasis_kind": "real_solidharmonics",
        "lmax": int(lmax),
        "n_rnl": n_rnl, "n_pair": n_pair, "n_ylm": n_ylm, "n_A": n_A,
        "n_AA": n_AA, "n_B": n_B, "len_basis": int(len_basis),
        "aa_orders": [int(g.shape[1]) for g in cpl.aa_specs],
        "aa_lens": [int(g.shape[0]) for g in cpl.aa_specs],
        "rnl_spline": None, "pair_spline": None,
        "rcut": float(rcut),
        "nnll": [[list(b) for b in bb] for bb in cpl.nnll_spec],
        "authoring": {
            "wL": float(wL), "rcut": float(rcut),
            "r0": float(tinit["rnl_transform"][0, 0, 4]),
            "rin": float(rin), "radial_mode": radial_mode,
            "pair_mode": pair_mode, "seed": int(seed),
            "pair_maxn": int(pair_maxn), "with_gamma": bool(with_gamma),
        },
    }
    return Authoring(model=model, meta=meta, nnll_spec=cpl.nnll_spec, Rnl_spec=tuple(Rnl),
                     Ylm_spec=tuple(Ylm), aa_sig=cpl.aa_sig, aspec=tuple(cpl.aspec),
                     aa_specs=tuple(np.asarray(g) for g in cpl.aa_specs),
                     nnll=nnll, gamma=gamma)
