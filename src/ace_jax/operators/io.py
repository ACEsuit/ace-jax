"""Operator models on disk: one `.npz`, laid out as the potentials' export schema (`eval.io`).

A file holds one site basis, optionally the bond basis on it, and the coefficients of an on-site
and / or an off-site model.  It is self-contained: the coupling is stored, so loading needs no
coupling library and no rebuild, and `load(save(...))` predicts the same blocks to the last bit.

    meta_json                       schema_version = 1, kind = "operator", and
        elements, channels          atomic numbers (the site basis' order); operator channels
        orbitals                    {"Z": [l, l, ...]}: the shells of each species
        site                        rcut, lmax, solid, lps [[L, parity], ...], n_B, and radial:
                                    {"kind": "table", "grid": [r_min, h, n]} or {"kind": "tensor",
                                    radial_kind, rnl_basis, rnl_grid}
        bond                        null, or rcut_bond, n_bond, n_cpl, l_cpl, r_env, n_env, l_env,
                                    n_ebond, l_ebond, env_emb
        info                        the caller's own dictionary (provenance, fit settings, errors)
    aspec_r, aspec_y                0-based A-basis indices (as the potential schema)
    aa_spec_{1..K}                  per-order A-index tuples
    A2B_{i}_{flat,cols,vals}        the coupling of lps[i]: feature * (2L+1) + m, AA column, value
    gamma_{i}                       the smoothness of each feature of lps[i] (absent: none stored)
    rtab_coefs                      radial kind "table": (NZ, NZ, n + 2, n_rnl) B-spline coefficients
    rnl_*, polys_*, pair_envelope   radial kind "tensor": `eval.radial.TensorRadial`'s arrays by name
    {onsite,offsite}_keys           (n, 4) (z, s1, s2, L) / (n, 5) (zi, zj, s1, s2, L): the blocks' L parts
    {onsite,offsite}_ptr            (n + 1,): part i has columns ptr[i]:ptr[i + 1] of
    {onsite,offsite}_coef           (n_channel, total columns) float64

Layer 3 (an application module, like `eval.io`)."""
import dataclasses
import json
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..eval.radial import TensorRadial
from . import bond as B
from .features import RadialTable, SiteBasis
from .offsite import OffsiteModel
from .onsite import OnsiteModel
from .orbitals import OrbitalBasis

SCHEMA_VERSION = 1
_BOND = ("rcut_bond", "n_bond", "n_cpl", "l_cpl", "r_env", "n_env", "l_env", "n_ebond", "l_ebond", "env_emb")
_RADIAL_STATIC = ("radial_kind", "rnl_basis", "rnl_grid")


class OperatorFile(NamedTuple):
    """What `load` returns: the models (None when the file has none) and the file's meta."""
    onsite: OnsiteModel
    offsite: OffsiteModel
    meta: dict


def _same(a, b):
    """Two site bases with the same arrays (the on-site and off-site models of one file share one)."""
    if a is b:
        return True
    fa, fb = (_site_arrays(x) for x in (a, b))
    return _site_meta(a) == _site_meta(b) and fa.keys() == fb.keys() and all(np.array_equal(fa[k], fb[k]) for k in fa)


def _site_meta(sb):
    r = sb.radial
    radial = dict(kind="table", grid=list(r.grid)) if isinstance(r, RadialTable) else dict(
        kind="tensor", **{k: (list(getattr(r, k)) if k == "rnl_grid" else getattr(r, k)) for k in _RADIAL_STATIC})
    return dict(rcut=sb.rcut, lmax=sb.lmax, solid=bool(sb.solid), lps=[list(lp) for lp in sb.lps], n_B=list(sb.n_B), radial=radial)


def _site_arrays(sb):
    out = {"aspec_r": np.asarray(sb.aspec_r), "aspec_y": np.asarray(sb.aspec_y)}
    out.update({f"aa_spec_{k + 1}": np.asarray(g) for k, g in enumerate(sb.aa_specs)})
    for i in range(len(sb.lps)):
        out.update({f"A2B_{i}_flat": np.asarray(sb.flat[i]), f"A2B_{i}_cols": np.asarray(sb.cols[i]), f"A2B_{i}_vals": np.asarray(sb.vals[i])})
        if sb.gamma is not None:
            out[f"gamma_{i}"] = np.asarray(sb.gamma[i], np.float64)
    if isinstance(sb.radial, RadialTable):
        out["rtab_coefs"] = np.asarray(sb.radial.coefs)
    elif isinstance(sb.radial, TensorRadial):
        out.update({f.name: np.asarray(getattr(sb.radial, f.name)) for f in dataclasses.fields(sb.radial)
                    if f.name not in _RADIAL_STATIC and getattr(sb.radial, f.name) is not None})
    else:
        raise TypeError(f"operators.io: cannot store a radial of type {type(sb.radial).__name__}")
    return out


def _site(z, m):
    r = m["radial"]
    if r["kind"] == "table":
        radial = RadialTable(coefs=jnp.asarray(z["rtab_coefs"]), grid=(float(r["grid"][0]), float(r["grid"][1]), int(r["grid"][2])))
    else:
        names = [f.name for f in dataclasses.fields(TensorRadial) if f.name not in _RADIAL_STATIC]
        radial = TensorRadial(**{k: jnp.asarray(z[k]) for k in names if k in z}, radial_kind=r["radial_kind"], rnl_basis=r["rnl_basis"],
                              rnl_grid=tuple(r["rnl_grid"]))
    n = len(m["lps"])
    order = sorted(int(k.rsplit("_", 1)[1]) for k in z.files if k.startswith("aa_spec_"))
    return SiteBasis(
        radial=radial, aspec_r=jnp.asarray(z["aspec_r"]), aspec_y=jnp.asarray(z["aspec_y"]),
        aa_specs=tuple(jnp.asarray(z[f"aa_spec_{k}"]) for k in order),
        flat=tuple(jnp.asarray(z[f"A2B_{i}_flat"]) for i in range(n)), cols=tuple(jnp.asarray(z[f"A2B_{i}_cols"]) for i in range(n)),
        vals=tuple(jnp.asarray(z[f"A2B_{i}_vals"]) for i in range(n)),
        lps=tuple((int(L), int(p)) for L, p in m["lps"]), n_B=tuple(int(v) for v in m["n_B"]),
        elements=tuple(int(e) for e in z["elements"]), rcut=float(m["rcut"]), lmax=int(m["lmax"]), solid=bool(m["solid"]),
        gamma=tuple(tuple(float(g) for g in z[f"gamma_{i}"]) for i in range(n)) if "gamma_0" in z or n == 0 else None)


def _pack(coef):
    keys = sorted(coef)
    cs = [np.asarray(coef[k], np.float64) for k in keys]
    return (np.asarray(keys, np.int64).reshape(len(keys), -1), np.concatenate([[0], np.cumsum([c.shape[1] for c in cs])]).astype(np.int64),
            np.concatenate(cs, axis=1) if cs else np.zeros((0, 0)))


def _unpack(z, name):
    keys, ptr, c = z[f"{name}_keys"], z[f"{name}_ptr"], z[f"{name}_coef"]
    return {tuple(int(v) for v in k): np.array(c[:, a:b]) for k, a, b in zip(keys, ptr[:-1], ptr[1:])}


def save(path, onsite=None, offsite=None, info=None):
    """Write an on-site and / or an off-site model (of one site basis, one orbital basis and the
    same channels) to `path` (.npz).  info: a JSON-serialisable dictionary kept in the meta."""
    models = [m for m in (onsite, offsite) if m is not None]
    if not models:
        raise ValueError("operators.io.save: give an on-site model, an off-site model or both")
    sb = onsite.basis if onsite is not None else offsite.bond.site
    if len(models) == 2 and not (_same(sb, offsite.bond.site) and onsite.orbitals == offsite.orbitals and onsite.channels == offsite.channels):
        raise ValueError("operators.io.save: the two models differ in site basis, orbitals or channels")
    out = _site_arrays(sb)
    out["elements"] = np.asarray(sb.elements, np.int64)
    meta = dict(schema_version=SCHEMA_VERSION, kind="operator", elements=[int(z) for z in sb.elements], channels=list(models[0].channels),
                orbitals={str(z): list(ls) for z, ls in models[0].orbitals.shells}, site=_site_meta(sb), bond=None, info=info or {})
    if offsite is not None:
        b = offsite.bond
        meta["bond"] = {k: (getattr(b, k) if k != "env_emb" or b.env_emb is None else [list(r) for r in b.env_emb]) for k in _BOND}
    for name, m in (("onsite", onsite), ("offsite", offsite)):
        if m is not None:
            out[f"{name}_keys"], out[f"{name}_ptr"], out[f"{name}_coef"] = _pack(m.coef)
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(path, **out)


def load(path):
    """`OperatorFile(onsite, offsite, meta)` of a file `save` wrote."""
    z = np.load(path)
    if "meta_json" not in z:
        raise ValueError(f"{path}: not an ace-jax model file (no meta_json)")
    meta = json.loads(bytes(z["meta_json"]).decode())
    if meta.get("kind") != "operator":
        raise ValueError(f"{path}: not an operator model (kind {meta.get('kind')!r}); potentials load with ace_jax.load")
    if meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {meta['schema_version']}")
    sb = _site(z, meta["site"])
    orb, ch = OrbitalBasis.of({int(k): v for k, v in meta["orbitals"].items()}), tuple(meta["channels"])
    on = OnsiteModel(basis=sb, coef=_unpack(z, "onsite"), orbitals=orb, channels=ch) if "onsite_keys" in z else None
    off = None
    if "offsite_keys" in z:
        b = dict(meta["bond"])
        b["env_emb"] = None if b["env_emb"] is None else tuple(tuple(float(v) for v in r) for r in b["env_emb"])
        off = OffsiteModel(bond=B.BondBasis(site=sb, **b), coef=_unpack(z, "offsite"), orbitals=orb, channels=ch)
    return OperatorFile(on, off, meta)
