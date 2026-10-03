"""Per-atom error + served-array core shared by the calibrate driver (same npz schema as
modal_bench365.big_errors, whose deployed copy is left untouched)."""
import os
import sys
import time


def concat_xyz(paths, out):
    """Concatenate extxyz files (frame order = path order) into `out`; returns the frame count."""
    n = 0
    with open(out, "wb") as fo:
        for p in paths:
            with open(p, "rb") as fi:
                b = fi.read()
            fo.write(b if b.endswith(b"\n") else b + b"\n")
    from ase.io import read
    n = len(read(out, ":"))
    return n


def err_npz(model: str, post: str, xyz: str, out_npz: str) -> str:
    """big_errors logic with explicit model/posterior/out paths."""
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from ase.io import read
    from ace_jax import ACECalculator
    calc = ACECalculator(model, posterior=post) if os.path.exists(post) else ACECalculator(model)
    sys.path.insert(0, "/root")
    import served_arrays
    props = ["forces_std"]
    if os.path.exists(post) and "group_table_json" in np.load(post).files:      # schema 3
        props += ["forces_q", "forces_group", "forces_cov"]
    err, dfv, srv, fam, rc, fx, cid, t0 = [], [], [], [], [], [], [], time.time()
    for k, a in enumerate(read(xyz, ":")):
        fam.append(np.full(len(a), a.info.get("family", "?")))
        rc.append(a.arrays.get("r_core", np.full(len(a), np.nan)))
        fx.append(a.arrays.get("fixed", np.zeros(len(a), bool)).astype(bool))
        cid.append(np.full(len(a), k))
        a.calc = calc
        d = a.get_forces() - a.arrays["mace_force"]
        dfv.append(d.astype(np.float32))
        err.append(np.linalg.norm(d, axis=1))
        if os.path.exists(post):
            srv.append(served_arrays.collect(calc, [a], props))
    out = dict(err=np.concatenate(err), dF=np.concatenate(dfv), family=np.concatenate(fam),
               r_core=np.concatenate(rc), fixed=np.concatenate(fx), cfg=np.concatenate(cid))
    if srv:
        out.update({k: np.concatenate([x[k] for x in srv]) for k in srv[0]})
    np.savez(out_npz, **out)
    e = out["err"]
    return (f"{len(e)} atoms, median |dF| {np.median(e):.3f}, 99th {np.percentile(e, 99):.3f} eV/A"
            + (f"; median sigma {np.median(out['sd']):.3f}" if "sd" in out else "") + f"; {time.time() - t0:.0f} s")
