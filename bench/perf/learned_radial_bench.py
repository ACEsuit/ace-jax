"""Learned (analytic) tensor radials vs splined ones: the dense E/F/V model call
of a benchmark ACE model in its stock spline form, converted to the analytic
branch at several polynomial spans n_q (`to_analytic`) with Wnlq perturbed on the
active rows to mimic learning, and the lean form of each, interleaved round by
round on one device (docs/dev/learned-radial-splining.md).

    PYTHONPATH=bench:src python bench/perf/learned_radial_bench.py <system>_<size> <n> \
        [--nqs 8,12,16,20] [--filled-nq 12] [--rounds 3] [--reps 10]

Cases, per n_q:
  spline              the stock (Julia-splined) model
  spline:lean         lean(spline): prune + pairfold + species-compact l-blocks
  aN                  analytic, n_q = N, Wnlq perturbed on its active rows
  aN:lean_exact       lean(aN, spline_tol=None): exact, analytic R_nl kept, so
                      the l-blocks are not species-compact
  aN:lean             lean(aN): to_spline first (--spline-tol, default 1e-10), then the spline lean
  aN_filled[...]      aN with every z_j block of Wnlq filled (no zero pattern):
                      compaction cannot apply
  X:oldgather         X evaluated with the spline radial's previous formula,
                      vmap(spline_eval)(x, coefs[zi, zj]) (before
                      `spline_eval_pairs`), to A/B the gather on one device

Prints one JSON object: per case the median over rounds of the per-round median
efv time, and parity of each lean case against its own full model.
"""
import argparse
import inspect
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS = os.path.join(ROOT, "bench", "scaling", "models")


def perturb(model, rel=0.2, seed=0, fill=False):
    """Wnlq + rel * (row rms) * N(0, 1) on the active (not identically zero)
    rows, so the zero pattern -- which radial columns belong to which z_j -- is
    kept, as learning keeps it (radial_model.normalise freezes inactive rows).
    fill=True also fills the inactive rows (scale: the mean active row rms), a
    radial with no per-species pattern, e.g. a glorot-initialised one."""
    import dataclasses

    import jax.numpy as jnp
    import numpy as np
    W = np.asarray(model.rnl_Wnlq, np.float64)
    act = np.abs(W).max(-1) > 0
    rms = np.sqrt((W ** 2).mean(-1, keepdims=True))
    noise = np.random.default_rng(seed).normal(size=W.shape)
    W2 = W + rel * rms * noise * act[..., None]
    if fill:
        W2 = np.where(act[..., None], W2, rel * rms[act].mean() * noise)
    return dataclasses.replace(model, rnl_Wnlq=jnp.asarray(W2, model.rnl_Wnlq.dtype), radial_learned=True)


def old_gather(model):
    """`model` evaluated with the spline radial's previous formula: vmap over
    edges of spline_eval on coefs[zi, zj] (same values, bit for bit)."""
    import dataclasses

    import jax

    from ace_jax.eval.model import ACEModel
    from ace_jax.eval.radial import agnesi_normalized, spline_eval

    class OldGather(ACEModel):
        def _radial_one(self, r, zi, zj, kind, trans, coefs, grid, Wnlq, ABC, env):
            if kind != "spline":
                return super()._radial_one(r, zi, zj, kind, trans, coefs, grid, Wnlq, ABC, env)
            x = agnesi_normalized(r, trans[zi, zj])
            x0, h, n = grid
            val = jax.vmap(lambda xx, c: spline_eval(xx, c, x0, h, n))(x, coefs[zi, zj])
            return val * env[:, None]

    return OldGather(**{f.name: getattr(model, f.name) for f in dataclasses.fields(model)})


def parity(ref, m, x):
    import jax
    import numpy as np
    f = jax.jit(lambda mm: mm.energy_forces_virial_dense(*x))
    (E0, F0, V0), (E1, F1, V1) = f(ref), f(m)
    return {"dE_rel": float(abs(E1 - E0) / max(1.0, abs(float(E0)))),
            "dF": float(np.abs(np.asarray(F1 - F0)).max()),
            "dV": float(np.abs(np.asarray(V1 - V0)).max()),
            "F_max": float(np.abs(np.asarray(F0)).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="e.g. Cantor_medium")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--nqs", default="8,12,16,20")
    ap.add_argument("--filled-nq", type=int, default=12, help="0: no filled variant")
    ap.add_argument("--spline-tol", type=float, default=1e-10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--oldgather", action="store_true", help="also the pre-spline_eval_pairs gather")
    a = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", a.dtype == "float64")
    import jax.numpy as jnp
    sys.path.insert(0, HERE)
    from ace_gap_bench import dense_args, timed

    from ace_jax.eval.io import load
    from ace_jax.eval.model import highest_precision, lean
    from ace_jax.fit.radial_model import to_analytic
    dt = getattr(jnp, a.dtype)
    has_to_spline = "spline_tol" in inspect.signature(lean).parameters
    lean_exact = (lambda m: lean(m, spline_tol=None)) if has_to_spline else lean
    system = a.model.split("_")[0]
    # the conversions run in float64 on the host; the timed models are cast after
    ace, meta, _ = load(os.path.join(MODELS, f"ace_{a.model}.npz"))
    at, x = dense_args(meta, system, a.n, dt)
    cast = lambda m: jax.tree.map(lambda v: v.astype(dt) if hasattr(v, "dtype")
                                  and jnp.issubdtype(v.dtype, jnp.floating) else v, m)
    cases = {"spline": ace, "spline:lean": lean(ace)}
    out = {"model": a.model, "n": a.n, "dtype": a.dtype, "K": int(x[4].shape[1]),
           "device": str(jax.devices()[0].device_kind), "jax": jax.__version__,
           "has_to_spline": has_to_spline, "spline_tol": a.spline_tol,
           "nqs": [int(q) for q in a.nqs.split(",") if q], "info": {}, "parity": {}, "cases": {}}
    variants = [(f"a{q}", int(q), False) for q in a.nqs.split(",") if q]
    if a.filled_nq:
        variants.append((f"a{a.filled_nq}_filled", a.filled_nq, True))
    for name, q, fill in variants:
        m, rel = to_analytic(ace, q)
        m = perturb(m, fill=fill)
        cases[name] = m
        cases[f"{name}:lean_exact"] = lean_exact(m)
        info = {"to_analytic_relres_max": float(rel.max())}
        if has_to_spline:
            from ace_jax.fit.radial_model import to_spline
            t0 = time.perf_counter()
            ms, err = to_spline(m, tol=a.spline_tol)
            info.update(to_spline_s=time.perf_counter() - t0, to_spline_err=float(err),
                        n_intervals=int(ms.rnl_grid[2] - 1))
            ml = lean(m, spline_tol=a.spline_tol)
            info["lean_compact"] = bool(ml.blk_compact)
            cases[f"{name}:lean"] = ml
        info["lean_exact_compact"] = bool(cases[f"{name}:lean_exact"].blk_compact)
        out["info"][name] = info
    out["info"]["spline"] = {"lean_compact": bool(cases["spline:lean"].blk_compact)}
    if a.oldgather:
        for k in [k for k in ("spline", "spline:lean", f"a{a.filled_nq or 12}:lean") if k in cases]:
            cases[f"{k}:oldgather"] = old_gather(cases[k])
    cases = {k: cast(v) for k, v in cases.items()}
    efv = jax.jit(lambda mm, *xx: mm.energy_forces_virial_dense(*xx))
    with highest_precision():
        for k, m in cases.items():
            if ":" in k:
                out["parity"][k] = parity(cases[k.split(":")[0]], m, x)
                if k.endswith(":oldgather"):
                    out["parity"][k] = parity(cases[k.rsplit(":", 1)[0]], m, x)
            t0 = time.perf_counter()
            jax.block_until_ready(efv(m, *x))
            out["cases"][k] = {"compile_s": time.perf_counter() - t0, "rounds": []}
        for rnd in range(a.rounds):
            order = list(cases) if rnd % 2 == 0 else list(cases)[::-1]
            for k in order:
                m = cases[k]
                out["cases"][k]["rounds"].append(timed(lambda: efv(m, *x), a.reps))  # noqa: B023
    ref = statistics.median(out["cases"]["spline"]["rounds"])
    for c in out["cases"].values():
        c["efv_s"] = statistics.median(c["rounds"])
        c["atom_steps_per_s"] = a.n / c["efv_s"]
        c["vs_spline"] = c["efv_s"] / ref                 # cost relative to the stock spline
    print(json.dumps(out))


if __name__ == "__main__":
    main()
