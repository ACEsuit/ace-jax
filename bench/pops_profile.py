"""Where does POPS time go?  Runs the real fitting pipeline (linear arm, --uq pops)
with a timer around every POPS stage and prints a breakdown.

    python bench/pops_profile.py --model M.npz --data D.xyz --ntrain 800 --ntest 200 \
        --energy-key .. --force-key .. --virial-key .. [--grid 1e-4,1e-6,...]

Timers block on the stage's outputs, so each figure is device time, compile
included (the first call of a stage pays its compile).
"""
import argparse
import collections
import time

import jax

jax.config.update("jax_enable_x64", True)

import ace_jax.fit.pipeline.predict as PP      # noqa: E402
import ace_jax.fit.pops as POPS                # noqa: E402
import ace_jax.fit.predict as P                # noqa: E402
from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data   # noqa: E402

T, N = collections.defaultdict(float), collections.Counter()


def timed(name, f):
    def g(*a, **k):
        t = time.time()
        out = f(*a, **k)
        try:
            jax.block_until_ready(out)
        except Exception:
            pass
        T[name] += time.time() - t; N[name] += 1
        return out
    return g


def instrument():
    for name in ("pops_leverage_residual", "pops_moment_sums", "pops_projection_bounds",
                 "pops_envelope_streamed", "_train_stats", "select_pops_ridge", "predict_fixed"):
        setattr(P, name, timed(name, getattr(P, name)))
    PP.select_pops_ridge = P.select_pops_ridge      # the pipeline imported these by name
    PP.predict_fixed = P.predict_fixed
    POPS.hypercube_support = timed("hypercube_support (eigh A W A)", POPS.hypercube_support)
    POPS.hypercube_cov = timed("hypercube_cov", POPS.hypercube_cov)
    init = P.PopsRidgePath.__init__
    P.PopsRidgePath.__init__ = timed("PopsRidgePath.__init__ (stats + eigh)", init)
    P.PopsRidgePath.A = timed("PopsRidgePath.A (L^3)", P.PopsRidgePath.A)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--model", "--data", "--energy-key", "--force-key", "--virial-key"):
        ap.add_argument(k, required=True)
    ap.add_argument("--ntrain", type=int, default=800); ap.add_argument("--ntest", type=int, default=200)
    ap.add_argument("--grid", default=None, help="comma-separated ridge grid (default: FitConfig's)")
    ap.add_argument("--r0", type=float, default=2.5)
    a = ap.parse_args()
    kw = {} if a.grid is None else {"pops_ridge_grid": tuple(float(x) for x in a.grid.split(","))}
    cfg = FitConfig(model=a.model, arm="linear", uq="pops", pops_ridge="auto", rungs=("map",), map_steps=40,
                    energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key, r0=a.r0,
                    ntrain=a.ntrain, ntest=a.ntest, test_start=a.ntrain, predict_train=False,
                    **kw).validate()
    instrument()
    t0 = time.time()
    data = load_fit_data(cfg, data=a.data)
    res = fit(cfg, data, log=lambda *s: print(*s, flush=True))
    total = time.time() - t0
    print(f"\nlen_basis {res.built.prob.cfg.len_basis}  ntrain {a.ntrain}  ntest {a.ntest}  "
          f"grid {len(cfg.pops_ridge_grid)}  total {total:.0f}s")
    print("pipeline stages:", {k: round(v, 1) for k, v in res.timings.items()})
    print(f"{'stage':44s} {'calls':>5s} {'secs':>9s} {'s/call':>8s}")
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"{k:44s} {N[k]:5d} {v:9.1f} {v / N[k]:8.2f}")


if __name__ == "__main__":
    main()
