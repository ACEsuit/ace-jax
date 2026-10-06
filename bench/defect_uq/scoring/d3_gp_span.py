"""D3 of docs/dev/specs/2026-10-05-gp-discrepancy-design.md: how much of the residual GP's prior lies outside
its inducing span at crack-tip, dislocation-core and bulk atoms?

No Cantor GP model file was saved (bench365_gp kept theta_map.json and predictions only), so the fit is
rebuilt from what defines its prior: the GP arm of modal/fit_bench.py (PCA-128 density map, 100 inducing
sites per species by FPS -- deterministic), on bench365 train.xyz, at the saved theta_MAP.  q and k - q are
prior quantities (no training statistics needed):
  site level   k(x, x),  q = k_xM K_MM^-1 k_Mx,  k - q            (the energy-row DTC term per site)
  force level  k_F = <o, o>_k,  q_F = K_oM K_MM^-1 K_Mo,  k_F - q_F  (predict._dtc_deriv_residual, summed
               over the 3 components of the target atom's force)
The SoR *posterior* variance (q minus what the data explain) needs the training statistics, a GPU pass; it is
not computed here.

Each target atom is evaluated in a non-periodic cluster of every atom (periodic images included) within
2 r_cut + 0.5 A of it: the target's force depends on the sites within r_cut, whose descriptors depend on the
atoms within r_cut of them, so the target's rows are exact.

    uv run python d3_gp_span.py --data <bench365 dir> --model <cantor_embed_d16_deg10.npz> \
        --theta <bench365_gp>/theta_map.json --big <big3_mh1.xyz> --out d3.md [--ind-cache ind.npz]
"""
import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from jax.scipy.linalg import solve_triangular  # noqa: E402

from ace_jax.fit.data import Config, build_dataset  # noqa: E402
from ace_jax.fit.hypers import Hypers  # noqa: E402
from ace_jax.fit.kernels import K_MM, k_rows  # noqa: E402
from ace_jax.fit.pipeline import FitConfig, load_fit_data  # noqa: E402
from ace_jax.fit.pipeline.problem import build_problem  # noqa: E402
from ace_jax.fit.predict import _dtc_deriv_residual  # noqa: E402
from ace_jax.fit.rows import batch_rows_parts, residual_inputs  # noqa: E402

RCUT = 6.25
NCAP, KCAP = 1024, 128     # fixed padding: one compile for every cluster (a 13 A fcc sphere holds ~750 atoms)


def gp_config(data, model):
    """The GP arm of modal/fit_bench.py (bench365_gp, 2026-09-28)."""
    return FitConfig(model=model, energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial",
                     r0=2.5, batch=4, rungs=("map",), predict_train=False, e0="lsq", opt="lbfgs",
                     arm="gp", m_per_species=100, density="pca", pca_d=128, lml="host-cache", map_restarts=4)


def problem(a, log=print):
    cfg = gp_config(a.data, a.model)
    d = load_fit_data(cfg, train=f"{a.data}/train.xyz", test=f"{a.data}/test.xyz")
    t = time.time()
    built = build_problem(cfg, d)
    log(f"build_problem {time.time() - t:.0f} s: M = {built.prob.ind.XM.shape[0]}, d = {built.prob.ind.XM.shape[1]}")
    if a.ind_cache:
        i = built.prob.ind
        np.savez(a.ind_cache, XM=i.XM, SM=i.SM, ZM=i.ZM, scale=i.scale, Pmap=i.Pmap, embed=i.embed, warp=i.warp)
    return built.prob, d


def cluster(a, i, rc):
    """Non-periodic cluster of atom i and every atom (images included) within rc of it; the target first."""
    I, J, D = neighbor_list("ijD", a, rc)
    m = I == i
    pos = np.vstack([np.zeros(3), D[m]]) + a.positions[i]
    num = np.r_[a.numbers[i], a.numbers[J[m]]]
    c = Atoms(num, pos)
    c.center(vacuum=10.0)
    return c


def target_rows(prob, theta, d, c):
    cf = Config(c.positions, c.numbers, np.asarray(c.cell), np.zeros(3, bool), 0.0, np.zeros((len(c), 3)),
                None, 1.0, 1.0, 0.0)
    ds = build_dataset([cf], d.meta, d.E0, 1, n_cap=NCAP, k_cap=KCAP, pack="off", log=lambda *s: None)
    b = jax.tree.map(lambda x: x[0], ds)
    ind, spec = prob.ind, prob.spec
    lin, res, X, JU0 = batch_rows_parts(theta, spec, prob.model, ind, prob.cfg, b, with_X=True, with_JU0=True)
    Fv, _ = _dtc_deriv_residual(theta, prob, b, X, res=res, JU0=JU0)          # (Ncap, 3), = k_F - q_F, clamped
    L = jnp.linalg.cholesky(K_MM(theta, spec, ind.XM, ind.SM, ind.ZM, ind.embed))
    vF = solve_triangular(L, res.F[0].T, lower=True)                           # target = node 0: (M, 3)
    qF = jnp.sum(vF * vF, 0)
    U, s, _ = residual_inputs(ind, prob.cfg, b, X)
    z = b.node_z
    kxx = k_rows(theta, spec, U[:1], s[:1], z[:1], U[:1], s[:1], z[:1], ind.embed)[0, 0]
    kxM = k_rows(theta, spec, U[:1], s[:1], z[:1], ind.XM, ind.SM, ind.ZM, ind.embed)[0]
    v = solve_triangular(L, kxM, lower=True)
    return dict(k=float(kxx), q=float(v @ v), kF=float(jnp.sum(Fv[0]) + jnp.sum(qF)), qF=float(jnp.sum(qF)),
                dF=float(jnp.sum(Fv[0])), s=float(s[0]))


def targets(big, test, n, seed=0):
    rng = np.random.default_rng(seed)
    frames = read(big, ":")
    out = []
    pick = lambda idx, k: rng.choice(idx, min(k, len(idx)), replace=False) if len(idx) else []
    for fam, bands in (("crack", (("tip r<5", 0, 5), ("crack 10-20", 10, 20))),
                       ("edge", (("edge core r<5", 0, 5),)), ("screw", (("screw core r<5", 0, 5),)),
                       ("crack", (("big-cell bulk r 18-26", 18, 26),))):
        fr = [(k, a) for k, a in enumerate(frames) if a.info.get("family") == fam and float(a.info.get("rattle", 0)) == 0]
        for lab, lo, hi in bands:
            for _, a in fr:
                rc = a.arrays["r_core"]
                idx = np.flatnonzero(~a.arrays["fixed"].astype(bool) & (rc >= lo) & (rc < hi))
                out += [(lab, a, int(i)) for i in pick(idx, max(1, n // len(fr)))]
    T = read(test, ":")
    for k in rng.choice(len(T), n, replace=False):
        out.append(("bench365 test (ID)", T[k], int(rng.integers(len(T[k])))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--theta", required=True); ap.add_argument("--big", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ind-cache")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    th = json.load(open(a.theta))
    theta = Hypers(*[jnp.asarray(th[f], jnp.float64) for f in Hypers._fields])
    prob, d = problem(a)
    rows = []
    tg = targets(a.big, f"{a.data}/test.xyz", a.n)
    if int(os.environ.get("D3_LIMIT", "0")):
        tg = tg[:int(os.environ["D3_LIMIT"])]
    for lab, at, i in tg:
        t = time.time()
        r = target_rows(prob, theta, d, cluster(at, i, 2 * RCUT + 0.5))
        rows.append(dict(label=lab, **r))
        print(f"{lab:24s} k {r['k']:.3e} q/k {r['q'] / r['k']:.3f}  kF {r['kF']:.3e} (k-q)F/kF {r['dF'] / max(r['kF'], 1e-300):.3f}"
              f"  s {r['s']:.3f}  ({time.time() - t:.0f} s)", flush=True)
    json.dump(rows, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    labs = list(dict.fromkeys(r["label"] for r in rows))
    lines = ["# D3 residual-GP prior inside / outside the inducing span\n",
             f"theta `{a.theta}`; inducing set rebuilt (fit_bench.py GP arm); clusters 2 r_cut + 0.5 A.\n",
             "| atoms | n | median k (eV^2) | median (k-q)/k | median k_F (eV^2/A^2) | median (k_F-q_F)/k_F [p10, p90] | median s (A) |",
             "|---|---|---|---|---|---|---|"]
    for lab in labs:
        R = [r for r in rows if r["label"] == lab]
        k = np.array([r["k"] for r in R]); q = np.array([r["q"] for r in R])
        kF = np.array([r["kF"] for r in R]); dF = np.array([r["dF"] for r in R])
        sh = dF / np.maximum(kF, 1e-300)
        lines.append(f"| {lab} | {len(R)} | {np.median(k):.3e} | {np.median((k - q) / k):.3f} | {np.median(kF):.3e} | "
                     f"{np.median(sh):.3f} [{np.quantile(sh, 0.1):.3f}, {np.quantile(sh, 0.9):.3f}] | "
                     f"{np.median([r['s'] for r in R]):.3f} |")
    open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
