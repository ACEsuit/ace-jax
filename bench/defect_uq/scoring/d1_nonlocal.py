"""D1 of docs/dev/specs/2026-10-05-gp-discrepancy-design.md: does the served force-UQ score depend on the
environment beyond r_cut, inside a Mondrian group, in a way the r_cut-local feature does not explain?

Per atom i, the smoothed non-local feature g_i = sum_j f_c(r_ij; R) u_j / sum_j f_c(r_ij; R) (j = i included,
f_c(r; R) = (1 - (r/R)^2)^2), with u_j one of
  z: soft coordination - z*  (sum_k sigmoid((r1 - r_jk)/w), w = 0.05 A; r1, z* from the posterior),
  d: soft first-shell distortion (std/mean of r_jk under the same soft weights),
  s: the GP summary s_j (fit/summary.py: power-mean nearest-neighbour distance, r0 = 2.5, p = 10, r_cut),
for R in {0 (u_i alone), r_cut, 8, 10, 12 = MACE-MH-1 receptive field (2 interactions x r_max 6.0 A)}.

Score: the conformal ratio t_i = s_i / q_g (covered iff t_i <= 1), s_i the served aniso Mahalanobis score.
  T_val: the posterior's own calibration scores (cal_scores; P_fit errors, transfer factor applied);
  big cells: <run>/big3*_err.npz against the served posterior (as validate_shape._hit).
Within each group: Spearman rho(t, g_R); coverage across g_R quintiles (spread = max - min); and the part
not explained by the R = r_cut feature: partial Spearman rho(t, g_R | g_rcut) (rank residuals), and the
conditional split -- within g_rcut quintiles, coverage of the upper minus the lower half of g_R, pooled.
90 % intervals from a bootstrap over cells (T_val: configurations; big: (file, cfg // 3)).

    uv run python d1_nonlocal.py --post <run>/posterior.npz --train <bench365>/train.xyz \
        --big-run <rev2 run dir> --big-xyz <big3 dir> --out d1.md [--cache feats.npz]
"""
import argparse
import json
import os

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.stats import rankdata, spearmanr

RCUT, R0, P = 6.25, 2.5, 10         # the Cantor fit's r_cut, GP r0 (fit_bench.py), summary power
RF = 12.0                           # MACE-MH-1: num_interactions 2 x r_max 6.0 (read from the torch model)
RS = (0.0, RCUT, 8.0, 10.0, RF)
W_SOFT = 0.05
FEATS = ("z", "d", "s")
TIP = 10.0
LIMIT = int(os.environ.get("D1_LIMIT", "0"))      # smoke runs: configs per set (0 = all)
BIG_FILES = {"big3_err.npz": "big3_mh1.xyz", **{f"big3x_r{p}_err.npz": f"big3_cracks_r{p}.xyz"
                                                 for p in ("2-3", "4-5", "6-7", "8-9")}}


def fc(r, R):
    return np.where(r < R, (1.0 - (r / R) ** 2) ** 2, 0.0)


def features(atoms, r1, z_star):
    """g (len(FEATS), len(RS), n) and the hard shell features (z, d) of conformal.shell_features."""
    n = len(atoms)
    i, j, r = neighbor_list("ijd", atoms, RF)
    sw = 1.0 / (1.0 + np.exp((r - r1) / W_SOFT))
    zt = np.bincount(i, sw, n)
    mean = np.bincount(i, sw * r, n) / np.maximum(zt, 1e-12)
    var = np.bincount(i, sw * (r - mean[i]) ** 2, n) / np.maximum(zt, 1e-12)
    dt = np.sqrt(var) / np.maximum(mean, 1e-12)
    S = np.bincount(i, fc(r, RCUT) * (R0 / r) ** P, n) + 1e-30
    s = R0 * S ** (-1.0 / P)
    hard = r < r1
    zh = np.bincount(i, hard, n).astype(int)
    mh = np.bincount(i, hard * r, n) / np.maximum(zh, 1)
    vh = np.bincount(i, hard * (r - mh[i]) ** 2, n) / np.maximum(zh, 1)
    dh = np.where(zh >= 2, np.sqrt(vh) / np.maximum(mh, 1e-12), np.nan)
    U = np.stack([zt - z_star, dt, s])                                    # (3, n)
    g = np.empty((len(FEATS), len(RS), n))
    for k, R in enumerate(RS):
        if R == 0:
            g[:, k] = U
            continue
        w = fc(r, R)
        den = 1.0 + np.bincount(i, w, n)
        for f in range(len(FEATS)):
            g[f, k] = (U[f] + np.bincount(i, w * U[f][j], n)) / den
    return g, zh, dh


def assign_groups(z, d, z_star, edges):
    band = np.searchsorted(np.asarray(edges, float), np.nan_to_num(d, nan=np.inf), side="right")
    return (band * 2 + (z != z_star)).astype(np.int64)


def load_post(path):
    z = np.load(path, mmap_mode="r")
    js = lambda k: json.loads(bytes(z[k]).decode())
    gc, gt = js("group_consts_json"), js("group_table_json")
    q = np.array([np.inf if x == "inf" else x for x in gt["q"]], float)
    lam = np.asarray(gt["lam_rms"], float)
    return dict(r1=gc["r1"], z_star=gc["z_star"], edges=np.asarray(gc["edges"], float), q=q, lam=lam,
                eps=float(z["eps"]), shape=str(z["force_shape"]), cal_s=np.asarray(z["cal_scores"], float),
                cal_g=np.asarray(z["cal_groups"], int), cal_cfg=np.asarray(z["cal_cfg"], int))


def tval_table(post, train_xyz, log=print):
    frames = read(train_xyz, ":")
    cfgs, starts = np.unique(post["cal_cfg"], return_index=True)
    order = np.argsort(starts)
    out = {k: [] for k in ("g", "t", "grp", "cell")}
    n_bad = n_done = 0
    for c in (cfgs[order][:LIMIT] if LIMIT else cfgs[order]):
        m = np.flatnonzero(post["cal_cfg"] == c)
        a = frames[c]
        if len(m) != len(a):
            n_bad += 1
            continue
        g, zh, dh = features(a, post["r1"], post["z_star"])
        grp = assign_groups(zh, dh, post["z_star"], post["edges"])
        if not np.array_equal(grp, post["cal_g"][m]):
            n_bad += 1
            continue
        out["g"].append(g); out["grp"].append(grp)
        out["t"].append(post["cal_s"][m] / post["q"][grp])
        out["cell"].append(np.full(len(a), c))
        n_done += 1
    log(f"T_val: {n_done}/{n_done + n_bad} configs aligned (atom count and recomputed groups match)")
    return {"g": np.concatenate(out["g"], -1), "t": np.concatenate(out["t"]), "grp": np.concatenate(out["grp"]),
            "cell": np.concatenate(out["cell"]), "fam": np.full(sum(len(x) for x in out["t"]), "tval")}


def big_table(post, run, xyz_dir, log=print):
    out = {k: [] for k in ("g", "t", "grp", "cell", "fam", "rc", "edge")}
    for fi, (ef, xf) in enumerate(BIG_FILES.items()):
        if not os.path.exists(f"{run}/{ef}"):
            continue
        A = np.load(f"{run}/{ef}", allow_pickle=True)
        frames = read(f"{xyz_dir}/{xf}", ":")
        grp = A["forces_group"].astype(int)
        lam = post["lam"][grp]
        V = A["forces_cov"].astype(float) / lam[:, None, None] ** 2
        M = V + post["eps"] * (np.trace(V, axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)
        e = A["dF"].astype(float)
        s = np.sqrt(np.einsum("na,na->n", e, np.linalg.solve(M, e[:, :, None])[:, :, 0]))
        t = s / post["q"][grp]
        cfg = A["cfg"]
        for c in (np.unique(cfg)[:LIMIT] if LIMIT else np.unique(cfg)):
            m = np.flatnonzero(cfg == c)
            a = frames[c]
            assert len(a) == len(m), (ef, c)
            g, zh, dh = features(a, post["r1"], post["z_star"])
            if not np.array_equal(assign_groups(zh, dh, post["z_star"], post["edges"]), grp[m]):
                log(f"WARNING {ef} cfg {c}: recomputed groups differ from forces_group")
            xy = a.positions[:, :2]
            rho = np.linalg.norm(xy - 0.5 * (xy.max(0) + xy.min(0)), axis=1)
            free = ~A["fixed"][m].astype(bool)
            out["g"].append(g[..., free]); out["t"].append(t[m][free]); out["grp"].append(grp[m][free])
            out["cell"].append(np.full(free.sum(), fi * 10 ** 6 + c // 3))
            out["fam"].append(A["family"][m][free].astype(str)); out["rc"].append(A["r_core"][m][free])
            out["edge"].append((rho.max() - rho)[free])
        log(f"big: {ef} ({len(np.unique(cfg))} configs)")
    return {k: np.concatenate(v, -1) for k, v in out.items()}


def _qbins(x, k=5):
    e = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(e, x, side="right")


def _partial_rho(t, g, g0):
    rt, rg, r0 = (rankdata(v) for v in (t, g, g0))
    X = np.c_[np.ones_like(r0), r0]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = res(rt), res(rg)
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-300))


def _cond_split(hit, g, g0, k=5):
    """Within g0 quintiles, coverage of g >= the bin's median minus coverage below it, atom-weighted."""
    b0 = _qbins(g0, k)
    num = den = 0.0
    for b in range(k):
        m = b0 == b
        if m.sum() < 20:
            continue
        hi = g[m] >= np.median(g[m])
        if hi.all() or (~hi).all():
            continue
        num += m.sum() * (hit[m][hi].mean() - hit[m][~hi].mean())
        den += m.sum()
    return num / den if den else np.nan


def stats(t, g, g0, cell, B=200, seed=0):
    """Point estimates and 90 % cell-bootstrap intervals of (rho, spread, partial rho, conditional split)."""
    hit = t <= 1.0

    def est(idx):
        tt, gg, g00, hh = t[idx], g[idx], g0[idx], hit[idx]
        qb = _qbins(gg)
        cov = np.array([hh[qb == b].mean() for b in range(5) if (qb == b).any()])
        return np.array([spearmanr(tt, gg)[0], cov.max() - cov.min(), _partial_rho(tt, gg, g00),
                         _cond_split(hh, gg, g00)])

    pt = est(np.arange(len(t)))
    u, inv = np.unique(cell, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    bs = np.array([est(np.concatenate([members[c] for c in rng.integers(0, len(u), len(u))])) for _ in range(B)])
    lo, hi = np.nanpercentile(bs, [5, 95], axis=0)
    return pt, lo, hi, float(hit.mean())


def populations(T, Bg):
    pops = [("T_val", T, np.ones(len(T["t"]), bool))]
    if Bg is not None:
        inner = Bg["edge"] >= RF
        pops += [("big, all free (>= 12 A from vacuum edge)", Bg, inner)]
        for f in ("crack", "edge", "screw"):
            pops.append((f"{f}", Bg, inner & (Bg["fam"] == f)))
        pops.append(("crack tip (r_core <= 10 A)", Bg, (Bg["fam"] == "crack") & (Bg["rc"] <= TIP)))
    return pops


def report(T, Bg, n_min=300, B=200, log=print):
    k0 = RS.index(RCUT)
    lines, gate = [], []
    for name, D, mask in populations(T, Bg):
        if mask.sum() < n_min:
            continue
        lines += [f"\n### {name}\n", "| group | n | cov | feature | R | rho [90% CI] | bin spread [CI] | "
                  "partial rho given g(r_cut) [CI] | cond. split [CI] |", "|---|---|---|---|---|---|---|---|---|"]
        for gr in np.unique(D["grp"][mask]):
            m = mask & (D["grp"] == gr)
            if m.sum() < n_min:
                continue
            for fi, f in enumerate(FEATS):
                for k, R in enumerate(RS):
                    pt, lo, hi, cov = stats(D["t"][m], D["g"][fi, k, m], D["g"][fi, k0, m], D["cell"][m], B)
                    nl = R > RCUT
                    lines.append(f"| {gr} | {m.sum()} | {cov:.3f} | {f} | {R:g} | {pt[0]:+.3f} [{lo[0]:+.2f}, {hi[0]:+.2f}] | "
                                 f"{pt[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}] | "
                                 + (f"{pt[2]:+.3f} [{lo[2]:+.2f}, {hi[2]:+.2f}] | {pt[3]:+.3f} [{lo[3]:+.3f}, {hi[3]:+.3f}] |"
                                    if nl else "– | – |"))
                    if nl:
                        gate.append(dict(pop=name, group=int(gr), n=int(m.sum()), feat=f, R=R, rho=pt[0],
                                         prho=pt[2], prho_ci=(lo[2], hi[2]), split=pt[3], split_ci=(lo[3], hi[3]),
                                         spread=pt[1]))
        # pooled over groups (the score is already group-normalised): the headline line per population
        lines.append("")
        lines.append(f"pooled over groups ({mask.sum()} atoms):")
        for fi, f in enumerate(FEATS):
            for k, R in enumerate(RS):
                if R <= RCUT:
                    continue
                pt, lo, hi, cov = stats(D["t"][mask], D["g"][fi, k, mask], D["g"][fi, k0, mask], D["cell"][mask], B)
                lines.append(f"- {f}, R = {R:g}: partial rho {pt[2]:+.3f} [{lo[2]:+.2f}, {hi[2]:+.2f}], cond. split "
                             f"{pt[3]:+.3f} [{lo[3]:+.3f}, {hi[3]:+.3f}] (raw rho {pt[0]:+.3f}, raw bin spread {pt[1]:.3f})")
        log(f"done {name}")
    return lines, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--big-run"); ap.add_argument("--big-xyz")
    ap.add_argument("--out", required=True); ap.add_argument("--cache")
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--n-min", type=int, default=300)
    a = ap.parse_args()
    post = load_post(a.post)
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        T = {k[2:]: z[k] for k in z.files if k.startswith("T_")}
        Bg = {k[2:]: z[k] for k in z.files if k.startswith("B_")} or None
    else:
        T = tval_table(post, a.train)
        Bg = big_table(post, a.big_run, a.big_xyz) if a.big_run else None
        if a.cache:
            np.savez(a.cache, **{f"T_{k}": v for k, v in T.items()},
                     **({f"B_{k}": v for k, v in Bg.items()} if Bg else {}))
    if Bg is not None:      # sanity: the rev2 table's coverage, all free atoms
        tip = (Bg["fam"] == "crack") & (Bg["rc"] <= TIP)
        print(f"check: big all-free cov {np.mean(Bg['t'] <= 1):.3f} (atoms), tip {np.mean(Bg['t'][tip] <= 1):.3f}; "
              f"T_val cov {np.mean(T['t'] <= 1):.3f}")
    lines, gate = report(T, Bg, a.n_min, a.boot)
    head = ["# D1 non-local dependence of the force-UQ score\n", f"posterior `{a.post}`; big cells `{a.big_run}`",
            f"R grid {RS} (r_cut {RCUT}, teacher receptive field {RF}); features {FEATS}; groups with >= {a.n_min} atoms; "
            f"{a.boot} cell-bootstrap resamples.\n"]
    with open(a.out, "w") as f:
        f.write("\n".join(head + lines) + "\n")
    with open(os.path.splitext(a.out)[0] + "_gate.json", "w") as f:
        json.dump(gate, f, indent=1, default=float)


if __name__ == "__main__":
    main()
