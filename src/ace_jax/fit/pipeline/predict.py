import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..data import build_dataset
from ..metrics import summarise
from ..predict import POPS_MEAN, PopsRidgePath, predict_fixed, predict_mixture, select_pops_ridge

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


class Preds(NamedTuple):
    arrays: dict; metrics: dict; pops: dict; timings: dict


def label_metrics(E, Em, Ev, F, Fm, Fv, V, Vm, Vv, nat):
    """Summaries on the observed labels only: NaN rows (no label) are dropped."""
    e = np.isfinite(E); f = np.isfinite(F).all(1); v = np.isfinite(V).all(1)
    return {"E": summarise(1e3 * E[e] / nat[e], 1e3 * Em[e] / nat[e], 1e3 * np.sqrt(Ev[e]) / nat[e]),
            "F": summarise(F[f].reshape(-1), Fm[f].reshape(-1), np.sqrt(Fv[f]).reshape(-1)),
            "V": summarise(V[v].reshape(-1), Vm[v].reshape(-1), np.sqrt(Vv[v]).reshape(-1))}


def _labels(cfgs):
    E = np.array([np.nan if c.energy is None else c.energy for c in cfgs])
    F = np.concatenate([np.full((len(c.numbers), 3), np.nan) if c.forces is None else c.forces for c in cfgs])
    V = np.array([np.full(6, np.nan) if c.virial is None else [c.virial[i, j] for i, j in VOIGT]
                  for c in cfgs])
    return E, F, V


def _pops_setup(cfg, d, b, stats, theta, log):
    """Ridge (auto: selected on a train hold-out), the BLR-mean path, and the test
    envelope -- run.py's POPS block verbatim in behaviour."""
    from ..rows import linear_rows
    prob, out = b.prob, {}
    if cfg.pops_ridge == "auto":
        nval = max(1, int(cfg.pops_val_frac * len(d.train)))
        ds_fit = build_dataset(d.train[:-nval], d.meta, d.E0, cfg.batch)
        ds_val = build_dataset(d.train[-nval:], d.meta, d.E0, cfg.batch)
        ridge, scores = select_pops_ridge(theta, prob, ds_fit, ds_val, list(cfg.pops_ridge_grid),
                                          form=cfg.pops_posterior, leverage_pct=cfg.pops_leverage_pct)
        out["ridge_scores"] = {"grid": list(cfg.pops_ridge_grid), "ridge": ridge, "n_val": nval,
                               "scores_crps": {q: [float(x) for x in v] for q, v in scores.items()}}
    else:
        ridge = cfg.pops_ridge
    out["ridge"] = ridge
    log(f"POPS (paper) ridge: {ridge}")
    rd = ridge if isinstance(ridge, dict) else {q: ridge for q in "EFV"}
    path = PopsRidgePath(theta, prob, d.ds_train, stats=stats)
    path.use_mean(POPS_MEAN)
    cst = np.asarray(path.c_star)
    rowsE, rowsF = ([], [], []), ([], [])
    for i in range(d.ds_test.n_batches):
        bt = jax.tree.map(lambda x_: x_[i], d.ds_test)
        lin, _, _ = linear_rows(prob.model, prob.cfg, bt)
        Lb, C = lin.E.shape[-1], bt.y_E.shape[0]
        nat_b = np.zeros(C + 1); np.add.at(nat_b, np.asarray(bt.node_cfg), np.asarray(bt.node_mask, float))
        kE = np.asarray(bt.w_E) > 0
        phE = np.asarray(lin.E)[kE]
        rowsE[0].append(phE); rowsE[1].append(np.asarray(bt.y_E)[kE] - phE @ cst); rowsE[2].append(nat_b[:C][kE])
        kF = np.repeat(np.asarray(bt.w_F) > 0, 3)
        phF = np.asarray(lin.F).reshape(-1, Lb)[kF]
        rowsF[0].append(phF); rowsF[1].append(np.asarray(bt.y_F).reshape(-1)[kF] - phF @ cst)
    phE, rE, natE = (np.concatenate(x_) for x_ in rowsE)
    phF, rF = (np.concatenate(x_) for x_ in rowsF)
    sel = np.random.default_rng(cfg.seed).choice(len(rF), size=min(cfg.pops_env_nf, len(rF)), replace=False)
    phF, rF = phF[np.sort(sel)], rF[np.sort(sel)]
    env, env_m = {}, {}
    for q, ph, r, sc in (("E", phE, rE, natE), ("F", phF, rF, np.ones(len(rF)))):
        lo, hi = path.envelope(jnp.asarray(ph), rd[q], cfg.pops_leverage_pct)
        lo, hi = np.asarray(lo), np.asarray(hi)
        unit = 1e3 if q == "E" else 1.0
        env_m[q] = {"env_cover": float(np.mean((lo <= r) & (r <= hi))),
                    "env_width_median": float(np.median(unit * (hi - lo) / sc)), "env_n": int(len(r))}
        env.update({f"{q}_lo": lo / sc * unit, f"{q}_hi": hi / sc * unit, f"{q}_resid": r / sc * unit})
    env["F_index"] = np.sort(sel)
    out.update(envelope=env, envelope_metrics=env_m, path=path)
    return out


def predict_splits(cfg, d, b, stats, theta, draws, log=print):
    prob, arrays, metrics, tm = b.prob, {}, {}, {}
    pops = {}
    if cfg.uq == "pops":
        t = time.time()
        pops = _pops_setup(cfg, d, b, stats, theta, log)
        tm["pops_paper_setup"] = time.time() - t
    splits = [("test", d.test_o, d.ds_test, d.base_test)]
    if cfg.predict_train:
        splits.append(("train", d.train_o, d.ds_train, d.base_train))
    if d.ds_ood is not None:
        splits.append(("ood", d.ood_o, d.ds_ood, d.base_ood))
    for rung, dr in draws.items():
        sub = dr if len(dr) <= cfg.n_draws else dr[np.linspace(0, len(dr) - 1, cfg.n_draws).astype(int)]
        for split, cfgs, ds, base in splits:
            t = time.time()
            if cfg.uq == "pops":
                pred = predict_fixed(theta, prob, d.ds_train, ds, deriv_dtc=cfg.deriv_dtc, uq="pops",
                                     pops_form=cfg.pops_posterior, leverage_pct=cfg.pops_leverage_pct,
                                     pops_ridge=pops["ridge"], pops_path=pops["path"], stats=stats)
            else:
                pred = predict_mixture(sub, prob, d.ds_train, ds, deriv_dtc=cfg.deriv_dtc, stats=stats)
            tm[f"predict_{split}_{rung}"] = time.time() - t
            nat = np.array([len(c.numbers) for c in cfgs])
            bE = np.array([x[0] for x in base])
            bF = np.concatenate([x[1] for x in base]) if base else np.zeros((0, 3))
            bV6 = np.array([[x[2][i, j] for i, j in VOIGT] for x in base])
            E, F, V = _labels(cfgs)
            Em = np.asarray(pred.E_mean) + bE; Fm = np.asarray(pred.F_mean) + bF
            Vm = np.asarray(pred.V_mean) + bV6
            s2 = {k: float(np.mean(np.exp(2 * sub[:, i]))) for k, i in (("E", 7), ("F", 8), ("V", 9))}
            arrays[f"{split}/{rung}"] = dict(
                nat=nat, E=E, E_mean=Em, E_var=np.asarray(pred.E_var), F=F, F_mean=Fm,
                F_var=np.asarray(pred.F_var), V=V, V_mean=Vm, V_var=np.asarray(pred.V_var),
                noise_E=s2["E"] * nat, noise_F=np.full(len(F), s2["F"]), noise_V=s2["V"] * nat)
            m = label_metrics(E, Em, np.asarray(pred.E_var), F, Fm, np.asarray(pred.F_var),
                              V, Vm, np.asarray(pred.V_var), nat)
            if split == "test":
                for q, v in pops.get("envelope_metrics", {}).items():
                    m[q].update(v)
            metrics[f"{split}/{rung}"] = m
            log(f"{split} {rung} " + str({q: {k: round(x, 4) for k, x in v.items()
                                              if k in ("rmse", "crps", "coverage", "rho", "rms_z")}
                                          for q, v in m.items()}))
    pops.pop("path", None)
    return Preds(arrays, metrics, pops, tm)
