"""UQ assessment of the production runs (prod_out/<run>/pred_{test,ood}_map.npz).

Per run, split (test / ood) and quantity (E, V per atom in meV; F per component in eV/A):

  sharpness+calibration  rmse, median sigma, rms-z, Gaussian NLL, CRPS, coverage at
                         50/68/90/95 % (reliability), ENCE over 10 equal-count sigma bins
  ranking                Spearman rho(sigma, |err|)
  post-hoc               global scale s = sqrt<z^2> and split-conformal |z| quantile,
                         fit on one config-half of TEST and scored on the other; the
                         same test-fitted s is then carried to OOD (does it transfer?)
  OOD                    sigma_ood/sigma_test vs rmse_ood/rmse_test (does sigma grow with
                         the error?) and AUROC of per-config energy sigma for OOD-vs-test

sigma variants: 'latent' (what run.py's metrics.json scores) and, for BLR/GP, 'label'
= latent + fitted noise (the predictive for a label).  POPS already is the full
predictive (no noise term), so it only gets 'latent'.  POPS runs also report the
min/max envelope coverage/width from pops_envelope_test.npz.

  python prod_uq.py prod_out/cantor_linear prod_out/cantor_gp ...  [--out uq.json]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm, spearmanr

LEVELS = (0.5, 0.68, 0.9, 0.95)


def crps(y, mu, s):
    w = (y - mu) / s
    return float(np.mean(s * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - 1 / np.sqrt(np.pi))))


def ence(err, s, nbins=10):
    o = np.argsort(s)
    parts = np.array_split(o, nbins)
    rms_s = np.array([np.sqrt(np.mean(s[p] ** 2)) for p in parts])
    rmse = np.array([np.sqrt(np.mean(err[p] ** 2)) for p in parts])
    return float(np.mean(np.abs(rms_s - rmse) / rms_s)), rms_s.tolist(), rmse.tolist()


def auroc(neg, pos):
    """P(score_pos > score_neg), ties half."""
    r = np.concatenate([neg, pos]).argsort().argsort() + 1.0
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def scores(y, mu, s):
    err = y - mu
    z = err / s
    e, rs, rm = ence(err, s)
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(err ** 2))), "median_sigma": float(np.median(s)),
            "rms_z": float(np.sqrt(np.mean(z ** 2))),
            "nll": float(np.mean(0.5 * z ** 2 + np.log(s) + 0.5 * np.log(2 * np.pi))),
            "crps": crps(y, mu, s),
            **{f"cov{int(100 * l)}": float(np.mean(np.abs(z) <= norm.ppf(0.5 + l / 2))) for l in LEVELS},
            "ence": e, "rho": float(spearmanr(s, np.abs(err))[0]),
            "bins_rms_sigma": rs, "bins_rmse": rm}


def rows(d, q, variant, pops):
    """(y, mu, sigma, config-id) per observation for quantity q."""
    nat = d["nat"].astype(float)
    nz = 0.0 if (pops or variant == "latent") else 1.0
    if q == "E":
        sc = 1e3 / nat
        return (d["E"] * sc, d["E_mean"] * sc, np.sqrt(np.maximum(d["E_var"] + nz * d["noise_E"], 1e-300)) * sc,
                np.arange(len(nat)))
    if q == "V":
        sc = (1e3 / nat)[:, None]
        v = d["V_var"] + nz * d["noise_V"][:, None]
        cid = np.repeat(np.arange(len(nat)), 6)
        return ((d["V"] * sc).ravel(), (d["V_mean"] * sc).ravel(), (np.sqrt(np.maximum(v, 1e-300)) * sc).ravel(), cid)
    v = d["F_var"] + nz * d["noise_F"][:, None]
    cid = np.repeat(np.repeat(np.arange(len(nat)), nat.astype(int)), 3)
    return d["F"].ravel(), d["F_mean"].ravel(), np.sqrt(np.maximum(v, 1e-300)).ravel(), cid


def posthoc(y, mu, s, cid, alpha=0.1, seed=0):
    """Fit s_glob and the conformal |z| quantile on a random half of the configs,
    score on the other half."""
    cfgs = np.unique(cid)
    p = np.random.default_rng(seed).permutation(cfgs)
    cal = np.isin(cid, p[: len(p) // 2])
    zc = (y[cal] - mu[cal]) / s[cal]
    ze = (y[~cal] - mu[~cal]) / s[~cal]
    sg = float(np.sqrt(np.mean(zc ** 2)))
    lvl = min(np.ceil((len(zc) + 1) * (1 - alpha)) / len(zc), 1.0)
    q = float(np.quantile(np.abs(zc), lvl, method="higher"))
    return {"s": sg, "rms_z_after": float(np.sqrt(np.mean((ze / sg) ** 2))),
            "crps_after": crps(y[~cal], mu[~cal], sg * s[~cal]),
            "cov90_scaled": float(np.mean(np.abs(ze) <= 1.645 * sg)),
            "cov90_conformal": float(np.mean(np.abs(ze) <= q)), "q90_conformal": q}


OOD_TYPES = None        # per-config ood_type labels (file order), from --ood-types <xyz>


def _subset(R, keep_cfg):
    m = np.isin(R[3], keep_cfg)
    return tuple(a[m] for a in R)


def assess(run):
    name = os.path.basename(run.rstrip("/"))
    pops = os.path.exists(os.path.join(run, "pops_envelope_test.npz"))
    preds = {sp: np.load(os.path.join(run, f"pred_{sp}_map.npz"))
             for sp in ("test", "ood") if os.path.exists(os.path.join(run, f"pred_{sp}_map.npz"))}
    out = {"run": name, "pops": pops, "splits": sorted(preds)}
    for variant in (("latent",) if pops else ("latent", "label")):
        res = out.setdefault(variant, {})
        for q in "EFV":
            r = res.setdefault(q, {})
            T = rows(preds["test"], q, variant, pops)
            r["test"] = scores(*T[:3])
            r["posthoc_test"] = posthoc(*T)
            if "ood" in preds:
                O = rows(preds["ood"], q, variant, pops)
                r["ood"] = scores(*O[:3])
                sg = r["posthoc_test"]["s"]                    # test-fitted scale carried to OOD
                zo = (O[0] - O[1]) / (sg * O[2])
                r["ood_with_test_scale"] = {"rms_z": float(np.sqrt(np.mean(zo ** 2))),
                                            "cov90": float(np.mean(np.abs(zo) <= 1.645)),
                                            "cov90_conformal": float(np.mean(np.abs((O[0] - O[1]) / O[2])
                                                                             <= r["posthoc_test"]["q90_conformal"]))}
                r["ood_shift"] = {"sigma_ratio": r["ood"]["median_sigma"] / r["test"]["median_sigma"],
                                  "rmse_ratio": r["ood"]["rmse"] / r["test"]["rmse"]}
                if q == "E":
                    r["ood_shift"]["auroc_sigma_E"] = auroc(T[2], O[2])
                if OOD_TYPES is not None:
                    types = np.asarray(OOD_TYPES)
                    assert len(types) == len(preds["ood"]["nat"]), "ood_type count != OOD configs"
                    r["ood_by_type"] = {}
                    for t in sorted(set(types)):
                        Ot = _subset(O, np.flatnonzero(types == t))
                        sc = scores(*Ot[:3])
                        zt = (Ot[0] - Ot[1]) / (sg * Ot[2])
                        r["ood_by_type"][t] = {
                            "n": sc["n"], "rmse": sc["rmse"], "rms_z": sc["rms_z"], "cov90": sc["cov90"],
                            "rho": sc["rho"], "crps": sc["crps"],
                            "rms_z_test_scale": float(np.sqrt(np.mean(zt ** 2))),
                            "sigma_ratio": sc["median_sigma"] / r["test"]["median_sigma"],
                            "rmse_ratio": sc["rmse"] / r["test"]["rmse"],
                            **({"auroc_sigma_E": auroc(T[2], Ot[2])} if q == "E" else {})}
    if pops:
        e = np.load(os.path.join(run, "pops_envelope_test.npz"))
        env = {}
        for q in "EF":
            if f"{q}_lo" in e:
                lo, hi, res_ = e[f"{q}_lo"], e[f"{q}_hi"], e[f"{q}_resid"]
                env[q] = {"cover": float(np.mean((res_ >= lo) & (res_ <= hi))),
                          "median_width_over_rmse": float(np.median(hi - lo) / np.sqrt(np.mean(res_ ** 2)))}
        out["envelope_test"] = env
    return out


def table(results):
    hdr = ("| run | σ | q | RMSE | rms-z | cov90 | CRPS | ENCE | ρ | s (post-hoc) | rms-z after | "
           "OOD rms-z | OOD rms-z (test s) | σ↑ / err↑ OOD |")
    lines = [hdr, "|" + "---|" * 14]
    for R in results:
        for variant in ("latent", "label"):
            if variant not in R:
                continue
            for q in "EFV":
                r = R[variant][q]
                t, p = r["test"], r["posthoc_test"]
                o = r.get("ood"); ot = r.get("ood_with_test_scale"); sh = r.get("ood_shift")
                lines.append(
                    f"| {R['run']} | {variant} | {q} | {t['rmse']:.3g} | {t['rms_z']:.2f} | {t['cov90']:.2f} | "
                    f"{t['crps']:.3g} | {t['ence']:.2f} | {t['rho']:.2f} | {p['s']:.2f} | {p['rms_z_after']:.2f} | "
                    + (f"{o['rms_z']:.2f} | {ot['rms_z']:.2f} | {sh['sigma_ratio']:.2f} / {sh['rmse_ratio']:.2f} |"
                       if o else "– | – | – |"))
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    dest = "prod_uq.json"
    if "--out" in args:
        i = args.index("--out"); dest = args[i + 1]; del args[i:i + 2]
    if "--ood-types" in args:
        i = args.index("--ood-types")
        from ase.io import read
        OOD_TYPES = [str(a.info.get("ood_type", "?")) for a in read(args[i + 1], ":")]
        del args[i:i + 2]
    res = [assess(r) for r in args]
    json.dump(res, open(dest, "w"), indent=1)
    print(table(res))
    for R in res:
        if "envelope_test" in R:
            print(R["run"], "POPS envelope:", R["envelope_test"])
        for v in ("latent", "label"):
            if v in R and "ood_shift" in R[v]["E"]:
                print(R["run"], v, "AUROC(σ_E: ood vs test) =", round(R[v]["E"]["ood_shift"]["auroc_sigma_E"], 3))
            for q in "EFV":
                for t, b in (R.get(v, {}).get(q, {}).get("ood_by_type") or {}).items():
                    print(f"  {R['run']:24s} {v:6s} {q} {t:9s} rmse {b['rmse']:.3g} (x{b['rmse_ratio']:.1f} test) "
                          f"rms-z {b['rms_z']:.2f} cov90 {b['cov90']:.2f} rho {b['rho']:.2f} "
                          f"sigma x{b['sigma_ratio']:.1f}" + (f" AUROC {b['auroc_sigma_E']:.3f}" if "auroc_sigma_E" in b else ""))
