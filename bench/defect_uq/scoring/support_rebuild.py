"""Rebuild a fitted posterior's covariate-shift support reference under other features, without refitting.

Plan: docs/dev/force-uq-vs-calm.md, Step 2 (raw vs normalised support features).  The support reference is a
diagnostic downstream of the fit: it needs the training descriptors (for the per-species PCA) and the
T_val atoms with their conformal scores, groups and configuration ids -- which the posterior stores (`cal_*`)
and ard.json's split records (val_idx into the training configurations).  So it can be rebuilt with
`--ard-support-features raw|normalised` semantics from the same calibration points as the fit's own.

The PCA pool here is capped at --pca-atoms per species (the fit uses ard_support_max_atoms = 50 000); the
"raw" rebuild against the stored reference measures what that changes.

    python support_rebuild.py build --model M --posterior P --ard-json J --train train.xyz --out DIR
        [--features raw,normalised] [--pca-atoms 10000] [--energy-key ...]
    python support_rebuild.py eval --model M --posterior P --ref DIR/support_<kind>.npz [--stored]
        --set NAME=FILE.xyz[:desc.npz][@fam_key] ... --out report.md
"""
import argparse
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np  # noqa: E402
from _frames import read_atoms  # noqa: E402


def descriptors(calc, frames, log=print):
    """(X, Z, cfg) of every atom of frames (raw compact descriptors, species index, frame index)."""
    Xs, Zs, cs = [], [], []
    for i, fr in enumerate(frames):
        X, Z = calc.support_descriptors(fr)
        Xs.append(X); Zs.append(Z); cs.append(np.full(len(Z), i))
        if i % 200 == 0:
            log(f"  descriptors {i + 1}/{len(frames)}")
    return np.concatenate(Xs), np.concatenate(Zs), np.concatenate(cs)


def build(a):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.support import build_support, explained_variance, fit_pca, flatten_support, n_pass, \
        support_body, support_features
    post = ARDPosterior.load(a.posterior)
    calc = ACECalculator(a.model, posterior=a.posterior)
    split = json.load(open(a.ard_json))["split"]
    train = read_atoms(a.train)
    cal = post.cal
    # cal["cfg"] is the training-set index of each T_val atom's configuration (ard.py run_ard_stage); the
    # atoms of one configuration are stored in node order
    u = np.unique(cal["cfg"])
    if not set(u.tolist()) <= set(split["val_idx"]):
        raise ValueError("the posterior's calibration configurations are not in ard.json's val_idx")
    cnt = np.array([np.sum(cal["cfg"] == c) for c in u])
    nat = np.array([len(train[c]) for c in u])
    if not np.array_equal(cnt, nat):
        bad = np.flatnonzero(cnt != nat)
        raise ValueError(f"cal atoms per T_val config do not match {a.train} at {len(bad)} configs "
                         f"(first {u[bad[:5]]}: {cnt[bad[:5]]} vs {nat[bad[:5]]}); is it the fit's training file?")
    fit_idx = np.setdiff1d(np.arange(len(train)), split["val_idx"])
    Xv, Zv, _ = descriptors(calc, [train[c] for c in u])        # config order u, node order within
    o = np.argsort(cal["cfg"], kind="stable")                   # cal order -> the same blocks
    Xv, Zv = Xv[np.argsort(o)], Zv[np.argsort(o)]               # the descriptors in cal order
    rng = np.random.default_rng(0)
    pool = {}
    for i in rng.permutation(fit_idx):
        if pool and all(len(v) >= a.pca_atoms for v in pool.values()) and len(pool) == len(np.unique(Zv)):
            break
        X, Z = calc.support_descriptors(train[i])
        for z in np.unique(Z):
            pool.setdefault(int(z), []).append(X[Z == z])
    pool = {z: np.concatenate(v)[:a.pca_atoms] for z, v in pool.items()}
    body = support_body({"nnll": calc.meta["nnll"], "n_pair": calc.meta["n_pair"]})
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # raw descriptors of the calibration atoms (cal order) and of the training pool, for novelty_groups.py
    np.savez(out / "descriptors.npz", X_cal=Xv.astype(np.float32), Z_cal=Zv,
             **{f"X_pool_{z}": v.astype(np.float32) for z, v in pool.items()}, body=body)
    for kind in a.features.split(","):
        feat = {"kind": kind, "body": body}
        F = {z: support_features(v, feat) for z, v in pool.items()}
        pca = fit_pca(F, n_pass=n_pass(feat))
        ev = explained_variance(F, pca, n_pass(feat))
        print(f"{kind}: components " + ", ".join(f"{z}: {pca[z][2].shape[1] - n_pass(feat)} ({ev[z]:.3f})"
                                                   for z in sorted(pca)))
        k = np.isin(Zv, list(pca))
        ref = build_support(pca, Xv[k], Zv[k], np.asarray(cal["scores"], float)[k], np.asarray(cal["cfg"])[k],
                            50000, 0, grp=np.asarray(cal["groups"])[k], features=None if kind == "raw" else feat)
        np.savez(out / f"support_{kind}.npz", **flatten_support(ref))
    return out


def load_ref(path):
    from ace_jax.fit.support import unflatten_support
    z = np.load(path)
    return unflatten_support({k[8:]: z[k] for k in z.files if k.startswith("support_")})


def evaluate(ref, sets, alpha, calc=None):
    """{name: {family: (frac flagged, n_eff median, atoms)}}; support_check per configuration (as served)."""
    from ace_jax.fit.support import support_check
    res = {}
    for name, (X, Z, cfg, fam) in sets.items():
        bad, ne, fams = np.zeros(len(Z), bool), [], np.asarray(fam)
        for c in np.unique(cfg):
            m = cfg == c
            r = support_check(ref, X[m], Z[m], alpha)
            bad[m] = ~r["support_ok"]
            ne.append(np.median(list(r["n_eff"].values())))
        res[name] = {f: (float(bad[fams == f].mean()), int((fams == f).sum())) for f in np.unique(fams)}
        res[name]["_bad"] = bad
        res[name]["all"] = (float(bad.mean()), len(bad))
        res[name]["_neff"] = float(np.median(ne))
    return res


def load_set(spec, calc, max_cfg=None):
    """NAME=FILE.xyz[:desc.npz][@family_key] -> (X, Z, cfg, family)."""
    name, rest = spec.split("=", 1)
    fam_key = None
    if "@" in rest:
        rest, fam_key = rest.split("@")
    xyz, desc = (rest.split(":") + [None])[:2]
    frames = read_atoms(xyz)
    if max_cfg and not desc:
        frames = frames[::max(1, len(frames) // max_cfg)][:max_cfg]
    if desc:
        D = np.load(desc)
        if len(D["Z"]) != sum(len(f) for f in frames):
            raise ValueError(f"{desc}: {len(D['Z'])} rows but {xyz} has {sum(len(f) for f in frames)} atoms")
        X, Z = D["X"].astype(float), D["Z"].astype(int)
        cfg = np.concatenate([np.full(len(f), i) for i, f in enumerate(frames)])
    else:
        X, Z, cfg = descriptors(calc, frames, log=lambda *_: None)
    if fam_key:
        fam = np.concatenate([np.asarray(f.arrays[fam_key]).astype(str) if fam_key in f.arrays
                              else np.full(len(f), str(f.info.get(fam_key, "?"))) for f in frames])
    else:
        fam = np.full(len(Z), name)
    free = np.concatenate([~np.asarray(f.arrays.get("fixed", np.zeros(len(f), bool))).astype(bool) for f in frames])
    return name, (X[free], Z[free], cfg[free], fam[free])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for k in ("--model", "--posterior", "--ard-json", "--train", "--out"):
        b.add_argument(k, required=True)
    b.add_argument("--features", default="raw,normalised")
    b.add_argument("--pca-atoms", type=int, default=10000)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--posterior", required=True)
    e.add_argument("--ref", action="append", default=[], help="support_<kind>.npz from build")
    e.add_argument("--stored", action="store_true", help="also the posterior's own stored reference")
    e.add_argument("--set", action="append", required=True)
    e.add_argument("--max-cfg", type=int, default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--save", help="npz of per-atom flags, keys <reference>@<set>")
    a = p.parse_args(argv)
    if a.cmd == "build":
        build(a)
        return
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    post = ARDPosterior.load(a.posterior)
    calc = ACECalculator(a.model, posterior=a.posterior)
    alpha = float(np.asarray(post.group_table["alpha"]))
    refs = {pathlib.Path(r).stem: load_ref(r) for r in a.ref}
    if a.stored:
        refs = {"stored (fit)": post.support, **refs}
    sets = dict(load_set(s, calc, a.max_cfg) for s in a.set)
    lines = ["| reference | set | family | atoms | flagged (support_ok=False) |", "|---|---|---|---|---|"]
    flags = {}
    for rn, ref in refs.items():
        res = evaluate(ref, sets, alpha)
        flags.update({f"{rn}@{sn}": d["_bad"] for sn, d in res.items()})
        for sn, d in res.items():
            for f, v in d.items():
                if not f.startswith("_"):
                    lines.append(f"| {rn} | {sn} | {f} | {v[1]} | {v[0]:.3f} |")
            lines.append(f"| {rn} | {sn} | median n_eff | | {d['_neff']:.0f} |")
    md = "\n".join(lines) + "\n"
    pathlib.Path(a.out).write_text(md)
    if a.save:
        np.savez(a.save, **flags)
    print(md)


if __name__ == "__main__":
    main()
