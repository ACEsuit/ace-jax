"""Bond-scan diagnostic (CALM Test 1, arXiv:2609.40060; plan: docs/dev/force-uq-vs-calm.md, Step 1).

A rattled elemental (or random-alloy) prototype is scaled hydrostatically from 0.5 d_eq to r_cut (d the
nearest-neighbour distance) and every point is served by `ACECalculator(model, posterior=...)`.  Per point,
max over atoms of: the unscaled shape v = tr V, forces_std, forces_q, the group, support_q, the fraction of
atoms with support_ok False, and the whitened support-PCA radius |x_c| (the classifier's input).

The cell must be rattled: in a perfect fcc / bcc / diamond lattice every per-atom vector vanishes by symmetry,
the force design rows phi_a included, so v = 0 at every scale.  The rattle is ONE fixed unit-Gaussian pattern
times `--rattle` x d (homologous: the relative distortion is the same at every scale); the unrattled cell's
max v is printed as that symmetry check.  The supercell repeats the conventional cell n >= 2 r_cut / a times,
so it is wider than r_cut down to 0.5 d_eq.

Detection, following the paper: a metric detects a point when it exceeds `f` x its value at d_eq; the report
gives the fraction of points detected in the compression (d < 0.8 d_eq) and stretching (1.4 d_eq < d < r_cut)
windows for a sweep of f (SI2 style).  support_ok has no factor: a point is detected when any atom is flagged.

    python bond_scan.py --model M.npz --posterior P.npz --proto Si:diamond:5.431 --out DIR [--n 121]
    python bond_scan.py ... --proto CrMnFeCoNi:fcc:3.6502 --proto Ni:fcc:3.6502 --proto Fe:bcc:2.8971
Writes DIR/<proto>.npz, DIR/bond_scan.md (table) and DIR/bond_scan.png.
"""
import argparse
import pathlib
import re
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

FACTORS = (1.5, 2.0, 3.0, 5.0, 10.0)
COMP, STRETCH = 0.8, 1.4                      # windows in d / d_eq (stretching up to r_cut)
NN = {"fcc": np.sqrt(0.5), "bcc": np.sqrt(3) / 2, "diamond": np.sqrt(3) / 4, "sc": 1.0}   # d_eq / a


def prototype(spec, rcut, seed=0):
    """(atoms at d_eq, d_eq) for "Els:lattice:a": a cubic conventional cell repeated n >= 2 rcut / a times;
    several elements (e.g. CrMnFeCoNi) give an equiatomic random occupation (seeded)."""
    from ase.build import bulk
    from ase.data import chemical_symbols
    els, lat, a = spec.split(":")
    a = float(a)
    sym = re.findall("[A-Z][a-z]?", els)
    if "".join(sym) != els or not set(sym) <= set(chemical_symbols):
        raise ValueError(f"cannot split {els!r} into element symbols")
    at = bulk(sym[0], lat, a=a, cubic=True)
    n = int(np.ceil(2 * rcut / a))
    at = at.repeat((n, n, n))
    if len(sym) > 1:
        rng = np.random.default_rng(seed)
        at.set_chemical_symbols(rng.permutation(np.resize(sym, len(at))))
    return at, NN[lat] * a


def scaled(at0, s, u, rattle, d_eq):
    at = at0.copy()
    at.set_cell(at0.cell * s, scale_atoms=True)
    at.positions += rattle * s * d_eq * u
    return at


def evaluate(calc, at, post, support=True):
    from ace_jax.fit.support import _proj
    std = calc.get_property("forces_std", at)
    out = {"forces_std": std}
    if post.group_table is not None:
        g = np.asarray(calc.get_property("forces_group", at))
        lam = np.asarray(post.group_table["lam_rms"], float)[g]
        out.update(v=(std / lam) ** 2, forces_q=calc.get_property("forces_q", at), group=g)
    if support and post.support is not None:
        sup = calc.get_property("forces_support", at)
        X, Z = calc.support_descriptors(at)
        xc = np.zeros(len(Z))
        for z in np.unique(Z):
            if int(z) in post.support["pca"]:
                xc[Z == z] = np.linalg.norm(_proj(post.support["pca"], z, X[Z == z]), axis=1)
        out.update(support_q=np.asarray(sup["support_q"]), support_bad=~np.asarray(sup["support_ok"]),
                   xc=xc, phinorm=np.linalg.norm(X, axis=1))
    return out


def scan(calc, post, spec, rcut, n, rattle, support=True, seed=0, log=print):
    at0, d_eq = prototype(spec, rcut, seed)
    u = np.random.default_rng(seed + 1).standard_normal((len(at0), 3))
    v0 = evaluate(calc, scaled(at0, 1.0, u, 0.0, d_eq), post, support=False)
    if "v" in v0:
        log(f"{spec}: {len(at0)} atoms, d_eq {d_eq:.4f} A; unrattled max v = {np.max(v0['v']):.2e} (0 by symmetry for one element)")
    s = np.unique(np.r_[np.linspace(0.5, rcut / d_eq, n), 1.0])
    rows = []
    for k, sk in enumerate(s):
        t = time.time()
        r = evaluate(calc, scaled(at0, sk, u, rattle, d_eq), post, support)
        rows.append(r)
        if k % 20 == 0:
            log(f"  {k + 1}/{len(s)} d/d_eq {sk:.3f}: max forces_std {np.max(r['forces_std']):.3g} "
                f"({time.time() - t:.1f} s)")
    agg = {"d_rel": s, "d_eq": d_eq, "rcut": rcut, "n_atoms": len(at0)}
    for key in rows[0]:
        A = np.stack([r[key] for r in rows])
        agg[key] = A.mean(1) if key == "support_bad" else np.median(A, 1) if key == "group" else A.max(1)
        if key == "group":
            agg["group_max"] = A.max(1)
    return agg


def detection(agg, factors=FACTORS):
    """{metric: {"comp"|"stretch": [fraction of window points detected per factor]}}; support_bad has one
    entry (any atom flagged)."""
    s, rmax = agg["d_rel"], agg["rcut"] / agg["d_eq"]
    win = {"comp": s < COMP, "stretch": (s > STRETCH) & (s < rmax)}
    i_eq = int(np.argmin(np.abs(s - 1.0)))
    out = {}
    for key in ("v", "forces_std", "forces_q", "support_q", "xc"):
        if key in agg:
            y = agg[key]
            out[key] = {w: [float(np.mean(y[m] > f * y[i_eq])) for f in factors] for w, m in win.items()}
    if "support_bad" in agg:
        out["support_ok=False"] = {w: [float(np.mean(agg["support_bad"][m] > 0))] for w, m in win.items()}
        out["support_ok=False"]["eq"] = float(agg["support_bad"][i_eq])
    return out


def report(results, factors=FACTORS):
    lines = [f"Fraction of scan points detected (metric > f x its d_eq value; support: any atom flagged). "
             f"Compression d < {COMP} d_eq, stretching {STRETCH} d_eq < d < r_cut.", "",
             "| prototype | metric | window | " + " | ".join(f"f={f:g}" for f in factors) + " |",
             "|---|---|---|" + "---|" * len(factors)]
    for spec, agg in results.items():
        for key, d in detection(agg, factors).items():
            for w in ("comp", "stretch"):
                vals = d[w] + [None] * (len(factors) - len(d[w]))
                lines.append(f"| {spec} | {key} | {w} | " + " | ".join("" if x is None else f"{x:.2f}" for x in vals)
                             + " |")
    lines += ["", "| prototype | atoms | d_eq (A) | support flagged at d_eq | groups (median atom) over the scan |",
              "|---|---|---|---|---|"]
    for spec, agg in results.items():
        g = agg.get("group")
        gs = "" if g is None else ", ".join(f"{int(a)}: {agg['d_rel'][g == a].min():.2f}-{agg['d_rel'][g == a].max():.2f}"
                                           for a in np.unique(g))
        eq = detection(agg).get("support_ok=False", {}).get("eq")
        lines.append(f"| {spec} | {agg['n_atoms']} | {agg['d_eq']:.3f} | "
                     f"{'' if eq is None else f'{eq:.2f}'} | {gs} |")
    return "\n".join(lines) + "\n"


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ("v", "forces_std", "forces_q", "xc") if any(k in a for a in results.values())]
    fig, axs = plt.subplots(len(keys) + 1, 1, figsize=(7, 2.2 * (len(keys) + 1)), sharex=True)
    for spec, agg in results.items():
        s = agg["d_rel"]
        i_eq = int(np.argmin(np.abs(s - 1.0)))
        for ax, k in zip(axs, keys):
            if k in agg:
                ax.semilogy(s, np.maximum(agg[k] / agg[k][i_eq], 1e-6), label=spec)
                ax.set_ylabel(f"{k} / eq")
        if "support_bad" in agg:
            axs[-1].plot(s, agg["support_bad"], label=spec)
    axs[-1].set_ylabel("frac. support_ok=False")
    axs[-1].set_xlabel("d / d_eq")
    for ax in axs:
        ax.axvspan(s.min(), COMP, color="0.9")
        ax.axvspan(STRETCH, max(a["rcut"] / a["d_eq"] for a in results.values()), color="0.9")
        ax.axvline(1.0, color="0.5", lw=0.5)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--posterior", required=True)
    p.add_argument("--proto", action="append", required=True, help="Els:lattice:a, e.g. Si:diamond:5.431")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=121)
    p.add_argument("--rattle", type=float, default=0.01, help="rattle std as a fraction of d")
    p.add_argument("--no-support", action="store_true")
    a = p.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calc = ACECalculator(a.model, posterior=a.posterior)
    post = ARDPosterior.load(a.posterior)
    rcut = float(calc.cutoff)
    results = {}
    for spec in a.proto:
        results[spec] = scan(calc, post, spec, rcut, a.n, a.rattle, support=not a.no_support)
        np.savez(out / f"{spec.replace(':', '_')}.npz", **results[spec])
    md = report(results)
    (out / "bond_scan.md").write_text(md)
    plot(results, out / "bond_scan.png")
    print(md)


if __name__ == "__main__":
    main()
