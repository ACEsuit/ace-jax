"""Numerical-precision audit of the served force-uncertainty shape (plan: docs/dev/force-uq-vs-calm.md, Step 4, H4).

The shape is v(x) = sum_a |R^T u_a|^2, u_a = D^-1 phi_a, with R = S^-1 D^-1 (G - g_bar) (jackknife.shape_factor)
computed in float64 at fit time but STORED in float32 (ARDPosterior.save): the conformal scores are made with the
float64 R and the served shape uses the rounded one.  The float64 R is not kept, so the rounding is modelled:
R (1 + delta), delta ~ U(-2^-24, 2^-24) per entry (float32 round-to-nearest), several seeds.  Variants, per atom
against the shape served from the stored posterior:
- B32: the force design rows phi rounded to float32 (a float32 row evaluation);
- R32 seed k: the stored R perturbed by another float32 rounding;
- S-floor tau (with --spectrum): R with its components along the eigenvectors of S = C C^T whose eigenvalue
  lambda < tau lambda_max removed (cond(S) is floored at ard_cond_max, 1e14 by default) -- how much of v
  comes from directions S resolves poorly (cf. CALM's LLPR finding (e), arXiv:2609.40060).
Reported: Spearman rho(v, v') over atoms (on v to 5 significant figures, so tied symmetric atoms stay tied), max and p99 of |v'/v - 1|, the fraction of atoms moving > 1 %, and
the singular-value spectrum of R (and of S with --spectrum).

    python precision_audit.py --model M --posterior P --cells a.xyz[:N] [--cells b.xyz[:N]] --out audit.md [--spectrum]
"""
import argparse
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np  # noqa: E402
from _frames import read_atoms, sig_round  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

TAUS = (1e-14, 1e-12, 1e-10, 1e-8)


def force_rows(calc, at):
    """(N, 3, L) float64 force design rows of the live atoms of `at` (as ACECalculator serves them)."""
    from ace_jax.fit.rows import chunked_rows_fn
    from ace_jax.eval.model import highest_precision
    ds, b, live = calc._one_config_dataset(at)
    if getattr(calc, "_rows_fn", None) is None:
        calc._rows_fn = chunked_rows_fn(calc._fit_model, calc._fit_cfg)
    with highest_precision():
        F = calc._rows_fn(b).F
    return np.asarray(F)[np.asarray(live)]


def v_of(R, dinv, F):
    from ace_jax.fit.jackknife import atom_shape
    return np.trace(atom_shape(R, dinv, F), axis1=1, axis2=2)


def rounding(R, seed):
    d = np.random.default_rng(seed).uniform(-2.0 ** -24, 2.0 ** -24, R.shape)
    return R * (1.0 + d)


def compare(v, w):
    rel = np.abs(w / np.where(v > 0, v, np.nan) - 1.0)
    rel = rel[np.isfinite(rel)]
    return {"rho": float(spearmanr(sig_round(v), sig_round(w))[0]), "max": float(rel.max()), "p99": float(np.quantile(rel, 0.99)),
            "frac>1%": float(np.mean(rel > 0.01))}


def main(argv=None):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--posterior", required=True)
    p.add_argument("--cells", action="append", required=True, help="FILE.xyz[:N] (N evenly spaced frames)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--spectrum", action="store_true", help="eigendecompose S (L x L; minutes, ~4 L^2 x 8 bytes)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    post = ARDPosterior.load(a.posterior)
    if post.R is None:
        raise SystemExit("posterior has no schema-3 shape factor R")
    calc = ACECalculator(a.model, posterior=a.posterior)
    L0 = calc._fit_cfg.len_basis
    R = np.asarray(post.R, np.float64)[:L0]
    dinv = np.asarray(post.dinv, np.float64)[:L0]
    Rs = [rounding(R, k) for k in range(a.seeds)]
    proj = {}
    lines = []
    sv = np.linalg.svd(R, compute_uv=False)
    lines += [f"R: {R.shape[0]} x {R.shape[1]}; singular values max {sv[0]:.3e}, median {np.median(sv):.3e}, "
              f"min {sv[-1]:.3e} (sigma_max/sigma_min {sv[0] / max(sv[-1], 1e-300):.2e}); "
              f"{int(np.sum(sv < sv[0] * 2.0 ** -24))} below sigma_max x 2^-24", ""]
    if a.spectrum:
        C = np.asarray(post.chol, np.float64)[:L0, :L0]
        U, s, _ = np.linalg.svd(C, full_matrices=False)               # S = C C^T = U s^2 U^T
        lam = s ** 2
        del C
        lines += [f"S: lambda_max {lam[0]:.3e}, lambda_min {lam[-1]:.3e}, cond {lam[0] / lam[-1]:.2e}; "
                  + ", ".join(f"{int(np.sum(lam < t * lam[0]))} eigenvalues < {t:g} lambda_max" for t in TAUS), ""]
        UR = U.T @ R
        for t in TAUS:
            small = lam < t * lam[0]
            if small.any():
                proj[f"S-floor {t:g}"] = R - U[:, small] @ UR[small]
        del U, UR
    acc = {k: [] for k in ["base", "B32", *(f"R32 seed {k}" for k in range(a.seeds)), *proj]}
    for spec in a.cells:
        path, n = (spec.split(":") + [None])[:2]
        frames = read_atoms(path)
        if n:
            frames = frames[::max(1, len(frames) // int(n))][:int(n)]
        for at in frames:
            F = force_rows(calc, at)
            acc["base"].append(v_of(R, dinv, F))
            acc["B32"].append(v_of(R, dinv, F.astype(np.float32).astype(np.float64)))
            for k in range(a.seeds):
                acc[f"R32 seed {k}"].append(v_of(Rs[k], dinv, F))
            for name, Rp in proj.items():
                acc[name].append(v_of(Rp, dinv, F))
        print(f"{spec}: {len(frames)} frames")
    v = {k: np.concatenate(x) for k, x in acc.items()}
    lines += [f"{len(v['base'])} atoms ({', '.join(a.cells)}).", "",
              "| variant | Spearman rho(v, v') | max abs(v'/v - 1) | p99 | atoms moving > 1 % |", "|---|---|---|---|---|"]
    for k in acc:
        if k != "base":
            c = compare(v["base"], v[k])
            lines.append(f"| {k} | {c['rho']:.9f} | {c['max']:.2e} | {c['p99']:.2e} | {c['frac>1%']:.4f} |")
    md = "\n".join(lines) + "\n"
    pathlib.Path(a.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
