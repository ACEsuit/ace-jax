"""Spike: reproduce the POPS behaviour on a 1D misspecified fit with the REFERENCE
`popsregression` package, to understand why 'samples'(=ensemble) vs 'hypercube'
looked SO far apart on SiGe/Cantor (rms-z ~19 vs ~1).

What the package actually does (popsregression/_pops.py, predict):
  * return_epistemic_std -> sqrt(X . sigma_ . X)              = ordinary BayesianRidge
        posterior variance. This is FORM-INDEPENDENT (no POPS term).
  * return_std           -> sqrt(X . misspecification_sigma_ . X + X . sigma_ . X)
        = POPS predictive: the misspecification term is the ONLY place the posterior
        form ('ensemble' centred committee cov vs 'hypercube' uncentred box 2nd
        moment) enters.
So the honest 'ensemble vs hypercube' comparison is return_std; the two differ only
by the mean term (phi.mean(delta))^2, which the box keeps and the committee cov drops.

Setup: y=sin(w x), low-degree polynomial (Vandermonde) = misspecification knob.
Calibration is against the NOISELESS truth (pure model error), which is exactly what
the POPS misspecification term must cover.

Run:  uv run python spike/pops_1d.py
"""
import numpy as np
from popsregression import POPSRegression

rng = np.random.default_rng(0)
W = 3.0
NOISE = 0.02
NTR, NTE = 120, 600

def truth(x):
    return np.sin(W * x)

def design(x, deg):
    return np.vander(np.asarray(x), deg + 1, increasing=True)

xtr = np.sort(rng.uniform(0, 3, NTR))       # asymmetric domain (odd truth => nonzero mean(delta))
xte = np.linspace(0, 3, NTE)
ytr = truth(xtr) + NOISE * rng.standard_normal(NTR)
yte = truth(xte)

def rmsz(y, mu, std):
    return float(np.sqrt(np.mean(((y - mu) / std) ** 2)))

def fit(post, X, y, clip=0.0):
    return POPSRegression(posterior=post, fit_intercept=False,
                          leverage_percentile=0.0, percentile_clipping=clip).fit(X, y)

print(f"1D: y=sin({W}x) on [0,3], {NTR} train / {NTE} test, label noise {NOISE}")
print("Degree = misspecification knob (lower deg = worse fit).  rms-z: ideal 1.\n")
print(f"{'deg':>3} {'fitRMSE':>8} | {'Bayes-only':>10} | {'POPS ens':>9} {'POPS hcube':>10} | {'mean-term':>9}")
print(f"{'':>3} {'':>8} | {'(no POPS)':>10} | {'return_std':>9} {'return_std':>10} | {'fraction':>9}")

for deg in (1, 2, 3, 4, 6, 8):
    Xtr, Xte = design(xtr, deg), design(xte, deg)
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    fit_rmse = float(np.sqrt(np.mean((yte - Xte @ beta) ** 2)))
    me = fit("ensemble", Xtr, ytr); mh = fit("hypercube", Xtr, ytr)
    mu, ens_std = me.predict(Xte, return_std=True)              # POPS predictive, ensemble
    _, bayes_std = me.predict(Xte, return_epistemic_std=True)   # ordinary Bayes (form-indep.)
    _, hc_std = mh.predict(Xte, return_std=True)                # POPS predictive, hypercube
    # mean-term fraction of the hypercube misspecification variance
    Sig0 = mh.sigma_; r = ytr - Xtr @ mh.coef_
    pc = Xtr @ Sig0; h = np.einsum("ij,ij->i", pc, Xtr)
    delta = pc * (r / h)[:, None]; mbar = delta.mean(0)
    centered = np.einsum("ij,jk,ik->i", Xte, np.cov(delta.T, bias=True), Xte).mean()
    meanterm = ((Xte @ mbar) ** 2).mean()
    frac = meanterm / (centered + meanterm)
    print(f"{deg:>3} {fit_rmse:>8.3f} | {rmsz(yte, mu, bayes_std):>10.1f} | "
          f"{rmsz(yte, mu, ens_std):>9.2f} {rmsz(yte, mu, hc_std):>10.2f} | {frac:>9.3f}")

print("\nTakeaways:")
print(" 1. 'Bayes-only' (no POPS misspecification term) is wildly overconfident under")
print("    misspecification (rms-z up to ~100) -- POPS's misspecification term is what")
print("    calibrates it. That gap, NOT centred-vs-uncentred, is the bulk of the SiGe")
print("    '19 vs 1' we saw (samples-run was misspec-thin; hcube-run carried the term).")
print(" 2. POPS ensemble vs hypercube (return_std) differ only by the mean-term fraction")
print("    -- small here (few %). The reference package does NOT show a ~5x form gap on")
print("    a benign 1D problem, so our ACE 5x is regime-specific (whitened multi-quantity")
print("    design => larger mean(delta)) or a port quirk -- worth a direct check.")
