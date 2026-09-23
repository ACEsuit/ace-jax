# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["ace-jax", "POPSRegression>=0.4"]
#
# [tool.uv.sources]
# ace-jax = { path = "..", editable = true }
# ///
"""Does POPS need a noise floor in the WELL-DETERMINED regime (production has lots
of data), or only in the underdetermined SiGe toy?

pops_aliasing.py showed the reference POPS calibrates on the ACE energy design with
no floor -- but there N_features(1110) >> N_train(175), so the huge BayesianRidge
posterior variance (null-space epistemic) did the calibrating and the
misspecification box added almost nothing. Production is the opposite limit.

Here we PCA-reduce the ACE energy features to k so we can sweep N_train from
underdetermined to well-determined, and DECOMPOSE the reference POPS predictive:
  full  = sqrt(misspec_var + epistemic_var)   (return_std)
  epi   = sqrt(epistemic_var)                 (return_epistemic_std, ordinary Bayes)
  mis   = sqrt(full^2 - epi^2)                (the POPS misspecification box alone)
and report rms-z of each vs the noiseless labels. The question: as N_train grows
(epi -> small), does the misspecification box (mis) stay ~1 on its own?

No ACE MAP fit needed. ~30 s.  Run:  uv run spike/pops_regime.py
"""
import os, pathlib
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import equinox as eqx
from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.inducing import GPConfig
from ace_jax.fit.rows import linear_rows
from popsregression import POPSRegression

D = pathlib.Path(os.path.expanduser("~/acegp-data/sige"))
model, meta, z = load(D / "sige_embed_d16.npz")
keys = dict(energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial")
configs = load_configs(str(D / "sige_mh1.xyz"), **keys)
els = [int(e) for e in meta["elements"]]
counts = np.array([[np.sum(c.numbers == e) for e in els] for c in configs], float)
E = np.array([c.energy for c in configs]); nat = np.array([len(c.numbers) for c in configs])
E0, *_ = np.linalg.lstsq(counts, E, rcond=None)
model = eqx.tree_at(lambda m: m.E0, model, __import__("jax.numpy", fromlist=["asarray"]).asarray(E0))
ds = build_dataset(list(configs), meta, E0, 4)
cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=len(els), C=4)
with highest_precision():
    P = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        lin, _, _ = linear_rows(model, cfg, b)
        P.append(np.asarray(lin.E)[np.asarray(b.w_E) > 0])
phiE = np.concatenate(P)                       # (N, L)
y = E - counts @ E0                            # eV per config, noiseless
N = phiE.shape[0]

# PCA-reduce features to K so we can reach the WELL-DETERMINED regime (N_train > K)
K = 30
Uc = phiE - phiE.mean(0)
_, _, Vt = np.linalg.svd(Uc, full_matrices=False)
Phi = Uc @ Vt[:K].T                            # (N, K) reduced ACE energy design

def rmsz(yv, mu, s): return float(np.sqrt(np.mean(((yv - mu) / s) ** 2)))
rng = np.random.default_rng(0)
print(f"ACE SiGe energy design PCA-reduced to K={K} features; sweep N_train (noiseless labels).")
print("reference popsregression, hypercube.  rms-z (ideal 1); mis = misspec box ALONE.\n")
print(f"{'N_train':>7} {'regime':>14} | {'rms-z epi':>9} {'rms-z mis':>9} {'rms-z full':>10}")
for ntr in (20, 40, 80, 150, 300):
    if ntr >= N: continue
    pi = rng.permutation(N); tr, te = pi[:ntr], pi[ntr:]
    m = POPSRegression(posterior="hypercube", fit_intercept=False,
                       leverage_percentile=0.0, percentile_clipping=0.0).fit(Phi[tr], y[tr])
    mu, full = m.predict(Phi[te], return_std=True)
    _, epi = m.predict(Phi[te], return_epistemic_std=True)
    mis = np.sqrt(np.maximum(full**2 - epi**2, 1e-30))     # misspecification box alone
    regime = "underdet." if ntr < K else ("~critical" if ntr < 2*K else "well-determined")
    print(f"{ntr:>7} {regime:>14} | {rmsz(y[te], mu, epi):>9.2f} {rmsz(y[te], mu, mis):>9.2f} {rmsz(y[te], mu, full):>10.2f}")

print("\nRead: if 'rms-z mis' stays ~1 as N_train grows (epi rms-z blows up, i.e. epi")
print("shrinks), the POPS misspecification box self-calibrates in the determined regime")
print("-> production needs NO floor and NO epistemic crutch. If 'mis' also blows up with")
print("N, the box does not carry it and a floor (or richer features) is genuinely needed.")
