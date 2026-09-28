"""Probe reason (1a): the anisotropic smoothness prior variance sigma_c^2/Gamma^2
starves the null-space directions, so the BLR epistemic variance is too tight
(overconfident, rms-z ~3.7 on SiGe energy). Add an ISOTROPIC prior-variance floor
eps^2 -- prior var per weight = sigma_c^2/Gamma^2 + eps^2 -- and see if the
epistemic-only calibration improves.

This is a CHEAP probe on the EXISTING MAP fit: reconstruct the data Gram G = A - Lam
once, then sweep eps^2 in numpy (mean held at the baseline; only the variance's
prior changes). A proper confirmation re-optimises MAP with eps as a hyper (see
notes). Laplace is NOT the fix here (already tried; ~MAP), consistent with the
hypers being pinned and the problem being the prior SHAPE.

The fitted sigma_E/sigma_F are kept so we also report the +noise (label-predictive)
rms-z, not just epistemic-only. ~4 min f64 CPU.
Run:  uv run python spike/pops_isoprior.py
"""
import os, pathlib
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.ladder import run_map
from ace_jax.fit.objective import Problem, make_lml, posterior, prior_precision
from ace_jax.fit.stats import sufficient_statistics
from ace_jax.fit.rows import linear_rows
from ace_jax.fit.predict import predict_fixed

D = pathlib.Path(os.path.expanduser("~/acegp-data/sige"))
R0, NTRAIN, TEST_START, NTEST, SEED, BATCH, MAP_STEPS = 2.35, 200, 250, 100, 0, 4, 300
model, meta, z = load(D / "sige_embed_d16.npz")
keys = dict(energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial")
configs = load_configs(str(D / "sige_mh1.xyz"), **keys)
perm = np.random.default_rng(SEED).permutation(len(configs))
train = [configs[i] for i in perm[:NTRAIN]]; test = [configs[i] for i in perm[TEST_START:TEST_START + NTEST]]
els = [int(e) for e in meta["elements"]]
counts = np.array([[np.sum(c.numbers == e) for e in els] for c in train], float)
E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in train]), rcond=None)
model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
ds_train = build_dataset(train, meta, E0, BATCH); ds_test = build_dataset(test, meta, E0, BATCH)
cfg = GPConfig(r0=R0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=len(els), C=BATCH)
natte = np.array([len(c.numbers) for c in test])
yE = np.array([c.energy for c in test]); yF = np.concatenate([c.forces for c in test]).reshape(-1)

with highest_precision():
    X, S = site_features(model, cfg, ds_train)
    ind = select_inducing(X, S, ds_train.node_z, ds_train.node_mask, 0, descriptor_scale(X, ds_train.node_mask))  # M=0
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]), default_prior(R0))
    lik = make_lml(prob, ds_train, cache_linear=True); jax.block_until_ready(lik(to_array(prob.prior.mu)))
    theta = run_map(lik, prob.prior, steps=MAP_STEPS, lr=0.02, seed=SEED)

    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)
    mu, cholA = posterior(theta, st, prob)
    Lam, _ = prior_precision(theta, prob)
    A = np.asarray(cholA) @ np.asarray(cholA).T
    G = A - np.asarray(Lam)                          # noise-scaled data Gram (M=0)
    sc2 = float(np.exp(2 * theta.log_sigma_c)); gam2 = np.asarray(prob.gamma) ** 2
    v0 = sc2 / gam2                                  # baseline prior variance per weight
    # test linear rows
    XE, XF = [], []
    for i in range(ds_test.n_batches):
        b = jax.tree.map(lambda a: a[i], ds_test)
        lin, _, _ = linear_rows(prob.model, prob.cfg, b)
        me = np.asarray(b.w_E) > 0; mf = np.asarray(b.w_F) > 0
        XE.append(np.asarray(lin.E)[me]); XF.append(np.asarray(lin.F)[mf].reshape(-1, lin.E.shape[-1]))
    XE = np.concatenate(XE); XF = np.concatenate(XF)
    pblr = predict_fixed(theta, prob, ds_train, ds_test)     # baseline mean (E0 handled)
sE, sF = float(np.exp(theta.log_sigma_E)), float(np.exp(theta.log_sigma_F))
Em, Fm = np.asarray(pblr.E_mean), np.asarray(pblr.F_mean).reshape(-1)

def rmsz(y, m, var): return float(np.sqrt(np.mean((y - m) ** 2 / var)))
print(f"SiGe BLR epistemic calibration vs isotropic prior-variance floor eps^2.")
print(f"baseline prior var sigma_c^2/Gamma^2: min {v0.min():.2e}  median {np.median(v0):.2e}  max {v0.max():.2e}")
print(f"fitted sigma_E {sE*1e3:.2f} meV/atom-ish, sigma_F {sF:.4f} eV/A\n")
print(f"{'eps^2/sigma_c^2':>15} | {'rmsz E (epi)':>12} {'rmsz F (epi)':>12} | {'rmsz E (+noise)':>15} {'rmsz F (+noise)':>15}")
for frac in (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
    eps2 = frac * sc2
    lin = 1.0 / (v0 + eps2)                          # floored prior precision
    Ae = G + np.diag(lin)
    S0 = np.linalg.inv(Ae)
    vE = np.einsum("ij,jk,ik->i", XE, S0, XE)        # per-config energy epistemic var (eV^2)
    vF = np.einsum("ij,jk,ik->i", XF, S0, XF)        # per-component force epistemic var
    rE, rF = rmsz(yE, Em, vE), rmsz(yF, Fm, vF)
    rEn, rFn = rmsz(yE, Em, vE + sE**2 * natte), rmsz(yF, Fm, vF + sF**2)
    print(f"{frac:>15g} | {rE:>12.2f} {rF:>12.2f} | {rEn:>15.2f} {rFn:>15.2f}")
print("\nRead: if 'rmsz E (epi)' falls toward 1 as eps^2 grows, the anisotropic prior")
print("shape WAS starving the null-space variance -> add eps as a hyper and re-opt MAP.")
print("Forces stay high (out-of-span error) regardless -> they still need sigma_F.")
