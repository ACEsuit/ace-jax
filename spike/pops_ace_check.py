"""Direct check: WHY does our port's POPS 'ensemble' vs 'hypercube' differ ~5x on
the ACE (SiGe) design, when the reference package shows only a few-% difference on
a benign 1D problem (spike/pops_1d.py)?

We (A) reproduce the ensemble-vs-hypercube rms-z gap on the real SiGe linear fit,
then (B) diagnose it from the pointwise corrections `deltas` directly:

  hypercube_cov = support @ (diag((hi-lo)^2/12) + mbar mbar^T) @ support^T   (BOX)
  ensemble var  = phi* . Cov_committee(delta) . phi*                          (CENTERED)

So per PCA axis the two differ by two things:
  * box_var/centered_var   -- the uniform BOX over the full min..max range is much
                              wider than the committee variance when the projected
                              corrections are HEAVY-TAILED (a few high-leverage
                              points dominate). percentile_clipping=0 keeps the
                              full range.
  * mean term (mbar^2)     -- the systematic-bias term the docstring emphasised.
Whichever dominates on ACE tells us if the gap is regime-specific (heavy tails from
whitened multi-quantity leverage) or the small mean term.

Needs ~/acegp-data/sige/{sige_embed_d16.npz,sige_mh1.xyz}. ~3-4 min f64 CPU.
Run:  uv run python spike/pops_ace_check.py
"""
import os, pathlib
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from scipy.stats import kurtosis

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.ladder import run_map
from ace_jax.fit.objective import Problem, make_lml, posterior
from ace_jax.fit.stats import sufficient_statistics, pops_statistics
from ace_jax.fit.predict import predict_fixed

D = pathlib.Path(os.path.expanduser("~/acegp-data/sige"))
R0, NTRAIN, TEST_START, NTEST, SEED, BATCH, MAP_STEPS = 2.35, 200, 250, 100, 0, 4, 300

model, meta, z = load(D / "sige_embed_d16.npz")
keys = dict(energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial")
configs = load_configs(str(D / "sige_mh1.xyz"), **keys)
perm = np.random.default_rng(SEED).permutation(len(configs))
train = [configs[i] for i in perm[:NTRAIN]]
test = [configs[i] for i in perm[TEST_START:TEST_START + NTEST]]
els = [int(e) for e in meta["elements"]]
counts = np.array([[np.sum(c.numbers == e) for e in els] for c in train], float)
E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in train]), rcond=None)
model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
ds_train = build_dataset(train, meta, E0, BATCH)
ds_test = build_dataset(test, meta, E0, BATCH)
cfg = GPConfig(r0=R0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=len(els), C=BATCH)

with highest_precision():
    X, S = site_features(model, cfg, ds_train)
    scale = descriptor_scale(X, ds_train.node_mask)
    ind = select_inducing(X, S, ds_train.node_z, ds_train.node_mask, 0, scale)   # M=0 linear
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]), default_prior(R0))
    lik = make_lml(prob, ds_train, cache_linear=True)
    jax.block_until_ready(lik(to_array(prob.prior.mu)))
    theta = run_map(lik, prob.prior, steps=MAP_STEPS, lr=0.02, seed=SEED)

    # ---- (A) reproduce the ensemble-vs-hypercube rms-z gap (aleatoric OFF) ----
    def rmsz_E(p):
        y = np.array([c.energy for c in test]); mu = np.asarray(p.E_mean); s = np.sqrt(np.asarray(p.E_var))
        return float(np.sqrt(np.mean(((y - mu) / s) ** 2)))
    def rmsz_F(p):
        y = np.concatenate([c.forces for c in test]).reshape(-1)
        mu = np.asarray(p.F_mean).reshape(-1); s = np.sqrt(np.asarray(p.F_var)).reshape(-1)
        return float(np.sqrt(np.mean(((y - mu) / s) ** 2)))
    p_ens = predict_fixed(theta, prob, ds_train, ds_test, uq="pops", pops_form="ensemble", aleatoric=False)
    p_hc = predict_fixed(theta, prob, ds_train, ds_test, uq="pops", pops_form="hypercube", aleatoric=False)
    p_hc_a = predict_fixed(theta, prob, ds_train, ds_test, uq="pops", pops_form="hypercube", aleatoric=True)

    # ---- (B) diagnose from the pointwise corrections deltas ----
    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)
    c_star, cholA = posterior(theta, st, prob)
    from jax.scipy.linalg import cho_solve
    Sigma0 = cho_solve((cholA, True), jnp.eye(c_star.shape[0]))
    sigma = {q: float(np.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}
    deltas = np.asarray(pops_statistics(c_star, Sigma0, prob, ds_train, sigma, leverage_pct=0.0))

print(f"\nSiGe linear (M=0), {len(train)} train / {len(test)} test, {deltas.shape[0]} corrections x {deltas.shape[1]} params")
print("\n(A) predictive rms-z (ideal 1):")
print(f"    ensemble,  aleatoric OFF : E {rmsz_E(p_ens):6.2f}   F {rmsz_F(p_ens):6.2f}")
print(f"    hypercube, aleatoric OFF : E {rmsz_E(p_hc):6.2f}   F {rmsz_F(p_hc):6.2f}   <- misspecification-only")
print(f"    hypercube, aleatoric ON  : E {rmsz_E(p_hc_a):6.2f}   F {rmsz_F(p_hc_a):6.2f}   <- the default")
print(f"    hypercube/ensemble sigma-E ratio = {float(np.sqrt(np.asarray(p_hc.E_var)/np.asarray(p_ens.E_var)).mean()):.1f}x")
# how much of the default predictive variance is the aleatoric floor?
for q in ("E", "F"):
    v_off = np.asarray(getattr(p_hc, f"{q}_var")).reshape(-1)
    v_on = np.asarray(getattr(p_hc_a, f"{q}_var")).reshape(-1)
    frac = float(np.mean((v_on - v_off) / v_on))
    print(f"    {q}: aleatoric is {frac:.0%} of the hypercube+aleatoric predictive variance (mean)")

# PCA of the corrections; per-axis centered var, box var, mean term
G = deltas.T @ deltas
evals, evecs = np.linalg.eigh(G)
keep = evals > 1e-8 * evals.max()
supp = evecs[:, keep]                    # (L, d)
proj = deltas @ supp                     # (K, d)
cvar = proj.var(0)                       # centered committee variance per axis
lo, hi = proj.min(0), proj.max(0)
bvar = (hi - lo) ** 2 / 12.0             # uniform-box variance per axis (percentile_clipping=0)
msq = proj.mean(0) ** 2                  # mean (systematic) term per axis
kurt = kurtosis(proj, axis=0, fisher=False)   # 3 = Gaussian; >>3 = heavy tails

print(f"\n(B) corrections in PCA basis ({proj.shape[1]} kept axes):")
print(f"    sum centered var (ensemble)      = {cvar.sum():.4g}")
print(f"    sum box var      (hypercube box) = {bvar.sum():.4g}   -> box/centered = {bvar.sum()/cvar.sum():.1f}x")
print(f"    sum mean-term    (mbar^2)        = {msq.sum():.4g}   -> mean-term fraction = {msq.sum()/(cvar.sum()+msq.sum()):.3f}")
print(f"    projected-correction kurtosis    = median {np.median(kurt):.1f}, max {kurt.max():.1f}  (3=Gaussian; >>3 heavy-tailed)")
frac_top = np.sort(np.linalg.norm(deltas, axis=1))[::-1][:5].sum() / np.linalg.norm(deltas, axis=1).sum()
print(f"    top-5 of {deltas.shape[0]} corrections carry {frac_top:.1%} of total ||delta||")
print("\nVERDICT: if box/centered >> 1 with small mean-term fraction and high kurtosis,")
print("the ACE gap is HEAVY-TAILED corrections (a few high-leverage points) widening the")
print("uniform box -- regime-specific and by-design, NOT the centred-vs-uncentred mean term.")
