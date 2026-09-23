"""Try the fix: add the linear posterior variance phi.Sigma0.phi (pops_epistemic)
to the POPS predictive, matching the reference popsregression (misspec + Bayes).
Does the SiGe 'aleatoric floor' need drop out?

Reports rms-z for hypercube POPS over the epistemic x aleatoric grid, E and F.
~4 min f64 CPU.  Run:  uv run python spike/pops_epistemic_fix.py
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
from ace_jax.fit.objective import Problem, make_lml
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

yE = np.array([c.energy for c in test]); yF = np.concatenate([c.forces for c in test]).reshape(-1)
def rz(y, mu, v): return float(np.sqrt(np.mean(((y - mu) / np.sqrt(v)) ** 2)))

with highest_precision():
    X, S = site_features(model, cfg, ds_train)
    ind = select_inducing(X, S, ds_train.node_z, ds_train.node_mask, 0, descriptor_scale(X, ds_train.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]), default_prior(R0))
    lik = make_lml(prob, ds_train, cache_linear=True); jax.block_until_ready(lik(to_array(prob.prior.mu)))
    theta = run_map(lik, prob.prior, steps=MAP_STEPS, lr=0.02, seed=SEED)
    grid = {}
    for epi in (False, True):
        for alea in (False, True):
            p = predict_fixed(theta, prob, ds_train, ds_test, uq="pops", pops_form="hypercube",
                              pops_epistemic=epi, aleatoric=alea)
            grid[(epi, alea)] = (rz(yE, np.asarray(p.E_mean), np.asarray(p.E_var)),
                                 rz(yF, np.asarray(p.F_mean).reshape(-1), np.asarray(p.F_var).reshape(-1)))

print("\nSiGe hypercube POPS rms-z (ideal 1):")
print(f"    {'':22} {'E':>7} {'F':>7}")
print(f"    box only (old ens-era)  {grid[(False,False)][0]:>7.2f} {grid[(False,False)][1]:>7.2f}")
print(f"    box + aleatoric (old default) {grid[(False,True)][0]:>7.2f} {grid[(False,True)][1]:>7.2f}")
print(f"    box + epistemic (THE FIX)     {grid[(True,False)][0]:>7.2f} {grid[(True,False)][1]:>7.2f}")
print(f"    box + epistemic + aleatoric   {grid[(True,True)][0]:>7.2f} {grid[(True,True)][1]:>7.2f}")
print("\nNote: SiGe is under-determined (1110 params, 200 configs), so epistemic is LARGE")
print("here; production (>>params configs) will lean on the misspecification box instead.")
