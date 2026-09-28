"""Test the reconciliation (Swinburne & Perez, arXiv:2402.01810): POPS characterises
misspecification in the LOW-NOISE limit with Sigma_Y FIXED small (NOT evidence-fit).
The paper's coverage theorem: the pointwise-optimal ensemble ENVELOPE must bracket
every training observation as N/P -> inf.  Our pipeline instead evidence-fits
sigma_E/F/V ~ RMSE, which the paper says reverts to plain ML and defeats POPS.

Here we redo POPS the paper's way and sweep the ridge lambda -> 0 (can't be exactly
0; the Gram needs regularising), REUSING one eigendecomposition of the data Gram:
    A(lambda)^-1 = U diag(1/(Lambda+lambda)) U^T .
Fixed STRUCTURAL weights (w_E=1/sqrt(nat), w_F=1); no sigma_q.  Gram is
over-determined by forces.  Question: does the ensemble cover the ENERGY residuals
(rms-z -> 1, envelope coverage -> ~1) as lambda shrinks -- with NO noise floor?

~5 min f64 CPU.  Run:  uv run python spike/pops_lownoise.py
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
from ace_jax.fit.objective import Problem, make_lml, posterior
from ace_jax.fit.stats import sufficient_statistics
from ace_jax.fit.rows import linear_rows

D = pathlib.Path(os.path.expanduser("~/acegp-data/sige"))
R0, NTRAIN, TEST_START, NTEST, SEED, BATCH, MAP_STEPS = 2.35, 200, 250, 100, 0, 4, 300
model, meta, z = load(D / "sige_embed_d16.npz")
keys = dict(energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial")
configs = load_configs(str(D / "sige_mh1.xyz"), **keys)
perm = np.random.default_rng(SEED).permutation(len(configs))
train = [configs[i] for i in perm[:NTRAIN]]; test = [configs[i] for i in perm[TEST_START:TEST_START + NTEST]]
els = [int(e) for e in meta["elements"]]
countsTr = np.array([[np.sum(c.numbers == e) for e in els] for c in train], float)
E0, *_ = np.linalg.lstsq(countsTr, np.array([c.energy for c in train]), rcond=None)
model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
ds_train = build_dataset(train, meta, E0, BATCH); ds_test = build_dataset(test, meta, E0, BATCH)
cfg = GPConfig(r0=R0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=len(els), C=BATCH)

def collect(ds, cfgs):
    """physical-weighted energy+force rows, targets (E0-removed), and labels."""
    PE, WE, PF, WF = [], [], [], []
    with highest_precision():
        for i in range(ds.n_batches):
            b = jax.tree.map(lambda a: a[i], ds)
            lin, _, _ = linear_rows(model, cfg, b)
            me = np.asarray(b.w_E) > 0; mf = np.asarray(b.w_F) > 0
            L = lin.E.shape[-1]
            PE.append(np.asarray(lin.E)[me]); WE.append(np.asarray(b.w_E)[me])
            PF.append(np.asarray(lin.F)[mf].reshape(-1, L)); WF.append(np.repeat(np.asarray(b.w_F)[mf], 3))
    PE, WE, PF, WF = map(np.concatenate, (PE, WE, PF, WF))
    nat = np.array([len(c.numbers) for c in cfgs]); cnt = np.array([[np.sum(c.numbers == e) for e in els] for c in cfgs], float)
    tE = np.array([c.energy for c in cfgs]) - cnt @ E0          # E0-removed energy target
    tF = np.concatenate([c.forces for c in cfgs]).reshape(-1)
    return PE, WE, tE, nat, PF, WF, tF

with highest_precision():
    X, S = site_features(model, cfg, ds_train)
    ind = select_inducing(X, S, ds_train.node_z, ds_train.node_mask, 0, descriptor_scale(X, ds_train.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]), default_prior(R0))
    lik = make_lml(prob, ds_train, cache_linear=True); jax.block_until_ready(lik(to_array(prob.prior.mu)))
    theta = run_map(lik, prob.prior, steps=MAP_STEPS, lr=0.02, seed=SEED)
    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)
    cstar, _ = posterior(theta, st, prob)
cstar = np.asarray(cstar)

PEtr, WEtr, tEtr, natTr, PFtr, WFtr, tFtr = collect(ds_train, train)
PEte, WEte, tEte, natTe, PFte, WFte, tFte = collect(ds_test, test)
L = PEtr.shape[1]
# residuals at the loss-min fit (E0-removed): r = target - phi.cstar
rEtr = tEtr - PEtr @ cstar; rFtr = tFtr - PFtr @ cstar
rEte = tEte - PEte @ cstar; rFte = tFte - PFte @ cstar
print(f"SiGe M=0: L={L} params.  train rows: {len(rEtr)} E + {len(rFtr)} F  (over-determined by forces)")
print(f"train E RMSE {1e3*np.sqrt(np.mean((rEtr/natTr)**2)):.2f} meV/atom, F RMSE {np.sqrt(np.mean(rFtr**2)):.4f} eV/A")

# stacked physically-weighted design (Phi_w = w.phi) and residuals; POPS members = all train rows
Phw = np.vstack([WEtr[:, None] * PEtr, WFtr[:, None] * PFtr])   # (nrows, L)
rw = np.concatenate([WEtr * rEtr, WFtr * rFtr])                 # w.r  (whitened residual)
# ONE eigendecomposition of the data Gram, reused for every ridge lambda
M = Phw.T @ Phw
Lam, U = np.linalg.eigh(M)                                      # M = U diag(Lam) U^T
eta = Phw @ U                                                   # members projected to eigenbasis (nrows, L)
phiEte = PEte @ U                                               # raw energy test rows, eigenbasis
rng = np.random.default_rng(0); fsel = rng.choice(len(rFte), size=min(2000, len(rFte)), replace=False)
phiFte = (PFte[fsel]) @ U; rFte_s = rFte[fsel]                  # subsample of force test components

lam_scale = Lam.max()
print(f"\nGram eigenvalues: max {Lam.max():.2e} min {Lam.min():.2e} (cond {Lam.max()/max(Lam.min(),1e-30):.1e})")
print(f"\n{'lambda/lam_max':>14} | {'rmsz E':>7} {'cover E':>8} | {'rmsz F':>7} {'cover F':>8}   (ideal 1.0 / ~1.0; NO floor)")
def stats(phite, rte, per=None):
    A = phite * inv[None, :]; G = A @ eta.T; shift = G * c[None, :]
    if per is not None:
        shift = shift / per[:, None]; rte = rte / per
    var = shift.var(axis=1)
    rmsz = float(np.sqrt(np.mean(rte ** 2 / np.maximum(var, 1e-30))))
    lo, hi = shift.min(1), shift.max(1)
    return rmsz, float(np.mean((rte >= lo) & (rte <= hi)))
for frac in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7):
    lam = frac * lam_scale
    inv = 1.0 / (Lam + lam)                                     # reuse eigendecomp
    h = (eta ** 2) @ inv; c = rw / h
    rzE, cvE = stats(phiEte, rEte, per=natTe)
    rzF, cvF = stats(phiFte, rFte_s)
    print(f"{frac:>14.0e} | {rzE:>7.2f} {cvE:>8.2f} | {rzF:>7.2f} {cvF:>8.2f}")
print("\nRead: if rms-z E -> ~1 and envelope coverage -> ~1 as lambda shrinks, POPS in the")
print("low-noise limit covers the residuals with NO fitted noise -> reconciles the paper.")
print("If it stays overconfident even at lambda->0, our port diverges from the paper's ansatz.")
