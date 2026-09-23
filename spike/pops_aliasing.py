"""Is the 'aleatoric' floor genuine irreducible scatter (descriptor aliasing) or
an artifact of undersized POPS? POPS is designed to need NO noise floor, so if the
floor is real it must be error the ACE FEATURES cannot resolve: two configs with
(near-)identical linear energy features phi_E but different labels E/atom.

We build the intensive energy descriptor x = phi_E / nat for every config (this is
exactly what the linear energy model sees: E_pred/nat = x . c), and ask, for the
nearest-neighbour pairs in x, how big the label gap |Delta(E/atom)| is. If the gap
stays ~ the model RMSE (0.97 meV/atom) as ||Delta x|| -> 0, the features alias
(irreducible, a floor is legitimate). If the gap -> 0 with distance, there is no
aliasing and the floor points to an undersized POPS term instead.

No MAP fit needed (design rows + labels only). ~30 s.
Run:  uv run python spike/pops_aliasing.py
"""
import os, pathlib
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.inducing import GPConfig
from ace_jax.fit.rows import linear_rows

D = pathlib.Path(os.path.expanduser("~/acegp-data/sige"))
model, meta, z = load(D / "sige_embed_d16.npz")
keys = dict(energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial")
configs = load_configs(str(D / "sige_mh1.xyz"), **keys)
els = [int(e) for e in meta["elements"]]
counts = np.array([[np.sum(c.numbers == e) for e in els] for c in configs], float)
E = np.array([c.energy for c in configs]); nat = np.array([len(c.numbers) for c in configs])
E0, *_ = np.linalg.lstsq(counts, E, rcond=None)          # per-species E0 (removed by the model)
model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
ds = build_dataset(list(configs), meta, E0, 4)
cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=len(els), C=4)

# energy design rows phi_E (one L-vector per config), assembled batch by batch
with highest_precision():
    phiE = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        lin, _, _ = linear_rows(model, cfg, b)
        m = np.asarray(b.w_E) > 0                         # drop padded configs
        phiE.append(np.asarray(lin.E)[m])
phiE = np.concatenate(phiE)                               # (Ncfg, L)
assert phiE.shape[0] == len(configs), (phiE.shape, len(configs))

# intensive descriptor the linear energy model sees, and the label it must predict
x = phiE / nat[:, None]                                   # (N, L)  E_pred/nat = x . c
epa = 1e3 * (E - counts @ E0) / nat                       # meV/atom, E0 removed (what the model fits)

# standardise features (unit variance) and reduce to the informative subspace so
# nearest-neighbour distances are meaningful (L ~ 1100 >> N ~ 350)
xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
U, S, Vt = np.linalg.svd(xs - xs.mean(0), full_matrices=False)
k = int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.99) + 1)   # 99% variance
xr = (xs @ Vt[:k].T)                                      # (N, k)
xr /= xr.std(0) + 1e-12                                   # whiten the kept axes

N = xr.shape[0]
d2 = np.sum((xr[:, None, :] - xr[None, :, :]) ** 2, axis=2)
np.fill_diagonal(d2, np.inf)
dist = np.sqrt(d2)
nn = np.argmin(dist, axis=1)                              # nearest neighbour of each config
nnd = dist[np.arange(N), nn]
dlabel = np.abs(epa - epa[nn])                            # |Delta(E/atom)| to the NN, meV/atom

print(f"SiGe: {N} configs, {phiE.shape[1]} energy features -> {k} PCA axes (99% var)")
print(f"model energy RMSE (this linear base) ~ 0.97 meV/atom (from the fit); label spread = {epa.std():.1f} meV/atom\n")

order = np.argsort(nnd)
print("closest 10 config pairs in feature space:")
print(f"  {'||dx|| (whitened)':>18}  {'|dE/atom| meV':>13}")
for i in order[:10]:
    print(f"  {nnd[i]:>18.3f}  {dlabel[i]:>13.2f}")

# binned: does the label gap shrink as feature distance -> 0?
print("\n|dE/atom| to nearest neighbour, binned by feature distance (closest first):")
qs = np.quantile(nnd, [0, .2, .4, .6, .8, 1.0])
for a, b in zip(qs[:-1], qs[1:]):
    m = (nnd >= a) & (nnd <= b)
    print(f"  ||dx|| in [{a:5.2f},{b:5.2f}]  n={m.sum():3d}   median |dE/atom| = {np.median(dlabel[m]):5.2f} meV")
r = np.corrcoef(nnd, dlabel)[0, 1]
print(f"\ncorr(||dx||, |dE/atom|) = {r:.2f}")
print("Read: if the closest bin's |dE/atom| stays ~ the model RMSE (does NOT fall to 0),")
print("the ACE energy features ALIAS -> the floor is genuine irreducible scatter, and a")
print("noise-free POPS is simply the wrong tool for a representation-limited surrogate.")
print("If |dE/atom| -> 0 with distance, there is no aliasing and the floor is an artifact.")


# --- decisive: does the REFERENCE POPS calibrate on the ACE energy design WITHOUT
# a noise floor? (single-quantity, no whitening -- isolates the POPS term itself) ---
from popsregression import POPSRegression
y = (E - counts @ E0)                                    # eV per config (model target)
rng = np.random.default_rng(0); pi = rng.permutation(N)
tr, te = pi[:N//2], pi[N//2:]
def rmsz(yv, mu, s): return float(np.sqrt(np.mean(((yv-mu)/s)**2)))
m = POPSRegression(posterior="hypercube", fit_intercept=False,
                   leverage_percentile=0.0, percentile_clipping=0.0).fit(phiE[tr], y[tr])
mu, std = m.predict(phiE[te], return_std=True)           # POPS predictive, NO explicit noise floor
_, epi = m.predict(phiE[te], return_epistemic_std=True)  # ordinary Bayes only
natte = nat[te]
print("\n--- reference POPSRegression on the ACE ENERGY design (single quantity, no floor) ---")
print(f"    test RMSE            = {1e3*np.sqrt(np.mean(((y[te]-mu)/natte)**2)):.2f} meV/atom")
print(f"    rms-z, Bayes only    = {rmsz(y[te], mu, epi):.2f}   (no misspecification term)")
print(f"    rms-z, POPS (return_std, hypercube, NO floor) = {rmsz(y[te], mu, std):.2f}")
print("    => if this is ~1, the reference POPS self-calibrates with NO noise floor on")
print("       the ACE features -> our port's 93%-aleatoric need is a port/assembly mismatch,")
print("       not irreducible physics. If >>1, the floor is real for this design.")
