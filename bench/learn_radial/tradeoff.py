"""Energy/force trade-off curve of a linear ACE fit at fixed radials, cheaply.

For each candidate radial set W the per-quantity fit statistics (G_q, b_q for
q in E, F, V) are streamed once, and the validation design rows are cached
once (energies per atom, force components). Every point on the curve is then
a re-solve of the L x L normal equations with a different energy noise level,
followed by a matrix product for the validation errors -- no further passes
over the data. The other hyperparameters (sigma_F, sigma_V, sigma_c) stay at
the candidate's theta-MAP, obtained as in the held-out gate. sigma_E is scaled
by each multiplier: a larger multiplier weights energies less, i.e. is more
force-heavy. Compares directly with a pacemaker kappa sweep (E/F RMSE on the
same split).

    uv run python bench/learn_radial/tradeoff.py --model M.npz --data D.xyz --n-q 12 \
        --cand init --cand runs/nq12/sige/lam_0.1/rnl_Wnlq.npy --out tradeoff.json
"""
import argparse
import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ace_jax.basis.prior import prior_diagonal
from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import default_prior, from_array
from ace_jax.fit.ladder import run_map
from ace_jax.fit.objective import log_marginal_likelihood
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem
from ace_jax.fit.radial_learn import theta_map_linear
from ace_jax.fit.radial_model import normalise, radial_gram, row_active, to_analytic, with_radial
from ace_jax.fit.rows import linear_rows
from ace_jax.fit.stats import linear_statistics

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--energy-key", default="energy"); p.add_argument("--force-key", default="forces")
p.add_argument("--virial-key", default="virial")
p.add_argument("--ntrain", type=int, default=200); p.add_argument("--nval", type=int, default=200)
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--r0", type=float, default=2.35)
p.add_argument("--n-q", type=int, default=12, help="must match the run that produced the candidates")
p.add_argument("--map-steps", type=int, default=300)
p.add_argument("--mult", default="0.03,0.1,0.3,1,3,10,30,100,300",
               help="multipliers on the MAP sigma_E (larger = weight energies less)")
p.add_argument("--gamma-powers", default=None,
               help="instead of the sigma_E sweep: prior-shape sweep, Gamma -> Gamma**alpha for each alpha "
                    "(0 = uniform ridge, 1 = the model's prior), sigmas re-MAP'd on the cached statistics")
p.add_argument("--cand", action="append", required=True,
               help='"init" or a path to a learned rnl_Wnlq.npy (repeatable); label = its parent dir')
p.add_argument("--out", required=True)
a = p.parse_args()

model, meta, z = load(a.model)
model, _ = to_analytic(model, a.n_q)
NZ = len(meta["elements"])
configs = load_configs(a.data, a.energy_key, a.force_key, a.virial_key)
perm = np.random.default_rng(a.seed).permutation(len(configs))
E0 = np.asarray(z["E0"]) if "E0" in z else np.zeros(NZ)
ds_fit = build_dataset([configs[i] for i in perm[:a.ntrain]], meta, E0, configs_per_batch=a.batch)
ds_val = build_dataset([configs[i] for i in perm[a.ntrain:a.ntrain + a.nval]], meta, E0,
                       configs_per_batch=a.batch)
cfg = GPConfig(r0=a.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=NZ, C=a.batch)
X, S = site_features(model, cfg, ds_fit)
ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
gamma = np.asarray(prior_diagonal(z, meta, source=a.model))
prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(gamma), default_prior(a.r0))
W0 = model.rnl_Wnlq
W_init = normalise(W0, radial_gram(model, ds_fit), row_active(W0))
a0 = theta_map_linear(prob, ds_fit, W_init, steps=a.map_steps)


@jax.jit
def rows(m, b):
    r, _, _ = linear_rows(m, cfg, b)
    return r.E, r.F


def val_rows(m):
    """Cached validation design: per-atom energy rows/targets and force-component rows/targets."""
    PE, yE, PF, yF = [], [], [], []
    for i in range(ds_val.n_batches):
        b = jax.tree.map(lambda x: x[i], ds_val)
        rE, rF = (np.asarray(x) for x in rows(m, b))
        live_E = np.asarray(b.cfg_mask) & (np.asarray(b.w_E) > 0)
        n = np.maximum(np.asarray(b.n_atoms), 1)[:, None]
        PE.append((rE / n)[live_E]); yE.append((np.asarray(b.y_E)[:, None] / n)[live_E, 0])
        live_F = np.asarray(b.node_mask) & (np.asarray(b.w_F) > 0)
        PF.append(rF[live_F].reshape(-1, rF.shape[-1])); yF.append(np.asarray(b.y_F)[live_F].ravel())
    return np.concatenate(PE), np.concatenate(yE), np.concatenate(PF), np.concatenate(yF)


mults = [float(x) for x in a.mult.split(",")]
res = {"mult": mults, "gamma_powers": a.gamma_powers}
for cand in a.cand:
    label, W = ("init", W_init) if cand == "init" else (pathlib.Path(cand).parent.name, jnp.asarray(np.load(cand)))
    th = from_array(theta_map_linear(prob, ds_fit, W, steps=a.map_steps, init=a0))
    m = with_radial(model, W)
    st = jax.tree.map(np.asarray, linear_statistics(m, cfg, ds_fit))
    PE, yE, PF, yF = val_rows(m)
    if a.gamma_powers:
        pts = []
        stj = jax.tree.map(jnp.asarray, st)
        for alpha in (float(x) for x in a.gamma_powers.split(",")):
            g_a = gamma ** alpha
            prob_a = prob._replace(gamma=jnp.asarray(g_a))
            lml = jax.jit(lambda v, pa=prob_a: log_marginal_likelihood(from_array(v), stj, pa))
            # warm start: shift log sigma_c by the change in Gamma's geometric mean, so the prior
            # strength on a typical basis function is unchanged; then re-MAP on the cached Gram
            shift = float(np.mean(np.log(g_a)) - np.mean(np.log(gamma)))
            th0 = th._replace(log_sigma_c=th.log_sigma_c + shift)
            h = run_map(lml, prob.prior, steps=a.map_steps, init=th0)
            sE, sF, sV, sc = (float(np.exp(getattr(h, f"log_sigma_{k}"))) for k in "EFVc")
            G = st.G_E / sE ** 2 + st.G_F / sF ** 2 + st.G_V / sV ** 2 + np.diag(g_a ** 2 / sc ** 2)
            c = np.linalg.solve(G, st.b_E / sE ** 2 + st.b_F / sF ** 2 + st.b_V / sV ** 2)
            pts.append({"alpha": alpha, "sigma_c": sc, "sigma_E": sE, "sigma_F": sF,
                        "E_rmse_meV_atom": float(1e3 * np.sqrt(np.mean((PE @ c - yE) ** 2))),
                        "F_rmse_meV_A": float(1e3 * np.sqrt(np.mean((PF @ c - yF) ** 2)))})
            print(f"{label:14s} Gamma^{alpha:<5g} sigma_c {sc:.3e}  val E {pts[-1]['E_rmse_meV_atom']:7.3f} meV/atom"
                  f"  F {pts[-1]['F_rmse_meV_A']:7.2f} meV/A", flush=True)
        res[label] = {"points": pts}
        continue
    sE, sF, sV, sc = (float(np.exp(getattr(th, f"log_sigma_{k}"))) for k in "EFVc")
    lam = np.diag(gamma ** 2 / sc ** 2)
    pts = []
    for k in mults:
        G = st.G_E / (k * sE) ** 2 + st.G_F / sF ** 2 + st.G_V / sV ** 2 + lam
        bb = st.b_E / (k * sE) ** 2 + st.b_F / sF ** 2 + st.b_V / sV ** 2
        c = np.linalg.solve(G, bb)
        pts.append({"mult": k, "sigma_E": k * sE, "E_rmse_meV_atom": float(1e3 * np.sqrt(np.mean((PE @ c - yE) ** 2))),
                    "F_rmse_meV_A": float(1e3 * np.sqrt(np.mean((PF @ c - yF) ** 2)))})
        print(f"{label:14s} x{k:<6g} sigma_E {k * sE:.3e}  val E {pts[-1]['E_rmse_meV_atom']:7.3f} meV/atom"
              f"  F {pts[-1]['F_rmse_meV_A']:7.2f} meV/A", flush=True)
    res[label] = {"sigma_E_map": sE, "sigma_F": sF, "sigma_V": sV, "sigma_c": sc, "points": pts}
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
