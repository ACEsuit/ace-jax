"""Validation errors of exported models (model.npz, plain or with a frozen
sqrt-density term) on the run.py split: energy RMSE/MAE in meV/atom and force
RMSE/MAE in meV/Å, evaluated with the model itself (energy_forces_virial), so
it measures exactly what qoi.py and MD will use.

    uv run python bench/learn_radial/rmse_npz.py --model BASE.npz --data D.xyz \
        --model-npz runs/x/model.npz [--model-npz ...] --out rmse_npz.json

--model (the run's input model) only fixes the element order and E0 for the
dataset, exactly as run.py builds it.
"""
import argparse
import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--energy-key", default="energy"); p.add_argument("--force-key", default="forces")
p.add_argument("--virial-key", default="virial")
p.add_argument("--ntrain", type=int, default=200); p.add_argument("--nval", type=int, default=200)
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--model-npz", action="append", required=True); p.add_argument("--out", required=True)
a = p.parse_args()

_, meta, z = load(a.model)
NZ = len(meta["elements"])
configs = load_configs(a.data, a.energy_key, a.force_key, a.virial_key)
perm = np.random.default_rng(a.seed).permutation(len(configs))
E0 = np.asarray(z["E0"]) if "E0" in z else np.zeros(NZ)
ds = build_dataset([configs[i] for i in perm[a.ntrain:a.ntrain + a.nval]], meta, E0, configs_per_batch=a.batch)


@jax.jit
def efv(m, b):
    """Per-config energies (C,) and per-node forces (Ncap, 3) of one padded
    batch, from a single site_energies_dense call (its value and its
    gradient wrt rij, via has_aux) -- no need for a second forward pass to
    split the per-node energies into per-config sums."""
    Ncap, K = b.nbr.shape
    C = b.y_E.shape[0]
    zi = jnp.broadcast_to(b.node_z[:, None], (Ncap, K))

    def total(r):
        e = m.site_energies_dense(r, zi, b.node_z[b.nbr], b.nbr_mask, b.node_z)
        return jnp.sum(e), e

    (_, e), g_r = jax.value_and_grad(total, has_aux=True)(b.rij)
    g_r = jnp.where(b.nbr_mask[..., None], g_r, 0.0)
    F = (jnp.zeros((Ncap, 3), b.rij.dtype).at[jnp.arange(Ncap)].add(g_r.sum(axis=1))
         .at[b.nbr.reshape(-1)].add(-g_r.reshape(-1, 3)))
    return jax.ops.segment_sum(e, b.node_cfg, num_segments=C + 1)[:C], F


res = {}
for path in a.model_npz:
    m, _, _ = load(path)
    dE, dF = [], []
    with highest_precision():
        for i in range(ds.n_batches):
            b = jax.tree.map(lambda x: x[i], ds)
            Ec, F = (np.asarray(x) for x in efv(m, b))
            E0sum = np.asarray(jax.ops.segment_sum(jnp.where(b.node_mask, jnp.asarray(E0)[b.node_z], 0.0),
                                                   b.node_cfg, num_segments=b.y_E.shape[0] + 1)[:b.y_E.shape[0]])
            live_E = np.asarray(b.cfg_mask) & (np.asarray(b.w_E) > 0)
            n = np.maximum(np.asarray(b.n_atoms), 1)
            dE.append(((Ec - E0sum - np.asarray(b.y_E)) / n)[live_E])
            live_F = np.asarray(b.node_mask) & (np.asarray(b.w_F) > 0)
            dF.append((F - np.asarray(b.y_F))[live_F].ravel())
    dE, dF = np.concatenate(dE), np.concatenate(dF)
    res[str(path)] = {"E_rmse_meV_atom": float(1e3 * np.sqrt(np.mean(dE ** 2))),
                      "E_mae_meV_atom": float(1e3 * np.mean(np.abs(dE))),
                      "F_rmse_meV_A": float(1e3 * np.sqrt(np.mean(dF ** 2))),
                      "F_mae_meV_A": float(1e3 * np.mean(np.abs(dF)))}
    print(f"{path}: E {res[str(path)]['E_rmse_meV_atom']:.3f} meV/atom  F {res[str(path)]['F_rmse_meV_A']:.2f} meV/A",
          flush=True)
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
