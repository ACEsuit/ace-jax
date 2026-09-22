"""python -m acegp.cli: fit the hybrid GP on an extxyz dataset with an exported
ACE model, run the requested rungs of the ladder, write draws and the metrics
table (per atom energies in meV/atom, forces in eV/A, virials in eV)."""
import argparse
import csv
import json
import pathlib

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh

from .eval import highest_precision, load
from .construct.prior import gamma_from_model
from .fit.data import build_dataset, load_configs
from .fit.hypers import Hypers, default_prior, to_array
from .fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from .fit.kernels import KernelSpec
from .fit.ladder import run_laplace, run_laplace_fd, run_map, run_nuts, run_pathfinder, run_vi
from .fit.metrics import summarise
from .fit.objective import Problem, make_log_density
from .fit.predict import predict_mixture


def _pad_to_multiple(ds, n):
    """Append copies of the last batch, weights, cfg_mask, node_mask and
    nbr_mask all zeroed/False, so ds.n_batches is a multiple of n (Task 15:
    sufficient_statistics_sharded requires n_batches % n_devices == 0).
    Zeroing node_mask/nbr_mask too (matching build_dataset's own padding
    convention) makes the padded batch inert to every consumer -- not just
    the weighted statistics, but also anything reading node_mask, such as
    site_features/select_inducing/descriptor_scale, should this helper ever
    be applied before those run (fix round 1: cli.py now calls it only after,
    on a copy handed solely to make_log_density, but the invariant should
    hold regardless of call site)."""
    k = (-ds.n_batches) % n
    if k == 0:
        return ds
    last = jax.tree.map(lambda a: a[-1:], ds)
    last = last._replace(w_E=jnp.zeros_like(last.w_E), w_F=jnp.zeros_like(last.w_F),
                         w_V=jnp.zeros_like(last.w_V), cfg_mask=jnp.zeros_like(last.cfg_mask),
                         node_mask=jnp.zeros_like(last.node_mask),
                         nbr_mask=jnp.zeros_like(last.nbr_mask))
    pad = jax.tree.map(lambda a: jnp.concatenate([a] * k, axis=0), last)
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), ds, pad)


def _add_fit_args(p):
    p.add_argument("--model", required=True); p.add_argument("--train", required=True)
    p.add_argument("--test"); p.add_argument("--energy-key", default="energy")
    p.add_argument("--force-key", default="forces"); p.add_argument("--virial-key", default="virial")
    p.add_argument("--weights", default=None, help="JSON, ACEfit weights dict")
    p.add_argument("--configs-per-batch", type=int, default=8)
    p.add_argument("--m-per-species", type=int, default=500)
    p.add_argument("--kernel", default="cosine", choices=["cosine", "matern32"])
    p.add_argument("--no-bump", action="store_true"); p.add_argument("--objective", default="lml", choices=["lml", "loo"])
    p.add_argument("--rungs", default="map,laplace"); p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--map-steps", type=int, default=500); p.add_argument("--vi-steps", type=int, default=2000)
    p.add_argument("--nuts-warmup", type=int, default=500); p.add_argument("--nuts-samples", type=int, default=500)
    p.add_argument("--nuts-chains", type=int, default=4); p.add_argument("--r0", type=float, required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
    p.add_argument("--devices", type=int, default=1,
                   help="shard sufficient statistics over this many devices (Task 15, stretch)")
    return p


def _metrics(pred, test_configs, ds_test):
    nat = np.array([len(c.numbers) for c in test_configs])
    # E rows are one per config, F rows one per atom in config order: keep only
    # the configs that carry the observation (mirrors the virial has_v below).
    has_e = np.array([c.energy is not None for c in test_configs])
    has_f = np.array([c.forces is not None for c in test_configs])
    yE = np.array([c.energy for c, h in zip(test_configs, has_e) if h])
    E_mean, E_var, nat_e = pred.E_mean[has_e], pred.E_var[has_e], nat[has_e]
    yF = np.concatenate([c.forces for c, h in zip(test_configs, has_f) if h]).reshape(-1, 3)
    f_rows = np.repeat(has_f, nat)
    F_mean, F_var = np.asarray(pred.F_mean)[f_rows], np.asarray(pred.F_var)[f_rows]
    # yV/pred.V_* are one row per real (cfg_mask-true) config, in dataset order,
    # which is test_configs order.  configs[0] (the isolated atom) has no
    # virial, so its y_V row is a placeholder zero -- build yV only from
    # configs that actually carry a virial, and drop the matching rows from
    # pred.V_mean/V_var so the two stay aligned (brief, Task 14 context).
    has_v = np.array([c.virial is not None for c in test_configs])
    yV = np.asarray(ds_test.y_V).reshape(-1, 6)[np.asarray(ds_test.cfg_mask).reshape(-1)][has_v]
    V_mean = np.asarray(pred.V_mean)[has_v]
    V_var = np.asarray(pred.V_var)[has_v]
    out = {"E": summarise(1e3 * yE / nat_e, 1e3 * E_mean / nat_e, 1e3 * np.sqrt(E_var) / nat_e),
           "F": summarise(yF.reshape(-1), F_mean.reshape(-1), np.sqrt(F_var).reshape(-1)),
           "V": summarise(yV.reshape(-1), V_mean.reshape(-1), np.sqrt(V_var).reshape(-1))}
    return out


def run(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, meta, z = load(a.model)
    weights = json.loads(a.weights) if a.weights else None
    keys = dict(energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key, weights=weights)
    train_cfgs = load_configs(a.train, **keys)
    test_cfgs = load_configs(a.test, **keys) if a.test else train_cfgs
    E0 = np.asarray(z["E0"])
    C = a.configs_per_batch
    ds_train = build_dataset(train_cfgs, meta, E0, C)
    ds_test = build_dataset(test_cfgs, meta, E0, C)
    cfg = GPConfig(r0=a.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=C)
    mesh = None
    if a.devices > 1:
        devices = jax.devices()[:a.devices]
        if len(devices) != a.devices:
            raise ValueError(f"--devices {a.devices} requested, only {len(devices)} available")
        mesh = Mesh(np.array(devices), ("data",))
    # site_features/select_inducing/descriptor_scale always run on the
    # UNPADDED ds_train: a padded batch's node_mask==True atoms are literal
    # duplicates of the last real batch's atoms, and letting them into
    # descriptor_scale's std or farthest_point's candidate set would make the
    # fitted model silently depend on --devices (review finding, fix round 1).
    with highest_precision():
        X, S = site_features(model, cfg, ds_train)
        ind = select_inducing(X, S, ds_train.node_z, ds_train.node_mask, a.m_per_species,
                              descriptor_scale(X, ds_train.node_mask))
        if "gamma" in z.files:
            gamma = jnp.asarray(z["gamma"])
        else:
            print(f"gamma missing from {a.model} -- rebuilt via construct.prior "
                  "(algebraic smoothness prior)", flush=True)
            gamma = jnp.asarray(gamma_from_model(meta))
        prob = Problem(KernelSpec(a.kernel, not a.no_bump, cfg.D), model, ind, cfg,
                       gamma, default_prior(a.r0))
        # ds_fit is what the (possibly sharded) objective sees; ds_train stays
        # unpadded for predict_mixture below (single-device, brief says so).
        ds_fit = _pad_to_multiple(ds_train, a.devices) if mesh is not None else ds_train
        logdens = make_log_density(prob, ds_fit, a.objective, mesh=mesh)
        lik = logdens.likelihood
        rungs = [r.strip() for r in a.rungs.split(",")]
        theta_map = run_map(lik, prob.prior, steps=a.map_steps, seed=a.seed)
        with open(out / "theta_map.json", "w") as fh:
            json.dump(theta_map._asdict(), fh, indent=1)
        draws = {}
        if "map" in rungs:
            draws["map"] = np.asarray(to_array(theta_map))[None]
        if "laplace" in rungs:
            draws["laplace"], _ = run_laplace(lik, prob.prior, n_draws=a.n_draws, steps=a.map_steps,
                                              seed=a.seed, init=theta_map)
        if "pathfinder" in rungs:
            draws["pathfinder"], _ = run_pathfinder(lik, prob.prior, theta_map, n_draws=a.n_draws, seed=a.seed)
        if "vi" in rungs:
            draws["vi"], _ = run_vi(lik, prob.prior, n_draws=a.n_draws, steps=a.vi_steps,
                                    seed=a.seed, init=theta_map)
        if "nuts" in rungs:
            draws["nuts"], summ = run_nuts(lik, prob.prior, num_warmup=a.nuts_warmup,
                                           num_samples=a.nuts_samples, num_chains=a.nuts_chains,
                                           seed=a.seed, init=theta_map)
            with open(out / "nuts_summary.json", "w") as fh:
                json.dump(summ, fh, indent=1)
        results = {}
        with open(out / "metrics.csv", "w", newline="") as fh:
            w = None
            for rung, d in draws.items():
                np.save(out / f"draws_{rung}.npy", d)
                sub = d if len(d) <= a.n_draws else d[np.linspace(0, len(d) - 1, a.n_draws).astype(int)]
                pred = predict_mixture(sub, prob, ds_train, ds_test)
                results[rung] = _metrics(pred, test_cfgs, ds_test)
                for q, m in results[rung].items():
                    row = {"rung": rung, "quantity": q, **m}
                    if w is None:
                        w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader()
                    w.writerow(row)
    with open(out / "config.json", "w") as fh:
        json.dump({**vars(a), "M": int(ind.XM.shape[0]), "len_basis": cfg.len_basis,
                   "n_train": len(train_cfgs), "n_test": len(test_cfgs)}, fh, indent=1)
    return results


def cmd_eval(a):
    """Evaluate a fitted/exported model on a dataset: predicted energy (and,
    with --forces, forces/virial) per configuration, and RMSE vs the labels
    when present.  Native E/F/V (no ASE), one forward pass per config."""
    import jax.numpy as jnp
    from .eval import highest_precision, load, sparse_graph, species_indices
    model, meta, z = load(a.model)
    rcut = float(meta["rcut"])
    keys = dict(energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key)
    configs = load_configs(a.data, **keys)
    esq = ecnt = fsq = fcnt = 0.0
    rows = []
    with highest_precision():
        for i, c in enumerate(configs):
            g = sparse_graph(c.positions, c.cell, c.pbc, rcut)
            nz = jnp.asarray(species_indices(meta, c.numbers))
            send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
            E, F, V = model.energy_forces_virial(jnp.asarray(g.rij), nz[send], nz[recv],
                                                 send, recv, g.n_nodes, nz)
            E = float(E); F = np.asarray(F); nat = len(c.numbers)
            rows.append({"config": i, "natoms": nat, "energy": E,
                         "energy_per_atom": E / nat, "fmax": float(np.abs(F).max())})
            if c.energy is not None:
                esq += ((E - c.energy) / nat) ** 2; ecnt += 1
            if c.forces is not None:
                fsq += float(((F - c.forces) ** 2).sum()); fcnt += c.forces.size
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {len(rows)} predictions to {a.out}")
    else:
        for r in rows[:10]:
            print(r)
        if len(rows) > 10:
            print(f"... ({len(rows)} configs)")
    if ecnt:
        print(f"E RMSE {1e3 * np.sqrt(esq / ecnt):.3f} meV/atom  ({ecnt} configs)")
    if fcnt:
        print(f"F RMSE {np.sqrt(fsq / fcnt):.4f} eV/A  ({fcnt} components)")
    return rows


def cmd_construct(a):
    """Author a frozen ACE model in memory (seeded radial init, zero readout,
    algebraic smoothness prior) and package it as an npz bridge file.  Requires
    the `authoring` extra; the saved file evaluates with the plain eval path."""
    from .construct.model import build_model
    from .construct.export import save_npz
    els = [int(e) if e.strip().isdigit() else e.strip()
           for e in a.elements.split(",")]
    auth = build_model(els, a.order, a.max_degree, wL=a.wL, rcut=a.rcut,
                       rin=a.rin, radial_mode=a.radial_mode, pair_mode=a.pair_mode,
                       seed=a.seed, with_gamma=not a.no_gamma,
                       coupling_cache=not a.no_coupling_cache,
                       coupling_cache_dir=a.coupling_cache_dir)
    out = pathlib.Path(a.out).expanduser()
    if out.parent and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, auth)
    m, meta = auth.model, auth.meta
    print(f"authored {m.A2B.shape[0]} B functions ({meta['n_AA']} AA), "
          f"{meta['n_pair']} pair, {meta['len_basis']} basis entries, "
          f"lmax {meta['lmax']}, rcut {meta['rcut']} -> {a.out}")
    return auth


def main(argv=None):
    top = argparse.ArgumentParser(prog="ace-jax", description="Fit and evaluate ACE models in JAX")
    sub = top.add_subparsers(dest="cmd", required=True)
    _add_fit_args(sub.add_parser("fit", help="fit the hybrid linear-ACE + residual GP (--m-per-species 0 = linear-only fit)"))
    ev = sub.add_parser("eval", help="evaluate a model on a dataset (energy/forces, RMSE vs labels)")
    ev.add_argument("--model", required=True); ev.add_argument("--data", required=True)
    ev.add_argument("--energy-key", default="energy"); ev.add_argument("--force-key", default="forces")
    ev.add_argument("--virial-key", default="virial"); ev.add_argument("--forces", action="store_true")
    ev.add_argument("--out", default=None, help="CSV of per-config predictions (default: print head)")
    con = sub.add_parser("construct", help="author a frozen ACE model (seeded radial init) and save it")
    con.add_argument("--elements", required=True, help="comma-separated Z numbers or symbols")
    con.add_argument("--order", type=int, required=True, help="correlation order")
    con.add_argument("--max-degree", type=int, required=True, help="TotalDegree level bound")
    con.add_argument("--wL", type=float, default=1.5)
    con.add_argument("--rcut", type=float, default=5.5)
    con.add_argument("--rin", type=float, default=0.0)
    con.add_argument("--radial-mode", default="glorot_normal")
    con.add_argument("--pair-mode", default="onehot")
    con.add_argument("--seed", type=int, default=0)
    con.add_argument("--no-gamma", action="store_true", help="skip the smoothness prior")
    con.add_argument("--no-coupling-cache", action="store_true",
                     help="always run the Julia coupling shim instead of the per-shape cache")
    con.add_argument("--coupling-cache-dir", default=None,
                     help="override the coupling cache directory (default: $ACEJAX_COUPLING_CACHE "
                          "or ~/.cache/ace-jax/coupling)")
    con.add_argument("--out", required=True)
    a = top.parse_args(argv)
    if a.cmd == "eval":
        return cmd_eval(a)
    if a.cmd == "construct":
        return cmd_construct(a)
    return run(a)


if __name__ == "__main__":
    main()
