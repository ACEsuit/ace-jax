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

from .eval import highest_precision, load
from .fit.data import load_configs
from .fit.pipeline.objective import _pad_to_multiple  # noqa: F401  (moved to the pipeline; kept importable)


def _add_fit_args(p):
    p.add_argument("--model", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--train", help="training extxyz (with --test, or tested on itself)")
    src.add_argument("--data", help="one extxyz split by a seeded permutation (--ntrain/--ntest/--test-start)")
    p.add_argument("--test"); p.add_argument("--ood", help="extra out-of-distribution test extxyz")
    p.add_argument("--ntrain", type=int, default=800); p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--test-start", type=int, default=None)
    p.add_argument("--energy-key", default="energy")
    p.add_argument("--force-key", default="forces"); p.add_argument("--virial-key", default="virial")
    p.add_argument("--weights", default=None,
                   help='JSON: an ACEfit weights dict {"default": {"E":..,"F":..,"V":..}, <config_type>: ..} '
                        'or a list of weight factors [{"Structural": {}}, {"ConfigType": {...}}]')
    p.add_argument("--e0", choices=["model", "lsq"], default="model",
                   help="per-species E0: the model's (default) or least squares on the training energies")
    p.add_argument("--baseline", default=None, help="dimer_mean.npz: fit the residual to this pair mean")
    p.add_argument("--configs-per-batch", type=int, default=8)
    p.add_argument("--m-per-species", type=int, default=500)
    p.add_argument("--kernel", default="cosine", choices=["cosine", "matern32"])
    p.add_argument("--no-bump", action="store_true")
    p.add_argument("--density", choices=["none", "pair", "pca"], default="none",
                   help="GP feature map: full descriptor, pair densities, or a PCA view (--pca-d)")
    p.add_argument("--pca-d", type=int, default=128)
    p.add_argument("--embedding", default=None, help="MACE element table (JSON): frozen species coregionalization")
    p.add_argument("--objective", default="lml", choices=["lml", "loo"])
    p.add_argument("--lml", choices=["device", "host-cache"], default="device",
                   help="host-cache: cache the linear design rows in host RAM (GP, pair|pca, L-BFGS, map only)")
    p.add_argument("--opt", choices=["adam", "lbfgs"], default="adam")
    p.add_argument("--map-restarts", type=int, default=1, help="L-BFGS multi-start (best log-posterior)")
    p.add_argument("--init", default=None, help="theta_map.json to start the MAP from")
    p.add_argument("--rungs", default="map,laplace"); p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--laplace", choices=["svi", "fd"], default="svi")
    p.add_argument("--map-steps", type=int, default=500); p.add_argument("--vi-steps", type=int, default=2000)
    p.add_argument("--nuts-warmup", type=int, default=500); p.add_argument("--nuts-samples", type=int, default=500)
    p.add_argument("--nuts-chains", type=int, default=4); p.add_argument("--r0", type=float, required=True)
    p.add_argument("--uq", choices=["blr", "pops"], default="blr", help="pops: linear arm (--m-per-species 0)")
    p.add_argument("--pops-ridge", default="auto")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
    p.add_argument("--devices", type=int, default=1,
                   help="shard sufficient statistics over this many devices (Task 15, stretch)")
    return p


def _parse_weights(s):
    """(ACEfit weights dict, factor list): exactly one is set, from one JSON string."""
    from .fit.weights import ConfigType, PerConfig, Quantity, Structural
    if not s:
        return None, None
    v = json.loads(s)
    if isinstance(v, dict):
        return v, None
    if isinstance(v, list):
        cls = {"Structural": Structural, "Quantity": Quantity, "ConfigType": ConfigType, "PerConfig": PerConfig}
        return None, [cls[name](**kw) for entry in v for name, kw in entry.items()]
    raise ValueError("--weights must be a JSON object (ACEfit weights) or a JSON list (weight factors)")


def run(a):
    from .fit.pipeline import FitConfig, fit, load_fit_data, write_outputs
    weights, factors = _parse_weights(a.weights)
    rungs = tuple(r.strip() for r in a.rungs.split(","))
    ridge = a.pops_ridge if a.pops_ridge in ("auto", "blr") else float(a.pops_ridge)
    cfg = FitConfig(
        model=a.model, arm="gp" if a.m_per_species > 0 else "linear", energy_key=a.energy_key,
        force_key=a.force_key, virial_key=a.virial_key, ntrain=a.ntrain, ntest=a.ntest,
        test_start=a.test_start, seed=a.seed, batch=a.configs_per_batch, weights=weights, factors=factors,
        baseline=a.baseline, e0=a.e0, m_per_species=a.m_per_species, kernel=a.kernel, bump=not a.no_bump,
        density=a.density, pca_d=a.pca_d, embedding=a.embedding, r0=a.r0, objective=a.objective,
        lml=a.lml, devices=a.devices, opt=a.opt, map_steps=a.map_steps, map_restarts=a.map_restarts,
        init=json.load(open(a.init)) if a.init else None, rungs=rungs, laplace=a.laplace,
        n_draws=a.n_draws, vi_steps=a.vi_steps, nuts_warmup=a.nuts_warmup, nuts_samples=a.nuts_samples,
        nuts_chains=a.nuts_chains, uq=a.uq, predict_train=False, pops_ridge=ridge,
        predict_stats="recompute")
    cfg.validate()
    data = (load_fit_data(cfg, data=a.data, ood=a.ood) if a.data
            else load_fit_data(cfg, train=a.train, test=a.test, ood=a.ood))
    res = fit(cfg, data)
    write_outputs(res, a.out, layout=("cli",), argv=vars(a))
    return {key.split("/")[1]: m for key, m in res.preds.metrics.items() if key.startswith("test/")}


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
    from .construct.export import save_npz
    els = [int(e) if e.strip().isdigit() else e.strip() for e in a.elements.split(",")]
    if a.embedding:
        from .construct.model import build_embedding_model
        auth = build_embedding_model(els, a.order, a.max_degree, embedding=a.embedding, d_max=a.d_max,
                                     wL=a.wL, maxl=a.maxl, rcut=a.rcut, reduction=a.reduction,
                                     with_gamma=not a.no_gamma, coupling_cache=not a.no_coupling_cache,
                                     coupling_cache_dir=a.coupling_cache_dir)
    else:
        from .construct.model import build_model
        auth = build_model(els, a.order, a.max_degree, wL=a.wL, rcut=5.5 if a.rcut is None else a.rcut,
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
    con.add_argument("--rcut", type=float, default=None,
                     help="cutoff (default 5.5; with --embedding, 2.5 x mean bond length)")
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
    con.add_argument("--embedding", default=None,
                     help="frozen element embedding: a JSON table {Z, emb} or 'identity' "
                          "(builds ace_embedding_model: ace1-compatible, factorised radial)")
    con.add_argument("--d-max", type=int, default=None, help="cap on per-order channel widths (default lossless)")
    con.add_argument("--maxl", type=int, default=None)
    con.add_argument("--reduction", choices=["pca", "truncate"], default="pca")
    con.add_argument("--out", required=True)
    a = top.parse_args(argv)
    if a.cmd == "eval":
        return cmd_eval(a)
    if a.cmd == "construct":
        return cmd_construct(a)
    return run(a)


if __name__ == "__main__":
    main()
