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
    p.add_argument("--model", default=None,
                   help="an ACE basis/model .npz (or give --order/--max-degree to build the basis)")
    src = p.add_mutually_exclusive_group()
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
    p.add_argument("--rungs", default="map",
                   help="comma-separated from map,laplace,pathfinder,vi,nuts (default map; the others add "
                        "hyperparameter draws and cost far more than the MAP)")
    p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--laplace", choices=["svi", "fd"], default="svi")
    p.add_argument("--map-steps", type=int, default=500); p.add_argument("--vi-steps", type=int, default=2000)
    p.add_argument("--nuts-warmup", type=int, default=500); p.add_argument("--nuts-samples", type=int, default=500)
    p.add_argument("--nuts-chains", type=int, default=4); p.add_argument("--r0", type=float, default=None,
                   help="typical nearest-neighbour distance (A); centres the GP hyperprior "
                        "(default when building the basis: its mean bond length)")
    p.add_argument("--uq", choices=["blr", "pops"], default="blr", help="pops: linear arm (--m-per-species 0)")
    p.add_argument("--pops-ridge", default="auto")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default=None)
    p.add_argument("--model-draws", type=int, default=1,
                   help="GP arm: hyperparameter draws stored in gp_model.npz (1 = the MAP; more = "
                        "evenly spaced draws of the last rung, each adding a (Dt, Dt) factor)")
    p.add_argument("--no-save-model", action="store_true",
                   help="skip writing the fitted model (model.npz linear / gp_model.npz GP)")
    p.add_argument("--devices", type=int, default=1,
                   help="shard sufficient statistics over this many devices (experimental)")
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


def _fit_config(a):
    """FitConfig for `ace-jax fit` arguments, keeping the CLI's historical defaults
    (model E0, per-draw statistics recompute, run_pathfinder's own 16 samples /
    15 iterations)."""
    from .fit.pipeline import FitConfig
    weights, factors = _parse_weights(a.weights)
    rungs = tuple(r.strip() for r in a.rungs.split(","))
    ridge = a.pops_ridge if a.pops_ridge in ("auto", "blr") else float(a.pops_ridge)
    cfg = FitConfig(
        model=a.model if a.model is not None else _basis_spec(a, embedding=a.basis_embedding),
        arm="gp" if a.m_per_species > 0 else "linear", energy_key=a.energy_key,
        force_key=a.force_key, virial_key=a.virial_key, ntrain=a.ntrain, ntest=a.ntest,
        test_start=a.test_start, seed=a.seed, batch=a.configs_per_batch, weights=weights, factors=factors,
        baseline=a.baseline, e0=a.e0, m_per_species=a.m_per_species, kernel=a.kernel, bump=not a.no_bump,
        density=a.density, pca_d=a.pca_d, embedding=a.embedding, r0=a.r0, objective=a.objective,
        lml=a.lml, devices=a.devices, opt=a.opt, map_steps=a.map_steps, map_restarts=a.map_restarts,
        init=json.load(open(a.init)) if a.init else None, rungs=rungs, laplace=a.laplace,
        n_draws=a.n_draws, vi_steps=a.vi_steps, nuts_warmup=a.nuts_warmup, nuts_samples=a.nuts_samples,
        nuts_chains=a.nuts_chains, uq=a.uq, predict_train=False, pops_ridge=ridge,
        predict_stats="recompute", pf_samples=16, pf_maxiter=15)
    return cfg.validate()


def _check_fit_args(p, a):
    """Cross-flag rules for `aj fit` that argparse cannot express (run after any
    fit.yaml merge, so the file may supply what the command line leaves out)."""
    building = a.order is not None or a.max_degree is not None
    if a.model is None and not building:
        p.error("give one of --model, --order/--max-degree, or --config")
    if a.model is not None and building:
        p.error("--model and --order/--max-degree are alternatives: give one")
    if building and (a.order is None or a.max_degree is None):
        p.error("--order and --max-degree go together")
    if a.train is None and a.data is None:
        p.error("one of the arguments --train --data is required")
    if a.out is None:
        p.error("the following arguments are required: --out")
    if a.model is not None and a.r0 is None:
        p.error("--r0 is required with --model")


def run(a):
    from .fit.pipeline import fit, load_fit_data, write_outputs
    cfg = _fit_config(a)
    data = (load_fit_data(cfg, data=a.data, ood=a.ood) if a.data
            else load_fit_data(cfg, train=a.train, test=a.test, ood=a.ood))
    if a.r0 is None:
        print(f"r0 {data.r0:.3f} A (mean bond length of the basis; pass --r0 to override)")
    res = fit(cfg, data)
    write_outputs(res, a.out, layout=("cli",), argv=vars(a), save_model=not a.no_save_model,
                  model_draws=a.model_draws)
    return {key.split("/")[1]: m for key, m in res.preds.metrics.items() if key.startswith("test/")}


def cmd_eval(a):
    """Evaluate a fitted/exported model on a dataset: predicted energy (and,
    with --forces, forces/virial) per configuration, and RMSE vs the labels
    when present.  Native E/F/V (no ASE), one forward pass per config."""
    from .eval import sparse_graph, species_indices
    keys = dict(energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key)
    configs = load_configs(a.data, **keys)
    gp = str(a.model).endswith(".npz") and "gp_json" in np.load(a.model).files   # gp_model.npz from `fit`
    if gp:
        from ase import Atoms
        from .calc.gp import GPCalculator
        calc = GPCalculator.from_file(a.model)
    else:
        model, meta, z = load(a.model)
        rcut = float(meta["rcut"])
    esq = ecnt = fsq = fcnt = 0.0
    rows = []
    with highest_precision():
        for i, c in enumerate(configs):
            if gp:
                at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
                at.calc = calc
                E, F = at.get_potential_energy(), at.get_forces()
            else:
                g = sparse_graph(c.positions, c.cell, c.pbc, rcut)
                nz = jnp.asarray(species_indices(meta, c.numbers))
                send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
                E, F, V = model.energy_forces_virial(jnp.asarray(g.rij), nz[send], nz[recv],
                                                     send, recv, g.n_nodes, nz)
            E = float(E); F = np.asarray(F); nat = len(c.numbers)
            rows.append({"config": i, "natoms": nat, "energy": E,
                         "energy_per_atom": E / nat, "fmax": float(np.abs(F).max())})
            if gp:
                rows[-1]["energy_std"] = float(calc.results["energy_std"])
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


def cmd_basis(a):
    from .basis.export import save_npz
    from .basis.model import build_basis
    auth = build_basis(_basis_spec(a, embedding=a.embedding), seed=a.seed)
    out = pathlib.Path(a.out).expanduser()
    if out.parent and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, auth)
    m, meta = auth.model, auth.meta
    print(f"authored {m.A2B.shape[0]} B functions ({meta['n_AA']} AA), "
          f"{meta['n_pair']} pair, {meta['len_basis']} basis entries, "
          f"lmax {meta['lmax']}, rcut {meta['rcut']} -> {a.out}")
    return auth


def add_basis_args(p, *, fit):
    """The basis-definition flags, shared by `aj basis` and `aj fit`.  On `aj fit`
    the embedding flag is --basis-embedding (--embedding is the GP species
    table there) and --seed is the fit's own."""
    p.add_argument("--elements", default=None, help="comma-separated Z numbers or symbols"
                   + (" (default: the species in the data)" if fit else ""))
    p.add_argument("--order", type=int, required=not fit, help="correlation order")
    p.add_argument("--max-degree", type=int, required=not fit, help="TotalDegree level bound")
    p.add_argument("--wL", type=float, default=1.5)
    p.add_argument("--rcut", type=float, default=None,
                   help="cutoff (default 5.5; with an embedding, 2.5 x mean bond length)")
    p.add_argument("--rin", type=float, default=0.0)
    p.add_argument("--radial-mode", default="glorot_normal")
    p.add_argument("--pair-mode", default="onehot")
    p.add_argument("--no-gamma", action="store_true", help="skip the smoothness prior")
    p.add_argument("--no-coupling-cache", action="store_true",
                   help="always compute the coupling instead of using the per-shape cache")
    p.add_argument("--coupling-cache-dir", default=None,
                   help="coupling cache directory (default: $ACEJAX_COUPLING_CACHE or ~/.cache/ace-jax/coupling)")
    p.add_argument("--basis-embedding" if fit else "--embedding", default=None,
                   help="frozen element embedding of the basis: a JSON table {Z, emb} or 'identity' "
                        "(builds ace_embedding_model: ace1-compatible, factorised radial)")
    p.add_argument("--d-max", type=int, default=None, help="cap on per-order channel widths (default lossless)")
    p.add_argument("--maxl", type=int, default=None)
    p.add_argument("--reduction", choices=["pca", "truncate"], default="pca")
    if not fit:
        p.add_argument("--seed", type=int, default=0)


def _basis_spec(a, *, embedding):
    from .basis.model import BasisSpec
    kw = {f: getattr(a, f) for f in BasisSpec.FIELDS if f not in ("embedding", "elements")}
    els = None if a.elements is None else tuple(
        str(e).strip() for e in (a.elements if isinstance(a.elements, (list, tuple)) else a.elements.split(",")))
    return BasisSpec(elements=els, embedding=embedding, **kw)


def _parser():
    top = argparse.ArgumentParser(prog="ace-jax",
                                  description="Fit and evaluate ACE models in JAX (short alias: aj)")
    sub = top.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser("fit", help="fit the hybrid linear-ACE + residual GP (--m-per-species 0 = linear-only fit)")
    _add_fit_args(fit_p)
    add_basis_args(fit_p.add_argument_group("basis (built in memory; instead of --model)"), fit=True)
    ev = sub.add_parser("eval", help="evaluate a model on a dataset (energy/forces, RMSE vs labels)")
    ev.add_argument("--model", required=True); ev.add_argument("--data", required=True)
    ev.add_argument("--energy-key", default="energy"); ev.add_argument("--force-key", default="forces")
    ev.add_argument("--virial-key", default="virial"); ev.add_argument("--forces", action="store_true")
    ev.add_argument("--out", default=None, help="CSV of per-config predictions (default: print head)")
    con = sub.add_parser("basis", help="author a new ACE basis: a frozen model (seeded radial init) saved as .npz")
    add_basis_args(con, fit=False)
    con.add_argument("--out", required=True)
    return top


def _parse(argv=None):
    top = _parser()
    a = top.parse_args(argv)
    if a.cmd == "fit":
        _check_fit_args(top._subparsers._group_actions[0].choices["fit"], a)
    return a


def main(argv=None):
    """Console entry point; returns 0 because the script wrapper passes the
    result to sys.exit (a returned dict would print and exit 1)."""
    a = _parse(argv)
    {"eval": cmd_eval, "basis": cmd_basis}.get(a.cmd, run)(a)
    return 0


if __name__ == "__main__":
    main()
