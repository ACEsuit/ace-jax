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
    p.add_argument("--config", default=None,
                   help="a fit.yaml run file (flag names as keys, the basis in a `basis:` block); "
                        "command-line flags override it")
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
    p.add_argument("--uq", choices=["blr", "pops", "ard"], default="blr",
                   help="pops/ard: linear arm (--m-per-species 0); ard = ARD posterior with a "
                        "calibrated per-atom forces_std (see --ard-variance), writes posterior.npz")
    p.add_argument("--ard-mode", choices=["joint", "sequential"], default="joint",
                   help="joint: noise + ARD scales by evidence; sequential: ARD only, one Gram (low memory)")
    p.add_argument("--ard-variance", choices=["sandwich", "kappa"], default="sandwich",
                   help="ARD force-uncertainty shape: sandwich (default) = delete-one-cluster PRESS jackknife "
                        "(misspecification-robust); kappa = the posterior A^-1 shape. Both get the "
                        "per-group scales")
    p.add_argument("--ard-val-frac", type=float, default=0.2,
                   help="stratified train fraction held out to score the per-group force scales "
                        "(rms factor and conformal quantile) with the hold-out posterior")
    p.add_argument("--force-shape", choices=["iso", "aniso"], default="iso",
                   help="ard: isotropic (|e|/sqrt(v/3)) or anisotropic (Mahalanobis) conformal force scores")
    p.add_argument("--ard-coverage", type=float, default=0.9,
                   help="ard: nominal coverage 1 - alpha of the per-group conformal quantile")
    p.add_argument("--ard-groups", choices=["distortion", "none"], default="distortion",
                   help="ard: conformal groups = distortion bands x [z = z*] (8), or none (2)")
    p.add_argument("--ard-cluster-size", type=float, default=3.0,
                   help="ard: sandwich block side in units of r_cut ('inf': whole configurations)")
    p.add_argument("--ard-press", choices=["exact", "block"], default="exact",
                   help="ard: PRESS correction of the jackknife scores (exact: per-cluster solve; block: block approximation)")
    p.add_argument("--ard-n-min", type=int, default=20,
                   help="ard: groups with fewer configurations borrow a neighbouring group's scales")
    p.add_argument("--no-ard-support", action="store_true",
                   help="ard: skip the covariate-shift support reference")
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
        nuts_chains=a.nuts_chains, uq=a.uq, ard_mode=a.ard_mode, ard_variance=a.ard_variance, ard_val_frac=a.ard_val_frac,
        ard_force_shape=a.force_shape, ard_coverage=a.ard_coverage, ard_groups=a.ard_groups,
        ard_cluster_size=a.ard_cluster_size, ard_press=a.ard_press, ard_n_min=a.ard_n_min,
        ard_support=not a.no_ard_support,
        predict_train=False, pops_ridge=ridge,
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
    from . import runfile
    fit_p = _parser()._subparsers._group_actions[0].choices["fit"]
    runfile.write(pathlib.Path(a.out) / "fit.yaml",
                  runfile.resolved(a, data, fit_dests=runfile._dests(fit_p), argv=getattr(a, "_argv", [])))
    return {key.split("/")[1]: m for key, m in res.preds.metrics.items() if key.startswith("test/")}


def cmd_eval(a):
    """Evaluate a fitted/exported model on a dataset: predicted energy (and,
    with --forces, forces/virial) per configuration, and RMSE vs the labels
    when present.  Native E/F/V (no ASE), one forward pass per config."""
    from .eval import sparse_graph, species_indices
    keys = dict(energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key)
    configs = load_configs(a.data, **keys)
    gp = str(a.model).endswith(".npz") and "gp_json" in np.load(a.model).files   # gp_model.npz from `fit`
    ard = getattr(a, "posterior", None) is not None
    if gp and ard:
        raise ValueError("--posterior is for a linear model.npz from `fit --uq ard`, not a gp_model.npz")
    if gp or ard:
        from ase import Atoms
        if gp:
            from .calc.gp import GPCalculator
            calc = GPCalculator.from_file(a.model)
        else:
            from .calc.point import ACECalculator
            calc = ACECalculator(a.model, posterior=a.posterior)
    else:
        model, meta, z = load(a.model)
        rcut = float(meta["rcut"])
    esq = ecnt = fsq = fcnt = 0.0
    rows, per_atom = [], []
    served = {}
    if ard:
        post = calc.posterior
        want_pa = bool(getattr(a, "per_atom", None))
        if want_pa and post.group_table is None:
            print("note: schema-2 posterior: forces_q/forces_group/forces_cov are not written "
                  "(refit with --uq ard)")
        served = {"q": want_pa and post.group_table is not None,
                  "mahal": want_pa and post.group_table is not None and post.force_shape == "aniso",
                  "support": want_pa and getattr(a, "support", False)}
        if served["support"] and post.support is None:
            print("note: this posterior has no support reference: support_ok/support_q are not written")
            served["support"] = False
    with highest_precision():
        for i, c in enumerate(configs):
            if gp or ard:
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
            if ard:
                s = np.asarray(calc.get_property("forces_std", at))   # on request: E/F reused
                rows[-1]["fmax_std"] = float(s.max())
                if getattr(a, "per_atom", None):
                    out_at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
                    out_at.arrays["forces_pred"] = F
                    out_at.arrays["forces_std"] = s
                    if served["q"]:
                        out_at.arrays["forces_q"] = np.asarray(calc.get_property("forces_q", at))
                        out_at.arrays["forces_group"] = np.asarray(calc.get_property("forces_group", at))
                        if served["mahal"]:
                            out_at.arrays["forces_cov"] = np.asarray(
                                calc.get_property("forces_cov", at)).reshape(len(at), 9)
                            out_at.arrays["forces_q_mahal"] = np.asarray(calc.get_property("forces_q_mahal", at))
                    if served["support"]:
                        sup = calc.get_property("forces_support", at)
                        out_at.arrays["support_ok"] = np.asarray(sup["support_ok"], bool)
                        out_at.arrays["support_q"] = np.asarray(sup["support_q"], float)
                    per_atom.append(out_at)
            if c.energy is not None:
                esq += ((E - c.energy) / nat) ** 2; ecnt += 1
            if c.forces is not None:
                fsq += float(((F - c.forces) ** 2).sum()); fcnt += c.forces.size
    if per_atom:
        from ase.io import write as _write
        _write(a.per_atom, per_atom)
        print(f"wrote per-atom uncertainty arrays for {len(per_atom)} configs to {a.per_atom}")
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


_NEED3 = "this posterior predates schema 3; refit with --uq ard"


def _print_group_table(t):
    d = t.to_dict() if hasattr(t, "to_dict") else t
    merged = {int(k): int(src) for k, src in d["merged"]}
    print(f"{'g':>2} {'n_cfg (T_val/U)':>16} {'lam_rms':>9} {'q':>8} {'r':>7}")
    for g in range(len(d["q"])):
        line = (f"{g:>2} {d['n_cfg'][g]:>6} ({d['n_cfg_val'][g]}/{d['n_cfg_cal'][g]})".ljust(26)
                + f" {d['lam_rms'][g]:9.4g} {d['q'][g]:8.4g} {d['r'][g]:7.3g}")
        print(line + (f"  -> g_src {merged[g]}" if g in merged else ""))


def cmd_calibrate(a):
    """Recalibrate the per-group conformal scales of an ARD posterior on a labelled set U (the posterior
    mean and covariance are untouched): per-group replace by default (groups with >= n_min U configurations
    use U only, the rest pool T_val + U), --append pools everywhere, --replace uses U only."""
    import dataclasses
    import hashlib

    from ase import Atoms

    from .calc.point import ACECalculator
    from .fit.ard import ARDPosterior, conformal_scores
    from .fit.conformal import group_scales
    post = ARDPosterior.load(a.posterior)
    if post.group_table is None or post.cal is None:
        raise ValueError(_NEED3)
    configs = load_configs(a.data, energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key)
    if not any(c.forces is not None for c in configs):
        raise ValueError(f"calibrate: no forces under --force-key {a.force_key!r} in {a.data}")
    calc = ACECalculator(a.model, posterior=a.posterior)
    t0 = post.group_table
    lam_g = np.asarray(t0["lam_rms"], float)
    G, n_min = len(t0["q"]), int(t0["n_min"])
    want_support = post.support is not None
    s_u, g_u, c_u, X_u, Z_u = [], [], [], [], []
    with highest_precision():
        for ci, c in enumerate(configs):
            if c.forces is None:
                continue
            at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
            at.calc = calc
            cov = np.asarray(calc.get_property("forces_cov", at))
            g = np.asarray(calc.get_property("forces_group", at)).astype(int)
            e = np.asarray(c.forces) - at.get_forces()
            lam = lam_g[g]
            V = cov / np.where(lam > 0, lam, np.nan)[:, None, None] ** 2          # exact: cov = lam^2 V
            v = np.trace(V, axis1=1, axis2=2)
            ok = np.isfinite(v) & (v > 0)
            if not ok.any():
                continue
            s = conformal_scores(e[ok], V[ok], post.force_shape, post.eps)
            s_u.append(s); g_u.append(g[ok]); c_u.append(np.full(int(ok.sum()), ci))
            if want_support:
                X, Z = calc.support_descriptors(at)
                X_u.append(X[ok]); Z_u.append(Z[ok])
    if not s_u:
        raise ValueError("calibrate: no atoms with a positive posterior shape and a force label")
    s_u, g_u, c_u = map(np.concatenate, (s_u, g_u, c_u))
    cal = post.cal
    src_new = int(np.max(cal["src"])) + 1
    c_old = np.asarray(cal["cfg"], np.int64)
    c_u = c_u + (int(c_old.max()) + 1 if len(c_old) else 0)
    u_cfg = np.array([len(np.unique(c_u[g_u == g])) for g in range(G)])
    g_old = np.asarray(cal["groups"]).astype(int)
    if a.replace:
        keep_old = np.zeros(len(g_old), bool)
    elif a.append:
        keep_old = np.ones(len(g_old), bool)
    else:
        keep_old = u_cfg[g_old] < n_min                                   # per-group replace
    S = np.r_[np.asarray(cal["scores"], np.float32)[keep_old], s_u.astype(np.float32)]
    Gg = np.r_[g_old[keep_old], g_u]
    Cc = np.r_[c_old[keep_old], c_u]
    Ss = np.r_[np.asarray(cal["src"]).astype(int)[keep_old], np.full(len(s_u), src_new)]
    tab = group_scales(S.astype(float), Gg, Cc, G, t0["alpha"], n_min, src=Ss)
    h = hashlib.sha256()
    with open(a.data, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    mode = "replace" if a.replace else ("append" if a.append else "per-group")
    sources = list(t0.get("sources", [])) + [{"path": str(a.data), "sha256": h.hexdigest(),
                                               "n_cfg": int(len(np.unique(c_u))), "n_atoms": int(len(s_u)),
                                               "mode": mode}]
    tab = dataclasses.replace(tab, sources=sources)
    support = None
    if want_support:
        from .fit.support import recalibrate_support
        support = recalibrate_support(post.support, mode, u_cfg, n_min, np.concatenate(X_u),
                                      np.concatenate(Z_u), s_u, c_u, g_u)
    new = post._replace(group_table=tab.to_dict(),
                        cal={"scores": S.astype(np.float32), "groups": Gg.astype(np.int8),
                             "cfg": Cc.astype(np.int64), "src": Ss.astype(np.int16)},
                        support=support)
    new.save(a.out)
    _print_group_table(tab)
    print(f"wrote recalibrated posterior ({mode}, {len(s_u)} atoms in {len(np.unique(c_u))} configs) to {a.out}")
    return 0


def cmd_basis(a):
    from .basis.export import save_npz
    from .basis.model import build_basis
    auth = build_basis(_basis_spec(a, embedding=a.embedding), seed=a.seed)
    out = pathlib.Path(a.out).expanduser()
    if out.parent and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, auth)
    m, meta = auth.model, auth.meta
    print(f"basis: {m.A2B.shape[0]} B functions ({meta['n_AA']} AA), "
          f"{meta['n_pair']} pair, {meta['len_basis']} basis entries, "
          f"lmax {meta['lmax']}, rcut {meta['rcut']} -> {a.out}")
    return auth


def add_basis_args(p, *, fit):
    """The basis-definition flags, shared by `aj basis` and `aj fit`.  On `aj fit`
    the embedding flag is --basis-embedding (--embedding is the GP species
    table there) and --seed is the fit's own."""
    p.add_argument("--elements", default=None, required=not fit, help="comma-separated Z numbers or symbols"
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
    ev.add_argument("--posterior", default=None, help="posterior.npz from `fit --uq ard`: adds forces_std")
    ev.add_argument("--per-atom", default=None, help="extxyz with per-atom forces and forces_std arrays")
    ev.add_argument("--support", action="store_true",
                    help="with --posterior --per-atom: add support_ok and support_q (covariate-shift support)")
    cal = sub.add_parser("calibrate", help="recalibrate the per-group conformal scales of an ARD posterior "
                                           "on a labelled set (per-group replace by default)")
    cal.add_argument("--model", required=True); cal.add_argument("--posterior", required=True)
    cal.add_argument("--data", required=True, help="labelled extxyz U (needs forces under --force-key)")
    cal.add_argument("--energy-key", default="energy"); cal.add_argument("--force-key", default="forces")
    cal.add_argument("--virial-key", default="virial")
    mode = cal.add_mutually_exclusive_group()
    mode.add_argument("--append", action="store_true", help="pool U with the stored T_val scores in every group")
    mode.add_argument("--replace", action="store_true", help="use U only in every group")
    cal.add_argument("--out", required=True)
    con = sub.add_parser("basis", help="author a new ACE basis: a frozen model (seeded radial init) saved as .npz")
    add_basis_args(con, fit=False)
    con.add_argument("--out", required=True)
    con.add_argument("--config", default=None,
                     help="a fit.yaml: its `basis:` block (and `seed`); command-line flags override it")
    return top


def _config_path(argv):
    for i, t in enumerate(argv):
        if t == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if t.startswith("--config="):
            return t.split("=", 1)[1]
    return None


def _apply_config(sub, cmd, path):
    """Load a fit.yaml into `sub`'s defaults (so explicit flags still win) and
    un-require what the file supplies.  Returns the defaults applied."""
    from . import runfile
    from .basis.model import BasisSpec
    try:
        cfg = runfile.read(path)
        if cmd == "basis":             # aj basis: the basis block (+ seed); the fit keys are not its business
            flat = {**cfg.get("basis", {}), **({"seed": cfg["seed"]} if "seed" in cfg else {})}
            defaults = runfile.defaults_for(sub, flat, basis_fields=(), basis_dest_map={})
        else:
            defaults = runfile.defaults_for(sub, cfg, basis_fields=BasisSpec.FIELDS,
                                            basis_dest_map={"embedding": "basis_embedding"})
    except (OSError, ValueError, TypeError) as e:
        sub.error(str(e))
    for act in sub._actions:
        if act.dest in defaults:
            act.required = False
    sub.set_defaults(**defaults)
    return defaults


def _parse(argv=None):
    """Parse the command line, layering a --config fit.yaml under it."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    top = _parser()
    subs = top._subparsers._group_actions[0].choices
    cmd = next((t for t in argv if not t.startswith("-")), None)
    path = _config_path(argv) if cmd in ("fit", "basis") else None
    defaults = _apply_config(subs[cmd], cmd, path) if path else {}
    a = top.parse_args(argv)
    a._argv = argv                              # as typed: recorded in out/fit.yaml provenance
    if defaults:
        from . import runfile
        given = runfile.explicit_dests(subs[cmd], argv[argv.index(cmd) + 1:])
        for k in sorted(set(defaults) & given):
            print(f"override: {k} {defaults[k]} -> {getattr(a, k)} (command line)")
        # alternatives: choosing one side on the command line drops the file's other side
        # (argparse's mutual exclusion never sees a default, and _check_fit_args would refuse both)
        for mine, other in (({"train"}, {"data"}), ({"data"}, {"train"}),
                            ({"model"}, {"order", "max_degree"}), ({"order", "max_degree"}, {"model"})):
            if a.cmd == "fit" and mine & given:
                for k in sorted((other & set(defaults)) - given):
                    print(f"override: {k} {defaults[k]} -> None (command line gives "
                          f"{'/'.join('--' + m.replace('_', '-') for m in sorted(mine & given))})")
                    setattr(a, k, None)
    if a.cmd == "fit":
        _check_fit_args(subs["fit"], a)
    return a


def main(argv=None):
    """Console entry point; returns 0 because the script wrapper passes the
    result to sys.exit (a returned dict would print and exit 1)."""
    import sys

    from .basis.coupling import BasisUnavailable
    a = _parse(argv)
    try:
        {"eval": cmd_eval, "basis": cmd_basis, "calibrate": cmd_calibrate}.get(a.cmd, run)(a)
    except BasisUnavailable as e:                     # a user-facing condition, not a crash
        print(f"aj {a.cmd}: error: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    return 0


if __name__ == "__main__":
    main()
