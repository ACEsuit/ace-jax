"""python -m acegp.cli: fit the hybrid GP on an extxyz dataset with an exported
ACE model, run the requested rungs of the ladder, write draws and the metrics
table (per atom energies in meV/atom, forces in eV/A, virials in eV)."""
import argparse
import json
import pathlib

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from .eval import highest_precision
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
    p.add_argument("--stress-key", default=None,
                   help="stress label (eV/A^3, 3x3 or Voigt-6) used as the virial, virial = -stress * volume, when a config has no virial label")
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
                   help="ARD force variance: sandwich = configuration-clustered misspecification-robust "
                        "(scaled by lam on the train hold-out); kappa = kappa^2 x the posterior variance")
    p.add_argument("--ard-val-frac", type=float, default=0.2,
                   help="train fraction held out to fit the force-variance scale: lam (sandwich) "
                        "and kappa")
    p.add_argument("--learn-radial", action="store_true",
                   help="learn the tensor radials (VarPro, held-out gate) before the fit; the saved model "
                        "is marked radial_learned and splined at deploy time")
    p.add_argument("--radial-n-q", type=int, default=12,
                   help="tensor-radial polynomial span after widening (with --learn-radial)")
    p.add_argument("--radial-steps", type=int, default=40,
                   help="L-BFGS steps per roughness weight (with --learn-radial)")
    p.add_argument("--radial-lam-grid", default="0,1e-2",
                   help="comma-separated relative roughness weights the gate picks among, "
                        "alongside the initial radials (with --learn-radial)")
    p.add_argument("--radial-val-frac", type=float, default=0.2,
                   help="fraction of the training configs held out for the gate (with --learn-radial)")
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
        force_key=a.force_key, virial_key=a.virial_key, stress_key=a.stress_key, ntrain=a.ntrain, ntest=a.ntest,
        test_start=a.test_start, seed=a.seed, batch=a.configs_per_batch, weights=weights, factors=factors,
        baseline=a.baseline, e0=a.e0, m_per_species=a.m_per_species, kernel=a.kernel, bump=not a.no_bump,
        density=a.density, pca_d=a.pca_d, embedding=a.embedding, r0=a.r0, objective=a.objective,
        lml=a.lml, devices=a.devices, opt=a.opt, map_steps=a.map_steps, map_restarts=a.map_restarts,
        init=json.load(open(a.init)) if a.init else None, rungs=rungs, laplace=a.laplace,
        n_draws=a.n_draws, vi_steps=a.vi_steps, nuts_warmup=a.nuts_warmup, nuts_samples=a.nuts_samples,
        nuts_chains=a.nuts_chains, uq=a.uq, ard_mode=a.ard_mode, ard_variance=a.ard_variance, ard_val_frac=a.ard_val_frac,
        learn_radial=a.learn_radial, radial_n_q=a.radial_n_q, radial_steps=a.radial_steps,
        radial_lam_grid=tuple(float(x) for x in str(a.radial_lam_grid).split(",") if x.strip()),
        radial_val_frac=a.radial_val_frac,
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
    """Evaluate a fitted/exported model on a dataset.  Prints the per-config-type E/F/V
    RMSE table (fit/report.py) for the labels present, and with --out writes the input
    structures back as extxyz with every original label kept and the predictions added:
    info <prefix>energy (eV) and <prefix>stress (3x3, eV/A^3, periodic cells only),
    arrays <prefix>forces (eV/A); a gp_model.npz adds info <prefix>energy_std and arrays
    <prefix>forces_std, --posterior (ARD) arrays <prefix>forces_std.  The structures go
    through the ase-extxyz plugin (create_calc=False: no label moves into a calculator).
    Every model goes through its jitted calculator: the edge list is padded to buckets,
    so configs of similar size share one compile.  A linear model is evaluated exactly
    (lean, never splined: spline_tol=None)."""
    from ase import Atoms
    from ase.stress import voigt_6_to_full_3x3_stress
    from .fit.data import VOIGT
    from .fit.report import format_rmse_table, rmse_by_type
    keys = dict(energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key, stress_key=a.stress_key)
    configs = load_configs(a.data, **keys)
    frames = None
    if a.out:                 # passed through as ASE Atoms: create_calc=False keeps every label as written
        from ase_extxyz.io import read_cextxyz
        frames = list(read_cextxyz(a.data, index=":", create_calc=False))
    gp = str(a.model).endswith(".npz") and "gp_json" in np.load(a.model).files   # gp_model.npz from `fit`
    ard = getattr(a, "posterior", None) is not None
    if gp and ard:
        raise ValueError("--posterior is for a linear model.npz from `fit --uq ard`, not a gp_model.npz")
    if gp:
        from .calc.gp import GPCalculator
        calc = GPCalculator.from_file(a.model)
    else:
        from .calc.point import ACECalculator
        calc = ACECalculator(a.model, posterior=a.posterior if ard else None, spline_tol=None)
    p = a.prefix
    Em, Fm, Vm = [], [], []
    with highest_precision():
        for i, c in enumerate(configs):
            at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
            at.calc = calc
            E, F = float(at.get_potential_energy()), np.asarray(at.get_forces())
            S = np.asarray(at.get_stress()) if at.pbc.all() else None          # Voigt xx yy zz yz xz xy
            Em.append(E); Fm.append(F)
            Vm.append(np.full(6, np.nan) if S is None else -S * at.get_volume())   # the virial, VOIGT order
            if frames is not None:
                f = frames[i]
                f.info[f"{p}energy"] = E
                f.arrays[f"{p}forces"] = F
                if S is not None:
                    f.info[f"{p}stress"] = voigt_6_to_full_3x3_stress(S)
                if gp:
                    f.info[f"{p}energy_std"] = float(calc.results["energy_std"])
                    f.arrays[f"{p}forces_std"] = np.asarray(calc.results["forces_std"])
                if ard:
                    f.arrays[f"{p}forces_std"] = np.asarray(calc.get_property("forces_std", at))
    nat = np.array([len(c.numbers) for c in configs])
    E = np.array([np.nan if c.energy is None else c.energy for c in configs])
    F = np.concatenate([np.full((len(c.numbers), 3), np.nan) if c.forces is None else c.forces for c in configs])
    V = np.array([np.full(6, np.nan) if c.virial is None else [c.virial[i, j] for i, j in VOIGT] for c in configs])
    print(format_rmse_table(rmse_by_type([c.config_type for c in configs], nat, E, np.array(Em),
                                         F, np.concatenate(Fm), V, np.array(Vm)), f"{a.data} vs {a.model}"))
    if frames is not None:
        out = pathlib.Path(a.out).expanduser()
        if out.parent and str(out.parent) != ".":
            out.parent.mkdir(parents=True, exist_ok=True)
        from ase_extxyz.io import write_cextxyz
        write_cextxyz(str(out), frames)
        print(f"wrote {len(frames)} configurations with predictions ({p}energy, {p}forces, ...) to {out}")
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
    p.add_argument("--radial-mode", default="onehot", choices=["onehot", "glorot_normal", "zero"],
                   help="initial tensor radials: onehot (R_n = P_n, the frozen-fit default) or seeded "
                        "glorot_normal mixtures (a random start, e.g. for learned radials)")
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
    ev = sub.add_parser("eval", help="evaluate a model on a dataset: RMSE table vs the labels, predictions to extxyz")
    ev.add_argument("--model", required=True); ev.add_argument("--data", required=True)
    ev.add_argument("--energy-key", default="energy"); ev.add_argument("--force-key", default="forces")
    ev.add_argument("--virial-key", default="virial")
    ev.add_argument("--stress-key", default=None,
                    help="stress label (eV/A^3, 3x3 or Voigt-6) used as the virial, virial = -stress * volume, when a config has no virial label")
    ev.add_argument("--out", default=None,
                    help="extxyz to write: the input structures, every label kept, plus the predictions "
                         "(<prefix>energy, <prefix>forces, <prefix>stress, and *_std for UQ models)")
    ev.add_argument("--prefix", default="ace_", help="name prefix of the predicted keys (default ace_)")
    ev.add_argument("--posterior", default=None, help="posterior.npz from `fit --uq ard`: adds <prefix>forces_std")
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
    if a.cmd == "fit" and not a.learn_radial:   # typed options only: a fit.yaml records every default
        from . import runfile
        typed = runfile.explicit_dests(subs["fit"], argv[argv.index(cmd) + 1:])
        stray = sorted(d for d in typed if d.startswith("radial_"))
        if stray:
            subs["fit"].error(f"--{stray[0].replace('_', '-')} needs --learn-radial")
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
        {"eval": cmd_eval, "basis": cmd_basis}.get(a.cmd, run)(a)
    except BasisUnavailable as e:                     # a user-facing condition, not a crash
        print(f"aj {a.cmd}: error: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    return 0


if __name__ == "__main__":
    main()
