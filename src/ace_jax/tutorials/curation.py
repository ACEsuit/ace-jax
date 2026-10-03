"""The curation campaign of tutorial 7 (from MLIP-school E3; tutorial support, not stable
API): drive MD with the current model, pick a few frames, label them, refit, repeat, and
track the error of the target property, gamma(111).

One loop serves both the tutorial and the label generator (make_labels.py e3): offline it
runs MD and labels live; the tutorial passes the shipped `pools` and labels from the
shipped cache, so its picks always hit (every pool frame is labelled)."""
import pathlib

import numpy as np

from . import campaign as C
from . import structures as T

CAMPAIGN = dict(drivers=("random", "novelty", "uncertainty"), rounds=2, per_round=4,
                temperature=400.0, n_steps=60, every=4, rng_seed=20260907)


def fit_model(train, out, uq="blr"):
    """The tutorials' evidence fit (order 3, degree 10, rcut 5.5, joint E0) of `train`;
    returns (model.npz, posterior.npz or None). uq='ard' adds the per-atom force std."""
    import jax
    if not jax.config.jax_enable_x64:          # a float32 fit is silently poor, and its MD crawls
        raise RuntimeError("fitting needs float64: call jax.config.update('jax_enable_x64', True) first")
    from ..basis.model import BasisSpec
    from ..fit.pipeline import FitConfig, fit, load_fit_data, write_outputs
    cfg = FitConfig(model=BasisSpec(order=3, max_degree=10, rcut=5.5, elements=("Si",)), arm="linear",
                    m_per_species=0, e0="lsq", opt="lbfgs", r0=None, rungs=("map",), predict_stats="recompute",
                    predict_train=False, uq=uq, energy_key="energy", force_key="forces", virial_key="virial")
    res = fit(cfg.validate(), load_fit_data(cfg, train=train, log=lambda *a: None), log=lambda *a: None)
    out = pathlib.Path(out)
    write_outputs(res, out, layout=("cli",), log=lambda *a: None)
    return str(out / "model.npz"), (str(out / "posterior.npz") if uq == "ard" else None)


def gamma_error(model_file, targets):
    """|gamma(111)| of the fitted model minus the labeller's, eV/A^2; targets = labelled (bulk, slab)."""
    from ..calc.point import ACECalculator
    calc, (b, s) = ACECalculator(model_file, skin=0), targets
    E = []
    for a in (b, s):
        x = a.copy(); x.calc = calc
        E.append(x.get_potential_energy())
    model = T.surface_energy(E[0], len(b), E[1], s)
    truth = T.surface_energy(b.info["energy"], len(b), s.info["energy"], s)
    return abs(model - truth)


def max_force_std(pool, model_file, posterior):
    """Per frame, the largest per-atom force std of an ARD fit (ACECalculator(posterior=))."""
    from ..calc.point import ACECalculator
    calc, out = ACECalculator(model_file, posterior=posterior, skin=0), []
    for a in pool:
        x = a.copy(); x.calc = calc
        out.append(float(np.max(calc.get_property("forces_std", x))))
    return np.array(out)


def default_pick(driver):
    """The built-in selectors: random, descriptor novelty (C.score_novelty), ARD uncertainty."""
    def pick(pool, train, count, rng, model_file, posterior):
        if driver == "random":
            return list(rng.choice(len(pool), size=count, replace=False))
        if driver == "novelty":
            B = C.reference_basis()
            s = C.score_novelty(C.atom_descriptors(pool, B), C.atom_descriptors(train, B))
        else:
            s = max_force_std(pool, model_file, posterior)
        return [int(i) for i in np.argsort(s)[::-1][:count]]
    return pick


def run_campaign(driver, labeller, work, *, seed=0, pools=None, pick=None, rounds=None, per_round=None,
                 n_steps=None, log=lambda *a: None):
    """One driver's campaign. labeller(list[Atoms]) -> labelled copies (tutorials.labels.label).
    pools: the MD pool per round (the tutorial passes the shipped ones); None runs MD from the
    current model. pick(pool, train, count, rng, model_file, posterior) -> indices (default:
    the driver's built-in). Returns {"history": [{labels, err, picks}], "pools": [...]}."""
    cfgd = CAMPAIGN
    rounds, per_round = rounds or cfgd["rounds"], per_round or cfgd["per_round"]
    work, uq = pathlib.Path(work), ("ard" if driver == "uncertainty" else "blr")
    pick = pick or default_pick(driver)
    targets = labeller(list(T.e3_targets()))
    seed_set = labeller(T.e3_seed())
    start, post = fit_model(labeller(T.e1_cells(0.08, 0.02)), work / f"{driver}-start", uq=uq)
    history = [dict(labels=0, err=gamma_error(start, targets), picks=[])]
    train, model, used_pools = list(seed_set), start, []
    rng = np.random.default_rng(cfgd["rng_seed"] + seed)
    for r in range(rounds):
        if pools is not None:
            pool = list(pools[r])
        else:
            pool = [f for f in C.md_pool(model, T.e3_md_starts(), temperature=cfgd["temperature"],
                                         n_steps=n_steps or cfgd["n_steps"], every=cfgd["every"],
                                         seed=1000 * seed + r) if C.physical(f)]
        used_pools.append(pool)
        idx = pick(pool, train, min(per_round, len(pool)), rng, model, post)
        train += labeller([pool[i] for i in idx])
        model, post = fit_model(train, work / f"{driver}-r{r}", uq=uq)
        history.append(dict(labels=len(train) - len(seed_set), err=gamma_error(model, targets),
                            picks=[int(i) for i in idx]))
        log(f"{driver} round {r + 1}: {history[-1]['labels']} labels, |gamma error| {history[-1]['err']:.2e}")
    return {"history": history, "pools": used_pools}
