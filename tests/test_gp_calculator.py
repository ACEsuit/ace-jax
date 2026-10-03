import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.calc.gp import GPCalculator, fit_posteriors
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem
from ace_jax.fit.predict import predict_mixture

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def _fitted_si():
    """A two-draw fitted GP on the first six si_tiny configs: (fitted, meta, E0,
    configs, prob, train, draws)."""
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    E0 = np.asarray(z["E0"])
    train = build_dataset(configs[:6], meta, E0, 3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, train)
    ind = select_inducing(X, S, train.node_z, train.node_mask, 6, descriptor_scale(X, train.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    theta = Hypers(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=0.0, log_r0=np.log(2.35),
                   log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
                   log_sigma_E=np.log(1e-3), log_sigma_F=np.log(0.02), log_sigma_V=np.log(0.02))
    a = np.asarray(to_array(theta))
    draws = np.stack([a, a + np.array([0.2] + [0.0] * 9)])
    with highest_precision():
        fitted = fit_posteriors(prob, train, draws)
    return fitted, meta, E0, configs, prob, train, draws


def test_calculator_deriv_dtc_false_is_sor_forces_std():
    """GPCalculator(deriv_dtc=False): forces_std from the SoR-only force variance, the mixture of
    predict_mixture(deriv_dtc=False); the default keeps the (larger) derivative-DTC term."""
    from ase import Atoms
    fitted, meta, E0, configs, prob, train, draws = _fitted_si()
    with highest_precision():
        c = configs[7]
        test = build_dataset([c], meta, E0, 1)
        sd = {}
        for flag in (True, False):
            atoms = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
            atoms.calc = GPCalculator(fitted, meta, deriv_dtc=flag)
            sd[flag] = atoms.calc.get_property("forces_std", atoms)
        p = predict_mixture(draws, prob, train, test, deriv_dtc=False)
    assert np.allclose(sd[False] ** 2, p.F_var, rtol=1e-8, atol=1e-14)
    assert np.all(sd[True] >= sd[False] - 1e-12) and not np.allclose(sd[True], sd[False])


def test_calculator_agrees_with_predict_mixture():
    from ase import Atoms
    fitted, meta, E0, configs, prob, train, draws = _fitted_si()
    with highest_precision():
        c = configs[7]
        atoms = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
        atoms.calc = GPCalculator(fitted, meta)
        E, F, s = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        E_std = atoms.calc.get_property("energy_std", atoms)
        test = build_dataset([c], meta, E0, 1)
        p = predict_mixture(draws, prob, train, test)
    assert abs(E - p.E_mean[0]) < 1e-8 and np.abs(F - p.F_mean).max() < 1e-8
    assert abs(E_std - np.sqrt(p.E_var[0])) < 1e-8
    assert np.allclose(s, -p.V_mean[0] / atoms.get_volume())


def test_calculator_compiles_its_predictor_once():
    """GPCalculator runs the jitted per-batch predictor (predict._predict_fn), built
    once per calculator. An eager _predict_batch dispatched the derivative-DTC op by
    op: ~7.5 s per structure on si_tiny, 5x the jitted path. A second structure of
    the same shape must reuse the compiled predictor."""
    from ase import Atoms
    fitted, meta, E0, configs, prob, train, draws = _fitted_si()
    calc = GPCalculator(fitted, meta)
    c = configs[7]
    for shift in (0.0, 1e-3):                       # a new calculation, same padded shape
        atoms = Atoms(numbers=c.numbers, positions=c.positions + shift, cell=c.cell, pbc=c.pbc)
        atoms.calc = calc
        atoms.get_potential_energy()
    assert calc._predict._cache_size() == 1


def test_slot_buckets_are_quarter_octaves():
    from ace_jax.calc.gp import slot_bucket
    ks = range(1, 2000)
    assert all(slot_bucket(k) >= k for k in ks)
    assert max(slot_bucket(k) / k for k in ks if k >= 8) <= 2 ** 0.25 + 1e-9    # <= 19% padding
    assert len({slot_bucket(k) for k in range(39, 77)}) <= 4                       # si_tiny's range


def test_calculator_buckets_neighbour_slots():
    """Neighbour slots are padded to a quarter-octave bucket (slot_bucket), not the exact
    maximum neighbour count: si_tiny's 53 configs had 27 distinct exact counts (39-76), so
    nearly every structure (and every MD step) compiled the predictor again; multiples of
    8 still gave 7 compiles.  Every dimer evaluated: one compile per bucket."""
    from ase import Atoms
    from ace_jax.calc.gp import slot_bucket
    from ace_jax.fit.data import sparse_graph
    fitted, meta, E0, configs, prob, train, draws = _fitted_si()
    rcut = float(meta["rcut"])
    dimers = [c for c in configs if len(c.numbers) == 2]
    k = [int(np.bincount(sparse_graph(c.positions, c.cell, c.pbc, rcut).senders, minlength=2).max())
         for c in dimers]
    picks = {}                                    # two dimers with different counts per bucket
    for c, ki in zip(dimers, k):
        picks.setdefault(slot_bucket(ki), {}).setdefault(ki, c)
    picks = [c for v in picks.values() for c in list(v.values())[:2]]
    calc = GPCalculator(fitted, meta)
    for c in picks:
        atoms = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
        atoms.calc = calc
        atoms.get_potential_energy()
    assert len(picks) > len({slot_bucket(x) for x in k})                 # some bucket is shared
    assert calc._predict._cache_size() == len({slot_bucket(x) for x in k}) <= 4, sorted(set(k))
