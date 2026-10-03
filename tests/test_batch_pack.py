"""Size-aware (atom-budget) batching in build_dataset: order-preserving greedy packing of mixed-size
training sets (bulk + big cells), and its index consistency with every config-ordered consumer."""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from ase.build import bulk

from ace_jax.eval import highest_precision
from ace_jax.fit.data import Config, Dataset, build_dataset
from conftest import FIXTURE_DIR

META = {"elements": [14], "rcut": 5.0}


def _si(rep, rng, rattle=0.05):
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat(rep)
    pos = at.positions + rng.normal(scale=rattle, size=at.positions.shape)
    n = len(at)
    return Config(pos, at.numbers, np.asarray(at.cell), np.asarray(at.pbc), float(rng.normal()),
                  rng.normal(size=(n, 3)), rng.normal(size=(3, 3)), 1 / np.sqrt(n), 1.0, 1 / np.sqrt(n))


def _mixed(rng, n_small=80, big_at=(5, 40, 81)):
    """n_small configs of 8-40 atoms with three 600-atom cells inserted at big_at."""
    reps = [(1, 1, k) for k in rng.integers(1, 6, n_small)]            # 8 .. 40 atoms
    cfgs = [_si(r, rng) for r in reps]
    for i in big_at:
        cfgs.insert(i, _si((5, 5, 3), rng))                              # 600 atoms
    return cfgs


def _live_order(ds):
    """(batch, slot) of every live config, in Dataset order, and its atom count."""
    cm, nat = np.asarray(ds.cfg_mask), np.asarray(ds.n_atoms)
    return [(b, c, int(nat[b, c])) for b in range(cm.shape[0]) for c in np.flatnonzero(cm[b])]


def _assert_same(a, b):
    for k in Dataset._fields:
        x, y = np.asarray(getattr(a, k)), np.asarray(getattr(b, k))
        assert x.dtype == y.dtype and x.shape == y.shape, k
        assert x.tobytes() == y.tobytes(), k


def test_packed_mixed_sizes_preserve_order_and_budget():
    rng = np.random.default_rng(0)
    cfgs = _mixed(rng)
    ds = build_dataset(cfgs, META, np.zeros(1), 4, pack=True, log=lambda *a: None)
    n_cap = ds.node_z.shape[1]
    assert n_cap == 608                                                  # round_up(600, 32)
    live = _live_order(ds)
    assert len(live) == len(cfgs)                                        # each config in exactly one batch
    assert [n for *_, n in live] == [len(c.numbers) for c in cfgs]       # in input order
    # concatenating the live configs batch by batch reproduces the input labels in order
    yE = np.asarray(ds.y_E)[np.asarray(ds.cfg_mask)]
    np.testing.assert_array_equal(yE, [c.energy for c in cfgs])
    yF = np.asarray(ds.y_F)[np.asarray(ds.node_mask)]
    np.testing.assert_array_equal(yF, np.concatenate([c.forces for c in cfgs]))
    # every node's config slot is a live config of its batch; padded nodes point at C_eff
    C_eff = ds.y_E.shape[1]
    nm, nc = np.asarray(ds.node_mask), np.asarray(ds.node_cfg)
    assert np.all(nc[~nm] == C_eff)
    assert np.all(np.take_along_axis(np.asarray(ds.cfg_mask), np.where(nm, nc, 0), 1)[nm])
    total = sum(len(c.numbers) for c in cfgs)
    assert ds.n_batches * n_cap <= 2 * total + n_cap
    # within budget and the per-batch cap; padded config slots are masked with zero weight
    assert np.all(nm.sum(1) <= n_cap)
    assert np.all(np.asarray(ds.w_E)[~np.asarray(ds.cfg_mask)] == 0)
    # neighbours stay inside their own configuration
    nb = np.asarray(ds.nbr)
    own = np.take_along_axis(nc, nb.reshape(len(nb), -1), 1).reshape(nb.shape)
    assert np.all((own == nc[:, :, None])[np.asarray(ds.nbr_mask)])


def test_auto_packs_mixed_but_not_homogeneous():
    rng = np.random.default_rng(1)
    cfgs = _mixed(rng)
    lines = []
    auto = build_dataset(cfgs, META, np.zeros(1), 4, log=lines.append)
    on = build_dataset(cfgs, META, np.zeros(1), 4, pack=True, log=lambda *a: None)
    off = build_dataset(cfgs, META, np.zeros(1), 4, pack=False)
    _assert_same(auto, on)
    assert off.n_batches * off.node_z.shape[1] > 2 * auto.n_batches * auto.node_z.shape[1]
    assert len(lines) == 1 and "pack" in lines[0] and "608" in lines[0]
    # homogeneous: the 8-40 atom configs alone -> auto is the fixed-C layout, byte for byte
    small = [c for c in cfgs if len(c.numbers) < 600]
    lines.clear()
    _assert_same(build_dataset(small, META, np.zeros(1), 4, log=lines.append),
                 build_dataset(small, META, np.zeros(1), 4, pack=False))
    assert lines == []


@pytest.mark.parametrize("C", [1, 2, 3, 4, 8])
def test_auto_is_byte_identical_on_the_tiny_fixture(C):
    from ace_jax.fit.data import load_configs
    from ace_jax.eval import load
    _, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    cfgs = load_configs(str(FIXTURE_DIR / "si_tiny_train.xyz"), "dft_energy", "dft_force", "dft_virial")
    E0 = np.asarray(z["E0"])
    for sub in (cfgs, cfgs[:7], list(reversed(cfgs))[:30]):
        _assert_same(build_dataset(sub, meta, E0, C), build_dataset(sub, meta, E0, C, pack=False))


def test_pack_on_rejects_explicit_n_cap_below_largest_config():
    rng = np.random.default_rng(2)
    cfgs = [_si((1, 1, 1), rng), _si((2, 2, 2), rng)]                   # 8, 64 atoms
    with pytest.raises(ValueError, match="n_cap"):
        build_dataset(cfgs, META, np.zeros(1), 2, n_cap=32, pack=True)
    ds = build_dataset(cfgs, META, np.zeros(1), 2, n_cap=96, pack=True, log=lambda *a: None)
    assert ds.node_z.shape[1] == 96 and [n for *_, n in _live_order(ds)] == [8, 64]


def test_pack_rejects_unknown_mode():
    with pytest.raises(ValueError, match="pack"):
        build_dataset([_si((1, 1, 1), np.random.default_rng(0))], META, np.zeros(1), 1, pack="sometimes")


def _pipe_cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear", uq="ard",
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False, ard_n_min=1)
    return FitConfig(**{**base, **kw})


def test_fit_config_validates_batch_pack():
    with pytest.raises(ValueError, match="batch_pack"):
        _pipe_cfg(batch_pack="maybe").validate()
    for v in ("auto", "on", "off"):
        assert _pipe_cfg(batch_pack=v).validate().batch_pack == v


def test_cli_batch_pack_flag():
    from ace_jax import cli
    p = cli._add_fit_args(__import__("argparse").ArgumentParser())
    assert p.parse_args([]).batch_pack == "auto"
    assert p.parse_args(["--batch-pack", "off"]).batch_pack == "off"


def test_ard_stage_with_a_big_cell_is_pack_invariant():
    """The tiny fixture plus one 1000-atom Si cell: the schema-3 ARD stage (PRESS sandwich, groups,
    T_fit/T_val split, shell table, row clusters) gives the same posterior with pack auto as with
    pack off -- every Dataset-order -> config-list mapping survives packing."""
    from ace_jax.fit import ard
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    res = {}
    for mode in ("off", "auto"):
        cfg = _pipe_cfg(ard_variance="sandwich", batch_pack=mode).validate()
        d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None)
        big = _si((5, 5, 5), np.random.default_rng(3), rattle=0.1)
        big = big._replace(energy=float(np.mean([c.energy / len(c.numbers) for c in d.train if c.energy])
                                        * 1000 + 0.3),
                           forces=0.1 * np.random.default_rng(4).normal(size=(1000, 3)),
                           virial=np.random.default_rng(5).normal(size=(3, 3)))
        train = d.train[:12] + [big] + d.train[12:]
        ds = build_dataset(train, d.meta, d.E0, cfg.batch, pack=mode, log=lambda *a: None)
        d = d._replace(train=train, ds_train=ds)
        if mode == "auto":
            assert ds.y_E.shape[1] != cfg.batch and ds.n_batches < res["off"][3]   # it packed
        b = build_problem(cfg, d)
        lines = []
        with highest_precision():
            r = ard.run_ard_stage(cfg, d, b, default_prior(2.35).mu, log=lines.append)
        # auto: the stage's own T_fit / T_val datasets pack too (the big cell is in one of them)
        assert any("size-aware packing" in s for s in lines) == (mode == "auto")
        res[mode] = (r.posterior, r.report, b, ds.n_batches, ds)
    (p0, r0, *_), (p1, r1, *_) = res["off"], res["auto"]
    # the evidence statistics agree to summation order; the L-BFGS hyperparameter optimum (and so
    # the mean) to the optimizer's tolerance
    with highest_precision():
        s0 = ard.ard_statistics(default_prior(2.35).mu, res["off"][2].prob, res["off"][4], "joint")
        s1 = ard.ard_statistics(default_prior(2.35).mu, res["auto"][2].prob, res["auto"][4], "joint")
    for a, b in zip(jax.tree.leaves(s1), jax.tree.leaves(s0)):
        a, b = np.asarray(a, float), np.asarray(b, float)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * max(1.0, np.abs(b).max()))
    np.testing.assert_allclose(p1.h, p0.h, rtol=0, atol=1e-5)
    # the statistics agree to 1e-12 and h to 1e-5 above; the means come from two optimizer runs whose endpoints
    # drift by ~2e-6 relative across platforms (CI) -- a layout/index bug would be O(1)
    np.testing.assert_allclose(p1.mean, p0.mean, rtol=0, atol=1e-5 * np.abs(p0.mean).max())
    for k in ("lam_rms", "q", "r", "n_cfg", "n_cfg_val", "n_atoms"):
        np.testing.assert_allclose(np.asarray(p1.group_table[k], float),
                                   np.asarray(p0.group_table[k], float), rtol=1e-6, atol=1e-12)
    assert p1.group_table["merged"] == p0.group_table["merged"]
    assert r1["n_fit_configs"] == r0["n_fit_configs"] and r1["n_val_configs"] == r0["n_val_configs"]
    assert r1["n_val_atoms"] == r0["n_val_atoms"]


def test_pack_mode_keeps_the_fixed_layout_for_loo():
    assert _pipe_cfg().pack_mode == "auto"
    assert _pipe_cfg(objective="loo").pack_mode == "off"
    assert _pipe_cfg(objective="loo", batch_pack="on").pack_mode == "on"


def test_load_fit_data_passes_batch_pack(monkeypatch):
    from ace_jax.fit.pipeline import data as P, load_fit_data
    seen = []
    real = P.build_dataset
    monkeypatch.setattr(P, "build_dataset", lambda *a, **k: (seen.append(k.get("pack")), real(*a, **k))[1])
    load_fit_data(_pipe_cfg(arm="linear", uq="blr", batch_pack="off"),
                  data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None)
    assert seen == ["off", "off"]
