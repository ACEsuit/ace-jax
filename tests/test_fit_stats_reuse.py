"""The fit's sufficient statistics are computed once and reused by prediction and by the
model file, wherever reuse is exact; and each training set's neighbour lists are built once.

predict_stats "auto" (the CLI's) reuses the objective's statistics on the QR path, where they
are bitwise the statistics a recompute streams (`linear_qr_statistics` on the same data), and
recomputes elsewhere (the GP and Cholesky objectives cache a split Gram, which differs from a
full recompute in summation order).  The model file reuses the posterior the MAP prediction
already factored.  So "auto" must reproduce "recompute" exactly, with fewer passes."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR  # noqa: E402

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")

BASE = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
            virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, rungs=("map",),
            predict_train=False, e0="lsq", opt="lbfgs", map_steps=5)
TOL = 1e-12


class Passes:
    """Counts the full statistics passes over the training set (by dataset identity)."""

    def __init__(self, monkeypatch):
        import ace_jax.fit.pipeline.objective as pobj
        import ace_jax.fit.predict as pred
        import ace_jax.fit.stats as st
        self.calls, self.ds = [], None
        for mod, name in ((st, "linear_qr_statistics"), (st, "linear_statistics"), (pobj, "linear_statistics"),
                          (st, "sufficient_statistics"), (pred, "sufficient_statistics")):
            monkeypatch.setattr(mod, name, self._wrap(name, getattr(mod, name)))

    def _wrap(self, name, f):
        def g(*a, **kw):
            ds = a[-1] if a else None
            self.calls.append((name, ds))
            return f(*a, **kw)
        return g

    def on_train(self, name):
        return sum(1 for n, ds in self.calls if n == name and ds is self.ds)


def _fit(passes=None, **kw):
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(**{**BASE, **kw}).validate()
    d = load_fit_data(cfg, data=str(XYZ), log=lambda *a: None)
    if passes is not None:
        passes.ds = d.ds_train
    return fit(cfg, d, log=lambda *a: None)


def _close(a, b, where):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, where
    assert np.allclose(a, b, rtol=TOL, atol=TOL * max(1.0, float(np.max(np.abs(b), initial=0))),
                       equal_nan=True), (where, float(np.max(np.abs(a - b), initial=0)))


def _same_metrics(a, b, where):
    if isinstance(a, dict):
        assert set(a) == set(b), where
        for k in a:
            _same_metrics(a[k], b[k], f"{where}.{k}")
    else:
        _close(a, b, where)


def _equivalent(r, c):
    """Predictions, metrics, the MAP posterior and the model file of r match control c."""
    from ace_jax.fit.pipeline.export import _posterior, gp_model_arrays, linear_model_arrays
    assert set(r.preds.arrays) == set(c.preds.arrays)
    for k in c.preds.arrays:
        for q in c.preds.arrays[k]:
            _close(r.preds.arrays[k][q], c.preds.arrays[k][q], f"{k}/{q}")
    _same_metrics(r.preds.metrics, c.preds.metrics, "metrics")
    if r.ard is None and "mean" not in r.preds.pops:
        for x, y, n in zip(_posterior(r, r.theta), _posterior(c, c.theta), ("mu", "L")):
            _close(x, y, n)
    arrs = (linear_model_arrays if r.built.prob.ind.XM.shape[0] == 0 else gp_model_arrays)
    ar, ac = arrs(r), arrs(c)
    assert set(ar) == set(ac)
    for k in ac:
        if ac[k].dtype.kind in "fc":
            _close(ar[k], ac[k], f"model.npz/{k}")
        else:
            assert np.array_equal(ar[k], ac[k]), k


CASES = {
    "linear_qr": dict(arm="linear"),
    "linear_cholesky": dict(arm="linear", lml_solver="cholesky"),
    "shared_noise": dict(arm="linear", noise="shared"),
    "pops": dict(arm="linear", uq="pops", e0="prefit", pops_ridge=1e-6),
    "ard": dict(arm="linear", uq="ard", m_per_species=0),
    "gp": dict(arm="gp", m_per_species=6, density="pair"),
    "linear_loo": dict(arm="linear", objective="loo"),     # a QR problem, a Gram-form objective
}


@pytest.mark.parametrize("case", list(CASES))
def test_reuse_reproduces_recompute(case):
    kw = CASES[case]
    _equivalent(_fit(predict_stats="auto", **kw), _fit(predict_stats="recompute", **kw))


@pytest.mark.parametrize("case", ["linear_qr", "shared_noise", "pops"])
def test_qr_fit_streams_its_statistics_once(case, monkeypatch, tmp_path):
    """One QR statistics pass over the training set, for the objective; prediction, the
    model file and the outputs (fitted E0 included) reuse it."""
    from ace_jax.fit.pipeline import write_outputs
    p = Passes(monkeypatch)
    res = _fit(p, predict_stats="auto", **CASES[case])
    write_outputs(res, tmp_path, layout=("run", "cli"), log=lambda *a: None)
    assert p.on_train("linear_qr_statistics") == 1, p.calls
    assert p.on_train("sufficient_statistics") == 0


def test_explicit_recompute_still_recomputes(monkeypatch, tmp_path):
    from ace_jax.fit.pipeline import write_outputs
    p = Passes(monkeypatch)
    res = _fit(p, predict_stats="recompute", arm="linear")
    write_outputs(res, tmp_path, layout=("cli",), log=lambda *a: None)
    assert p.on_train("linear_qr_statistics") == 2       # the objective's, and the prediction's
    assert (tmp_path / "model.npz").exists()            # written from the prediction's MAP posterior


def test_gp_model_file_reuses_the_map_prediction_posterior(monkeypatch, tmp_path):
    """The GP recomputes per draw (its LML caches a split Gram); the model file at the MAP
    takes the posterior the MAP prediction factored instead of a further pass."""
    from ace_jax.fit.pipeline import write_outputs
    p = Passes(monkeypatch)
    res = _fit(p, predict_stats="auto", **CASES["gp"])
    write_outputs(res, tmp_path, layout=("cli",), log=lambda *a: None)
    assert p.on_train("sufficient_statistics") == 1, p.calls     # the MAP prediction's only
    assert (tmp_path / "gp_model.npz").exists()


def test_split_gram_objective_streams_its_linear_gram_once(monkeypatch, tmp_path):
    """The Cholesky (and GP) objective's cached linear Gram serves the predictions too (no
    second linear pass).  Its split statistics are not bitwise a full recompute's, so the
    model file still takes one full pass (as before: the goldens hold it)."""
    from ace_jax.fit.pipeline import write_outputs
    p = Passes(monkeypatch)
    res = _fit(p, predict_stats="cached", arm="linear", lml_solver="cholesky")
    write_outputs(res, tmp_path, layout=("cli",), log=lambda *a: None)
    assert p.on_train("linear_statistics") == 1, p.calls
    assert p.on_train("sufficient_statistics") == 1, p.calls


def test_no_test_file_reuses_the_training_dataset(tmp_path):
    from ase.io import read, write
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    tr, te = tmp_path / "tr.xyz", tmp_path / "te.xyz"
    cfgs = read(XYZ, ":")
    write(tr, cfgs[:8]); write(te, cfgs[8:12])
    cfg = FitConfig(**BASE).validate()
    d = load_fit_data(cfg, train=str(tr), log=lambda *a: None)
    assert d.ds_test is d.ds_train
    d2 = load_fit_data(cfg, train=str(tr), test=str(te), log=lambda *a: None)
    assert d2.ds_test is not d2.ds_train


def test_build_dataset_lists_each_config_once(monkeypatch):
    """k_cap and the dense rows come from one neighbour list per config, and the batches
    are the ones a separate k_cap pass and per-batch list gave (bitwise)."""
    import ace_jax.eval.nlist as nl
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.eval import load
    monkeypatch.setenv("ACEJAX_NLIST", "ase")
    _, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    cfgs = load_configs(str(XYZ), energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
    E0 = np.zeros(len(meta["elements"]))
    real, calls = nl._neighbour_list, []
    monkeypatch.setattr(nl, "_neighbour_list", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ds = build_dataset(cfgs, meta, E0, 4, log=lambda *a: None)
    assert len(calls) == len(cfgs)
    k_cap = ds.nbr.shape[-1]
    calls.clear()
    ref = build_dataset(cfgs, meta, E0, 4, k_cap=k_cap, log=lambda *a: None)    # the per-batch path
    assert len(calls) == len(cfgs)
    for a, b, n in zip(ds, ref, ds._fields):
        assert np.array_equal(np.asarray(a), np.asarray(b)), n
