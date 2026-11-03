"""Resumable fits: the streamed QR statistics and the MAP's best hyperparameters checkpointed to a directory
(`aj fit --checkpoint DIR`; stats.QRCheckpoint, pipeline.mapfit)."""
import jax

jax.config.update("jax_enable_x64", True)
import json  # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.fit import stats  # noqa: E402


class _DS(NamedTuple):
    P: np.ndarray        # (n_batches, rows, L)
    y: np.ndarray        # (n_batches, rows)

    @property
    def n_batches(self):
        return self.P.shape[0]


def _problem(nb=9, r=7, L=5, seed=0):
    g = np.random.default_rng(seed)
    return _DS(g.normal(size=(nb, r, L)), g.normal(size=(nb, r))), [(np.zeros((L, L)), np.zeros(L))] * 2


def _rows(calls, extra, fail_at=None):
    def f(b):
        calls.append(1)
        if fail_at is not None and len(calls) > fail_at:
            raise KeyboardInterrupt("killed")
        extra["s"] = extra.get("s", 0.0) + float(np.sum(b.y))
        return [(b.P, b.y), (2.0 * b.P, b.y)]
    return f


def _run(ds, init, ck, key="k", fail_at=None):
    calls, extra = [], {}
    out = stats.host_qr_stream(_rows(calls, extra, fail_at), ds, init, merge_rows=10, checkpoint=ck, key=key,
                               extra=extra)
    return [(np.asarray(R), np.asarray(c)) for R, c in out], extra, len(calls)


def test_qr_checkpoint_resume_matches_uninterrupted(tmp_path):
    ds, init = _problem()
    ref, ref_extra, _ = _run(ds, init, stats.QRCheckpoint(tmp_path / "ref", every=0))
    ck = stats.QRCheckpoint(tmp_path / "run", every=0)
    with pytest.raises(KeyboardInterrupt):
        _run(ds, init, ck, fail_at=5)                                  # killed after 5 batches
    out, extra, n = _run(ds, init, ck)
    assert n == ds.n_batches - 5                                       # only the batches not yet merged
    for (R, c), (R0, c0) in zip(out, ref):
        np.testing.assert_array_equal(R, R0)
        np.testing.assert_array_equal(c, c0)
    assert extra == pytest.approx(ref_extra, rel=1e-15)


def test_qr_checkpoint_matches_no_checkpoint_to_roundoff(tmp_path):
    ds, init = _problem()
    a, _, _ = _run(ds, init, None)
    b, _, _ = _run(ds, init, stats.QRCheckpoint(tmp_path, every=0))
    for (R, c), (R0, c0) in zip(a, b):
        np.testing.assert_allclose(R.T @ R, R0.T @ R0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(R.T @ c, R0.T @ c0, rtol=1e-12, atol=1e-12)


def test_qr_checkpoint_complete_is_reused(tmp_path):
    ds, init = _problem()
    ck = stats.QRCheckpoint(tmp_path, every=3600)                      # no mid-stream saves, only the final one
    a, ea, n0 = _run(ds, init, ck)
    b, eb, n1 = _run(ds, init, ck)
    assert n0 == ds.n_batches and n1 == 0
    for (R, _c), (R0, _c0) in zip(a, b):
        np.testing.assert_array_equal(R, R0)
    assert ea == eb


def test_qr_checkpoint_other_key_is_ignored(tmp_path):
    ds, init = _problem()
    ck = stats.QRCheckpoint(tmp_path, every=0)
    _run(ds, init, ck, key="k1")
    _, _, n = _run(ds, init, ck, key="k2")
    assert n == ds.n_batches                                           # a different problem: recomputed


def test_fit_checkpoint_writes_theta_and_resumes(tmp_path, capsys):
    from ace_jax.cli import main
    from conftest import FIXTURE_DIR
    args = ["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(FIXTURE_DIR / "si_tiny_train.xyz"),
            "--ntrain", "20", "--ntest", "6", "--energy-key", "dft_energy", "--force-key", "dft_force",
            "--virial-key", "dft_virial", "--m-per-species", "0", "--opt", "lbfgs", "--map-steps", "6",
            "--configs-per-batch", "4", "--r0", "2.35", "--checkpoint", str(tmp_path / "ck")]
    assert main(args + ["--out", str(tmp_path / "a")]) == 0
    files = sorted(p.name for p in (tmp_path / "ck").iterdir())
    assert "theta_ckpt.json" in files and any(f.startswith("qrstats-") for f in files)
    th = json.loads((tmp_path / "ck" / "theta_ckpt.json").read_text())
    assert "log_sigma_F" in th
    import re
    evals = lambda log: [float(v) for v in re.findall(r"eval \d+  logpost = (\S+)", log)]  # noqa: E731
    best_a = max(evals(capsys.readouterr().out))
    assert main(args + ["--out", str(tmp_path / "b")]) == 0
    log = capsys.readouterr().out
    assert "checkpoint: QR statistics complete" in log and "checkpoint: MAP starts from" in log
    assert evals(log)[0] >= best_a - 1e-6 * abs(best_a)              # resumes from the best point, not the prior
