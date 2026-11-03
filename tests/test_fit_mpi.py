"""MPI-sharded training statistics (`aj fit --mpi`, the [mpi] extra): each rank streams every size-th batch into its
own QR factors, a binary tree merges them onto rank 0, and rank 0 alone runs the rest of the fit."""
import jax

jax.config.update("jax_enable_x64", True)
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.fit import stats  # noqa: E402


class _DS(NamedTuple):
    P: np.ndarray
    y: np.ndarray

    @property
    def n_batches(self):
        return self.P.shape[0]


def _problem(nb=11, r=6, L=5, seed=0):
    g = np.random.default_rng(seed)
    return _DS(g.normal(size=(nb, r, L)), g.normal(size=(nb, r))), [(np.zeros((L, L)), np.zeros(L))] * 2


def _rows(b):
    return [(b.P, b.y), (3.0 * b.P, b.y)]


def _gram(f):
    return [(np.asarray(R).T @ np.asarray(R), np.asarray(R).T @ np.asarray(c)) for R, c in f]


def test_shard_batches_is_interleaved_and_complete():
    for n, size in ((11, 1), (11, 3), (4, 4), (3, 5)):
        parts = [stats.shard_batches(n, r, size) for r in range(size)]
        assert sorted(i for p in parts for i in p) == list(range(n))
        assert all(list(p) == list(range(r, n, size)) for r, p in enumerate(parts))


def test_merged_shards_match_the_serial_pass():
    ds, init = _problem()
    serial = stats.host_qr_stream(_rows, ds, init, merge_rows=4)
    for size in (2, 3, 5):
        shards = [stats.host_qr_stream(_rows, ds, init, merge_rows=4, batches=stats.shard_batches(ds.n_batches, r, size))
                  for r in range(size)]
        merged = shards[0]
        for other in shards[1:]:
            merged = stats.merge_factors(merged, other)
        for (G, b), (G0, b0) in zip(_gram(merged), _gram(serial)):
            np.testing.assert_allclose(G, G0, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(b, b0, rtol=1e-12, atol=1e-12)


def test_tree_schedule_reaches_rank_zero_once_per_rank():
    for size in (1, 2, 3, 4, 7, 8):
        rounds = stats.tree_schedule(size)
        sent = set()
        for rnd in rounds:
            for src, dst in rnd:
                assert src not in sent and dst not in sent            # once sent, a rank is done
                assert src != 0
            sent |= {src for src, _ in rnd}
        assert sent == set(range(1, size))                            # every other rank's factors reach rank 0
        assert len(rounds) == (0 if size == 1 else int(np.ceil(np.log2(size))))


def _mpiexec():
    exe = shutil.which("mpiexec", path=str(__import__("pathlib").Path(sys.executable).parent)) or shutil.which("mpiexec")
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return exe


def test_mpi_fit_matches_serial(tmp_path):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip("mpi4py / mpiexec not installed (the [mpi] extra)")
    from conftest import FIXTURE_DIR
    args = ["-m", "ace_jax.cli", "fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"),
            "--data", str(FIXTURE_DIR / "si_tiny_train.xyz"), "--ntrain", "20", "--ntest", "6",
            "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
            "--m-per-species", "0", "--opt", "lbfgs", "--map-steps", "8", "--configs-per-batch", "2", "--r0", "2.35"]
    env = {**__import__("os").environ, "JAX_PLATFORMS": "cpu"}
    subprocess.run([sys.executable, *args, "--out", str(tmp_path / "serial")], check=True, env=env,
                   capture_output=True)
    p = subprocess.run([mpiexec, "-n", "3", sys.executable, *args, "--mpi", "--out", str(tmp_path / "mpi")],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "mpi: 3 ranks" in p.stdout
    a = json.loads((tmp_path / "serial" / "theta_map.json").read_text())
    b = json.loads((tmp_path / "mpi" / "theta_map.json").read_text())
    for k in a:
        assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-8), k
    assert not (tmp_path / "mpi" / "progress.rank1.jsonl").exists()   # only rank 0 reports
