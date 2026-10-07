"""fit.progress: the JSON-lines event sink `aj fit` writes as it runs (tests/test_fit_learn_radial.py
covers the events of a real fit)."""
import json

import numpy as np
import pytest

from ace_jax.fit.progress import active, emit, progress_file, stage


def _read(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_emit_is_a_no_op_without_a_sink():
    assert not active()
    emit("anything", x=1)                     # nothing to write to, nothing raised


def test_events_are_written_live_and_plain_json(tmp_path):
    p = tmp_path / "progress.jsonl"
    with progress_file(p):
        assert active()
        emit("a", x=np.float64(1.5), v=np.arange(3), n=np.int32(4), d={"k": np.float32(2.0)})
        assert _read(p)[0]["event"] == "a"     # flushed per line: readable before the block ends
    assert not active()
    (r,) = _read(p)
    assert r["x"] == 1.5 and r["v"] == [0, 1, 2] and r["n"] == 4 and r["d"] == {"k": 2.0}
    assert {"time", "elapsed_s"} <= set(r)


def test_stage_records_start_done_and_failure(tmp_path):
    p = tmp_path / "progress.jsonl"
    with progress_file(p):
        with stage("ok", k=1):
            pass
        with pytest.raises(ValueError), stage("bad"):
            raise ValueError("boom")
    ev = [(e["name"], e["status"]) for e in _read(p)]
    assert ev == [("ok", "start"), ("ok", "done"), ("bad", "start"), ("bad", "failed")]
    assert "boom" in _read(p)[-1]["error"] and _read(p)[1]["seconds"] >= 0


def test_lbfgs_loop_reports_its_starting_value():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.fit.radial_learn import lbfgs_loop
    info = {}
    _, fx, _, _ = lbfgs_loop(lambda x: jnp.sum((x - 1.0) ** 2), jnp.zeros(3), steps=5, info=info)
    assert info["f0"] == 3.0 and fx < info["f0"]
