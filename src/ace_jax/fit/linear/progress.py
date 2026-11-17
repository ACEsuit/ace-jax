"""Machine-readable progress of a running fit: one JSON object per line.

Library code calls `emit(event, **fields)` at the points a person watching a long fit wants to see
(a stage starting or finishing, an L-BFGS evaluation, a radial-learning round).  It is a no-op unless
a driver has opened a sink with `progress_file(path)`; `aj fit` opens `<out>/progress.jsonl`.  The
human log (`log=`) is unchanged and independent: events add structure, they never replace a line.

Each record is {"time": UTC ISO-8601, "elapsed_s": since the sink opened, "event": name, **fields},
written and flushed as it happens, so `tail -f` or a JSON reader follows a fit live, and a killed fit
keeps every event up to the kill.  Field values are plain JSON (numpy and JAX scalars and small arrays
are converted).  Nothing here touches the numerics."""
import contextlib
import contextvars
import datetime
import json
import time

import numpy as np

_SINK = contextvars.ContextVar("ace_jax_progress", default=None)


def _plain(v):
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    a = np.asarray(v)
    if a.dtype.kind in "biuf":
        return a.item() if a.ndim == 0 else a.tolist()
    return str(v)


def emit(event, **fields):
    """Record one event in the open sink, if any (else nothing happens)."""
    sink = _SINK.get()
    if sink is not None:
        sink(event, fields)


def active():
    """Whether a sink is open (to skip computing a field only an event would use)."""
    return _SINK.get() is not None


@contextlib.contextmanager
def progress_file(path):
    """Append events to `path` (JSON lines, line-buffered) while the block runs."""
    t0 = time.time()
    f = open(path, "a", buffering=1)

    def sink(event, fields):
        rec = {"time": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
               "elapsed_s": round(time.time() - t0, 3), "event": event,
               **{k: _plain(v) for k, v in fields.items()}}
        f.write(json.dumps(rec, allow_nan=True) + "\n")
    tok = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(tok)
        f.close()


@contextlib.contextmanager
def stage(name, **fields):
    """A `stage` event with status "start", then "done" (or "failed", with the error) and its seconds."""
    t = time.time()
    emit("stage", name=name, status="start", **fields)
    try:
        yield
    except BaseException as e:
        emit("stage", name=name, status="failed", seconds=round(time.time() - t, 3),
             error=f"{type(e).__name__}: {e}"[:500])
        raise
    emit("stage", name=name, status="done", seconds=round(time.time() - t, 3))
