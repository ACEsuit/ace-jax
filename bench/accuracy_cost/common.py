"""Paths, the sweep table and the run-directory protocol shared by every
accuracy-cost driver.  Data and runs live on /storage ($ACEJAX_AC_ROOT): only
results/*.jsonl and figs/ are committed.  $ACEJAX_AC_RUNS and $ACEJAX_AC_RESULTS
move the run tree and the metrics file on their own (a smoke run beside the real one);
$ACEJAX_AC_HOST and $ACEJAX_AC_GPU stamp status.json (Modal fits: host "modal")."""
import json
import os
import pathlib
import resource
import socket
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("ACEJAX_AC_ROOT", "/storage/eng/essswb/projects/ace-jax"))
DATA = ROOT / "data" / "si_accuracy_cost"
RUNS = pathlib.Path(os.environ.get("ACEJAX_AC_RUNS") or ROOT / "runs" / "accuracy_cost")
RESULTS = pathlib.Path(os.environ.get("ACEJAX_AC_RESULTS") or HERE / "results")
RCUT = 5.0                                    # every family (spec: GAP-18's SOAP cutoff)
E0_SI = -158.54496821                         # the isolated_atom frame's dft_energy


def load_sweep(path=HERE / "sweep.yaml"):
    import yaml
    return yaml.safe_load(pathlib.Path(path).read_text())


def entry(family, size, sweep=None):
    for e in (sweep or load_sweep())[family]:
        if e["size"] == size:
            return e
    raise KeyError(f"{family}/{size} not in sweep.yaml")


def run_dir(family, size):
    return RUNS / family / size


def read_status(d):
    p = pathlib.Path(d) / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_status(d, **kw):
    d = pathlib.Path(d); d.mkdir(parents=True, exist_ok=True)
    gpu = {"gpu": os.environ["ACEJAX_AC_GPU"]} if os.environ.get("ACEJAX_AC_GPU") else {}
    tmp = d / "status.json.tmp"             # atomic: a Volume commit or reader never sees half a file
    tmp.write_text(json.dumps({"host": os.environ.get("ACEJAX_AC_HOST") or socket.gethostname(), **gpu, **kw}, indent=1))
    os.replace(tmp, d / "status.json")


def tail(path, n=40):
    p = pathlib.Path(path)
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else ""


def run_logged(cmd, cwd, log="fit.log", env=None):
    """Run `cmd` in `cwd` with output to `log`: (returncode, wall_s, peak_rss_gb).
    Peak RSS is RUSAGE_CHILDREN's max, so one fit per driver process."""
    cwd = pathlib.Path(cwd); cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "cmd.txt").write_text(" ".join(map(str, cmd)) + "\n")
    t0 = time.perf_counter()
    with open(cwd / log, "w") as f:
        rc = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return rc, time.perf_counter() - t0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6


def fit_once(family, size, model_name, cmd, env=None, split_hash=None):
    """The resumable protocol: an existing model is never refitted; a failure
    writes status 'failed' with the log tail and is not retried."""
    d = run_dir(family, size)
    if (d / model_name).exists() and (read_status(d) or {}).get("status") == "ok":
        return read_status(d)
    rc, wall, peak = run_logged(cmd, d, env=env)
    ok = rc == 0 and (d / model_name).exists()
    write_status(d, family=family, size=size, status="ok" if ok else "failed", model=str(d / model_name),
                 fit_wall_s=wall, fit_peak_rss_gb=peak, split_hash=split_hash,
                 **({} if ok else {"returncode": rc, "error": tail(d / "fit.log")}))
    return read_status(d)


def upsert_jsonl(path, row, key):
    """Replace the row with the same `key` fields, else append; one row per key."""
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    k = tuple(row.get(f) for f in key)
    rows = [r for r in rows if tuple(r.get(f) for f in key) != k] + [row]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
