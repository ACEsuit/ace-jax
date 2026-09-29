"""Expand the benchmark matrix for a host and run it resumably.

    python bench/scaling/sweep.py moriarty-gpu [--only acejax-pace] [--dry-run]
"""
import dataclasses
import json
import os
import pathlib
import subprocess
import sys

from scaling.models import planned_models
from scaling.structures import n_ladder

HOSTS = {
    # Xeon Silver 4216: 16 cores x 2 hyperthreads -- one MPI rank per physical core
    # rss_cap_gb: a case's whole process tree is killed past this (62 GB node)
    "moriarty-cpu": {"device": "cpu", "n_max": 32768, "ranks": 16, "rss_cap_gb": 48},
    "moriarty-gpu": {"device": "gpu", "n_max": 1 << 20, "ranks": 1, "rss_cap_gb": 48},
    "modal-a100": {"device": "gpu", "n_max": 1 << 21, "ranks": 1},
    "local-cpu": {"device": "cpu", "n_max": 8192, "ranks": 8},
}
MODES = {"acejax-pace": ("standalone", "lammps"), "acejax-ace": ("standalone", "lammps"),
         "mlpace": ("lammps",), "mace": ("standalone", "lammps")}
DTYPES = {"acejax-pace": ("float64", "float32"), "acejax-ace": ("float64", "float32"),
          "mlpace": ("float64",), "mace": ("float64", "float32")}
# lammps-jax ships only pair_style jax/kk, which needs KOKKOS built with CUDA
GPU_ONLY_LAMMPS = ("acejax-pace", "acejax-ace")


def lammps_supported(code, device):
    return device == "gpu" or code not in GPU_ONLY_LAMMPS


@dataclasses.dataclass(frozen=True)
class Case:
    code: str
    model: str
    mode: str
    n_atoms: int
    dtype: str
    device: str
    ranks: int

    def key(self):
        return (self.model, self.mode, self.n_atoms, self.dtype, self.device)


# Symmetrix evaluates in double whatever the input precision: one LAMMPS dtype
LAMMPS_DTYPES = {"mace": ("float64",)}


def cases(host):
    h = HOSTS[host]
    out = []
    for m in planned_models():
        for mode in MODES[m["code"]]:
            if mode == "lammps" and not lammps_supported(m["code"], h["device"]):
                continue
            dtypes = LAMMPS_DTYPES.get(m["code"], DTYPES[m["code"]]) if mode == "lammps" \
                else DTYPES[m["code"]]
            for dtype in dtypes:
                for n in n_ladder(m["system"], h["n_max"]):
                    out.append(Case(m["code"], m["name"], mode, n, dtype, h["device"], h["ranks"]))
    return out


def merge_rows(existing, incoming):
    """(existing + incoming rows whose _key is new, number added); keeps order."""
    seen = {json.dumps(json.loads(l).get("_key")) for l in existing.splitlines() if l.strip()}
    out, added = existing, 0
    for l in incoming.splitlines():
        if not l.strip():
            continue
        k = json.dumps(json.loads(l).get("_key"))
        if k not in seen:
            seen.add(k)
            out += ("" if not out or out.endswith("\n") else "\n") + l + "\n"
            added += 1
    return out, added


def seed_for(results_text, snapshot_dir, code):
    """Resume point for one code: the local results plus a saved snapshot of it."""
    snap = pathlib.Path(snapshot_dir) / f"{code}.jsonl"
    return merge_rows(results_text, snap.read_text() if snap.exists() else "")[0]


def _line(c):
    return (c.model, c.mode, c.dtype, c.device)


_DEVICE_NAMES = {}


def device_name(device):
    """What a case ran on: the GPU's name (nvidia-smi; Modal's A100-80GB is an
    SXM4 or a PCIe card, which time differently), else the CPU model."""
    if device not in _DEVICE_NAMES:
        name = ""
        if device == "gpu":
            try:
                p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=30)
                name = p.stdout.strip().splitlines()[0] if p.returncode == 0 and p.stdout.strip() else ""
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            try:
                name = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                             if l.startswith("model name")), "")
            except OSError:
                pass
            import platform
            name = name or platform.processor() or platform.machine()
        _DEVICE_NAMES[device] = name or "unknown"
    return _DEVICE_NAMES[device]


def run_sweep(host, runner, results_path, select=lambda c: True):
    """Run the cases in order (line by line, ascending n), resumably.  Each case
    gets the previous ok row of its line, which sizes its step count."""
    results_path = pathlib.Path(results_path)
    done, dead, prev = set(), set(), {}
    if results_path.exists():
        for l in results_path.read_text().splitlines():
            r = json.loads(l)
            done.add(tuple(r["_key"]))
            if r["status"] in ("oom", "error", "unstable", "parity_fail", "unsupported"):
                dead.add(tuple(r["_line"]))
            elif r["status"] == "ok" and r.get("mode") != "parity":
                ln = tuple(r["_line"])
                if ln not in prev or r["_key"][2] > prev[ln]["_key"][2]:
                    prev[ln] = r
    todo = sorted((c for c in cases(host) if select(c)), key=lambda c: (_line(c), c.n_atoms))
    for c in todo:
        if c.key() in done or _line(c) in dead:
            continue
        row = runner(c, prev.get(_line(c)))
        if row.get("status") == "error":        # transient (e.g. GPU state): retry once
            first = row.get("error")
            row = runner(c, prev.get(_line(c)))
            row["retried"], row["first_error"] = True, first
        row["_key"], row["_line"], row["host"] = list(c.key()), list(_line(c)), host
        row["device_name"] = device_name(c.device)
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if row["status"] != "ok":
            dead.add(_line(c))
        else:
            prev[_line(c)] = row


def child_env(env, mode=None, cpus=None):
    """The login environment (the lmp wrappers `module load`), plus overrides.
    Threads: a standalone case is one process and gets every core (torch reads
    OMP_NUM_THREADS; moriarty's login env pins it to 1); MPI LAMMPS runs one
    thread per rank."""
    here = pathlib.Path(__file__).parent
    out = {**os.environ, **env.get("os_env", {}),
           "PYTHONPATH": env.get("pythonpath", str(here.parent)),
           **({"PJRT_PLUGIN": env["pjrt"]} if env.get("pjrt") else {})}   # run_lammps reads it
    # torch reads MKL_NUM_THREADS as well (moriarty's login env pins both to 1)
    if mode == "standalone" and cpus:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            out[v] = str(cpus)
    elif mode == "lammps":
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            out[v] = "1"
    return out


def gate_in_subprocess(host, env):
    """Parity rows from a child process: the gate imports JAX and torch, and a
    parent holding GPU memory would starve every case after it."""
    p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), host, "--gate-rows"],
                       capture_output=True, text=True, timeout=7200, env=child_env(env))
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    if not rows:
        raise RuntimeError(f"parity gate produced no rows:\n{p.stderr[-2000:]}")
    return rows


def _env_path(host):
    return pathlib.Path(__file__).parent / "envs" / f"{host}.json"


def subprocess_runner(host, env):
    """Real runner: one case per fresh process (so peak memory is per case)."""
    here = pathlib.Path(__file__).parent

    def run(c, prev=None):
        if c.mode == "standalone":
            cmd = [sys.executable, str(here / "run_standalone.py"), c.model, str(c.n_atoms),
                   c.dtype, c.device]
        else:
            lmp = env.get("lmp_jax", env["lmp"]) if c.code.startswith("acejax") else env["lmp"]
            cmd = [sys.executable, str(here / "run_lammps.py"), c.model, str(c.n_atoms), c.dtype,
                   c.device, lmp, str(c.ranks), f"/tmp/bench_{host}_{c.n_atoms}"]
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        e = child_env(env, mode=c.mode, cpus=cpus)
        if prev and prev.get("step_s"):                  # sizes the LAMMPS step count
            e["BENCH_PREV"] = json.dumps({"step_s": prev["step_s"], "n_atoms": prev["n_atoms"],
                                          "layout": prev.get("layout")})
        cap = HOSTS[host].get("rss_cap_gb")
        rc, out, err, peak, capped = run_capped(cmd, e, timeout=7200,
                                                cap_bytes=cap and cap * 2**30)
        return row_from_process(c, rc, out, err, capped=capped, peak_rss=peak)
    return run


def row_from_process(c, returncode, stdout, stderr, capped=False, peak_rss=None):
    """The case's JSON row, or -- when it died without printing one -- a row
    saying why: over the host memory cap (run_capped killed it), or SIGKILL
    (the kernel OOM killer: dmesg "Out of memory: Killed process") is an oom,
    anything else an error."""
    lines = [l for l in stdout.splitlines() if l.startswith("{")]
    if lines and not capped:
        return json.loads(lines[-1])
    row = {"code": c.code, "mode": c.mode, "model": c.model, "n_atoms": c.n_atoms,
           "dtype": c.dtype, "device": c.device, "returncode": returncode,
           "status": "oom" if (capped or returncode == -9) else "error", "error": stderr[-300:]}
    if capped:
        row["capped_rss"] = peak_rss
    return row


def run_capped(cmd, env, timeout, cap_bytes=None, poll=0.5):
    """Run cmd in its own session; kill the whole process tree (MPI ranks too)
    if its summed resident memory passes cap_bytes, so the case -- not the
    kernel OOM killer, which on moriarty took systemd and dbus first -- pays.
    Returns (returncode, stdout, stderr, peak_rss, capped); returncode None
    means it was killed for running past `timeout`."""
    import signal
    import tempfile
    import time
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, text=True, env=env, start_new_session=True)
        t0, peak, capped = time.monotonic(), 0, False
        while p.poll() is None:
            if cap_bytes:
                rss = _tree_rss(p.pid)
                peak = max(peak, rss)
                if rss > cap_bytes:
                    capped = True
                    _kill_tree(p.pid, signal.SIGKILL)
                    break
            if time.monotonic() - t0 > timeout:
                _kill_tree(p.pid, signal.SIGKILL)
                p.wait()
                fo.seek(0); fe.seek(0)
                return None, fo.read(), fe.read() + f"\ntimeout after {timeout} s", peak, False
            time.sleep(poll)
        rc = p.wait()
        fo.seek(0); fe.seek(0)
        return rc, fo.read(), fe.read(), peak, capped


def _tree_rss(pid):
    import psutil
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def _kill_tree(pid, sig):
    try:
        os.killpg(pid, sig)                    # the case's own session / process group
    except ProcessLookupError:
        pass


def main(argv=None):
    import argparse
    import collections
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("host", choices=sorted(HOSTS))
    ap.add_argument("--only", help="run only this code (e.g. acejax-pace)")
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="count cases, run nothing")
    ap.add_argument("--gate-rows", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--results", help="default: bench/scaling/results/<host>.jsonl")
    a = ap.parse_args(argv)
    here = pathlib.Path(__file__).parent
    select = (lambda c: c.code == a.only) if a.only else (lambda c: True)
    if a.dry_run:
        n = collections.Counter((c.code, c.mode) for c in cases(a.host) if select(c))
        for k, v in sorted(n.items()):
            print(f"{k[0]:12s} {k[1]:10s} {v}")
        return
    env = json.loads(_env_path(a.host).read_text())
    env.setdefault("pythonpath", str(here.parent))
    if a.gate_rows:                                    # child side of gate_in_subprocess
        from scaling import parity
        for r in parity.gate(a.host, env):
            print(json.dumps(r))
        return
    res = pathlib.Path(a.results or here / "results" / f"{a.host}.jsonl")
    res.parent.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(l) for l in res.read_text().splitlines()] if res.exists() else []
    rows = [r for r in prior if r.get("mode") == "parity"]
    if a.parity_only or not rows:                      # a resumed sweep keeps its gate
        rows = gate_in_subprocess(a.host, env)
        with res.open("a") as f:
            for r in rows:
                # the bundle layout keeps the dense and sparse ace-jax gates distinct
                r["_key"] = ["parity", r["gate"], r["system"], r["code"]] + (
                    [r["bundle_layout"]] if r.get("bundle_layout") else [])
                r["_line"] = ["parity"]
                r["device_name"] = device_name(r.get("device", "gpu"))
                f.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"parity {r['gate']:7s} {r['system']:7s} {r['model']:28s} {r['status']}"
              + (f"  dE/atom {r['dE_per_atom']:.1e} dF {r['max_dF']:.1e}" if "dE_per_atom" in r else ""))
    if a.parity_only:
        return
    from scaling.parity import blocked                 # plain python: no JAX/torch import
    block = blocked(rows)
    run_sweep(a.host, subprocess_runner(a.host, env), res,
              select=lambda c: select(c) and (c.code, c.mode) not in block)


if __name__ == "__main__":
    main()
