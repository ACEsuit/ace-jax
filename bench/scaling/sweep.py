"""Expand the benchmark matrix for a host and run it resumably.

    python bench/scaling/sweep.py moriarty-gpu [--only acejax-pace] [--dry-run]
"""
import dataclasses
import json
import os
import pathlib
import subprocess
import sys

from scaling.models import SIZES, planned_models
from scaling.structures import n_ladder

HOSTS = {
    "moriarty-cpu": {"device": "cpu", "n_max": 32768, "ranks": 32},
    "moriarty-gpu": {"device": "gpu", "n_max": 1 << 20, "ranks": 1},
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


def cases(host):
    h = HOSTS[host]
    out = []
    for m in planned_models():
        for mode in MODES[m["code"]]:
            if mode == "lammps" and not lammps_supported(m["code"], h["device"]):
                continue
            for dtype in DTYPES[m["code"]]:
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


def run_sweep(host, runner, results_path, select=lambda c: True):
    results_path = pathlib.Path(results_path)
    done, dead = set(), set()
    if results_path.exists():
        for l in results_path.read_text().splitlines():
            r = json.loads(l)
            done.add(tuple(r["_key"]))
            if r["status"] in ("oom", "error", "unstable", "parity_fail", "unsupported"):
                dead.add(tuple(r["_line"]))
    todo = sorted((c for c in cases(host) if select(c)), key=lambda c: (_line(c), c.n_atoms))
    for c in todo:
        if c.key() in done or _line(c) in dead:
            continue
        row = runner(c)
        if row.get("status") == "error":        # transient (e.g. GPU state): retry once
            first = row.get("error")
            row = runner(c)
            row["retried"], row["first_error"] = True, first
        row["_key"], row["_line"], row["host"] = list(c.key()), list(_line(c)), host
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if row["status"] != "ok":
            dead.add(_line(c))


def child_env(env):
    """The login environment (the lmp wrappers `module load`), plus overrides."""
    here = pathlib.Path(__file__).parent
    return {**os.environ, **env.get("os_env", {}),
            "PYTHONPATH": env.get("pythonpath", str(here.parent)),
            **({"PJRT_PLUGIN": env["pjrt"]} if env.get("pjrt") else {})}   # run_lammps reads it


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

    def run(c):
        if c.mode == "standalone":
            cmd = [sys.executable, str(here / "run_standalone.py"), c.model, str(c.n_atoms),
                   c.dtype, c.device]
        else:
            lmp = env.get("lmp_jax", env["lmp"]) if c.code.startswith("acejax") else env["lmp"]
            cmd = [sys.executable, str(here / "run_lammps.py"), c.model, str(c.n_atoms), c.dtype,
                   c.device, lmp, str(c.ranks), f"/tmp/bench_{host}_{c.n_atoms}"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=child_env(env))
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else {"status": "error", "error": p.stderr[-300:]}
    return run


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
                r["_key"], r["_line"] = ["parity", r["gate"], r["system"], r["code"]], ["parity"]
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
