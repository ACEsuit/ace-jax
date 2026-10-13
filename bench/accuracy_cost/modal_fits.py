"""The PACE ladder and the MACE sweep fitted on Modal GPUs (fit time is recorded,
with the GPU, but is not the primary measurement; scoring and timing stay local).

    modal run bench/accuracy_cost/modal_fits.py --mode smoke     # tiny fits -> runs/accuracy_cost_smoke
    modal run bench/accuracy_cost/modal_fits.py --mode probe     # PACE cost per ladder rung -> runs/accuracy_cost_probe
    modal deploy bench/accuracy_cost/modal_fits.py               # then, detached:
    modal run bench/accuracy_cost/modal_fits.py --mode launch    # spawn on the deployed app, save call ids
    modal run bench/accuracy_cost/modal_fits.py --mode collect   # download + Symmetrix export; re-runnable

Images: pacemaker is envs/pacemaker's uv.lock (Python 3.9, python-ace 0.2.8, TF 2.15.1 with
its pip CUDA 12.2 / cuDNN 8.9 wheels) plus tensorpotential at the data.md commit; MACE is
envs/mace's uv.lock (mace-torch 0.3.16, torch 2.14.1 cu130).  Each venv lives in /opt and the
fit runs as `<venv>/bin/python -m accuracy_cost.fit_*`, exactly as locally; the container's own
Python only drives it and commits the Volume.  No Symmetrix in the image: the .json export
runs locally at collect time (modal_collect.symmetrix_local).

The Volume holds data/si_accuracy_cost (uploaded once) and runs/<root>/{pace,mace}, with
ACEJAX_AC_ROOT=/vol so common.DATA and common.RUNS point into it; it is committed every two
minutes.  A finished fit is never refitted (common.fit_once); a retried or relaunched call
resumes: MACE from ckpt/ (--restart_latest), PACE from its last ladder checkpoint in
ladder/resume<k>/ (fit_pace.resume; the full ladder is ~22-26 h of A100 compute by the probe, so
it may need a second 24 h attempt).  `--mode launch --only pace` relaunches one fit by hand.  status.json records host "modal" and the GPU nvidia-smi names.
"""
import json
import pathlib
import sys

import modal

# the image paths below are only read locally; in the container this file may sit at /root
HERE = pathlib.Path(__file__).resolve().parent if modal.is_local() else pathlib.Path("/ace-jax/bench/accuracy_cost")
if modal.is_local():
    sys.path.insert(0, str(HERE.parent))                # accuracy_cost.* for the local entrypoint
ROOT = HERE.parents[1]
APP = "ace-jax-accuracy-cost"
GPU = ["A100-40GB", "A100-80GB"]                        # FP64 fits (TF float64, MACE float64): A100, not L40S
TP_COMMIT = "1e44b2558356800ae070658c0bb856ff9bf74538"   # ICAMS/TensorPotential, as data.md
PACE_PY, MACE_PY = "/opt/pacemaker/bin/python", "/opt/mace/bin/python"
REAL = "runs/accuracy_cost"
SMOKE = "runs/accuracy_cost_smoke"
SMOKE_PACE = {"max_functions": 20, "ladder_step": 10, "take": [10, 20], "maxiter": 20, "n_train": 100}
SMOKE_MACE = {"epochs": 2, "n_train": 100}
PROBE = "runs/accuracy_cost_probe"
PROBE_PACE = {"maxiter": 3}            # the full ladder, 3 BFGS steps per rung: time/eval at every size


def _env_image(name, py, extra=()):
    envs = HERE / "envs" / name
    return (modal.Image.debian_slim(python_version="3.11")
            .apt_install("git")
            .pip_install("uv")
            .add_local_file(envs / "pyproject.toml", f"/opt/{name}-env/pyproject.toml", copy=True)
            .add_local_file(envs / "uv.lock", f"/opt/{name}-env/uv.lock", copy=True)
            .run_commands(f"cd /opt/{name}-env && UV_PROJECT_ENVIRONMENT=/opt/{name} uv sync --frozen"
                          f" --no-install-project -p {py}", *extra))


pace_image = _env_image("pacemaker", "3.9", (
    f"git clone https://github.com/ICAMS/TensorPotential /tmp/tp && cd /tmp/tp && git checkout {TP_COMMIT}"
    " && uv pip install --python /opt/pacemaker/bin/python --no-deps . && rm -rf /tmp/tp",))
mace_image = _env_image("mace", "3.12").env(
    # torch's cu130 wheel JIT-compiles with NVRTC, which dlopens libnvrtc-builtins.so.13.0 by name
    {"LD_LIBRARY_PATH": "/opt/mace/lib/python3.12/site-packages/nvidia/cu13/lib"})


def _with_bench(img):
    return img.add_local_dir(ROOT / "bench", "/ace-jax/bench",
                             ignore=["**/__pycache__", "scaling/results/*", "pace_modal/*", "**/*.png"])


app = modal.App(APP)
vol = modal.Volume.from_name(APP, create_if_missing=True)


def _drive(cmd, runs):
    """Run one driver command against the Volume, committing it every two minutes; the
    resulting status.json files (container paths) are returned."""
    import os
    import subprocess
    import time
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    env = {**os.environ, "ACEJAX_AC_ROOT": "/vol", "ACEJAX_AC_RUNS": f"/vol/{runs}", "ACEJAX_AC_HOST": "modal",
           "ACEJAX_AC_GPU": gpu[0] if gpu else "unknown", "PYTHONPATH": "/ace-jax/bench",
           "PYTHONUNBUFFERED": "1"}
    vol.reload()
    print("gpu:", env["ACEJAX_AC_GPU"], "cmd:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd="/ace-jax/bench", env=env)
    while proc.poll() is None:
        time.sleep(120)
        vol.commit()
    vol.commit()
    out = {str(p): json.loads(p.read_text()) for p in pathlib.Path(f"/vol/{runs}").glob("*/*/status.json")}
    if proc.returncode:
        raise RuntimeError(f"{' '.join(cmd)}: exit {proc.returncode}")
    return out


@app.function(image=_with_bench(pace_image), gpu=GPU, timeout=24 * 3600, volumes={"/vol": vol}, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))   # a retry resumes the ladder (fit_pace.resume)
def fit_pace(overrides: str = "", runs: str = REAL):
    """The whole ladder plus the .yace conversion, as fit_pace.main()."""
    import subprocess
    gpus = subprocess.run([PACE_PY, "-c", "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"],
                          capture_output=True, text=True).stdout.strip()
    print("tensorflow GPUs:", gpus, flush=True)
    if "GPU:0" not in gpus:                             # TF would silently fit on the CPU
        raise RuntimeError(f"tensorflow sees no GPU: {gpus!r}")
    return _drive([PACE_PY, "-m", "accuracy_cost.fit_pace", *([overrides] if overrides else [])], runs)


@app.function(image=_with_bench(mace_image), gpu=GPU, timeout=12 * 3600, volumes={"/vol": vol}, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=1, initial_delay=60.0))
def fit_mace(size: str, overrides: str = "", runs: str = REAL):
    """One MACE size with --device=cuda; the Symmetrix .json is exported locally (collect)."""
    import os
    os.environ["ACEJAX_AC_DEVICE"] = "cuda"
    return _drive([MACE_PY, "-m", "accuracy_cost.fit_mace", size, "--no-symmetrix",
                   *(["--overrides", overrides] if overrides else [])], runs)


def _upload(force=False):
    from accuracy_cost import common
    from accuracy_cost.modal_collect import DATA_FILES, REMOTE_DATA
    have = set()
    try:
        have = {e.path.lstrip("/").rsplit("/", 1)[-1] for e in vol.listdir(REMOTE_DATA)}
    except Exception:
        pass
    want = json.loads((common.DATA / "split.json").read_text())["hash"]
    if not force and set(DATA_FILES) <= have:
        import io
        b = io.BytesIO(); vol.read_file_into_fileobj(f"{REMOTE_DATA}/split.json", b)
        if json.loads(b.getvalue())["hash"] == want:
            print(f"data on the Volume (split {want})"); return
    with vol.batch_upload(force=True) as up:
        for f in DATA_FILES:
            up.put_file(str(common.DATA / f), f"{REMOTE_DATA}/{f}")
    print(f"uploaded {len(DATA_FILES)} data files (split {want})")


def _collect(runs):
    from accuracy_cost.modal_collect import MACE_SIZES, collect, complete, statuses, symmetrix_local
    dest = collect(vol, runs)
    done = symmetrix_local(dest, MACE_SIZES)
    print("symmetrix .json:", done)
    for k, st in statuses(dest).items():
        state = st.get("status") if complete(k.split("/")[0], st) or st.get("status") != "ok" else "finishing"
        print(f"  {k:12s} {state:9s} host={st.get('host')} gpu={st.get('gpu')} "
              f"fit_wall_s={st.get('fit_wall_s')} n_params={st.get('n_params')}")
    return dest


@app.local_entrypoint()
def main(mode: str = "smoke", reupload: bool = False, only: str = ""):
    from accuracy_cost.modal_collect import MACE_SIZES, local_runs
    if mode == "collect":
        _collect(REAL)
        rec = local_runs(REAL) / "modal_fits.json"
        if rec.exists():
            for name, cid in json.loads(rec.read_text())["calls"].items():
                try:
                    modal.FunctionCall.from_id(cid).get(timeout=0); state = "done"
                except TimeoutError:
                    state = "running"
                except Exception as e:
                    state = f"failed: {type(e).__name__}: {str(e)[:200]}"
                print(f"  call {name:12s} {cid} {state}")
        return
    _upload(reupload)
    if mode == "smoke":
        calls = [fit_pace.spawn(json.dumps(SMOKE_PACE), SMOKE), fit_mace.spawn("c8L0", json.dumps(SMOKE_MACE), SMOKE)]
        for c in calls:
            try:
                c.get()
            except Exception as e:
                print("smoke call failed:", e)
        _collect(SMOKE)
    elif mode == "probe":
        try:
            fit_pace.remote(json.dumps(PROBE_PACE), PROBE)
        except Exception as e:
            print("probe failed:", e)
        _collect(PROBE)
    elif mode == "launch":                                # on the deployed app: the calls outlive this client
        import datetime
        fp = modal.Function.from_name(APP, "fit_pace"); fm = modal.Function.from_name(APP, "fit_mace")
        names = ["pace", *(f"mace/{s}" for s in MACE_SIZES)]
        names = [n for n in names if not only or n in only.split(",")]   # --only pace: relaunch one (it resumes)
        calls = {n: (fp.spawn("", REAL) if n == "pace" else fm.spawn(n.split("/")[1], "", REAL)) for n in names}
        rec = local_runs(REAL) / "modal_fits.json"; rec.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(rec.read_text()) if rec.exists() else {}
        now = datetime.datetime.now().isoformat(timespec="seconds")
        rec.write_text(json.dumps({"app": APP, "app_id": modal.App.lookup(APP).app_id, "volume": APP, "gpu": GPU,
                                   "launched": old.get("launched", now),
                                   "calls": {**old.get("calls", {}), **{k: c.object_id for k, c in calls.items()}},
                                   "history": old.get("history", []) + [{"at": now, "calls": {k: c.object_id for k, c in calls.items()}}]},
                                  indent=1))
        print(rec.read_text())
    else:
        raise SystemExit(f"--mode smoke|probe|launch|collect, not {mode!r}")
