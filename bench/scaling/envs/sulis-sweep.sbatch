#!/usr/bin/env bash
# One benchmark sweep on a Sulis A100 (host sulis-a100), from ~/bench-scaling/ace-jax:
#
#   sbatch bench/scaling/envs/sulis-sweep.sbatch acejax-ace-learned   # sweep.py --only CODE
#   sbatch bench/scaling/envs/sulis-sweep.sbatch all                  # every code
#   sbatch bench/scaling/envs/sulis-sweep.sbatch parity               # --parity-only, then
#       exit 1 unless every acejax / mlpace / spline gate is parity_ok, so sweeps
#       queued with --dependency=afterok:<parity job> run only on a verified build
#
# (or CODE=... via --export).  Rows: bench/scaling/results/sulis-a100.jsonl, each
# with device_name from nvidia-smi.  One GPU of a node that other jobs may share:
# add --exclusive for undisturbed timings (it charges the whole node).
#SBATCH --job-name=acejax-sweep
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
CODE=${1:-${CODE:-all}}
cd "$ROOT/ace-jax"
# no modules in the parent: JAX and torch bring their own CUDA libraries, and
# lmp.sh / lmp-jax.sh load the LAMMPS modules themselves (as on moriarty)
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
# torch cu130 JIT (NVRTC) dlopens libnvrtc-builtins.so.13 by name (as on Modal)
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax/bench
PY=$ROOT/venv/bin/python
echo "job $SLURM_JOB_ID code $CODE on $(hostname) $(date -Is)"
scontrol show job "$SLURM_JOB_ID" | grep -E "NodeList=|TRES=|OverSubscribe"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

case $CODE in
  parity)
    "$PY" bench/scaling/sweep.py sulis-a100 --parity-only
    "$PY" - <<'PYEOF'
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("bench/scaling/results/sulis-a100.jsonl").read_text().splitlines()]
last = {json.dumps(r["_key"]): r for r in rows if r.get("mode") == "parity"}
bad = [k for k, r in last.items() if r["gate"] != "mace" and r["status"] != "parity_ok"]
print("gates not parity_ok (mace excluded):", bad or "none")
raise SystemExit(1 if bad else 0)
PYEOF
    ;;
  all) "$PY" bench/scaling/sweep.py sulis-a100 ;;
  *)   "$PY" bench/scaling/sweep.py sulis-a100 --only "$CODE" ;;
esac
