#!/usr/bin/env bash
# ace-jax's real GPU memory need: lammps-jax's XLA pool fills to its limit (0.75 of the card: every bundle,
# o3d8 to o4d20, peaked at 30.04 GiB), so time three linear-ACE sizes at 32768 atoms with the pool capped by
# LAMMPS_JAX_MEM_FRACTION.  The smallest fraction a size runs at (at its layout) is what it needs.
#
#   for f in 0.05 0.1 0.2 0.4; do sbatch -p gpu-devel --exclusive -t 00:58:00 --job-name=ac-memfrac \
#       --dependency=singleton envs/sulis-memfrac.sbatch $f; done
#
# Rows: results/sulis-gpu-memfrac-<f>.jsonl (row "mem_fraction"; a failed or OOM case is a row too).
#SBATCH --job-name=ac-memfrac
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:58:00
#SBATCH --output=%x-%j.out
set -euo pipefail
F=$1; ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-ac/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
export ACEJAX_AC_ACE_SPLINE_TOL=1e-10 LAMMPS_JAX_MEM_FRACTION=$F     # as the ace rows were timed (splined)
echo "job $SLURM_JOB_ID mem_fraction $F on $(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for s in o3d8 o4d12 o4d20; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only ace/$s --repeats 1 --redo \
      --env accuracy_cost/envs/sulis-gpu.json --results accuracy_cost/results/sulis-gpu-memfrac-$F.jsonl || true
done
python3 - "accuracy_cost/results/sulis-gpu-memfrac-$F.jsonl" "$F" <<'EOF'
import json, sys
p, f = sys.argv[1], float(sys.argv[2])
rows = [json.loads(l) for l in open(p)]
for r in rows:
    r.setdefault("mem_fraction", f)
open(p, "w").writelines(json.dumps(r) + "\n" for r in rows)
EOF
echo "done $(date -Is)"
