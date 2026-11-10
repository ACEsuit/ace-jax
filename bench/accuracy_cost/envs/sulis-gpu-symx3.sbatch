#!/usr/bin/env bash
# MACE in LAMMPS on a Sulis A100: Symmetrix v0.3.0 (mace-symx3, bench/scaling/envs/sulis-symx3.sh) and, on the same
# node, the "mace" rows' Symmetrix 0d86e1e again (same-node pairs), both float64 symmetrix/mace/kk, each gated
# against MACE in PyTorch (float64).  Rows: results/sulis-gpu-symx3.jsonl.  Args: mace or mace/<size>.
#   sbatch accuracy_cost/envs/sulis-gpu-symx3.sbatch
#SBATCH --job-name=ac-gpu-symx3
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-gracert/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0
export PYTHONPATH=$ROOT/ace-jax-gracert/bench:$ROOT/ace-jax-ac/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/VERSIONS-symx3 || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in ${@:-mace}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes mace-symx3 mace --modes lammps \
      --env accuracy_cost/envs/sulis-gpu-symx3.json --results accuracy_cost/results/sulis-gpu-symx3.jsonl || true
done
echo "done $(date -Is)"
