#!/usr/bin/env bash
# grace-runtime in LAMMPS on a Sulis A100: grace/runtime/kk (GRACE-1L and 2L kokkos.npz, "mixed": fp32 network) and
# pace/runtime/kk (pacemaker .yaml, float64); bench/scaling/envs/sulis-grace-runtime.sh gpu.  Each model gated
# (64 atoms): GRACE against gracemaker's TPCalculator (env json `grace_python`), PACE against ace-jax on the .yace.
# Rows: results/sulis-gpu-gracert.jsonl.  Args: families or family/size (default grace grace_2l pace).
#   sbatch accuracy_cost/envs/sulis-gpu-gracert.sbatch [grace/c16l4_s0 ...]
#SBATCH --job-name=ac-gpu-gracert
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-gracert/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0
export PYTHONPATH=$ROOT/ace-jax-gracert/bench:$ROOT/ace-jax-ac/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in ${@:-grace grace_2l pace}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes grace-rt pace-rt \
      --env accuracy_cost/envs/sulis-gpu-gracert.json --results accuracy_cost/results/sulis-gpu-gracert.jsonl || true
done
echo "done $(date -Is)"
