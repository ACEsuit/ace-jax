#!/usr/bin/env bash
# NEP on a Sulis A100 with GPUMD's own MD code (gpumd; bench/scaling/envs/sulis-nep.sh gpumd builds it): mixed
# precision (dtype "mixed"), gated against calorine's CPUNEP (float64; needs GCC/12.3.0's libstdc++).
# Rows: results/sulis-gpu-nep.jsonl.  Args: cases (family/size).
#   sbatch accuracy_cost/envs/sulis-gpu-nep.sbatch [nep/n10 ...]
#SBATCH --job-name=ac-gpu-nep
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
cd "$ROOT/ace-jax-ac/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-ac/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
[ -n "${ACEJAX_AC_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$ACEJAX_AC_GPU
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in ${@:-nep/n10 nep/n30 nep/n50 nep/n80 nep/n120}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes nep-gpumd \
      --env accuracy_cost/envs/sulis-gpu-nep.json --results accuracy_cost/results/sulis-gpu-nep.jsonl || true
done
echo "done $(date -Is)"
