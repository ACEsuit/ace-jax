#!/usr/bin/env bash
# Symmetrix-XL MACE ("mace-xl", "mace-xl-f32") on a Sulis A100: one rank, `-k on g 1 -sf kk`, the device R1
# artifacts from bench/scaling/envs/sulis-xl-build-gpu.sbatch (lmp-xl.sh).  Args: [ONLY] [timing args...]
#   sbatch accuracy_cost/envs/sulis-timing-xl-gpu.sbatch mace          # rows: results/sulis-gpu-mace_xl.jsonl
#SBATCH --job-name=ac-xl-gpu
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
ONLY=${1:-mace}; shift $(( $# < 1 ? $# : 1 ))
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-ac/src ACEJAX_AC_ROOT=$ROOT/ac
[ -n "${ACEJAX_AC_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$ACEJAX_AC_GPU
export ACEJAX_AC_MAX_LOAD1=16
echo "job $SLURM_JOB_ID gpu ${ONLY} on $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true; cat $ROOT/VERSIONS-xl
"$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only "$ONLY" --codes mace-xl mace-xl-f32 \
    --env accuracy_cost/envs/sulis-gpu-xl.json --results accuracy_cost/results/sulis-gpu-mace_xl.jsonl "$@"
echo "done $(date -Is)"
