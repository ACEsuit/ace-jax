#!/usr/bin/env bash
# Batch size against device memory on one Sulis A100 (40 GB): bench/batch/memory.py, arguments passed on.
# Same tree and venv as sulis-gpu.sbatch.  One job per (kind, dtype) fits the gpu-devel hour:
#   sbatch bench/batch/sulis-gpu-memory.sbatch --kind points --dtype float32     (from $ROOT/ace-jax-batch/code)
# Results: $ROOT/ace-jax-batch/results/sulis-a100-memory.jsonl
#SBATCH --job-name=batch-mem
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:59:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}; T=$ROOT/ace-jax-batch
cd "$T/code"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true
module load GCCcore/12.3.0; set -u      # matscipy-neighbours was built with GCC 12.3 (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH=$T/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # peak_bytes_in_use is then what the program needed
export JAX_COMPILATION_CACHE_DIR=$T/jax_cache
mkdir -p "$T/results"
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is): $*"; cat REV 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
$T/venv/bin/python bench/batch/memory.py --model "$T/model/o4d12.npz" --out "$T/results/sulis-a100-memory.jsonl" "$@"
echo "done $(date -Is)"
