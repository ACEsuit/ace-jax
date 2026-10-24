#!/usr/bin/env bash
# Batched evaluation on one Sulis A100 (docs/dev/batched-evaluation-results.md): bench/batch/run.py (single
# points, BatchCalculator against ACECalculator one cell per call; float32 and float64; builders jax and msn)
# and bench/batch/md.py (vmapped JAX-MD NVE, the JAX model) for the Si o4d12 linear ACE.
# Tree $ROOT/ace-jax-batch: code/ (branch feat/batched-eval), venv/ (uv sync --extra cuda --extra jaxmd, then
# matscipy-neighbours built from libAtoms/matscipy-neighbours#11 with ENABLE_CUDA, sm_80 and JAX's FFI
# headers), model/o4d12.npz (runs/accuracy_cost/ace/o4d12/model.npz).
#   sbatch bench/batch/sulis-gpu.sbatch            (from $ROOT/ace-jax-batch/code)
# Results: $ROOT/ace-jax-batch/results/sulis-a100-{points,md}.jsonl
#SBATCH --job-name=batch-gpu
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}; T=$ROOT/ace-jax-batch
cd "$T/code"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$T/jax_cache
mkdir -p "$T/results"
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat REV 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu --format=csv || true
PY=$T/venv/bin/python
$PY -c "import jax, matscipy_neighbours.xla as x; print(jax.__version__, jax.devices(), hasattr(x, 'neighbour_matrix_batched'))"
$PY bench/batch/run.py --model "$T/model/o4d12.npz" --dtype float32 float64 --builder jax msn \
    --out "$T/results/sulis-a100-points.jsonl"
$PY bench/batch/md.py --model "$T/model/o4d12.npz" --sizes 8 64 216 --batch 1 16 64 256 1024 \
    --max-atoms 262144 | tee -a "$T/results/sulis-a100-md.jsonl"
echo "done $(date -Is)"
