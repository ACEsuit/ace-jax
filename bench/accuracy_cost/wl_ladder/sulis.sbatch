#!/usr/bin/env bash
# The wL ladder (run.sh) on a Sulis A100, with the benchmark branch's code (~/bench-scaling/ace-jax-ac, the
# code of the baseline ace ladder) in its own fitting venv, ~/bench-scaling/fitenv (uv sync --frozen --extra cuda
# from that copy's uv.lock; the timing venv lacks numpyro/optax), with the desktop's coupling cache copied over.
#   ONLY="o4d12_wL1 ..." sbatch wl_ladder/sulis.sbatch
#SBATCH --job-name=ac-wl
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
B=$HOME/bench-scaling
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export PYTHONPATH=$B/ace-jax-ac/src
export DATA=$B/ac/data/si_accuracy_cost OUT=$B/ac/runs/accuracy_cost/ace_wl AJ="$B/fitenv/bin/python -m ace_jax.cli"
export UV_CACHE_DIR=$HOME/.cache/uv
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) ONLY=${ONLY:-all}"; nvidia-smi --query-gpu=name --format=csv,noheader
$B/fitenv/bin/python -c "import ace_jax, jax; print(ace_jax.__file__, jax.devices())"
bash $B/ace-jax-ac/bench/accuracy_cost/wl_ladder/run.sh
