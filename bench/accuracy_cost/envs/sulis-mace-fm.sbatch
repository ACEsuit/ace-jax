#!/usr/bin/env bash
# Fine-tune one foundation MACE model (family mace_fm, fit_mace_fm.py) on a Sulis GPU.  The L40 nodes (48 GB) are
# enough and usually free; the fit's hardware is not part of the benchmark.
#   sbatch accuracy_cost/envs/sulis-mace-fm.sbatch mh1-omat-ft-lr1e-3        (from ~/bench-scaling/ace-jax-fdn/bench)
#SBATCH --job-name=ac-mace-fm
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-fdn/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export PYTHONPATH=$ROOT/ace-jax-fdn/bench ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MACE_FM=${ACEJAX_AC_MACE_FM:-$HOME/.cache/mace}
PY=${MACE_FM_PYTHON:-$HOME/mh1ft-train-venv/bin/python}       # mace-torch 0.3.16 with CUDA
echo "job $SLURM_JOB_ID mace_fm $1 on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -m accuracy_cost.fit_mace_fm fit "$1"
echo "done $(date -Is)"
