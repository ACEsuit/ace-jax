#!/usr/bin/env bash
# One NEP training run on a Sulis A100 (GPUMD's nep; bench/scaling/envs/sulis-nep.sh gpumd builds it):
# python -m accuracy_cost.fit_nep SIZE, the data and runs under $ACEJAX_AC_ROOT.
#   sbatch envs/sulis-nep.sbatch n30   (two-stage: the second stage --dependency=afterok on the first)
#SBATCH --job-name=ac-nep
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
CODE=${ACEJAX_AC_CODE:-$ROOT/ace-jax-ac}   # a staged checkout (sbatch --export=ALL,ACEJAX_AC_CODE=...)
cd "$CODE/bench"
export PYTHONPATH=$CODE/bench:$CODE/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_NEP=$ROOT/nep.sh
echo "job $SLURM_JOB_ID nep $1 on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv
"$ROOT/venv/bin/python" -m accuracy_cost.fit_nep "$1"
