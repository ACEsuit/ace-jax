#!/usr/bin/env bash
# A100 training-throughput probe for the GRACE plan: every grace / grace_fs size (and the ablations) for 10 epochs
# (target_total_updates 1330 at batch 16 on the 2,115-frame train split) into a scratch run tree, $ROOT/ac-probe;
# fit_grace.probe_summary reads s/epoch from the gracemaker logs.  Not a fit: nothing is scored.
#   sbatch accuracy_cost/envs/sulis-grace-probe.sbatch
#SBATCH --job-name=ac-grace-probe
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-grace/bench"
export PYTHONPATH=$ROOT/ace-jax-grace/bench ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_RUNS=$ROOT/ac-probe/runs TF_CPP_MIN_LOG_LEVEL=1
export ACEJAX_AC_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "job $SLURM_JOB_ID grace probe on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv
P="$ROOT/venv-grace/bin/python"; O='{"target_total_updates": 1330}'
$P -m accuracy_cost.fit_grace grace c8l3 c16l3 c16l4 c32l4 c64l4 --overrides "$O" || true
$P -m accuracy_cost.fit_grace grace_fs fs_small fs_medium --overrides "$O" || true
$P -m accuracy_cost.fit_grace grace_abl c16l4_linro c16l4_frozenR c16l4_o3 --overrides "$O" || true
$P -m accuracy_cost.fit_grace grace --probe-summary "$ROOT/ac-probe/runs" | tee "$ROOT/ac-probe/summary.txt"
echo "done $(date -Is)"
