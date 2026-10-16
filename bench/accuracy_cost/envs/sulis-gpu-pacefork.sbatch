#!/usr/bin/env bash
# Linear ACE run by ML-PACE's pace/kk on an A100 with the prototype kernels (pace_ref/fast/0003, 0005, 0006, 0007 on
# ICAMS/lammps-user-pace#31; scaling/envs/sulis-pacetab.sh gpufork) in their fastest mode, PACE_KK_RHOW=3 (flat-offset
# ComputeRho, segmented ComputeWeights): the ace_tab ladder (Tabulated .yace, o4d12-o4d20), gated and timed by
# timing.py.  Rows: results/sulis-gpu-ace_tabfork.jsonl, plotted with the CPU rows as the patched-ML-PACE line; with
# STOCK=1 stock pace/kk (tabulated radials only) is timed after each size on the same GPU, into sulis-gpu-ace_tab5.jsonl.
#   sbatch [--partition=gpu-devel] accuracy_cost/envs/sulis-gpu-pacefork.sbatch [o4d20 ...]   (from ~/bench-scaling/ace-jax-ac/bench)
#SBATCH --job-name=ac-gpu-pacefork
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-ac/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 PACE_KK_RHOW=${PACE_KK_RHOW:-3}
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) PACE_KK_RHOW=$PACE_KK_RHOW"
nvidia-smi --query-gpu=name,driver_version --format=csv || true
for s in ${@:-o4d12 o4d14 o4d16 o4d20}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only ace_tab/$s --codes mlpace ${REDO:+--redo} \
      --env accuracy_cost/envs/sulis-gpu-pacefork.json --results accuracy_cost/results/sulis-gpu-ace_tabfork.jsonl || true
  [ -n "${STOCK:-}" ] && { "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only ace_tab/$s --codes mlpace ${REDO:+--redo} \
      --env accuracy_cost/envs/sulis-gpu-pacetab.json --results accuracy_cost/results/sulis-gpu-ace_tab5.jsonl || true; }
done
echo "done $(date -Is)"
