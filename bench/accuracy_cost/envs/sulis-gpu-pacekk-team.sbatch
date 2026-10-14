#!/usr/bin/env bash
# Linear ACE run by ML-PACE's pace/kk on an A100 with the fused team kernel (pace_ref/fast/0003; sulis-pacetab.sh
# gputeam) against the stock kernels in the same binary (PACE_KK_RHOW=0): the ace_tab ladder (Tabulated .yace,
# o4d12-o4d20), gated and timed by timing.py.  Rows: results/sulis-gpu-ace_tab-team.jsonl (team) and
# sulis-gpu-ace_tab-teamstock.jsonl (stock kernels, same build, same node).
#   sbatch accuracy_cost/envs/sulis-gpu-pacekk-team.sbatch [o4d20 ...]     (from ~/bench-scaling/ace-jax-ac/bench)
#SBATCH --job-name=ac-gpu-pacekk-team
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-ac/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,driver_version --format=csv || true
ENV=accuracy_cost/envs/sulis-gpu-pacetab-team.json; R=accuracy_cost/results
for s in ${@:-o4d12 o4d14 o4d16 o4d20}; do
  PACE_KK_RHOW=1 "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only ace_tab/$s --codes mlpace \
      --env $ENV --results $R/sulis-gpu-ace_tab-team.jsonl || true
  PACE_KK_RHOW=0 "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only ace_tab/$s --codes mlpace \
      --env $ENV --results $R/sulis-gpu-ace_tab-teamstock.jsonl || true
done
echo "done $(date -Is)"
