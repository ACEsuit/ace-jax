#!/usr/bin/env bash
# Linear ACE run by ML-PACE from ICAMS/lammps-user-pace#31 (scaling/envs/sulis-pacetab.sh cpufork: tabulated radials
# plus the recursive-evaluator speed-ups): the ace_tab ladder (Tabulated .yace, o4d12-o4d20) at 128 MPI ranks x
# 1 thread, gated and timed by timing.py.  Rows: results/sulis-cpu-128x1-ace_tabfork.jsonl; with STOCK=1 stock ML-PACE
# (tabulated radials only) is timed after each size on the same node, into sulis-cpu-128x1-ace_tab5.jsonl.
#   [REDO=1] sbatch accuracy_cost/envs/sulis-cpu-pacefork.sbatch [o4d20 ...]   (from ~/bench-scaling/ace-jax-ac/bench; REDO re-times ok rows)
#SBATCH --job-name=ac-cpu-pacefork
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-cpu/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu   # 128 ML-PACE ranks: o4d20 ~79 GB
export ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) $(grep -m1 'model name' /proc/cpuinfo)"
R=accuracy_cost/results                    # results/sulis/ in the repository
for s in ${@:-o4d12 o4d14 o4d16 o4d20}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only ace_tab/$s --codes mlpace ${REDO:+--redo} \
      --env accuracy_cost/envs/sulis-cpu-pacefork.json --results $R/sulis-cpu-128x1-ace_tabfork.jsonl || true
  [ -n "${STOCK:-}" ] && { "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only ace_tab/$s --codes mlpace ${REDO:+--redo} \
      --env accuracy_cost/envs/sulis-cpu-pacetab.json --results $R/sulis-cpu-128x1-ace_tab5.jsonl || true; }
done
echo "done $(date -Is)"
