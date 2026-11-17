#!/usr/bin/env bash
# GRACE foundation models (family grace_fm) on a Sulis CPU node (128 cores) through grace-runtime's pair_style
# grace/runtime (plain MPI + OpenMP, as sulis-cpu-gracert.sbatch), at each layout RxT of $LAYOUTS (default 128x1 32x4
# 8x16); the plots take each model's fastest gated layout.  Gated against the model's SavedModel in gracemaker's
# TPCalculator.  Rows: results/sulis-cpu-<RxT>-grace-fm.jsonl.
#   sbatch accuracy_cost/envs/sulis-cpu-grace-fm.sbatch [grace_fm/2l-omat-large ...]   (from ~/bench-scaling/ace-jax-fdn/bench)
#SBATCH --job-name=ac-cpu-grace-fm
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-fdn/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-fdn/bench:$ROOT/ace-jax-fdn/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) layouts ${LAYOUTS:-128x1 32x4 8x16}"
for L in ${LAYOUTS:-128x1 32x4 8x16}; do
  export ACEJAX_AC_CPU_RANKS=${L%x*} ACEJAX_AC_CPU_THREADS=${L#*x}
  for c in ${@:-grace_fm}; do
    "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes grace-rt \
        --env accuracy_cost/envs/sulis-cpu-grace-fm.json --results accuracy_cost/results/sulis-cpu-$L-grace-fm.jsonl ${REDO:+--redo} || true
  done
done
echo "done $(date -Is)"
