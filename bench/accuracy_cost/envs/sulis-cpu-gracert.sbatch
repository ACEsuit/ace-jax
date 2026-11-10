#!/usr/bin/env bash
# grace-runtime in LAMMPS on a Sulis CPU node (128 cores): grace/runtime and pace/runtime, plain MPI + OpenMP styles
# (OMP_NUM_THREADS threads a rank, each rank bound to that many cores), at each layout RxT of $LAYOUTS
# (default 128x1 32x4 8x16: grace-runtime's docs recommend OpenMP threads within each rank); the plots take each
# model's fastest gated layout.  Gates as sulis-gpu-gracert.sbatch.  bench/scaling/envs/sulis-grace-runtime.sh cpu.
# Rows: results/sulis-cpu-<RxT>-gracert.jsonl.  Args: families or family/size (default grace grace_2l pace).
#   sbatch accuracy_cost/envs/sulis-cpu-gracert.sbatch [pace/f800 ...]
#SBATCH --job-name=ac-cpu-gracert
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-gracert/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-gracert/bench:$ROOT/ace-jax-cpu/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) layouts ${LAYOUTS:-128x1 32x4 8x16}"
for L in ${LAYOUTS:-128x1 32x4 8x16}; do
  export ACEJAX_AC_CPU_RANKS=${L%x*} ACEJAX_AC_CPU_THREADS=${L#*x}
  for c in ${@:-grace grace_2l pace}; do
    "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes grace-rt pace-rt \
        --env accuracy_cost/envs/sulis-cpu-gracert.json --results accuracy_cost/results/sulis-cpu-$L-gracert.jsonl || true
  done
done
echo "done $(date -Is)"
