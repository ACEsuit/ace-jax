#!/usr/bin/env bash
# MPI ranks vs threads per rank (128 cores) for ace-jax (lammps-jax host backend) and ML-PACE, on a few sizes;
# rows: results/sulis-cpu-layout-<R>x<T>.jsonl.  Args: layouts as RxT (default 128x1 64x2 32x4 16x8 8x16).
#   sbatch accuracy_cost/envs/sulis-cpu-layouts.sbatch
#SBATCH --job-name=ac-cpu-layouts
#SBATCH --account=su007-jrk
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:58:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ac/bench:${ACEJAX_SRC:-$ROOT/ace-jax-cpu/src}:$ROOT/lammps-jax-cpu/python
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_ACE_SPLINE_TOL=1e-10
echo "job $SLURM_JOB_ID layouts ${*:-default} on $(hostname) $(date -Is)"
for L in ${@:-128x1 64x2 32x4 16x8 8x16}; do
  export ACEJAX_AC_CPU_RANKS=${L%x*} ACEJAX_AC_CPU_THREADS=${L#*x}
  for c in ${LAYOUT_CASES:-ace/o4d12 ace/o4d20 pace/f800 pace/f3200}; do
    "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --repeats 1 \
        --env accuracy_cost/envs/sulis-cpu-lj.json --results accuracy_cost/results/sulis-cpu-layout-$L.jsonl || true
  done
done
echo "done $(date -Is)"
