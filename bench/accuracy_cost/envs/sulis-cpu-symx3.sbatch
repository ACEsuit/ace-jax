#!/usr/bin/env bash
# MACE in LAMMPS on a Sulis CPU node (128 cores): Symmetrix v0.3.0's plain symmetrix/mace (mace-symx3: OpenMP threads
# within each rank, its README's "8 threads per rank" rule of thumb) at each layout RxT of $LAYOUTS (default 128x1
# 32x4 16x8), and the "mace" rows' Symmetrix 0d86e1e at 128x1 on the same node (its published layout).  Gated
# against MACE in PyTorch (float64).  Rows: results/sulis-cpu-<RxT>-symx3.jsonl.
#   sbatch accuracy_cost/envs/sulis-cpu-symx3.sbatch
#SBATCH --job-name=ac-cpu-symx3
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
cd "$ROOT/ace-jax-gracert/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-gracert/bench:$ROOT/ace-jax-ac/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) layouts ${LAYOUTS:-128x1 32x4 16x8}"; cat $ROOT/VERSIONS-symx3 || true
for L in ${LAYOUTS:-128x1 32x4 16x8}; do
  export ACEJAX_AC_CPU_RANKS=${L%x*} ACEJAX_AC_CPU_THREADS=${L#*x}
  codes="mace-symx3"; [ "$L" = 128x1 ] && codes="mace-symx3 mace"
  for c in ${@:-mace}; do
    "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes $codes --modes lammps \
        --env accuracy_cost/envs/sulis-cpu-symx3.json --results accuracy_cost/results/sulis-cpu-$L-symx3.jsonl || true
  done
done
echo "done $(date -Is)"
