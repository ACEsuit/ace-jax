#!/usr/bin/env bash
# GRACE through LAMMPS (grace/1l/cpu/kk, Kokkos OpenMP 1 thread a rank; grace/fs; float64; scaling/envs/sulis-grace.sh cpu)
# at 128 MPI ranks x 1 thread, gated against gracemaker's TPCalculator (grace venv, env json `grace_python`).
# Rows: results/sulis-cpu-128x1-grace.jsonl.  Args: cases (family or family/size_s0; default every grace and grace_fs size, seed 0).
#   sbatch accuracy_cost/envs/sulis-cpu-grace.sbatch [grace/c16l4_s0 ...]
#SBATCH --job-name=ac-cpu-grace
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-grace/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-grace/bench:$ROOT/ace-jax-cpu/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
export ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/VERSIONS-cpu || true
for c in ${@:-grace grace_fs}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c \
      --env accuracy_cost/envs/sulis-cpu-grace.json --results accuracy_cost/results/sulis-cpu-128x1-grace.jsonl || true
done
echo "done $(date -Is)"
