#!/usr/bin/env bash
# ace-jax with the recursive (DAG) product basis (feat/recursive-aa + the lammps-cpu export changes, synced to
# $ROOT/ace-jax-dag-cpu/src) at 128 MPI ranks x 1 thread, lammps-jax host backend; ace-jax codes only (ML-PACE's
# rows are in sulis-cpu-128x1.jsonl).  Rows: results/sulis-cpu-128x1-dag.jsonl.  Args: cases (family/size).
#   sbatch accuracy_cost/envs/sulis-cpu-dag.sbatch [ace/o4d20 ...]
#SBATCH --job-name=ac-cpu-dag
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
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-dag-cpu/src:$ROOT/lammps-jax-cpu/python
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_ACE_SPLINE_TOL=1e-10 ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
export ACEJAX_EXPORT_AA_DAG=${ACEJAX_EXPORT_AA_DAG:-greedy}
echo "job $SLURM_JOB_ID dag=$ACEJAX_EXPORT_AA_DAG on $(hostname) $(date -Is)"; cat $ROOT/VERSIONS-cpu; cat $ROOT/ace-jax-dag-cpu/REV
for c in ${@:-ace/o4d12 ace/o4d16 ace/o4d20 pace/f800 pace/f3200}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes acejax-ace acejax-pace \
      --env accuracy_cost/envs/sulis-cpu-lj-dag.json --results accuracy_cost/results/sulis-cpu-128x1-dag.jsonl || true
done
echo "done $(date -Is)"
