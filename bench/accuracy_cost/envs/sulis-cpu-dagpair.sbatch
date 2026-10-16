#!/usr/bin/env bash
# ace-jax's product basis in the chain form and as a DAG (aa_dag="greedy", eval/aa_dag.py), timed one after the
# other for each case on the same node (Sulis CPU nodes differ by ~20% for one build): LAMMPS + lammps-jax host
# backend, 128 MPI ranks x 1 thread, ace-jax from $ROOT/ace-jax-main/src.  Analytic linear ACE radials splined at
# 1e-10 as the plotted ace rows (o3d8 unsplined, as plotted).  Rows: results/sulis-cpu-128x1-dagpair-{chain,dag}-$TAG.jsonl (one file per job:
# timing.py rewrites its results file, so concurrent jobs must not share one).  Args: cases (family/size).
#   sbatch accuracy_cost/envs/sulis-cpu-dagpair.sbatch ace/o3d8 ace/o4d20 ...   (from ~/bench-scaling/ace-jax-ac/bench)
#SBATCH --job-name=ac-cpu-dagpair
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
cd "$ROOT/ace-jax-ac/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-main/src:$ROOT/lammps-jax-cpu/python
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/ace-jax-main/REV
R=accuracy_cost/results; ENV=accuracy_cost/envs/sulis-cpu-lj-main.json
for c in "$@"; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=      # o3d8 splined misses the 1e-9 eV/A gate (1.75e-9): its plotted rows are unsplined
  export ACEJAX_AC_ACE_SPLINE_TOL=$st
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes acejax-ace --redo \
      --env $ENV --results $R/sulis-cpu-128x1-dagpair-chain-${TAG:-$SLURM_JOB_ID}.jsonl || true
  ACEJAX_EXPORT_AA_DAG=greedy "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes acejax-ace --redo \
      --env $ENV --results $R/sulis-cpu-128x1-dagpair-dag-${TAG:-$SLURM_JOB_ID}.jsonl || true
done
echo "done $(date -Is)"
