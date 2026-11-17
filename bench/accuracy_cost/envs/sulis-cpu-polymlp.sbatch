#!/usr/bin/env bash
# Seko's polynomial MLPs through LAMMPS `pair_style polymlp` (lammps-polymlp-package, float64; built by
# bench/scaling/envs/sulis-polymlp.sh) on one exclusive CPU node.  The pair style has no MPI support, so the
# benchmark's 128 ranks x 1 thread layout cannot be run: timing.py runs it as ONE rank with the node's cores as
# OpenMP threads (rows record ranks 1 and threads).  Two layouts, each its own file:
#   LAYOUT=128x1 (default)  1 rank x 128 OpenMP threads  -> results/sulis-cpu-128x1-polymlp.jsonl
#   LAYOUT=1x1              1 rank x 1 thread             -> results/sulis-cpu-1x1-polymlp.jsonl
# The gate compares it with pypolymlp's evaluator (env `polymlp_python`).  Args: cases (family/size).
#   sbatch [--export=ALL,LAYOUT=1x1] accuracy_cost/envs/sulis-cpu-polymlp.sbatch [polymlp/m3g6-l44 ...]
#SBATCH --job-name=ac-cpu-polymlp
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
cd "$ROOT/ace-jax-ye3t/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ye3t/bench:$ROOT/ace-jax-ye3t/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
LAYOUT=${LAYOUT:-128x1}
export ACEJAX_AC_CPU_RANKS=${LAYOUT%x*} ACEJAX_AC_CPU_THREADS=${LAYOUT#*x}
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) layout $LAYOUT"; cat $ROOT/VERSIONS-cpu $ROOT/VERSIONS-polymlp || true
for c in ${@:-polymlp}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes polymlp ${REDO:+--redo} \
      --env accuracy_cost/envs/sulis-cpu-polymlp.json --results accuracy_cost/results/sulis-cpu-$LAYOUT-polymlp.jsonl || true
done
echo "done $(date -Is)"
