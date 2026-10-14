#!/usr/bin/env bash
# ACEpotentials.jl's LAMMPS export of the linear ACE fits (`acepotentials-trim`, pair_style ace) at 128 MPI
# ranks x 1 thread, the CPU layout of every other row.  Builds the juliac --trim libraries first, on this node
# (juliac targets the build CPU; accuracy_cost/build_trim.py keeps fresh ones), then gates (LAMMPS vs the ace-jax
# fit) and times them.  Needs scaling/envs/sulis-acepot.sh's Julia, depot, plugin and wrapper, run from the
# checkout at $ROOT/ace-jax-trim.  Rows: results/sulis-cpu-128x1-trim.jsonl.  Args: ace sizes (default all).
#   sbatch accuracy_cost/envs/sulis-cpu-trim.sbatch [o4d20 ...]          (from ~/bench-scaling/ace-jax-trim/bench)
#SBATCH --job-name=ac-cpu-trim
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
cd "$ROOT/ace-jax-trim/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-trim/bench:$ROOT/ace-jax-trim/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
export ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
ENV=accuracy_cost/envs/sulis-cpu-trim.json
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) $(grep -m1 'model name' /proc/cpuinfo)"
JULIA_DEPOT_PATH=$ROOT/julia-pr309 $ROOT/julia-1.12.6/bin/julia --startup-file=no --project=$ROOT/ace-jax-trim/bench/scaling/julia \
    -e 'using Pkg; Pkg.precompile(); using ACEpotentials; println(pkgversion(ACEpotentials))'
"$ROOT/venv/bin/python" -m accuracy_cost.build_trim --env $ENV "$@"
for s in ${@:-o3d8 o3d10 o3d12 o4d12 o4d14 o4d16 o4d20}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only ace/$s --codes acepotentials-trim \
      --env $ENV --results accuracy_cost/results/sulis-cpu-128x1-trim.jsonl || true
done
echo "done $(date -Is)"
