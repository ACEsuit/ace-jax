#!/usr/bin/env bash
# CPU LAMMPS rows like for like: every code (ace-jax via lammps-jax's host backend, ML-PACE, ...) in one LAMMPS
# build (lammps-dev build-kk-cpu: Kokkos OpenMP, ML-PACE, PLUGIN), RANKS MPI ranks x THREADS cores each, the same
# processors grid (run_lammps.proc_grid).  Needs scaling/envs/sulis-lammps-cpu-build.sbatch's build and the XLA
# CPU plugin (docs/dev/lammps-jax-cpu-port.md).  Args: RANKS THREADS [ONLY] [timing args...]
#   sbatch accuracy_cost/envs/sulis-timing-cpu-lj.sbatch 128 1 ace        # rows: results/sulis-cpu-128x1.jsonl
#   ACEJAX_SRC=$ROOT/ace-jax-sb-cpu/src sbatch ... 128 1 ace_pshape --env accuracy_cost/envs/sulis-cpu-lj-sb.json
#SBATCH --job-name=ac-cpu-lj
#SBATCH --account=su007-jrk
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:58:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
R=$1; T=$2; ONLY=${3:-}; shift $(( $# < 3 ? $# : 3 ))
cd "$ROOT/ace-jax-ac/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ac/bench:${ACEJAX_SRC:-$ROOT/ace-jax-cpu/src}:$ROOT/lammps-jax-cpu/python
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_CPU_RANKS=$R ACEJAX_AC_CPU_THREADS=$T
export ACEJAX_AC_ACE_SPLINE_TOL=${ACEJAX_AC_ACE_SPLINE_TOL-1e-10}     # as the standalone CPU ace rows
echo "job $SLURM_JOB_ID cpu ${R}x${T} ${ONLY:-all} on $(hostname) $(date -Is)"; cat $ROOT/VERSIONS-cpu
"$ROOT/venv/bin/python" -m accuracy_cost.timing cpu ${ONLY:+--only "$ONLY"} \
    --env accuracy_cost/envs/sulis-cpu-lj.json --results accuracy_cost/results/sulis-cpu-${R}x${T}.jsonl "$@"
echo "done $(date -Is)"
