#!/usr/bin/env bash
# The linear ACE ladder with ace-jax's C++ FFI kernel (export_lammps(ffi=True): one XLA FFI call per step in place of
# the XLA program, radials tabulated in r; envs/sulis-cpu-lj-ffi.json: ACEJAX_EXPORT_FFI and LAMMPS_JAX_FFI_HANDLERS),
# then the same model's XLA bundle (aa_dag="auto", as plotted) on the same node, for the speed-up (Sulis CPU nodes
# differ by ~20% for one build): LAMMPS + lammps-jax host backend, 128 MPI ranks x 1 thread.  Needs
# sulis-cpu-ffi-build.sbatch's build.  Analytic radials splined at 1e-10 as the plotted ace rows (o3d8 unsplined).
# Rows: results/sulis-cpu-128x1-ffi-$TAG.jsonl and sulis-cpu-128x1-ffi-xla-$TAG.jsonl (one pair per job).
#   [TAG=x] sbatch accuracy_cost/envs/sulis-cpu-ffi.sbatch [ace/o3d8 ...]   (from ~/bench-scaling/ace-jax-ffi/bench)
#SBATCH --job-name=ac-cpu-ffi
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
cd "$ROOT/ace-jax-ffi/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ffi/bench:$ROOT/ace-jax-ffi/src:$ROOT/lammps-jax-ffi/python
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/ace-jax-ffi/REV
R=accuracy_cost/results; T=${TAG:-$SLURM_JOB_ID}
for c in ${@:-ace/o3d8 ace/o3d10 ace/o3d12 ace/o4d12 ace/o4d14 ace/o4d16 ace/o4d20}; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=
  export ACEJAX_AC_ACE_SPLINE_TOL=$st
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes acejax-ace --redo \
      --env accuracy_cost/envs/sulis-cpu-lj-ffi.json --results $R/sulis-cpu-128x1-ffi-$T.jsonl || true
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes acejax-ace --redo \
      --env accuracy_cost/envs/sulis-cpu-lj-ffi-xla.json --results $R/sulis-cpu-128x1-ffi-xla-$T.jsonl || true
done
echo "done $(date -Is)"
