#!/usr/bin/env bash
# lammps-jax CPU parity + MD timing on one Sulis compute node (bench/scaling/lammps_cpu_smoke.py), after
# sulis-lammps-cpu-build.sbatch.  Args: SIZE (ace/<size> under $ROOT/ac/runs/accuracy_cost) REP RANKS... ; extra
# smoke options via SMOKE_ARGS.
#   sbatch bench/scaling/envs/sulis-lammps-cpu-smoke.sbatch o4d12 16 16 32 64 128
#SBATCH --job-name=lj-cpu-smoke
#SBATCH --account=su007-jrk
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:58:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SIZE=$1; REP=$2; shift 2
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/lammps-jax-cpu/python:$ROOT/ace-jax-cpu/src:$ROOT/ace-jax-cpu/bench
export JAX_PLATFORMS=cpu
OUT=$ROOT/ac/lammps-cpu/$SIZE-rep$REP-$SLURM_JOB_ID
echo "job $SLURM_JOB_ID $SIZE rep $REP ranks $* on $(hostname) $(date -Is); $(nproc) cpus"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|L3"
cat $ROOT/VERSIONS-cpu
"$ROOT/venv/bin/python" $ROOT/ace-jax-cpu/bench/scaling/lammps_cpu_smoke.py \
  $ROOT/ac/runs/accuracy_cost/ace/$SIZE/model.npz --lmp $ROOT/lmp-jax-cpu.sh \
  --plugin-dir $ROOT/lammps-jax-cpu/build-plugin-cpu-pjrt --pjrt $ROOT/pjrt-cpu-xla-91888df/pjrt_c_api_cpu_plugin.so \
  --rep $REP --ranks "$@" --out $OUT ${SMOKE_ARGS:-}
echo "done $(date -Is)"
