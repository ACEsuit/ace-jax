#!/usr/bin/env bash
# The multi-component speed rows (accuracy_cost/multicomponent.py: Cantor, medium random-weight models, 8,192 atoms,
# float64) on one exclusive 128-core Sulis node, 128 MPI ranks x 1 thread, with the silicon benchmark's builds:
# ace-jax's C++ kernel (linear ACE; built here, as sulis-cpu-ffi-build.sbatch) and its XLA bundles (linear ACE and
# PACE) in LAMMPS + lammps-jax's host backend, ML-PACE (LAMMPS develop ec02ed0, lmp-pacetab-cpu.sh) and MACE-MP-0b2
# medium on Symmetrix v0.3.0 (lmp-symx3-cpu.sh; needs sulis-gpu-cantor.sbatch's extraction, or makes it).
# Rows: results/sulis/sulis-cpu-128x1-cantor.jsonl and sulis-cpu-128x1-cantor-kernel.jsonl.
#   sbatch accuracy_cost/envs/sulis-cpu-cantor.sbatch          (from ~/bench-scaling/ace-jax-mc/bench)
#SBATCH --job-name=ac-cpu-cantor
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}; T=$ROOT/ace-jax-mc; PY=$ROOT/venv/bin/python
cd "$T/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$T/bench:$T/src:$ROOT/lammps-jax-ffi/python BENCH_MODELS_DIR=$ROOT/ace-jax/bench/scaling/models
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1 ACEJAX_FFI_CACHE=$T/ffi-cache-cpu
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $T/REV; cat $ROOT/VERSIONS-symx3 | head -1
M=$BENCH_MODELS_DIR; S=$M/mace_medium_Cantor-symx3.json
[ -s "$S" ] || "$ROOT/venv-symx3/bin/symmetrix_extract_mace" --model $M/mace_medium.model \
    --atomic-numbers 24 25 26 27 28 --output "$S" > $M/symx3-extract-Cantor.log 2>&1 || echo "extraction FAILED"
E=accuracy_cost/envs/sulis-cpu-cantor.json; R=accuracy_cost/results/sulis
LIB=$($PY -c "from ace_jax.eval import ffi; print(ffi.library())")
echo "kernel $LIB"
ACEJAX_EXPORT_FFI=1 LAMMPS_JAX_FFI_HANDLERS=$($PY -c "from ace_jax.eval import ffi; print(ffi.handlers('$LIB'))") \
    $PY -m accuracy_cost.multicomponent cpu --env $E --codes acejax-ace \
    --results $R/sulis-cpu-128x1-cantor-kernel.jsonl || true       # ("$@", e.g. --codes, is for the pass below / above)
$PY -m accuracy_cost.multicomponent cpu --env $E --results $R/sulis-cpu-128x1-cantor.jsonl "$@" || true
echo "done $(date -Is)"
