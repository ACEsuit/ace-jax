#!/usr/bin/env bash
# The Si linear ACE ladder (o3d8-o4d20) with ace-jax's CUDA kernel (export_lammps(ffi=True) on the GPU: AceDirectGpu,
# one FFI call per step in place of the XLA program, radials tabulated in r; envs/sulis-gpu-cudak.json:
# ACEJAX_EXPORT_FFI), then the same model's XLA bundle (aa_dag="auto", as plotted; sulis-gpu-cudak-xla.json) on the
# same A100 (a shared node, not --exclusive: the pair's ratio is like for like), float64 and float32
# (ACEJAX_AC_ACE_DTYPES).  LAMMPS + lammps-jax (the GPU plugin, 4a7f4fb), 1 rank.
# The kernel is built here with the CUDA 12.6 module's nvcc (the A100s' driver is 560: no CUDA 13) into
# $T/ffi-cache, and LAMMPS_JAX_FFI_HANDLERS names it.  Analytic radials splined at 1e-10 as the plotted ace rows
# (o3d8 unsplined).  Tree $ROOT/ace-jax-cudak: src/ and bench/ of this branch (REV), the bench venv ($ROOT/venv).
# Rows: results/sulis-gpu-cudak-$TAG.jsonl and sulis-gpu-cudak-xla-$TAG.jsonl (one pair per job).
#   [TAG=x] sbatch accuracy_cost/envs/sulis-gpu-cudak.sbatch [ace/o3d8 ...]   (from ~/bench-scaling/ace-jax-cudak/bench)
#SBATCH --job-name=ac-gpu-cudak
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}; T=$ROOT/ace-jax-cudak
cd "$T/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true
module load GCC/12.3.0 CUDA/12.6.0; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$T/bench:$T/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=48 ACEJAX_AC_ACE_DTYPES=float64,float32   # a shared GPU node
export NVCC=$(command -v nvcc) ACEJAX_FFI_CACHE=$T/ffi-cache
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $T/REV
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
LIB=$("$ROOT/venv/bin/python" -c "from ace_jax.eval import ffi; print(ffi.library(gpu=True))")
export LAMMPS_JAX_FFI_HANDLERS=$("$ROOT/venv/bin/python" -c "from ace_jax.eval import ffi; print(ffi.handlers('$LIB', gpu=True))")
echo "kernel $LIB"; echo "LAMMPS_JAX_FFI_HANDLERS=$LAMMPS_JAX_FFI_HANDLERS"
R=accuracy_cost/results; TG=${TAG:-$SLURM_JOB_ID}
for c in ${@:-ace/o3d8 ace/o3d10 ace/o3d12 ace/o4d12 ace/o4d14 ace/o4d16 ace/o4d20}; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=      # as the plotted ace rows
  export ACEJAX_AC_ACE_SPLINE_TOL=$st
  echo "== $c $(date -Is)"
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes acejax-ace --modes lammps --redo \
      --env accuracy_cost/envs/sulis-gpu-cudak.json --results $R/sulis-gpu-cudak-$TG.jsonl || true
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes acejax-ace --modes lammps --redo \
      --env accuracy_cost/envs/sulis-gpu-cudak-xla.json --results $R/sulis-gpu-cudak-xla-$TG.jsonl || true
done
echo "done $(date -Is)"
