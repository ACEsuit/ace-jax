#!/usr/bin/env bash
# The multi-component speed rows (accuracy_cost/multicomponent.py: Cantor, medium random-weight models, 32,768 atoms,
# float64) on a Sulis A100, with the silicon benchmark's builds: ace-jax's XLA bundles (linear ACE and PACE) and its
# CUDA kernel (linear ACE; built here with the CUDA 12.6 module's nvcc, as sulis-gpu-cudak.sbatch), ML-PACE pace/kk
# (LAMMPS develop ec02ed0, lmp-pacetab.sh) and MACE-MP-0b2 medium on Symmetrix v0.3.0 (lmp-symx3.sh).
# Tree $ROOT/ace-jax-mc: src/ and bench/ of this branch (REV); models $ROOT/ace-jax/bench/scaling/models.
# Rows: results/sulis/sulis-gpu-cantor.jsonl and sulis-gpu-cantor-kernel.jsonl.
#   sbatch accuracy_cost/envs/sulis-gpu-cantor.sbatch          (from ~/bench-scaling/ace-jax-mc/bench)
#SBATCH --job-name=ac-gpu-cantor
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}; T=$ROOT/ace-jax-mc; PY=$ROOT/venv/bin/python
cd "$T/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true
module load GCC/12.3.0 CUDA/12.6.0; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$T/bench:$T/src BENCH_MODELS_DIR=$ROOT/ace-jax/bench/scaling/models
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=48                      # a shared GPU node
export NVCC=$(command -v nvcc) ACEJAX_FFI_CACHE=$T/ffi-cache
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $T/REV; cat $ROOT/VERSIONS-symx3 | head -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
M=$BENCH_MODELS_DIR; S=$M/mace_medium_Cantor-symx3.json                    # Symmetrix v0.3.0's own extraction
[ -s "$S" ] || "$ROOT/venv-symx3/bin/symmetrix_extract_mace" --model $M/mace_medium.model \
    --atomic-numbers 24 25 26 27 28 --output "$S" > $M/symx3-extract-Cantor.log 2>&1 || echo "extraction FAILED"
E=accuracy_cost/envs/sulis-gpu-cantor.json; R=accuracy_cost/results/sulis
$PY -m accuracy_cost.multicomponent gpu --env $E --results $R/sulis-gpu-cantor.jsonl "$@" || true
LIB=$($PY -c "from ace_jax.eval import ffi; print(ffi.library(gpu=True))")
export LAMMPS_JAX_FFI_HANDLERS=$($PY -c "from ace_jax.eval import ffi; print(ffi.handlers('$LIB', gpu=True))")
echo "kernel $LIB"
ACEJAX_EXPORT_FFI=1 $PY -m accuracy_cost.multicomponent gpu --env $E --codes acejax-ace \
    --results $R/sulis-gpu-cantor-kernel.jsonl || true       # ("$@", e.g. --codes, is for the pass below / above)
echo "done $(date -Is)"
