#!/usr/bin/env bash
# ace-jax's CUDA kernel in LAMMPS on an A100, Si 32768 atoms: a candidate tree ($NEW, here branch
# perf/ffi-cpu-dag) and then the staged tree of main ($OLD) on the same GPU, per model, float64 and float32
# (as run for "CUDA kernel: batches without atomic adds", docs/dev/accuracy-cost-results.md; gpu-devel's hour
# takes 3-4 models). The two kernels take different plans (ffi.kernel_inputs), so each runs from its own tree:
# src/ and bench/, kernel built here by the CUDA 12.6 module's nvcc into $T/ffi-cache. Env files: envs/
# sulis-gpu-cudak.json with the tree's paths. Rows: $P/gpu-retime-new.jsonl (-> results/sulis/
# sulis-gpu-cudak.jsonl) and $P/gpu-retime-old.jsonl (-> ...-cudak-main, the same-GPU control, not plotted).
#   sbatch accuracy_cost/envs/sulis-gpu-cudak-retime.sbatch ace/o4d20 ace/o4d16 ace/o4d14
#SBATCH --job-name=cudak-retime
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:59:00
#SBATCH --output=%x-%j.out
set -uo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}; NEW=${NEW:-$ROOT/ace-jax-gpudag}; OLD=${OLD:-$ROOT/ace-jax-gracert}
P=$ROOT/prof-ffi; PY=$ROOT/venv/bin/python; B=$P/gpu-retime-$SLURM_JOB_ID; mkdir -p $B
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true
module load GCC/12.3.0 CUDA/12.6.0; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=48 ACEJAX_AC_ACE_DTYPES=${DTYPES:-float64,float32}
export NVCC=$(command -v nvcc)
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $NEW/REV $OLD/REV 2>/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in "$@"; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=      # as the plotted ace rows
  export ACEJAX_AC_ACE_SPLINE_TOL=$st
  for v in new old; do
    T=$([ $v = new ] && echo $NEW || echo $OLD)
    cd "$T/bench"
    export PYTHONPATH=$T/bench:$T/src ACEJAX_FFI_CACHE=$T/ffi-cache
    LIB=$($PY -c "from ace_jax.eval import ffi; print(ffi.library(gpu=True))")
    export LAMMPS_JAX_FFI_HANDLERS=$($PY -c "from ace_jax.eval import ffi; print(ffi.handlers('$LIB', gpu=True))")
    sed "s#bench-scaling/ace-jax-cudak/#bench-scaling/$(basename $T)/#g" $NEW/bench/accuracy_cost/envs/sulis-gpu-cudak.json > $B/$v.json
    echo "== $c $v $(date -Is) $LIB"
    $PY -m accuracy_cost.timing gpu --only $c --codes acejax-ace --modes lammps --redo \
        --env $B/$v.json --results $P/gpu-retime-$v.jsonl 2>&1 | tail -2 || true
  done
done
echo "done $(date -Is)"
