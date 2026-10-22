#!/usr/bin/env bash
# The GPU memory harness on one Sulis A100 (40 GB), float64: `python -m gpu_memory.run <args>`.
#
# Staging (from a checkout, login node):
#   rsync -a --delete src bench pyproject.toml sulis:bench-scaling/ace-jax-gpumem/
#   sbatch -p gpu-devel -t 00:58:00 bench/gpu_memory/sulis.sbatch sweep --variant baseline \
#       --dump-dir $HOME/bench-scaling/gpumem/dumps --out gpu_memory/results/sulis-a100.jsonl
#   sbatch ... sulis.sbatch bisect --variant baseline --out gpu_memory/results/sulis-a100.jsonl
# (run from ~/bench-scaling/ace-jax-gpumem: --out is relative to its bench/).  The venv is the
# benchmark one (bench/scaling/envs/sulis.sh: JAX CUDA, matscipy-neighbours with CUDA).
#SBATCH --job-name=gpumem
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:58:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SRC=${GPUMEM_SRC:-$ROOT/ace-jax-gpumem}
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
# matscipy-neighbours was built with GCC 12.3: the system libstdc++ lacks GLIBCXX_3.4.29
export LD_PRELOAD=/sulis/easybuild/software/GCCcore/12.3.0/lib64/libstdc++.so.6
export PYTHONPATH=$SRC/bench:$SRC/src
export JAX_COMPILATION_CACHE_DIR=$ROOT/tmp/jaxcache-gpumem
MODEL=${GPUMEM_MODEL:-$ROOT/ac/runs/accuracy_cost/ace/o4d20/model.npz}
cd "$SRC/bench"
echo "job ${SLURM_JOB_ID:-} on $(hostname) $(date -Is): $*"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
"$ROOT/venv/bin/python" -c "import jax, ace_jax; from ace_jax.eval.nlist import have_matscipy_neighbours as h; print(jax.__version__, jax.devices(), ace_jax.__file__, 'matscipy-neighbours', h())"
cmd=$1; shift
"$ROOT/venv/bin/python" -m gpu_memory.run "$cmd" --model "$MODEL" "$@"
echo "done $(date -Is)"
