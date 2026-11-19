#!/usr/bin/env bash
# Foundation MACE models (family mace_fm) on a Sulis CPU node (128 cores) through Symmetrix v0.3.0's pair_style
# symmetrix/mace/foundation (float64) and .../float32 (OpenMP threads within each rank; the float32-mixed style is
# GPU only), at each layout RxT of $LAYOUTS (default 128x1 32x4 16x8, as sulis-cpu-symx3.sbatch); the plots take each
# model's fastest gated layout.  Gated against the model in PyTorch (float64).  Rows: results/sulis-cpu-<RxT>-fdn.jsonl.
#   sbatch accuracy_cost/envs/sulis-cpu-fdn.sbatch [mace_fm/mh1-omat ...]      (from ~/bench-scaling/ace-jax-fdn/bench)
#SBATCH --job-name=ac-cpu-fdn
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
cd "$ROOT/ace-jax-fdn/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-fdn/bench:$ROOT/ace-jax-fdn/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) layouts ${LAYOUTS:-128x1 32x4 16x8}"; cat $ROOT/VERSIONS-symx3 || true
for c in ${@:-mace_fm}; do
  for s in $("$ROOT/venv/bin/python" -c "
import sys; from accuracy_cost import common
f, _, s = sys.argv[1].partition('/'); print(' '.join(e['size'] for e in common.load_sweep()[f] if not s or e['size'] == s))" $c); do
    PATH=$ROOT/venv-symx3/bin:$PATH "$ROOT/venv-symx3/bin/python" -m accuracy_cost.fit_mace_fm export $s || true
  done
done
for L in ${LAYOUTS:-128x1 32x4 16x8}; do
  export ACEJAX_AC_CPU_RANKS=${L%x*} ACEJAX_AC_CPU_THREADS=${L#*x}
  for c in ${@:-mace_fm}; do
    "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes mace-fdn mace-fdn-f32 --modes lammps \
        --env accuracy_cost/envs/sulis-cpu-fdn.json --results accuracy_cost/results/sulis-cpu-$L-fdn.jsonl ${REDO:+--redo} || true
  done
done
echo "done $(date -Is)"
