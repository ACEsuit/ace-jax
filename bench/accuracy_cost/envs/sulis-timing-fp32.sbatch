#!/usr/bin/env bash
# The linear ACE order-4 ladder on a Sulis A100 with its mixed-precision lines (docs/dev/fp32-results.md):
# ace-jax (lammps-jax bundles, analytic radials splined at 1e-10 as the plotted rows) in float64 and float32,
# and the same models as tabulated .yace in ML-PACE's pace/kk with the pace_ref/fast prototype kernels (PR #102) in
# their fastest mode, PACE_KK_RHOW=3: KOKKOS_PREC double (lmp-pacekk-fork.sh, sulis-pacetab.sh gpufork on main) and
# mixed (lmp-pacekk-fork-mixed.sh, sulis-pacetab.sh gpufork_mixed).  Code from ~/bench-scaling/ace-jax-ac-fp32.
#
#   sbatch --exclusive envs/sulis-timing-fp32.sbatch [size ...]     (from ace-jax-ac-fp32/bench/accuracy_cost)
#
# One job per size backfills far sooner than the whole ladder (2026-10-16: the 4-size job waited >24 h):
#   for s in o4d12 o4d14 o4d16 o4d20; do sbatch --exclusive envs/sulis-timing-fp32.sbatch $s; done
# Each size goes to results/sulis-gpu-fp32-<size>.jsonl, so concurrent jobs never append to one file, and a rerun
# of a size redoes only its rows that are not ok (a failed gate is re-gated).
#SBATCH --job-name=ac-timing-fp32
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
CODE=$ROOT/ace-jax-ac-fp32
cd "$CODE/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$CODE/bench:$CODE/src
export ACEJAX_AC_ROOT=$ROOT/ac
export ACEJAX_AC_MAX_LOAD1=16
export ACEJAX_AC_ACE_SPLINE_TOL=1e-10 ACEJAX_AC_ACE_DTYPES=float64,float32 PACE_KK_RHOW=${PACE_KK_RHOW:-3}
[ -n "${ACEJAX_AC_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$ACEJAX_AC_GPU
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) PACE_KK_RHOW=$PACE_KK_RHOW"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
"$ROOT/venv/bin/python" -c "import ace_jax; print(ace_jax.__file__)"
SIZES=${*:-o4d12 o4d14 o4d16 o4d20}
for fam in ace ace_tab; do
  for size in $SIZES; do
    OUT=accuracy_cost/results/sulis-gpu-fp32-$size.jsonl
    "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $fam/$size --env accuracy_cost/envs/sulis-gpu-fp32.json \
        --results $OUT
  done
done
