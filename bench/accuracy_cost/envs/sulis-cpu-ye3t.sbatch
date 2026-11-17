#!/usr/bin/env bash
# YE3T through LAMMPS `pair_style ye3t` (ye3t-lammps ML-YE3T, float64, MPI; built by bench/scaling/envs/sulis-ye3t.sh),
# with stock ML-PACE v.2025.12.4.p1 in the same binary, on one exclusive CPU node.  Two layouts, each its own file:
#   LAYOUT=128x1 (default)  128 MPI ranks x 1 thread, the benchmark's CPU protocol -> results/sulis-cpu-128x1-ye3t.jsonl
#   LAYOUT=1x1              one rank x one thread, the paper's setting              -> results/sulis-cpu-1x1-ye3t.jsonl
# Families: ace_cheb (the linear ACE ladder as ChebExpCos .yace: codes ye3t-sym, mlpace = `pace` recursive,
# mlpace-product = `pace product`; gated against ace-jax on the source fit) and the fitted ye3t_* families (code
# ye3t; gated against ye3t-methods' ASE calculator, env `ye3t_python`).  Args: cases (family/size).
#   sbatch [--export=ALL,LAYOUT=1x1,COMPARE=1] accuracy_cost/envs/sulis-cpu-ye3t.sbatch [ace_cheb/o4d12 ...]
#SBATCH --job-name=ac-cpu-ye3t
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ye3t/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ye3t/bench:$ROOT/ace-jax-ye3t/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 JAX_PLATFORMS=cpu
LAYOUT=${LAYOUT:-128x1}
export ACEJAX_AC_CPU_RANKS=${LAYOUT%x*} ACEJAX_AC_CPU_THREADS=${LAYOUT#*x}
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) layout $LAYOUT"; cat $ROOT/VERSIONS-cpu $ROOT/VERSIONS-ye3t || true
for c in ${@:-ace_cheb/o3d12 ace_cheb/o4d12 ace_cheb/o4d14 ace_cheb/o4d16 ace_cheb/o4d20}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes ye3t-sym ye3t mlpace mlpace-product ${REDO:+--redo} \
      --env accuracy_cost/envs/sulis-cpu-ye3t.json --results accuracy_cost/results/sulis-cpu-$LAYOUT-ye3t.jsonl || true
  # COMPARE=1 (with LAYOUT=1x1: the paper's single-thread setting): the same fit in the benchmark's other linear-ACE
  # codes at this layout, on this node -- ML-PACE with the additions of ICAMS/lammps-user-pace#31 and stock ML-PACE
  # on the exact Tabulated file (ace_tab; envs sulis-cpu-pacefork / -pacetab), and ace-jax's C++ kernel on the fit
  # (ace; env sulis-cpu-lj-ffi, radials splined at 1e-10 as its plotted rows).  One results file per code
  # (timing.KEY has no LAMMPS build in it)
  if [ -n "${COMPARE:-}" ] && [ "${c%/*}" = ace_cheb ]; then
    s=${c#*/}; R=accuracy_cost/results
    "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only ace_tab/$s --codes mlpace ${REDO:+--redo} \
        --env accuracy_cost/envs/sulis-cpu-pacefork.json --results $R/sulis-cpu-$LAYOUT-ace_tabfork.jsonl || true
    "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only ace_tab/$s --codes mlpace ${REDO:+--redo} \
        --env accuracy_cost/envs/sulis-cpu-pacetab.json --results $R/sulis-cpu-$LAYOUT-ace_tab.jsonl || true
    BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_ACE_SPLINE_TOL=1e-10 "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu \
        --only ace/$s --codes acejax-ace ${REDO:+--redo} \
        --env accuracy_cost/envs/sulis-cpu-lj-ffi.json --results $R/sulis-cpu-$LAYOUT-ffi.jsonl || true
  fi
done
echo "done $(date -Is)"
