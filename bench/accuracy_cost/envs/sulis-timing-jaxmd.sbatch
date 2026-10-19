#!/usr/bin/env bash
# JAX-MD vs LAMMPS throughput, linear ACE ladder (ace o3d8..o4d20), one Sulis A100, float64.
#
# Rows (results/sulis-gpu-jaxmd.jsonl), all from ONE ace-jax source (~/bench-scaling/ace-jax-jaxmd, feat/jax-md:
# ace_jax.jaxmd) and spline_tol 1e-10, as the sulis-gpu-ace-splined-v2 lammps rows:
#   acejax-ace jaxmd         JAX-MD NVE, forces by autodiff (quantity.force of ace_jax.jaxmd's energy)
#   acejax-ace lammps        lammps-jax (Kokkos), re-timed here from the same source
# ML-PACE on the same models (`ace_tab`, o4 only) is results/sulis/sulis-gpu-ace_tab3.jsonl: independent
# of the ace-jax source, not re-timed.  Same protocol for every row: rattled Si supercell, N=32768 (+4096
# plateau check), 0.1 fs NVE from rest, skin 1 A checked every step, median of 3 (accuracy_cost.timing).
#
# Staging (login node, once):
#   rsync bench/ -> ~/bench-scaling/ace-jax-ac/bench/  (timing.py, scaling/run_jaxmd.py)
#   git -C ~/bench-scaling clone -b feat/jax-md <ace-jax> ace-jax-jaxmd   (or rsync its src/)
#   ~/bench-scaling/venv/bin/pip install --no-deps jax-md flax jraph e3nn-jax ml-collections
#     (--no-deps keeps the venv's jax; jax_md's top-level import needs those four, nothing pulls jax)
#
#   sbatch -p gpu-devel -t 00:50:00 envs/sulis-timing-jaxmd.sbatch ace/o3d8 --force-busy \
#       --results accuracy_cost/results/sulis-gpu-jaxmd.smoke.jsonl                       # smoke (own rows)
#   sbatch --exclusive envs/sulis-timing-jaxmd.sbatch                                    # the ladder
#   per size on gpu-devel (1 h): own results file and work dir each (timing.py rewrites its file):
#   sbatch -p gpu-devel -t 01:00:00 --exclusive --export=ALL,MODES=lammps envs/sulis-timing-jaxmd.sbatch ace/o4d20 \
#       --results accuracy_cost/results/sulis-gpu-jaxmd-o4d20-lammps.jsonl --work $ROOT/ac/runs/accuracy_cost/timing/jaxmd-o4d20-lammps
#SBATCH --job-name=ac-jaxmd
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
ONLY=${1:-ace}; shift $(( $# < 1 ? $# : 1 ))
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
SRC=$ROOT/ace-jax-jaxmd/src          # also the children's, from envs/sulis-gpu-jaxmd.json `pythonpath`
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$SRC
# RSS cap per child (scaling.sweep.run_capped): the default 48 GB killed A100 gates at 52-65 GB host RSS
# (JAX preallocates 75% of the 40 GB GPU); under the job's --mem 120G
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_ACE_SPLINE_TOL=1e-10 ACEJAX_AC_RSS_CAP_GB=${ACEJAX_AC_RSS_CAP_GB:-110}
[ -n "${ACEJAX_AC_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$ACEJAX_AC_GPU
echo "job $SLURM_JOB_ID $ONLY on $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
"$ROOT/venv/bin/python" -c "import ace_jax, ace_jax.jaxmd, jax, jax_md; print(ace_jax.__file__, jax.__version__, jax.devices())"
git -C "$SRC/.." log -1 --format='ace-jax %h %s' 2>/dev/null || true
T=("$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --codes acejax-ace
   --env accuracy_cost/envs/sulis-gpu-jaxmd.json --results accuracy_cost/results/sulis-gpu-jaxmd.jsonl
   --work "$ACEJAX_AC_ROOT/runs/accuracy_cost/timing/sulis-gpu-jaxmd")
# smallest models first within each mode (sweep order); the JAX-MD lines first, the LAMMPS re-time last
# MODES: "all" (default: jaxmd, then lammps), or any space-separated list of modes run in one invocation,
# e.g. MODES=lammps, MODES=jaxmd, MODES="jaxmd-tight lammps-tight" (each fits a 1 h gpu-devel job per size)
rc=0      # every step runs; the job fails if any did (a smoke job's afterok then holds the ladder)
for only in $ONLY; do
    if [ "${MODES:-all}" = all ]; then
        "${T[@]}" --only "$only" --modes jaxmd "$@" || { rc=$?; echo "$only jaxmd: exit $rc"; }
        "${T[@]}" --only "$only" --modes lammps "$@" || { rc=$?; echo "$only lammps: exit $rc"; }
    else
        "${T[@]}" --only "$only" --modes $MODES "$@" || { rc=$?; echo "$only $MODES: exit $rc"; }
    fi
done
exit $rc
