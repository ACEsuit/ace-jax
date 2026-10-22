#!/usr/bin/env bash
# The Si linear ACE ladder (o3d8-o4d20) with NVIDIA cuequivariance as the product basis (export_lammps(aa_dag="cueq"))
# in lammps-jax bundles on one A100: 32,768 atoms, float64, each model exported once (timing.LAMMPS_CUEQ; o4d20
# ~13 min), gated on the three 64-atom cells with that bundle, then timed 3x.  Radials as the plotted ace rows:
# analytic splined at 1e-10, o3d8 unsplined.  LAMMPS runs each bundle in its sidecar environment
# (run_lammps.bundle_env: replay file, shipped NVRTC cache, the ffi-replay shim, cueq's libraries).
# Tree $ROOT/ace-jax-cueqfeat: code/ (this branch), venv/ (uv sync --extra cuda --extra cueq, then lammps-jax
# 4a7f4fb editable from $ROOT/lammps-jax: the plugin's build), ffi-replay/ (lammps-jax contrib/ffi-replay's
# recorder + ffi_replay_shim.so built against the system libffi.so.6, -lffi).
#   sbatch accuracy_cost/envs/sulis-gpu-cueq.sbatch [ace/o3d8 ...]   (from $ROOT/ace-jax-cueqfeat/code/bench)
# Rows: accuracy_cost/results/sulis-gpu-cueq.jsonl (copied to results/sulis/); resumable (no --redo).
#SBATCH --job-name=ac-gpu-cueq
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}; T=$ROOT/ace-jax-cueqfeat
cd "$T/code/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
S=$T/venv/lib/python3.12/site-packages        # cueq's libraries for the export children (LAMMPS: bundle_env)
export LD_LIBRARY_PATH=$S/cuequivariance_ops/lib:$S/nvidia/cuda_nvrtc/lib:$S/nvidia/cublas/lib:$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=48     # a shared GPU node: other jobs' load
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $T/code/REV 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in ${@:-ace/o3d8 ace/o3d10 ace/o3d12 ace/o4d12 ace/o4d14 ace/o4d16 ace/o4d20}; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=      # as the plotted ace rows (sulis-gpu-dagpair.sbatch)
  echo "== $c $(date -Is)"
  ACEJAX_AC_ACE_SPLINE_TOL=$st "$T/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes acejax-ace \
      --modes lammps-cueq --env accuracy_cost/envs/sulis-gpu-cueq.json \
      --results accuracy_cost/results/sulis-gpu-cueq.jsonl --work $T/work || true
done
echo "done $(date -Is)"
