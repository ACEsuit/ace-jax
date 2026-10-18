#!/usr/bin/env bash
# ace-jax's product basis in the chain form and as a DAG (aa_dag="greedy", eval/aa_dag.py), timed one after the
# other for each case on the same A100: LAMMPS + lammps-jax, ace-jax from $ROOT/ace-jax-main/src.  Analytic linear
# ACE radials splined at 1e-10 as the plotted ace rows (o3d8 unsplined, as plotted).  Rows: results/sulis-gpu-dagpair-{chain,dag}-$TAG.jsonl (one file per job:
# timing.py rewrites its results file, so concurrent jobs must not share one).
#   sbatch [--partition=gpu-devel] accuracy_cost/envs/sulis-gpu-dagpair.sbatch ace/o3d8 ...   (from ~/bench-scaling/ace-jax-ac/bench)
#SBATCH --job-name=ac-gpu-dagpair
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-main/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/ace-jax-main/REV
nvidia-smi --query-gpu=name,driver_version --format=csv || true
R=accuracy_cost/results; ENV=accuracy_cost/envs/sulis-gpu-main.json
for c in "$@"; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=      # o3d8 splined misses the 1e-9 eV/A gate (1.75e-9): its plotted rows are unsplined
  export ACEJAX_AC_ACE_SPLINE_TOL=$st
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes acejax-ace --redo \
      --env $ENV --results $R/sulis-gpu-dagpair-chain-${TAG:-$SLURM_JOB_ID}.jsonl || true
  ACEJAX_EXPORT_AA_DAG=greedy "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes acejax-ace --redo \
      --env $ENV --results $R/sulis-gpu-dagpair-dag-${TAG:-$SLURM_JOB_ID}.jsonl || true
done
echo "done $(date -Is)"
