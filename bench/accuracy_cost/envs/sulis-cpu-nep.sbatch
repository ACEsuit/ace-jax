#!/usr/bin/env bash
# NEP through LAMMPS `pair_style nep` (NEP_CPU's USER-NEP, float64; built by bench/scaling/envs/sulis-nep.sh) at
# 128 MPI ranks x 1 thread.  The gate compares it with calorine's CPUNEP (needs GCC/12.3.0's libstdc++).
# Rows: results/sulis-cpu-128x1-nep.jsonl.  Args: cases (family/size).
#   sbatch accuracy_cost/envs/sulis-cpu-nep.sbatch [nep/n10 ...]
#SBATCH --job-name=ac-cpu-nep
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-cpu/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 JAX_PLATFORMS=cpu
export ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/VERSIONS-cpu || true
for c in ${@:-nep/n10 nep/n30 nep/n50 nep/n80 nep/n120}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only $c --codes nep \
      --env accuracy_cost/envs/sulis-cpu-nep.json --results accuracy_cost/results/sulis-cpu-128x1-nep.jsonl || true
done
echo "done $(date -Is)"
