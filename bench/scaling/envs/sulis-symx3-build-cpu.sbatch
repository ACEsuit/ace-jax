#!/usr/bin/env bash
# Symmetrix v0.3.0 on a compute node: its Python package (extractor), the MACE fits' model-symx3.json, and the CPU
# LAMMPS (-march=native).  After `bash sulis-symx3.sh sources venv` on the login node; from bench/scaling/envs:
#   sbatch sulis-symx3-build-cpu.sbatch
#SBATCH --job-name=symx3-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3850
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
R=${BENCH_ROOT:-$HOME/bench-scaling}
for i in $(seq 120); do      # submitted while `sulis-symx3.sh sources venv` ran on the login node: wait for it
  [ -f "$R/VERSIONS-symx3" ] && "$R/venv-symx3/bin/python" -c "import mace, torch, scikit_build_core, pybind11" 2>/dev/null && break; sleep 30
done
[ -f "$R/VERSIONS-symx3" ] || { echo "sources not staged"; exit 1; }
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-symx3.sh" py models lammps_cpu
