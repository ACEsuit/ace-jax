#!/usr/bin/env bash
# Symmetrix-XL CPU build on a compute node (sulis-xl.sh: Python CPU backend, Kokkos-OpenMP LAMMPS, wrapper,
# model-xl.json extraction + host R1 artifacts for the MACE run dirs, VERSIONS-xl).  `sources venv` first, on
# the login node.  Native arch: build where the timings run (EPYC 7742).
#   sbatch bench/scaling/envs/sulis-xl-build-cpu.sbatch [steps...]
#SBATCH --job-name=xl-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
lscpu | grep "Model name"
[ $# -gt 0 ] || set -- py_cpu lammps_cpu wrappers models_cpu versions
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-xl.sh" "$@"      # $0 is the spooled copy
