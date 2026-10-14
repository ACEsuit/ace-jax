#!/usr/bin/env bash
# Build the CPU-only LAMMPS tree (sulis.sh lammps_cpu: ML-PACE, ML-QUIP, Symmetrix on Kokkos OpenMP) on a compute
# node.  ML-QUIP's DOWNLOAD_QUIP clones QUIP from GitHub, so the node needs outbound internet (checked first).
#   sbatch bench/scaling/envs/sulis-build-cpu.sbatch
#SBATCH --job-name=acejax-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
curl -sS --max-time 20 -o /dev/null -w "github: %{http_code}\n" https://github.com || { echo "no internet on $(hostname)"; exit 2; }
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis.sh" lammps_cpu     # $0 is the spooled copy
