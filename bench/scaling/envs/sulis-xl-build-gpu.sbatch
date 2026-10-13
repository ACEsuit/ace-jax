#!/usr/bin/env bash
# Symmetrix-XL CUDA build on an A100 (sulis-xl.sh: Python CUDA sm_80 backend, Kokkos-CUDA LAMMPS, wrapper,
# device R1 artifacts for the MACE run dirs -- after the CPU job's model-xl.json extraction, VERSIONS-xl).
#   sbatch [--dependency=afterok:<cpu job>] bench/scaling/envs/sulis-xl-build-gpu.sbatch [steps...]
#SBATCH --job-name=xl-build-gpu
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
[ $# -gt 0 ] || set -- py_cuda lammps_cuda wrappers models_cuda versions
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-xl.sh" "$@"
