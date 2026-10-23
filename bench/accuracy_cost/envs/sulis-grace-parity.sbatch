#!/usr/bin/env bash
# The 64-atom LAMMPS parity check (fit_grace --parity) of finished GRACE fits on a Sulis A100 (grace/1l/kk,
# grace/fs/kk) and with the CPU build (grace/1l/cpu/kk, grace/fs; one rank): FAMILY SIZE [SEED ...].
#   sbatch accuracy_cost/envs/sulis-grace-parity.sbatch grace c16l3 0
#SBATCH --job-name=ac-grace-parity
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-grace/bench"
FAM=$1 SIZE=$2; shift 2
export PYTHONPATH=$ROOT/ace-jax-grace/bench ACEJAX_AC_ROOT=$ROOT/ac TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_PROC_BIND=false
P="$ROOT/venv-grace/bin/python"
$P -m accuracy_cost.fit_grace "$FAM" "$SIZE" --seed ${@:-0} --parity "$ROOT/lmp-grace.sh" --device gpu
CUDA_VISIBLE_DEVICES= $P -m accuracy_cost.fit_grace "$FAM" "$SIZE" --seed ${@:-0} --parity "$ROOT/lmp-grace-cpu.sh" --device cpu
