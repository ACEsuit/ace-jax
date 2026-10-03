#!/usr/bin/env bash
# Build the Sulis benchmark environment (envs/sulis.sh) on an A100 node, then
# check it: the GPU, JAX's devices, the lammps-jax plugin and ML-PACE.
#
#   sbatch bench/scaling/envs/sulis-build.sbatch          # from ~/bench-scaling/ace-jax
#
# `sources` and `venv` (clone / download only) run on the login node first:
#   bash bench/scaling/envs/sulis.sh sources venv
#SBATCH --job-name=acejax-build
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SH=$ROOT/ace-jax/bench/scaling/envs/sulis.sh
export JOBS=${SLURM_CPUS_PER_TASK:-48}
scontrol show job "$SLURM_JOB_ID" | grep -E "NodeList=|TRES=|Gres"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# ace-jax's path first (neighbours, the dev tree, the plugin); the Symmetrix
# tree last, so a Symmetrix failure costs only the MACE rows
fail=()
for s in torch_cu126 neighbours lammps_dev plugin lammps lmp_python symmetrix_py env_json; do
  bash "$SH" "$s" || fail+=("$s")
done
echo "=== failed steps: ${fail[*]:-none}"

echo "=== checks"
"$ROOT/venv/bin/python" -c "import jax; print(jax.devices())"
"$ROOT/venv/bin/python" -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
"$ROOT/lmp-jax.sh" -h | grep -i -E "jax" | head -5
"$ROOT/lmp.sh" -h | grep -i -E "pace|symmetrix" | head -10
cat "$ROOT/VERSIONS"
# the Symmetrix tree is optional (lmp.sh then runs ML-PACE on the dev tree)
for s in "${fail[@]}"; do
  case $s in lammps|lmp_python|symmetrix_py) ;; *) exit 1 ;; esac
done
