#!/bin/bash
#SBATCH --job-name=ham-gold
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --time=47:00:00
#SBATCH --output=logs/%x-%j.out
# One training run of MACE-H or DeepH-E3 on the MACE-H paper's bulk gold data
# (bench/ham/maceh_train.py).  Single node, one A100.
#   sbatch [--time=..] train_gold.sh <MACE-H|DeepH-E3> <n_train> [epochs]
# Training structures: the first <n_train> of gold_order.txt (the seeded order ace-jax's nested
# training sets use); validation: the last max(1, n_train / 4) of it.  Epochs: 2500 (the authors'
# gold recipe) times max(1, 16 / n_train), so that small sets get as many steps as 16 structures.
set -euo pipefail
code=$1; n=$2
root=$HOME/projects/ham-bench
epochs=${3:-$(python3 -c "import math; print(math.ceil(2500 * max(1, 16 / $n)))")}
nval=$(( n / 4 > 1 ? n / 4 : 1 ))
train=$(head -n "$n" $root/scripts/gold_order.txt | paste -sd,)
val=$(tail -n "$nval" $root/scripts/gold_order.txt | paste -sd,)
work=$root/runs/gold_${code}_n${n}
mkdir -p "$work"
batch=$([ "$code" = MACE-H ] && [ "$n" -ge 2 ] && echo 2 || echo 1)
cd $root
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
venv/bin/python scripts/maceh_train.py $root/$code $root/data/bulk_gold_data/train "$work" "$train" "$val" \
    num_epoch="$epochs" batch_size="$batch" > "$work/train.log" 2>&1
grep -E "Epoch #" "$work/train.log" | tail -2
# The predictions for the test structures are made afterwards on the desktop from the fetched
# `run` folder (bench/ham/maceh_eval.py): that step is a Python loop over the bonds, about 4.5
# minutes per 64-atom structure on a CPU core, and does not need the GPU.
