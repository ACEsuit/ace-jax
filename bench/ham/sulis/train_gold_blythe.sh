#!/bin/bash
#SBATCH --job-name=ham-gold
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=47:00:00
#SBATCH --output=logs/%x-%j.out
# train_gold.sh for Blythe's L40s (48 GB): one training run of MACE-H or DeepH-E3 on the bulk
# gold data, from the group share (home has a 2 GB quota).
#   sbatch [--time=..] train_gold_blythe.sh <MACE-H|DeepH-E3> <n_train> [epochs]
set -euo pipefail
code=$1; n=$2
root=/springbrook/share/engjk/essswb/ham-bench
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
