#!/bin/bash
#SBATCH --job-name=ham-gold-size
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%j.out
# A smaller network of MACE-H or DeepH-E3 on the bulk gold data, for the cost axis of the Pareto
# comparison (train_gold.sh is the authors' size).  Single node, one A100.
#   sbatch train_gold_size.sh <MACE-H|DeepH-E3> <n_train> <half|quarter>
# half / quarter: the multiplicities of the hidden and edge irreps divided by 2 / 4, every l kept.
set -euo pipefail
code=$1; n=$2; size=$3
root=$HOME/projects/ham-bench
case $size in
  half)    mid=64x0e+32x1o+16x2e+8x3o+8x4e; hid=64x0e+64x1o+64x2e ;;
  quarter) mid=32x0e+16x1o+8x2e+4x3o+4x4e;  hid=32x0e+32x1o+32x2e ;;
  *) echo "size: half or quarter"; exit 2 ;;
esac
extra="irreps_mid=$mid"; [ "$code" = MACE-H ] && extra="$extra hidden_irreps=$hid"
epochs=$(python3 -c "import math; print(math.ceil(2500 * max(1, 16 / $n)))")
nval=$(( n / 4 > 1 ? n / 4 : 1 ))
train=$(head -n "$n" $root/scripts/gold_order.txt | paste -sd,)
val=$(tail -n "$nval" $root/scripts/gold_order.txt | paste -sd,)
work=$root/runs/gold_${code}_n${n}_${size}
mkdir -p "$work"
batch=$([ "$code" = MACE-H ] && [ "$n" -ge 2 ] && echo 2 || echo 1)
cd $root
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
venv/bin/python scripts/maceh_train.py $root/$code $root/data/bulk_gold_data/train "$work" "$train" "$val" \
    num_epoch="$epochs" batch_size="$batch" $extra > "$work/train.log" 2>&1
grep -E "Epoch #" "$work/train.log" | tail -2
