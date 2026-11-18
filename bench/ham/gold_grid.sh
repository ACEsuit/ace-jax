#!/bin/bash
# The ace-jax grid on the MACE-H paper's bulk gold data: model size x training-set size, for the
# accuracy / cost / data Pareto comparison.  Fits run where JAX runs (a GPU: compiled design rows
# and QR merges there); accuracy only -- inference is timed separately on one CPU.
#   bench/ham/gold_grid.sh <gold dir> <out.jsonl> [python command]
# A point already in <out.jsonl> (same model and n_train) is skipped, so the script can be re-run.
gold=$1; out=$2; py=${3:-"uv run --no-sync python"}
models=(
  "site=0,0 env=none"
  "site=2,2 env=none"
  "site=0,0 env=5.0,4,4,2,4"
  "site=0,0 env=5.0,6,4,4,4"
  "site=2,2 env=5.0,6,4,4,4"
  "site=2,2 env=5.0,8,4,6,4"
  "site=4,4 env=5.0,8,4,6,4 nbond=20 nsite=16"
)
for n in 1 2 4 8 16 32 64 152; do
  for m in "${models[@]}"; do
    tag="n=$n $m"
    grep -qsF "\"tag\": \"$tag\"" "$out.tags" && continue
    before=$(wc -l < "$out" 2>/dev/null || echo 0)
    mkdir -p "$out.models"                                      # each model as a file (operators.io), named by its tag
    $py bench/ham/deeph_fit.py "$gold" "$n" 11 bonds=0 $m out="$out" save_model="$out.models/$(echo "$tag" | tr ' =,' '_--').npz" 2>&1 \
      | grep -E "training structures|test set|bands |Fermi level|density of states|Error" | cut -c1-300
    [ "$(wc -l < "$out" 2>/dev/null || echo 0)" -gt "$before" ] && echo "{\"tag\": \"$tag\"}" >> "$out.tags"
  done
done
