#!/usr/bin/env bash
# wL ladder for the Si accuracy-cost benchmark: linear ACE fits at the fit_ace.py command plus --wL.
# Design and the basis-size table: PLAN.md.  Fits run smallest first; a point whose metrics.csv exists is
# skipped; a failed fit is logged and the ladder moves on.  Run on a GPU box (the baseline ladder ran on
# lestrade's RTX 4000 Ada), one ladder at a time:
#     bash bench/accuracy_cost/wl_ladder/run.sh                 # the 23 core points
#     OPTIONAL=1 bash bench/accuracy_cost/wl_ladder/run.sh      # + the two optional ~3.6k points
#     ONLY="o4d15_wL2 o5d14_wL2" bash .../run.sh                # a subset, by tag
set -euo pipefail

# DATA, OUT and AJ (the aj command) can be set from outside, e.g. on Sulis (sulis.sbatch): no coupling library
# there, so ACEJAX_COUPLING_CACHE_ONLY=1 with the shapes copied into ~/.cache/ace-jax/coupling
ROOT=/home/eng/essswb/projects/ace-jax
DATA=${DATA:-/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost}
OUT=${OUT:-/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace_wl}
read -r -a AJ <<<"${AJ:-aj}"
[[ -d "$ROOT/.venv/bin" ]] && export PATH="$ROOT/.venv/bin:$PATH"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/storage/eng/essswb/cache/uv}
# new (order, max_degree, wL) shapes need the coupling library once, unless the cache already has them
[[ "${ACEJAX_COUPLING_CACHE_ONLY:-}" == 1 ]] || unset ACEJAX_COUPLING_CACHE_ONLY
OPTIONAL=${OPTIONAL:-0}
ONLY=${ONLY:-}

# wL order max_degree configs_per_batch(0 = aj default) n_B tier ; sorted by n_B (PLAN.md table)
POINTS=(
  "1 3 9 0 111 T1"
  "2 3 11 0 120 T1"
  "2 4 13 0 342 T2"
  "1.75 4 13 0 373 T2"
  "2.5 4 14 0 373 T2"
  "1 4 11 0 398 T2"
  "1.25 4 12 0 399 T2"
  "1.25 4 13 0 598 T3"
  "2 5 14 0 626 O5"
  "1 4 12 0 637 T3"
  "1.5 5 13 0 638 O5"
  "2 4 15 0 649 T3"
  "2.5 4 16 0 662 T3"
  "2.5 5 15 0 672 O5"
  "1.75 4 15 0 735 T3"
  "1.25 4 15 2 1317 T4"
  "1.75 4 17 2 1405 T4"
  "2.5 4 19 2 1492 T4"
  "1 4 14 2 1546 T4"
  "2 4 18 2 1624 T4"
  "2.5 5 18 2 1693 O5"
  "2 5 17 2 1764 O5"
  "2 4 21 1 3817 T5"
  "2 4 22 1 5031 T6"            # added 2026-10-11: wL 2.0 past o4d21 (o4d20's n_B 5456; o4d24 beyond it)
  "2 4 24 1 8559 T6"
  "1.75 4 20 1 3557 opt"
  "1 4 16 1 3603 opt"
)

mkdir -p "$OUT"
for p in "${POINTS[@]}"; do
  read -r wL order md cpb nB tier <<<"$p"
  tag="o${order}d${md}_wL${wL}"
  [[ "$tier" == opt && "$OPTIONAL" != 1 && -z "$ONLY" ]] && continue
  [[ -n "$ONLY" && " $ONLY " != *" $tag "* ]] && continue
  d="$OUT/$tag"
  if [[ -f "$d/metrics.csv" ]]; then echo "skip $tag (done)"; continue; fi
  mkdir -p "$d"
  cmd=("${AJ[@]}" fit --train "$DATA/train.xyz" --test "$DATA/test.xyz" --order "$order" --max-degree "$md" --wL "$wL"
       --rcut 5.0 --virial-key virial --weights "$DATA/weights_acejax.json" --e0 lsq --rungs map
       --m-per-species 0 --out "$d" --noise shared)
  [[ "$cpb" != 0 ]] && cmd+=(--configs-per-batch "$cpb")
  echo "${cmd[*]}" > "$d/cmd.txt"
  echo "$(date '+%F %T') fit $tag (tier $tier, n_B $nB) on $(hostname -s)"
  t0=$(date +%s)
  if (cd "$d" && "${cmd[@]}") > "$d/fit.log" 2>&1 && [[ -f "$d/model.npz" ]]; then st=ok; else st=failed; fi
  wall=$(( $(date +%s) - t0 ))
  printf '{"host": "%s", "family": "ace_wl", "size": "%s", "status": "%s", "model": "%s", "fit_wall_s": %d, "wL": %s, "order": %d, "max_degree": %d, "n_B": %d, "tier": "%s"}\n' \
    "$(hostname -f)" "$tag" "$st" "$d/model.npz" "$wall" "$wL" "$order" "$md" "$nB" "$tier" > "$d/status.json"
  echo "$(date '+%F %T') $tag $st in ${wall}s"
  [[ "$st" == failed ]] && tail -5 "$d/fit.log" || true
done
