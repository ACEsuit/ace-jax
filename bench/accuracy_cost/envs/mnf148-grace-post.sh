#!/bin/bash
# Score and QoI-test finished GRACE fits on mnf148 (CPU; fetched to /storage first): FAMILY SIZE SEEDS... per
# argument, e.g.  mnf148-grace-post.sh "grace c8l3 0 1 2" "grace_fs fs_small 0 1 2"
# Test metrics -> results/metrics.jsonl (evaluate, one at a time: it upserts the file); QoIs -> the framework's
# test-runner (run_pairs.sh; bulk_diamond with SI_TESTS_ELASTIC_FMAX=1e-3, as tight_elastic.sh).  QOI_PAR sets the
# number of concurrent tests (4 threads each; default 7).
set -u
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$B"
export CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=2 OMP_PROC_BIND=false
P=/storage/eng/essswb/venvs/ac-grace/bin/python
models=()
for spec in "$@"; do
  set -- $spec; fam=$1 size=$2; shift 2
  for k in "$@"; do
    $P -m accuracy_cost.evaluate "$fam" "${size}_s$k" 2>&1 | grep -E "rmse_E|Error" || true
    models+=("$($P -m accuracy_cost.fit_grace "$fam" "$size" --seed "$k" --register-qoi 2>/dev/null | tail -1)")
  done
done
M=$(IFS=,; echo "${models[*]}")
echo "QoI models: $M"
cd ~/silicon-testing-framework/test-runner
export SI_PY=$P
SI_TESTS_ELASTIC_FMAX=1e-3 ./run_pairs.sh 2 "$M" bulk_diamond &
./run_pairs.sh ${QOI_PAR:-7} "$M" bulk_bc8,bulk_bcc,bulk_beta-Sn,bulk_fcc,bulk_hcp,bulk_hcp_short_and_fat,bulk_hex_diamond,bulk_sh,bulk_st12,interstitial-energy,phonon_diamond,stacking-fault-surface,surface-decohesion-100-unrelaxed,surface-energy-100-relaxed,vacancy-energy
wait
echo POST_DONE
