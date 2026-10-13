#!/usr/bin/env bash
# PACE ladder continued from f800 (761 functions, target nradmax 15,6,4,3,2,2 / lmax 0,4,3,2,1,0) to 3200
# functions on a larger target (nradmax 15,8,6,4,3,2 / lmax 0,5,4,3,2,1: 5857 available), steps
# max(100, 30%), maxiter 1500 per step as before.  Re-submit to resume: a restart starts from the
# newest interim_potential_ladder_step_*.yaml.
#SBATCH --job-name=ac-pace-ladder
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd $HOME/ac-pace
PY=$HOME/pacemaker-env/.venv/bin/python
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('TF GPUs', g); assert g"
last=$(ls -t seg*/interim_potential_ladder_step_*.yaml 2>/dev/null | head -1 || true)
k=$(ls -d seg* 2>/dev/null | wc -l || true); seg=seg$k; mkdir -p $seg
cp train.pckl.gzip weights.pckl.gzip $seg/
init=${last:-output_potential.yaml}; cp "$init" $seg/initial_potential.yaml
$PY - "$seg" <<'PYEOF'
import sys, yaml
y = yaml.safe_load(open("input.yaml")); y["potential"]["initial_potential"] = "initial_potential.yaml"
yaml.safe_dump(y, open(f"{sys.argv[1]}/input.yaml", "w"), sort_keys=False)
PYEOF
echo "segment $seg from $init"
cd $seg && $PY ../pacemaker_np.py input.yaml > fit.out 2>&1
echo "done $(date -Is)"
