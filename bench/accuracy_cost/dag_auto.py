"""The ace-jax rows of the cost figure in the product-basis form aa_dag="auto" picks (`eval.aa_dag.resolve_aa_dag`):
from the paired timings (envs/sulis-{cpu,gpu}-dagpair.sbatch: each model in the chain form and as a DAG, one
after the other on the same node), each model's gate and timing rows from the DAG file when its chain-form
multiplies per site reach AUTO_MIN_CHAIN_MUL, else from the chain file.  The multiplies come from
results/dag_chain_multiplies.json (`chain_multiplies` of each model's aa_specs; ace_lr/o4d16, fitted elsewhere,
shares ace/o4d16's basis).

    python -m accuracy_cost.dag_auto        (from bench/; writes results/sulis/sulis-{cpu-128x1,gpu}-acejax-auto.jsonl)
"""
import glob
import json
import pathlib

from ace_jax.eval.aa_dag import AUTO_MIN_CHAIN_MUL

R = pathlib.Path(__file__).parent / "results"


def rows(dev, mode):
    out = {}
    for f in sorted(glob.glob(str(R / "sulis" / f"sulis-{dev}-dagpair-{mode}-*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out.setdefault((r["family"], r["size"]), []).append(r)
    return out


def main():
    mul = json.loads((R / "dag_chain_multiplies.json").read_text())
    for dev in ("cpu-128x1", "gpu"):
        chain, dag = rows(dev, "chain"), rows(dev, "dag")
        with open(R / "sulis" / f"sulis-{dev}-acejax-auto.jsonl", "w") as f:
            for k in sorted(set(chain) | set(dag)):
                use_dag = mul[f"{k[0]}/{k[1]}"] >= AUTO_MIN_CHAIN_MUL
                for r in (dag if use_dag else chain).get(k, []):
                    f.write(json.dumps(r) + "\n")
                print(dev, *k, "dag" if use_dag else "chain")


if __name__ == "__main__":
    main()
