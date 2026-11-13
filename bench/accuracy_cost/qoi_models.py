"""Register fitted polymlp / YE3T models with the silicon QoI test framework (bench/si_qoi/compare.py's FW):
one `models/<NAME>/model.py` per fit, loading the benchmark's own calculator (evaluate.load_calc).

    python -m accuracy_cost.qoi_models FAMILY [SIZE ...]        (from bench/; default: every ok fit of the family)

NAME is plot_cost.py's QoI model name for the family (NAMES).  Run the tests with that family's venv, e.g.

    cd ~/silicon-testing-framework/test-runner
    SI_PY=/storage/eng/essswb/venvs/ac-polymlp/bin/python ./run_pairs.sh 8 POLYMLP_m3g6-l44,... $TESTS
"""
import os
import pathlib
import sys

from . import common

FW = pathlib.Path(os.path.expanduser(os.environ.get("ACEJAX_SI_FW", "~/silicon-testing-framework")))
NAMES = {"polymlp": "POLYMLP_{s}", "ye3t_sym_fit": "YE3T_{s}", "ye3t_mix_A": "YE3TmixA_{s}", "ye3t_mix_T": "YE3TmixT_{s}"}
TESTS = ("bulk_diamond,bulk_bc8,bulk_bcc,bulk_beta-Sn,bulk_fcc,bulk_hcp,bulk_hcp_short_and_fat,bulk_hex_diamond,"
         "bulk_sh,bulk_st12,vacancy-energy,interstitial-energy,surface-energy-100-relaxed,"
         "surface-decohesion-100-unrelaxed,phonon_diamond,stacking-fault-surface")
TEMPLATE = '''# {family}/{size} of the Si accuracy-cost benchmark (bench/accuracy_cost/{fit}.py), through the benchmark's own
# calculator (evaluate.load_calc): float64, total energies (E0 per atom included)
import sys
sys.path.insert(0, {bench!r})
from accuracy_cost.evaluate import load_calc
calculator = load_calc({family!r}, {model!r})
no_checkpoint = True
name = {name!r}
'''


def name(family, size):
    return NAMES[family].format(s=size)


def register(family, size):
    st = common.read_status(common.run_dir(family, size))
    if not st or st.get("status") != "ok":
        return None
    d = FW / "models" / name(family, size); d.mkdir(parents=True, exist_ok=True)
    (d / "model.py").write_text(TEMPLATE.format(family=family, size=size, model=st["model"], name=name(family, size),
                                                bench=str(common.HERE.parent),
                                                fit="fit_polymlp" if family == "polymlp" else "fit_ye3t"))
    return name(family, size)


if __name__ == "__main__":
    fam, sizes = sys.argv[1], sys.argv[2:] or [e["size"] for e in common.load_sweep()[sys.argv[1]]]
    done = [n for n in (register(fam, s) for s in sizes) if n]
    print(",".join(done)); print(TESTS)
