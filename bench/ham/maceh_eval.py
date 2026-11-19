"""Predict Hamiltonians with a trained MACE-H or DeepH-E3 model (`bench/ham/maceh_train.py`) for
the structures of a DeepH-layout folder: `<work dir>/pred/<structure>/hamiltonians_pred.h5`, which
`bench/ham/deeph_fit.py pred=` scores.  Run in the code's own environment:
    python bench/ham/maceh_eval.py <MACE-H or DeepH-E3 checkout> <test dir> <work dir> [device=cuda]
<work dir>: the training work dir (its `run` folder holds the model)."""
import configparser
import os
import sys

repo, data, work = sys.argv[1:4]
device = dict(a.split("=", 1) for a in sys.argv[4:]).get("device", "cuda")
sys.path.insert(0, repo)
pkg = "maceh" if os.path.isdir(os.path.join(repo, "maceh")) else "deephe3"
DeepHE3Kernel = __import__(pkg).DeepHE3Kernel

cfg = configparser.ConfigParser(inline_comment_prefixes=(';',))
dflt = os.path.join(repo, pkg, "default_configs")
cfg.read([os.path.join(dflt, "base_default.ini"), os.path.join(dflt, "eval_default.ini")])
real = os.path.join(work, "test_structures")                  # real folders of linked files (os.walk follows no links)
for name in sorted(os.listdir(data)):
    os.makedirs(os.path.join(real, name), exist_ok=True)
    for f in os.listdir(os.path.join(data, name)):
        if not os.path.lexists(os.path.join(real, name, f)):
            os.symlink(os.path.abspath(os.path.join(data, name, f)), os.path.join(real, name, f))
out = os.path.join(work, "pred")
os.makedirs(out, exist_ok=True)
settings = {"device": device, "trained_model_dir": os.path.join(work, "run"), "output_dir": out, "target": "hamiltonian",
            "inference": "False", "test_only": "False", "graph_dir": "", "DFT_data_dir": "", "processed_data_dir": real,
            "save_graph_dir": os.path.join(work, "graph_test"), "dataset_name": "test", "get_overlap": "False"}
for key, value in settings.items():
    for s in cfg.sections():
        if cfg.has_option(s, key):
            cfg[s][key] = value
os.makedirs(settings["save_graph_dir"], exist_ok=True)
ini = os.path.join(work, "eval.ini")
with open(ini, "w") as f:
    cfg.write(f)
DeepHE3Kernel().eval(ini)
