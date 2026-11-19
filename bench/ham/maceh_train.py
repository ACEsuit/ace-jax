"""Train MACE-H (maurergroup/MACE-H) or DeepH-E3 (Xiaoxun-Gong/DeepH-E3, the code MACE-H is built
on: the same data pipeline and config files) on a DeepH-layout dataset with a fixed
train/validation split.  MACE-H starts from the settings its authors published for bulk gold (Zenodo 10.5281/zenodo.15223696,
`configs/Au/Au_soc_Shift&Scale_train_maceh.ini`: 2 blocks, hidden 128x0e+128x1o+128x2e,
correlation 2, max_ell 5, Bessel basis, learning rate 0.008, shift and scale).  Run in MACE-H's
own environment:
    python bench/ham/maceh_train.py <MACE-H or DeepH-E3 checkout> <dataset dir> <work dir> <train names> <val names>
        [key=value ...]
DeepH-E3 starts from its own defaults with gold's cutoff, learning rate, epochs and batch size
(the network options of `AU` that it has).
<train names>, <val names>: comma-separated structure folders of <dataset dir>.  key=value
overrides an option of any section of the config (`num_epoch=300`, `cutoff_radius=9.0`,
`hidden_irreps=64x0e+64x1o+64x2e`, `device=cpu`, ...).  MACE-H shuffles the structures and
splits by size; here the shuffle is replaced by an ordering that puts the named training
structures first, so every code in a comparison trains on the same structures."""
import configparser
import os
import sys

repo, data, work, train, val = sys.argv[1:6]
train, val = train.split(","), val.split(",")
over = dict(a.split("=", 1) for a in sys.argv[6:])
import numpy as np

sys.path.insert(0, repo)
pkg = "maceh" if os.path.isdir(os.path.join(repo, "maceh")) else "deephe3"
DeepHE3Kernel = __import__(pkg).DeepHE3Kernel

AU = {"cutoff_radius": "10.0", "irreps_mid": "128x0e+64x1o+32x2e+16x3o+16x4e", "num_blocks": "2", "max_ell": "5",
      "hidden_irreps": "128x0e+128x1o+128x2e", "correlation": "2", "basis_func": "Bessel", "learning_rate": "0.008",
      "num_epoch": "2500", "batch_size": "2"}
cfg = configparser.ConfigParser(inline_comment_prefixes=(';',))
dflt = os.path.join(repo, pkg, "default_configs")
cfg.read([os.path.join(dflt, "base_default.ini"), os.path.join(dflt, "train_default.ini")])
os.makedirs(work, exist_ok=True)
both = os.path.join(work, "structures")                       # links: only the named structures
os.makedirs(both, exist_ok=True)
for name in train + val:                                      # real folders (os.walk follows no links) of linked files
    os.makedirs(os.path.join(both, name), exist_ok=True)
    for f in os.listdir(os.path.join(data, name)):
        if not os.path.lexists(os.path.join(both, name, f)):
            os.symlink(os.path.abspath(os.path.join(data, name, f)), os.path.join(both, name, f))
settings = {**AU, "device": "cuda", "save_dir": os.path.join(work, "run"), "additional_folder_name": "", "checkpoint_dir": "",
            "graph_dir": "", "DFT_data_dir": "", "processed_data_dir": both, "save_graph_dir": os.path.join(work, "graph"),
            "dataset_name": "bench", "get_overlap": "False", "train_size": str(len(train)), "val_size": str(len(val)),
            "test_size": "0", "simplified_output": "True", **over}
for key, value in settings.items():
    sections = [s for s in cfg.sections() if cfg.has_option(s, key)]
    if not sections and key in AU and key not in over:
        continue                                               # a MACE-H option DeepH-E3 does not have
    if not sections:
        raise SystemExit(f"no option {key} in {pkg}'s default configs")
    for s in sections:
        cfg[s][key] = value
ini = os.path.join(work, "train.ini")
with open(ini, "w") as f:
    cfg.write(f)

k = DeepHE3Kernel()
get_graph, seen = k.get_graph, {}


def graph(*a, **kw):
    seen["dataset"] = get_graph(*a, **kw)
    return seen["dataset"]


def ordered(indices):
    """In place of the shuffle of the train/validation split: training structures first."""
    ids = seen["dataset"].data.stru_id
    missing = [n for n in train + val if n not in ids]
    if missing:
        raise SystemExit(f"structures not in the graph: {missing}; it holds {ids}")
    indices.sort(key=lambda i: (train + val).index(ids[i]))
    print("train:", [ids[i] for i in indices[:len(train)]], "validation:", [ids[i] for i in indices[len(train):]])


k.get_graph = graph
np.random.shuffle = ordered
k.train(ini)
