"""Inference time of MACE-H (maurergroup/MACE-H) on a DeepH-layout dataset, with untrained
weights: the forward pass and the conversion of the network output to Hamiltonian blocks,
as `DeepHE3Kernel.eval` runs them.  Run in MACE-H's own environment (torch, e3nn, PyG):
    python bench/ham/maceh_time.py <MACE-H or DeepH-E3 checkout> <dataset dir> <work dir> <cpu|cuda> [threads] [train.ini]
<dataset dir> holds one folder per structure (hamiltonians.h5, element.dat, ...).  The network
is the [network] section of MACE-H's configs/train.ini (its Bi2Te3 example: 3 blocks,
correlation 3, hidden 64x0e+64x1o+64x2e, l_max 4, cutoff 7.2 angstrom), or that of the given
train.ini (e.g. the one a `maceh_train.py` run wrote: the network that was trained)."""
import configparser
import os
import sys
import time

repo, data, work, device = sys.argv[1:5]
threads = int(sys.argv[5]) if len(sys.argv) > 5 else None
config = sys.argv[6] if len(sys.argv) > 6 else None
if threads:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
import torch

if threads:
    torch.set_num_threads(threads)
sys.path.insert(0, repo)
pkg = "maceh" if os.path.isdir(os.path.join(repo, "maceh")) else "deephe3"
DeepHE3Kernel = __import__(pkg).DeepHE3Kernel
exec(f"from {pkg}.utils import *")                           # as kernel.py: names the saved model script may need

cfg = configparser.ConfigParser(inline_comment_prefixes=(';',))
dflt = os.path.join(repo, pkg, "default_configs")       # the example file leaves some required options out
cfg.read([os.path.join(dflt, "base_default.ini"), os.path.join(dflt, "train_default.ini"),
          config or os.path.join(repo, "configs", "train.ini")])
if not cfg.has_option("basic", "inference"):
    cfg["basic"]["inference"] = "False"
os.makedirs(work, exist_ok=True)
cfg["basic"].update({"device": device, "save_dir": os.path.join(work, "run"), "additional_folder_name": "", "checkpoint_dir": ""})
cfg["data"].update({"graph_dir": "", "DFT_data_dir": "", "processed_data_dir": data,
                    "save_graph_dir": os.path.join(work, "graph"), "dataset_name": "bench", "get_overlap": "False"})
ini = os.path.join(work, "train.ini")
with open(ini, "w") as f:
    cfg.write(f)

k = DeepHE3Kernel()
k.load_config(train_config_path=ini)
c = k.train_config
torch.set_default_dtype(c.torch_dtype)
os.makedirs(c.save_dir, exist_ok=True)
dataset = k.get_graph(c)
k.config_set_target()
dataset.set_mask(c.target_blocks, convert_to_net=c.convert_net_out)
k.register_constructor(device=c.device)
src = os.path.join(c.save_dir, "src")
k.save_script()                                              # copies the package to <save_dir>/src, which save_model fills
k.save_model(src)
net = k.load_model(src, device=c.device)
net.eval()
nparam = sum(p.numel() for p in net.parameters())

from torch_geometric.data import Batch

batch = Batch.from_data_list([dataset[0]]).to(c.device)
n_atoms, n_edges = batch.x.shape[0], batch.edge_index.shape[1]


def run():
    with torch.no_grad():
        out, out_edge = net(batch)
        H = k.construct_kernel.get_H(out_edge)
    if device == "cuda":
        torch.cuda.synchronize()
    return H


run()
ts = []
for _ in range(5):
    t = time.perf_counter(); H = run(); ts.append(time.perf_counter() - t)
t = min(ts)
fs = []
for _ in range(3):                                           # the network alone, without the conversion to blocks
    t0 = time.perf_counter()
    with torch.no_grad():
        net(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    fs.append(time.perf_counter() - t0)
print(f"{'MACE-H' if pkg == 'maceh' else 'DeepH-E3'} {device} {c.torch_dtype} threads {torch.get_num_threads()}: {n_atoms} atoms, {n_edges / n_atoms:.0f} edges per atom, "
      f"{nparam} parameters, output {tuple(H.shape)}; {1e3 * t:.1f} ms per structure, {1e6 * t / n_atoms:.0f} us per atom, "
      f"{1e6 * t / n_edges:.1f} us per block; network alone {1e6 * min(fs) / n_atoms:.0f} us per atom")
