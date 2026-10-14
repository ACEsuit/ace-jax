"""GAP refitted on the train split with GAP-18's own gap_fit command line (the
published recipe, glue core included); only the data, n_sparse and output
change.  Run under the quip
venv, on mnf148: 9k sparse points needs tens of GB."""
import pathlib
import re
import shutil
import sys

from . import common
from .common import DATA


def refit_command(cl, n_sparse, train, gp_file):
    c = re.sub(r"\bat_file=\S+", f"at_file={train}", cl.strip())
    c = re.sub(r"\bn_sparse=\d+", f"n_sparse={n_sparse}", c)
    c = re.sub(r"\bgp_file=\S+", f"gp_file={gp_file}", c)
    c = re.sub(r"\benergy_parameter_name=\S+", "energy_parameter_name=energy", c)
    c = re.sub(r"\bforce_parameter_name=\S+", "force_parameter_name=forces", c)
    return re.sub(r"\bvirial_parameter_name=\S+", "virial_parameter_name=virial", c)


def xml_label(path):
    m = re.search(r'<Potential\s+label="([^"]+)"', open(path).read())
    if not m:
        raise ValueError(f"{path}: no top-level <Potential label=...>")
    return m.group(1)


def gap_fit_exe():
    """gap_fit from this interpreter's venv: running `venv/bin/python` does not put venv/bin on PATH."""
    p = pathlib.Path(sys.executable).parent / "gap_fit"
    return str(p) if p.exists() else "gap_fit"


def fit(size):
    from .split import GLUE, XML, ensure_split
    from .weights import command_line
    h = ensure_split()
    e, d = common.entry("gap", size), common.run_dir("gap", size)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / GLUE, d / GLUE)                            # core_param_file is read from cwd
    args = refit_command(command_line(DATA / XML), e["n_sparse"], str(DATA / "train.xyz"), "gap.xml")
    st = common.fit_once("gap", size, "gap.xml", [gap_fit_exe(), *args.split()], split_hash=h)
    if st["status"] == "ok" and not st.get("quip_label"):
        st.update(n_params=e["n_sparse"], quip_label=xml_label(d / "gap.xml"))
        common.write_status(d, **{k: v for k, v in st.items() if k != "host"})
    return st


if __name__ == "__main__":
    which = sys.argv[1]
    for s in ([e["size"] for e in common.load_sweep()["gap"]] if which == "all" else [which]):
        st = fit(s)
        print(s, st["status"], flush=True)
        if st["status"] != "ok":
            print(st.get("error", ""), flush=True)
