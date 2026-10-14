"""GAP-18's per-config-type sigmas, the one source of fit weights, in each
code's convention.  ace-jax (ACEfit dict) multiplies residuals.  Its default is
ACEpotentials.jl's default_weights() (E 30, F 1, V 1; model.jl:13), and each
config type scales it by sigma_default/sigma_type.  No N factor is needed: ACEfit
and ace-jax's Structural weighting apply 1/sqrt(N) to total-energy and virial
rows, and gap_fit's sigma_per_atom=T is the same family, so GAP-18's type ratios
transfer.  The marginal likelihood still learns each quantity's noise.
pacemaker multiplies squared residuals: w = 1/sigma^2.  MACE scales a config's
whole loss by config_type_weights, so the force sigma sets the ratio (its own
energy/forces/virials weights set the balance).  Virial sigmas reach ace-jax
and gap_fit; MACE uses one global virials weight; pacemaker fits no virials."""
import json
import pathlib
import re

import numpy as np


def command_line(xml_path):
    t = pathlib.Path(xml_path).read_text()
    m = re.search(r"<command_line><!\[CDATA\[(.*?)\]\]></command_line>", t, re.S)
    if not m:
        raise ValueError(f"{xml_path}: no <command_line>")
    return m.group(1)


def gap_sigmas(cl):
    d = tuple(map(float, re.search(r"default_sigma=\{([^}]*)\}", cl).group(1).split()[:3]))
    per, m = {}, re.search(r"config_type_sigma=\{([^}]*)\}", cl)
    if m:
        tok = m.group(1).split(":")
        if len(tok) % 5:
            raise ValueError("config_type_sigma: expected type:E:F:V:H groups")
        for i in range(0, len(tok), 5):
            per[tok[i]] = (float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3]))
    return d, per


ACEPOTENTIALS_DEFAULT = {"E": 30.0, "F": 1.0, "V": 1.0}      # ACEpotentials/src/model.jl:13


def acejax(d, per):
    D = ACEPOTENTIALS_DEFAULT
    w = lambda s: {"E": D["E"] * d[0] / s[0], "F": D["F"] * d[1] / s[1], "V": D["V"] * d[2] / s[2]}
    return {"default": dict(D), **{t: w(s) for t, s in per.items()}}


def pacemaker(d, per, atoms):
    out = []
    for at in atoms:
        sE, sF, _ = per.get(at.info.get("config_type"), d)
        out.append((1 / sE ** 2, np.full(len(at), 1 / sF ** 2)))
    return out


def mace(d, per):
    return {"Default": 1.0, **{t: (d[1] / s[1]) ** 2 for t, s in per.items()}}


def write_all(out_dir):
    from .common import DATA
    from .split import XML
    d, per = gap_sigmas(command_line(DATA / XML))
    out = pathlib.Path(out_dir)
    (out / "weights_acejax.json").write_text(json.dumps(acejax(d, per), indent=1))
    (out / "weights_mace.json").write_text(json.dumps(mace(d, per), indent=1))
