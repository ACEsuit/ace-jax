"""Regenerate train.xyz and test.xyz from the ace-jax test fixture si_tiny_train.xyz
(run from the repository root: python docs/user/tutorials/data/si/make_si.py).

The labels are renamed to the names `aj` reads by default (dft_energy -> energy,
dft_force -> forces, dft_virial -> virial), so the tutorial commands need no
--energy-key/--force-key/--virial-key; the gap_* predictions of an older model are
dropped.  Split: every 4th bulk configuration is the test set (13), the rest plus the
isolated atom the training set (40)."""
import pathlib

import extxyz

from ace_jax.fit.xyz import write_raw          # writes the lattice the way extxyz reads it

HERE = pathlib.Path(__file__).parent
SRC = HERE.parents[4] / "fixtures" / "si_tiny_train.xyz"
RENAME = {"dft_energy": "energy", "dft_virial": "virial"}


def clean(f):
    """Rename/drop keys in place (replacing the dicts crashes the C writer)."""
    for k in list(f.info):
        v = f.info.pop(k)
        if not k.startswith("gap_"):
            f.info[RENAME.get(k, k)] = v
    for k in [k for k in f.arrays if k not in ("species", "pos")]:
        v = f.arrays.pop(k)
        if k == "dft_force":
            f.arrays["forces"] = v
    return f


frames = [clean(f) for f in list(extxyz.iread_dicts(str(SRC)))]
iso = [f for f in frames if f.info["config_type"] == "isolated_atom"]
bulk = [f for f in frames if f.info["config_type"] != "isolated_atom"]
write_raw(HERE / "test.xyz", bulk[::4])
write_raw(HERE / "train.xyz", iso + [f for i, f in enumerate(bulk) if i % 4])
