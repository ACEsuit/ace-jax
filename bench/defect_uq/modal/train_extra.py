"""--train-extra support for fit_bench.py: validate extra training frames and append them to train.xyz."""
import numpy as np

ENERGY, FORCE, VIRIAL = "mace_energy", "mace_force", "mace_virial"


def _has(a, key):
    return key in a.info or key in a.arrays


def prepare(train_path, extra_paths, out_path, log=print, fixed="keep"):
    """Write train.xyz + the extra frames to out_path (extxyz) and return the counts dict.

    Every extra frame must carry mace_energy and mace_force (ValueError names file, frame and key).
    A missing mace_virial is allowed and explicit: the fit loader then builds no virial row for the frame
    (w_V = 0), counted and logged.  Atoms flagged `fixed` (boundary atoms held during relaxation) are
    counted and logged; fixed="keep" leaves them in (their MACE forces are valid labels of the geometry;
    the loader has one force weight per configuration, so a per-atom zero weight is not available), and
    fixed="error" refuses any frame that has them.
    """
    from ase.io import read, write
    frames = read(train_path, ":")
    n_train = len(frames)
    n_novirial = n_fixed = n_extra = 0
    for p in extra_paths:
        for i, a in enumerate(read(p, ":")):
            for key in (ENERGY, FORCE):
                if not _has(a, key):
                    raise ValueError(f"--train-extra {p} frame {i}: missing required key {key!r}")
            if not _has(a, VIRIAL):
                n_novirial += 1
            fx = a.arrays.get("fixed")
            nf = 0 if fx is None else int(np.asarray(fx, bool).sum())
            if nf and fixed == "error":
                raise ValueError(f"--train-extra {p} frame {i}: {nf} fixed boundary atoms (fixed='error')")
            n_fixed += nf
            n_extra += 1
            frames.append(a)
    write(out_path, frames, format="extxyz")
    out = {"n_train": n_train, "n_extra": n_extra, "n_extra_no_virial": n_novirial, "n_extra_fixed_atoms": n_fixed}
    log("train-extra:", out)
    return out
