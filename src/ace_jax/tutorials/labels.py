"""Labels for the school-derived tutorials (tutorial support, not stable API).

The tutorials' default structures are labelled once, offline, by an MIT-licensed
MACE foundation model (docs/user/tutorials/data/school/make_labels.py) and shipped
as extxyz caches keyed on structure content. `label()` serves those, so a tutorial's
default path needs neither torch nor mace-torch; only a structure the cache does not
hold (a student moved a slider, or brought their own system) is labelled live, and
only if mace-torch is installed."""
import hashlib
import pathlib
import urllib.request

import numpy as np

MODELS = {"mpa-0": "medium-mpa-0", "mp-0b3": "medium-0b3"}      # tutorial name -> mace_mp(model=...)
EXTRA = ('pip install mace-torch --extra-index-url https://download.pytorch.org/whl/cpu '
         '(CPU torch; about 1 GB installed)')
_used = [0]


class LabelsUnavailable(RuntimeError):
    pass


def labels_used():
    return _used[0]


def reset_labels_used():
    _used[0] = 0


def structure_key(atoms):
    """Content hash: numbers, pbc, cell (1e-6 A) and wrapped fractional positions (1e-6);
    atom order is part of the identity (labels are per atom)."""
    cell = np.asarray(atoms.cell.array, float)
    if np.any(atoms.pbc) and abs(np.linalg.det(cell)) > 1e-12:
        frac = np.linalg.solve(cell.T, np.asarray(atoms.positions, float).T).T
        frac = np.where(np.asarray(atoms.pbc), np.mod(np.round(frac, 6), 1.0), frac)
        pos = np.round(np.mod(np.round(frac, 6), 1.0), 6)
    else:
        pos = np.round(np.asarray(atoms.positions, float), 6)
    h = hashlib.sha256()
    for part in (np.asarray(atoms.numbers, np.int64), np.asarray(atoms.pbc, bool),
                 np.round(cell, 6) + 0.0, pos + 0.0):          # + 0.0: no -0.0
        h.update(np.ascontiguousarray(part).tobytes())
    return h.hexdigest()


def _labelled_copy(atoms, energy, forces, stress, model):
    from ase.stress import voigt_6_to_full_3x3_stress
    a = atoms.copy(); a.calc = None
    a.info["energy"] = float(energy); a.arrays["forces"] = np.asarray(forces, float)
    if np.all(a.pbc) and stress is not None:
        s = np.asarray(stress, float)
        S = voigt_6_to_full_3x3_stress(s) if s.shape == (6,) else s.reshape(3, 3)
        a.info["virial"] = -S * a.get_volume()
    a.info["label_model"] = model
    return a


class LabelCache:
    def __init__(self, entries):
        self._d = entries                                    # (model, key) -> labelled Atoms

    @classmethod
    def from_file(cls, path_or_url):
        from ase import Atoms
        from ..fit.xyz import read_extxyz
        src = str(path_or_url)
        if src.startswith(("http://", "https://")):
            # keyed on the whole URL: e1/ and c/ ship files with the same name
            tag = hashlib.sha256(src.encode()).hexdigest()[:16]
            local = pathlib.Path.home() / ".cache" / "ace-jax" / "tutorial-labels" / f"{tag}-{src.rsplit('/', 1)[-1]}"
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                urllib.request.urlretrieve(src, local)
            src = local
        d = {}
        for f in read_extxyz(src):
            a = Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
            a.info.update(f.info); a.arrays["forces"] = np.asarray(f.arrays["forces"])
            # the key of the structure as labelled (written at write time): a re-hash of the
            # 8-decimal positions read back could flip on a rounding boundary and miss
            d[(str(f.info["label_model"]), str(f.info.get("structure_key") or structure_key(a)))] = a
        return cls(d)

    @property
    def models(self):
        return {m for m, _ in self._d}

    def get(self, atoms, model):
        hit = self._d.get((model, structure_key(atoms)))
        if hit is None:
            return None
        out = atoms.copy(); out.calc = None
        out.info["energy"] = hit.info["energy"]; out.arrays["forces"] = hit.arrays["forces"].copy()
        if "virial" in hit.info:
            out.info["virial"] = np.asarray(hit.info["virial"], float).reshape(3, 3)
        out.info["label_model"] = model
        return out


def _mace_calculator(model):
    from mace.calculators import mace_mp                    # optional: the live path only
    return mace_mp(model=MODELS[model], default_dtype="float64", device="cpu")


def label(structures, *, model="mpa-0", cache=None, live=True, calculator=None):
    """Labelled copies of `structures` (energy, forces, virial for periodic cells,
    label_model), from `cache` where it holds them, else from the live labeller."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    structures = list(structures)
    out = [None if cache is None else cache.get(a, model) for a in structures]
    missing = [i for i, o in enumerate(out) if o is None]
    if missing:
        if calculator is None:
            if not live:
                raise LabelsUnavailable(f"{len(missing)} of {len(structures)} structures are not in the "
                                        f"shipped {model} labels and live labelling is off")
            try:
                calculator = _mace_calculator(model)
            except ImportError as e:
                raise LabelsUnavailable(
                    f"{len(missing)} of {len(structures)} structures are not in the shipped {model} labels; "
                    f"labelling them needs mace-torch: {EXTRA}. (The tutorial's default settings need "
                    f"nothing extra.)") from e
        for i in missing:
            a = structures[i].copy(); a.calc = calculator
            stress = a.get_stress() if np.all(a.pbc) else None
            out[i] = _labelled_copy(structures[i], a.get_potential_energy(), a.get_forces(), stress, model)
    _used[0] += len(structures)
    return out


def write_cache(path, labelled):
    """Write labelled Atoms (from label()) as a cache file the tutorials ship."""
    from ase.io import write
    frames = []
    for a in labelled:
        b = a.copy(); b.calc = None
        b.info = {k: v for k, v in a.info.items()}
        b.info["structure_key"] = structure_key(a)
        frames.append(b)
    write(str(path), frames, format="extxyz")
