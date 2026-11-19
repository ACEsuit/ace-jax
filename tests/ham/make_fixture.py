"""Write fixtures/ham/sic_deeph: a subset of Quoll.jl's SiC regression data
(test-data-v0.4.0, `siliconcarbide_deeph.tar.gz`: FHI-aims 250610 converted to the
DeepH layout by Quoll), MIT.  Datasets are copied byte for byte (no conversion by
ace-jax), keeping the blocks of atom pairs closer than CUT angstrom.
    uv run --extra ham python tests/ham/make_fixture.py <unpacked siliconcarbide_deeph dir>

With `spin` as a second argument, fixtures/ham/fe_spin_deeph instead: the collinear-spin bcc Fe
calculation made for this work (FHI-aims 261002; `ham/julia/ingest.jl fhi-aims-spin` wrote the
source directory, with `up` and `dn` inside), blocks of atom pairs closer than CUT_FE.
    uv run --extra ham python tests/ham/make_fixture.py <ingested spin dir> spin"""
import json
import pathlib
import shutil
import sys

import h5py
import numpy as np

CUT = 3.2            # on-site, first and second neighbours
OUT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "ham" / "sic_deeph"


CUT_FE = 3.0         # on-site, first and second neighbours of bcc Fe


def subset(src, out, names, cut):
    out.mkdir(parents=True, exist_ok=True)
    lat = np.loadtxt(src / "lat.dat")                    # columns
    pos = np.loadtxt(src / "site_positions.dat")         # columns
    for f in ("lat.dat", "site_positions.dat", "element.dat", "orbital_types.dat", "info.json"):
        shutil.copy(src / f, out / f)
    cells = set()
    for name in names:
        with h5py.File(src / name, "r") as a, h5py.File(out / name, "w") as b:
            for key in a:
                Rx, Ry, Rz, i, j = json.loads(key)
                if np.linalg.norm(pos[:, j - 1] + lat @ np.array([Rx, Ry, Rz]) - pos[:, i - 1]) < cut:
                    b.create_dataset(key, data=a[key][()], compression="gzip", compression_opts=9, shuffle=True)
                    cells.add((Rx, Ry, Rz))
            print(out.name, name, len(b), "of", len(a), "blocks")
    np.savetxt(out / "R_list.dat", np.asarray(sorted(cells)), fmt="%d", delimiter="\t")


def spin(src):
    src, out = pathlib.Path(src), OUT.parent / "fe_spin_deeph"
    subset(src / "up", out / "up", ("hamiltonians.h5", "overlaps.h5"), CUT_FE)
    subset(src / "dn", out / "dn", ("hamiltonians.h5",), CUT_FE)
    (out / "README.md").write_text(
        "bcc Fe (2 atoms, a distorted cell), collinear spin: H of each spin channel (up/, dn/) and S\n"
        "(up/), blocks of atom pairs closer than 3.0 angstrom, eV and angstrom.  Generated for ace-jax:\n"
        "FHI-aims 261002, PBE, light species defaults, atomic ZORA, 8x8x8 k-points, Gaussian smearing\n"
        "0.1 eV, moment 4.31 mu_B per cell; converted by Quoll.jl through ham/julia/ingest.jl\n"
        "fhi-aims-spin.  Made by tests/ham/make_fixture.py; the datasets are Quoll's bytes.  Do not edit.\n")


def main(src):
    src = pathlib.Path(src)
    OUT.mkdir(parents=True, exist_ok=True)
    lat = np.loadtxt(src / "lat.dat")                    # columns
    pos = np.loadtxt(src / "site_positions.dat")         # columns
    for f in ("lat.dat", "site_positions.dat", "element.dat", "orbital_types.dat", "info.json"):
        shutil.copy(src / f, OUT / f)
    cells = set()
    for name in ("hamiltonians.h5", "overlaps.h5"):
        with h5py.File(src / name, "r") as a, h5py.File(OUT / name, "w") as b:
            for key in a:
                Rx, Ry, Rz, i, j = json.loads(key)
                if np.linalg.norm(pos[:, j - 1] + lat @ np.array([Rx, Ry, Rz]) - pos[:, i - 1]) < CUT:
                    b.create_dataset(key, data=a[key][()], compression="gzip", compression_opts=9, shuffle=True)
                    cells.add((Rx, Ry, Rz))
            print(name, len(b), "of", len(a), "blocks")
    np.savetxt(OUT / "R_list.dat", np.asarray(sorted(cells)), fmt="%d", delimiter="\t")
    (OUT / "README.md").write_text(
        "SiC (4 atoms), H and S blocks of atom pairs closer than 3.2 angstrom, eV and angstrom.\n"
        "A subset of Quoll.jl's regression data (maurergroup/Quoll.jl, MIT; test-data-v0.4.0,\n"
        "siliconcarbide_deeph: FHI-aims 250610, converted by Quoll).  Made by tests/ham/make_fixture.py;\n"
        "the datasets are Quoll's bytes.  Do not edit.\n")


if __name__ == "__main__":
    spin(sys.argv[1]) if sys.argv[2:] == ["spin"] else main(sys.argv[1])
