"""Regenerate the Cantor (CrMnFeCoNi) tutorial subsets from the full MACE-MH-1 sets.

    python docs/user/tutorials/data/cantor/make_subsets.py --src /path/to/acegp-data/cantor

Frames are copied verbatim (the extxyz text of each selected frame, labels untouched),
chosen by a seeded permutation, so the output is byte-for-byte reproducible from the
same source files.  The source files and their provenance are described in README.md
next to this script.  Standard library only.
"""
import argparse, hashlib, pathlib, random

# (output, source, number of frames, offset into the seeded permutation)
SUBSETS = [
    ("cantor_train.xyz", "cantor1k_b_mh1.xyz", 40, 0),      # bulk: training set
    ("cantor_test.xyz", "cantor1k_b_mh1.xyz", 30, 40),      # bulk: in-distribution test, disjoint from train
    ("cantor_vacancy.xyz", "cantor_vac_mh1.xyz", 30, 0),    # OOD: one atom removed (unrelaxed)
    ("cantor_compressed.xyz", "cantor_ood_mh1.xyz", 30, 0),  # OOD: compressed bulk
]
SEED = 0


def frames(path):
    """The extxyz text of each frame (count line + comment line + atom lines)."""
    lines = pathlib.Path(path).read_text().splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1; continue
        n = int(lines[i]); out.append("".join(lines[i:i + n + 2])); i += n + 2
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="directory holding the full MACE-MH-1 Cantor sets")
    p.add_argument("--out", default=str(pathlib.Path(__file__).parent), help="output directory (default: here)")
    a = p.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    cache = {}
    for name, source, n, offset in SUBSETS:
        if source not in cache:
            fr = frames(src / source)
            perm = list(range(len(fr))); random.Random(SEED).shuffle(perm)
            cache[source] = (fr, perm)
        fr, perm = cache[source]
        idx = perm[offset:offset + n]
        text = "".join(fr[j] for j in idx)
        (out / name).write_text(text)
        print("%-24s %3d frames from %-22s  sha256 %s" % (name, n, source, hashlib.sha256(text.encode()).hexdigest()[:16]))


if __name__ == "__main__":
    main()
