"""Copy a libetcouple bundle keeping only the shared libraries the runtime
actually loads (traced while running every case in cases.json) plus all
non-library files.  Symlinked names that were loaded become regular files
(wheels cannot carry symlinks).
    python coupling/tools/prune_bundle.py coupling/build/bundle coupling/build/pruned"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
EXT = "dylib" if sys.platform == "darwin" else "so"
DRIVER = """
import json, sys
sys.path.insert(0, sys.argv[1])
import ace_jax_coupling as ajc
for c in json.load(open(sys.argv[2])).values():
    ajc.couple_raw(c["mb"], c["R"], c["Y"])
print("traced-ok")
"""


def _is_lib(p):
    n = p.name
    return n.endswith(".dylib") if sys.platform == "darwin" else (n.endswith(".so") or ".so." in n)


def traced(src):
    # Trace in a clean environment: with a Julia depot reachable (HOME/.julia,
    # JULIA_DEPOT_PATH) the image's JLLs resolve their artifacts from the depot
    # instead of the bundle, and those bundle files would be pruned.
    env = {k: v for k, v in os.environ.items() if not k.startswith("JULIA")}
    env["HOME"] = tempfile.mkdtemp(prefix="prune-home-")
    env["ACEJAX_COUPLING_LIB"] = str(src / "lib" / f"libetcouple.{EXT}")
    if sys.platform == "darwin":
        env["DYLD_PRINT_LIBRARIES"] = "1"
    else:
        env["LD_DEBUG"] = "files"
    r = subprocess.run([sys.executable, "-c", DRIVER, str(REPO / "coupling/python/src"),
                        str(REPO / "coupling/python/tests/data/cases.json")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "traced-ok" in r.stdout, r.stderr[-3000:]
    root = str(src) + os.sep
    used = set()
    for line in r.stderr.splitlines():
        for tok in line.replace("=", " ").split():
            if tok.startswith(root):
                used.add(pathlib.Path(tok).relative_to(src))
    return used


def main(src, dst):
    src, dst = pathlib.Path(src).resolve(), pathlib.Path(dst).resolve()
    used = {pathlib.Path(os.path.realpath(src / u)).relative_to(src) for u in traced(src)}
    assert pathlib.Path("lib") / f"libetcouple.{EXT}" in used, "trace did not see libetcouple"
    # The trace reports resolved files, but binaries refer to them by alias
    # (install names / SONAMEs / dlopen strings, e.g. @rpath/libunwind.1.dylib ->
    # libunwind.1.0.dylib).  Keep a name of a traced library iff that name occurs
    # in some other traced binary (fallback: its real name); each kept name becomes a
    # regular file.
    data = {u: (src / u).read_bytes() for u in sorted(used)}
    names = {}                                      # real rel path -> kept rel names
    for p in sorted(src.rglob("*")):
        if p.is_dir() or not _is_lib(p):
            continue
        real = pathlib.Path(os.path.realpath(p)).relative_to(src)
        # referenced from ANOTHER traced binary (a library's own bytes name itself)
        if real in used and any(p.name.encode() in d for u, d in data.items() if u != real):
            names.setdefault(real, []).append(p.relative_to(src))
    for real in used:
        names.setdefault(real, [real])
    shutil.rmtree(dst, ignore_errors=True)
    kept = 0
    for p in sorted(src.rglob("*")):                # non-library files: all of them
        if p.is_file() and not _is_lib(p):
            out = dst / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p.resolve(), out)
            kept += 1
    for real, rels in sorted(names.items()):
        for rel in rels:
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / real, out)           # dereference symlinks
            kept += 1
    n_names = sum(map(len, names.values()))
    print(f"kept {kept} files: {len(used)} traced libs under {n_names} names -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
