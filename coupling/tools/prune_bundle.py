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

# GPL-2.0+ SuiteSparse modules.  The shim never calls them (EquivariantTensors'
# nullspace_solver = :dense), but SuiteSparse_jll's __init__ dlopens every
# SuiteSparse library, so each is replaced by an empty placeholder of the same
# name: the wheel ships no GPL code.  check_bundle.py enforces the marker.
GPL_LIBS = ("libumfpack", "libspqr", "librbio", "libcholmod")
# also replaced where present: KLU's CHOLMOD interface (LGPL) imports CHOLMOD symbols,
# which Windows resolves when the DLL loads, so it cannot load against a placeholder
PLACEHOLDER_LIBS = GPL_LIBS + ("libklu_cholmod",)
PLACEHOLDER_MARK = "ace-jax-coupling GPL placeholder"

REPO = pathlib.Path(__file__).resolve().parents[2]
EXT = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
LIBDIR = "bin" if sys.platform == "win32" else "lib"
# Two process states: the full cases (numpy loaded first, as for any user),
# and a bare ctypes load (no numpy: Julia's loader then opens its own libgcc_s
# etc. from the bundle instead of reusing the system copies numpy pulled in).
DRIVERS = ["""
import json, sys
sys.path.insert(0, sys.argv[1])
import ace_jax_coupling as ajc
for c in json.load(open(sys.argv[2])).values():
    ajc.couple_raw(c["mb"], c["R"], c["Y"])
print("traced-ok")
""", """
import ctypes, os, sys
h = ctypes.CDLL(os.environ["ACEJAX_COUPLING_LIB"], mode=os.RTLD_NOW | os.RTLD_LOCAL)
assert h.etc_abi_version() == 1
print("traced-ok")
"""]


def _is_lib(p):
    n = p.name
    if sys.platform == "win32":
        return n.lower().endswith(".dll")
    return n.endswith(".dylib") if sys.platform == "darwin" else (n.endswith(".so") or ".so." in n)


# Windows: no loader trace variable, so each driver lists the modules mapped into its
# own process (the package's _loaded_images, loaded from source: no package import in
# the bare driver, so numpy stays out of that process).
WIN_DRIVERS = ["""
import json, sys
sys.path.insert(0, sys.argv[1])
import ace_jax_coupling as ajc
from ace_jax_coupling import _loader
for c in json.load(open(sys.argv[2])).values():
    ajc.couple_raw(c["mb"], c["R"], c["Y"])
for p in _loader._loaded_images(): print("MOD", p)
print("traced-ok")
""", """
import ctypes, importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location("_l", pathlib.Path(sys.argv[1]) / "ace_jax_coupling" / "_loader.py")
_l = importlib.util.module_from_spec(spec); spec.loader.exec_module(_l)
lib = pathlib.Path(os.environ["ACEJAX_COUPLING_LIB"])
for d in sorted({f.parent for f in lib.parent.parent.rglob("*.dll")}):
    os.add_dll_directory(str(d))
h = ctypes.CDLL(str(lib))
assert h.etc_abi_version() == 1
for p in _l._loaded_images(): print("MOD", p)
print("traced-ok")
"""]


def _traced_windows(src):
    env = {k: v for k, v in os.environ.items() if not k.startswith("JULIA")}
    home = tempfile.mkdtemp(prefix="prune-home-")
    env.update(HOME=home, USERPROFILE=home, APPDATA=home, LOCALAPPDATA=home)
    env["ACEJAX_COUPLING_LIB"] = str(src / LIBDIR / f"libetcouple.{EXT}")
    root = os.path.normcase(str(src)) + os.sep
    used = set()
    for drv in WIN_DRIVERS:
        r = subprocess.run([sys.executable, "-c", drv, str(REPO / "coupling/python/src"),
                            str(REPO / "coupling/python/tests/data/cases.json")],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0 and "traced-ok" in r.stdout, (r.stdout[-2000:], r.stderr[-3000:])
        for line in r.stdout.splitlines():
            if line.startswith("MOD ") and os.path.normcase(line[4:]).startswith(root):
                used.add(pathlib.Path(line[4:]).relative_to(src))
    return used


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
    root = str(src) + os.sep
    used = set()
    err = ""
    for drv in DRIVERS:
        r = subprocess.run([sys.executable, "-c", drv, str(REPO / "coupling/python/src"),
                            str(REPO / "coupling/python/tests/data/cases.json")],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0 and "traced-ok" in r.stdout, r.stderr[-3000:]
        err += r.stderr
    for line in err.splitlines():
        for tok in line.replace("=", " ").split():
            tok = tok.strip(":;,[]()'\"")            # LD_DEBUG writes "file=/p/lib.so:  ..."
            if tok.startswith(root) and os.path.isfile(tok):
                used.add(pathlib.Path(tok).relative_to(src))
    return used


def main(src, dst):
    src, dst = pathlib.Path(src).resolve(), pathlib.Path(dst).resolve()
    if sys.platform == "win32":
        # DLLs are found by name in the directories the loader registers, and each
        # traced file is one image, so keep exactly the traced files
        used = _traced_windows(src)
        assert pathlib.Path(LIBDIR) / f"libetcouple.{EXT}" in used, "trace did not see libetcouple"
        _write(src, dst, {u: [u] for u in used})
        _replace_gpl(dst)
        return
    opened = traced(src)
    if sys.platform != "darwin":
        # LD_DEBUG reports each library under the name it was opened by
        # (SONAME via rpath, or the dlopen string): keep exactly those names.
        names = {}
        for u in opened:
            names.setdefault(pathlib.Path(os.path.realpath(src / u)).relative_to(src), []).append(u)
        # Static closure over DT_NEEDED: a dependency the trace resolved outside
        # the bundle (e.g. a host libunwind) must still ship, or a clean machine
        # fails with "libunwind.so.8: cannot open shared object file".
        libdirs = [src / "lib", src / "lib" / "julia"]
        todo = [src / r for rels in names.values() for r in rels]
        while todo:
            f = todo.pop()
            needed = subprocess.run(["patchelf", "--print-needed", str(f)], capture_output=True,
                                    text=True, check=True).stdout.split()
            for n in needed:
                cand = next((d / n for d in libdirs if (d / n).exists()), None)
                if cand is None:
                    continue                                  # system library (libc, libm, ...)
                rel = cand.relative_to(src)
                real = pathlib.Path(os.path.realpath(cand)).relative_to(src)
                if rel not in names.get(real, []):
                    names.setdefault(real, []).append(rel)
                    todo.append(cand)
        # one file per library (see the macOS branch): keep the name libjulia's
        # loader dlopens by path, point every DT_NEEDED at it
        loader_names = b"".join((src / u).read_bytes() for u in opened
                                if "libjulia." in u.name and "internal" not in u.name)
        renames = {}
        for real, rels in names.items():
            if len(rels) > 1:
                keep = next((r for r in sorted(rels) if r.name.encode() in loader_names), sorted(rels)[0])
                renames.update({r.name: keep.name for r in rels if r != keep})
                names[real] = [keep]
        _write(src, dst, names)
        for f in sorted(p for p in dst.rglob("*") if p.is_file() and _is_lib(p)):
            needed = subprocess.run(["patchelf", "--print-needed", str(f)], capture_output=True,
                                    text=True, check=True).stdout.split()
            args = [x for old in needed if old in renames for x in ("--replace-needed", old, renames[old])]
            if args:
                subprocess.run(["patchelf", *args, str(f)], check=True)
        _replace_gpl(dst)
        return
    used = {pathlib.Path(os.path.realpath(src / u)).relative_to(src) for u in opened}
    assert pathlib.Path("lib") / f"libetcouple.{EXT}" in used, "trace did not see libetcouple"
    # libjulia's loader opens a fixed dependency list (libgcc_s, libstdc++,
    # libjulia-internal, ...) by path from lib/julia -- unless the process already
    # has a library of that name (numpy pulls in the system libgcc_s first, so the
    # trace never sees the bundled one).  Keep every bundle library the loader
    # names, whatever the trace saw.
    loader = [u for u in used if "libjulia." in u.name and "internal" not in u.name]
    loader_names = b"".join((src / u).read_bytes() for u in loader)
    for p in src.rglob("*"):
        if p.is_file() and _is_lib(p) and p.name.encode() in loader_names:
            used.add(pathlib.Path(os.path.realpath(p)).relative_to(src))
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
    # One file per library: dyld keys images by path, so a library shipped under
    # two names (upstream: a symlink) loads as two images -- a second, never
    # initialised libjulia-internal.  Keep the name libjulia's loader opens by
    # path (a string we cannot rewrite), else the first referenced one, and
    # point every load command at it.
    renames = {}
    for real, rels in names.items():
        if len(rels) > 1:
            keep = next((r for r in rels if r.name.encode() in loader_names), rels[0])
            renames.update({r.name: keep.name for r in rels if r != keep})
            names[real] = [keep]
    _write(src, dst, names)
    _relink_macos(dst, renames)
    _replace_gpl(dst)


def _replace_gpl(dst):
    """Swap each GPL SuiteSparse library for an empty placeholder under the same
    name (SONAME / install name kept, so the JLL's dlopen still succeeds), then
    re-run the trace drivers on the final bundle: they must still pass."""
    done = []
    for f in sorted(p for p in dst.rglob("*") if p.is_file() and _is_lib(p)):
        if not f.name.startswith(PLACEHOLDER_LIBS):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            c = pathlib.Path(tmp) / "placeholder.c"
            # the name makes each placeholder's bytes unique (check_bundle: one file per library)
            c.write_text('#ifdef _WIN32\n__declspec(dllexport)\n#endif\n'
                         f'const char acejax_gpl_placeholder[] = "{PLACEHOLDER_MARK}: {f.name} '
                         '(no SuiteSparse code; never called)";\n')
            out = pathlib.Path(tmp) / f.name
            if sys.platform == "win32":
                subprocess.run(["gcc", "-shared", "-static-libgcc", "-o", str(out), str(c)], check=True)
            elif sys.platform == "darwin":
                ident = subprocess.run(["otool", "-D", str(f)], capture_output=True, text=True,
                                       check=True).stdout.split()[-1]
                subprocess.run(["cc", "-dynamiclib", "-mmacosx-version-min=11.0", "-install_name", ident,
                                "-o", str(out), str(c)], check=True)
                subprocess.run(["codesign", "--force", "--sign", "-", str(out)], check=True, capture_output=True)
            else:
                soname = subprocess.run(["patchelf", "--print-soname", str(f)], capture_output=True,
                                        text=True, check=True).stdout.strip() or f.name
                subprocess.run(["cc", "-shared", "-fPIC", "-nostdlib", f"-Wl,-soname,{soname}",
                                "-o", str(out), str(c)], check=True)
            shutil.copy2(out, f)
        done.append(f.name)
    missing = [g for g in GPL_LIBS if not any(n.startswith(g + ".") or n.startswith(g + "-") for n in done)]
    # Linux/macOS: SuiteSparse_jll's __init__ opens every SuiteSparse library, so all four are
    # traced and must have been replaced (a miss means a renamed file slipped through).  Windows:
    # the trimmed coupling never maps them, so the trace drops them and there is nothing to ship.
    if sys.platform != "win32":
        assert not missing, f"expected one each of {GPL_LIBS} in the bundle; missing {missing}, replaced {done}"
    (_traced_windows if sys.platform == "win32" else traced)(dst)   # the coupling still runs on the placeholders
    print(f"GPL libraries replaced by placeholders: {', '.join(done) or 'none (none traced)'}")


def _relink_macos(dst, renames):
    """Rewrite @rpath/<old> load commands to the kept names, then ad-hoc re-sign
    every rewritten binary (arm64 refuses unsigned code).  Install names are left
    alone: dyld matches a later dlopen("@rpath/<install name>") -- e.g.
    OpenLibm_jll's "@rpath/libopenlibm.4.dylib" -- against images already
    loaded, which is what keeps the dropped aliases unnecessary."""
    for f in sorted(dst.rglob("*.dylib")):
        out = subprocess.run(["otool", "-L", str(f)], capture_output=True, text=True, check=True).stdout
        ident = subprocess.run(["otool", "-D", str(f)], capture_output=True, text=True, check=True).stdout.split()[-1]
        args = []
        for old, new in renames.items():
            if f"@rpath/{old} " in out and ident != f"@rpath/{old}":
                args += ["-change", f"@rpath/{old}", f"@rpath/{new}"]
        if args:
            subprocess.run(["install_name_tool", *args, str(f)], check=True, capture_output=True)
            subprocess.run(["codesign", "--force", "--sign", "-", str(f)], check=True, capture_output=True)


def _write(src, dst, names):
    """Copy all non-library files, and each kept library under each kept name."""
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
            out.chmod(out.stat().st_mode | 0o200)   # artifacts are read-only; strip/patchelf write in place
            kept += 1
    if sys.platform != "darwin":
        # Linux JLL libraries ship with debug info (libstdc++ 21 MB -> 3.7 MB).
        # Never strip libetcouple or the privatized libjulia*: patchelf rewrote
        # their layout and strip corrupts it ("ELF load command ... not aligned").
        for rels in names.values():
            for rel in rels:
                if rel.name != f"libetcouple.{EXT}" and "_libjulia" not in rel.name:
                    subprocess.run(["strip", "--strip-debug", str(dst / rel)], check=True)
    n_names = sum(map(len, names.values()))
    print(f"kept {kept} files: {len(names)} libs under {n_names} names -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
