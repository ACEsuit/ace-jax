"""Check a libetcouple bundle: loads, ABI 1, max order 8, the platform floor
(macOS: every Mach-O minos <= 11.0; Linux: no GLIBC_ symbol newer than 2.28; Windows:
none), and no GPL code: each GPL SuiteSparse library is prune_bundle.py's placeholder.
    python coupling/tools/check_bundle.py coupling/build/bundle"""
import ctypes
import json
import os
import pathlib
import platform
import re
import subprocess
import sys

MACOS_MAX = (11, 0)
GLIBC_MAX = (2, 28)


def _libs(root):
    pats = {"darwin": ("*.dylib",), "win32": ("*.dll",)}.get(sys.platform, ("*.so", "*.so.*"))
    return sorted({p for pat in pats for p in root.rglob(pat) if p.is_file() and not p.is_symlink()})


def _pe_exports(f):
    """Exported names of a PE (Windows DLL) file, read from its export directory
    (no binutils/dumpbin needed on the runner)."""
    b = f.read_bytes()
    u = lambda o, n: int.from_bytes(b[o:o + n], "little")
    pe = u(0x3C, 4)
    assert b[pe:pe + 4] == b"PE\0\0", f"{f}: not a PE file"
    nsec, optsz, opt = u(pe + 6, 2), u(pe + 20, 2), pe + 24
    exp_rva = u(opt + (112 if u(opt, 2) == 0x20B else 96), 4)    # data directory 0 (PE32+ / PE32)
    if exp_rva == 0:
        return set()
    sects = [(u(s + 12, 4), max(u(s + 8, 4), u(s + 16, 4)), u(s + 20, 4))
             for s in (opt + optsz + 40 * i for i in range(nsec))]
    off = lambda rva: next(raw + rva - va for va, size, raw in sects if va <= rva < va + size)
    e = off(exp_rva)
    names = off(u(e + 32, 4))
    out = set()
    for i in range(u(e + 24, 4)):
        p = off(u(names + 4 * i, 4))
        out.add(b[p:b.index(b"\0", p)].decode())
    return out


def _exports(f):
    """Defined dynamic symbols, minus linker-reserved (_-prefixed) ones.  A placeholder
    exports only `acejax_gpl_placeholder`; a real SuiteSparse library exports its API.
    (Not a size test: aarch64's 64 KiB page alignment pads even an empty library.)"""
    if sys.platform == "win32":
        return {n for n in _pe_exports(f) if not n.startswith("_")}
    if sys.platform == "darwin":
        out = subprocess.run(["nm", "-gU", str(f)], capture_output=True, text=True, check=True).stdout
        names = [ln.split()[-1][1:] for ln in out.splitlines() if ln.strip()]   # Mach-O adds one '_'
    else:
        out = subprocess.run(["nm", "-D", "--defined-only", str(f)], capture_output=True, text=True,
                             check=True).stdout
        names = [ln.split()[-1] for ln in out.splitlines() if ln.strip()]
    return {n for n in names if not n.startswith("_")}


def check(root):
    root = pathlib.Path(root).resolve()
    info = json.loads((root / "build_info.json").read_text())
    if sys.platform == "win32":
        for d in sorted({f.parent for f in _libs(root)}):
            os.add_dll_directory(str(d))
        lib = ctypes.CDLL(str(root / "bin" / "libetcouple.dll"))
    else:
        ext = "dylib" if sys.platform == "darwin" else "so"
        lib = ctypes.CDLL(str(root / "lib" / f"libetcouple.{ext}"), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    assert lib.etc_abi_version() == 1 == info["abi"], "ABI mismatch"
    assert lib.etc_max_order() == 8 == info["max_order"], "max order mismatch"
    bad = []
    for f in _libs(root):
        if sys.platform == "win32":
            continue                                  # no OS-version floor encoded in the DLLs to check
        if sys.platform == "darwin":
            out = subprocess.run(["otool", "-l", str(f)], capture_output=True, text=True).stdout
            for v in re.findall(r"minos (\d+)\.(\d+)", out):
                if tuple(map(int, v)) > MACOS_MAX:
                    bad.append(f"{f.relative_to(root)} minos {'.'.join(v)}")
        else:
            out = subprocess.run(["objdump", "-T", str(f)], capture_output=True, text=True).stdout
            for v in set(re.findall(r"GLIBC_(\d+)\.(\d+)", out)):
                if tuple(map(int, v)) > GLIBC_MAX:
                    bad.append(f"{f.relative_to(root)} GLIBC_{'.'.join(v)}")
    assert not bad, "platform floor violated:\n  " + "\n  ".join(bad)
    # one file per library: two copies under different names load as two
    # images with separate state (a second, uninitialised libjulia-internal)
    import hashlib
    seen = {}
    for f in _libs(root):
        seen.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), []).append(str(f.relative_to(root)))
    dups = [v for v in seen.values() if len(v) > 1]
    assert not dups, f"duplicate library copies (would load twice): {dups}"
    gpl = ("libumfpack", "libspqr", "librbio", "libcholmod", "libklu_cholmod")   # == prune_bundle.PLACEHOLDER_LIBS
    real = [f"{f.relative_to(root)} exports {sorted(x)[:3]}" for f in _libs(root) if f.name.startswith(gpl)
            and (x := _exports(f) - {"acejax_gpl_placeholder"} or
                 ({"<no marker>"} if b"ace-jax-coupling GPL placeholder" not in f.read_bytes() else set()))]
    assert not real, f"GPL SuiteSparse libraries in the bundle (not placeholders): {real}"
    print(f"bundle OK: {info['platform']} et_rev={info['et_rev'][:12]} libs={len(_libs(root))} {platform.machine()}")


if __name__ == "__main__":
    check(sys.argv[1])
