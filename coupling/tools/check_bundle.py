"""Check a libetcouple bundle: loads, ABI 1, max order 8, and the platform floor
(macOS: every Mach-O minos <= 11.0; Linux: no GLIBC_ symbol newer than 2.28).
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
    pats = ("*.dylib",) if sys.platform == "darwin" else ("*.so", "*.so.*")
    return sorted({p for pat in pats for p in root.rglob(pat) if p.is_file() and not p.is_symlink()})


def check(root):
    root = pathlib.Path(root).resolve()
    info = json.loads((root / "build_info.json").read_text())
    ext = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(str(root / "lib" / f"libetcouple.{ext}"), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    assert lib.etc_abi_version() == 1 == info["abi"], "ABI mismatch"
    assert lib.etc_max_order() == 8 == info["max_order"], "max order mismatch"
    bad = []
    for f in _libs(root):
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
    print(f"bundle OK: {info['platform']} et_rev={info['et_rev'][:12]} libs={len(_libs(root))} {platform.machine()}")


if __name__ == "__main__":
    check(sys.argv[1])
