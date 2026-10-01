"""Locate, load and type the libetcouple shared library (lazily)."""
import ctypes
import json
import os
import pathlib
import sys
import threading
from ctypes import c_int32, c_int64, c_void_p

ABI_VERSION = 1
_EXT = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
_LIBDIR = "bin" if sys.platform == "win32" else "lib"      # Windows bundles keep the DLLs in bin/
_PKG = pathlib.Path(__file__).resolve().parent
_LIB = None
LOCK = threading.Lock()


class CouplingLibError(RuntimeError):
    """The native coupling library is missing, unloadable or incompatible."""


def lib_path() -> pathlib.Path:
    env = os.environ.get("ACEJAX_COUPLING_LIB")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return _PKG / "_lib" / _LIBDIR / f"libetcouple.{_EXT}"


def bundle_root() -> pathlib.Path:
    return lib_path().parent.parent


def build_info() -> dict:
    """The bundle's build_info.json. Never loads the library."""
    p = bundle_root() / "build_info.json"
    if not p.is_file():
        raise CouplingLibError(
            f"no coupling library build found ({p} missing). This is a source/development "
            "build of ace-jax-coupling without the compiled bundle: install a platform wheel, "
            "or set ACEJAX_COUPLING_LIB to <bundle>/lib/libetcouple." + _EXT)
    return json.loads(p.read_text())


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with LOCK:
        if _LIB is not None:
            return _LIB
        build_info()                                   # clear error if the bundle is absent
        p = lib_path()
        try:
            if sys.platform == "win32":
                # dependent DLLs resolve from directories registered here, not from PATH
                root = bundle_root()
                for d in sorted({f.parent for f in root.rglob("*.dll")}):
                    os.add_dll_directory(str(d))
                h = ctypes.CDLL(str(p))
            else:
                h = ctypes.CDLL(str(p), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        except OSError as e:
            raise CouplingLibError(f"cannot load {p}: {e}") from e
        h.etc_abi_version.restype = c_int32
        h.etc_max_order.restype = c_int32
        if h.etc_abi_version() != ABI_VERSION:
            raise CouplingLibError(f"{p}: ABI {h.etc_abi_version()} != expected {ABI_VERSION}")
        h.etc_couple.restype = c_int32
        h.etc_couple.argtypes = ([c_int64, c_void_p, c_void_p, c_void_p,
                                  c_int64, c_void_p, c_void_p,
                                  c_int64, c_void_p, c_void_p,
                                  c_void_p] + [c_void_p] * 10)
        _LIB = h
        return _LIB
