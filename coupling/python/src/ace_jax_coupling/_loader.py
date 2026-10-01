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


def _win_dll_dirs(root):
    """Make the bundle's DLL directories win every DLL lookup in this process.
    add_dll_directory covers ctypes' load of libetcouple and its import table, but the
    embedded Julia runtime loads its dependencies by name at run time through the
    standard search (application directory, then PATH), which add_dll_directory does
    not reach. Prepending to PATH makes the bundle's copies win over any other DLL of
    the same name elsewhere on PATH."""
    dirs = sorted({str(f.parent) for f in pathlib.Path(root).rglob("*.dll")})
    for d in dirs:
        os.add_dll_directory(d)
    path = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(dirs + [d for d in path if d and d not in dirs])


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
                _win_dll_dirs(bundle_root())
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


def _loaded_images():
    """Paths of every shared library mapped into this process (prune tracing on
    Windows, and the load-once test). Diagnostics only; it never loads anything."""
    if sys.platform == "darwin":
        d = ctypes.CDLL(None)
        d._dyld_get_image_name.restype = ctypes.c_char_p
        return [d._dyld_get_image_name(i).decode() for i in range(d._dyld_image_count())]
    if sys.platform == "win32":
        from ctypes import wintypes
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                               wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        k32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        mods, need = (wintypes.HMODULE * 4096)(), wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(k32.GetCurrentProcess(), mods, ctypes.sizeof(mods),
                                          ctypes.byref(need), 3):           # LIST_MODULES_ALL
            raise OSError(ctypes.get_last_error(), "EnumProcessModulesEx failed")
        buf = ctypes.create_unicode_buffer(32768)
        out = []
        for m in mods[: need.value // ctypes.sizeof(wintypes.HMODULE)]:
            if k32.GetModuleFileNameW(m, buf, len(buf)):
                out.append(buf.value)
        return out
    with open("/proc/self/maps") as f:
        return [ln.split()[-1] for ln in f if len(ln.split()) >= 6]
