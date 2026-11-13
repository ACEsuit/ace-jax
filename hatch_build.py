"""Wheel hook: embed the compiled C++ kernel and tag the wheel for its platform.

ACEJAX_KERNEL_DIR   a directory written by `ace_jax.eval.ffi.build_prebuilt` (manifest.json + the libraries)
ACEJAX_KERNEL_PLAT  wheel platform tag, e.g. manylinux_2_28_x86_64, manylinux_2_28_aarch64, macosx_11_0_arm64
Without ACEJAX_KERNEL_DIR the wheel is pure Python (py3-none-any): the kernel is then compiled on first use.
The libraries are loaded with ctypes, so the wheel is tagged for a platform, not for a Python version."""
import hashlib
import json
import os
import pathlib

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        kernel = os.environ.get("ACEJAX_KERNEL_DIR")
        if self.target_name != "wheel" or not kernel:
            return
        root = pathlib.Path(kernel).resolve()
        plat = os.environ.get("ACEJAX_KERNEL_PLAT")
        if not (root / "manifest.json").is_file():
            raise RuntimeError(f"{root} is not a prebuilt kernel (no manifest.json)")
        if not plat:
            raise RuntimeError("ACEJAX_KERNEL_PLAT must be set with ACEJAX_KERNEL_DIR")
        info = json.loads((root / "manifest.json").read_text())
        source = pathlib.Path(self.root) / "src" / "ace_jax" / "eval" / "ace_ffi.cc"
        if info["source"] != hashlib.sha256(source.read_bytes()).hexdigest()[:16]:   # = eval.ffi.source_key()
            raise RuntimeError(f"{root} was compiled from another ace_ffi.cc: the installed package would ignore it")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{plat}"
        for f in ["manifest.json", *(v["file"] for v in info["variants"])]:
            build_data["force_include"][str(root / f)] = f"ace_jax/eval/_prebuilt/{f}"
