"""Wheel hook: embed the compiled bundle and tag the wheel for its platform.

ACEJAX_COUPLING_BUNDLE  path to a (pruned) bundle dir with build_info.json
ACEJAX_COUPLING_PLAT    wheel platform tag, e.g. manylinux_2_28_x86_64, macosx_11_0_arm64, win_amd64
Without ACEJAX_COUPLING_BUNDLE a lib-less py3-none-any development wheel is built."""
import os
import pathlib
import warnings

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        bundle = os.environ.get("ACEJAX_COUPLING_BUNDLE")
        if not bundle:
            warnings.warn("ACEJAX_COUPLING_BUNDLE unset: building a lib-less development wheel", stacklevel=1)
            return
        root = pathlib.Path(bundle).resolve()
        if not (root / "build_info.json").is_file():
            raise RuntimeError(f"{root} is not a coupling bundle (no build_info.json)")
        plat = os.environ.get("ACEJAX_COUPLING_PLAT")
        if not plat:
            raise RuntimeError("ACEJAX_COUPLING_PLAT must be set with ACEJAX_COUPLING_BUNDLE")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{plat}"
        for p in root.rglob("*"):
            if p.is_file():
                build_data["force_include"][str(p)] = f"ace_jax_coupling/_lib/{p.relative_to(root)}"
