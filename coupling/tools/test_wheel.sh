#!/usr/bin/env bash
# Install a built wheel into a fresh venv with an EMPTY HOME and no Julia on
# PATH, run the package tests, and check the library wrote nothing to HOME.
#   bash coupling/tools/test_wheel.sh dist/ace_jax_coupling-0.2.0-py3-none-<tag>.whl [python]
set -euo pipefail
WHEEL=$(realpath "$1"); PY=${2:-python3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mktemp -d); export HOME="$WORK/home"; mkdir -p "$HOME"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) WIN=1 ;; *) WIN=0 ;; esac
if [ "$WIN" = 1 ]; then                                    # Windows resolves "home" from these, not HOME
  export USERPROFILE="$HOME" APPDATA="$HOME/AppData/Roaming" LOCALAPPDATA="$HOME/AppData/Local"
  mkdir -p "$APPDATA" "$LOCALAPPDATA"
fi
unset JULIA_DEPOT_PATH JULIA_PROJECT ACEJAX_COUPLING_LIB PYTHONPATH
# drop every PATH entry that holds a Julia (by name or by content: the Windows runner
# image has a Chocolatey julia shim in C:\ProgramData\Chocolatey\bin)
NEWPATH=""
while IFS= read -r d; do
  case "$d" in *[Jj]ulia*) continue ;; esac
  [ -e "$d/julia" ] || [ -e "$d/julia.exe" ] || [ -e "$d/libjulia.dll" ] && continue
  NEWPATH="${NEWPATH:+$NEWPATH:}$d"
done < <(echo "$PATH" | tr ':' '\n')
export PATH="$NEWPATH"
if command -v julia >/dev/null; then echo "julia still on PATH: $(command -v julia)"; exit 1; fi
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$PY" -m venv "$WORK/venv"
VBIN="$WORK/venv/$([ "$WIN" = 1 ] && echo Scripts || echo bin)"
"$VBIN/python" -m pip install -q "$WHEEL" pytest
cp -r "$REPO/coupling/python/tests" "$WORK/tests"          # run from outside the source tree
BEFORE=$(find "$HOME" | sort)                              # after pip: only the library's writes count
(cd "$WORK" && "$VBIN/python" -m pytest -q tests -p no:cacheprovider)
AFTER=$(find "$HOME" | sort)
if [ "$BEFORE" != "$AFTER" ]; then echo "library wrote to HOME:"; diff <(echo "$BEFORE") <(echo "$AFTER"); exit 1; fi
du -h "$WHEEL"; echo "wheel OK"
