#!/usr/bin/env bash
# Install a built wheel into a fresh venv with an EMPTY HOME and no Julia on
# PATH, run the package tests, and check the library wrote nothing to HOME.
#   bash coupling/tools/test_wheel.sh dist/ace_jax_coupling-0.1.0-py3-none-<tag>.whl [python]
set -euo pipefail
WHEEL=$(realpath "$1"); PY=${2:-python3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mktemp -d); export HOME="$WORK/home"; mkdir -p "$HOME"
unset JULIA_DEPOT_PATH JULIA_PROJECT ACEJAX_COUPLING_LIB PYTHONPATH
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -i julia | paste -sd: -); export PATH
if command -v julia >/dev/null; then echo "julia still on PATH: $(command -v julia)"; exit 1; fi
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q "$WHEEL" pytest
cp -r "$REPO/coupling/python/tests" "$WORK/tests"          # run from outside the source tree
BEFORE=$(find "$HOME" | sort)                              # after pip: only the library's writes count
(cd "$WORK" && "$WORK/venv/bin/python" -m pytest -q tests -p no:cacheprovider)
AFTER=$(find "$HOME" | sort)
if [ "$BEFORE" != "$AFTER" ]; then echo "library wrote to HOME:"; diff <(echo "$BEFORE") <(echo "$AFTER"); exit 1; fi
du -h "$WHEEL"; echo "wheel OK"
