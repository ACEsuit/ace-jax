#!/usr/bin/env bash
# YE3T Python stack for bench/accuracy_cost/fit_ye3t.py: torch (CPU), ye3t f8ff96d, ye3t-methods 6b85c54 (neither is on PyPI),
# built from clones under $ACEJAX_AC_ROOT/ext (git clone https://github.com/ye3t-equivariance/{ye3t,ye3t-methods}).
# Needs a C++20 compiler (GCC 11.5 here) and CMake.  envs/ye3t/freeze.txt is the resulting pip freeze.
set -euo pipefail
export UV_CACHE_DIR=/storage/eng/essswb/cache/uv PATH=$HOME/.local/bin:$PATH
V=/storage/eng/essswb/venvs/ac-ye3t; X=/storage/eng/essswb/projects/ace-jax/ext
[ -d $V ] || uv venv -q --python 3.12 $V
uv pip install -q --python $V/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -q --python $V/bin/python 'setuptools>=77,<82' wheel cmake numpy ase sympy pyyaml psutil matplotlib scikit-learn matscipy spglib phonopy pytest pip
export OMP_NUM_THREADS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 MAKEFLAGS=-j16
cd $X && $V/bin/python -m pip install --no-build-isolation ./ye3t
cd $X && $V/bin/python -m pip install --no-build-isolation './ye3t-methods[examples,fit,neighbors]'
$V/bin/python -c "import ye3t, ye3t_methods, torch; print('ok', torch.__version__)"
$V/bin/python -m pip freeze > $X/logs/ac-ye3t.freeze.txt
