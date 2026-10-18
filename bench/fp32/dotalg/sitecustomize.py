# bench/fp32/dot_algorithm.py's hook: on PYTHONPATH, every Python process installs $ACEJAX_EXP_DOT_ALGORITHM
import os
import sys

if os.environ.get("ACEJAX_EXP_DOT_ALGORITHM"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import dot_algorithm
    dot_algorithm.install_from_env()
