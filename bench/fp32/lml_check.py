"""Plan step 1.3 (LML): `analyse.lml_check` for each basis named on the command line, as JSON."""
import json
import sys

from analyse import lml_check

print(json.dumps([lml_check(m) for m in sys.argv[1:]], indent=1))
