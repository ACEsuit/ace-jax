import json
import pathlib

import numpy as np
import pytest

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cases():
    return json.loads((DATA / "cases.json").read_text())


@pytest.fixture(scope="session")
def reference():
    return np.load(DATA / "et_reference.npz")
