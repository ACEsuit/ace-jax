"""The tutorials' cell graphs: each notebook, and the edits its exercises ask for, must
form a DAG (marimo refuses to run a notebook with a cycle). marimo is a docs dependency
(docs/requirements.txt), so the docs CI job runs this file; the test shards skip it."""
import importlib.util
import pathlib

import pytest

pytest.importorskip("marimo")

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "docs" / "user" / "tutorials" / "notebooks"


def _graph(src, tmp_path):
    from marimo._ast.app import InternalApp
    p = tmp_path / "nb.py"; p.write_text(src)
    spec = importlib.util.spec_from_file_location("nb", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return InternalApp(m.app).graph


def test_tutorial_4_exercise_3_adds_the_vacancy_to_the_training_set_without_a_cycle(tmp_path):
    src = (NB / "school_dataset_si.py").read_text()
    edit = "load_fit_data(cfg, train=labelled_bulk, "
    assert edit in src
    _graph(src.replace(edit, "load_fit_data(cfg, train=labelled_bulk + [labelled_probe[1]], "), tmp_path)


def test_tutorial_3_exercise_3_drops_a_table_column_without_a_cycle(tmp_path):
    src = (NB / "multi_element.py").read_text()
    edit = "np.array([_valence, _pauling], float).T"
    assert edit in src
    _graph(src.replace(edit, "np.array([_valence], float).T"), tmp_path)
