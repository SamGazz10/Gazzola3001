"""The notebooks must run from top to bottom against the current code and model.

`network_by_hand.ipynb` also asserts that its NumPy forward pass and gradient
match the library, so this test fails if the model and the notebook drift apart.
"""
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["network_by_hand.ipynb", "walkthrough.ipynb"])
def test_notebook_runs(name):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    NotebookClient(notebook, timeout=300, resources={"metadata": {"path": str(ROOT)}}).execute()
