"""Smoke test for examples/quickstart.py."""
import importlib.util
from pathlib import Path

import numpy as np


def test_quickstart_runs():
    path = Path(__file__).parents[1] / "examples" / "quickstart.py"
    spec = importlib.util.spec_from_file_location("quickstart", path)
    quickstart = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quickstart)
    J, dJ = quickstart.main(num_time_steps=2)
    assert J > 0
    assert np.all(np.isfinite(dJ)) and np.linalg.norm(dJ) > 0
