"""Material models: exact at tabulated alloys/temperatures, linear blend between two alloys."""
import numpy as np
import pandas as pd
import pytest

from toga.mat import build_material_properties
from toga.mat.material_properties import DATA_DIR

ALLOYS = ["IN625", "SS316"]


@pytest.fixture(scope="module")
def props():
    return build_material_properties(ALLOYS, {
        "E": {"model": "rbf"},
        "logA": {"model": "rbf", "temperature_dependent": False},
        "n": {"model": "rbf", "temperature_dependent": False},
    })


@pytest.fixture(scope="module")
def tables():
    return {a: pd.read_csv(DATA_DIR / f"{a}.csv", skipinitialspace=True) for a in ALLOYS}


@pytest.mark.parametrize("i", [0, 1])
def test_reproduces_tabulated_E(props, tables, i):
    table = tables[ALLOYS[i]][["T", "E"]].dropna()
    x = np.zeros((len(table), 3))
    x[:, i] = 1.0
    x[:, 2] = table["T"]
    np.testing.assert_allclose(props.E(x), table["E"], rtol=1e-8)


@pytest.mark.parametrize("name", ["logA", "n"])
def test_composition_only_properties(props, tables, name):
    y = [tables[a][name].iloc[0] for a in ALLOYS]
    pure = np.eye(2)
    np.testing.assert_allclose(getattr(props, name)(pure), y, rtol=1e-10)
    # With two alloys and a linear kernel the RBF fit is exactly the rule of mixtures.
    for c in (0.25, 0.5, 0.9):
        mix = getattr(props, name)(np.array([c, 1 - c]))
        np.testing.assert_allclose(mix, c * y[0] + (1 - c) * y[1], rtol=1e-10)


@pytest.fixture(scope="module")
def table_props():
    return build_material_properties(ALLOYS, {
        "E": {"model": "table"},
        "n": {"model": "table", "temperature_dependent": False},
    })


def test_table_model_is_piecewise_linear_and_mixes_linearly(table_props, tables):
    E = {a: tables[a][["T", "E"]].dropna() for a in ALLOYS}
    for T in (0.0, 600.0, 749.0, 750.0, 800.0, 1000.0):
        pure = [np.interp(T, E[a]["T"], E[a]["E"]) for a in ALLOYS]
        np.testing.assert_allclose(table_props.E(np.array([1.0, 0.0, T])), pure[0], rtol=1e-12)
        np.testing.assert_allclose(table_props.E(np.array([0.0, 1.0, T])), pure[1], rtol=1e-12)
        np.testing.assert_allclose(table_props.E(np.array([0.3, 0.7, T])), 0.3 * pure[0] + 0.7 * pure[1], rtol=1e-12)
    # Held at the end values outside the tabulated range.
    np.testing.assert_allclose(table_props.E(np.array([1.0, 0.0, 1200.0])), E["IN625"]["E"].iloc[-1], rtol=1e-12)
    # Batched queries and temperature-independent properties.
    assert table_props.E(np.array([[1.0, 0.0, 600.0], [0.0, 1.0, 800.0]])).shape == (2,)
    n = [tables[a]["n"].iloc[0] for a in ALLOYS]
    np.testing.assert_allclose(table_props.n(np.array([0.25, 0.75])), 0.25 * n[0] + 0.75 * n[1], rtol=1e-12)
