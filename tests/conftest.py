"""Shared fixtures: float64 JAX on CPU, quiet logging, small meshes and Norton materials."""
import logging

import jax
import pytest

import toga
from toga.fem.generate_mesh import Mesh, rectangle_mesh
from toga.mat import build_material_properties

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
toga.logger.setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def make_mesh():
    def _make(nx, ny, lx, ly):
        m = rectangle_mesh(Nx=nx, Ny=ny, domain_x=lx, domain_y=ly)
        return Mesh(m.points, m.cells_dict["quad"], ele_type="QUAD4")
    return _make


@pytest.fixture(scope="session")
def norton_material():
    def _make(E, A, n):
        return build_material_properties(
            material_names=["dummy"],
            property_configs={
                "E": {"model": "constant", "value": E},
                "A": {"model": "constant", "value": A},
                "n": {"model": "constant", "value": n},
            },
        )
    return _make
