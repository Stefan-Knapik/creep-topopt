"""Density filter: a normalized, local weighted average of the design field."""
import jax.numpy as jnp
import numpy as np
import pytest

from toga.opt.filtering import applyDensityFilter, compute_filter_kd_tree
from toga.physics.creep_explicit import CreepExplicit


@pytest.fixture(scope="module")
def fe_and_filter(make_mesh, norton_material):
    mesh = make_mesh(12, 6, 2.0, 1.0)
    problem = CreepExplicit(mesh, vec=2, dim=2, ele_type="QUAD4",
                            dirichlet_bc_info=[[lambda p: jnp.isclose(p[0], 0.0)], [0], [lambda p: 0.0]],
                            additional_info=(norton_material(1.0, 1.0, 1.0),))
    H, Hs = compute_filter_kd_tree(problem.fe, rmin=1.5)
    return problem.fe, {"H": H, "Hs": Hs}


def test_filter_preserves_constant_field(fe_and_filter):
    fe, ft = fe_and_filter
    rho = jnp.full((fe.num_cells, 1), 0.37)
    np.testing.assert_allclose(applyDensityFilter(ft, rho), rho, rtol=1e-12)


def test_filter_is_local_convex_average(fe_and_filter):
    fe, ft = fe_and_filter
    W = np.asarray(ft["H"].todense()) / np.asarray(ft["Hs"])[:, None]
    assert np.all(W >= 0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(W > 0, W.T > 0)  # symmetric neighbourhoods

    rho = np.random.default_rng(0).uniform(0.0, 1.0, (fe.num_cells, 1))
    out = np.asarray(applyDensityFilter(ft, jnp.asarray(rho)))
    assert rho.min() <= out.min() and out.max() <= rho.max()
    assert np.std(out) < np.std(rho)
