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


@pytest.fixture(scope="module")
def fe_3d():
    """A HEX8 grid with graded spacing, as the minimal `fe` the filters read (points, cells, JxW, flex_inds)."""
    from types import SimpleNamespace
    n = 10
    x = np.linspace(0.0, 1.0, n + 1) ** 1.3 * n   # element sizes from about 0.5 to 1.3
    y = z = np.arange(n + 1, dtype=float)
    points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    node = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
    cells = np.array([[node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                       node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)]
                      for i in range(n) for j in range(n) for k in range(n)])
    volume = np.repeat(np.diff(x), n * n)
    JxW = np.repeat(volume[:, None] / 8.0, 8, axis=1)
    return SimpleNamespace(points=points, cells=cells, JxW=JxW, flex_inds=np.arange(len(cells)),
                           num_cells=len(cells), dim=3)


def test_radius_filter_3d(fe_3d):
    from toga.opt.filtering import compute_filter_radius
    fe = fe_3d
    radius = 2.5   # 2.5 average element sizes (the average element has unit volume)
    H, Hs = compute_filter_radius(fe, radius)
    ft = {"H": H, "Hs": Hs}
    W = np.asarray(H.todense()) / np.asarray(Hs)[:, None]
    assert np.all(W >= 0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-12)

    # a composition stays on the simplex and a constant is preserved
    rho = jnp.full((fe.num_cells, 2), 0.37)
    np.testing.assert_allclose(applyDensityFilter(ft, rho), rho, rtol=1e-12)
    x = np.random.default_rng(0).uniform(0.0, 1.0, (fe.num_cells, 1))
    out = np.asarray(applyDensityFilter(ft, jnp.asarray(np.concatenate((x, 1.0 - x), axis=1))))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-12)
    assert x.min() <= out[:, 0].min() and out[:, 0].max() <= x.max()

    # the neighbourhood is the whole ball: more than the 20 cells the k-d tree filter keeps
    centroids = fe.points[fe.cells].mean(axis=1)
    interior = np.all((centroids > 3.0) & (centroids < 7.0), axis=1)
    neighbours = (W > 0).sum(axis=1)
    assert neighbours[interior].min() > 20
    dist = np.linalg.norm(centroids[:, None] - centroids[None], axis=-1)
    np.testing.assert_array_equal(W > 0, dist < radius)

    # weights: cone in the distance times the neighbour's volume
    i = int(np.flatnonzero(interior)[0])
    volume = fe.JxW.sum(axis=1)
    expected = np.maximum(radius - dist[i], 0.0) * volume
    np.testing.assert_allclose(W[i], expected / expected.sum(), rtol=1e-12)
