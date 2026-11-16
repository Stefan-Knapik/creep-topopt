"""
Copied and modified from https://github.com/deepmodeling/jax-fem
Under GNU General Public License v3.0
"""

import numpy as np


def compute_filter_kd_tree(fe, rmin):
    """Density-filter matrix H and row sums Hs over the flexible cells.

    Linear cone weights with radius ``rmin`` given in average element sizes; neighbours from a k-d tree (20 nearest).
    """
    import scipy
    from jax.experimental.sparse import BCOO
    cell_centroids = np.mean(np.take(fe.points, fe.cells, axis=0), axis=1)
    flex_num_cells = len(fe.flex_inds)
    flex_cell_centroids = np.take(cell_centroids, fe.flex_inds, axis=0)

    V = np.sum(fe.JxW)
    avg_elem_V = V/fe.num_cells

    avg_elem_size = avg_elem_V**(1./fe.dim)
    rmin = rmin * avg_elem_size

    kd_tree = scipy.spatial.KDTree(flex_cell_centroids)
    num_nbs = 20
    dd, ii = kd_tree.query(flex_cell_centroids, num_nbs)
    I = np.repeat(np.arange(flex_num_cells), num_nbs)
    J = ii.reshape(-1)
    V = np.where(rmin - dd > 0., rmin - dd, 0.).reshape(-1)
    H_sp = scipy.sparse.csc_array((V, (I, J)), shape=(flex_num_cells, flex_num_cells))

    H = BCOO.from_scipy_sparse(H_sp).sort_indices()
    Hs = H.sum(1).todense()
    return H, Hs


def applyDensityFilter(ft, rho):
    return ft['H'] @ rho / ft['Hs'][:, None]


def compute_filter_radius(fe, radius, volume_weighted=True):
    """Density-filter matrix H and row sums Hs with a cone kernel of a given physical radius.

    H_ij = max(radius - |x_i - x_j|, 0) * v_j over all cells j whose centroid lies within ``radius``
    (length units of the mesh) of that of cell i, with v_j the cell volume (1 if not
    ``volume_weighted``). Unlike :func:`compute_filter_kd_tree` the neighbourhood is not capped at
    20 cells, so the radius is honoured in 3D, and it does not depend on the mesh size. Use with
    :func:`applyDensityFilter`; the rows of H / Hs sum to one, so the filter is a local average that
    keeps a composition on the simplex.
    """
    import scipy
    from jax.experimental.sparse import BCOO
    cell_centroids = np.mean(np.take(np.asarray(fe.points), np.asarray(fe.cells), axis=0), axis=1)
    flex_inds = np.asarray(fe.flex_inds)
    centroids = np.take(cell_centroids, flex_inds, axis=0)
    num = len(flex_inds)
    volume = np.take(np.sum(np.asarray(fe.JxW), axis=1), flex_inds) if volume_weighted else np.ones(num)

    pairs = scipy.spatial.cKDTree(centroids).query_pairs(radius, output_type='ndarray')
    dist = np.linalg.norm(centroids[pairs[:, 0]] - centroids[pairs[:, 1]], axis=1)
    diag = np.arange(num)
    I = np.concatenate((diag, pairs[:, 0], pairs[:, 1]))
    J = np.concatenate((diag, pairs[:, 1], pairs[:, 0]))
    W = np.concatenate((np.full(num, radius), radius - dist, radius - dist)) * volume[J]
    H_sp = scipy.sparse.csr_array((W, (I, J)), shape=(num, num))

    H = BCOO.from_scipy_sparse(H_sp).sort_indices()
    Hs = H.sum(1).todense()
    return H, Hs
