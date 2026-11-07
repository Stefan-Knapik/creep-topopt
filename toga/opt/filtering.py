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
