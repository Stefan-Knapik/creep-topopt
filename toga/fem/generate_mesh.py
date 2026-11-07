"""
Copied and modified from https://github.com/deepmodeling/jax-fem
Under GNU General Public License v3.0
"""
import jax
import meshio
import numpy as onp

from toga.fem.basis import get_face_shape_vals_and_grads


class Mesh():
    """Mesh manager.

    Attributes
    ----------
    points : NumpyArray
        Shape is (num_total_nodes, dim).
    cells: NumpyArray
        Shape is (num_cells, num_nodes).
    """
    def __init__(self, points, cells, ele_type='TET4'):
        self.points = points
        self.cells = cells
        self.ele_type = ele_type

    def count_selected_faces(self, location_fn):
        """Given location functions, compute the number of faces that satisfy the location function. 
        Useful for setting up distributed load conditions.

        Parameters
        ----------
        location_fns : list
            :attr:`~jax_fem.problem.Problem.location_fns`

        Returns
        -------
        face_count : int
        """
        _, _, _, _, face_inds = get_face_shape_vals_and_grads(self.ele_type)
        cell_points = onp.take(self.points, self.cells, axis=0)
        cell_face_points = onp.take(cell_points, face_inds, axis=1)

        vmap_location_fn = jax.vmap(location_fn)

        def on_boundary(cell_points):
            boundary_flag = vmap_location_fn(cell_points)
            return onp.all(boundary_flag)

        vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
        boundary_flags = vvmap_on_boundary(cell_face_points)
        boundary_inds = onp.argwhere(boundary_flags)
        return boundary_inds.shape[0]


def get_meshio_cell_type(ele_type):
    """Convert element type into a compatible string with 
    `meshio <https://github.com/nschloe/meshio/blob/9dc6b0b05c9606cad73ef11b8b7785dd9b9ea325/src/meshio/xdmf/common.py#L36>`_.

    Parameters
    ----------
    ele_type : str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`

    Returns
    -------
    cell_type : str
        Compatible with meshio.
    """
    if ele_type == 'TET4':
        cell_type = 'tetra'
    elif ele_type == 'TET10':
        cell_type = 'tetra10'
    elif ele_type == 'HEX8':
        cell_type = 'hexahedron'
    elif ele_type == 'HEX27':
        cell_type = 'hexahedron27'
    elif  ele_type == 'HEX20':
        cell_type = 'hexahedron20'
    elif ele_type == 'TRI3':
        cell_type = 'triangle'
    elif ele_type == 'TRI6':
        cell_type = 'triangle6'
    elif ele_type == 'QUAD4':
        cell_type = 'quad'
    elif ele_type == 'QUAD8':
        cell_type = 'quad8'
    else:
        raise NotImplementedError
    return cell_type


def rectangle_mesh(Nx, Ny, domain_x, domain_y):
    """Generate QUAD4 mesh.

    Parameters
    ----------
    Nx : int
        Number of elements along x-axis.
    Ny : int
        Number of elements along y-axis.
    domain_x : float
        Length of side along x-axis.
    domain_y : float
        Length of side along y-axis.
    """
    dim = 2
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing='ij')
    points_xy = onp.stack((xv, yv), axis=dim)
    points = points_xy.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xy = points_inds.reshape(Nx + 1, Ny + 1)
    inds1 = points_inds_xy[:-1, :-1]
    inds2 = points_inds_xy[1:, :-1]
    inds3 = points_inds_xy[1:, 1:]
    inds4 = points_inds_xy[:-1, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4), axis=dim).reshape(-1, 4)
    out_mesh = meshio.Mesh(points=points, cells={'quad': cells})
    return out_mesh
