"""File I/O, timing and device-array helpers."""
import hashlib
import os
import time
from functools import wraps

import jax
import jax.numpy as jnp
import meshio
import numpy as onp

from toga import logger
from toga.fem.generate_mesh import get_meshio_cell_type

_shared_device_arrays = {}

def _device_arrays_hoisted():
    """True if closed-over jax.Arrays are passed to compiled programs as arguments.

    That needs JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1 in the environment before jax is
    imported: jax.config.update after import sets the flag but leaves jax.Array constants
    embedded in the executable.
    """
    try:
        from jax._src import core as jax_core
        from jax._src.array import ArrayImpl
        return jax.config.jax_use_simplified_jaxpr_constants and ArrayImpl in jax_core.literalable_types
    except (ImportError, AttributeError):
        return False

def shared_device_array(x, dtype=None):
    """Device copy of a NumPy array, shared by all arrays with the same contents.

    With JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1, traced code that closes over a NumPy array
    hoists a separate device copy at every use; a shared device array is passed to the
    compiled program as one buffer. Without that flag constants are embedded in the
    executable, where a device copy would only duplicate them, so the NumPy array is returned.
    """
    a = onp.asarray(x, dtype=dtype)
    if not _device_arrays_hoisted():
        return a
    key = (a.shape, a.dtype.str, hashlib.sha1(a.tobytes()).digest())
    if key not in _shared_device_arrays:
        with jax.ensure_compile_time_eval():  # concrete even when first called while tracing
            _shared_device_arrays[key] = jnp.asarray(a)
    return _shared_device_arrays[key]

def clean_data_dir(data_dir, keep_subdir="jax_cache"):
    """Delete all files in data_dir except those inside keep_subdir."""
    keep_dir = os.path.abspath(os.path.join(data_dir, keep_subdir))
    for root, dirs, files in os.walk(data_dir):
        if os.path.abspath(root) == keep_dir:
            continue
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != keep_dir]
        for f in files:
            os.remove(os.path.join(root, f))


def remove_orphan_nodes(meshio_mesh, cell_type, return_index_map=False):
    """Return (points, cells) with unreferenced nodes removed.

    Orphan nodes produce all-zero rows in the stiffness matrix and cause
    direct solvers (e.g. PARDISO) to report a singular system.

    If ``return_index_map`` is True, also returns a callable mapping original
    node indices to their new indices; orphan nodes map to -1.
    """
    raw_cells = meshio_mesh.cells_dict[cell_type]
    used_mask = onp.zeros(len(meshio_mesh.points), dtype=bool)
    used_mask[raw_cells.flatten()] = True
    remap = onp.full(len(meshio_mesh.points), -1, dtype=onp.int32)
    remap[used_mask] = onp.arange(used_mask.sum(), dtype=onp.int32)
    if return_index_map:
        old_to_new = lambda idx: remap[onp.asarray(idx)]
        return meshio_mesh.points[used_mask], remap[raw_cells], old_to_new
    return meshio_mesh.points[used_mask], remap[raw_cells]


def _prepare_field(arr, expected_len, name):
    arr = onp.asarray(arr, dtype=onp.float32)
    if arr.shape[0] != expected_len:
        raise ValueError(f"Field '{name}' wrong shape: got {arr.shape}, expected first dim = {expected_len}")
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def quad_to_cell_average(fe, x):
    """Cell-wise volume average of a field x defined at quadrature points.

    Parameters
    ----------
    fe : FiniteElement
        Finite element object with ``JxW`` of shape ``(num_cells, num_quads)``.
    x : array, shape (num_cells, num_quads, ...)
        Field values at quadrature points.

    Returns
    -------
    array, shape (num_cells, ...)
    """
    x_sum = jnp.sum(x * fe.JxW[:, :, None, None], axis=1)
    vol = jnp.sum(fe.JxW, axis=1)
    return x_sum / vol[:, None, None]


def save_vtu(fe, sol_file, cell_infos=None, point_infos=None):
    """Write FE mesh and optional field data to a VTU file.

    Parameters
    ----------
    fe : FiniteElement
        Finite element object with attributes ``ele_type``, ``points``, ``cells``, ``num_cells``.
    sol_file : str
        Output file path (will create parent directories as needed).
    cell_infos : list of (str, array), optional
        Cell-centred fields as ``[(name, data), ...]``.  Each array's first
        dimension must equal ``fe.num_cells``; higher dimensions are flattened.
    point_infos : list of (str, array), optional
        Node-centred fields as ``[(name, data), ...]``.  Each array's first
        dimension must equal the number of nodes.
    """
    cell_type = get_meshio_cell_type(fe.ele_type)
    sol_dir = os.path.dirname(sol_file)
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
    out_mesh = meshio.Mesh(
        points=onp.asarray(fe.points, dtype=onp.float32),
        cells={cell_type: onp.asarray(fe.cells)},
    )
    if cell_infos is not None:
        for name, data in cell_infos:
            out_mesh.cell_data[name] = [_prepare_field(data, fe.num_cells, name)]
    if point_infos is not None:
        for name, data in point_infos:
            out_mesh.point_data[name] = _prepare_field(data, len(fe.points), name)
    out_mesh.write(sol_file)


class timer:
    """Context manager that measures wall time of a code block.

    The elapsed seconds are available as ``t.elapsed`` after the block exits.
    The object also supports ``float()`` and format specs so it can be used
    directly in f-strings::

        with timer() as t:
            do_work()
        print(f"took {t:.3f} s")
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __float__(self):
        return self.elapsed

    def __format__(self, fmt):
        return format(self.elapsed, fmt)


class oom_hint:
    """Context manager that logs what to try if the wrapped JAX computation runs out of GPU memory.

    The error is re-raised. ``extra`` is appended to the message, e.g. a problem's measured
    memory need::

        with oom_hint("Rough need at 10 time steps: ~0.17 GB per 1k cells."):
            dF, (F, aux) = evaluate_performance_grad(theta, ft)
    """

    def __init__(self, extra=''):
        self.extra = extra

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _):
        if exc_type is not None and issubclass(exc_type, jax.errors.JaxRuntimeError) and 'RESOURCE_EXHAUSTED' in str(exc):
            logger.error(
                "Out of GPU memory. Options: a GPU with more memory; "
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 (JAX uses at most 75% of the GPU by default); "
                "a coarser mesh; fewer time steps (time-dependent problems); "
                "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=0 (slower compile; can lower memory for single-solve "
                "problems, raises it for time stepping). " + self.extra)
        return False


def timeit(func):
    """ Decorator for printing the timing results of a function.

    Parameters
    ----------
    func : callable
        Function to be timed.

    Returns
    -------
    callable
        Wrapped function with timing logic.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def walltime(txt_dir=None, filename=None):
    """Wrapper for writing timing results to a file

    Parameters
    ----------
    txt_dir : str
        Directory to save timing data.
    filename : str
        Base filename (default: 'walltime_{platform}.txt').

    Returns
    -------
    callable
        Decorator function.
    """

    def decorate(func):

        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            logger.info(
                f"Time elapsed {time_elapsed} of function {func.__name__} "
                f"on platform {platform}"
            )
            if txt_dir is not None:
                os.makedirs(txt_dir, exist_ok=True)
                fname = 'walltime'
                if filename is not None:
                    fname = filename
                with open(os.path.join(txt_dir, f"{fname}_{platform}.txt"),
                          'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values

        return wrapper

    return decorate
