"""Differentiable linear solves with configurable host backends."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SolverFn = Callable[..., np.ndarray]


def _coo_to_csr(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
    *,
    backend: str,
):
    if backend == "scipy":
        return sp.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    if backend == "cupy":
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx_sparse
        except ImportError as exc:
            raise ImportError(
                "csr_backend='cupy' requires cupy and cupyx.scipy.sparse."
            ) from exc
        rows_cp = cp.asarray(rows)
        cols_cp = cp.asarray(cols)
        data_cp = cp.asarray(data)
        return cpx_sparse.coo_matrix((data_cp, (rows_cp, cols_cp)), shape=shape).tocsr()
    raise ValueError("csr_backend must be 'scipy' or 'cupy'.")


def _coo_host_to_csr(indices_host, data_host, shape, backend):
    indices_arr = np.asarray(indices_host)
    if indices_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D sparse indices, got shape {indices_arr.shape}."
        )
    if indices_arr.shape[1] == 2:
        indices_2d = indices_arr
    elif indices_arr.shape[0] == 2:
        indices_2d = indices_arr.T
    else:
        raise ValueError(
            f"Could not interpret sparse indices with shape {indices_arr.shape}."
        )

    rows = indices_2d[:, 0]
    cols = indices_2d[:, 1]
    data_arr = np.asarray(data_host)
    valid = (
        (rows >= 0)
        & (rows < shape[0])
        & (cols >= 0)
        & (cols < shape[1])
    )
    return _coo_to_csr(rows[valid], cols[valid], data_arr[valid], shape, backend=backend)


_CSR_PATTERNS: dict = {}


def _csr_pattern(indices: np.ndarray, shape: tuple[int, int]):
    """CSR structure of fixed COO ``indices`` plus the CSR slot of every COO entry.

    Cached by content, so the sort runs once per sparsity pattern rather than on
    every solve. Out-of-range (padding) entries go to a dump slot at the end.
    """
    key = (shape, hashlib.sha1(np.ascontiguousarray(indices)).hexdigest())
    if key not in _CSR_PATTERNS:
        rows = indices[:, 0].astype(np.int64)
        cols = indices[:, 1].astype(np.int64)
        valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        keys, slot_valid = np.unique(rows[valid] * shape[1] + cols[valid], return_inverse=True)
        slot = np.full(len(rows), len(keys), dtype=np.int64)
        slot[valid] = slot_valid
        indptr = np.concatenate(([0], np.cumsum(np.bincount(keys // shape[1], minlength=shape[0]))))
        _CSR_PATTERNS[key] = (slot, (keys % shape[1]).astype(np.int32), indptr.astype(np.int32))
    return _CSR_PATTERNS[key]


def _to_scipy_csr(A_csr):
    if isinstance(A_csr, sp.csr_matrix):
        return A_csr
    if hasattr(A_csr, "get"):
        return sp.csr_matrix(A_csr.get())
    raise TypeError(
        "Built-in scipy/pardiso solvers require scipy CSR input. "
        "Pass solver callable compatible with selected csr_backend."
    )


def _solve_with_scipy(
    A_csr, rhs: np.ndarray, *, transpose: bool, **solver_kwargs: Any
) -> np.ndarray:
    A_csr = _to_scipy_csr(A_csr)
    A_use = A_csr.T if transpose else A_csr
    return np.asarray(spla.spsolve(A_use, rhs, **solver_kwargs))


def _solve_with_pardiso(
    A_csr, rhs: np.ndarray, *, transpose: bool, **solver_kwargs: Any
) -> np.ndarray:
    import pypardiso

    A_csr = _to_scipy_csr(A_csr)
    A_use = A_csr.T if transpose else A_csr
    return np.asarray(pypardiso.spsolve(A_use, rhs, **solver_kwargs))


class CachedPardiso:
    """Pardiso with the symbolic analysis cached for a fixed sparsity pattern.

    pypardiso.spsolve reruns the symbolic analysis (fill-reducing ordering) for every new
    matrix. Here it runs once per sparsity pattern, and a factorization is reused while the
    matrix values are unchanged. mtype 11 (general, as pypardiso) works for any A; mtype -2
    (symmetric indefinite) factorizes only the upper triangle and is valid only for A = A^T.
    """

    def __init__(self, mtype: int = 11):
        import pypardiso

        self.ps = pypardiso.PyPardisoSolver(mtype=mtype)
        self.symmetric = mtype != 11
        self.indptr = self.indices = self.data = None

    def _run(self, phase: int, b: np.ndarray) -> np.ndarray:
        self.ps.set_phase(phase)
        return self.ps._call_pardiso(self.U, b)

    def __call__(self, A_csr, rhs: np.ndarray, *, transpose: bool, **_: Any) -> np.ndarray:
        A = _to_scipy_csr(A_csr)
        if transpose and not self.symmetric:
            A = A.T.tocsr()
        n = A.shape[0]
        if self.indptr is None or not (
            np.array_equal(A.indptr, self.indptr) and np.array_equal(A.indices, self.indices)
        ):  # new sparsity pattern: build the matrix Pardiso sees and run the symbolic analysis
            A.sort_indices()
            rows = np.repeat(np.arange(n), np.diff(A.indptr))
            self.keep = np.flatnonzero(A.indices >= rows) if self.symmetric else np.arange(A.nnz)
            indptr = np.searchsorted(rows[self.keep], np.arange(n + 1))
            self.U = sp.csr_matrix((A.data[self.keep], A.indices[self.keep], indptr), shape=A.shape)
            self.indptr, self.indices, self.data = A.indptr.copy(), A.indices.copy(), None
            self._run(11, np.zeros((n, 1)))
        if self.data is None or not np.array_equal(A.data, self.data):
            self.U.data[:] = A.data[self.keep]
            self._run(22, np.zeros((n, 1)))
            self.data = A.data.copy()
        b = np.asfortranarray(np.asarray(rhs, dtype=np.float64).reshape(n, -1))
        return np.asarray(self._run(33, b)).reshape(np.shape(rhs))


SOLVER_REGISTRY: dict[str, SolverFn] = {
    "scipy": _solve_with_scipy,
    # TOGA_PARDISO=cached: CachedPardiso (general); =sym: symmetric, only for A = A^T
    "pardiso": {"cached": lambda: CachedPardiso(11), "sym": lambda: CachedPardiso(-2)}.get(
        os.environ.get("TOGA_PARDISO"), lambda: _solve_with_pardiso)(),
}


def bcoo_custom_linear_solve(
    A_bcoo: jsparse.BCOO,
    b: jax.Array,
    solver: str | SolverFn = "pardiso",
    *,
    symmetric: bool = False,
    callback_vmap_method: str = "expand_dims", # 'sequential', 'expand_dims' or 'broadcast_all'
    transpose_callback_vmap_method: str = "expand_dims", # 'sequential', 'expand_dims' or 'broadcast_all'
    csr_backend: str = "scipy", # 'scipy' or 'cupy'
    indices_host: np.ndarray | None = None,
    **solver_kwargs: Any,
) -> jax.Array:
    """Solve ``A x = b`` for sparse ``A`` and make it differentiable in JAX.

    Args:
        A_bcoo: Sparse matrix in JAX BCOO format.
        b: Dense right-hand side array.
        solver: Either backend name in ``SOLVER_REGISTRY`` or callable with
            signature ``solver(A_csr, rhs, transpose: bool, **kwargs)``.
        symmetric: Passed to ``jax.lax.custom_linear_solve``.
        callback_vmap_method: ``jax.pure_callback`` batching behavior used
            under ``vmap`` transforms for the forward solve callback.
        transpose_callback_vmap_method: ``jax.pure_callback`` batching behavior
            used under ``vmap`` transforms for the transpose (backward) solve
            callback. ``"expand_dims"`` batches right-hand sides into one
            callback without copying an unbatched matrix per batch element.
        csr_backend: Backend used for COO->CSR conversion inside callback.
            Supported: ``"scipy"``, ``"cupy"``.
        indices_host: Optional NumPy copy of ``A_bcoo.indices`` for a fixed
            sparsity pattern. The CSR pattern is then built once and only the
            matrix values are sent to the host on each solve.
        **solver_kwargs: Extra kwargs forwarded to the host solver.
    """
    if not isinstance(A_bcoo, jsparse.BCOO):
        raise TypeError("A_bcoo must be jax.experimental.sparse.BCOO.")
    if A_bcoo.ndim != 2:
        raise ValueError(f"A_bcoo must be rank-2. Got ndim={A_bcoo.ndim}.")
    if csr_backend not in {"scipy", "cupy"}:
        raise ValueError("csr_backend must be 'scipy' or 'cupy'.")

    if isinstance(solver, str):
        try:
            solver_fn = SOLVER_REGISTRY[solver]
        except KeyError as exc:
            raise ValueError(
                f"Unknown solver {solver!r}. Use one of {list(SOLVER_REGISTRY)} "
                "or pass a callable."
            ) from exc
    else:
        solver_fn = solver

    shape = A_bcoo.shape
    indices = A_bcoo.indices
    data = A_bcoo.data
    nrows = shape[0]
    scalar_rhs_input = jnp.ndim(b) == 0
    b_vec = jnp.reshape(b, (1,)) if scalar_rhs_input else b

    fixed_pattern = indices_host is not None and csr_backend == "scipy"
    if fixed_pattern:
        slot, csr_indices, indptr = _csr_pattern(np.asarray(indices_host), shape)

    def _solve_callback(rhs: jax.Array, *, transpose: bool) -> jax.Array:
        out_spec = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)

        def _host_solve(
            indices_host: np.ndarray | None, data_host: np.ndarray, rhs_host: np.ndarray
        ) -> np.ndarray:
            # vmap (possibly nested) adds leading batch dims to the operands.
            data_host = data_host.reshape(-1, data_host.shape[-1])
            if data_host.shape[0] > 1 and not (data_host == data_host[:1]).all():
                raise NotImplementedError(
                    "vmap over the sparse matrix is not supported, only over right-hand sides."
                )
            data_host = data_host[0]
            if fixed_pattern:
                csr_data = np.bincount(slot, weights=data_host, minlength=len(csr_indices) + 1)[:-1]
                A_csr = sp.csr_matrix((csr_data, csr_indices, indptr), shape=shape)
            else:
                indices_host = indices_host.reshape(-1, *indices_host.shape[-2:])[0]
                A_csr = _coo_host_to_csr(indices_host, data_host, shape, csr_backend)

            # Sparse solvers expect RHS columns, i.e. shape (nrows, nrhs).
            # All-zero columns (e.g. jacrev cotangents of outputs that don't
            # depend on this solve) have zero solutions and are skipped.
            rhs_arr = np.asarray(rhs_host)
            rhs_cols = rhs_arr.reshape(-1, nrows).T
            x_cols = np.zeros(rhs_cols.shape, dtype=rhs_arr.dtype)
            nonzero = np.any(rhs_cols != 0, axis=0)
            if nonzero.any():
                x_host = solver_fn(A_csr, rhs_cols[:, nonzero], transpose=transpose, **solver_kwargs)
                x_cols[:, nonzero] = np.asarray(x_host).reshape(nrows, -1)
            return x_cols.T.reshape(rhs_arr.shape)

        return jax.pure_callback(
            _host_solve,
            out_spec,
            None if fixed_pattern else indices,
            data,
            rhs,
            vmap_method=(
                transpose_callback_vmap_method
                if transpose
                else callback_vmap_method
            ),
        )

    def matvec(x: jax.Array) -> jax.Array:
        return A_bcoo @ x

    x = jax.lax.custom_linear_solve(
        matvec,
        b_vec,
        solve=lambda _, rhs: _solve_callback(rhs, transpose=False),
        transpose_solve=lambda _, rhs: _solve_callback(rhs, transpose=True),
        symmetric=symmetric,
    )
    return x[0] if scalar_rhs_input else x


def bcoo_spineax_solve(
    A_bcoo: jsparse.BCOO,
    b: jax.Array,
    indices_host: np.ndarray,
    **analyze_kwargs: Any,
) -> jax.Array:
    """Solve ``A x = b`` on the GPU with cuDSS via spineax (differentiable).

    Needs the fixed sparsity pattern ``indices_host`` (NumPy copy of
    ``A_bcoo.indices``): duplicates are summed into CSR order on the device
    and the matrix never leaves the GPU. ``analyze_kwargs`` go to
    ``spineax.cudss.analyze`` (default: general LU, full pattern).
    """
    from spineax import cudss

    slot, csr_columns, csr_offsets = _csr_pattern(np.asarray(indices_host), A_bcoo.shape)
    nnz = len(csr_columns)
    values = jax.ops.segment_sum(A_bcoo.data, slot.astype(np.int32), num_segments=nnz + 1)[:nnz]
    analyze_kwargs = {"mtype_id": "general", "mview_id": "full", **analyze_kwargs}
    token = cudss.analyze(values, csr_offsets, csr_columns, **analyze_kwargs)
    token = cudss.factorize(token, values)
    return cudss.solve(token, b)


__all__ = ["SOLVER_REGISTRY", "bcoo_custom_linear_solve", "bcoo_spineax_solve"]
