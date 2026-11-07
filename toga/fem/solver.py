"""
Copied and modified from https://github.com/deepmodeling/jax-fem
Under GNU General Public License v3.0
"""
import time

import jax
import jax.experimental.sparse as jsparse
import jax.flatten_util
import jax.numpy as np
import numpy as onp
from jax import config
from jax import core as jax_core
from jax.experimental.sparse import BCOO

from toga import logger
from toga.fem.sparse_linear_solvers import bcoo_custom_linear_solve, bcoo_spineax_solve
from toga.utils import shared_device_array

config.update("jax_enable_x64", True)


def _is_tracer(x):
    return isinstance(x, jax_core.Tracer)


def _log_scalar(logger_fn, prefix, value):
    if _is_tracer(value):
        logger_fn(f"{prefix}: <traced {value.aval}>")
    else:
        logger_fn(f"{prefix}: {value}")


def _log_solver_state(logger_fn, res_val, rel_res_val):
    if _is_tracer(res_val) or _is_tracer(rel_res_val):
        logger_fn(
            "Newton residuals: "
            f"res=<traced {getattr(res_val, 'aval', type(res_val))}>, "
            f"rel=<traced {getattr(rel_res_val, 'aval', type(rel_res_val))}>"
        )
    else:
        logger_fn(f"Newton residuals: res={res_val}, rel={rel_res_val}")

def _get_problem_p_mat_bcoo(problem):
    if not hasattr(problem, 'P_mat'):
        return None
    if not hasattr(problem, 'P_mat_BCOO'):
        problem.P_mat_BCOO = BCOO.from_scipy_sparse(problem.P_mat).sort_indices()
    return problem.P_mat_BCOO


def _get_bc_rows(problem):
    if hasattr(problem, '_bc_rows'):
        return problem._bc_rows

    bc_rows = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.asarray(
                fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
                dtype=onp.int32,
            ).reshape(-1)
            if row_inds.size:
                bc_rows.append(row_inds)

    if bc_rows:
        problem._bc_rows = onp.unique(onp.concatenate(bc_rows)).astype(onp.int32, copy=False)
    else:
        problem._bc_rows = onp.empty((0,), dtype=onp.int32)

    return problem._bc_rows


def _get_bc_elimination_info(problem):
    if hasattr(problem, '_bc_elimination_info'):
        return problem._bc_elimination_info

    row_inds = _get_bc_rows(problem)
    indices = onp.stack((problem.I, problem.J), axis=1)
    if row_inds.size == 0:
        row_keep_mask = onp.ones((indices.shape[0],), dtype=bool)
        row_col_keep_mask = onp.ones((indices.shape[0],), dtype=bool)
        kept_indices = indices
        diag_indices = onp.empty((0, 2), dtype=indices.dtype)
    else:
        row_keep_mask = ~onp.isin(indices[:, 0], row_inds)
        row_col_keep_mask = row_keep_mask & ~onp.isin(indices[:, 1], row_inds)
        kept_indices = indices[row_col_keep_mask]
        diag_indices = onp.stack((row_inds, row_inds), axis=1)

    final_indices = onp.concatenate((kept_indices, diag_indices), axis=0).astype(onp.int32, copy=False)
    diag_mask = final_indices[:, 0] == final_indices[:, 1]
    diag_rows = final_indices[diag_mask, 0].astype(onp.int32, copy=False)

    # Masks are stored as int32 gather indices on device: indexing with a NumPy bool mask
    # would build a new int64 index array at every trace.
    problem._bc_elimination_info = {
        'row_keep_idx': shared_device_array(onp.flatnonzero(row_keep_mask), onp.int32),
        'row_col_keep_idx': shared_device_array(onp.flatnonzero(row_col_keep_mask), onp.int32),
        'added_diag_count': diag_indices.shape[0],
        'indices': final_indices,
        'indices_device': shared_device_array(final_indices),
        'diag_idx': shared_device_array(onp.flatnonzero(diag_mask), onp.int32),
        'diag_rows': shared_device_array(diag_rows),
    }
    return problem._bc_elimination_info


def _get_A_indices_host(problem):
    """NumPy copy of the (fixed) sparsity indices of get_A's matrix, or None with P_mat."""
    if hasattr(problem, 'P_mat'):
        return None
    return _get_bc_elimination_info(problem)['indices']


def _bcoo_diagonal(A, problem):
    bc_elimination_info = _get_bc_elimination_info(problem)
    diag_rows = bc_elimination_info['diag_rows']
    diag_data = A.data[bc_elimination_info['diag_idx']]
    return jax.ops.segment_sum(diag_data, diag_rows, num_segments=A.shape[0])


def jax_solve(problem, A, b, x0, precond):
    """Solves the equilibrium equation using a JAX solver.

    Parameters
    ----------
    precond
        Whether to calculate the preconditioner or not
    """
    logger.debug("JAX Solver - Solving linear system")
    if not isinstance(A, jsparse.BCOO):
        raise TypeError("jax_solve expects A as a JAX BCOO matrix.")

    jacobi = _bcoo_diagonal(A, problem)
    pc = lambda x: x * (1. / np.where(jacobi == 0, 1., jacobi)) if precond else None

    if np.issubdtype(b.dtype, np.complexfloating):
        A = A.astype(complex)
        b = b.astype(complex)
        if x0 is not None:
            x0 = x0.astype(complex)

    x, info = jax.scipy.sparse.linalg.bicgstab(A,
                                               b,
                                               x0=x0,
                                               M=pc,
                                               tol=1e-10,
                                               atol=1e-10,
                                               maxiter=10000)

    # Verify convergence
    err = np.linalg.norm(A @ x - b)
    _log_scalar(logger.debug, "JAX Solver - Finished solving, res", err)
    x = np.where(err < 0.1, x, np.nan)

    return x


def umfpack_solve(A, b, solver_options, indices_host=None):
    """scipy.sparse.linalg.spsolve: UMFPACK if scikit-umfpack is installed, otherwise SuperLU."""
    logger.debug("Scipy Solver - Solving linear system with spsolve")
    x = bcoo_custom_linear_solve(A, b, solver="scipy", indices_host=indices_host, **solver_options)
    logger.debug(f"Scipy Solver - Finished solving, linear solve res = {np.linalg.norm(A @ x - b)}")
    return x


def pardiso_solve(A, b, solver_options, indices_host=None):
    logger.debug("PARDISO Solver - Solving linear system")
    x = bcoo_custom_linear_solve(A, b, solver="pardiso", indices_host=indices_host, **solver_options)
    logger.debug(f"PARDISO Solver - Finished solving, linear solve res = {np.linalg.norm(A @ x - b)}")
    return x


def spineax_solve(A, b, solver_options, indices_host):
    logger.debug("spineax (cuDSS) Solver - Solving linear system on the GPU")
    if indices_host is None:
        raise NotImplementedError("spineax_solver needs a fixed sparsity pattern (not supported with P_mat).")
    return bcoo_spineax_solve(A, b, indices_host, **solver_options)


def linear_solver(problem, A, b, x0, solver_options, indices_host=None):
    # If user does not specify any solver, set jax_solver as the default one.
    if  len(solver_options.keys() & {'jax_solver', 'umfpack_solver', 'pardiso_solver', 'spineax_solver'}) == 0: 
        solver_options['jax_solver'] = {}

    if 'jax_solver' in solver_options:      
        precond = solver_options['jax_solver']['precond'] if 'precond' in solver_options['jax_solver'] else True
        x = jax_solve(problem, A, b, x0, precond)
    elif 'umfpack_solver' in solver_options:
        x = umfpack_solve(A, b, solver_options['umfpack_solver'], indices_host)
    elif 'pardiso_solver' in solver_options:
        x = pardiso_solve(A, b, solver_options['pardiso_solver'], indices_host)
    elif 'spineax_solver' in solver_options:
        x = spineax_solve(A, b, solver_options['spineax_solver'], indices_host)
    else:
        raise NotImplementedError("Unknown linear solver.")
    return x


################################################################################
# Dirichlet B.C. helpers

def apply_bc_vec(res_vec, dofs, problem, scale=1.):
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = (res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True))
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(-fe.vals_list[i]*scale)

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn, problem, scale=1.):
    def res_fn_bc(dofs):
        """Apply Dirichlet boundary conditions
        """
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)
    return res_fn_bc


def assign_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs, problem):
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)
  
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = (new_sol.at[fe.node_inds_list[i],
                                  fe.vec_inds_list[i]].set(sol[fe.node_inds_list[i],
                                          fe.vec_inds_list[i]]))
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def get_flatten_fn(fn_sol_list, problem):

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def linear_incremental_solver(problem, res_vec, A, dofs, solver_options):
    """
    Linear solver at each Newton's iteration
    """
    logger.debug("Solving linear system...")
    # x0 will always be correct at boundary locations
    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    if hasattr(problem, 'P_mat'):
        P_bcoo = _get_problem_p_mat_bcoo(problem)
        x0_2 = copy_bc(P_bcoo @ dofs, problem)
        x0 = P_bcoo.T @ (x0_1 - x0_2)
    else:
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2

    raw_A = solver_options.get('raw_A')
    eliminate_columns = solver_options.get('column_elimination', True)
    b = -res_vec
    if eliminate_columns and raw_A is not None:
        b = b - raw_A @ x0
        bc_rows = _get_bc_rows(problem)
        if bc_rows.size:
            b = b.at[bc_rows].set(x0[bc_rows])

    inc = linear_solver(problem, A, b, x0, solver_options, _get_A_indices_host(problem))

    line_search_flag = solver_options['line_search_flag'] if 'line_search_flag' in solver_options else False
    if line_search_flag:
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs


def line_search(problem, dofs, inc):
    """Backtracking line search: halve the Newton step up to three times while the residual decreases."""
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha*inc)
        return np.linalg.norm(res_vec)

    alpha = 1.
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        logger.debug(f"Line search i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.
            break
        res_norm = res_norm_half

    return dofs + alpha*inc


def get_A(problem, V, *, eliminate_columns=True):
    mode = "row+column elimination" if eliminate_columns else "row elimination"
    logger.debug(f"Creating sparse matrix with JAX BCOO using {mode}...")
    ndofs = problem.num_total_dofs_all_vars
    data = np.asarray(V)
    bc_elimination_info = _get_bc_elimination_info(problem)
    keep_idx = bc_elimination_info['row_col_keep_idx'] if eliminate_columns else bc_elimination_info['row_keep_idx']
    kept_data = data[keep_idx]
    diag_data = np.ones((bc_elimination_info['added_diag_count'],), dtype=data.dtype)
    indices = bc_elimination_info['indices_device']
    data = np.concatenate((kept_data, diag_data), axis=0)
    A = BCOO((data, indices), shape=(ndofs, ndofs))

    # Linear multipoint constraints
    if hasattr(problem, 'P_mat'):
        P_bcoo = _get_problem_p_mat_bcoo(problem)
        A = P_bcoo.T @ (A @ P_bcoo)

    return A


def get_A_raw(problem, V):
    ndofs = problem.num_total_dofs_all_vars
    if not hasattr(problem, '_IJ_device'):
        problem._IJ_device = shared_device_array(onp.stack((problem.I, problem.J), axis=1), onp.int32)
    A = BCOO((np.asarray(V), problem._IJ_device), shape=(ndofs, ndofs))

    if hasattr(problem, 'P_mat'):
        P_bcoo = _get_problem_p_mat_bcoo(problem)
        A = P_bcoo.T @ (A @ P_bcoo)

    return A


################################################################################
# The "row elimination" solver

def solver(problem, solver_options={}):
    r"""Solve the nonlinear problem using Newton's method with configurable linear solvers.

    The solver imposes Dirichlet B.C. with "row elimination" method. Conceptually,

    .. math::
        r(u) = D \, r_{\text{unc}}(u) + (I - D)u - u_b \\
        A = \frac{\text{d}r}{\text{d}u} = D \frac{\text{d}r}{\text{d}u} + (I - D)

    where:

    - :math:`r_{\text{unc}}: \mathbb{R}^N\rightarrow\mathbb{R}^N` is the residual function without considering Dirichlet boundary conditions.

    - :math:`u\in\mathbb{R}^N` is the FE solution vector.

    - :math:`u_b\in\mathbb{R}^N` is the vector for Dirichlet boundary conditions, e.g.,

      .. math::
            u_b = \begin{bmatrix}
                  0 \\
                  0 \\
                  2 \\
                  3
                  \end{bmatrix}
    
    - :math:`D\in\mathbb{R}^{N\times N}` is the auxiliary matrix for masking, e.g.,

      .. math::
            D = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0
            \end{bmatrix}
    
    - :math:`I\in\mathbb{R}^{N\times N}` is the identity matrix, e.g., 

      .. math::
            I = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}

    - :math:`A\in\mathbb{R}^{N\times N}` is the tangent stiffness matrix (the global Jacobian matrix).

    Parameters
    ----------
    problem : Problem
        The nonlinear problem to solve
    solver_options : dict
        Configuration dictionary for solver parameters and algorithms.
        Four linear solvers are available:

        - `JAX solver <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.bicgstab.html>`_
        - `UMFPACK solver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html>`_
        - `PARDISO solver <https://pypi.org/project/pypardiso/>`_
        - cuDSS on the GPU via `spineax`
    
        The empty choice ::

            solver_options = {}

        will default to JAX solver as ::
        
            solver_options = {'jax_solver': {}}

        which will further default to ::

            solver_options = {'jax_solver': {'precond': True}}

        The UMFPACK solver can be specified as ::

            solver_options = {'umfpack_solver': {}}

        The PARDISO solver can be specified as ::

            solver_options = {'pardiso_solver': {}}

        The GPU cuDSS solver (requires ``pip install spineax``) can be specified as ::

            solver_options = {'spineax_solver': {}}

        Other available options are :: 

            solver_options = {
                'line_search_flag': False, # Line search method
                'initial_guess': initial_guess, # Same shape as sol_list
                'tol': 1e-6, # Absolute tolerance for residual vector (l2 norm), used in Newton's method
                'rel_tol': 1e-8, # Relative tolerance for residual vector (l2 norm), used in Newton's method
                'max_iters': 50, # Maximum number of Newton iterations
                'column_elimination': True, # Also eliminate Dirichlet columns (keeps A symmetric)
            }

    Returns
    -------
    sol_list : list

    """
    logger.debug("Calling the row elimination solver for imposing Dirichlet B.C.")
    logger.debug("Start timing")
    start = time.time()

    if 'initial_guess' in solver_options:
        # Keep the initial guess out of the differentiation chain.
        initial_guess = jax.lax.stop_gradient(solver_options['initial_guess'])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        if hasattr(problem, 'P_mat'):
            dofs = np.zeros(problem.P_mat.shape[1]) # reduced dofs
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = solver_options['rel_tol'] if 'rel_tol' in solver_options else 1e-8
    tol = solver_options['tol'] if 'tol' in solver_options else 1e-6
    max_iters = solver_options['max_iters'] if 'max_iters' in solver_options else 50

    eliminate_columns = solver_options['column_elimination'] if 'column_elimination' in solver_options else True

    def newton_update_helper(dofs):
        if hasattr(problem, 'P_mat'):
            dofs = _get_problem_p_mat_bcoo(problem) @ dofs

        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list, V = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        if hasattr(problem, 'P_mat'):
            res_vec = _get_problem_p_mat_bcoo(problem).T @ res_vec

        raw_A = get_A_raw(problem, V) if eliminate_columns else None
        A = get_A(problem, V, eliminate_columns=eliminate_columns)
        return res_vec, A, raw_A

    res_vec, A, raw_A = newton_update_helper(dofs)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = np.where(res_val > 0.0, res_val, 1.0)
    rel_res_val = res_val / res_val_initial
    _log_solver_state(logger.debug, res_val, rel_res_val)

    def loop_cond(state):
        iter_num, _, _, _, _, res_val, rel_res_val = state
        return (iter_num < max_iters) & (rel_res_val > rel_tol) & (res_val > tol)

    def loop_body(state):
        iter_num, dofs, res_vec, A, raw_A, _, _ = state
        loop_solver_options = dict(solver_options)
        loop_solver_options['raw_A'] = raw_A
        loop_solver_options['column_elimination'] = eliminate_columns
        dofs = linear_incremental_solver(problem, res_vec, A, dofs, loop_solver_options)
        res_vec, A, raw_A = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val / res_val_initial
        _log_solver_state(logger.debug, res_val, rel_res_val)
        return iter_num + 1, dofs, res_vec, A, raw_A, res_val, rel_res_val

    _, dofs, res_vec, A, raw_A, res_val, rel_res_val = jax.lax.while_loop(
        loop_cond,
        loop_body,
        (0, dofs, res_vec, A, raw_A, res_val, rel_res_val),
    )

    dofs = np.where(np.all(np.isfinite(dofs)), dofs, np.nan)

    if hasattr(problem, 'P_mat'):
        dofs = _get_problem_p_mat_bcoo(problem) @ dofs

    # If sol_list = [[[u1x, u1y], 
    #                 [u2x, u2y], 
    #                 [u3x, u3y], 
    #                 [u4x, u4y]], 
    #                [[p1], 
    #                 [p2]]],
    # the flattend DOF vector will be [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y, p1, p2]
    sol_list = problem.unflatten_fn_sol_list(dofs)

    end = time.time()
    solve_time = end - start
    logger.info(f"Solve took {solve_time} [s]")
    _log_scalar(logger.info, "max of dofs", np.max(dofs))
    _log_scalar(logger.info, "min of dofs", np.min(dofs))

    return sol_list


################################################################################
# Implicit differentiation

def implicit_jvp(problem, sol_list, params, params_dot, linear_solver_options):
    """Tangent of the solution by the implicit function theorem.

    At the solution c(u, p) = 0, so du = -A^{-1} (dc/dp dp) with A = dc/du. The
    linear solve is wrapped in ``jax.lax.custom_linear_solve``, so JAX can
    transpose this rule (reverse mode: one adjoint solve with A^T) and
    differentiate it again (higher-order derivatives).
    """
    dofs = jax.flatten_util.ravel_pytree(sol_list)[0]

    def constraint_fn(params):
        """c(u=u, p)
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    _, c_dot = jax.jvp(constraint_fn, (params,), (params_dot,))

    problem.set_params(params)
    _, V = problem.newton_update(sol_list)
    A = get_A(problem, V)

    indices_host = _get_A_indices_host(problem)
    indices_host_T = None if indices_host is None else indices_host[:, ::-1]
    solve = lambda _, b: linear_solver(problem, A, b, None, linear_solver_options, indices_host)
    transpose_solve = lambda _, b: linear_solver(problem, A.transpose(), b, None, linear_solver_options, indices_host_T)

    if hasattr(problem, 'P_mat'):
        c_dot = _get_problem_p_mat_bcoo(problem).T @ c_dot

    sol_dot = jax.lax.custom_linear_solve(lambda x: A @ x, -c_dot, solve, transpose_solve)

    if hasattr(problem, 'P_mat'):
        sol_dot = _get_problem_p_mat_bcoo(problem) @ sol_dot

    return problem.unflatten_fn_sol_list(sol_dot)


def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    """Automatic differentiation wrapper for the forward problem.

    Derivatives use a custom JVP rule (:func:`implicit_jvp`), so forward mode,
    reverse mode (JAX transposes the JVP) and higher-order derivatives all work.

    Parameters
    ----------
    problem : Problem
    solver_options : dictionary
    adjoint_solver_options : dictionary
        Linear solver options for the tangent / adjoint solves.

    Returns
    -------
    fwd_pred : callable
    """
    @jax.custom_jvp
    def fwd_pred(params):
        problem.set_params(params)
        sol_list = solver(problem, solver_options)
        return sol_list

    @fwd_pred.defjvp
    def fwd_pred_jvp(primals, tangents):
        params, = primals
        params_dot, = tangents
        sol_list = fwd_pred(params)
        sol_dot = implicit_jvp(problem, sol_list, params, params_dot, adjoint_solver_options)
        return sol_list, sol_dot

    return fwd_pred
