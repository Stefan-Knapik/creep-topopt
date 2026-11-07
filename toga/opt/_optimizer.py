"""
Base optimizer for TOGA.
"""
import numpy as np


class Optimizer:
    """Base class holding design-variable state and problem metadata.

    Parameters
    ----------
    desvars_init : array_like, shape (N,)
        Initial design variables.
    bounds : tuple of (float, float)
        Scalar lower and upper bounds on the design variables.
    is_independent : bool
        True if constraints are independent (one design variable affects only
        one constraint). Required True for OC.
    constraint_map : array_like, optional
        ``(m, m)`` mapping used by OC/PGD when ``m > 1``. Defaults to identity
        (built lazily once the number of constraints is known).
    is_terminal : bool
        True if no outer continuation loop (e.g., penalization) is active.
    max_iter : int, optional
        Hard iteration cap. :meth:`converged` returns True once
        ``self.iteration >= max_iter``. Default ``None`` (no cap).

    Notes
    -----
    The number of constraints ``m`` is inferred from the shape of ``vc``
    passed to :meth:`iter` on the first call; it does not need to be declared
    up front.
    """

    def __init__(self,
                 desvars_init,
                 bounds=(0.0, 1.0),
                 is_independent=True,
                 constraint_map=None,
                 is_terminal=True,
                 max_iter=None):
        self.desvars = np.asarray(desvars_init, dtype=float).reshape(-1).copy()
        self.N = self.desvars.size
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self._is_independent = bool(is_independent)
        self._is_terminal = bool(is_terminal)
        self._user_constraint_map = (None if constraint_map is None
                                     else np.asarray(constraint_map, dtype=float))
        self.max_iter = None if max_iter is None else int(max_iter) + 1

        self.m = None
        self.lambda_map = None
        self._J = None
        self._dJ = None
        self._vc = None
        self._dvc = None
        self._x_at_eval = None

        self.iteration = 0
        self.change = np.inf
        self.change_f = np.inf
        self.change_tol = np.inf
        self.fun_tol = np.inf

    def _store_values(self, J, dJ, vc, dvc):
        """Save the objective/constraint values supplied to :meth:`iter`.

        Shapes are normalized so optimizer code can rely on canonical layouts:
        J scalar, dJ (N,), vc (m,), dvc (m, N). On the first call, ``m`` is
        inferred from the size of ``vc`` and ``lambda_map`` is built.
        """
        vc_flat = np.asarray(vc, dtype=float).reshape(-1)
        if self.m is None:
            self.m = vc_flat.size
            self.lambda_map = (np.eye(self.m) if self._user_constraint_map is None
                               else self._user_constraint_map)

        self._J = float(np.asarray(J).reshape(-1)[0])
        self._dJ = np.asarray(dJ, dtype=float).reshape(-1)
        self._vc = vc_flat
        self._dvc = np.asarray(dvc, dtype=float).reshape(self.m, self.N)
        self._x_at_eval = self.desvars.copy()

    def _g_at(self, x):
        """Linear approximation of the constraint value at trial point ``x``.

        Exact for the standard linear volume constraint; matches the way OC
        and PGD probe the feasible set in the pyFANTOM implementation.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        return self._vc + self._dvc @ (x - self._x_at_eval)

    def iter(self, J, dJ, vc, dvc):
        raise NotImplementedError("iter method must be implemented in subclasses.")

    def converged(self):
        """True once the iteration cap is reached or change tolerances are met."""
        if self.max_iter is not None and self.iteration >= self.max_iter:
            return True
        if self.iteration < 1:
            return False
        if not self._is_terminal:
            return False
        return self.change <= self.change_tol and self.change_f <= self.fun_tol

    def logs(self):
        raise NotImplementedError("logs method must be implemented in subclasses.")
