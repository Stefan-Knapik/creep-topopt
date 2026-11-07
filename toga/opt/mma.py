"""
Method of Moving Asymptotes (MMA) optimizer for TOGA.
"""

import time

import numpy as np

from ._mma_core import mmasub
from ._optimizer import Optimizer


class MMA(Optimizer):
    """Method of Moving Asymptotes optimizer.

    Usage::

        optimizer = MMA(desvars_init=theta0, bounds=(1e-3, 1.0), num_constraints=1)
        while not optimizer.converged():
            optimizer.iter(J, dJ, vc, dvc)
    """

    def __init__(self,
                 desvars_init,
                 bounds=(0.0, 1.0),
                 is_independent=True,
                 constraint_map=None,
                 is_terminal=True,
                 max_iter=None,
                 sub_tol=1e-7,
                 sub_maxiter=100,
                 change_tol=1e-4,
                 fun_tol=1e-6,
                 move=0.5,
                 timer=False):
        super().__init__(desvars_init, bounds, is_independent,
                         constraint_map, is_terminal, max_iter)

        self.change_tol = change_tol
        self.fun_tol = fun_tol
        self.move = move
        self.sub_tol = sub_tol
        self.sub_maxiter = sub_maxiter
        self.timer = timer

        self.last_desvars = self.desvars.copy()
        self.last_f = None
        self.f0_val = None

        self.x_1 = self.desvars.copy()
        self.x_2 = self.desvars.copy()
        self.low = np.ones([self.N, 1]) * self.bounds[0]
        self.upp = np.ones([self.N, 1]) * self.bounds[1]

    def iter(self, J, dJ, vc, dvc):
        if self.timer:
            start_time = time.time()

        self._store_values(J, dJ, vc, dvc)

        desvars = self.desvars
        dg = self._dvc
        df = self._dJ
        if self.iteration == 0:
            self.f0_val = self._J
            self.last_f = self._J

        f_val = self._vc.reshape(-1, 1)

        a = np.zeros([self.m, 1])
        c = np.ones([self.m, 1]) * 100000
        d = np.zeros([self.m, 1])

        desvars_new, _, _, _, _, _, _, _, _, self.low, self.upp = mmasub(
            self.m,
            self.N,
            self.iteration + 1,
            desvars.reshape(-1, 1),
            np.ones_like(desvars).reshape(-1, 1) * self.bounds[0],
            np.ones_like(desvars).reshape(-1, 1) * self.bounds[1],
            self.x_1.reshape(-1, 1),
            self.x_2.reshape(-1, 1),
            0,
            df.reshape(-1, 1),
            f_val.reshape(-1, 1),
            dg,
            self.low,
            self.upp,
            1.0,
            a,
            c,
            d,
            move=self.move,
            sub_maxiter=self.sub_maxiter,
            sub_tol=self.sub_tol,
        )
        self.x_2 = self.x_1.copy()
        self.x_1 = desvars.copy()

        self.iteration += 1

        if self.timer:
            end_time = time.time()

        if np.isnan(desvars_new).any():
            desvars_new = np.clip(desvars + self.sub_tol, self.bounds[0], self.bounds[1])

        desvars_new = desvars_new.reshape(-1)
        self.desvars = desvars_new.copy()
        self.change = np.linalg.norm(self.last_desvars - desvars_new, ord=np.inf)
        self.change_f = np.abs((self._J - self.last_f) / self._J) if self._J != 0 else np.inf

        self.last_f = self._J
        self.last_desvars = self.desvars.copy()

        if self.timer:
            return end_time - start_time

    def logs(self):
        return {
            'objective': float(self.last_f) if self.last_f is not None else None,
            'variable change': float(self.change),
            'function change': float(self.change_f),
        }
