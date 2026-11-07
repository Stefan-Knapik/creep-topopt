"""MMA on a separable convex problem with a closed-form KKT solution."""
import numpy as np

from toga.opt import MMA


def test_mma_weighted_least_norm():
    # min sum w_i x_i^2  s.t.  1 - sum x_i <= 0,  0 <= x <= 1   ->   x_i = (1/w_i) / sum(1/w_j)
    w = np.array([1.0, 2.0, 4.0, 8.0])
    x_star = (1 / w) / np.sum(1 / w)

    opt = MMA(desvars_init=np.full(4, 0.5), bounds=(0.0, 1.0), max_iter=100,
              change_tol=1e-8, fun_tol=1e-10)
    while not opt.converged():
        x = opt.desvars
        opt.iter(np.sum(w * x**2), 2 * w * x, [1 - x.sum()], -np.ones((1, 4)))

    assert opt.iteration < 100
    np.testing.assert_allclose(opt.desvars, x_star, atol=1e-4)
    assert opt.desvars.sum() >= 1 - 1e-4
