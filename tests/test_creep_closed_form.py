"""Uniaxial Norton stress relaxation on a 2x2 QUAD4 plate against the closed-form solution."""
import jax.numpy as np
import numpy as onp
import pytest

from toga.physics.creep_explicit import CreepExplicit
from toga.physics.creep_implicit import CreepImplicit

E, A, N_EXP = 160e3, 1e-21, 3.5  # MPa, MPa^-n s^-1, -
L, U_BAR, END_TIME = 1.0, 0.002, 1e10
SCHEMES = {"explicit": CreepExplicit, "implicit": CreepImplicit}
STEPS = (20, 40)


def analytic_relaxation(t):
    sigma_0 = E * U_BAR / L
    return (sigma_0 ** (1 - N_EXP) + (N_EXP - 1) * E * A * t) ** (1 / (1 - N_EXP))


@pytest.fixture(scope="module")
def sigma_xx(make_mesh, norton_material):
    """sigma_xx(t) at every quadrature point, keyed by (scheme, num_time_steps)."""
    mesh = make_mesh(2, 2, L, L)

    def at(x):
        return lambda p: np.isclose(p[0], x, atol=1e-5)

    def bottom(p):
        return np.isclose(p[1], 0.0, atol=1e-5)

    # ux = 0 left, uy = 0 bottom, ux = u_bar right, top free -> uniaxial tension.
    bcs = [[at(0.0), bottom, at(L)], [0, 1, 0], [lambda p: 0.0, lambda p: 0.0, lambda p: U_BAR]]
    out = {}
    for name, cls in SCHEMES.items():
        problem = cls(mesh, vec=2, dim=2, ele_type="QUAD4", dirichlet_bc_info=bcs,
                      additional_info=(norton_material(E, A, N_EXP),))
        for n_steps in STEPS:
            _, (sigma, *_) = problem.forward_solve(
                theta=np.ones((len(mesh.cells), 1)), end_time=END_TIME, num_time_steps=n_steps,
                fwd_solver_options={"umfpack_solver": {}}, vjp_solver_options={"umfpack_solver": {}})
            out[name, n_steps] = onp.asarray(sigma[..., 0, 0]).reshape(n_steps + 1, -1)
    return out


def rel_error(sigma, n_steps):
    exact = analytic_relaxation(onp.linspace(0.0, END_TIME, n_steps + 1))
    return onp.max(onp.abs(sigma - exact[:, None])) / exact[0]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_relaxation_converges_first_order(sigma_xx, scheme):
    # Homogeneous stress state: every quadrature point carries the same history.
    for n_steps in STEPS:
        assert onp.ptp(sigma_xx[scheme, n_steps], axis=1).max() < 1e-8 * E * U_BAR
    coarse, fine = (rel_error(sigma_xx[scheme, n], n) for n in STEPS)
    assert fine < 0.02
    order = onp.log2(coarse / fine)
    assert 0.8 < order < 1.5


def test_schemes_bracket_exact_solution(sigma_xx):
    # Forward Euler over-relaxes and backward Euler under-relaxes a convex decay.
    n_steps = STEPS[-1]
    exact = analytic_relaxation(onp.linspace(0.0, END_TIME, n_steps + 1))[1:]
    assert onp.all(sigma_xx["explicit", n_steps][1:, 0] < exact)
    assert onp.all(sigma_xx["implicit", n_steps][1:, 0] > exact)
