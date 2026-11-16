"""Design-dependent centrifugal body force of experiments/5_blade (blade_problem.py) on a small HEX8 bar:
sign and resultant of the load, reaction at the clamp, and the design sensitivity of the creep
compliance through the load."""
import sys
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
import pytest

from toga.fem.generate_mesh import Mesh
from toga.physics.thermal_creep_explicit import ThermalCreepExplicit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "5_blade"))
import blade_problem as cb  # noqa: E402

WIDTH, LENGTH = 2.0, 20.0   # mm; the bar spans y = 0 ... LENGTH, centred on x = 0
NX, NY, NZ = 2, 10, 2
R_HUB, OMEGA = 250.0, 1500.0
SOLVER = {"umfpack_solver": {}}


def bar_mesh():
    x = onp.linspace(-WIDTH / 2, WIDTH / 2, NX + 1)
    y = onp.linspace(0.0, LENGTH, NY + 1)
    z = onp.linspace(0.0, WIDTH, NZ + 1)
    points = onp.stack(onp.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    node = lambda i, j, k: (i * (NY + 1) + j) * (NZ + 1) + k
    cells = [[node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
              node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)]
             for i in range(NX) for j in range(NY) for k in range(NZ)]
    return Mesh(points, onp.array(cells), ele_type="HEX8")


@pytest.fixture(scope="module")
def bar():
    class RotatingBar(cb.CentrifugalLoad, ThermalCreepExplicit):
        r_hub, omega = R_HUB, OMEGA

    clamp = lambda p: np.isclose(p[1], 0.0, atol=1e-8)
    materials = cb.build_materials(expansion=False)  # the explicit scheme is unstable with thermal expansion
    problem = RotatingBar(bar_mesh(), vec=3, dim=3, ele_type="HEX8",
                          dirichlet_bc_info=[[clamp] * 3, [0, 1, 2], [lambda p: 0.0] * 3],
                          additional_info=(materials,))
    points = onp.asarray(problem.fe.points)
    temperature = np.asarray(600.0 + 200.0 * points[:, 1:2] / LENGTH)  # 600 C at the clamp, 800 C at the tip

    def solve(composition, num_time_steps=2):
        theta = np.concatenate((np.ones((len(composition), 1)), composition), axis=1)
        (u, *_), _ = problem.forward_solve(
            theta=theta, T=temperature, reference_T=600.0, end_time=3600.0 * 100, num_time_steps=num_time_steps,
            fwd_solver_options=SOLVER, vjp_solver_options=SOLVER)
        return theta, u

    return problem, materials, temperature, solve


def test_centrifugal_load_sign_and_reaction(bar):
    problem, materials, temperature, solve = bar
    composition = np.full((problem.num_cells, 2), 0.5)
    theta, u = solve(composition)

    # Resultant: rho omega^2 A (r_hub L + L^2 / 2) along +y; rho from g/mm^3 to tonne/mm^3.
    rho = 0.5 * (8.44e-3 + 7.95e-3) * 1e-6
    expected = rho * OMEGA**2 * WIDTH**2 * (R_HUB * LENGTH + LENGTH**2 / 2)
    force = onp.asarray(problem.compute_total_force(theta))
    assert force[1] == pytest.approx(expected, rel=1e-12)
    assert abs(force[0]) < 1e-9 * expected and force[2] == 0.0

    # The bar is pulled outward; the tip displacement follows the 1D bar, the integral of N(y) / (E(y) A),
    # up to the Poisson constraint at the clamp.
    points = onp.asarray(problem.fe.points)
    tip = onp.isclose(points[:, 1], LENGTH)
    u0 = onp.asarray(u[0])
    assert u0[tip, 1].min() > 0.0
    y = onp.linspace(0.0, LENGTH, 2001)
    E = onp.array([float(materials.E(np.array([0.5, 0.5, 600.0 + 200.0 * s / LENGTH]))) for s in y])
    strain = rho * OMEGA**2 * (R_HUB * (LENGTH - y) + (LENGTH**2 - y**2) / 2) / E
    u_tip_1d = onp.sum(0.5 * (strain[1:] + strain[:-1]) * onp.diff(y))
    assert u0[tip, 1].mean() == pytest.approx(u_tip_1d, rel=0.02)

    # Reaction at the clamp balances the load; the free nodes are in equilibrium. (problem.dt is left as
    # the solve set it: the jitted element kernels bake it in, so changing it here would poison their cache.)
    problem.set_params([problem.init_params(theta, temperature, 600.0), []])
    residual = onp.asarray(problem.compute_residual([u[0]])[0])
    clamped = onp.isclose(points[:, 1], 0.0)
    assert onp.abs(residual[~clamped]).max() < 1e-9 * expected
    assert residual[clamped].sum(axis=0) == pytest.approx(-force, rel=1e-9, abs=1e-9 * expected)

    # Creep under a dead load only ever adds external work.
    assert float(problem.compute_creep_compliance(u, theta)) > 0.0


def test_creep_compliance_gradient_through_design_dependent_load(bar):
    problem, _, _, solve = bar
    rng = onp.random.default_rng(0)
    composition = rng.uniform(0.2, 0.8, (problem.num_cells, 2))

    def J(composition):
        theta, u = solve(composition)
        return problem.compute_creep_compliance(u, theta)

    J = jax.jit(J)
    grad = onp.asarray(jax.jit(jax.grad(J))(composition))
    direction = rng.uniform(-1.0, 1.0, composition.shape)
    h = 1e-4
    fd = (float(J(composition + h * direction)) - float(J(composition - h * direction))) / (2 * h)
    assert onp.sum(grad * direction) == pytest.approx(fd, rel=1e-5)

    # With the load density frozen the derivative changes: the load path is part of the gradient.
    model = problem.model_rho_load
    try:
        problem.model_rho_load = cb.FixedComposition(model, [0.5, 0.5])
        grad_fixed = onp.asarray(jax.grad(J.__wrapped__)(composition))
    finally:
        problem.model_rho_load = model
    assert onp.linalg.norm(grad - grad_fixed) > 0.01 * onp.linalg.norm(grad)
