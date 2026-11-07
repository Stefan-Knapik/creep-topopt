"""Adjoint/AD design sensitivities of the creep-compliance objective against central differences."""
import jax
import jax.numpy as np
import numpy as onp
import pytest

from toga.mat.material_properties import vmap_design_to_physical
from toga.opt.filtering import applyDensityFilter, compute_filter_kd_tree
from toga.opt.penalization import penalize
from toga.physics.creep_explicit import CreepExplicit
from toga.physics.creep_implicit import CreepImplicit

E, A, N_EXP = 160e3, 1e-21, 3.5
LX, LY, NX, NY = 200.0, 100.0, 10, 5
TRACTION = 100.0
END_TIME = 3600 * 24 * 365.0
SOLVER = {"umfpack_solver": {}}


def make_objective(base_cls, mesh, material):
    """Design -> density filter -> RAMP -> transient creep -> J = C(t_end) - C(0)."""

    class LoadedCreep(base_cls):
        def get_surface_maps(self):
            return [lambda u, x, *_: np.array([0.0, TRACTION])]

    def left(p):
        return np.isclose(p[0], 0.0, atol=1e-5)

    def load(p):
        return np.isclose(p[0], LX, atol=1e-5) & np.isclose(p[1], 0.0, atol=0.2 * LY + 1e-5)

    problem = LoadedCreep(mesh, vec=2, dim=2, ele_type="QUAD4",
                          dirichlet_bc_info=[[left] * 2, [0, 1], [lambda p: 0.0] * 2],
                          location_fns=[load], additional_info=(material,))
    problem.penalize_A = lambda x: 1 / penalize(x, {"RAMP": 8.0}) ** 3.5
    H, Hs = compute_filter_kd_tree(problem.fe, 1.5)

    def J(theta):
        geom, mats = vmap_design_to_physical(theta)
        rho = applyDensityFilter({"H": H, "Hs": Hs}, geom.reshape(-1, 1))
        (u, *_), _ = problem.forward_solve(
            theta=np.concatenate((rho, mats), axis=1), end_time=END_TIME, num_time_steps=2,
            fwd_solver_options=SOLVER, vjp_solver_options=SOLVER)
        return problem.compute_compliance(u[-1]) - problem.compute_compliance(u[0])

    return J


@pytest.mark.parametrize("scheme", [CreepExplicit, CreepImplicit], ids=["explicit", "implicit"])
def test_gradient_matches_central_differences(scheme, make_mesh, norton_material):
    J = jax.jit(make_objective(scheme, make_mesh(NX, NY, LX, LY), norton_material(E, A, N_EXP)))
    theta = onp.random.default_rng(0).uniform(0.3, 1.0, (NX * NY, 1))
    grad = onp.asarray(jax.jit(jax.grad(J))(theta)).ravel()
    assert float(J(theta)) > 0  # creep only ever increases compliance

    h = 1e-5
    # Cells are numbered column by column: clamped corner, interior, loaded corner.
    for cell in (0, (NX // 2) * NY + NY // 2, (NX - 1) * NY):
        step = onp.zeros_like(theta)
        step[cell] = h
        fd = (float(J(theta + step)) - float(J(theta - step))) / (2 * h)
        assert grad[cell] == pytest.approx(fd, rel=1e-6)
