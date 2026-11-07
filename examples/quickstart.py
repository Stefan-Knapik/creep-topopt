"""Quickstart: creep compliance of a cantilever and its gradient w.r.t. element densities."""
import logging

import jax
import jax.numpy as np

import toga
from toga.fem.generate_mesh import Mesh, rectangle_mesh
from toga.mat import build_material_properties
from toga.physics.creep_explicit import CreepExplicit

LX, LY, NX, NY = 200.0, 100.0, 40, 20  # mm
SERVICE_LIFE = 3600 * 24 * 365.0  # 1 year in s
jax.config.update("jax_enable_x64", True)
toga.logger.setLevel(logging.WARNING)


class Cantilever(CreepExplicit):
    def get_surface_maps(self):
        return [lambda u, x, *_: np.array([0.0, 100.0])]  # MPa, on the loaded edge


def main(num_time_steps=4):
    m = rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY)
    mesh = Mesh(m.points, m.cells_dict["quad"], ele_type="QUAD4")
    material = build_material_properties(["steel"], {
        "E": {"model": "constant", "value": 160e3},  # MPa
        "A": {"model": "constant", "value": 1e-21},  # MPa^-n / s
        "n": {"model": "constant", "value": 3.5},
    })

    def clamped(p):
        return np.isclose(p[0], 0.0, atol=1e-5)

    def loaded(p):
        return np.isclose(p[0], LX, atol=1e-5) & (p[1] <= 0.2 * LY + 1e-5)

    problem = Cantilever(mesh, vec=2, dim=2, ele_type="QUAD4",
                         dirichlet_bc_info=[[clamped] * 2, [0, 1], [lambda p: 0.0] * 2],
                         location_fns=[loaded], additional_info=(material,))
    solver = {"umfpack_solver": {}}  # or {"pardiso_solver": {}} with pypardiso installed

    def creep_compliance(rho):
        (u, *_), _ = problem.forward_solve(
            theta=rho, end_time=SERVICE_LIFE, num_time_steps=num_time_steps,
            fwd_solver_options=solver, vjp_solver_options=solver)
        return problem.compute_compliance(u[-1]) - problem.compute_compliance(u[0])

    rho = np.full((len(mesh.cells), 1), 0.5)
    J, dJ = jax.jit(jax.value_and_grad(creep_compliance))(rho)
    print(f"J = C(t_end) - C(0) = {J:.4f} N mm,  |dJ/drho| = {np.linalg.norm(dJ):.4f}")
    return J, dJ


if __name__ == "__main__":
    main()
