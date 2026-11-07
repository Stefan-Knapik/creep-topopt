"""Plain-Python replica of Problem.forward_solve's time loop that reduces to scalars every step, avoiding the
full stress/creep-strain history lax.scan stacks (tens of GB on the finest mesh)."""
import time

import jax.numpy as np
import numpy as onp

from toga.fem.solver import solver

from common import (SOLVER_OPTIONS, eq_creep_strain, node_index, p_norm,
                    von_mises, vol_avg)


def time_steps(end_time, num_time_steps):
    """The step sizes forward_solve uses: an initial dt = 0 purely elastic solve
    followed by num_time_steps uniform creep steps."""
    tv = onp.linspace(0.0, end_time, num_time_steps + 1)
    return onp.concatenate([[0.0], onp.diff(tv)])


def surface_params_for(problem):
    return [np.ones((sq.shape[0], problem.fe.num_face_quads))
            for sq in problem.physical_surface_quad_points]


def _make_step(problem, solver_options):
    """One time step: impose dt, solve equilibrium, update the state."""
    def step(bp, dt):
        bp = (bp[0], bp[1], bp[2],
              np.full((len(problem.fe.cells), problem.fe.num_quads), dt))
        problem.set_params([bp, surface_params_for(problem)])
        sol_list = solver(problem, solver_options)
        return sol_list, problem.update_stress_strain(sol_list[0], bp)
    return step


def run_history(problem, theta, end_time, num_time_steps,
                solver_options=None, probes=()):
    """Run the transient analysis, recording scalar QoIs at every step.

    probes  iterable of (name, x, y, component) nodal probes
    """
    solver_options = solver_options or SOLVER_OPTIONS
    problem.end_time = end_time
    dts = time_steps(end_time, num_time_steps)
    step = _make_step(problem, solver_options)
    probe_idx = [(name, node_index(problem, x, y), comp) for name, x, y, comp in probes]

    bp = problem.init_params(theta)
    rec = {k: [] for k in ("dt", "compliance", "umax", "vm_avg", "vm_max",
                           "vm_pnorm8", "eps_eq_avg", "eps_eq_max", "finite")}
    for name, _, _ in probe_idx:
        rec[f"probe_{name}"] = []

    t0 = time.perf_counter()
    for s, dt in enumerate(dts):
        sol_list, bp = step(bp, float(dt))
        sol = sol_list[0]
        sigma, eps_cr = bp[0], bp[1]

        vm = von_mises(sigma)
        eq = eq_creep_strain(eps_cr)
        rec["dt"].append(float(dt))
        rec["compliance"].append(float(problem.compute_compliance(sol)))
        rec["umax"].append(float(np.max(np.abs(sol))))
        rec["vm_avg"].append(vol_avg(problem, vm))
        rec["vm_max"].append(float(np.max(vm)))
        rec["vm_pnorm8"].append(p_norm(problem, vm, p=8))
        rec["eps_eq_avg"].append(vol_avg(problem, eq))
        rec["eps_eq_max"].append(float(np.max(eq)))
        # Neither the global solver nor the local return map raises on failure,
        # so divergence has to be detected here or it silently becomes data.
        rec["finite"].append(bool(np.all(np.isfinite(sol)) and np.all(np.isfinite(sigma))))
        for name, idx, comp in probe_idx:
            rec[f"probe_{name}"].append(float(sol[idx, comp]))

    out = {k: onp.asarray(v) for k, v in rec.items()}
    out["J"] = out["compliance"][-1] - out["compliance"][0]
    out["finite_all_steps"] = bool(onp.all(out["finite"]))
    out["wall_time_s"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------
# Fidelity gate
# --------------------------------------------------------------------------

def validate_against_forward_solve(problem, theta, end_time, num_time_steps,
                                   rtol_C=1e-11, rtol_J=1e-8):
    """Assert that the manual driver reproduces Problem.forward_solve.

    The gate is applied to the compliances C(0) and C(t_end), not to J.  J is
    their difference and the two are close -- C ~ 1e3 against J ~ 1 -- so J
    amplifies any round-off in C by ~1e3.  The manual loop and the scan sum the
    same terms in different orders, so they are not expected to agree bitwise;
    what has to hold is that they agree to round-off on the quantities actually
    being summed.  A loose secondary bound on J catches an outright algorithmic
    disagreement.
    """
    full_sol, _ = problem.forward_solve(
        theta=theta, end_time=end_time, num_time_steps=num_time_steps,
        fwd_solver_options=SOLVER_OPTIONS, vjp_solver_options=SOLVER_OPTIONS)
    u = full_sol[0]
    C0_ref = float(problem.compute_compliance(u[0]))
    Cend_ref = float(problem.compute_compliance(u[-1]))
    J_ref = Cend_ref - C0_ref

    out = run_history(problem, theta, end_time, num_time_steps)
    C0_drv = float(out["compliance"][0])
    Cend_drv = float(out["compliance"][-1])
    J_drv = float(out["J"])

    rel_C0 = abs(C0_drv - C0_ref) / abs(C0_ref)
    rel_Cend = abs(Cend_drv - Cend_ref) / abs(Cend_ref)
    rel_J = abs(J_drv - J_ref) / abs(J_ref)
    print(f"  driver  C(0) = {C0_drv:.16g}   C(t_end) = {Cend_drv:.16g}   J = {J_drv:.16g}")
    print(f"  scan    C(0) = {C0_ref:.16g}   C(t_end) = {Cend_ref:.16g}   J = {J_ref:.16g}")
    print(f"  relative difference: C(0) {rel_C0:.3e}, C(t_end) {rel_Cend:.3e}, "
          f"J {rel_J:.3e}")
    if max(rel_C0, rel_Cend) > rtol_C:
        raise AssertionError(
            f"manual driver disagrees with forward_solve on the compliance: "
            f"{max(rel_C0, rel_Cend):.3e} > {rtol_C:.0e}")
    if rel_J > rtol_J:
        raise AssertionError(
            f"manual driver disagrees with forward_solve on J: "
            f"{rel_J:.3e} > {rtol_J:.0e}")
    return J_drv, J_ref
