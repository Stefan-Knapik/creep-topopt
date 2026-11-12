"""Verification of the rotating-blade model on a coarse mesh (about 3,000 HEX8), before the full runs.

    python verify_coarse.py                 # all checks
    python verify_coarse.py --checks sign gradient

Checks, each written to results/6_blade_centrifugal/verification/<check>.csv:
  sign      the blade moves outward (+y) under the centrifugal load and the reaction at the clamped
            root balances the resultant of the body force
  energy    J = W_ext(u_end) - W_ext(u_0) equals the internal work of the creep steps,
            sum_s int sigma_s : (eps_s - eps_{s-1}) dV, and creep dissipation + change of elastic energy
  gradient  adjoint sensitivities of J, IN625 fraction, mass and moment of inertia against central
            differences, for single composition variables and along a random direction
  timestep  explicit and implicit schemes with 10, 40 and 160 steps for uniform and graded compositions
  load      stress level, creep strain and J of the uniform compositions at 7000, 8500 and 10000 rpm
"""
import os
import sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import csv

import jax.numpy as np
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import centrifugal_blade as cb

OUT = cb.RESULTS / "verification"


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written to {OUT / f'{name}.csv'}")


def span_graded(blade, root=0.1, tip=0.9):
    """IN625 fraction rising linearly from the root to the tip (a stand-in for an optimized design)."""
    y = onp.asarray(blade.mesh.points)[onp.asarray(blade.mesh.cells)].mean(axis=1)[:, 1]
    w = root + (tip - root) * y / cb.SPAN
    return np.stack((w, 1.0 - w), axis=1)


def check_sign(mesh_file):
    blade = cb.build_blade(mesh_file)
    problem = blade.problem_u
    composition = cb.homogeneous(blade.num_cells, 0.5)
    theta = blade.full_theta(composition)
    T, u_full, _ = blade.forward(theta)
    u0 = u_full[0]

    # Residual of the elastic step without the Dirichlet rows: zero at the free nodes, and at the
    # clamped nodes the force the support exerts on the blade.
    body_params = problem.init_params(theta, T, cb.COLD_TEMPERATURE)
    problem.dt = cb.END_TIME / cb.NUM_TIME_STEPS
    problem.set_params([body_params, []])
    residual = onp.asarray(problem.compute_residual([u0])[0])
    free = onp.setdiff1d(onp.arange(len(residual)), blade.bottom_nodes)
    reaction = residual[blade.bottom_nodes].sum(axis=0)
    force = onp.asarray(problem.compute_total_force(theta))

    # Resultant by hand: rho * omega^2 * (int x dV, int (y + r_hub) dV, 0), rho in tonne/mm^3.
    JxW = onp.asarray(problem.fe.JxW)
    x = onp.asarray(problem.physical_quad_points)
    rho = 0.5 * (8.44e-3 + 7.95e-3) * 1e-6
    force_hand = rho * cb.OMEGA**2 * onp.array([(x[..., 0] * JxW).sum(), ((x[..., 1] + cb.R_HUB) * JxW).sum(), 0.0])

    u_tip = onp.asarray(u0)[blade.top_nodes].mean(axis=0)
    rows = [{
        'F_x': force[0], 'F_y': force[1], 'F_z': force[2],
        'F_hand_x': force_hand[0], 'F_hand_y': force_hand[1], 'F_hand_z': force_hand[2],
        'reaction_x': reaction[0], 'reaction_y': reaction[1], 'reaction_z': reaction[2],
        'max_abs_residual_free': onp.abs(residual[free]).max(),
        'u_tip_mean_x': u_tip[0], 'u_tip_mean_y': u_tip[1], 'u_tip_mean_z': u_tip[2],
        'W_ext_0': float(problem.compute_external_work(u0, theta)),
    }]
    print("\n[sign] 50/50 blend, elastic step")
    for k, v in rows[0].items():
        print(f"  {k:24s} {v: .8e}")
    assert u_tip[1] > 0, "the tip must move outward"
    assert onp.allclose(reaction, -force, rtol=1e-8, atol=1e-8 * onp.abs(force).max())
    assert onp.allclose(force, force_hand, rtol=1e-10, atol=1e-10 * onp.abs(force).max())
    write_csv("sign", rows)


def check_energy(mesh_file):
    rows = []
    for scheme in ("explicit", "implicit"):
        for num_time_steps in (10, 40):
            blade = cb.build_blade(mesh_file, scheme=scheme, num_time_steps=num_time_steps, save_history=True)
            problem = blade.problem_u
            JxW = problem.fe.JxW
            integrate = lambda a: np.sum(a * JxW, axis=(-2, -1))
            ddot = lambda a, b: np.sum(a * b, axis=(-2, -1))

            @jax.jit
            def measures(composition):
                theta = blade.full_theta(composition)
                T, u_full, body_params = blade.forward(theta)
                sigmas, eps_creep = body_params[0], body_params[1]
                eps = jax.vmap(problem.fe.sol_to_grad)(u_full)
                eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))
                eps_el = eps - eps_creep
                U = 0.5 * integrate(ddot(sigmas, eps_el))
                d_eps_creep = np.diff(eps_creep, axis=0)
                return {
                    'J': problem.compute_creep_compliance(u_full, theta),
                    # internal work of the creep steps; equals J at equilibrium
                    'W_int': np.sum(integrate(ddot(sigmas[1:], np.diff(eps, axis=0)))),
                    'D_trap': np.sum(integrate(ddot(0.5 * (sigmas[1:] + sigmas[:-1]), d_eps_creep))),
                    'D_be': np.sum(integrate(ddot(sigmas[1:], d_eps_creep))),
                    'dU': U[-1] - U[0],
                    'U0': U[0],
                    'W_ext_0': problem.compute_external_work(u_full[0], theta),
                }

            for name, composition in (('blend_0.5', cb.homogeneous(blade.num_cells, 0.5)),
                                      ('span_graded', span_graded(blade))):
                m = {k: float(v) for k, v in measures(composition).items()}
                m['res_W_int'] = (m['J'] - m['W_int']) / m['J']
                m['res_D_trap'] = (m['J'] - m['D_trap'] - m['dU']) / m['J']
                m['res_D_be'] = (m['J'] - m['D_be'] - m['dU']) / m['J']
                m['res_clapeyron'] = (m['W_ext_0'] - 2 * m['U0']) / m['W_ext_0']
                rows.append({'scheme': scheme, 'num_time_steps': num_time_steps, 'design': name, **m})
                print(f"[energy] {scheme} N={num_time_steps} {name}: " + ", ".join(f"{k}={v:.6e}" for k, v in m.items()))
            jax.clear_caches()
    write_csv("energy", rows)


def check_gradient(mesh_file):
    blade = cb.build_blade(mesh_file)
    rng = onp.random.default_rng(0)
    composition = rng.uniform(0.2, 0.8, (blade.num_cells, 2))

    def functions(composition):
        f, _ = blade.evaluate(composition)
        return {k: f[k] for k in ('J', 'IN625_fraction', 'mass_g', 'inertia_g_mm2')}

    value = jax.jit(functions)
    grad = {k: onp.asarray(v) for k, v in jax.jit(jax.jacrev(functions))(composition).items()}

    def central(step, h):
        fp = value(composition + h * step)
        fm = value(composition - h * step)
        return {k: (float(fp[k]) - float(fm[k])) / (2 * h) for k in fp}

    centroids = onp.asarray(blade.mesh.points)[onp.asarray(blade.mesh.cells)].mean(axis=1)
    # A cell at the root, at midspan, at the tip, and two random ones; both alloy columns.
    cells = [int(onp.argmin(centroids[:, 1])), int(onp.argmin(onp.abs(centroids[:, 1] - 25.0))),
             int(onp.argmax(centroids[:, 1])), *rng.integers(0, blade.num_cells, 2).tolist()]
    rows = []
    for h in (1e-3, 1e-4):
        for cell in cells:
            for column in (0, 1):
                step = onp.zeros_like(composition)
                step[cell, column] = 1.0
                fd = central(step, h)
                for k in grad:
                    ad = grad[k][cell, column]
                    rows.append({'kind': 'single', 'cell': cell, 'alloy': cb.MATERIAL_NAMES[column], 'y_mm': centroids[cell, 1],
                                 'h': h, 'function': k, 'adjoint': ad, 'central_fd': fd[k], 'rel_error': abs(ad - fd[k]) / abs(fd[k])})
        direction = onp.random.default_rng(1).uniform(-1.0, 1.0, composition.shape)
        fd = central(direction, h)
        for k in grad:
            ad = float(onp.sum(grad[k] * direction))
            rows.append({'kind': 'directional', 'cell': -1, 'alloy': 'all', 'y_mm': onp.nan,
                         'h': h, 'function': k, 'adjoint': ad, 'central_fd': fd[k], 'rel_error': abs(ad - fd[k]) / abs(fd[k])})

    # The same derivative with the load held at a fixed density: the difference to the full one is
    # the path through the design-dependent load.
    blade_fixed = cb.build_blade(mesh_file, frozen=('rho_load',))
    grad_fixed = onp.asarray(jax.jit(jax.grad(lambda c: blade_fixed.evaluate(c)[0]['J']))(composition))
    load_path = onp.linalg.norm(grad['J'] - grad_fixed) / onp.linalg.norm(grad['J'])

    print("\n[gradient] random composition in [0.2, 0.8], adjoint vs central differences")
    for r in rows:
        print(f"  {r['kind']:11s} cell {r['cell']:5d} {r['alloy']:5s} h={r['h']:.0e} {r['function']:15s} "
              f"adjoint {r['adjoint']: .8e}  fd {r['central_fd']: .8e}  rel.err {r['rel_error']:.2e}")
    print(f"  |dJ - dJ(load density fixed)| / |dJ| = {load_path:.4f}  (share of the gradient that comes through the load)")
    rows.append({'kind': 'load_path_share', 'cell': -1, 'alloy': 'all', 'y_mm': onp.nan, 'h': onp.nan, 'function': 'J',
                 'adjoint': load_path, 'central_fd': onp.nan, 'rel_error': onp.nan})
    write_csv("gradient", rows)
    worst = max(r['rel_error'] for r in rows if r['kind'] == 'directional')
    assert worst < 1e-5, f"directional derivative off by {worst:.2e}"


def check_timestep(mesh_file, rpm=cb.RPM):
    rows = []
    for scheme in ("explicit", "implicit"):
        for num_time_steps in (10, 40, 160):
            blade = cb.build_blade(mesh_file, scheme=scheme, num_time_steps=num_time_steps, omega=cb.rpm_to_omega(rpm))
            evaluate = jax.jit(lambda c: blade.evaluate(c)[0])
            designs = {'IN625': cb.homogeneous(blade.num_cells, 1.0), 'blend_0.5': cb.homogeneous(blade.num_cells, 0.5),
                       'SS316': cb.homogeneous(blade.num_cells, 0.0), 'span_graded': span_graded(blade)}
            for name, composition in designs.items():
                f = evaluate(composition)
                rows.append({'scheme': scheme, 'num_time_steps': num_time_steps, 'design': name,
                             **{k: float(f[k]) for k in ('J', 'creep_eq_mean', 'creep_eq_max_quad', 'vm0_mean', 'vm0_max_quad')}})
                print(f"[timestep] {scheme} N={num_time_steps:3d} {name:12s} J = {float(f['J']):.6e}")
            jax.clear_caches()
    reference = {r['design']: r['J'] for r in rows if r['scheme'] == 'implicit' and r['num_time_steps'] == 160}
    for r in rows:
        r['J_rel_to_implicit_160'] = r['J'] / reference[r['design']] - 1.0
        same = [q for q in rows if q['scheme'] == r['scheme'] and q['num_time_steps'] == r['num_time_steps']]
        r['J_over_J_IN625'] = r['J'] / next(q['J'] for q in same if q['design'] == 'IN625')
    print(f"\n{'scheme':9s} {'N':>4s} {'design':12s} {'J [mJ]':>12s} {'vs impl. 160':>13s} {'J / J_IN625':>12s}")
    for r in rows:
        print(f"{r['scheme']:9s} {r['num_time_steps']:4d} {r['design']:12s} {r['J']:12.5e} {r['J_rel_to_implicit_160']:+13.3e} {r['J_over_J_IN625']:12.5f}")
    write_csv("timestep" if rpm == cb.RPM else f"timestep_rpm_{round(rpm)}", rows)


def check_load(mesh_file):
    rows = []
    for rpm in (7000.0, 8500.0, 10000.0):
        omega = rpm * 2.0 * onp.pi / 60.0
        blade = cb.build_blade(mesh_file, omega=omega)
        evaluate = jax.jit(lambda c: blade.evaluate(c)[0])
        for name, w in (('IN625', 1.0), ('blend_0.5', 0.5), ('SS316', 0.0)):
            f = evaluate(cb.homogeneous(blade.num_cells, w))
            rows.append({'rpm': rpm, 'omega_rad_s': omega, 'tip_speed_m_s': omega * (cb.R_HUB + cb.SPAN) * 1e-3,
                         'design': name, **{k: float(f[k]) for k in cb.SCALAR_KEYS}})
            print(f"[load] {rpm:6.0f} rpm {name:10s} J = {float(f['J']):.5e} mJ, J / W0 = {float(f['J'] / f['W0']):.3e}, "
                  f"von Mises mean / peak (cell) = {float(f['vm0_mean']):.2f} / {float(f['vm0_max_cell']):.2f} MPa, "
                  f"creep strain mean / peak (cell) = {float(f['creep_eq_mean']):.3e} / {float(f['creep_eq_max_cell']):.3e}")
        jax.clear_caches()
    write_csv("load", rows)


if __name__ == "__main__":
    checks = {'sign': check_sign, 'energy': check_energy, 'gradient': check_gradient, 'timestep': check_timestep,
              'load': check_load}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checks", nargs="+", default=list(checks), choices=list(checks))
    parser.add_argument("--elem-size", type=float, default=1.5, help="element size of the coarse mesh in mm")
    parser.add_argument("--rpm", type=float, default=cb.RPM,
                        help="rotation speed of the time-step check (default: %(default)s rpm)")
    args = parser.parse_args()
    mesh_file = cb.coarse_mesh(args.elem_size)
    for name in args.checks:
        if name == 'timestep':
            check_timestep(mesh_file, args.rpm)
        else:
            checks[name](mesh_file)
        jax.clear_caches()
