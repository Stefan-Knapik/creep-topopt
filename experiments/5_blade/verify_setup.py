"""Checks of the rotating blade with thermal expansion and the composition filter. Writes
results/5_blade/verification/{sanity,energy,gradient,time_integration}.csv.

    python verify_setup.py                              # sanity, energy and gradient on the 3k mesh
    python verify_setup.py --checks sanity              # load resultant, reaction, sign, expansion on/off
    python verify_setup.py --checks energy              # J against the internal work and dissipation
    python verify_setup.py --checks gradient            # adjoint vs central differences in the raw design
                                                        # variables: filter, expansion, body force, shroud
    python verify_setup.py --checks timestep --fine --root-stress 100 --hours 100 [--design <run>/design_final.npy]
                                                        # J at 10 / 40 / 160 implicit steps
"""
import argparse

from _common import bp, jax, release, write_csv, case_kwargs
import jax.numpy as np
import numpy as onp

from optimize_graded_blade import add_case_arguments

OUT = bp.RESULTS / "verification"


def load_kwargs(args):
    return dict(root_stress=args.root_stress, shroud=args.shroud, end_time=3600.0 * args.hours)


def check_sanity(args, kwargs):
    rows = []
    J = {}
    for label, expansion in (("expansion", True), ("no expansion", False)):
        blade = bp.build_blade(**{**kwargs, 'expansion': expansion}, **load_kwargs(args))
        problem = blade.problem_u
        blend = bp.homogeneous(blade.num_cells, 0.5)
        theta = blade.full_theta(blend)
        f, aux = jax.jit(blade.evaluate)(blend)
        # Resultant of the body force against m * omega^2 * r_cg (uniform density: r_cg from the volume).
        x, JxW = problem.physical_quad_points, problem.fe.JxW
        y_cg = float(np.sum((x[..., 1] + blade.r_hub) * JxW) / np.sum(JxW))
        force = onp.asarray(problem.compute_total_force(theta))
        force_hand = float(f['mass_g']) * bp.G_TO_TONNE * blade.omega**2 * y_cg
        # Reaction at the clamp: residual of the elastic step without the Dirichlet rows.
        u0 = aux['u_full'][0]
        body_params = (*problem.init_params(theta, aux['T'], blade.t_root), np.zeros((blade.num_cells, problem.fe.num_quads, 3, 3)))
        surface_params = [np.ones((q.shape[0], problem.fe.num_face_quads)) for q in problem.physical_surface_quad_points]
        problem.set_params([body_params, surface_params])
        residual = onp.asarray(problem.compute_residual([u0])[0])
        bottom = blade.node_sets['bottom']
        free = onp.setdiff1d(onp.arange(len(residual)), bottom)
        reaction_y = residual[bottom].sum(axis=0)[1]
        tip_area = blade.volume / bp.SPAN
        tip_force = blade.tip_traction * tip_area   # nominal; the true tip area differs slightly
        tip_uy = float(onp.mean(onp.asarray(u0)[blade.node_sets['top'], 1]))
        J[label] = float(f['J'])
        rows.append({'case': label, 'cells': blade.num_cells, 'rpm': blade.rpm, 'F_y_N': force[1], 'F_y_hand_N': force_hand,
                     'reaction_y_N': reaction_y, 'tip_force_nominal_N': tip_force,
                     'max_abs_residual_free': onp.abs(residual[free]).max(), 'tip_uy_0_mm': tip_uy,
                     **{k: float(f[k]) for k in bp.SCALAR_KEYS}})
        print(f"[sanity] {label:13s} {blade.num_cells} cells, {blade.rpm:.0f} rpm | F_y {force[1]:.2f} N (hand {force_hand:.2f}), "
              f"reaction {reaction_y:.2f} N, nominal tip force {tip_force:.2f} N | free residual {onp.abs(residual[free]).max():.1e} | "
              f"T {float(f['T_min']):.1f}..{float(f['T_max']):.1f} | vm0 mean {float(f['vm0_mean']):.1f}, peak "
              f"{float(f['vm0_max_cell']):.1f} MPa | J {float(f['J']):.6g} mJ | tip u_y(0) {tip_uy:.4f} mm")
        assert abs(force[1] / force_hand - 1.0) < 1e-6
        assert tip_uy > 0.0 and float(f['J']) > 0.0
        assert onp.abs(residual[free]).max() < 1e-6 * abs(force[1])
        # the clamp carries the body force and the tip traction (true tip area within a few % of the nominal)
        assert abs(-reaction_y - force[1] - tip_force) < 0.05 * tip_force + 1e-6 * abs(force[1])
        del blade
        release()
    rel = J["expansion"] / J["no expansion"] - 1.0
    rows.append({'case': 'J expansion / J no expansion - 1 (uniform blend)', 'J': rel})
    print(f"[sanity] uniform blend: J with expansion / without - 1 = {rel:+.3%}")
    write_csv(OUT / "sanity.csv", rows)


def check_energy(args, kwargs):
    rows = []
    for steps in (10, 40):
        blade = bp.build_blade(**{**kwargs, 'num_time_steps': steps}, **load_kwargs(args), save_history=True)
        problem = blade.problem_u
        JxW = problem.fe.JxW
        integrate = lambda a: np.sum(a * JxW, axis=(-2, -1))
        ddot = lambda a, b: np.sum(a * b, axis=(-2, -1))

        @jax.jit
        def measures(composition):
            f, aux = blade.evaluate(composition)
            sigmas, eps_creep = aux['body_params'][0], aux['body_params'][1]
            eps = jax.vmap(problem.fe.sol_to_grad)(aux['u_full'])
            eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))
            d_eps_creep = np.diff(eps_creep, axis=0)
            return {'J': f['J'],
                    # internal work of the stress on the strain increments; equals J at equilibrium
                    'W_int': np.sum(integrate(ddot(sigmas[1:], np.diff(eps, axis=0)))),
                    'D_be': np.sum(integrate(ddot(sigmas[1:], d_eps_creep))),
                    'D_trap': np.sum(integrate(ddot(0.5 * (sigmas[1:] + sigmas[:-1]), d_eps_creep)))}

        for name, composition in (('blend_0.5', bp.homogeneous(blade.num_cells, 0.5)), ('span_graded', bp.span_graded(blade))):
            m = {k: float(v) for k, v in measures(composition).items()}
            m['res_W_int'] = (m['J'] - m['W_int']) / m['J']
            m['D_be_over_J'] = m['D_be'] / m['J']
            m['D_trap_over_J'] = m['D_trap'] / m['J']
            rows.append({'num_time_steps': steps, 'design': name, **m})
            print(f"[energy] N={steps} {name}: " + ", ".join(f"{k}={v:.6e}" for k, v in m.items()))
            assert abs(m['res_W_int']) < 1e-5 and m['J'] > 0.0
        del blade, measures
        release()
    write_csv(OUT / "energy.csv", rows)


def check_gradient(args, kwargs):
    shroud = args.shroud if args.shroud else 0.3
    blade = bp.build_blade(**kwargs, root_stress=args.root_stress, shroud=shroud, end_time=3600.0 * args.hours)
    rng = onp.random.default_rng(0)
    composition = onp.asarray(rng.uniform(0.2, 0.8, (blade.num_cells, 2)))

    def functions(c):
        f = blade.evaluate(c)[0]
        return {k: f[k] for k in ('J', 'IN625_fraction', 'mass_g', 'inertia_g_mm2')}

    value = jax.jit(functions)
    grad = {k: onp.asarray(v) for k, v in jax.jit(jax.jacrev(functions))(composition).items()}
    rows = []
    for seed in (1, 2, 3):
        direction = onp.random.default_rng(seed).uniform(-1.0, 1.0, composition.shape)
        for h in (1e-3, 1e-4):
            plus, minus = value(composition + h * direction), value(composition - h * direction)
            for k in grad:
                ad = float(onp.sum(grad[k] * direction))
                fd = (float(plus[k]) - float(minus[k])) / (2 * h)
                rows.append({'function': k, 'direction': seed, 'h': h, 'adjoint': ad, 'central_fd': fd,
                             'rel_error': abs(ad - fd) / abs(fd)})
                print(f"[gradient] {k:15s} direction {seed} h={h:.0e}  adjoint {ad: .8e}  fd {fd: .8e}  rel.err {rows[-1]['rel_error']:.2e}")
    write_csv(OUT / "gradient.csv", rows)
    for k in grad:
        worst = min(max(r['rel_error'] for r in rows if r['h'] == h and r['function'] == k) for h in (1e-3, 1e-4))
        assert worst < 1e-4, f"directional derivative of {k} off by {worst:.2e}"
    del blade
    release()


def check_timestep(args, kwargs):
    rows = []
    for steps in args.timesteps:
        blade = bp.build_blade(**{**kwargs, 'num_time_steps': steps}, **load_kwargs(args))
        designs = {"SS316": bp.homogeneous(blade.num_cells, 0.0), "IN625": bp.homogeneous(blade.num_cells, 1.0),
                   "blend_0.5": bp.homogeneous(blade.num_cells, 0.5)}
        if args.design:
            designs["design"] = np.asarray(onp.load(args.design))
        evaluate = jax.jit(lambda c: blade.evaluate(c)[0])
        for design, composition in designs.items():
            f = evaluate(composition)
            rows.append({'num_time_steps': steps, 'design': design,
                         **{k: float(f[k]) for k in ('J', 'IN625_fraction', 'vm0_mean', 'vm0_max_cell', 'vm_end_max_cell',
                                                     'creep_eq_max_cell', 'u_max_end')}})
            print(f"[timestep] N={steps:4d} {design:10s} J {rows[-1]['J']: .6g} mJ  creep peak {rows[-1]['creep_eq_max_cell']:.2e}", flush=True)
        del blade, evaluate
        release()
    reference = {r['design']: r['J'] for r in rows if r['num_time_steps'] == max(args.timesteps)}
    for r in rows:
        r['J_rel_to_finest'] = r['J'] / reference[r['design']] - 1.0
        print(f"  N={r['num_time_steps']:4d} {r['design']:10s} {r['J_rel_to_finest']:+.3%}")
    write_csv(OUT / f"time_integration{'_' + args.tag if args.tag else ''}.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checks", nargs="+", default=["sanity", "energy", "gradient"],
                        choices=["sanity", "energy", "gradient", "timestep"])
    add_case_arguments(parser)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[10, 40, 160])
    parser.add_argument("--design", default=None, help="design_final.npy of a run, for the timestep check")
    parser.add_argument("--tag", default=None, help="suffix of time_integration.csv")
    args = parser.parse_args()
    if args.elem_size is None and not args.fine:
        args.elem_size = 1.5   # the 3k mesh
    kwargs = case_kwargs(args)
    for check in args.checks:
        {"sanity": check_sanity, "energy": check_energy, "gradient": check_gradient, "timestep": check_timestep}[check](args, kwargs)
