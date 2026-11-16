"""Sec. 4.5: IN625/SS316 compositional grading of a rotating turbine blade.

The blade creeps under the centrifugal force of its own mass (optionally with a tip shroud) with the
root at 750 C and the tip at 800 C, thermal expansion on, implicit time integration. The optimizer
minimizes the creep compliance J = W_ext(u_end) - W_ext(u_0) for an IN625 fraction of at most V*;
the composition is density-filtered (blade_problem.py).

    python optimize_graded_blade.py --fine --study deploy --vf 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                                                    # the paper setting: 50,336 HEX8, 10 steps, 50 MMA iterations
    TOGA_MAX_ITER=15 python optimize_graded_blade.py --steps 2 --root-stress 80 --shroud 0.3 --hours 30 --vf 0.3
                                                    # exploration: 25k mesh, 2 time steps, 15 iterations
    python optimize_graded_blade.py ... --radius-mm 0            # no filter
    python optimize_graded_blade.py ... --no-expansion           # thermal expansion off
    python optimize_graded_blade.py ... --alpha-mode secant --cte-mismatch-scale 0.5 --init perturbed --tag x

Writes results/5_blade/<study>/<case>/V<bound %>/ (blade_problem.case_name): case.json, history.csv,
scalars.csv, walltime.csv, design_final.npy (raw design variables), vtk/sol_XXX.vtu, and DONE or FAILED.
"""
import os
import sys
from pathlib import Path


def run(vf=0.3, root_stress=100.0, shroud=0.0, hours=100.0, radius_mm=None, num_time_steps=None, elem_size=None,
        fine=False, alpha_mode="tangent", expansion=True, cte_mismatch_scale=1.0, init="uniform", tag=None,
        study="explore", t_root=None, t_hot=None, r_hub=None, max_iter=None, output_freq=5):

    #%% Import modules

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
    # Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
    # embedding them as constants. Must be set before importing jax.
    os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as np
    import numpy as onp
    import json
    import time
    _t_script_start = time.perf_counter()

    import logging
    import toga
    toga.logger.setLevel(logging.DEBUG)

    from toga import logger
    from toga.opt import MMA
    from toga.utils import clean_data_dir, quad_to_cell_average, save_vtu, timer, oom_hint

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blade_problem as bp

    radius_mm = bp.RADIUS_MM if radius_mm is None else radius_mm
    num_time_steps = bp.NUM_TIME_STEPS if num_time_steps is None else num_time_steps
    elem_size = bp.ELEM_SIZE if elem_size is None else elem_size
    t_root = bp.T_ROOT if t_root is None else t_root
    t_hot = bp.T_HOT if t_hot is None else t_hot
    r_hub = bp.R_HUB if r_hub is None else r_hub
    max_iter = int(os.environ.get('TOGA_MAX_ITER', 50)) if max_iter is None else max_iter

    data_dir = str(bp.run_dir(study, vf, root_stress=root_stress, shroud=shroud, hours=hours, radius_mm=radius_mm,
                              mesh=bp.mesh_name(elem_size, fine), num_time_steps=num_time_steps, alpha_mode=alpha_mode,
                              expansion=expansion, cte_mismatch_scale=cte_mismatch_scale, init=init, tag=tag))

    # Persistent JAX compilation cache of this run - skips recompilation when the run is repeated
    jax.config.update("jax_compilation_cache_dir", os.path.join(data_dir, 'jax_cache'))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    print(f"Clearing old results in {data_dir}...")
    clean_data_dir(data_dir)

    #%% Thermal and creep problems of the rotating blade (blade_problem.py)

    blade = bp.build_blade(t_root=t_root, t_hot=t_hot, r_hub=r_hub, root_stress=root_stress, shroud=shroud,
                           end_time=3600.0 * hours, mesh_file=bp.FINE_MESH if fine else None, elem_size=elem_size,
                           num_time_steps=num_time_steps, expansion=expansion, alpha_mode=alpha_mode,
                           cte_mismatch_scale=cte_mismatch_scale, radius_mm=radius_mm)
    problem_u = blade.problem_u
    logger.info(f"root {t_root:.0f} C, tip {t_hot:.0f} C; hub radius {r_hub:.0f} mm, nominal root stress "
                f"{root_stress:.0f} MPa: {blade.rpm:.0f} rpm, tip speed {blade.tip_speed:.0f} m/s; shroud share "
                f"{shroud:.0%} (tip traction {blade.tip_traction:.1f} MPa, {blade.shroud_mass:.1f} g); {hours:g} h in "
                f"{num_time_steps} steps; filter radius {radius_mm} mm; expansion {alpha_mode if expansion else 'off'}; "
                f"{blade.num_cells} cells")

    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'case.json'), 'w') as cf:
        json.dump({'t_root': t_root, 't_hot': t_hot, 'r_hub': r_hub, 'root_stress': root_stress, 'vf': vf,
                   'shroud': shroud, 'hours': hours, 'fine': fine, 'elem_size': elem_size,
                   'num_time_steps': num_time_steps, 'radius_mm': radius_mm, 'expansion': expansion,
                   'alpha_mode': alpha_mode, 'cte_mismatch_scale': cte_mismatch_scale, 'init': init, 'tag': tag,
                   'study': study, 'max_iter': max_iter,
                   'tip_traction': blade.tip_traction, 'shroud_mass_g': blade.shroud_mass,
                   'num_cells': blade.num_cells, 'rpm': blade.rpm, 'omega': blade.omega,
                   'tip_speed_m_s': blade.tip_speed}, cf, indent=1)

    #%% Defining the forward pass with objective and constraints

    vf_constraint = vf

    def objective(params):
        f, aux = blade.evaluate(params)
        return f['J'], (f, aux)

    def constraints(params):
        # bound on the IN625 content of the filtered (physical) composition
        return {'IN vf': blade.in625_fraction(params) / vf_constraint - 1.0}

    # One adjoint pass for J; the constraint is explicit in the composition.
    objective_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    constraints_grad = jax.jit(lambda params: (constraints(params), jax.jacrev(constraints)(params)))

    def evaluate_performance_grad(params):
        (_, (scalars, aux)), dJ = objective_grad(params)
        g, dg = constraints_grad(params)
        F = {'J': scalars['J'], **g}
        dF = {'J': dJ, **dg}
        aux['scalars'] = scalars
        return dF, (F, aux)

    #%% Define the output solution function

    def save_fields(data_path, i, theta, dF, aux):
        logger.info(f"Saving solution for iteration {i:d}...")
        mat_raw = onp.asarray(theta / np.sum(theta, axis=1, keepdims=True))   # normalized design
        mat_filtered = onp.asarray(aux['theta'][:, 1:])                       # what the physics sees
        u_full = aux['u_full']
        body_params = aux['body_params']

        point_infos = [
            ('U_final', u_full[-1]),
            ('U_0', u_full[0]),
            ('T', onp.asarray(aux['T'])),
        ]

        cell_infos = [
            ('mat', mat_raw),
            ('mat_filtered', mat_filtered),
            ('creep_strain_final', quad_to_cell_average(problem_u.fe, body_params[1][-1])),
            ('stress_final', quad_to_cell_average(problem_u.fe, body_params[0][-1])),
            ('stress_0', quad_to_cell_average(problem_u.fe, body_params[0][0])),
            ('dJ_dmat', dF['J'].reshape(-1, mat_raw.shape[1])),
        ]

        save_vtu(problem_u.fe, os.path.join(data_path, f'vtk/sol_{i:03d}.vtu'), cell_infos=cell_infos, point_infos=point_infos)
        onp.save(os.path.join(data_path, 'design_final.npy'), onp.asarray(theta))   # raw design variables

    def output_sol(data_path, theta, F, dF, aux):
        os.makedirs(data_path, exist_ok=True)
        i = output_sol.counter

        constraint_keys = [k for k in F if k != 'J']
        history_log_path = os.path.join(data_path, 'history.csv')
        if i == 0:
            with open(history_log_path, 'w') as hf:
                hf.write('iter,\t     J,\t' + '\t,'.join(constraint_keys) + '\n')
        with open(history_log_path, 'a') as hf:
            vc_str = '\t,'.join(f'{float(F[k]):.4f}' for k in constraint_keys)
            hf.write(f'{i:4},\t{float(F["J"]):.5g},\t{vc_str}\n')

        # J, IN625 fraction, mass, inertia, stress, creep-strain and temperature scalars of every iterate
        scalars_path = os.path.join(data_path, 'scalars.csv')
        if i == 0:
            with open(scalars_path, 'w') as sf:
                sf.write('iter,' + ','.join(bp.SCALAR_KEYS) + '\n')
        with open(scalars_path, 'a') as sf:
            sf.write(f'{i},' + ','.join(f'{float(aux["scalars"][k]):.10g}' for k in bp.SCALAR_KEYS) + '\n')

        if i % optimization_params['output_freq'] == 0:
            save_fields(data_path, i, theta, dF, aux)

        output_sol.counter += 1
    output_sol.counter = 0

    # %% Define the optimization problem and solve it

    theta_ini = bp.homogeneous(blade.num_cells, vf_constraint) # uniform blend at the IN625 bound
    if init == "perturbed":
        # same blend on average, +-0.1 of noise in the IN625 fraction: is the layout start-dependent?
        noise = onp.random.default_rng(0).uniform(-0.1, 0.1, blade.num_cells)
        in625 = onp.clip(vf_constraint + noise, 0.02, 0.98)
        theta_ini = np.asarray(onp.stack((in625, 1.0 - in625), axis=1))

    optimization_params = {
        'max_iter': max_iter,
        'change_tol': 1e-5,
        'fun_tol': 1e-5,
        'move': 0.1,
        'output_freq': output_freq,
    }

    def optimize(fe, theta_ini, optimization_params, data_sub_dir=None):
        data_folder = os.path.join(data_dir, data_sub_dir) if data_sub_dir else data_dir

        # The filter acts on the normalized composition inside blade.evaluate (blade_problem.py).

        theta = theta_ini

        optimizer = MMA(
                desvars_init=onp.asarray(theta).reshape(-1),
                bounds=(1e-3, 1.0),
                is_independent=True,
                max_iter=optimization_params['max_iter'],
                change_tol=optimization_params['change_tol'],
                fun_tol=optimization_params['fun_tol'],
                move=optimization_params['move'],
        )

        loop = 0
        J_scale = None

        os.makedirs(data_folder, exist_ok=True)
        walltime_log_path = os.path.join(data_folder, 'walltime.csv')
        with open(walltime_log_path, 'w') as wf:
            wf.write('phase,\tt_eval,\tt_save,\tt_opt,\tt_iter\n')
            wf.write(f'setup,\t{time.perf_counter() - _t_script_start:.3f},,,\n')

        while not optimizer.converged():
            with timer() as t_iter:
                with timer() as t_eval, oom_hint():
                    dF, (F, aux) = evaluate_performance_grad(theta)

                if not (onp.isfinite(float(F['J'])) and onp.all(onp.isfinite(onp.asarray(dF['J'])))):
                    with open(os.path.join(data_folder, 'FAILED'), 'w') as ff:
                        ff.write(f'non-finite J or dJ at MMA iteration {loop}\n')
                    raise FloatingPointError(f"non-finite J or dJ at MMA iteration {loop}")

                # The optimizer sees J scaled to 10 at the initial design: J spans orders of magnitude
                # over the cases, and MMA weighs the objective against the constraint in absolute terms.
                if J_scale is None:
                    J_scale = 10.0 / abs(float(F['J']))
                J = F['J'].reshape(1,) * J_scale
                dJ = dF['J'].reshape(-1) * J_scale
                vc = np.array([F[key] for key in F if key != 'J']).reshape(-1)
                dvc = np.array([dF[key] for key in dF if key != 'J']).reshape(len(vc), -1)
                logger.debug(f"J = {F['J']}; vc = {vc.ravel()}")

                with timer() as t_save:
                    output_sol(data_folder, theta, F, dF, aux)

                with timer() as t_optimizer:
                    optimizer.iter(onp.array(J), onp.array(dJ), onp.array(vc), onp.array(dvc))
                    theta_new = np.array(optimizer.desvars.reshape(theta.shape))
                    last_state = (theta, F, dF, aux)
                    theta = theta_new

            with open(walltime_log_path, 'a') as wf:
                wf.write(f'iter_{loop},\t{t_eval:.3f},\t{t_save:.3f},\t{t_optimizer:.3f},\t{t_iter:.3f}\n')

            logger.info(f'MMA iter {loop:d}; J={float(F["J"]):.5g}; vc={vc.ravel()}; '
                        f'change={optimizer.change:.3e}; change_f={optimizer.change_f:.3e}')
            loop += 1

        # the last evaluated design is always written (the .vtu files are otherwise every few iterations)
        if (loop - 1) % optimization_params['output_freq'] != 0:
            theta_last, _, dF_last, aux_last = last_state
            save_fields(data_folder, loop - 1, theta_last, dF_last, aux_last)

        with open(os.path.join(data_folder, 'DONE'), 'w') as ff:
            ff.write(f'{loop} MMA iterations\n')

    optimize(problem_u.fe, theta_ini, optimization_params)

def _release():
    """Drop the finished run's compiled executables and assembly caches before the next run.
    Every run() builds a new Problem whose jitted closures JAX would otherwise keep alive."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()


def add_case_arguments(parser):
    """Arguments that define a case, shared by the scripts of this folder."""
    parser.add_argument("--root-stress", type=float, default=100.0,
                        help="nominal root stress [MPa], which sets the speed (default 100)")
    parser.add_argument("--shroud", type=float, default=0.0,
                        help="share of the nominal root stress that comes from a tip shroud (radial traction on the tip)")
    parser.add_argument("--hours", type=float, default=100.0, help="hold time [h] (default 100)")
    parser.add_argument("--radius-mm", type=float, default=None,
                        help="filter radius of the composition [mm] (default 1.4; 0: no filter)")
    parser.add_argument("--steps", type=int, default=None, help="number of time steps (default 10)")
    parser.add_argument("--elem-size", type=float, default=None,
                        help="mesh size [mm]: 0.72 (default, 25k HEX8), 0.98 (10.8k), 1.5 (3k)")
    parser.add_argument("--fine", action="store_true", help="use the paper mesh (50,336 HEX8)")
    parser.add_argument("--alpha-mode", choices=["tangent", "table", "secant"], default="tangent",
                        help="thermal-expansion model (blade_problem.ThermalExpansion)")
    parser.add_argument("--no-expansion", action="store_true", help="switch thermal expansion off")
    parser.add_argument("--cte-mismatch-scale", type=float, default=1.0,
                        help="scale of the difference between the alloys' expansion coefficients")
    parser.add_argument("--t-root", type=float, default=None, help="root temperature [C] (default 750)")
    parser.add_argument("--t-hot", type=float, default=None, help="tip temperature [C] (default 800)")
    parser.add_argument("--r-hub", type=float, default=None, help="hub radius [mm] (default 250)")
    parser.add_argument("--study", default="explore", help="subfolder of results/5_blade/ (default: explore)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vf", type=float, nargs="+", default=[0.3], help="upper bounds on the IN625 volume fraction")
    add_case_arguments(parser)
    parser.add_argument("--init", choices=["uniform", "perturbed"], default="uniform", help="starting design")
    parser.add_argument("--tag", default=None, help="suffix of the case folder")
    parser.add_argument("--output-freq", type=int, default=5, help="write a .vtu every this many iterations (and the last)")
    args = parser.parse_args()

    for vf in args.vf:
        print(f"\n\nRunning root stress {args.root_stress:g} MPa, shroud {args.shroud}, {args.hours:g} h, V* = {vf}...\n\n")
        run(vf=vf, root_stress=args.root_stress, shroud=args.shroud, hours=args.hours, radius_mm=args.radius_mm,
            num_time_steps=args.steps, elem_size=args.elem_size, fine=args.fine, alpha_mode=args.alpha_mode,
            expansion=not args.no_expansion, cte_mismatch_scale=args.cte_mismatch_scale, init=args.init, tag=args.tag,
            study=args.study, t_root=args.t_root, t_hot=args.t_hot, r_hub=args.r_hub, output_freq=args.output_freq)
        _release()
