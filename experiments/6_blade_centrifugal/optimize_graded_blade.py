"""Rotating blade: IN625/SS316 compositional grading under the centrifugal body force of the blade's
own mass (no tip traction), minimizing the creep compliance J = W_ext(u_end) - W_ext(u_0).

    python optimize_graded_blade.py                       # IN625 content sweep, V* = 0.1 ... 0.9
    python optimize_graded_blade.py --vf 0.3              # one run, written to data_creep_30/
    python optimize_graded_blade.py --mass 0.3            # mass <= m_SS316 + 0.3 (m_IN625 - m_SS316),
                                                          # written to data_creep_mass_30/
    python optimize_graded_blade.py --mass 0.3 --inertia 0.25
                                                          # in addition, moment of inertia about the rotation
                                                          # axis <= I_SS316 + 0.25 (I_IN625 - I_SS316),
                                                          # written to data_creep_mass_30_inertia_25/
    python optimize_graded_blade.py --inertia 0.3         # inertia bound only, data_creep_inertia_30/
    python optimize_graded_blade.py --coarse --vf 0.3     # on the coarse verification mesh, under coarse/
    python optimize_graded_blade.py --rpm 8500            # another rotation speed, written under rpm_8500/

Bounds on mass and inertia are given as fractions of the range between the all-SS316 and the
all-IN625 blade. Results go to results/6_blade_centrifugal/.
"""
import os
import sys
from pathlib import Path


def run_name(vf=None, mass=None, inertia=None):
    name = 'data_creep'
    if vf is not None:
        name += f'_{round(vf * 100)}'
    if mass is not None:
        name += f'_mass_{round(mass * 100)}'
    if inertia is not None:
        name += f'_inertia_{round(inertia * 100)}'
    return name


def run(vf=None, mass=None, inertia=None, coarse=False, rpm=None):

    #%% Import modules

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
    # Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
    # embedding them as constants. Must be set before importing jax.
    os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    ROOT = Path(__file__).resolve().parents[2] # repository root
    results_dir = ROOT / "results" / "6_blade_centrifugal"
    if rpm is not None and rpm != 7000.0:  # the reference speed, RPM in centrifugal_blade.py
        results_dir = results_dir / f"rpm_{round(rpm)}"
    data_dir = str(results_dir / ("coarse" if coarse else "") / run_name(vf, mass, inertia))

    import jax
    jax.config.update("jax_enable_x64", True)

    # Persistent JAX compilation cache - skips recompilation on repeat runs
    _cache_dir = os.path.join(data_dir, 'jax_cache')
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    import jax.numpy as np
    import numpy as onp
    import time
    _t_script_start = time.perf_counter()

    import logging
    import toga
    toga.logger.setLevel(logging.DEBUG)

    from toga import logger
    from toga.mat.material_properties import vmap_design_to_physical
    from toga.opt import MMA
    from toga.utils import clean_data_dir, quad_to_cell_average, save_vtu, timer, oom_hint

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import centrifugal_blade as cb

    print(f"Clearing old results in {data_dir}...")
    clean_data_dir(data_dir)

    #%% Thermal and creep problems of the rotating blade (centrifugal_blade.py)

    rpm = cb.RPM if rpm is None else rpm
    assert Path(data_dir).parent == cb.results_dir(rpm, coarse)
    blade = cb.build_blade(cb.coarse_mesh() if coarse else cb.MESH_FILE, omega=cb.rpm_to_omega(rpm))
    problem_u = blade.problem_u
    material_names = cb.MATERIAL_NAMES

    #%% Bounds on mass and moment of inertia, as fractions of the SS316 -> IN625 range

    def blend(fn, fraction):
        lo = float(fn(blade.full_theta(cb.homogeneous(blade.num_cells, 0.0))))
        hi = float(fn(blade.full_theta(cb.homogeneous(blade.num_cells, 1.0))))
        return lo + fraction * (hi - lo)

    mass_max = blend(problem_u.compute_mass, mass) if mass is not None else None            # g
    inertia_max = blend(problem_u.compute_inertia, inertia) if inertia is not None else None  # g mm^2
    logger.info(f"Bounds: V* = {vf}, mass <= {mass_max} g, inertia <= {inertia_max} g mm^2")

    #%% Defining the forward pass with objective and constraints

    def objective(params):
        f, aux = blade.evaluate(params)
        return f['J'], (f, aux)

    def constraints(params):
        theta = blade.full_theta(params)
        g = {}
        if vf is not None:
            g['IN vf'] = problem_u.compute_volume_fraction_of_material(theta, mat_idx=0) / vf - 1.0
        if mass is not None:
            g['mass'] = problem_u.compute_mass(theta) / mass_max - 1.0
        if inertia is not None:
            g['inertia'] = problem_u.compute_inertia(theta) / inertia_max - 1.0
        return g

    # One adjoint pass for J; the constraints are explicit in the composition.
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

    def save_solution(data_path, i, theta, dF, aux):
        logger.info(f"Saving solution for iteration {i:d}...")
        theta = onp.concatenate((onp.ones((len(theta), 1)), theta), axis=1)
        geom, mat = vmap_design_to_physical(theta)
        u_full = aux['u_full']
        body_params = aux['body_params']
        point_infos = [
            ('U_final', u_full[-1]),
            ('U_0', u_full[0]),
            ('T', onp.asarray(aux['T'])),
        ]
        cell_infos = [
            ('geom', geom),
            ('mat', mat),
            ('creep_strain_final', quad_to_cell_average(problem_u.fe, body_params[1][-1])),
            ('stress_final', quad_to_cell_average(problem_u.fe, body_params[0][-1])),
            ('stress_0', quad_to_cell_average(problem_u.fe, body_params[0][0])),
            ('dJ_dmat', dF['J'].reshape(-1, mat.shape[1])),
        ]
        save_vtu(problem_u.fe, os.path.join(data_path, f'vtk/sol_{i:03d}.vtu'),
                 cell_infos=cell_infos, point_infos=point_infos)

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

        # J, IN625 fraction, mass, inertia, stress and creep-strain scalars of every iterate
        scalars_path = os.path.join(data_path, 'scalars.csv')
        if i == 0:
            with open(scalars_path, 'w') as sf:
                sf.write('iter,' + ','.join(cb.SCALAR_KEYS) + '\n')
        with open(scalars_path, 'a') as sf:
            sf.write(f'{i},' + ','.join(f'{float(aux["scalars"][k]):.10g}' for k in cb.SCALAR_KEYS) + '\n')

        output_sol.saved = i % optimization_params['output_freq'] == 0
        if output_sol.saved:
            save_solution(data_path, i, theta, dF, aux)

        output_sol.counter += 1
    output_sol.counter = 0
    output_sol.saved = False

    # %% Define the optimization problem and solve it

    theta_ini = np.ones((blade.num_cells, len(material_names))) * (1.0 / len(material_names))

    optimization_params = {
        'max_iter': int(os.environ.get('TOGA_MAX_ITER', 50)),
        'change_tol': 1e-5,
        'fun_tol': 1e-5,
        'move': 0.1,
        'output_freq': 10, # .vtu every 10th iteration, and always of the last one
    }

    def optimize(fe, theta_ini, optimization_params, data_sub_dir=None):
        data_folder = os.path.join(data_dir, data_sub_dir) if data_sub_dir else data_dir

        # The composition field is not filtered: the geometry is fixed and intermediate
        # compositions are physical blends, so there is no checkerboarding to suppress.

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

                # The optimizer sees J scaled to 10 at the initial design. Under centrifugal loading J is
                # of order 0.1 mJ, and MMA weighs the objective against the constraints in absolute terms.
                if J_scale is None:
                    J_scale = 10.0 / float(F['J'])
                J = F['J'].reshape(1,) * J_scale
                dJ = dF['J'].reshape(-1) * J_scale
                vc = np.array([F[key] for key in F if key != 'J']).reshape(-1)
                dvc = np.array([dF[key] for key in dF if key != 'J']).reshape(len(vc), -1)
                logger.debug(f"J = {F['J']}; vc = {vc.ravel()}")
                logger.debug(f"dJ.shape = {dJ.shape}    dJ.max = {dJ.max()}    dJ.min = {dJ.min()}")
                logger.debug(f"dvc.shape = {dvc.shape}    dvc.max = {dvc.max()}    dvc.min = {dvc.min()}")

                with timer() as t_save:
                    output_sol(data_folder, theta, F, dF, aux)
                last = (loop, theta, dF, aux)

                with timer() as t_optimizer:
                    optimizer.iter(onp.array(J), onp.array(dJ), onp.array(vc), onp.array(dvc))
                    theta = np.array(optimizer.desvars.reshape(theta.shape))

            with open(walltime_log_path, 'a') as wf:
                wf.write(f'iter_{loop},\t{t_eval:.3f},\t{t_save:.3f},\t{t_optimizer:.3f},\t{t_iter:.3f}\n')

            logger.info(f'MMA iter {loop:d}; J={float(F["J"]):.5g}; vc={vc.ravel()}; '
                        f'change={optimizer.change:.3e}; change_f={optimizer.change_f:.3e}')
            loop += 1

        if not output_sol.saved: # the last analysed design
            save_solution(data_folder, *last)

    optimize(problem_u.fe, theta_ini, optimization_params)

def _release():
    """Drop the finished run's compiled executables and assembly caches before the next run.
    Every run() builds a new Problem whose jitted closures JAX would otherwise keep alive."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vf", type=float, nargs="+", default=None,
                        help="upper bounds on the IN625 volume fraction (default sweep: 0.1 ... 0.9)")
    parser.add_argument("--mass", type=float, nargs="+", default=None,
                        help="upper bounds on the mass, as fractions of the SS316 -> IN625 range")
    parser.add_argument("--inertia", type=float, default=None,
                        help="upper bound on the moment of inertia about the rotation axis, as a fraction "
                             "of the SS316 -> IN625 range; added to every run, or alone if no other bound is given")
    parser.add_argument("--coarse", action="store_true", help="use the coarse verification mesh")
    parser.add_argument("--rpm", type=float, default=None, help="rotation speed (default: 7000 rpm)")
    args = parser.parse_args()

    if args.vf is None and args.mass is None and args.inertia is None:
        args.vf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    runs = [dict(vf=vf) for vf in args.vf or []] + [dict(mass=m) for m in args.mass or []] or [dict()]

    for bounds in runs:
        print(f"\n\nRunning optimization with bounds {bounds}, inertia {args.inertia}...\n\n")
        run(**bounds, inertia=args.inertia, coarse=args.coarse, rpm=args.rpm)
        _release()
