"""Sec. 4.3 (Fig. 5): thermo-structural creep optimization at V* = 0.3 under cold, hot and cold-to-hot
temperature boundary conditions. Writes results/3_thermal_creep/data_creep_{cold,hot,cold_hot}_0.3_flex/."""

def run(cold_temp, hot_temp, vf, case_name=None):

    #%% Import modules

    import os
    from pathlib import Path
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
    # Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
    # embedding them as constants. Must be set before importing jax.
    os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    ROOT = Path(__file__).resolve().parents[2] # repository root
    data_dir = str(ROOT / "results" / "3_thermal_creep" / f'data_creep_{case_name}_flex')

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
    from toga.mat import build_material_properties
    from toga.mat.material_properties import vmap_design_to_physical
    from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
    from toga.physics.thermal import Thermal
    from toga.physics.thermal_creep_implicit import ThermalCreepImplicit
    from toga.fem.solver import ad_wrapper
    from toga.opt.filtering import compute_filter_kd_tree, applyDensityFilter
    from toga.opt import MMA
    from toga.utils import clean_data_dir, quad_to_cell_average, save_vtu, timer, oom_hint

    print(f"Clearing old results in {data_dir}...")
    clean_data_dir(data_dir)

    #%% Define materials and properties

    material_names = ["dummy"]

    material_properties = build_material_properties(
        material_names=material_names,
        property_configs={
            "k": {"model": "constant", "value": 1.0,},
            "E": {"model": "constant", "value": 160e3,}, # MPa
            "A": {"model": "constant", "value": 1e-21,}, # MPa^-n s^-1
            "n": {"model": "constant", "value": 3.5,},
            "Q": {"model": "constant", "value": 1e4,},
        },)

    #%% Define the mesh domain

    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    N = 100
    Nx, Ny = 2*N, N
    Lx, Ly = 200., 100.
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type=ele_type)

    # %% Define the Structural FEA boundary conditions

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def dirichlet_zero(point):
        return 0.

    def dirichlet_cold(point):
        return cold_temp

    def dirichlet_hot(point):
        return hot_temp

    dirichlet_bc_info_T = [[left, right],
                           [0, 0],
                           [dirichlet_cold, dirichlet_hot]]

    dirichlet_bc_info_u = [[left, left, right, right],
                           [0, 1, 0, 1],
                           [dirichlet_zero]*4]

    def load_location(point):
        on_bottom = np.isclose(point[1], 0., atol=1e-5)
        return on_bottom

    location_fns = [load_location]

    class LoadedThermalCreep(ThermalCreepImplicit):
        def get_surface_maps(self):
            def traction(u, x, *internal_vars_surface):
                return np.array([0., 100.])
            return [traction]

    problem_u = LoadedThermalCreep(
        mesh,
        vec=2,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info_u,
        location_fns=location_fns,
        additional_info=(material_properties,),
    )

    problem_T = Thermal(
        mesh,
        vec=1,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info_T,
        additional_info=(material_properties,),
    )

    from toga.opt.penalization import penalize
    def make_penalize_A(exponent, ramp=8.0):
        return lambda x: 1 / penalize(x, {'RAMP': ramp})**exponent
    problem_u.penalize_A = make_penalize_A(exponent=3.5, ramp=8.0)

    #%% Define the simulation

    solver_options = {'pardiso_solver': {}}
    fwd_pred_T = ad_wrapper(problem_T,
                            solver_options=solver_options,
                            adjoint_solver_options=solver_options)

    end_time = 3600 * 24 * 365 # 1 year in seconds
    num_time_steps = 10

    def fwd_pred_u(theta, temperature, reference_temperature):
        full_sol, body_params = problem_u.forward_solve(
            theta=theta,
            T=temperature,
            reference_T=reference_temperature,
            end_time=end_time,
            num_time_steps=num_time_steps,
            fwd_solver_options=solver_options,
            vjp_solver_options=solver_options,
        )
        return full_sol[0], body_params

    def fwd_pred(theta):
        T = fwd_pred_T(theta)[0]
        reference_temperature = 300.0 - 273.15 # 300 K, in °C (see the note in __main__)
        u_full, body_params = fwd_pred_u(theta, T, reference_temperature)
        # T is a jaxArray(num_nodes,1)
        # u_full is a jaxArray(num_time_steps + 1, num_nodes, vec)
        # body_params is a tuple of len==5, (sigmas, eps_creep, theta, T_quad, ref_T_quad),
        #    shapes((time+1, cells, quads, vec, dim),
        #           (time+1, cells, quads, vec, dim),
        #           (time+1, cells, quads, num_mat+1),
        #           (time+1, cells, quads, 1),
        #           (time+1, cells, quads, ))
        return T, u_full, body_params

    #%% Defining the forward pass with objective and constraints

    vf_constraint_max = vf

    def evaluate_performance(params, *args):
        ft = args[0]
        f, aux = {}, {}

        geom, mat = vmap_design_to_physical(params)
        geom = geom.at[0::Ny].set(1.0) # bottom edge solid
        geom_smoothed = applyDensityFilter(ft, geom.reshape(-1, 1))
        params_physical = np.concatenate((geom_smoothed.reshape(-1, 1), mat), axis=1)

        temperature, u_full, body_params = fwd_pred(params_physical)

        def compute_creep_compliance(u_full):
            sol_0 = u_full[0] # initial displacement field (num_nodes, vec)
            sol_final = u_full[-1] # final displacement field (num_nodes, vec)
            compliance_0 = problem_u.compute_compliance(sol_0)
            compliance_final = problem_u.compute_compliance(sol_final)
            creep_compliance = compliance_final - compliance_0
            return creep_compliance

        f['J'] = compute_creep_compliance(u_full)
        f['vf'] = problem_u.compute_volume_fraction(params) / vf_constraint_max - 1.0

        aux['u_full'] = u_full
        aux['body_params'] = body_params
        aux['temperature'] = temperature
        aux['reference_temperature'] = cold_temp

        evaluate_performance.counter += 1
        return f, (f, aux)
    evaluate_performance.counter = 0

    evaluate_performance_grad = jax.jit(jax.jacrev(evaluate_performance, has_aux=True))

    #%% Define the output solution function

    def output_sol(data_path, theta, F, dF, aux):
        theta = theta.at[0::Ny].set(1.0) # bottom edge solid
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

        u_full = aux['u_full']
        max_disp_0 = float(onp.max(onp.abs(u_full[0])))
        max_disp_final = float(onp.max(onp.abs(u_full[-1])))
        disp_log_path = os.path.join(data_path, 'displacement.csv')
        if i == 0:
            with open(disp_log_path, 'w') as df:
                df.write('iter,\tmax_disp_initial,\tmax_disp_final\n')
        with open(disp_log_path, 'a') as df:
            df.write(f'{i:4},\t{max_disp_0:.6g},\t{max_disp_final:.6g}\n')

        if i % optimization_params['save_every_n_iters'] == 0 or i == 0:
            logger.info(f"Saving solution for iteration {i:d}...")

            geom, mat = vmap_design_to_physical(theta)
            u_full = aux['u_full']
            body_params = aux['body_params']

            vtu_path = os.path.join(data_path, f'vtk/sol_{i:03d}.vtu')
            os.makedirs(os.path.dirname(vtu_path), exist_ok=True)

            point_infos = [
                ('U_final', u_full[-1]),
                ('U_0', u_full[0]),
                ('temperature', aux['temperature']),
            ]

            cell_infos = [
                ('geom', geom),
                ('creep_strain_final', quad_to_cell_average(problem_u.fe, body_params[1][-1])),
                ('stress_final', quad_to_cell_average(problem_u.fe, body_params[0][-1])),
                ('stress_0', quad_to_cell_average(problem_u.fe, body_params[0][0])),
            ]

            save_vtu(problem_u.fe, vtu_path, cell_infos=cell_infos, point_infos=point_infos)

        output_sol.counter += 1
    output_sol.counter = 0

    # %% Define the optimization problem and solve it

    theta_ini = np.ones((len(mesh.cells), 1))

    optimization_params = {
        'max_iter': int(os.environ.get('TOGA_MAX_ITER', 250)),
        'change_tol': 1e-5,
        'fun_tol': 1e-5,
        'move': 0.1,
        'rmin': 1.5,
        'save_every_n_iters': 1,
    }

    def optimize(fe, theta_ini, optimization_params, data_sub_dir=None):
        data_folder = os.path.join(data_dir, data_sub_dir) if data_sub_dir else data_dir

        H, Hs = compute_filter_kd_tree(fe, optimization_params['rmin'])
        ft = {'H': H, 'Hs': Hs}

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

        os.makedirs(data_folder, exist_ok=True)
        walltime_log_path = os.path.join(data_folder, 'walltime.csv')
        with open(walltime_log_path, 'w') as wf:
            wf.write('phase,\tt_eval,\tt_save,\tt_opt,\tt_iter\n')
            wf.write(f'setup,\t{time.perf_counter() - _t_script_start:.3f},,,\n')

        while not optimizer.converged():
            with timer() as t_iter:
                with timer() as t_eval, oom_hint():
                    dF, (F, aux) = evaluate_performance_grad(theta, ft)

                J = F['J'].reshape(1,)
                dJ = dF['J'].reshape(-1)
                vc = np.array([F[key] for key in F if key != 'J']).reshape(-1)
                dvc = np.array([dF[key] for key in dF if key != 'J']).reshape(len(vc), -1)
                logger.debug(f"J = {J}; vc = {vc.ravel()}")
                logger.debug(f"dJ.shape = {dJ.shape}    dJ.max = {dJ.max()}    dJ.min = {dJ.min()}")
                logger.debug(f"dvc.shape = {dvc.shape}    dvc.max = {dvc.max()}    dvc.min = {dvc.min()}")

                with timer() as t_save:
                    output_sol(data_folder, theta, F, dF, aux)

                with timer() as t_optimizer:
                    optimizer.iter(onp.array(J), onp.array(dJ), onp.array(vc), onp.array(dvc))
                    theta = np.array(optimizer.desvars.reshape(theta.shape))

            with open(walltime_log_path, 'a') as wf:
                wf.write(f'iter_{loop},\t{t_eval:.3f},\t{t_save:.3f},\t{t_optimizer:.3f},\t{t_iter:.3f}\n')

            logger.info(f'MMA iter {loop:d}; J={float(J[0]):.5g}; vc={vc.ravel()}; '
                        f'change={optimizer.change:.3e}; change_f={optimizer.change_f:.3e}')
            loop += 1

    optimize(problem_u.fe, theta_ini, optimization_params)

def _release():
    """Drop the finished run's compiled executables and assembly caches before the next run.
    Every run() builds a new Problem whose jitted closures JAX would otherwise keep alive."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()

if __name__ == "__main__":
    # Paper temperatures are in kelvin (300 K / 1100 K); toga's thermal creep takes °C
    # (it adds 273.15 inside the Arrhenius term), so every absolute temperature is shifted.
    K = 273.15
    run(cold_temp=300.0 - K,  hot_temp=300.0 - K,  vf=0.3, case_name='cold_0.3')
    _release()
    run(cold_temp=1100.0 - K, hot_temp=1100.0 - K, vf=0.3, case_name='hot_0.3')
    _release()
    run(cold_temp=300.0 - K,  hot_temp=1100.0 - K, vf=0.3, case_name='cold_hot_0.3')
    _release()
