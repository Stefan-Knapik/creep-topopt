"""Sec. 4.5 (Figs. 7-8, Table 2): IN625/SS316 compositional grading of a 3D turbine blade, swept over
the IN625 fraction V* = 0.1 ... 0.9. Writes results/5_blade/data_creep_{10,...,90}/."""

def run(vf):

    #%% Import modules

    import os
    from pathlib import Path
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
    # Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
    # embedding them as constants. Must be set before importing jax.
    os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    ROOT = Path(__file__).resolve().parents[2] # repository root
    data_dir = str(ROOT / "results" / "5_blade" / f'data_creep_{int(vf*100)}')

    import jax
    jax.config.update("jax_enable_x64", True)

    # Persistent JAX compilation cache - skips recompilation on repeat runs
    _cache_dir = os.path.join(data_dir, 'jax_cache')
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    import jax.numpy as np
    import numpy as onp
    import meshio
    import time
    _t_script_start = time.perf_counter()

    import logging
    import toga
    toga.logger.setLevel(logging.DEBUG)

    from toga import logger
    from toga.mat import build_material_properties
    from toga.mat.material_properties import vmap_design_to_physical
    from toga.fem.generate_mesh import get_meshio_cell_type, Mesh
    from toga.physics.thermal import NonlinearThermal
    from toga.physics.thermal_creep_explicit import ThermalCreepExplicit
    from toga.fem.solver import ad_wrapper
    from toga.opt.filtering import compute_filter_kd_tree
    from toga.opt import MMA
    from toga.utils import clean_data_dir, remove_orphan_nodes, quad_to_cell_average, save_vtu, timer, oom_hint

    print(f"Clearing old results in {data_dir}...")
    clean_data_dir(data_dir)

    #%% Define materials and properties

    material_names = ["IN625", "SS316"]

    material_properties = build_material_properties(
        material_names=material_names,
        property_configs={
            "k":     {"model": "rbf"},
            "E":     {"model": "rbf"},
            "logA":  {"model": "rbf", "temperature_dependent": False},
            "n":     {"model": "rbf", "temperature_dependent": False},
            "Q":     {"model": "rbf", "temperature_dependent": False},
            "cost":  {"model": "rbf", "temperature_dependent": False},
        },)

    #%% Define the mesh domain

    mesh_file = str(ROOT / "toga/mesh/blade/blade_050_000.inp")

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(mesh_file)
    compact_points, compact_cells, reorder_nodes = remove_orphan_nodes(meshio_mesh, cell_type, return_index_map=True)
    mesh = Mesh(compact_points, compact_cells)

    #%% Define the Thermal FEA boundary conditions

    bottom_node_indices = reorder_nodes(meshio_mesh.point_sets['bottom'])
    top_node_indices = reorder_nodes(meshio_mesh.point_sets['top'])

    def bottom(point, i): # returns true for nodes at the bottom of the blade (y=0mm)
        return np.isin(i, bottom_node_indices)

    def top(point, i): # returns true for nodes at the top of the blade (y=50mm)
        return np.isin(i, top_node_indices)

    cold_temp_value = 600. # temperature in Celsius for bottom
    hot_temp_value = 800. # temperature in Celsius for top

    def cold_temp(point):
        return cold_temp_value

    def hot_temp(point):
        return hot_temp_value

    dirichlet_bc_info_T = [[bottom, top], # callables that return true for nodes where BC is applied
                        [0,0],         # components/dimensions of the solution the BC is applied to
                        [cold_temp, hot_temp]] # callables that returns the value of the BC at the given point

    problem_T = NonlinearThermal(mesh, vec=1, dim=3, ele_type=ele_type,
                                dirichlet_bc_info=dirichlet_bc_info_T,
                                additional_info=(material_properties,),)

    # %% Define the Structural FEA boundary conditions

    # min by y, then min z, then min x
    node_idx_to_fix_xz = onp.lexsort((mesh.points[:,0],
                                    mesh.points[:,2],
                                    mesh.points[:,1]))[0]

    # min by y, then max z, then min x
    node_idx_to_fix_x = onp.lexsort(( mesh.points[:,0],
                                    -mesh.points[:,2],
                                    mesh.points[:,1]))[0]

    def point_xz(point, idx):
        return np.isin(idx, node_idx_to_fix_xz)

    def point_x(point, idx):
        return np.isin(idx, node_idx_to_fix_x)

    def zero_disp(point):
        return 0.

    # Dirichlet BCs - prescribe displacement
    dirichlet_bc_info_u = [[bottom]*3, # callables that return true for nodes where BC is applied
                        [0, 1, 2],  # components/dimensions of the solution the BC is applied to (x,y,z)
                        [zero_disp]*3] # callables that returns the value of the BC at the given point

    # Neumann BCs - prescribe forces (tractions)
    location_fns_u = [top]

    class LoadedThermalCreep(ThermalCreepExplicit):
        def get_surface_maps(self):
            def traction(u, x, *internal_vars_surface):
                return -np.array([0., 30., 0.])
            return [traction]
    # creep strain rate ~ proportional to traction magnitude ^ 5 since Norton n is about 5
    # 2x traction magnitude causes roughly 32x creep strain rate

    problem_u = LoadedThermalCreep(mesh, vec=3, dim=3, ele_type=ele_type,
                                dirichlet_bc_info=dirichlet_bc_info_u,
                                location_fns=location_fns_u,
                                additional_info=(material_properties,),)

    #%% Define the simulation

    solver_options = {'pardiso_solver': {}}

    fwd_pred_T = ad_wrapper(problem_T,
                            solver_options=solver_options,
                            adjoint_solver_options=solver_options)

    end_time = 3600 * 100 # 100 hours
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
            save_history=False, # body_params at the first and last step only (saves GPU memory)
        )
        return full_sol[0], body_params

    def fwd_pred(theta):
        T = fwd_pred_T(theta)[0]
        reference_temperature = cold_temp_value
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

    vf_constraint = vf

    def evaluate_performance(params, *args):
        f, aux = {}, {}

        params = np.concatenate((np.ones((len(params), 1)), params), axis=1)

        params_physical = params

        T, u_full, body_params = fwd_pred(params_physical)

        def compute_creep_compliance(u_full):
            sol_0 = u_full[0] # initial displacement field (num_nodes, vec)
            sol_final = u_full[-1] # final displacement field (num_nodes, vec)
            compliance_0 = problem_u.compute_compliance(sol_0)
            compliance_final = problem_u.compute_compliance(sol_final)
            creep_compliance = compliance_final - compliance_0
            return creep_compliance

        f['J'] = compute_creep_compliance(u_full)
        f['IN vf'] = problem_u.compute_volume_fraction_of_material(params, mat_idx=0) / vf_constraint - 1.0

        aux['u_full'] = u_full
        aux['body_params'] = body_params
        aux['T'] = T

        evaluate_performance.counter += 1
        return f, (f, aux)
    evaluate_performance.counter = 0

    evaluate_performance_grad = jax.jit(jax.jacrev(evaluate_performance, has_aux=True))

    #%% Define the output solution function

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

        u_full = aux['u_full']
        max_disp_0 = float(onp.max(onp.abs(u_full[0])))
        max_disp_final = float(onp.max(onp.abs(u_full[-1])))
        disp_log_path = os.path.join(data_path, 'displacement.csv')
        if i == 0:
            with open(disp_log_path, 'w') as df:
                df.write('iter,\tmax_disp_initial,\tmax_disp_final\n')
        with open(disp_log_path, 'a') as df:
            df.write(f'{i:4},\t{max_disp_0:.6g},\t{max_disp_final:.6g}\n')

        if i % optimization_params['output_freq'] == 0:
            logger.info(f"Saving solution for iteration {i:d}...")

            theta = onp.concatenate((onp.ones((len(theta), 1)), theta), axis=1)

            geom, mat = vmap_design_to_physical(theta)
            T = onp.asarray(aux['T'])
            u_full = aux['u_full']
            body_params = aux['body_params']

            vtu_path = os.path.join(data_path, f'vtk/sol_{i:03d}.vtu')
            os.makedirs(os.path.dirname(vtu_path), exist_ok=True)

            point_infos = [
                ('U_final', u_full[-1]),
                ('U_0', u_full[0]),
                ('T', T),
            ]

            cell_infos = [
                ('geom', geom),
                ('mat', mat),
                ('creep_strain_final', quad_to_cell_average(problem_u.fe, body_params[1][-1])),
                ('stress_final', quad_to_cell_average(problem_u.fe, body_params[0][-1])),
                ('stress_0', quad_to_cell_average(problem_u.fe, body_params[0][0])),
                ('dJ_dmat', dF['J'].reshape(-1, mat.shape[1])),
            ]

            save_vtu(problem_u.fe, vtu_path, cell_infos=cell_infos, point_infos=point_infos)

        output_sol.counter += 1
    output_sol.counter = 0

    # %% Define the optimization problem and solve it

    theta_ini = np.ones((len(mesh.cells), len(material_names))) * (1.0 / len(material_names))

    optimization_params = {
        'max_iter': int(os.environ.get('TOGA_MAX_ITER', 50)),
        'change_tol': 1e-5,
        'fun_tol': 1e-5,
        'move': 0.1,
        'rmin': 1.5,
        'output_freq': 1,
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
    volume_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for vf in volume_fractions:
        print(f"\n\nRunning optimization with volume fraction constraint vf <= {vf:.1f}...\n\n")
        run(vf)
        _release()
