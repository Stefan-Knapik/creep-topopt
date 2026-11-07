"""Sec. 4.1 (Fig. 3): elastic-compliance baseline design at V* = 0.5, compared against the
creep-aware V* = 0.5 design from experiments/2_pareto_vf. Writes results/1_creep_elastic/data_elastic_50/."""

#%% Import modules

import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
ROOT = Path(__file__).resolve().parents[2] # repository root
data_dir = str(ROOT / "results" / "1_creep_elastic" / "data_elastic_50")

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
from toga.physics.linear_elasticity import Elasticity
from toga.fem.solver import ad_wrapper
from toga.opt.filtering import compute_filter_kd_tree, applyDensityFilter
from toga.opt import MMA
from toga.utils import clean_data_dir, save_vtu, timer, oom_hint

print(f"Clearing old results in {data_dir}...")
clean_data_dir(data_dir)

#%% Define materials and properties

material_names = ["dummy"]

material_properties = build_material_properties(
    material_names=material_names,
    property_configs={
        "E": {"model": "constant", "value": 160e3,}, # MPa
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

def dirichlet_zero(point):
    return 0.

dirichlet_bc_info = [[left]*2,
                     [0, 1],
                     [dirichlet_zero]*2]

def load_location(point):
    return np.logical_and(
        np.isclose(point[0], Lx, atol=1e-5),
        np.isclose(point[1], 0., atol=0.2*Ly + 1e-5))  # lower 20% of the right edge

location_fns = [load_location]


class LoadedElasticity(Elasticity):
    def get_surface_maps(self):
        def traction(u, x, *internal_vars_surface):
            return np.array([0., 100.])  # MPa
        return [traction]

problem = LoadedElasticity(mesh, vec=2, dim=2, ele_type=ele_type,
                        dirichlet_bc_info=dirichlet_bc_info,
                        location_fns=location_fns,
                        additional_info=(material_properties,),)


#%% Define the simulation

solver_options = {'pardiso_solver': {}}

fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

#%% Defining the forward pass with objective and constraints

vf_constraint_max = 0.5

def evaluate_performance(params, *args):
    ft = args[0]
    f, aux = {}, {}

    geom, mat = vmap_design_to_physical(params)
    geom_smoothed = applyDensityFilter(ft, geom.reshape(-1, 1))
    params_physical = np.concatenate((geom_smoothed.reshape(-1, 1), mat), axis=1)

    u_list = fwd_pred(params_physical)

    f['J'] = problem.compute_compliance(u_list[0])
    f['vf'] = problem.compute_volume_fraction(params) / vf_constraint_max - 1.0

    aux['u_list'] = u_list

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

    if i % optimization_params['save_every_n_iters'] == 0 or i == 0:
        logger.info(f"Saving solution for iteration {i:d}...")

        geom, mat = vmap_design_to_physical(theta)
        u_list = aux['u_list']

        vtu_path = os.path.join(data_path, f'vtk/sol_{i:03d}.vtu')
        os.makedirs(os.path.dirname(vtu_path), exist_ok=True)

        point_infos = [
            ('U', u_list[0]),
        ]

        cell_infos = [
            ('geom', geom),
        ]

        save_vtu(problem.fe, vtu_path, cell_infos=cell_infos, point_infos=point_infos)

    output_sol.counter += 1
output_sol.counter = 0


# %% Define the optimization problem and solve it

theta_ini = np.ones((len(mesh.cells), 1)) * vf_constraint_max

optimization_params = {
    'max_iter': 250,
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

if __name__ == "__main__":
    optimize(problem.fe, theta_ini, optimization_params)
