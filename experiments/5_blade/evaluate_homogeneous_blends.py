"""Sec. 4.5 (Fig. 8, Table 2): creep compliance of uniform IN625/SS316 blends, IN625 fraction 0.0 ... 1.0.
Writes results/5_blade/data_evaluation_explicit/homo_design_{0,...,10}.pkl."""

#%% Import modules

import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2] # repository root
data_dir = str(ROOT / "results" / "5_blade" / "data_evaluation_explicit")

import jax
jax.config.update("jax_enable_x64", True)

# Persistent JAX compilation cache - skips recompilation on repeat runs
_cache_dir = os.path.join(data_dir, 'jax_cache')
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import jax.numpy as np
import numpy as onp
import meshio
import pickle
import time
_t_script_start = time.perf_counter()

import logging
import toga
toga.logger.setLevel(logging.DEBUG)

from toga.mat import build_material_properties
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh
from toga.physics.thermal import NonlinearThermal
from toga.physics.thermal_creep_explicit import ThermalCreepExplicit
from toga.fem.solver import ad_wrapper
from toga.utils import clean_data_dir, remove_orphan_nodes

print(f"Clearing old results in {data_dir}...")
clean_data_dir(data_dir)

#%% Define materials and properties

material_names = ["IN625", "SS316"]

material_properties = build_material_properties(
    material_names=material_names,
    property_configs={
        "k":     {"model": "rbf"},
        "E":     {"model": "rbf"},
        "logA":     {"model": "rbf", "temperature_dependent": False},
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
    f['cost'] = problem_u.compute_total_by_volume(params,
                                                  property="cost",
                                                  normalize_by_volume=True)

    aux['u_full'] = u_full
    aux['body_params'] = body_params
    aux['T'] = T

    return f, aux

if __name__ == "__main__":
    theta = np.ones((len(mesh.cells), len(material_names))) * 0.5 # initial guess: 50% IN625, 50% SS316

    for i in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        theta_homo = np.zeros_like(theta)
        theta_homo = theta_homo.at[:,0].set(i)   # IN625
        theta_homo = theta_homo.at[:,1].set(1-i) # SS316
        F, aux = evaluate_performance(theta_homo)
        with open(os.path.join(data_dir, f'homo_design_{int(i*10)}.pkl'), 'wb') as f:
            pickle.dump((F, aux), f)
        print(f"Data successfully saved to {data_dir}/homo_design_{int(i*10)}.pkl")
