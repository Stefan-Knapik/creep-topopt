"""Sec. 4.1 (Fig. 3): one-year creep simulation of the elastic and creep-aware V* = 0.5 designs.
Writes results/1_creep_elastic/data_evaluation/creep_solution_for_{elastic,creep}_design.pkl."""

#%% Import modules

import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
ROOT = Path(__file__).resolve().parents[2] # repository root
data_dir = str(ROOT / "results" / "1_creep_elastic" / "data_evaluation")

import jax
jax.config.update("jax_enable_x64", True)

# Persistent JAX compilation cache - skips recompilation on repeat runs
_cache_dir = os.path.join(data_dir, 'jax_cache')
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import jax.numpy as np
import meshio
import pickle
import time
_t_script_start = time.perf_counter()

import logging
import toga
toga.logger.setLevel(logging.DEBUG)

from toga.mat import build_material_properties
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.physics.creep_implicit import CreepImplicit
from toga.utils import clean_data_dir

print(f"Clearing old results in {data_dir}...")
clean_data_dir(data_dir)

#%% Define materials and properties

material_names = ["dummy"]

material_properties = build_material_properties(
    material_names=material_names,
    property_configs={
        "E": {"model": "constant", "value": 160e3,}, # MPa
        "A": {"model": "constant", "value": 1e-21,}, # MPa^-n s^-1
        "n": {"model": "constant", "value": 3.5,},
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

class LoadedCreep(CreepImplicit):
    def get_surface_maps(self):
        def traction(u, x, *internal_vars_surface):
            return np.array([0., 100.])  # MPa
        return [traction]

problem = LoadedCreep(mesh, vec=2, dim=2, ele_type=ele_type,
                        dirichlet_bc_info=dirichlet_bc_info,
                        location_fns=location_fns,
                        additional_info=(material_properties,),)

from toga.opt.penalization import penalize
def make_penalize_A(exponent, ramp=8.0):
    return lambda x: 1 / penalize(x, {'RAMP': ramp})**exponent
problem.penalize_A = make_penalize_A(exponent=3.5, ramp=8.0)


#%% Define the simulation

solver_options = {'pardiso_solver': {}}

end_time = 3600 * 24 * 365 # 1 year in seconds
num_time_steps = 60

def fwd_pred(theta):
    full_sol, body_params = problem.forward_solve(
        theta=theta,
        end_time=end_time,
        num_time_steps=num_time_steps,
        fwd_solver_options=solver_options,
        vjp_solver_options=solver_options,
    )
    return full_sol[0], body_params

if __name__ == "__main__":
    mesh = meshio.read(ROOT / 'results/1_creep_elastic/data_elastic_50/vtk/sol_250.vtu')
    geom = mesh.cell_data.get('geom')[0]
    theta = geom.reshape(-1, 1)
    full_sol, body_params = fwd_pred(theta)
    with open(os.path.join(data_dir, 'creep_solution_for_elastic_design.pkl'), 'wb') as f:
        pickle.dump(full_sol, f)

    mesh = meshio.read(ROOT / 'results/2_pareto_vf/data_creep_50/vtk/sol_250.vtu')
    geom = mesh.cell_data.get('geom')[0]
    theta = geom.reshape(-1, 1)
    full_sol, body_params = fwd_pred(theta)
    with open(os.path.join(data_dir, 'creep_solution_for_creep_design.pkl'), 'wb') as f:
        pickle.dump(full_sol, f)
