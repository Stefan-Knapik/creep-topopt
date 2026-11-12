"""Sec. 4.3 (Fig. 6): effect of temperature-dependent E and k on the cold-to-hot design. The design optimized with
constant properties and the one optimized with E(T), k(T) of IN625 are each analysed once under both
property models, and their density and temperature fields are compared.
Each design is analysed as its optimizer analysed it: stored design variables -> density filter -> solver.
Reads results/3_thermal_creep/data_creep_cold_hot{,_tdep}_0.3_flex/, writes
results/3_thermal_creep/tdep_comparison.json."""

#%% Import modules

import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2] # repository root
data_dir = str(ROOT / "results" / "3_thermal_creep" / "data_evaluation_tdep")

import jax
jax.config.update("jax_enable_x64", True)

# Persistent JAX compilation cache - skips recompilation on repeat runs
_cache_dir = os.path.join(data_dir, 'jax_cache')
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import jax.numpy as np
import numpy as onp
import json
import meshio

from toga.mat import build_material_properties
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.physics.thermal import NonlinearThermal, Thermal
from toga.physics.thermal_creep_implicit import ThermalCreepImplicit
from toga.fem.solver import ad_wrapper
from toga.opt.filtering import compute_filter_kd_tree, applyDensityFilter
from toga.opt.penalization import penalize
from toga.utils import clean_data_dir

print(f"Clearing old results in {data_dir}...")
clean_data_dir(data_dir)

#%% Define the mesh domain and boundary conditions (as in optimize_thermal_boundary_conditions.py)

K = 273.15
cold_temp, hot_temp = 300.0 - K, 1100.0 - K # °C

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
N = 100
Nx, Ny = 2*N, N
Lx, Ly = 200., 100.
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type=ele_type)

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
    return np.isclose(point[1], 0., atol=1e-5)

class LoadedThermalCreep(ThermalCreepImplicit):
    def get_surface_maps(self):
        def traction(u, x, *internal_vars_surface):
            return np.array([0., 100.])
        return [traction]

solver_options = {'pardiso_solver': {}}
end_time = 3600 * 24 * 365 # 1 year in seconds
num_time_steps = 10
rmin = 1.5

#%% Define the two property models

def build_model(temperature_dependent):
    """Return ``analyse(rho)`` for the filtered density ``rho`` and the E model, as in the optimization."""
    property_configs = {
        "k": {"model": "constant", "value": 0.01,}, # W/(mm K)
        "E": {"model": "constant", "value": 160e3,}, # MPa
        "A": {"model": "constant", "value": 1e-21,}, # MPa^-n s^-1
        "n": {"model": "constant", "value": 3.5,},
        "Q": {"model": "constant", "value": 1e4,},
    }
    if temperature_dependent: # E(T) and k(T) piecewise linear between the tabulated rows of IN625
        property_configs["k"] = {"model": "table"}
        property_configs["E"] = {"model": "table"}

    material_properties = build_material_properties(
        material_names=["IN625"] if temperature_dependent else ["dummy"],
        property_configs=property_configs,)

    problem_u = LoadedThermalCreep(mesh, vec=2, dim=2, ele_type=ele_type,
                                   dirichlet_bc_info=dirichlet_bc_info_u,
                                   location_fns=[load_location],
                                   additional_info=(material_properties,),)
    problem_u.penalize_A = lambda x: 1 / penalize(x, {'RAMP': 8.0})**3.5

    problem_T = (NonlinearThermal if temperature_dependent else Thermal)(
        mesh, vec=1, dim=2, ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info_T,
        additional_info=(material_properties,),)

    fwd_pred_T = ad_wrapper(problem_T, solver_options=solver_options, adjoint_solver_options=solver_options)

    def analyse(rho):
        theta = rho.reshape(-1, 1)
        if temperature_dependent: # the table models take [composition, T]; one alloy, so composition = 1
            theta = np.concatenate((theta, np.ones_like(theta)), axis=1)
        T = fwd_pred_T(theta)[0]
        full_sol, _ = problem_u.forward_solve(
            theta=theta, T=T, reference_T=300.0 - K,
            end_time=end_time, num_time_steps=num_time_steps,
            fwd_solver_options=solver_options, vjp_solver_options=solver_options,)
        u_full = full_sol[0]
        J = problem_u.compute_compliance(u_full[-1]) - problem_u.compute_compliance(u_full[0])
        return float(J), onp.asarray(T).reshape(-1)

    return analyse, material_properties.E, problem_u.fe

#%% Load the two designs

def load_filtered_density(run_name, ft):
    """Filtered density of the final design of a run (its highest-numbered vtk/sol_*.vtu)."""
    run_dir = ROOT / "results" / "3_thermal_creep" / run_name
    geom = meshio.read(max((run_dir / 'vtk').glob('sol_*.vtu'))).cell_data['geom'][0] # raw design variables
    rho = applyDensityFilter(ft, np.asarray(geom, dtype=np.float64).reshape(-1, 1)).reshape(-1)
    J_final = float(onp.genfromtxt(run_dir / 'history.csv', delimiter=',', names=True)['J'][-1])
    return rho, J_final

def region_means(rho, x):
    """Mean filtered density in the cold/hot halves and in the four quarters along x."""
    half = [float(rho[(x >= a) & (x < a + Lx/2)].mean()) for a in (0., Lx/2)]
    quarter = [float(rho[(x >= a) & (x < a + Lx/4)].mean()) for a in (0., Lx/4, Lx/2, 3*Lx/4)]
    return {'halves_cold_to_hot': half, 'quarters_cold_to_hot': quarter}

if __name__ == "__main__":
    models = {'const': build_model(False), 'tdep': build_model(True)}
    fe = models['const'][2]

    H, Hs = compute_filter_kd_tree(fe, rmin)
    ft = {'H': H, 'Hs': Hs}

    runs = {'const': 'data_creep_cold_hot_0.3_flex', 'tdep': 'data_creep_cold_hot_tdep_0.3_flex'}
    rho, out = {}, {'J_optimizer_final': {}, 'J': {}, 'density': {}, 'temperature_K': {}, 'E_MPa': {}}
    for design, run_name in runs.items():
        rho[design], out['J_optimizer_final'][design] = load_filtered_density(run_name, ft)

    # J and temperature of each design under each property model
    T = {}
    for design in runs:
        for model, (analyse, _, _) in models.items():
            J, T[design, model] = analyse(rho[design])
            out['J'][f'{design}_design_under_{model}_model'] = J

    # Density fields
    x_cell = onp.asarray(fe.points)[onp.asarray(fe.cells), 0].mean(axis=1)
    rho = {design: onp.asarray(r) for design, r in rho.items()}
    diff = onp.abs(rho['tdep'] - rho['const'])
    n_solid = round(0.3 * diff.size) # the densest 30% of cells, as the figures threshold the designs
    solid = {design: onp.argsort(-r, kind='stable')[:n_solid] for design, r in rho.items()}
    out['density'] = {
        'mean_abs_diff': float(diff.mean()),
        'max_abs_diff': float(diff.max()),
        'fraction_diff_gt_0.5': float((diff > 0.5).mean()),
        'fraction_diff_gt_0.1': float((diff > 0.1).mean()),
        'fraction_solid_cells_shared': float(len(onp.intersect1d(solid['const'], solid['tdep'])) / n_solid),
        'mean_const': {'all': float(rho['const'].mean()), **region_means(rho['const'], x_cell)},
        'mean_tdep': {'all': float(rho['tdep'].mean()), **region_means(rho['tdep'], x_cell)},
    }

    # Temperature fields (nodal). 'solid' restricts to the nodes of cells with filtered density > 0.5.
    cells = onp.asarray(fe.cells)
    for design in runs:
        dT = onp.abs(T[design, 'tdep'] - T[design, 'const'])
        solid_nodes = onp.unique(cells[rho[design] > 0.5])
        out['temperature_K'][f'{design}_design'] = {
            'max_abs_diff_tdep_vs_const_model': float(dT.max()),
            'mean_abs_diff_tdep_vs_const_model': float(dT.mean()),
            'max_abs_diff_tdep_vs_const_model_solid': float(dT[solid_nodes].max()),
            'mean_abs_diff_tdep_vs_const_model_solid': float(dT[solid_nodes].mean()),
        }
    out['temperature_K']['max_abs_diff_between_designs_own_model'] = float(
        onp.abs(T['tdep', 'tdep'] - T['const', 'const']).max())

    # E(T) over the domain, at the nodal temperatures of each design under the E(T), k(T) model
    model_E = models['tdep'][1]
    for design in runs:
        E = onp.asarray(model_E(np.stack((np.ones(len(T[design, 'tdep'])), np.asarray(T[design, 'tdep'])), axis=1)))
        out['E_MPa'][f'{design}_design'] = {'min': float(E.min()), 'max': float(E.max())}
    out['E_MPa']['constant_model'] = 160e3

    print(json.dumps(out, indent=2))
    with open(ROOT / "results" / "3_thermal_creep" / "tdep_comparison.json", 'w') as f:
        json.dump(out, f, indent=2)
