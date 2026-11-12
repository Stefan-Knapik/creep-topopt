"""Sec. 4.2 (Fig. 4): elastic compliance C(0) and creep compliance J of the elastic-optimized
(experiments/1_creep_elastic) and creep-optimized designs at V* = 0.3 ... 0.8, under the creep
analysis of the creep optimizer (filtered density, 10 implicit steps).
Writes results/2_pareto_vf/pareto_comparison.csv."""

#%% Import modules

import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2] # repository root

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
import meshio

from toga.mat import build_material_properties
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.physics.creep_implicit import CreepImplicit
from toga.opt.filtering import compute_filter_kd_tree, applyDensityFilter

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


#%% Define the simulation (as in optimize_volume_fraction_sweep.py)

solver_options = {'pardiso_solver': {}}

end_time = 3600 * 24 * 365 # 1 year in seconds
num_time_steps = 10
rmin = 1.5

@jax.jit
def compliances(theta):
    full_sol, _ = problem.forward_solve(
        theta=theta,
        end_time=end_time,
        num_time_steps=num_time_steps,
        fwd_solver_options=solver_options,
        vjp_solver_options=solver_options,
    )
    compliance_0 = problem.compute_compliance(full_sol[0][0])
    compliance_final = problem.compute_compliance(full_sol[0][-1])
    return compliance_0, compliance_final - compliance_0

def last_history_J(run_dir):
    """Objective of the final iteration of an optimization run (5 significant digits)."""
    return onp.genfromtxt(ROOT / run_dir / 'history.csv', delimiter=',', names=True)['J'][-1]

def evaluate(run_dir):
    """C(0) and J of the final design of ``run_dir`` (highest-numbered vtk/sol_*.vtu)."""
    geom = meshio.read(max((ROOT / run_dir / 'vtk').glob('sol_*.vtu'))).cell_data.get('geom')[0]
    # The optimizers analyse the filtered density, so filter before solving.
    theta = applyDensityFilter(ft, np.asarray(geom, dtype=np.float64).reshape(-1, 1))
    return [float(x) for x in compliances(theta)]

if __name__ == "__main__":
    H, Hs = compute_filter_kd_tree(problem.fe, rmin)
    ft = {'H': H, 'Hs': Hs}

    volume_fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    out_path = ROOT / 'results' / '2_pareto_vf' / 'pareto_comparison.csv'
    rows = []
    for vf in volume_fractions:
        elastic_dir = f'results/1_creep_elastic/data_elastic_{round(vf*100)}'
        creep_dir = f'results/2_pareto_vf/data_creep_{round(vf*100)}'
        C0_e, J_e = evaluate(elastic_dir)
        C0_c, J_c = evaluate(creep_dir)
        rows.append((vf, C0_e, J_e, C0_c, J_c))
        # Each optimizer's own objective must be reproduced to the precision of its history.csv.
        C0_e_hist, J_c_hist = last_history_J(elastic_dir), last_history_J(creep_dir)
        print(f"V* = {vf:.1f}: elastic design C(0) = {C0_e:.8f}, J = {J_e:.8f}; "
              f"creep design C(0) = {C0_c:.8f}, J = {J_c:.8f}; "
              f"rel. diff. to history.csv: C(0) elastic {C0_e/C0_e_hist - 1:+.1e}, J creep {J_c/J_c_hist - 1:+.1e}")

    with open(out_path, 'w') as f:
        f.write('vf,C0_elastic_design,J_elastic_design,C0_creep_design,J_creep_design\n')
        for row in rows:
            f.write(f'{row[0]:.1f},' + ','.join(f'{x:.10g}' for x in row[1:]) + '\n')
    print(f"Saved {out_path}")
