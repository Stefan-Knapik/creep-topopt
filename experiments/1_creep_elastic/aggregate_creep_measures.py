"""Sec. 4.1 (Table 1): aggregate creep measures of the elastic and creep-aware V* = 0.5 designs, and the
energy balance of the dead load, J = D + dU, that links creep compliance to creep dissipation.
Each design is analysed on its filtered density. Plane stress, thickness 1 mm; energies in mJ.
Writes results/1_creep_elastic/creep_measures.csv."""

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
import meshio

from toga.mat import build_material_properties
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.physics.creep_implicit import CreepImplicit
from toga.opt.filtering import compute_filter_kd_tree, applyDensityFilter
from toga.utils import quad_to_cell_average

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
rmin = 1.5
JxW = problem.fe.JxW # (num_cells, num_quads), mm^2; thickness 1 mm

def equivalent_strain(eps):
    """Von Mises equivalent of an in-plane creep strain (..., 2, 2); incompressible, e33 = -(e11 + e22)."""
    e33 = -(eps[..., 0, 0] + eps[..., 1, 1])
    return np.sqrt(2. / 3. * (np.sum(eps * eps, axis=(-2, -1)) + e33**2))

def measures(theta, num_time_steps):
    full_sol, body_params = problem.forward_solve(
        theta=theta,
        end_time=end_time,
        num_time_steps=num_time_steps,
        fwd_solver_options=solver_options,
        vjp_solver_options=solver_options,
    )
    sols = full_sol[0]                             # (num_steps+1, num_nodes, 2); step 0 is elastic
    sigmas, eps_creep = body_params[0], body_params[1]  # (num_steps+1, num_cells, num_quads, 2, 2)
    integrate = lambda x: np.sum(x * JxW, axis=(-2, -1))  # over the domain
    ddot = lambda a, b: np.sum(a * b, axis=(-2, -1))

    # Plane stress: sigma_33 = 0, so the out-of-plane strains do no work.
    eps = jax.vmap(problem.fe.sol_to_grad)(sols)
    eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))
    eps_el = eps - eps_creep

    compliance = jax.vmap(problem.compute_compliance)(sols)
    U = 0.5 * integrate(ddot(sigmas, eps_el))
    d_eps_creep = np.diff(eps_creep, axis=0)
    d_eps_el = np.diff(eps_el, axis=0)
    d_sigmas = np.diff(sigmas, axis=0)
    # Creep dissipation, sum over steps of the integral of sigma : d(eps_cr).
    # Backward Euler (end-of-step stress, as in the constitutive update) and trapezoidal (mean stress).
    D_be_cells = np.sum(ddot(sigmas[1:], d_eps_creep) * JxW, axis=(0, -1))
    D_trap = np.sum(integrate(ddot(0.5 * (sigmas[1:] + sigmas[:-1]), d_eps_creep)))
    # J - D_be - dU for a linear-elastic stress: 1/2 sum of the integral of d(sigma) : d(eps_el).
    R_be = 0.5 * np.sum(integrate(ddot(d_sigmas, d_eps_el)))
    # Internal work of the creep steps, sum of the integral of sigma_s : d(eps)_s; equals J at equilibrium.
    W_int = np.sum(integrate(ddot(sigmas[1:], np.diff(eps, axis=0))))

    rho = theta[:, 0]
    solid = rho > 0.5
    vol = np.sum(JxW, axis=-1)
    eq = equivalent_strain(eps_creep[-1])           # final, at the quadrature points
    eq_cells = np.sum(eq * JxW, axis=-1)
    # Accumulated equivalent creep strain, sum over steps of the equivalent strain increment.
    eq_acc_cells = np.sum(equivalent_strain(d_eps_creep) * JxW, axis=(0, -1))
    # Peak over the solid cells of the equivalent strain of the cell-averaged tensor
    # (as visualization/1_creep_elastic_stress_strain.ipynb).
    eq_cell_avg = equivalent_strain(quad_to_cell_average(problem.fe, eps_creep[-1]))

    return {
        'C0': compliance[0],
        'J': compliance[-1] - compliance[0],
        'D_be': np.sum(D_be_cells),
        'D_trap': D_trap,
        'dU': U[-1] - U[0],
        'U0': U[0],
        'R_be': R_be,
        'W_int': W_int,
        'D_be_solid': np.sum(np.where(solid, D_be_cells, 0.)),
        'eq_int': np.sum(eq_cells),
        'eq_int_rho': np.sum(rho * eq_cells),
        'eq_avg_rho': np.sum(rho * eq_cells) / np.sum(rho * vol),
        'eq_avg_solid': np.sum(np.where(solid, eq_cells, 0.)) / np.sum(np.where(solid, vol, 0.)),
        'eq_acc_int_rho': np.sum(rho * eq_acc_cells),
        'eq_peak_solid': np.max(np.where(solid, eq_cell_avg, 0.)),
    }

designs = {
    'elastic_50': 'results/1_creep_elastic/data_elastic_50',
    'creep_50': 'results/2_pareto_vf/data_creep_50',
}

# 10 steps as in the optimizations, 60 as in evaluate_designs_under_creep.py, 240 for the trend.
time_steps = [10, 60, 240]

if __name__ == "__main__":
    H, Hs = compute_filter_kd_tree(problem.fe, rmin)
    ft = {'H': H, 'Hs': Hs}

    rows = []
    for tag, run_dir in designs.items():
        geom = meshio.read(max((ROOT / run_dir / 'vtk').glob('sol_*.vtu'))).cell_data.get('geom')[0]
        # The optimizers analyse the filtered density, so filter before solving.
        theta = applyDensityFilter(ft, np.asarray(geom, dtype=np.float64).reshape(-1, 1))
        for num_time_steps in time_steps:
            m = {k: float(v) for k, v in measures(theta, num_time_steps).items()}
            # Energy balance of the dead load: J = D + dU. Relative residuals for both forms of D.
            m['res_be'] = (m['J'] - m['D_be'] - m['dU']) / m['J']
            m['res_trap'] = (m['J'] - m['D_trap'] - m['dU']) / m['J']
            rows.append({'design': tag, 'num_time_steps': num_time_steps, **m})

    keys = list(rows[0])
    out_path = ROOT / 'results' / '1_creep_elastic' / 'creep_measures.csv'
    with open(out_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(row[k] if k == 'design' else f'{row[k]:.10g}' for k in keys) + '\n')

    print("\nEnergies in mJ, equivalent creep strains dimensionless (integrals in mm^3).")
    print(f"{'':>16s}" + ''.join(f"{row['design'] + ' N=' + str(row['num_time_steps']):>20s}" for row in rows))
    for k in keys[2:]:
        print(f"{k:>16s}" + ''.join(f"{row[k]:>20.8e}" for row in rows))
    print(f"Saved {out_path}")
