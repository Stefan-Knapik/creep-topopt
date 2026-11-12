"""Shared problem setup of the rotating-blade study (experiments/6_blade_centrifugal).

The blade of Sec. 4.5 (experiments/5_blade) loaded by the centrifugal body force of its own mass
instead of a tip traction. The span direction (+y) is radial, the root (y = 0) sits on a disk of
radius ``R_HUB`` and the rotation axis is parallel to z (the chordwise direction of the untwisted
root section) through (x, y) = (0, -R_HUB). At a material point x the body force is

    b(theta, x) = rho(theta) * omega^2 * r_perp(x),     r_perp = (x, y + R_HUB, 0),

with rho the rule-of-mixtures density of the local composition, so the load depends on the design.

Units: mm, N, MPa, s, mJ. The `rho` column of toga/mat/material_data is in g/mm^3; a body force in
N/mm^3 needs tonne/mm^3 (1 tonne mm/s^2 = 1 N), hence the factor ``G_TO_TONNE`` = 1e-6. Masses are
reported in g and moments of inertia in g mm^2.

Sign convention: the weak form assembled by toga.fem.problem is
    residual = int sigma : grad(v) dV + int mass_map . v dV + int surface_map . v dS,
so a body force b enters as mass_map = -b and a traction t as surface_map = -t.

Import this module after jax has been configured (x64, platform), as the scripts in this folder do.
"""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as np
import meshio
import numpy as onp

from toga.fem.generate_mesh import Mesh, get_meshio_cell_type
from toga.fem.solver import ad_wrapper
from toga.mat import build_material_properties
from toga.mat.material_properties import design_to_physical, vmap_design_to_physical
from toga.physics.thermal import NonlinearThermal
from toga.physics.thermal_creep_explicit import ThermalCreepExplicit
from toga.physics.thermal_creep_implicit import ThermalCreepImplicit
from toga.utils import quad_to_cell_average, remove_orphan_nodes

ROOT = Path(__file__).resolve().parents[2]  # repository root
RESULTS = ROOT / "results" / "6_blade_centrifugal"
MESH_FILE = ROOT / "toga" / "mesh" / "blade" / "blade_050_000.inp"

MATERIAL_NAMES = ["IN625", "SS316"]
G_TO_TONNE = 1e-6       # g/mm^3 -> tonne/mm^3

# Rotation. Chosen so that the mean initial von Mises stress matches the tip-traction study of Sec. 4.5
# (about 32 MPa); evaluate_homogeneous_blends.py writes the operating point and the stress levels, and
# `verify_coarse.py --checks load` shows how the creep compliance grows with the speed.
R_HUB = 250.0           # mm, radius of the disk rim that carries the blade root
RPM = 7000.0
OMEGA = RPM * 2.0 * onp.pi / 60.0   # 733.04 rad/s; tip speed OMEGA * (R_HUB + SPAN) = 220 m/s
SPAN = 50.0             # mm


def rpm_to_omega(rpm):
    return rpm * 2.0 * onp.pi / 60.0


def results_dir(rpm=RPM, coarse=False):
    """results/6_blade_centrifugal/ for the reference speed, rpm_<speed>/ below it for any other."""
    folder = RESULTS if rpm == RPM else RESULTS / f"rpm_{round(rpm)}"
    return folder / "coarse" if coarse else folder

COLD_TEMPERATURE = 600.0   # root, deg C
HOT_TEMPERATURE = 800.0    # tip, deg C
END_TIME = 3600.0 * 100    # 100 hours
NUM_TIME_STEPS = 10


class FixedComposition:
    """A property model evaluated at a fixed composition, whatever the local one is.

    Used to switch off the composition dependence of one property at a time (the temperature
    dependence is kept), to find out what the grading exploits.
    """

    def __init__(self, model, composition):
        self.model = model
        self.composition = np.asarray(composition, dtype=float)
        self.input_dim = model.input_dim

    def __call__(self, x_query):
        x_query = np.asarray(x_query)
        n = self.composition.shape[0]
        composition = np.broadcast_to(self.composition, x_query.shape[:-1] + (n,))
        return self.model(np.concatenate((composition, x_query[..., n:]), axis=-1))


def build_materials(frozen=(), frozen_composition=(1.0, 0.0)):
    """Table models of the IN625/SS316 pair, as in experiments/5_blade, plus the density.

    frozen: names of properties made independent of the composition by evaluating them at
        ``frozen_composition`` (default: pure IN625). 'rho_load' freezes the density that enters
        the centrifugal load only; mass and inertia always use the true density.
    """
    material_properties = build_material_properties(
        material_names=MATERIAL_NAMES,
        property_configs={
            "k":     {"model": "table"},
            "E":     {"model": "table"},
            "logA":  {"model": "table", "temperature_dependent": False},
            "n":     {"model": "table", "temperature_dependent": False},
            "Q":     {"model": "table", "temperature_dependent": False},
            "cost":  {"model": "table", "temperature_dependent": False},
            "rho":   {"model": "table", "temperature_dependent": False},  # g/mm^3
        },)
    material_properties.rho_load = material_properties.rho
    for name in frozen:
        setattr(material_properties, name, FixedComposition(getattr(material_properties, name), frozen_composition))
    return material_properties


class CentrifugalLoad:
    """Mixin for ThermalCreepExplicit / ThermalCreepImplicit: design-dependent centrifugal body
    force, an optional tensile tip traction, and the work of both on a displacement field."""

    r_hub = R_HUB               # mm
    omega = OMEGA               # rad/s
    tip_traction = 0.0          # MPa, along +y on the surface selected by location_fns[0]
    body_force_scale = 1.0      # 0 switches the body force off (tip-traction-only checks)

    def custom_init(self, material_models):
        super().custom_init(material_models)
        self.model_rho_load = getattr(material_models, 'rho_load', material_models.rho)

    def body_force(self, x, theta):
        """Centrifugal force per unit volume [N/mm^3] at point x for the design vector theta."""
        geom, mats = design_to_physical(theta)
        rho = self.model_rho_load(mats) * geom * G_TO_TONNE
        r_perp = np.array([x[0], x[1] + self.r_hub, 0.0])
        return self.body_force_scale * rho * self.omega**2 * r_perp

    def get_mass_map(self):
        # internal_vars = (sigma_old, eps_creep_old, theta, T, reference_T[, dt]) at the quadrature point
        def mass_map(u, x, sigma_old, epsilon_creep_old, theta, *rest):
            return -self.body_force(x, theta)
        return mass_map

    def get_surface_maps(self):
        if len(self.boundary_inds_list) == 0:
            return []

        def traction(u, x, *internal_vars_surface):
            return -np.array([0.0, self.tip_traction, 0.0])
        return [traction]

    def body_force_at_quads(self, theta):
        """b at every quadrature point, (num_cells, num_quads, 3), for theta of shape (num_cells, 1 + num_mat)."""
        thetas = np.repeat(theta[:, None, :], self.fe.num_quads, axis=1)
        return jax.vmap(jax.vmap(self.body_force))(self.physical_quad_points, thetas)

    def compute_external_work(self, sol, theta):
        """W_ext(u) = int b(theta) . u dV + int t . u dS  [mJ], the compliance of the dead loads."""
        u_quad = self.fe.convert_from_dof_to_quad(sol)  # (num_cells, num_quads, 3)
        body = np.sum(self.body_force_at_quads(theta) * u_quad * self.fe.JxW[:, :, None])
        return body + self.compute_compliance(sol)

    def compute_creep_compliance(self, u_full, theta):
        """J = W_ext(u_end) - W_ext(u_0)."""
        return self.compute_external_work(u_full[-1], theta) - self.compute_external_work(u_full[0], theta)

    def compute_total_force(self, theta):
        """Resultant of the body force [N]."""
        return np.sum(self.body_force_at_quads(theta) * self.fe.JxW[:, :, None], axis=(0, 1))

    def compute_mass(self, theta):
        """Total mass [g]."""
        return self.compute_total_by_volume(theta, property="rho")

    def compute_inertia(self, theta):
        """Moment of inertia about the rotation axis, I = int rho |r_perp|^2 dV  [g mm^2]."""
        geom, mat = vmap_design_to_physical(theta)
        x = self.physical_quad_points
        r2 = x[..., 0]**2 + (x[..., 1] + self.r_hub)**2
        return np.sum(self.model_rho(mat) * geom * np.sum(r2 * self.fe.JxW, axis=1))


def von_mises(sigma):
    """Von Mises equivalent of a stress (..., 3, 3)."""
    s = sigma - np.trace(sigma, axis1=-2, axis2=-1)[..., None, None] / 3.0 * np.eye(3)
    return np.sqrt(1.5 * np.sum(s * s, axis=(-2, -1)))


def equivalent_strain(eps):
    """Von Mises equivalent of a (deviatoric) creep strain (..., 3, 3)."""
    return np.sqrt(2.0 / 3.0 * np.sum(eps * eps, axis=(-2, -1)))


def load_mesh(mesh_file=MESH_FILE):
    cell_type = get_meshio_cell_type('HEX8')
    meshio_mesh = meshio.read(str(mesh_file))
    points, cells, reorder_nodes = remove_orphan_nodes(meshio_mesh, cell_type, return_index_map=True)
    mesh = Mesh(points, cells)
    return mesh, reorder_nodes(meshio_mesh.point_sets['bottom']), reorder_nodes(meshio_mesh.point_sets['top'])


def build_blade(mesh_file=MESH_FILE, scheme="explicit", num_time_steps=NUM_TIME_STEPS, end_time=END_TIME,
                r_hub=R_HUB, omega=OMEGA, tip_traction=0.0, body_force_scale=1.0, frozen=(),
                solver_options=None, save_history=False, material_properties=None):
    """Thermal and creep problems of the rotating blade and the functions the scripts share.

    Returns a namespace with
        mesh, problem_T, problem_u, bottom_nodes, top_nodes, num_cells
        forward(theta)                 -> T, u_full, body_params        (theta: (cells, 1 + num_mat))
        evaluate(composition)          -> dict of J, IN625 fraction, mass [g], inertia [g mm^2] and the
                                          stress / creep-strain scalars, and aux (T, u_full, body_params)
                                          (composition: (cells, num_mat), the design variables)
        full_theta(composition)        -> (cells, 1 + num_mat) with the geometry column of ones
    """
    solver_options = {'pardiso_solver': {}} if solver_options is None else solver_options
    if material_properties is None:
        material_properties = build_materials(frozen)
    ele_type = 'HEX8'
    mesh, bottom_nodes, top_nodes = load_mesh(mesh_file)

    def bottom(point, i):  # root, y = 0
        return np.isin(i, bottom_nodes)

    def top(point, i):  # tip, y = 50 mm
        return np.isin(i, top_nodes)

    dirichlet_bc_info_T = [[bottom, top], [0, 0], [lambda point: COLD_TEMPERATURE, lambda point: HOT_TEMPERATURE]]
    problem_T = NonlinearThermal(mesh, vec=1, dim=3, ele_type=ele_type,
                                 dirichlet_bc_info=dirichlet_bc_info_T,
                                 additional_info=(material_properties,),)

    dirichlet_bc_info_u = [[bottom] * 3, [0, 1, 2], [lambda point: 0.0] * 3]
    base = {"explicit": ThermalCreepExplicit, "implicit": ThermalCreepImplicit}[scheme]

    class CentrifugalBlade(CentrifugalLoad, base):
        pass

    CentrifugalBlade.r_hub = r_hub
    CentrifugalBlade.omega = omega
    CentrifugalBlade.tip_traction = tip_traction
    CentrifugalBlade.body_force_scale = body_force_scale

    problem_u = CentrifugalBlade(mesh, vec=3, dim=3, ele_type=ele_type,
                                 dirichlet_bc_info=dirichlet_bc_info_u,
                                 location_fns=[top] if tip_traction != 0.0 else None,
                                 additional_info=(material_properties,),)

    fwd_pred_T = ad_wrapper(problem_T, solver_options=solver_options, adjoint_solver_options=solver_options)

    def forward(theta):
        T = fwd_pred_T(theta)[0]
        full_sol, body_params = problem_u.forward_solve(
            theta=theta,
            T=T,
            reference_T=COLD_TEMPERATURE,
            end_time=end_time,
            num_time_steps=num_time_steps,
            fwd_solver_options=solver_options,
            vjp_solver_options=solver_options,
            save_history=save_history,
        )
        # T (num_nodes, 1); u_full (num_time_steps + 1, num_nodes, 3);
        # body_params = (sigmas, eps_creep, theta, T_quad, ref_T_quad[, dt]), first and last step
        # (every step with save_history).
        return T, full_sol[0], body_params

    def full_theta(composition):
        return np.concatenate((np.ones((len(composition), 1)), composition), axis=1)

    JxW = problem_u.fe.JxW
    volume = np.sum(JxW)

    def field_scalars(body_params):
        """Volume mean, peak at the quadrature points and peak of the cell average (as in the .vtu)."""
        out = {}
        for name, field, measure in (('vm0', body_params[0][0], von_mises),
                                     ('vm_end', body_params[0][-1], von_mises),
                                     ('creep_eq', body_params[1][-1], equivalent_strain)):
            at_quads = measure(field)
            out[f'{name}_mean'] = np.sum(at_quads * JxW) / volume
            out[f'{name}_max_quad'] = np.max(at_quads)
            out[f'{name}_max_cell'] = np.max(measure(quad_to_cell_average(problem_u.fe, field)))
        return out

    def evaluate(composition):
        theta = full_theta(composition)
        T, u_full, body_params = forward(theta)
        f = {
            'J': problem_u.compute_creep_compliance(u_full, theta),
            'W0': problem_u.compute_external_work(u_full[0], theta),
            'IN625_fraction': problem_u.compute_volume_fraction_of_material(theta, mat_idx=0),
            'mass_g': problem_u.compute_mass(theta),
            'inertia_g_mm2': problem_u.compute_inertia(theta),
            **field_scalars(body_params),
            'T_mean': np.sum(problem_u.fe.convert_from_dof_to_quad(T)[..., 0] * JxW) / volume,
            'u_max_0': np.max(np.abs(u_full[0])),
            'u_max_end': np.max(np.abs(u_full[-1])),
        }
        return f, {'T': T, 'u_full': u_full, 'body_params': body_params}

    return SimpleNamespace(
        mesh=mesh, problem_T=problem_T, problem_u=problem_u, material_properties=material_properties,
        bottom_nodes=onp.asarray(bottom_nodes), top_nodes=onp.asarray(top_nodes),
        num_cells=len(mesh.cells), forward=forward, evaluate=evaluate, full_theta=full_theta,
        field_scalars=field_scalars, volume=float(volume),
    )


def homogeneous(num_cells, in625_fraction):
    """Uniform blend, (num_cells, 2) with columns (IN625, SS316)."""
    return np.ones((num_cells, 1)) * np.array([in625_fraction, 1.0 - in625_fraction])


SCALAR_KEYS = ['J', 'W0', 'IN625_fraction', 'mass_g', 'inertia_g_mm2',
               'vm0_mean', 'vm0_max_quad', 'vm0_max_cell', 'vm_end_mean', 'vm_end_max_quad', 'vm_end_max_cell',
               'creep_eq_mean', 'creep_eq_max_quad', 'creep_eq_max_cell', 'T_mean', 'u_max_0', 'u_max_end']


def coarse_mesh(elem_size=1.5):
    """Coarse blade mesh for verification runs, generated with toga/mesh/blade/generate_twisted_blade_mesh.py
    (needs gmsh) into results/6_blade_centrifugal/mesh/. elem_size 1.5 mm gives about 3,000 HEX8."""
    import importlib.util
    mesh_dir = RESULTS / "mesh"
    name = f"blade_coarse_{elem_size:.2f}".replace(".", "p")
    mesh_file = mesh_dir / f"{name}.inp"
    if not mesh_file.exists():
        mesh_dir.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location(
            "generate_twisted_blade_mesh", ROOT / "toga" / "mesh" / "blade" / "generate_twisted_blade_mesh.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        generator.MESH_DIR = str(mesh_dir)  # the generator writes next to itself by default
        generator.make_blade_mesh(name, elem_size=elem_size)
    return mesh_file
