"""Shared problem setup of the graded turbine blade (Sec. 4.5, experiments/5_blade).

A twisted blade of fixed geometry, graded between IN625 and SS316, that creeps under the
centrifugal force of its own mass with a spanwise temperature gradient:

* loading: the span (+y) is radial, the root (y = 0) sits on a disk of radius ``r_hub`` and the
  rotation axis is parallel to z through (x, y) = (0, -r_hub),

      b(x) = rho(x_f) * omega^2 * (x, y + r_hub, 0),

  with rho the rule-of-mixtures density of the local composition (the load depends on the design).
  The speed is set through a nominal root stress: for a prismatic blade of span L and uniform
  density the centrifugal stress at the root is  sigma = rho * omega^2 * L * (r_hub + L/2).
  Optionally a share ``shroud`` of that stress comes from a tip shroud, a uniform radial traction
  on the tip, so that the stress does not fall to zero at the tip;

* temperature: the root is held at ``t_root`` and the tip at ``t_hot``, the airfoil surfaces are
  insulated. Both stay inside 750-800 deg C, the range in which both alloys were creep tested;

* thermal expansion is on, stress-free at the root temperature (the temperature of the disk the
  root is clamped to). The explicit scheme is unstable with it, so the creep problem is implicit
  (IncrementalCreepImplicit below);

* composition: the design variables w (cells, 2) are normalized to fractions x = w / sum(w) and
  then density-filtered, x_f = H x (cone kernel of radius ``radius_mm``, volume weighted, rows sum
  to one so x_f stays on the simplex). Everything physical is computed from x_f: the properties,
  the density in the body force, the IN625 content that is bounded, mass and moment of inertia.

Units: mm, N, MPa, s, mJ, W, deg C. `rho` is tabulated in g/mm^3; a body force in N/mm^3 needs
tonne/mm^3, hence ``G_TO_TONNE``.

Sign convention of toga.fem.problem:
    residual = int flux : grad(v) dV + int mass_map . v dV + int surface_map . v dS,
so a body force b enters as mass_map = -b and a traction t as surface_map = -t.

Import this module after jax has been configured (x64), as the scripts in this folder do.
"""
import importlib.util
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
from toga.opt.filtering import applyDensityFilter, compute_filter_radius
from toga.physics.thermal import NonlinearThermal
from toga.physics.thermal_creep_explicit import ThermalCreepExplicit
from toga.physics.thermal_creep_implicit import ThermalCreepImplicit
from toga.utils import quad_to_cell_average, remove_orphan_nodes

ROOT = Path(__file__).resolve().parents[2]  # repository root
FINE_MESH = ROOT / "toga" / "mesh" / "blade" / "blade_050_000.inp"   # the paper mesh, 50,336 HEX8
RESULTS = ROOT / "results" / "5_blade"
FIGURES = ROOT / "figures"

MATERIAL_NAMES = ["IN625", "SS316"]
G_TO_TONNE = 1e-6          # g/mm^3 -> tonne/mm^3
SPAN = 50.0                # mm
R_HUB = 250.0              # mm
T_ROOT = 750.0             # deg C, also the stress-free temperature
T_HOT = 800.0              # deg C, tip
END_TIME = 3600.0 * 100    # 100 hours
NUM_TIME_STEPS = 10
ELEM_SIZE = 0.72           # mm, about 25,000 HEX8, the exploration mesh (the paper mesh is 0.57 mm)
RADIUS_MM = 1.4            # filter radius of the composition [mm]

# Thermal expansion over 750-800 deg C (alpha_mode "tangent", the default): instantaneous
# coefficients, taken as constant over this 50 K window, strain = alpha * (T - T_root).
# IN625: 18.9e-6 /K, the tangent over 750-800 deg C of the mean coefficients (from 21 deg C) of the
#   Special Metals datasheet SMC-063: 15.3e-6 /K at 760 deg C and 15.8e-6 /K at 871 deg C.
# SS316: 21.0e-6 /K, from Desai & Ho, CINDAS Report 51 (1978), Table 4: dL/L0 = 1.342 % at 1000 K
#   and 1.552 % at 1100 K (reference 293 K).
ALPHA_TANGENT = {"IN625": 18.9e-6, "SS316": 21.0e-6}   # 1/K
ALPHA_T0 = 20.0   # deg C, reference of the tabulated mean coefficients (alpha_mode "secant")


def mesh_name(elem_size=None, fine=False):
    return "fine" if fine else f"h{(ELEM_SIZE if elem_size is None else elem_size):.2f}".replace(".", "p")


def case_name(root_stress, shroud=0.0, hours=100.0, radius_mm=RADIUS_MM, mesh="fine", num_time_steps=NUM_TIME_STEPS,
              alpha_mode="tangent", expansion=True, cte_mismatch_scale=1.0, init="uniform", tag=None):
    """Folder name of a case: S<root stress>_F<shroud %>_T<hours>h_r<filter radius, mm>_<mesh>_N<steps>, then
    whatever differs from the defaults: the expansion model (table / secant / noexp), cms<mismatch
    scale>, pert (perturbed start) and a free tag."""
    name = (f"S{round(root_stress):03d}_F{round(100 * shroud):02d}_T{hours:g}h_r{(radius_mm or 0.0):.1f}"
            f"_{mesh}_N{num_time_steps}")
    if not expansion:
        name += "_noexp"
    elif alpha_mode != "tangent":
        name += f"_{alpha_mode}"
    if expansion and cte_mismatch_scale != 1.0:
        name += f"_cms{cte_mismatch_scale:g}"
    if init != "uniform":
        name += f"_{init[:4]}"
    if tag:
        name += f"_{tag}"
    return name.replace(".", "p")


def run_dir(study, vf, **case):
    """results/5_blade/<study>/<case_name>/V<IN625 bound, %>/"""
    return RESULTS / study / case_name(**case) / f"V{round(100 * vf):02d}"


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


class ThermalExpansion:
    """Coefficient alpha(x, T) such that the thermal strain of the library physics,
    alpha * (T - t_ref), is the expansion between t_ref and T. Rule of mixtures in the composition.

    mode
        "tangent"  constant instantaneous coefficients ``ALPHA_TANGENT``
        "table"    the tabulated coefficient at T, used as is (as in the earlier studies)
        "secant"   the tables read as mean coefficients abar from ``ALPHA_T0``: the strain is
                   abar(T) (T - T0) - abar(t_ref) (t_ref - T0). Within one linear segment of the
                   table, abar = a + b T, this equals (T - t_ref) * (a + b (T + t_ref - T0)), which
                   is what is returned; t_ref and every temperature of the problem must lie in the
                   same segment (checked for [t_ref, t_max]).
    mismatch_scale
        scales the difference of the alloys' coefficients about their mean (1 = as is, 0 = both
        alloys expand alike).
    """

    def __init__(self, mode, t_ref, t_max, mismatch_scale=1.0, tables=None):
        self.mode = mode
        self.t_ref = float(t_ref)
        self.mismatch_scale = float(mismatch_scale)
        self.n_materials = len(MATERIAL_NAMES)
        self.input_dim = self.n_materials + 1
        if mode == "tangent":
            self.constants = [ALPHA_TANGENT[name] for name in MATERIAL_NAMES]
        elif mode == "table":
            self.tables = tables
        elif mode == "secant":
            self.segments = []
            for t, v in tables:
                t, v = onp.asarray(t), onp.asarray(v)
                i = int(onp.clip(onp.searchsorted(t, t_ref, side="right") - 1, 0, len(t) - 2))
                if not (t[i] <= t_ref and t_max <= t[i + 1]):
                    raise ValueError(f"secant expansion: {t_ref} and {t_max} C are not in one segment of the table")
                b = (v[i + 1] - v[i]) / (t[i + 1] - t[i])
                self.segments.append((float(v[i] - b * t[i]), float(b)))
        else:
            raise ValueError(f"unknown alpha_mode {mode!r}")

    def pure(self, temperature):
        if self.mode == "tangent":
            values = [c + 0.0 * temperature for c in self.constants]
        elif self.mode == "table":
            values = [np.interp(temperature, t, v) for t, v in self.tables]
        else:
            values = [a + b * (temperature + self.t_ref - ALPHA_T0) for a, b in self.segments]
        mean = sum(values) / len(values)
        return [mean + self.mismatch_scale * (v - mean) for v in values]

    def __call__(self, x_query):
        x_query = np.asarray(x_query)
        fractions = x_query[..., :self.n_materials]
        pure = self.pure(x_query[..., self.n_materials])
        return sum(fractions[..., i] * pure[i] for i in range(self.n_materials))


def build_materials(expansion=True, alpha_mode="tangent", cte_mismatch_scale=1.0, t_ref=T_ROOT, t_max=T_HOT,
                    frozen=(), frozen_composition=(1.0, 0.0)):
    """Table models of the IN625/SS316 pair (toga/mat/material_data), density and thermal expansion.

    frozen: names of properties made independent of the composition by evaluating them at
        ``frozen_composition`` (default: pure IN625). 'rho_load' freezes the density that enters
        the centrifugal load only; mass and inertia always use the true density.
    """
    property_configs = {
        "k":     {"model": "table"},
        "E":     {"model": "table"},
        "logA":  {"model": "table", "temperature_dependent": False},
        "n":     {"model": "table", "temperature_dependent": False},
        "Q":     {"model": "table", "temperature_dependent": False},
        "cost":  {"model": "table", "temperature_dependent": False},
        "rho":   {"model": "table", "temperature_dependent": False},  # g/mm^3
    }
    if expansion:
        tables = None
        if alpha_mode != "tangent":
            tables = build_material_properties(material_names=MATERIAL_NAMES,
                                               property_configs={"alpha": {"model": "table"}}).alpha.tables
        property_configs["alpha"] = ThermalExpansion(alpha_mode, t_ref, t_max, cte_mismatch_scale, tables)
    material_properties = build_material_properties(material_names=MATERIAL_NAMES, property_configs=property_configs)
    material_properties.rho_load = material_properties.rho
    for name in frozen:
        setattr(material_properties, name, FixedComposition(getattr(material_properties, name), frozen_composition))
    return material_properties


def omega_for_root_stress(root_stress, r_hub, material_properties):
    """Angular speed [rad/s] at which the blade's own mass gives the nominal root stress (50/50 blend)."""
    rho = float(onp.squeeze(material_properties.rho(np.array([0.5, 0.5])))) * G_TO_TONNE
    return float(onp.sqrt(root_stress / (rho * SPAN * (r_hub + 0.5 * SPAN))))


class IncrementalCreepImplicit(ThermalCreepImplicit):
    """ThermalCreepImplicit solved for the displacement increment of each time step.

    The library class solves every step for the total displacement with Newton started from u = 0.
    Once creep strain has accumulated, u = 0 is a state of large (negative) elastic strain, the local
    backward-Euler update is evaluated at stresses far above the real ones and the solve diverges.
    Solving for du = u - u_old with the old displacement gradient as an internal variable makes the
    zero initial guess the previous converged state. The equations and the adjoint are unchanged.

    internal_vars = (sigma_old, eps_creep_old, theta, T, reference_T, dt, u_grad_old)
    """

    def creep_stress_strain_map(self, du_grad, sigma_old, epsilon_creep_old, theta, T, reference_T, dt, u_grad_old):
        return super().creep_stress_strain_map(du_grad + u_grad_old, sigma_old, epsilon_creep_old, theta, T, reference_T, dt)

    def update_stress_strain(self, du, params):
        sigmas_old, epsilons_creep_old, thetas, T_quad, ref_T_quad, dts, u_grads_old = params
        du_grads = self.fe.sol_to_grad(du)
        stress, creep_strain_inc = self.stress_strain_fns()(
            du_grads, sigmas_old, epsilons_creep_old, thetas, T_quad, ref_T_quad, dts, u_grads_old)
        return stress, epsilons_creep_old + creep_strain_inc, thetas, T_quad, ref_T_quad, dts, du_grads + u_grads_old

    def forward_solve(self, theta, T, reference_T, end_time, num_time_steps,
                      fwd_solver_options, vjp_solver_options, save_history=True):
        """As ThermalCreepImplicit.forward_solve: ([u at every step], body params at every step, or at
        the first and last step only with save_history=False)."""
        fwd_pred = ad_wrapper(self, solver_options=fwd_solver_options, adjoint_solver_options=vjp_solver_options)

        body_params = self.init_params(theta, T, reference_T)
        body_params = (*body_params, np.zeros_like(body_params[0]))  # u_grad_old
        u = np.zeros((self.fe.num_total_nodes, self.fe.vec))
        dts = np.concatenate([np.array([0.0]), np.diff(np.linspace(0.0, end_time, num_time_steps + 1))])
        self.end_time = end_time

        # One array per loaded surface, as in the library class (the tip traction does not use it).
        surface_params = [np.ones((quads.shape[0], self.fe.num_face_quads)) for quads in self.physical_surface_quad_points]

        def step(carry, dt):
            bp, u = carry
            bp = (*bp[:5], np.full((len(self.fe.cells), self.fe.num_quads), dt), bp[6])
            du = fwd_pred([bp, surface_params])[0]
            return (self.update_stress_strain(du, bp), u + du)

        if save_history:
            def scan_fn(carry, dt):
                carry = step(carry, dt)
                return carry, carry
            _, (body_params_list, u_full) = jax.lax.scan(scan_fn, (body_params, u), dts)
            return [u_full], body_params_list

        def scan_fn(carry, dt):
            carry = step(carry, dt)
            return carry, carry[1]

        bp_first, u_first = step((body_params, u), dts[0])
        (bp_last, _), u_rest = jax.lax.scan(scan_fn, (bp_first, u_first), dts[1:])
        body_params_list = jax.tree_util.tree_map(lambda a, b: np.stack((a, b)), bp_first, bp_last)
        return [np.concatenate((u_first[None], u_rest))], body_params_list


class CentrifugalLoad:
    """Mixin for the creep problems: design-dependent centrifugal body force, an optional radial tip
    traction (the pull of a tip shroud), and the work of both on a displacement field."""

    r_hub = R_HUB           # mm
    omega = 0.0             # rad/s
    tip_traction = 0.0      # MPa along +y on the tip (location_fns[0])
    body_force_scale = 1.0  # 0 switches the body force off (checks)

    def custom_init(self, material_models):
        super().custom_init(material_models)
        self.model_rho_load = getattr(material_models, 'rho_load', material_models.rho)

    def body_force(self, x, theta):
        """Centrifugal force per unit volume [N/mm^3] at point x for the design vector theta."""
        geom, mats = design_to_physical(theta)
        rho = self.model_rho_load(mats) * geom * G_TO_TONNE
        return self.body_force_scale * rho * self.omega**2 * np.array([x[0], x[1] + self.r_hub, 0.0])

    def get_mass_map(self):
        # internal_vars = (sigma_old, eps_creep_old, theta, T, reference_T[, dt, u_grad_old]) at the quadrature point
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
        """W_ext(u) = int b(theta) . u dV + int t . u dS  [mJ], the work of the dead loads."""
        u_quad = self.fe.convert_from_dof_to_quad(sol)  # (num_cells, num_quads, 3)
        body = np.sum(self.body_force_at_quads(theta) * u_quad * self.fe.JxW[:, :, None])
        return body + self.compute_compliance(sol)

    def compute_creep_compliance(self, u_full, theta):
        """J = W_ext(u_end) - W_ext(u_0): the work of the dead loads on the creep displacement.
        The thermal displacement is part of u_0 and drops out."""
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


def blade_mesh(elem_size=ELEM_SIZE):
    """Blade mesh of the given element size, generated with toga/mesh/blade/generate_twisted_blade_mesh.py
    (needs gmsh) into results/5_blade/mesh/ on first use. 1.5 mm: 3k cells, 0.98: 10.8k, 0.72: 25k."""
    mesh_dir = RESULTS / "mesh"
    name = f"blade_{elem_size:.2f}".replace(".", "p")
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


AIRFOIL_SETS = ["suction_leading_edge", "suction_side", "suction_trailing_edge",
                "pressure_trailing_edge", "pressure_side", "pressure_leading_edge"]


def load_mesh(mesh_file):
    """Mesh and the node sets 'bottom' (root), 'top' (tip) and the airfoil surfaces."""
    cell_type = get_meshio_cell_type('HEX8')
    meshio_mesh = meshio.read(str(mesh_file))
    points, cells, reorder_nodes = remove_orphan_nodes(meshio_mesh, cell_type, return_index_map=True)
    sets = {name: onp.asarray(reorder_nodes(meshio_mesh.point_sets[name])) for name in ["bottom", "top"] + AIRFOIL_SETS}
    return Mesh(points, cells), sets


def build_blade(t_root=T_ROOT, t_hot=T_HOT, r_hub=R_HUB, root_stress=100.0, shroud=0.0,
                elem_size=ELEM_SIZE, mesh_file=None, scheme="implicit", num_time_steps=NUM_TIME_STEPS, end_time=END_TIME,
                expansion=True, alpha_mode="tangent", cte_mismatch_scale=1.0, radius_mm=RADIUS_MM,
                frozen=(), body_force_scale=1.0, solver_options=None, save_history=False):
    """Thermal and creep problems of the rotating blade and the functions the scripts share.

    Returns a namespace with
        mesh, problem_T, problem_u, node_sets, num_cells, volume, cell_volume, centroids,
        omega, rpm, tip_speed [m/s], tip_traction [MPa], shroud_mass [g]
        physical(composition)     -> x_f (cells, num_mat): normalized and filtered composition
        full_theta(composition)   -> (cells, 1 + num_mat): geometry column of ones and x_f
        forward(theta)            -> T, u_full, body_params
        evaluate(composition)     -> dict of scalars (SCALAR_KEYS) and aux (theta, T, u_full, body_params)
    ``composition`` is the raw design field (cells, num_mat), the optimizer's variables.
    """
    solver_options = {'pardiso_solver': {}} if solver_options is None else solver_options
    material_properties = build_materials(expansion, alpha_mode, cte_mismatch_scale, t_ref=t_root, t_max=t_hot,
                                          frozen=frozen)
    ele_type = 'HEX8'
    mesh, node_sets = load_mesh(blade_mesh(elem_size) if mesh_file is None else mesh_file)
    bottom_nodes, top_nodes = node_sets["bottom"], node_sets["top"]

    def bottom(point, i):  # root, y = 0
        return np.isin(i, bottom_nodes)

    def top(point, i):  # tip, y = 50 mm
        return np.isin(i, top_nodes)

    #%% Thermal problem: fixed root and tip temperatures, insulated airfoil surfaces

    dirichlet_bc_info_T = [[bottom, top], [0, 0], [lambda point: t_root, lambda point: t_hot]]
    problem_T = NonlinearThermal(mesh, vec=1, dim=3, ele_type=ele_type,
                                 dirichlet_bc_info=dirichlet_bc_info_T,
                                 additional_info=(material_properties,),)

    #%% Creep problem

    # ``shroud`` is the share of the nominal root stress carried for a tip shroud, applied as a uniform
    # radial traction on the tip; the blade's own mass gives the rest.
    omega = omega_for_root_stress((1.0 - shroud) * root_stress, r_hub, material_properties)
    tip_traction = shroud * root_stress
    base = {"explicit": ThermalCreepExplicit, "implicit": IncrementalCreepImplicit}[scheme]

    class CentrifugalBlade(CentrifugalLoad, base):
        pass

    CentrifugalBlade.r_hub = r_hub
    CentrifugalBlade.omega = omega
    CentrifugalBlade.tip_traction = tip_traction
    CentrifugalBlade.body_force_scale = body_force_scale

    dirichlet_bc_info_u = [[bottom] * 3, [0, 1, 2], [lambda point: 0.0] * 3]
    problem_u = CentrifugalBlade(mesh, vec=3, dim=3, ele_type=ele_type,
                                 dirichlet_bc_info=dirichlet_bc_info_u,
                                 location_fns=[top] if shroud else None,
                                 additional_info=(material_properties,),)

    #%% Composition: normalize, then filter

    if radius_mm:
        H, Hs = compute_filter_radius(problem_u.fe, radius_mm)
        ft = {'H': H, 'Hs': Hs}
    else:
        ft = None

    def physical(composition):
        x = composition / np.sum(composition, axis=1, keepdims=True)
        return x if ft is None else applyDensityFilter(ft, x)

    def full_theta(composition):
        return np.concatenate((np.ones((len(composition), 1)), physical(composition)), axis=1)

    #%% Simulation

    fwd_pred_T = ad_wrapper(problem_T, solver_options=solver_options, adjoint_solver_options=solver_options)

    def forward(theta):
        T = fwd_pred_T(theta)[0]
        full_sol, body_params = problem_u.forward_solve(
            theta=theta,
            T=T,
            reference_T=t_root,   # stress-free at the root temperature
            end_time=end_time,
            num_time_steps=num_time_steps,
            fwd_solver_options=solver_options,
            vjp_solver_options=solver_options,
            save_history=save_history,
        )
        # T (num_nodes, 1); u_full (num_time_steps + 1, num_nodes, 3);
        # body_params = (sigmas, eps_creep, theta, T_quad, ref_T_quad[, dt, u_grad]), first and last step
        # (every step with save_history).
        return T, full_sol[0], body_params

    JxW = problem_u.fe.JxW
    volume = np.sum(JxW)

    def field_scalars(body_params):
        """Volume mean and peak of the cell average (as in the .vtu) of stress and creep strain."""
        out = {}
        for name, field, measure in (('vm0', body_params[0][0], von_mises),
                                     ('vm_end', body_params[0][-1], von_mises),
                                     ('creep_eq', body_params[1][-1], equivalent_strain)):
            out[f'{name}_mean'] = np.sum(measure(field) * JxW) / volume
            out[f'{name}_max_cell'] = np.max(measure(quad_to_cell_average(problem_u.fe, field)))
        return out

    def in625_fraction(composition):
        """Volume-averaged IN625 fraction of the filtered composition: the bounded quantity."""
        return problem_u.compute_volume_fraction_of_material(full_theta(composition), mat_idx=0)

    def evaluate(composition):
        theta = full_theta(composition)
        T, u_full, body_params = forward(theta)
        s = np.sum(composition, axis=1)
        f = {
            'J': problem_u.compute_creep_compliance(u_full, theta),
            'W0': problem_u.compute_external_work(u_full[0], theta),
            'IN625_fraction': problem_u.compute_volume_fraction_of_material(theta, mat_idx=0),
            'mass_g': problem_u.compute_mass(theta),
            'inertia_g_mm2': problem_u.compute_inertia(theta),
            **field_scalars(body_params),
            'T_min': np.min(T),
            'T_max': np.max(T),
            'T_mean': np.sum(problem_u.fe.convert_from_dof_to_quad(T)[..., 0] * JxW) / volume,
            'u_max_0': np.max(np.abs(u_full[0])),
            'u_max_end': np.max(np.abs(u_full[-1])),
            'w_sum_min': np.min(s),
            'w_sum_max': np.max(s),
        }
        return f, {'theta': theta, 'T': T, 'u_full': u_full, 'body_params': body_params}

    cells = onp.asarray(mesh.cells)
    return SimpleNamespace(
        mesh=mesh, problem_T=problem_T, problem_u=problem_u, material_properties=material_properties,
        node_sets=node_sets, num_cells=len(cells), volume=float(volume),
        cell_volume=onp.asarray(np.sum(JxW, axis=1)), centroids=onp.asarray(mesh.points)[cells].mean(axis=1),
        omega=omega, rpm=omega * 60.0 / (2.0 * onp.pi), tip_speed=omega * (r_hub + SPAN) * 1e-3,
        tip_traction=tip_traction, r_hub=r_hub, t_root=t_root, t_hot=t_hot, end_time=end_time,
        num_time_steps=num_time_steps,
        # mass of a shroud at the tip radius that pulls with the tip traction [g] (tip area = volume / span)
        shroud_mass=tip_traction * float(volume) / SPAN / (omega**2 * (r_hub + SPAN)) / G_TO_TONNE,
        physical=physical, full_theta=full_theta, forward=forward, evaluate=evaluate,
        in625_fraction=in625_fraction, field_scalars=field_scalars,
    )


def build_blade_from_case(case, **overrides):
    """The blade of a run's case.json (written by optimize_graded_blade.py), with overrides."""
    kwargs = dict(t_root=case['t_root'], t_hot=case['t_hot'], r_hub=case['r_hub'], root_stress=case['root_stress'],
                  shroud=case['shroud'], elem_size=case['elem_size'], mesh_file=FINE_MESH if case['fine'] else None,
                  num_time_steps=case['num_time_steps'], end_time=3600.0 * case['hours'], expansion=case['expansion'],
                  alpha_mode=case['alpha_mode'], cte_mismatch_scale=case['cte_mismatch_scale'],
                  radius_mm=case['radius_mm'])
    kwargs.update(overrides)
    return build_blade(**kwargs)


def homogeneous(num_cells, in625_fraction):
    """Uniform blend, (num_cells, 2) with columns (IN625, SS316)."""
    return np.ones((num_cells, 1)) * np.array([in625_fraction, 1.0 - in625_fraction])


def span_graded(blade, in625_root=0.9, in625_tip=0.1):
    """Two-zone blade with a sharp interface at midspan: a hard case for the thermal mismatch stress."""
    in625 = onp.where(blade.centroids[:, 1] < 0.5 * SPAN, in625_root, in625_tip)
    return np.stack((in625, 1.0 - in625), axis=1)


def creep_rate(problem, composition, sigma_vm, T):
    """Norton-Arrhenius equivalent creep rate [1/s] of a blend at von Mises stress sigma_vm [MPa] and T [C]."""
    _, _, A, n, _, Q = problem.get_element_properties(np.asarray(composition), T)
    return A * np.exp(-Q / (problem.GAS_CONSTANT * (T + problem.CELSIUS_TO_KELVIN))) * sigma_vm**n


def alloy_indicator(blade, body_params, T):
    """Per cell: m = sigma_vm * (rate_SS316 - rate_IN625)(sigma_vm, T), the creep dissipation rate
    [mJ/(mm^3 s)] that swapping SS316 for IN625 removes at the cell's stress and temperature (taken
    at the end of the hold). Negative where SS316 is the slower alloy (low stress).
    Returns m, sigma_vm (end), sigma_vm (t = 0) and T of the cells."""
    fe = blade.problem_u.fe
    vm_end = von_mises(quad_to_cell_average(fe, body_params[0][-1]))
    vm_0 = von_mises(quad_to_cell_average(fe, body_params[0][0]))
    T_cell = np.mean(np.asarray(T)[np.asarray(blade.mesh.cells), 0], axis=1)
    rate = jax.vmap(lambda c, s, t: creep_rate(blade.problem_u, c, s, t), in_axes=(None, 0, 0))
    m = vm_end * (rate(np.array([0.0, 1.0]), vm_end, T_cell) - rate(np.array([1.0, 0.0]), vm_end, T_cell))
    return onp.asarray(m), onp.asarray(vm_end), onp.asarray(vm_0), onp.asarray(T_cell)


def top_fraction_design(score, cell_volume, vf, lo=1e-3):
    """Raw design that puts IN625 in the cells of highest ``score`` holding a share ``vf`` of the volume."""
    order = onp.argsort(-score)
    chosen = onp.zeros(len(score), bool)
    chosen[order[onp.cumsum(cell_volume[order]) <= vf * cell_volume.sum()]] = True
    return np.asarray(onp.where(chosen[:, None], [1.0, lo], [lo, 1.0]))


SCALAR_KEYS = ['J', 'W0', 'IN625_fraction', 'mass_g', 'inertia_g_mm2',
               'vm0_mean', 'vm0_max_cell', 'vm_end_mean', 'vm_end_max_cell', 'creep_eq_mean', 'creep_eq_max_cell',
               'T_min', 'T_max', 'T_mean', 'u_max_0', 'u_max_end', 'w_sum_min', 'w_sum_max']
