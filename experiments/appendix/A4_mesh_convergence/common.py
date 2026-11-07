"""Shared setup and quantities of interest for the mesh convergence study: the Sec. 4.1-4.2 cantilever
on the solid domain (theta = 1)."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT_DIR = os.path.join(ROOT, "results", "appendix", "A4_mesh_convergence")
DATA_DIR = os.path.join(OUT_DIR, "data")
RESULTS_DIR = OUT_DIR
for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

import jax
jax.config.update("jax_enable_x64", True)
# The local return map is a vmapped Newton over every quadrature point and is
# the part that benefits from the GPU; the linear solve is host-side PARDISO
# behind a jax.pure_callback and pays a device<->host round trip either way.
# TOGA_PLATFORM=cpu overrides, which is the fallback if the finest mesh in the
# ladder does not fit in device memory.
jax.config.update("jax_platform_name", os.environ.get("TOGA_PLATFORM", "gpu"))

_cache_dir = os.path.join(DATA_DIR, "jax_cache")
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import jax.numpy as np
import numpy as onp
import pandas as pd
import logging

import toga
toga.logger.setLevel(logging.WARNING)

from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.mat import build_material_properties
from toga.opt.penalization import penalize
from toga.physics.creep_explicit import CreepExplicit
from toga.physics.creep_implicit import CreepImplicit
from toga.utils_for_creep import pad_to_3x3, safe_divide, safe_sqrt, trim_to_dim


# --- Sec. 4.1-4.2 cantilever configuration, verbatim -----------------------
E_MOD = 160e3               # MPa
A_CR = 1e-21                # MPa^-n s^-1
N_CR = 3.5
LX, LY = 200.0, 100.0       # mm
TRACTION = 100.0            # MPa, on the lower 20% of the right edge
PENAL_A_EXPONENT = 3.5
RAMP_Q = 8.0
END_TIME = 3600 * 24 * 365  # 1 year in seconds
NUM_TIME_STEPS = 10
SOLVER_OPTIONS = {"pardiso_solver": {}}

# The paper's mesh: 200 x 100 elements over 200 x 100 mm, i.e. h = 1 mm.
PAPER_NX, PAPER_NY = 200, 100


def build_cantilever(nx, ny, method, penal_A_exponent=PENAL_A_EXPONENT):
    """The Sec. 4.1-4.2 cantilever at an arbitrary resolution."""
    material_properties = build_material_properties(
        material_names=["dummy"],
        property_configs={
            "E": {"model": "constant", "value": E_MOD},
            "A": {"model": "constant", "value": A_CR},
            "n": {"model": "constant", "value": N_CR},
        },
    )

    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=LX, domain_y=LY)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type=ele_type)

    def left(point):
        return np.isclose(point[0], 0.0, atol=1e-5)

    def dirichlet_zero(point):
        return 0.0

    def load_location(point):
        return np.logical_and(
            np.isclose(point[0], LX, atol=1e-5),
            np.isclose(point[1], 0.0, atol=0.2 * LY + 1e-5))

    BaseCls = {"explicit": CreepExplicit, "implicit": CreepImplicit}[method]

    class LoadedCreep(BaseCls):
        def get_surface_maps(self):
            def traction(u, x, *internal_vars_surface):
                return np.array([0.0, TRACTION])
            return [traction]

    problem = LoadedCreep(
        mesh, vec=2, dim=2, ele_type=ele_type,
        dirichlet_bc_info=[[left] * 2, [0, 1], [dirichlet_zero] * 2],
        location_fns=[load_location],
        additional_info=(material_properties,),
    )
    problem.penalize_A = lambda x: 1 / penalize(x, {"RAMP": RAMP_Q}) ** penal_A_exponent
    return problem, mesh


def solid_design(mesh):
    """theta = 1 on every cell.  Both penalisations are the identity here
    (RAMP(1) = 1), so the analysis sees the bulk material."""
    return np.ones((len(mesh.cells), 1))


# --------------------------------------------------------------------------
# Quantities of interest
# --------------------------------------------------------------------------

def creep_rate(sigma, A, n, dim=2):
    """Norton creep rate at one quadrature point: A sigma_vm^n (3/2) s_dev/sigma_vm.

    CreepImplicit exposes this as `creep_rate_from_sigma`, but CreepExplicit
    inlines the same expression, so it is written out once here (with the same
    safe_sqrt / safe_divide guards and the same 3x3 padding) and used for both
    schemes in the return-map residual check.
    """
    s3 = pad_to_3x3(sigma)
    dev = s3 - (np.trace(s3) / 3.0) * np.eye(3)
    s_norm = safe_sqrt(1.5 * np.sum(dev * dev))
    n_d = trim_to_dim(1.5 * safe_divide(dev, s_norm), dim)
    return A * s_norm**n * n_d


def von_mises(sigma):
    """von Mises stress from an in-plane (..., 2, 2) stress tensor.

    Padded to 3x3 first (plane stress, so sigma_33 = 0) exactly as
    CreepImplicit.creep_rate_from_sigma does, so this is consistent with what
    the constitutive model itself sees.
    """
    s3 = jax.vmap(jax.vmap(pad_to_3x3))(sigma)
    dev = s3 - (np.trace(s3, axis1=-2, axis2=-1)[..., None, None] / 3.0) * np.eye(3)
    return safe_sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))


def eq_creep_strain(eps_cr):
    """Equivalent (von Mises) creep strain from an in-plane (..., 2, 2) tensor.

    Only the in-plane components are stored -- trim_to_dim drops the 33 entry of
    the flow direction -- but Norton creep is deviatoric and hence
    incompressible, so e33 = -(e11 + e22).  Omitting it under-reports the
    equivalent strain by tens of percent.
    """
    e11 = eps_cr[..., 0, 0]
    e22 = eps_cr[..., 1, 1]
    e12 = eps_cr[..., 0, 1]
    e33 = -(e11 + e22)
    return np.sqrt(2.0 / 3.0 * (e11**2 + e22**2 + e33**2 + 2.0 * e12**2))


def vol_avg(problem, field_qp, weights=None):
    """Volume average of a per-quadrature-point scalar field."""
    JxW = problem.fe.JxW
    w = JxW if weights is None else JxW * weights
    return float(np.sum(field_qp * w) / np.sum(w))


def p_norm(problem, field_qp, p=8):
    """Volume-weighted p-norm; a mesh-convergent surrogate for the peak value,
    which is not convergent because the load patch edge and the clamped corners
    are stress singularities."""
    JxW = problem.fe.JxW
    vol = np.sum(JxW)
    return float((np.sum(field_qp**p * JxW) / vol) ** (1.0 / p))


def node_index(problem, x, y):
    """Index of the mesh node closest to (x, y)."""
    pts = onp.asarray(problem.fe.points)
    return int(onp.argmin((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2))


def append_row(path, row):
    """Append one dict as a CSV row, writing the header on first use."""
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
