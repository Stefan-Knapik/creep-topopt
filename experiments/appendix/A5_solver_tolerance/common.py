"""The Sec. 4.1-4.2 cantilever, set up as the optimizer in experiments/2_pareto_vf sees it."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT_DIR = os.path.join(ROOT, "results", "appendix", "A5_solver_tolerance")
DATA_DIR = os.path.join(OUT_DIR, "data")
RESULTS_DIR = OUT_DIR
for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", os.environ.get("TOGA_PLATFORM", "gpu"))

_cache_dir = os.path.join(DATA_DIR, "jax_cache")
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import logging

import jax.numpy as np

import toga
toga.logger.setLevel(logging.WARNING)

from toga.fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from toga.mat import build_material_properties
from toga.opt.penalization import penalize
from toga.physics.creep_implicit import CreepImplicit

# --- Configuration, verbatim from experiments/2_pareto_vf -------------------
E_MOD = 160e3                # MPa
A_CR = 1e-21                 # MPa^-n s^-1
N_CR = 3.5
LX, LY = 200.0, 100.0        # mm
TRACTION = 100.0             # MPa, on the lower 20% of the right edge
PENAL_A_EXPONENT = 3.5
RAMP_Q = 8.0
END_TIME = 3600 * 24 * 365   # 1 year in seconds
NUM_TIME_STEPS = 10
PAPER_NX, PAPER_NY = 200, 100

# The MMA stopping tolerance of the optimization scripts.  Any solver error far below
# this cannot influence which design the optimiser returns, and it is the line
# the figure draws.
MMA_FUN_TOL = 1e-5


def build_cantilever(nx=PAPER_NX, ny=PAPER_NY):
    """The Sec. 4.1-4.2 cantilever, implicit creep scheme."""
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
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type],
                ele_type=ele_type)

    def left(point):
        return np.isclose(point[0], 0.0, atol=1e-5)

    def dirichlet_zero(point):
        return 0.0

    def load_location(point):
        return np.logical_and(
            np.isclose(point[0], LX, atol=1e-5),
            np.isclose(point[1], 0.0, atol=0.2 * LY + 1e-5))

    class LoadedCreep(CreepImplicit):
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
    problem.penalize_A = lambda x: 1 / penalize(x, {"RAMP": RAMP_Q}) ** PENAL_A_EXPONENT
    return problem, mesh


def solid_design(mesh):
    """theta = 1 on every cell; both penalisations are the identity there."""
    return np.ones((len(mesh.cells), 1))


# The intermediate density used for the second, harder design case.  RAMP(0.5)
# with q = 8 is 0.1, so the stiffness drops by 10x and -- because the creep
# prefactor is penalised as 1 / RAMP^3.5 -- the creep prefactor rises by
# 1/0.1^3.5 = 3162x.  The stress field is unchanged (a uniform design scales E
# everywhere equally under a traction load), so the creep strain accumulated
# over the same year is ~3000x larger and the problem is genuinely nonlinear.
GREY_THETA = 0.5


def uniform_design(mesh, value=GREY_THETA):
    """theta = `value` on every cell."""
    return np.full((len(mesh.cells), 1), float(value))
