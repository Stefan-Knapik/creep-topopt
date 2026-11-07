"""Appendix B.1 (Fig. B1): Norton stress relaxation of a 1x1 plate (tension and mixed BCs), the TOGA side of
the Abaqus comparison. Writes results/appendix/A1_abaqus_validation/toga/."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT_DIR = os.path.join(ROOT, "results", "appendix", "A1_abaqus_validation")
DATA_DIR = os.path.join(OUT_DIR, "data")
CSV_DIR = os.path.join(OUT_DIR, "toga")
for _d in (DATA_DIR, CSV_DIR):
    os.makedirs(_d, exist_ok=True)

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

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

from toga.mat import build_material_properties
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.physics.creep_explicit import CreepExplicit
from toga.physics.creep_implicit import CreepImplicit


# Material and geometry: identical to the two Abaqus drivers in abaqus/.
E_MOD = 160e3      # MPa
A_CR = 1e-21       # MPa^-n s^-1
N_CR = 3.5
LX = LY = 1.0
N_ELE = 2
U_BAR = 0.002
END_TIME = 1e10     # seconds, ~317 years


def analytic_relaxation(t, e_total=U_BAR / LX, E=E_MOD, A=A_CR, n=N_CR):
    """Exact uniaxial Norton stress relaxation at constant total strain.

    With sigma = E (e_total - e_c) and de_c/dt = A sigma^n (uniaxial, so the
    von Mises stress equals sigma and the flow direction component is 1),

        d sigma / dt = -E A sigma^n
        sigma(t) = [ sigma_0^(1-n) + (n-1) E A t ]^(1/(1-n)),   sigma_0 = E e_total

    Returns (sigma, creep_strain) with the creep strain from e_c = e_total - sigma/E.
    Applies only to the `tension` case; the `mixed` case is biaxial.
    """
    t = onp.asarray(t, dtype=float)
    sigma_0 = E * e_total
    sigma = (sigma_0 ** (1.0 - n) + (n - 1.0) * E * A * t) ** (1.0 / (1.0 - n))
    return sigma, e_total - sigma / E


def build_problem(case, method):
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
    meshio_mesh = rectangle_mesh(Nx=N_ELE, Ny=N_ELE, domain_x=LX, domain_y=LY)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type=ele_type)

    def left(point):
        return np.isclose(point[0], 0.0, atol=1e-5)

    def right(point):
        return np.isclose(point[0], LX, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0.0, atol=1e-5)

    def zero(point):
        return 0.0

    def u_bar(point):
        return U_BAR

    if case == "tension":
        # ux = 0 on the left, uy = 0 on the bottom, ux = u_bar on the right.
        # Top edge free -> sigma_yy = 0 -> uniaxial.
        dirichlet_bc_info = [[left, bottom, right], [0, 1, 0], [zero, zero, u_bar]]
    elif case == "mixed":
        dirichlet_bc_info = [[left, left, right, right], [0, 1, 0, 1],
                             [zero, zero, u_bar, u_bar]]
    else:
        raise ValueError(f"Unknown case: {case!r}")

    if method == "explicit":
        ProblemCls = CreepExplicit
    elif method == "implicit":
        ProblemCls = CreepImplicit
    else:
        raise ValueError(f"Unknown method: {method!r}")

    problem = ProblemCls(
        mesh, vec=2, dim=2, ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(material_properties,),
    )
    return problem, mesh


def run(case, method, num_time_steps=10, end_time=END_TIME, write_csv=True):
    problem, mesh = build_problem(case, method)
    solver_options = {"pardiso_solver": {}}

    theta = np.ones((len(mesh.cells), 1))
    _, body_params = problem.forward_solve(
        theta=theta,
        end_time=end_time,
        num_time_steps=num_time_steps,
        fwd_solver_options=solver_options,
        vjp_solver_options=solver_options,
    )
    sigma, creep_strain, _, _ = body_params

    # Sampled quadrature point: first quadrature point of the first element.
    cell_id, qp_id = 0, 0
    time = onp.linspace(0.0, end_time, num_time_steps + 1)

    df_sig = pd.DataFrame({
        "time": time,
        "sigma_xx": onp.asarray(sigma[:, cell_id, qp_id, 0, 0]),
        "sigma_yy": onp.asarray(sigma[:, cell_id, qp_id, 1, 1]),
        "sigma_xy": onp.asarray(sigma[:, cell_id, qp_id, 0, 1]),
    })
    df_eps = pd.DataFrame({
        "time": time,
        "creep_strain_xx": onp.asarray(creep_strain[:, cell_id, qp_id, 0, 0]),
        "creep_strain_yy": onp.asarray(creep_strain[:, cell_id, qp_id, 1, 1]),
        "creep_strain_xy": onp.asarray(creep_strain[:, cell_id, qp_id, 0, 1]),
    })

    if write_csv:
        tag = f"{case}_{method}_{num_time_steps}"
        for name, df in (("stress", df_sig), ("creep_strain", df_eps)):
            path = os.path.join(CSV_DIR, f"{name}_{tag}.csv")
            df.to_csv(path, index=False)
            print(f"saved {path}")

    return df_sig, df_eps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["tension", "mixed"], default="tension")
    parser.add_argument("--method", choices=["explicit", "implicit"], default="explicit")
    parser.add_argument("--num_time_steps", type=int, default=10)
    args = parser.parse_args()
    run(args.case, args.method, args.num_time_steps)
