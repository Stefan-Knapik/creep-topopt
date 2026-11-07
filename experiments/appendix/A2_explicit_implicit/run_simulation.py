"""Appendix B.2 (Fig. B2): time-step refinement of the explicit and implicit creep schemes on the B.1 coupon.
Writes results/appendix/A2_explicit_implicit/."""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT_DIR = os.path.join(ROOT, "results", "appendix", "A2_explicit_implicit")
DATA_DIR = os.path.join(OUT_DIR, "data")
RESULTS_DIR = OUT_DIR
TRAJ_DIR = os.path.join(RESULTS_DIR, "traj")
for _d in (DATA_DIR, RESULTS_DIR, TRAJ_DIR):
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



E_MOD = 160e3      # MPa
A_CR = 1e-21       # MPa^-n s^-1
N_CR = 3.5
LX = LY = 1.0
N_ELE = 2
U_BAR = 0.002
END_TIME = 1e10     # seconds


def analytic_relaxation(t, e_total=U_BAR / LX, E=E_MOD, A=A_CR, n=N_CR):
    """Exact uniaxial Norton relaxation at constant total strain (tension case).

    sigma(t) = [sigma_0^(1-n) + (n-1) E A t]^(1/(1-n)) with sigma_0 = E e_total.
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


def run(case, method, num_time_steps, end_time=END_TIME):
    problem, mesh = build_problem(case, method)
    solver_options = {"pardiso_solver": {}}

    t0 = time.perf_counter()
    _, body_params = problem.forward_solve(
        theta=np.ones((len(mesh.cells), 1)),
        end_time=end_time,
        num_time_steps=num_time_steps,
        fwd_solver_options=solver_options,
        vjp_solver_options=solver_options,
    )
    wall = time.perf_counter() - t0

    sigma, creep_strain, _, _ = body_params
    cell_id, qp_id = 0, 0
    sig = onp.asarray(sigma[:, cell_id, qp_id])
    eps = onp.asarray(creep_strain[:, cell_id, qp_id])
    t = onp.linspace(0.0, end_time, num_time_steps + 1)

    df = pd.DataFrame({
        "time": t,
        "sigma_xx": sig[:, 0, 0], "sigma_yy": sig[:, 1, 1], "sigma_xy": sig[:, 0, 1],
        "creep_strain_xx": eps[:, 0, 0], "creep_strain_yy": eps[:, 1, 1],
        "creep_strain_xy": eps[:, 0, 1],
    })
    out = os.path.join(TRAJ_DIR, f"{case}_{method}_{num_time_steps}.csv")
    df.to_csv(out, index=False)

    # Both schemes can produce non-finite states when the step size exceeds the
    # explicit stability limit; neither solver raises, so this must be checked.
    stable = bool(onp.all(onp.isfinite(sig)) and onp.all(onp.isfinite(eps)))

    summary = {
        "case": case, "method": method, "num_time_steps": num_time_steps,
        "dt_s": end_time / num_time_steps, "end_time_s": end_time,
        "sigma_xx_end": float(sig[-1, 0, 0]),
        "sigma_yy_end": float(sig[-1, 1, 1]),
        "sigma_xy_end": float(sig[-1, 0, 1]),
        "creep_strain_xx_end": float(eps[-1, 0, 0]),
        "stable": stable, "wall_time_s": wall,
    }
    if case == "tension":
        s_ex, e_ex = analytic_relaxation(end_time)
        summary["sigma_xx_exact"] = float(s_ex)
        summary["err_vs_exact"] = abs(summary["sigma_xx_end"] - float(s_ex)) / float(s_ex)
        summary["err_eps_vs_exact"] = abs(
            summary["creep_strain_xx_end"] - float(e_ex)) / float(e_ex)
    print(f"  {case:8s} {method:8s} N={num_time_steps:<5d} "
          f"sigma_xx(t_end) = {summary['sigma_xx_end']:.6f}  "
          f"({wall:.1f}s)")
    return summary


NUM_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_REF = 8192  # implicit reference for the mixed case


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["tension", "mixed"], default="tension")
    parser.add_argument("--method", choices=["explicit", "implicit"], default="implicit")
    parser.add_argument("--num_time_steps", type=int, default=10)
    args = parser.parse_args()
    print(run(args.case, args.method, args.num_time_steps))
