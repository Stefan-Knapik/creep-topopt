"""Appendix B.3 (Fig. B3): AD/adjoint design sensitivities of the creep compliance against central finite differences.
Writes results/appendix/A3_sensitivity/."""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT_DIR = os.path.join(ROOT, "results", "appendix", "A3_sensitivity")
DATA_DIR = os.path.join(OUT_DIR, "data")
RESULTS_DIR = OUT_DIR
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", os.environ.get("TOGA_PLATFORM", "cpu"))

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
from toga.mat.material_properties import vmap_design_to_physical
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.physics.creep_explicit import CreepExplicit
from toga.physics.creep_implicit import CreepImplicit
from toga.opt.filtering import compute_filter_kd_tree, applyDensityFilter
from toga.opt.penalization import penalize


# --- Sec. 4.1-4.2 cantilever configuration, verbatim -----------------------
E_MOD = 160e3               # MPa
A_CR = 1e-21                # MPa^-n s^-1
N_CR = 3.5
LX, LY = 200.0, 100.0       # mm
TRACTION = 100.0            # MPa, on the lower 20% of the right edge
PENAL_A_EXPONENT = 3.5
RAMP_Q = 8.0
RMIN = 1.5                  # in units of average element size
END_TIME = 3600 * 24 * 365  # 1 year in seconds

# Coarsened mesh, so that a central difference for every design variable stays
# tractable: 800 variables is 1600 forward solves.
COARSE = (40, 20)


def build_problem(method, nx, ny):
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

    dirichlet_bc_info = [[left] * 2, [0, 1], [dirichlet_zero] * 2]

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
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[load_location],
        additional_info=(material_properties,),
    )
    problem.penalize_A = lambda x: 1 / penalize(x, {"RAMP": RAMP_Q}) ** PENAL_A_EXPONENT
    return problem, mesh


def make_objective(method, nx, ny, num_time_steps, end_time=END_TIME, rmin=RMIN):
    """Return (J_jit, grad_jit, num_cells) for the paper's creep-compliance
    objective, including the density filter, as in experiments/2_pareto_vf."""
    problem, mesh = build_problem(method, nx, ny)
    solver_options = {"pardiso_solver": {}}
    H, Hs = compute_filter_kd_tree(problem.fe, rmin)
    ft = {"H": H, "Hs": Hs}

    def J_of_theta(theta):
        geom, mat = vmap_design_to_physical(theta)
        geom_smoothed = applyDensityFilter(ft, geom.reshape(-1, 1))
        params_physical = np.concatenate((geom_smoothed.reshape(-1, 1), mat), axis=1)
        full_sol, _ = problem.forward_solve(
            theta=params_physical,
            end_time=end_time,
            num_time_steps=num_time_steps,
            fwd_solver_options=solver_options,
            vjp_solver_options=solver_options,
        )
        u_full = full_sol[0]
        return problem.compute_compliance(u_full[-1]) - problem.compute_compliance(u_full[0])

    return jax.jit(J_of_theta), jax.jit(jax.grad(J_of_theta)), len(mesh.cells)


def initial_design(num_cells, dummy_seed):
    """Uniform domain"""
    return jax.numpy.ones((num_cells, 1), dtype=jax.numpy.float64) * 0.5

# --------------------------------------------------------------------------
# Check 1: every design variable, coarse mesh
# --------------------------------------------------------------------------

def run_parity(method, num_time_steps=4, fd_step=1e-4, num_samples=None,
               nx=COARSE[0], ny=COARSE[1], theta_seed=0, sample_seed=1):
    J_jit, grad_jit, num_cells = make_objective(method, nx, ny, num_time_steps)
    theta0 = initial_design(num_cells, theta_seed)

    print(f"[{method}] parity: {nx}x{ny} = {num_cells} cells, "
          f"num_time_steps = {num_time_steps}, h = {fd_step:g}")
    t0 = time.perf_counter()
    J0 = float(J_jit(theta0))
    print(f"  J(theta0) = {J0:.8g}   ({time.perf_counter()-t0:.1f}s incl. compile)")
    t0 = time.perf_counter()
    dJ_ad = onp.asarray(grad_jit(theta0)).reshape(-1)
    print(f"  AD gradient ready   ({time.perf_counter()-t0:.1f}s)")

    if num_samples is None or num_samples >= num_cells:
        sample_idx = onp.arange(num_cells)
    else:
        rng = onp.random.default_rng(sample_seed)
        sample_idx = onp.sort(rng.choice(num_cells, size=num_samples, replace=False))

    theta_np = onp.asarray(theta0)
    dJ_fd = onp.zeros(len(sample_idx))
    t0 = time.perf_counter()
    for k, i in enumerate(sample_idx):
        tp, tm = theta_np.copy(), theta_np.copy()
        tp[i, 0] += fd_step
        tm[i, 0] -= fd_step
        dJ_fd[k] = (float(J_jit(np.asarray(tp))) - float(J_jit(np.asarray(tm)))) / (2 * fd_step)
        if (k + 1) % 100 == 0 or k == 0:
            el = time.perf_counter() - t0
            print(f"  {k+1:>4}/{len(sample_idx)}   elapsed {el:6.1f}s   "
                  f"ETA {el/(k+1)*(len(sample_idx)-k-1):6.0f}s")

    df = pd.DataFrame({
        "cell_idx": sample_idx,
        "theta": theta_np[sample_idx, 0],
        "dJ_ad": dJ_ad[sample_idx],
        "dJ_fd": dJ_fd,
    })
    out = os.path.join(RESULTS_DIR, f"sensitivity_{method}.csv")
    df.to_csv(out, index=False)
    print(f"saved {out}")

    meta = pd.DataFrame([{
        "method": method, "nx": nx, "ny": ny, "num_cells": num_cells,
        "num_time_steps": num_time_steps, "end_time_s": END_TIME, "rmin": RMIN,
        "fd_step": fd_step, "num_samples": len(sample_idx), "J0": J0,
        "penal_A_exponent": PENAL_A_EXPONENT,
    }])
    out = os.path.join(RESULTS_DIR, f"sensitivity_{method}_meta.csv")
    meta.to_csv(out, index=False)
    print(f"saved {out}")
    return df


# --------------------------------------------------------------------------
# Check 2: finite-difference step-size sweep
# --------------------------------------------------------------------------

FD_STEPS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7, 1e-8]


def run_fd_step(method, num_time_steps=4, num_probe_cells=8,
                nx=COARSE[0], ny=COARSE[1], theta_seed=0):
    J_jit, grad_jit, num_cells = make_objective(method, nx, ny, num_time_steps)
    theta0 = initial_design(num_cells, theta_seed)
    dJ_ad = onp.asarray(grad_jit(theta0)).reshape(-1)
    theta_np = onp.asarray(theta0)

    # Probe cells spanning the range of gradient magnitudes, so the V-curve is
    # not an artefact of one particularly well- or ill-scaled variable.
    order = onp.argsort(onp.abs(dJ_ad))
    probe = onp.unique(order[onp.linspace(0, num_cells - 1, num_probe_cells).astype(int)])

    rows = []
    for h in FD_STEPS:
        for i in probe:
            tp, tm = theta_np.copy(), theta_np.copy()
            tp[i, 0] += h
            tm[i, 0] -= h
            fd = (float(J_jit(np.asarray(tp))) - float(J_jit(np.asarray(tm)))) / (2 * h)
            rows.append({"method": method, "cell_idx": int(i), "h": h,
                         "dJ_ad": dJ_ad[i], "dJ_fd": fd,
                         "rel_err": abs(fd - dJ_ad[i]) / abs(dJ_ad[i])})
        med = onp.median([r["rel_err"] for r in rows if r["h"] == h])
        print(f"  h = {h:8.1e}   median rel. err = {med:.3e}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, f"fd_step_study_{method}.csv")
    df.to_csv(out, index=False)
    print(f"saved {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["parity", "fd_step"],
                        default="parity")
    parser.add_argument("--method", choices=["explicit", "implicit"], default="implicit")
    parser.add_argument("--num_time_steps", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--fd_step", type=float, default=1e-4)
    args = parser.parse_args()

    if args.check == "parity":
        run_parity(args.method, num_time_steps=args.num_time_steps or 4,
                   fd_step=args.fd_step, num_samples=args.num_samples)
    else:
        run_fd_step(args.method, num_time_steps=args.num_time_steps or 4)
