"""Replay the constitutive update over the saved displacement histories to recover initial stress
and final creep strain per cell. Writes results/1_creep_elastic/data_evaluation/reconstructed_fields.npz."""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Replay is constitutive-only (no linear solves) -> keep it on CPU so we don't
# grab a GPU that may be busy with real runs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as np
import numpy as onp
import meshio
import pickle

from toga.mat import build_material_properties
from toga.fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from toga.physics.creep_implicit import CreepImplicit
from toga.opt.penalization import penalize
from toga.utils import quad_to_cell_average

ROOT = Path(__file__).resolve().parents[2]  # repository root
data_dir = str(ROOT / "results" / "1_creep_elastic" / "data_evaluation")
# Elastic baseline and the creep-aware V* = 0.5 design from experiments/2_pareto_vf.
design_dirs = {
    "elastic_50": ROOT / "results" / "1_creep_elastic" / "data_elastic_50",
    "creep_50": ROOT / "results" / "2_pareto_vf" / "data_creep_50",
}

# --- Same problem as evaluate_designs_under_creep.py ------------------------

material_properties = build_material_properties(
    material_names=["dummy"],
    property_configs={
        "E": {"model": "constant", "value": 160e3},
        "A": {"model": "constant", "value": 1e-21},
        "n": {"model": "constant", "value": 3.5},
    },
)

ele_type = "QUAD4"
cell_type = get_meshio_cell_type(ele_type)
N = 100
Nx, Ny = 2 * N, N
Lx, Ly = 200.0, 100.0
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type=ele_type)


def left(point):
    return np.isclose(point[0], 0.0, atol=1e-5)


def dirichlet_zero(point):
    return 0.0


dirichlet_bc_info = [[left] * 2, [0, 1], [dirichlet_zero] * 2]


def load_location(point):
    return np.logical_and(
        np.isclose(point[0], Lx, atol=1e-5),
        np.isclose(point[1], 0.0, atol=0.2 * Ly + 1e-5),
    )


class LoadedCreep(CreepImplicit):
    def get_surface_maps(self):
        def traction(u, x, *internal_vars_surface):
            return np.array([0.0, 100.0])

        return [traction]


problem = LoadedCreep(
    mesh,
    vec=2,
    dim=2,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=[load_location],
    additional_info=(material_properties,),
)


def make_penalize_A(exponent, ramp=8.0):
    return lambda x: 1 / penalize(x, {"RAMP": ramp}) ** exponent


problem.penalize_A = make_penalize_A(exponent=3.5, ramp=8.0)

# --- Replay the constitutive update over the displacement history -----------

end_time = 3600 * 24 * 365
num_time_steps = 60
time_vector = np.linspace(0.0, end_time, num_time_steps + 1)
dts_seq = onp.concatenate([[0.0], onp.diff(onp.asarray(time_vector))])

num_cells = len(problem.fe.cells)
num_quads = problem.fe.num_quads


@jax.jit
def constitutive_step(sigmas_old, eps_creep_old, thetas, sol, dt):
    dts = np.full((num_cells, num_quads), dt)
    stress, creep_strain, _, _ = problem.update_stress_strain(
        sol, (sigmas_old, eps_creep_old, thetas, dts)
    )
    return stress, creep_strain


def reconstruct(design_tag, pkl_name):
    vtu = meshio.read(max((design_dirs[design_tag] / "vtk").glob("sol_*.vtu")))  # final design
    geom = onp.asarray(vtu.cell_data["geom"][0])
    theta = geom.reshape(-1, 1)

    with open(os.path.join(data_dir, pkl_name), "rb") as f:
        full_sol = onp.asarray(pickle.load(f))  # (num_steps+1, num_nodes, 2)

    sigmas_old, eps_creep_old, thetas, _ = problem.init_params(theta)

    stress_0 = None
    for i, dt in enumerate(dts_seq):
        sigmas_old, eps_creep_old = constitutive_step(
            sigmas_old, eps_creep_old, thetas, np.asarray(full_sol[i]), float(dt)
        )
        if i == 0:
            stress_0 = sigmas_old

    stress_0_cell = onp.asarray(quad_to_cell_average(problem.fe, stress_0)).reshape(-1, 4)
    creep_strain_final_cell = onp.asarray(
        quad_to_cell_average(problem.fe, eps_creep_old)
    ).reshape(-1, 4)
    return geom, stress_0_cell, creep_strain_final_cell


if __name__ == "__main__":
    out = {}
    for tag, pkl in [
        ("elastic_50", "creep_solution_for_elastic_design.pkl"),
        ("creep_50", "creep_solution_for_creep_design.pkl"),
    ]:
        print(f"Reconstructing {tag} ...")
        geom, sig0, epsf = reconstruct(tag, pkl)
        out[f"{tag}__geom"] = geom
        out[f"{tag}__stress_0"] = sig0
        out[f"{tag}__creep_strain_final"] = epsf

    # Sanity check against the creep-design VTU, which already stores these fields.
    ref = meshio.read(max((design_dirs["creep_50"] / "vtk").glob("sol_*.vtu")))
    ref_sig0 = onp.asarray(ref.cell_data["stress_0"][0])
    ref_epsf = onp.asarray(ref.cell_data["creep_strain_final"][0])
    d_sig = onp.abs(out["creep_50__stress_0"] - ref_sig0).max()
    d_eps = onp.abs(out["creep_50__creep_strain_final"] - ref_epsf).max()
    print(f"[sanity] max |stress_0 - VTU|           = {d_sig:.4e}")
    print(f"[sanity] max |creep_strain_final - VTU| = {d_eps:.4e}")

    out_path = os.path.join(data_dir, "reconstructed_fields.npz")
    onp.savez_compressed(out_path, **out)
    print(f"Saved {out_path}")
