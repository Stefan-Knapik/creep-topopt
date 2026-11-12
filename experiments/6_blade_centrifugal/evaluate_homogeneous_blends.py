"""Rotating blade: creep compliance of uniform IN625/SS316 blends, IN625 fraction 0.0 ... 1.0, under
centrifugal loading. Forward analyses only. Writes results/6_blade_centrifugal/homogeneous_blends.csv
(J, mass, moment of inertia, initial von Mises stress, creep strain) and the root reaction and
operating point to results/6_blade_centrifugal/operating_point.csv.

    python evaluate_homogeneous_blends.py
    python evaluate_homogeneous_blends.py --coarse     # coarse verification mesh, written under coarse/
    python evaluate_homogeneous_blends.py --rpm 8500   # another rotation speed, written under rpm_8500/
"""

#%% Import modules

import os
import sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
# Pass large closed-over arrays (mesh data, sparsity indices) to XLA as arguments instead of
# embedding them as constants. Must be set before importing jax.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import csv

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import centrifugal_blade as cb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coarse", action="store_true", help="use the coarse verification mesh")
    parser.add_argument("--rpm", type=float, default=cb.RPM, help="rotation speed (default: %(default)s rpm)")
    args = parser.parse_args()
    omega = cb.rpm_to_omega(args.rpm)
    out_dir = cb.results_dir(args.rpm, args.coarse)
    out_dir.mkdir(parents=True, exist_ok=True)

    blade = cb.build_blade(cb.coarse_mesh() if args.coarse else cb.MESH_FILE, omega=omega)
    problem_u = blade.problem_u
    evaluate = jax.jit(lambda composition: blade.evaluate(composition)[0])

    rows = []
    for i in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        f = evaluate(cb.homogeneous(blade.num_cells, i))
        rows.append({'V_star': i, **{k: float(f[k]) for k in cb.SCALAR_KEYS}})
        print(f"IN625 fraction {i:.1f}: J = {rows[-1]['J']:.5g} mJ, mass = {rows[-1]['mass_g']:.5g} g, "
              f"von Mises mean / peak = {rows[-1]['vm0_mean']:.5g} / {rows[-1]['vm0_max_cell']:.5g} MPa")
    rows.sort(key=lambda r: r['V_star'])

    out = out_dir / "homogeneous_blends.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written to {out}")

    # Operating point: resultant of the body force, nominal root stress and the simple estimate
    # rho omega^2 (r_hub L + L^2 / 2) for a prismatic bar.
    bottom_area = None
    points = onp.asarray(blade.mesh.points)
    op_rows = []
    for name, w in (("IN625", 1.0), ("blend_0.5", 0.5), ("SS316", 0.0)):
        theta = blade.full_theta(cb.homogeneous(blade.num_cells, w))
        force = onp.asarray(problem_u.compute_total_force(theta))
        rho = (w * 8.44e-3 + (1 - w) * 7.95e-3) * cb.G_TO_TONNE
        if bottom_area is None:
            bottom_area = blade.volume / (points[:, 1].max() - points[:, 1].min())  # mean section, prismatic blade
        op_rows.append({
            'design': name, 'rpm': args.rpm, 'omega_rad_s': omega, 'r_hub_mm': cb.R_HUB,
            'tip_speed_m_s': omega * (cb.R_HUB + cb.SPAN) * 1e-3,
            'F_x_N': force[0], 'F_y_N': force[1], 'section_area_mm2': bottom_area,
            'root_stress_nominal_MPa': force[1] / bottom_area,
            'root_stress_estimate_MPa': rho * omega**2 * (cb.R_HUB * cb.SPAN + cb.SPAN**2 / 2),
        })
        print(op_rows[-1])
    out = out_dir / "operating_point.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(op_rows[0]))
        writer.writeheader()
        writer.writerows(op_rows)
    print(f"written to {out}")
