"""Rotating blade: what the grading exploits. The best graded design of the IN625 content sweep and
the all-IN625 blade are analysed with one property at a time made independent of the composition
(evaluated for pure IN625, keeping its temperature dependence):

    full        the model of the optimization
    k_fixed     thermal conductivity k
    E_fixed     Young's modulus E
    rho_fixed   density in the centrifugal load (load of an all-IN625 blade)
    creep_only  k, E and the load density together, so only the creep parameters differ

For comparison the uniform blend with the IN625 content of the graded design and the saturated
tip-traction design of experiments/5_blade (if present) are analysed as well.
Writes results/6_blade_centrifugal/gain_factors.csv.

    python evaluate_gain_factors.py                    # best run of results/6_blade_centrifugal/data_creep_<V*>
    python evaluate_gain_factors.py --run data_creep_90
    python evaluate_gain_factors.py --coarse
"""

#%% Import modules

import os
import sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # hide PJRT logs
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import csv
import re

import jax.numpy as np
import meshio
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import centrifugal_blade as cb

VARIANTS = {
    'full': (),
    'k_fixed': ('k',),
    'E_fixed': ('E',),
    'rho_fixed': ('rho_load',),
    'creep_only': ('k', 'E', 'rho_load'),
}


def last_composition(run_dir):
    vtu = sorted((run_dir / "vtk").glob("sol_*.vtu"))[-1]
    return np.asarray(meshio.read(vtu).cell_data["mat"][0], dtype=float)  # (cells, 2), rows sum to one


def final_J(run_dir):
    with open(run_dir / "scalars.csv") as f:
        return float(list(csv.DictReader(f))[-1]['J'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="run directory with the graded design (default: lowest J of the V* sweep)")
    parser.add_argument("--coarse", action="store_true", help="use the coarse verification mesh")
    parser.add_argument("--rpm", type=float, default=cb.RPM, help="rotation speed (default: %(default)s rpm)")
    args = parser.parse_args()
    out_dir = cb.results_dir(args.rpm, args.coarse)

    if args.run is None:
        runs = [d for d in out_dir.glob("data_creep_*") if re.fullmatch(r"data_creep_\d+", d.name)]
        run_dir = min(runs, key=final_J)
    else:
        run_dir = out_dir / args.run
    graded = last_composition(run_dir)
    print(f"graded design: {run_dir}")

    rows = []
    for variant, frozen in VARIANTS.items():
        blade = cb.build_blade(cb.coarse_mesh() if args.coarse else cb.MESH_FILE, frozen=frozen,
                               omega=cb.rpm_to_omega(args.rpm))
        evaluate = jax.jit(lambda composition: blade.evaluate(composition)[0])
        fraction = float(blade.problem_u.compute_volume_fraction_of_material(blade.full_theta(graded), mat_idx=0))
        designs = {'IN625': cb.homogeneous(blade.num_cells, 1.0), 'graded': graded}
        if variant == 'full':
            designs['blend_same_content'] = cb.homogeneous(blade.num_cells, fraction)
            designs['SS316'] = cb.homogeneous(blade.num_cells, 0.0)
            tip_traction_run = cb.ROOT / "results" / "5_blade" / "data_creep_90"
            if not args.coarse and tip_traction_run.is_dir():
                designs['tip_traction_design'] = last_composition(tip_traction_run)
        for name, composition in designs.items():
            f = evaluate(composition)
            rows.append({'variant': variant, 'design': name, **{k: float(f[k]) for k in cb.SCALAR_KEYS}})
            print(f"{variant:11s} {name:20s} J = {rows[-1]['J']:.5e} mJ  T_mean = {rows[-1]['T_mean']:.3f} C  "
                  f"vm0_mean = {rows[-1]['vm0_mean']:.4f} MPa")
        jax.clear_caches()

    for r in rows:
        reference = next(q['J'] for q in rows if q['variant'] == r['variant'] and q['design'] == 'IN625')
        r['J_over_J_IN625'] = r['J'] / reference

    out = out_dir / "gain_factors.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'variant':11s} {'design':20s} {'J [mJ]':>12s} {'J / J_IN625':>12s}")
    for r in rows:
        print(f"{r['variant']:11s} {r['design']:20s} {r['J']:12.5e} {r['J_over_J_IN625']:12.5f}")
    print(f"written to {out}")
