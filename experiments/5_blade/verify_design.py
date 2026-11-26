"""Checks of Sec. 4.5 that need an optimized design. Writes into results/5_blade/verification/.

    python verify_design.py results/5_blade/deploy/<case>/V30                    # both checks
    python verify_design.py <run> --checks measures     # design_measures.json
    python verify_design.py <run> --checks explicit     # explicit_vs_implicit_25k.csv

measures  The optimized design, the uniform blend at its IN625 bound and the two monolithic blades,
          on the mesh and settings of the run: the scalars of blade_problem.SCALAR_KEYS and
              diss60              share of the creep dissipation from points whose von Mises stress at
                                  t = 0 is at least 60 MPa (the lower end of the tested stresses)
              diss_over_J         creep dissipation / creep compliance
              vol_share_vm0_ge60  volume share of those points
explicit  Explicit against implicit time integration on the 24,978-element mesh (--elem-size 0.72), with
          and without thermal expansion, for both alloys, the uniform blend and the optimized design
          (moved to the coarser mesh by nearest cell centroid). A diverged solve gives nan or a
          negative J.
"""
import argparse
import json
from pathlib import Path

from _common import bp, jax, release, write_csv
import jax.numpy as np
import numpy as onp
from scipy.spatial import cKDTree

OUT = bp.RESULTS / "verification"


def check_measures(run, case, design):
    blade = bp.build_blade_from_case(case, save_history=True)
    JxW = blade.problem_u.fe.JxW

    @jax.jit
    def measures(composition):
        f, aux = blade.evaluate(composition)
        sigmas, eps_creep = aux['body_params'][0], aux['body_params'][1]
        # backward-Euler creep dissipation of every quadrature point, sum over the steps of sigma_k : d eps_c,k
        dissipation = np.sum(np.sum(sigmas[1:] * np.diff(eps_creep, axis=0), axis=(-2, -1)), axis=0) * JxW
        tested = bp.von_mises(sigmas[0]) >= 60.0
        f['diss60'] = np.sum(np.where(tested, dissipation, 0.0)) / np.sum(dissipation)
        f['diss_over_J'] = np.sum(dissipation) / f['J']
        f['vol_share_vm0_ge60'] = np.sum(np.where(tested, JxW, 0.0)) / np.sum(JxW)
        return f

    designs = {"design": design, "uniform": bp.homogeneous(blade.num_cells, case['vf']),
               "SS316": bp.homogeneous(blade.num_cells, 0.0), "IN625": bp.homogeneous(blade.num_cells, 1.0)}
    name = "/".join(run.resolve().parts[-2:])
    result = {}
    for label, composition in designs.items():
        f = {k: float(v) for k, v in sorted(measures(composition).items())}
        result[f"{name}/{label}"] = f
        print(f"[measures] {label:8s} J {f['J']:.6g} mJ  creep peak {f['creep_eq_max_cell']:.3e}  diss60 {f['diss60']:.4f}  "
              f"vm0 mean {f['vm0_mean']:.1f} peak {f['vm0_max_cell']:.1f} MPa", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "design_measures.json", "w") as file:
        json.dump(result, file, indent=1)
    print(f"wrote {OUT / 'design_measures.json'}")
    del blade, measures
    release()


def check_explicit(case, design, elem_size, implicit_steps, explicit_steps):
    fine = bp.build_blade_from_case(case)
    centroids_fine = fine.centroids
    del fine
    release()
    rows = []
    for expansion in (True, False):
        for scheme, steps_list in (("implicit", implicit_steps), ("explicit", explicit_steps)):
            for steps in steps_list:
                blade = bp.build_blade_from_case(case, elem_size=elem_size, mesh_file=None, scheme=scheme,
                                                 num_time_steps=steps, expansion=expansion)
                nearest = cKDTree(centroids_fine).query(blade.centroids)[1]
                designs = {"SS316": bp.homogeneous(blade.num_cells, 0.0), "IN625": bp.homogeneous(blade.num_cells, 1.0),
                           f"blend_{case['vf']:g}": bp.homogeneous(blade.num_cells, case['vf']),
                           f"design_V{round(100 * case['vf']):02d}": np.asarray(onp.asarray(design)[nearest])}
                evaluate = jax.jit(lambda c: blade.evaluate(c)[0])
                for label, composition in designs.items():
                    try:
                        f = {k: float(v) for k, v in evaluate(composition).items()}
                    except Exception as error:   # a diverged solve may raise in the linear solver
                        print(f"[explicit] {scheme} N={steps} {label}: {type(error).__name__}: {error}")
                        f = {}
                    nan = float('nan')
                    rows.append({'expansion': expansion, 'scheme': scheme, 'steps': steps, 'cells': blade.num_cells,
                                 'design': label, 'J': f.get('J', nan), 'creep_peak': f.get('creep_eq_max_cell', nan),
                                 'vm_end_max': f.get('vm_end_max_cell', nan)})
                    print(f"[explicit] expansion {expansion!s:5s} {scheme} N={steps:4d} {label:11s} J {rows[-1]['J']: .6g} mJ  "
                          f"creep peak {rows[-1]['creep_peak']:.3e}", flush=True)
                del blade, evaluate
                release()
    reference = {(r['expansion'], r['design']): r['J'] for r in rows
                 if r['scheme'] == "implicit" and r['steps'] == max(implicit_steps)}
    for r in rows:
        r['rel_to_implicit_finest'] = r['J'] / reference[(r['expansion'], r['design'])] - 1.0
        # stable: finite, positive and within 10 % of the finest implicit solve
        r['stable'] = bool(onp.isfinite(r['J']) and r['J'] > 0.0 and abs(r['rel_to_implicit_finest']) < 0.1)
    write_csv(OUT / "explicit_vs_implicit_25k.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run folder of optimize_graded_blade.py (case.json, design_final.npy)")
    parser.add_argument("--checks", nargs="+", default=["measures", "explicit"], choices=["measures", "explicit"])
    parser.add_argument("--elem-size", type=float, default=0.72, help="mesh of the explicit check (0.72: 24,978 HEX8)")
    parser.add_argument("--implicit-steps", type=int, nargs="+", default=[160, 10])
    parser.add_argument("--explicit-steps", type=int, nargs="+", default=[10, 40, 160])
    args = parser.parse_args()
    case = json.load(open(args.run / "case.json"))
    design = np.asarray(onp.load(args.run / "design_final.npy"))
    if "measures" in args.checks:
        check_measures(args.run, case, design)
    if "explicit" in args.checks:
        check_explicit(case, design, args.elem_size, args.implicit_steps, args.explicit_steps)
