"""Sec. 4.5: creep compliance of the blade made of a uniform IN625/SS316 blend, IN625 fraction
0, 0.1, ..., 1, under the conditions of a case: the baseline of the graded designs.

    python evaluate_homogeneous_blends.py --fine --study deploy --root-stress 100 --hours 100

Writes results/5_blade/<study>/<case>/homogeneous_blends.csv (V_star, J and the scalars of
blade_problem.SCALAR_KEYS; mass in g, moment of inertia about the rotation axis in g mm^2).
"""
import argparse

from _common import bp, jax, write_csv, case_kwargs
import numpy as onp

from optimize_graded_blade import add_case_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_case_arguments(parser)
    parser.add_argument("--fractions", type=float, nargs="+", default=[i / 10 for i in range(11)])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    kwargs = case_kwargs(args)
    blade = bp.build_blade(**kwargs, root_stress=args.root_stress, shroud=args.shroud, end_time=3600.0 * args.hours)
    name = bp.case_name(args.root_stress, args.shroud, args.hours, radius_mm=kwargs['radius_mm'],
                        mesh=bp.mesh_name(kwargs['elem_size'], args.fine), num_time_steps=kwargs['num_time_steps'],
                        alpha_mode=args.alpha_mode, expansion=kwargs['expansion'],
                        cte_mismatch_scale=args.cte_mismatch_scale, tag=args.tag)
    evaluate = jax.jit(lambda c: blade.evaluate(c)[0])
    rows = []
    for fraction in args.fractions:
        f = evaluate(bp.homogeneous(blade.num_cells, fraction))
        rows.append({'V_star': fraction, **{k: float(f[k]) for k in bp.SCALAR_KEYS}})
        print(f"IN625 {fraction:.1f}: J {rows[-1]['J']:.6g} mJ, mass {rows[-1]['mass_g']:.3f} g, "
              f"vm0 mean {rows[-1]['vm0_mean']:.1f} peak {rows[-1]['vm0_max_cell']:.1f} MPa, "
              f"creep peak {rows[-1]['creep_eq_max_cell']:.2e}", flush=True)
    write_csv(bp.RESULTS / args.study / name / "homogeneous_blends.csv", rows)


if __name__ == "__main__":
    main()
