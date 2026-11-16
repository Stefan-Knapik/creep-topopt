"""Forward-only screen of load cases of the rotating blade: which root stress, shroud share and hold
time give a problem inside the range of the creep data and of a small-strain model.

For every case (root stress x shroud share x hold time) the uniform blends with IN625 fraction
0, 0.5 and 1 are analysed, at --steps time steps and, with --check-steps, at a finer step count.
One CSV row per case goes to results/5_blade/<study>/screen/<case>.csv with, for each blend b:

    J_b, J_b_check              creep compliance [mJ] at the two step counts (nan: the solve failed)
    creep_peak_b, creep_mean_b  peak (cell average) and volume mean of the equivalent creep strain
    vm0_peak_b, vm0_mean_b      von Mises stress at t = 0 [MPa]
    diss60_b                    share of the creep dissipation from points whose stress at t = 0 is at
                                least 60 MPa (the lower end of the tested stresses)

and <case>_indicator.npz holds, per cell of the 50/50 blend, the indicator
m = sigma_vm (rate_SS316 - rate_IN625)(sigma_vm, T) (blade_problem.alloy_indicator), sigma_vm and T.

    python screen_forward.py --root-stress 60 80 100 --shroud 0 0.3 --hours 10 30 100 --check-steps 40
"""
import argparse
import itertools

from _common import bp, jax, release, write_csv, case_kwargs
import jax.numpy as np
import numpy as onp

from optimize_graded_blade import add_case_arguments

BLENDS = (("SS316", 0.0), ("blend", 0.5), ("IN625", 1.0))


def analyse(blade):
    """Jitted composition -> scalars of one forward analysis with the history saved."""
    problem = blade.problem_u
    JxW = problem.fe.JxW

    def measures(composition):
        f, aux = blade.evaluate(composition)
        sigmas, eps_creep = aux['body_params'][0], aux['body_params'][1]
        # backward-Euler creep dissipation of every quadrature point, sum over the steps of sigma_k : d eps_c,k
        dissipation = np.sum(np.sum(sigmas[1:] * np.diff(eps_creep, axis=0), axis=(-2, -1)), axis=0) * JxW
        tested = bp.von_mises(sigmas[0]) >= 60.0
        f['diss60'] = np.sum(np.where(tested, dissipation, 0.0)) / np.sum(dissipation)
        return f, aux
    return jax.jit(measures)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_case_arguments(parser)
    for action in parser._actions:   # these three take lists here
        if action.dest in ("root_stress", "shroud", "hours"):
            action.nargs = "+"
            action.default = [action.default]
    parser.add_argument("--check-steps", type=int, default=None, help="second, finer step count for J")
    args = parser.parse_args()
    kwargs = case_kwargs(args)
    mesh = bp.mesh_name(kwargs['elem_size'], args.fine)

    for root_stress, shroud, hours in itertools.product(args.root_stress, args.shroud, args.hours):
        name = bp.case_name(root_stress, shroud, hours, radius_mm=kwargs['radius_mm'], mesh=mesh,
                            num_time_steps=kwargs['num_time_steps'], alpha_mode=args.alpha_mode,
                            expansion=kwargs['expansion'], cte_mismatch_scale=args.cte_mismatch_scale)
        out = bp.RESULTS / args.study / "screen"
        row = {'case': name, 'root_stress': root_stress, 'shroud': shroud, 'hours': hours,
               'num_time_steps': kwargs['num_time_steps'], 'check_steps': args.check_steps}
        for label, steps in (("", kwargs['num_time_steps']), ("_check", args.check_steps)):
            if steps is None:
                continue
            blade = bp.build_blade(**{**kwargs, 'num_time_steps': steps}, root_stress=root_stress, shroud=shroud,
                                   end_time=3600.0 * hours, save_history=True)
            row.update({'num_cells': blade.num_cells, 'rpm': blade.rpm, 'tip_speed_m_s': blade.tip_speed,
                        'tip_traction': blade.tip_traction})
            measures = analyse(blade)
            for blend, fraction in BLENDS:
                try:
                    f, aux = measures(bp.homogeneous(blade.num_cells, fraction))
                    f = {k: float(v) for k, v in f.items()}
                except Exception as error:   # a diverged solve may raise in the linear solver
                    print(f"[screen] {name} {blend}: {type(error).__name__}: {error}")
                    f = {}
                row[f'J_{blend}{label}'] = f.get('J', float('nan'))
                if label == "":
                    row.update({f'creep_peak_{blend}': f.get('creep_eq_max_cell', float('nan')),
                                f'creep_mean_{blend}': f.get('creep_eq_mean', float('nan')),
                                f'vm0_peak_{blend}': f.get('vm0_max_cell', float('nan')),
                                f'vm0_mean_{blend}': f.get('vm0_mean', float('nan')),
                                f'vm_end_peak_{blend}': f.get('vm_end_max_cell', float('nan')),
                                f'diss60_{blend}': f.get('diss60', float('nan'))})
                    if blend == "blend" and f:
                        m, vm_end, vm_0, T_cell = bp.alloy_indicator(blade, aux['body_params'], aux['T'])
                        out.mkdir(parents=True, exist_ok=True)
                        onp.savez(out / f"{name}_indicator.npz", indicator=m, vm_end=vm_end, vm_0=vm_0, T=T_cell,
                                  centroids=blade.centroids, cell_volume=blade.cell_volume)
                print(f"[screen] {name} N={steps} {blend:6s} J {row[f'J_{blend}{label}']:.5g} mJ", flush=True)
            del blade, measures
            release()
        if args.check_steps:
            for blend, _ in BLENDS:
                row[f'J_rel_check_{blend}'] = row[f'J_{blend}'] / row[f'J_{blend}_check'] - 1.0
        row['J_SS316_over_IN625'] = row['J_SS316'] / row['J_IN625']
        write_csv(out / f"{name}.csv", [row])


if __name__ == "__main__":
    main()
