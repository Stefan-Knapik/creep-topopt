"""Is an optimized design explained by stress and temperature? For each run folder given:

1. the uniform blend at the IN625 bound is analysed and the indicator
   m = sigma_vm (rate_SS316 - rate_IN625)(sigma_vm, T) is taken from its state at the end of the hold;
2. a heuristic design puts IN625 in the cells of highest m holding V* of the volume; it goes through
   the same filter as the optimizer's design;
3. capture ratio = (J_uniform - J_heuristic) / (J_uniform - J_optimized): the share of the
   optimizer's gain that the indicator alone reaches;
4. the uniform, heuristic and optimized designs are re-evaluated with thermal expansion off: the
   gain must survive (a design that only relaxes mismatch stress against the load would not).

    python heuristic_design.py results/5_blade/explore/<case>/V30 [more run folders]
    python heuristic_design.py <run> --steps 10      # re-evaluate a design found with 2 steps at 10

Writes <run>/heuristic.json (heuristic_N<steps>.json with --steps) and <run>/heuristic.npz (indicator,
heuristic design before and after the filter, optimized filtered design, centroids).
"""
import argparse
import json
from pathlib import Path

from _common import bp, jax, release
import jax.numpy as np
import numpy as onp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=None, help="time steps of the re-evaluation (default: as the run)")
    args = parser.parse_args()

    for run in args.runs:
        case = json.load(open(run / "case.json"))
        design = np.asarray(onp.load(run / "design_final.npy"))
        steps = case['num_time_steps'] if args.steps is None else args.steps
        result = {'run': str(run), 'num_time_steps': steps, 'vf': case['vf']}
        fields = {}
        for label, expansion in (("", case['expansion']), ("_noexp", False)):
            if label == "_noexp" and not case['expansion']:
                continue
            blade = bp.build_blade_from_case(case, num_time_steps=steps, expansion=expansion)
            evaluate = jax.jit(blade.evaluate)
            uniform = bp.homogeneous(blade.num_cells, case['vf'])
            f_uniform, aux = evaluate(uniform)
            if label == "":
                m, vm_end, vm_0, T_cell = bp.alloy_indicator(blade, aux['body_params'], aux['T'])
                heuristic = bp.top_fraction_design(m, blade.cell_volume, case['vf'])
                fields = dict(indicator=m, vm_end_uniform=vm_end, vm_0_uniform=vm_0, T=T_cell, centroids=blade.centroids,
                              cell_volume=blade.cell_volume, heuristic_raw=onp.asarray(heuristic)[:, 0],
                              heuristic_filtered=onp.asarray(blade.physical(heuristic))[:, 0],
                              optimized_filtered=onp.asarray(blade.physical(design))[:, 0])
            f_heuristic, _ = evaluate(heuristic)
            f_optimized, _ = evaluate(design)
            J = {k: float(f['J']) for k, f in (("uniform", f_uniform), ("heuristic", f_heuristic), ("optimized", f_optimized))}
            result.update({f'J_{k}{label}': v for k, v in J.items()})
            result[f'IN625_heuristic{label}'] = float(f_heuristic['IN625_fraction'])
            result[f'IN625_optimized{label}'] = float(f_optimized['IN625_fraction'])
            result[f'gain_optimized{label}'] = 1.0 - J['optimized'] / J['uniform']
            result[f'gain_heuristic{label}'] = 1.0 - J['heuristic'] / J['uniform']
            result[f'capture_ratio{label}'] = (J['uniform'] - J['heuristic']) / (J['uniform'] - J['optimized'])
            del blade, evaluate
            release()
        # agreement of the two layouts: volume-weighted correlation of the filtered IN625 fields
        v = fields['cell_volume'] / fields['cell_volume'].sum()
        a, b = fields['heuristic_filtered'], fields['optimized_filtered']
        cov = lambda p, q: float(onp.sum(v * (p - onp.sum(v * p)) * (q - onp.sum(v * q))))
        result['layout_correlation'] = cov(a, b) / max(onp.sqrt(cov(a, a) * cov(b, b)), 1e-30)
        suffix = "" if args.steps is None else f"_N{steps}"
        with open(run / f"heuristic{suffix}.json", "w") as f:
            json.dump(result, f, indent=1)
        onp.savez(run / f"heuristic{suffix}.npz", **fields)
        print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
