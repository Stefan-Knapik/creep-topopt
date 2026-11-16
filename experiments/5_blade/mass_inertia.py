"""Sec. 4.5: mass and moment of inertia of the graded designs and of the uniform blends of a case.

Reads results/5_blade/<study>/<case>/summary rows (scalars.csv of every V<bound>/ run, written by
optimize_graded_blade.py from the filtered composition) and homogeneous_blends.csv, and writes
<case>/mass_inertia.csv. The moment of inertia is about the rotation axis (parallel to z through
(x, y) = (0, -r_hub)); the density of a blend follows the rule of mixtures. As a check the mass is
recomputed from the last .vtu (mat_filtered) by 2x2x2 Gauss quadrature.

    python mass_inertia.py results/5_blade/deploy/<case>
"""
import argparse
import csv
import json
from pathlib import Path

import meshio
import numpy as np

from summarize import hex8_volumes

ROOT = Path(__file__).resolve().parents[2]
MATERIAL_DATA = ROOT / "toga" / "mat" / "material_data"
ALLOYS = ["IN625", "SS316"]  # order of the composition columns


def alloy_density(name):
    """Density in g/mm^3 (first row of the alloy's CSV; rho is temperature independent)."""
    with open(MATERIAL_DATA / f"{name}.csv", newline="") as f:
        return float(next(csv.DictReader(f, skipinitialspace=True))["rho"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    rho = np.array([alloy_density(a) for a in ALLOYS])
    rows = []
    volume = None
    for run in sorted(args.case.glob("V*")):
        if not (run / "scalars.csv").exists():
            continue
        last = list(csv.DictReader(open(run / "scalars.csv")))[-1]
        vf = json.load(open(run / "case.json"))['vf']
        row = {'design': 'graded', 'V_star': vf, 'IN625_fraction': float(last['IN625_fraction']),
               'mass_g': float(last['mass_g']), 'inertia_g_mm2': float(last['inertia_g_mm2']), 'J': float(last['J'])}
        vtus = sorted((run / "vtk").glob("sol_*.vtu"))
        if vtus:
            mesh = meshio.read(vtus[-1])
            if volume is None:
                cells = next(block.data for block in mesh.cells if block.type == "hexahedron")
                volume = hex8_volumes(np.asarray(mesh.points, dtype=float), cells)
            composition = np.asarray(mesh.cell_data["mat_filtered"][0], dtype=float)
            row['mass_g_from_vtu'] = float((composition @ rho) @ volume)
            row['IN625_fraction_from_vtu'] = float(composition[:, 0] @ volume / volume.sum())
        rows.append(row)
    homo = args.case / "homogeneous_blends.csv"
    if homo.exists():
        for r in csv.DictReader(open(homo)):
            rows.append({'design': 'homogeneous', 'V_star': float(r['V_star']), 'IN625_fraction': float(r['IN625_fraction']),
                         'mass_g': float(r['mass_g']), 'inertia_g_mm2': float(r['inertia_g_mm2']), 'J': float(r['J'])})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = args.case / "mass_inertia.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'design':12s} {'V*':>4s} {'IN625':>7s} {'mass [g]':>9s} {'I [g mm^2]':>12s} {'J [mJ]':>10s}")
    for r in rows:
        print(f"{r['design']:12s} {r['V_star']:4.1f} {r['IN625_fraction']:7.4f} {r['mass_g']:9.3f} {r['inertia_g_mm2']:12.1f} {r['J']:10.5g}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
