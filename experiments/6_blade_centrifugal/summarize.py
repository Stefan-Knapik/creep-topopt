"""Rotating blade: tables of the optimization runs and where the optimizer puts the IN625.

Reads the last iterate of every results/6_blade_centrifugal/data_creep_*/ run (scalars.csv and the
last .vtu) and writes, into results/6_blade_centrifugal/:

    graded_vf.csv             IN625 content sweep: V*, J, IN625 fraction used, mass, moment of inertia about
                              the rotation axis, initial von Mises stress (mean and peak), creep strain
    graded_mass.csv           the same for the runs bounded in mass and/or moment of inertia, with the bounds
    design_distribution.csv   IN625 fraction by span slice, by suction / pressure half and by leading /
                              trailing half, for these designs and for the tip-traction designs of
                              experiments/5_blade (results/5_blade/data_creep_*), if present

No analysis is run; needs only numpy, scipy and meshio.

    python summarize.py
    python summarize.py --coarse
"""
import argparse
import csv
import re
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "6_blade_centrifugal"
TIP_TRACTION_RESULTS = ROOT / "results" / "5_blade"
MESH_FILE = ROOT / "toga" / "mesh" / "blade" / "blade_050_000.inp"
MATERIAL_DATA = ROOT / "toga" / "mat" / "material_data"
R_HUB = 250.0  # mm, as in centrifugal_blade.py
SPAN = 50.0
NUM_SLICES = 10
KEYS = ['J', 'IN625_fraction', 'mass_g', 'inertia_g_mm2', 'vm0_mean', 'vm0_max_cell', 'vm0_max_quad',
        'creep_eq_mean', 'creep_eq_max_cell', 'creep_eq_max_quad', 'T_mean', 'u_max_0', 'u_max_end']


def alloy_density(name):
    """Density in g/mm^3."""
    with open(MATERIAL_DATA / f"{name}.csv", newline="") as f:
        return float(next(csv.DictReader(f, skipinitialspace=True))["rho"])


def last_row(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: float(v) for k, v in rows[-1].items()}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written to {path}")


def hex8_volumes(points, cells):
    """Cell volumes by 2x2x2 Gauss quadrature."""
    g = 1.0 / np.sqrt(3.0)
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    x = points[cells]
    volume = np.zeros(len(cells))
    for xi in corners * g:
        dshape = np.stack([corners[:, d] * np.prod(np.delete(1.0 + corners * xi, d, axis=1), axis=1) / 8.0
                           for d in range(3)], axis=1)
        volume += np.abs(np.linalg.det(np.einsum("cni,nj->cij", x, dshape)))
    return volume


def cell_regions(mesh_file):
    """Centroids, volumes and the halves each cell belongs to (nearest named surface of the mesh)."""
    mesh = meshio.read(mesh_file)
    points = np.asarray(mesh.points, dtype=float)
    cells = mesh.cells_dict["hexahedron"]
    centroids = points[cells].mean(axis=1)

    def distance(names):
        nodes = np.unique(np.concatenate([mesh.point_sets[n] for n in names]))
        return cKDTree(points[nodes]).query(centroids)[0]

    suction = distance(["suction_leading_edge", "suction_side", "suction_trailing_edge"]) \
        < distance(["pressure_leading_edge", "pressure_side", "pressure_trailing_edge"])
    leading = distance(["suction_leading_edge", "pressure_leading_edge"]) \
        < distance(["suction_trailing_edge", "pressure_trailing_edge"])
    return centroids, hex8_volumes(points, cells), suction, leading


def distribution(composition, centroids, volume, suction, leading, rho):
    w = composition[:, 0]
    mean = lambda mask: float(w[mask] @ volume[mask] / volume[mask].sum())
    row = {'IN625_fraction': mean(np.ones(len(w), bool))}
    edges = np.linspace(0.0, SPAN, NUM_SLICES + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        row[f'span_{a:.0f}_{b:.0f}mm'] = mean((centroids[:, 1] >= a) & (centroids[:, 1] < b + 1e-9))
    row['suction_half'] = mean(suction)
    row['pressure_half'] = mean(~suction)
    row['leading_half'] = mean(leading)
    row['trailing_half'] = mean(~leading)
    # volume-weighted mean spanwise position of the IN625, and of the blade
    row['IN625_centroid_y_mm'] = float((w * volume) @ centroids[:, 1] / (w @ volume))
    row['blade_centroid_y_mm'] = float(volume @ centroids[:, 1] / volume.sum())
    row['fraction_of_cells_above_0.9'] = float(volume[w > 0.9].sum() / volume.sum())
    row['fraction_of_cells_below_0.1'] = float(volume[w < 0.1].sum() / volume.sum())
    cell_rho = composition @ rho
    row['mass_g'] = float(cell_rho @ volume)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coarse", action="store_true", help="summarize the coarse-mesh runs under coarse/")
    parser.add_argument("--rpm", type=float, default=7000.0, help="rotation speed of the runs (default: 7000)")
    args = parser.parse_args()
    results = RESULTS if args.rpm == 7000.0 else RESULTS / f"rpm_{round(args.rpm)}"
    results = results / "coarse" if args.coarse else results
    mesh_file = RESULTS / "mesh" / "blade_coarse_1p50.inp" if args.coarse else MESH_FILE
    rho = np.array([alloy_density("IN625"), alloy_density("SS316")])

    homo = {}
    if (results / "homogeneous_blends.csv").exists():
        with open(results / "homogeneous_blends.csv") as f:
            homo = {round(float(r['V_star']), 1): {k: float(v) for k, v in r.items()} for r in csv.DictReader(f)}
    mass_range = (homo[0.0]['mass_g'], homo[1.0]['mass_g']) if homo else None
    inertia_range = (homo[0.0]['inertia_g_mm2'], homo[1.0]['inertia_g_mm2']) if homo else None

    vf_rows, mass_rows = [], []
    for run_dir in sorted(results.glob("data_creep_*")):
        if not (run_dir / "scalars.csv").exists():
            continue
        s = last_row(run_dir / "scalars.csv")
        m = re.fullmatch(r"data_creep(?:_(\d+))?(?:_mass_(\d+))?(?:_inertia_(\d+))?", run_dir.name)
        vf, mass, inertia = (int(g) / 100 if g else None for g in m.groups())
        row = {'run': run_dir.name, 'iterations': int(s['iter'])}
        if mass is None and inertia is None:
            row['V_star'] = vf
            if vf in homo:
                row['J_homogeneous_same_V_star'] = homo[vf]['J']
            vf_rows.append({**row, **{k: s[k] for k in KEYS}})
        else:
            row['mass_bound_fraction'] = mass if mass is not None else np.nan
            row['inertia_bound_fraction'] = inertia if inertia is not None else np.nan
            if mass_range:
                row['mass_max_g'] = mass_range[0] + mass * (mass_range[1] - mass_range[0]) if mass is not None else np.nan
                row['inertia_max_g_mm2'] = (inertia_range[0] + inertia * (inertia_range[1] - inertia_range[0])
                                            if inertia is not None else np.nan)
            mass_rows.append({**row, **{k: s[k] for k in KEYS}})
    vf_rows.sort(key=lambda r: r['V_star'])
    write_csv(results / "graded_vf.csv", vf_rows)
    write_csv(results / "graded_mass.csv", mass_rows)

    print(f"\n{'run':34s} {'J [mJ]':>11s} {'IN625':>8s} {'mass [g]':>9s} {'I [g mm^2]':>12s} {'vM mean':>8s} {'vM peak':>8s}")
    for r in vf_rows + mass_rows:
        print(f"{r['run']:34s} {r['J']:11.5g} {r['IN625_fraction']:8.5f} {r['mass_g']:9.4f} {r['inertia_g_mm2']:12.6g} "
              f"{r['vm0_mean']:8.4f} {r['vm0_max_cell']:8.4f}")

    # Where the IN625 is
    centroids, volume, suction, leading = cell_regions(mesh_file)
    rows = []
    studies = [('centrifugal', results)] + ([] if args.coarse else [('tip_traction', TIP_TRACTION_RESULTS)])
    for study, folder in studies:
        for run_dir in sorted(folder.glob("data_creep_*")):
            vtus = sorted((run_dir / "vtk").glob("sol_*.vtu"))
            if not vtus or 'cte' in run_dir.name:
                continue
            composition = np.asarray(meshio.read(vtus[-1]).cell_data["mat"][0], dtype=float)
            rows.append({'study': study, 'run': run_dir.name,
                         **distribution(composition, centroids, volume, suction, leading, rho)})
    write_csv(results / "design_distribution.csv", rows)
    slices = [k for k in rows[0] if k.startswith('span_')] if rows else []
    print(f"\nIN625 fraction by span slice (root -> tip, {NUM_SLICES} slices), suction | pressure half, leading | trailing half")
    for r in rows:
        print(f"{r['study']:12s} {r['run']:30s} " + ' '.join(f"{r[k]:.2f}" for k in slices)
              + f" | {r['suction_half']:.3f} {r['pressure_half']:.3f} | {r['leading_half']:.3f} {r['trailing_half']:.3f}")


if __name__ == "__main__":
    main()
