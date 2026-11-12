"""Sec. 4.5: mass and moment of inertia of the graded blade designs and of the uniform blends.

Reads the last .vtu of each results/5_blade/data_creep_{10,...,90}/ run and writes
results/5_blade/mass_inertia.csv. The density of a blend follows the rule of mixtures of the
`rho` column in toga/mat/material_data. The moment of inertia is taken about the chordwise (z) axis
through the root, I = int rho (x^2 + y^2) dV, with y the spanwise distance from the root.
"""
import csv
from pathlib import Path

import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "5_blade"
MATERIAL_DATA = ROOT / "toga" / "mat" / "material_data"
ALLOYS = ["IN625", "SS316"]  # order of the `mat` columns in the .vtu


def alloy_density(name):
    """Density in g/mm^3 (first row of the alloy's CSV; rho is temperature independent)."""
    with open(MATERIAL_DATA / f"{name}.csv", newline="") as f:
        return float(next(csv.DictReader(f, skipinitialspace=True))["rho"])


def hex8_volume_integrals(points, cells):
    """Per-cell volume and integral of x^2 + y^2, by 2x2x2 Gauss quadrature."""
    g = 1.0 / np.sqrt(3.0)
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    x = points[cells]  # (cells, 8, 3)
    volume = np.zeros(len(cells))
    second_moment = np.zeros(len(cells))
    for xi in corners * g:
        shape = np.prod(1.0 + corners * xi, axis=1) / 8.0  # (8,)
        dshape = np.stack([corners[:, d] * np.prod(np.delete(1.0 + corners * xi, d, axis=1), axis=1) / 8.0
                           for d in range(3)], axis=1)  # (8, 3)
        jac = np.einsum("cni,nj->cij", x, dshape)
        det = np.abs(np.linalg.det(jac))
        xq = np.einsum("n,cni->ci", shape, x)
        volume += det
        second_moment += det * (xq[:, 0] ** 2 + xq[:, 1] ** 2)
    return volume, second_moment


def main():
    rho = np.array([alloy_density(a) for a in ALLOYS])
    rows = []
    volume = second_moment = None

    for vf in range(10, 100, 10):
        vtu = sorted((RESULTS / f"data_creep_{vf}" / "vtk").glob("sol_*.vtu"))[-1]
        mesh = meshio.read(vtu)
        if volume is None:
            cells = next(block.data for block in mesh.cells if block.type == "hexahedron")
            volume, second_moment = hex8_volume_integrals(np.asarray(mesh.points, dtype=float), cells)
        composition = np.asarray(mesh.cell_data["mat"][0], dtype=float)  # (cells, 2), rows sum to one
        cell_rho = composition @ rho
        rows.append(("graded", vf / 100, composition[:, 0] @ volume / volume.sum(),
                     cell_rho @ volume, cell_rho @ second_moment))

    for i in range(11):
        w = i / 10
        blend_rho = w * rho[0] + (1 - w) * rho[1]
        rows.append(("homogeneous", w, w, blend_rho * volume.sum(), blend_rho * second_moment.sum()))

    out = RESULTS / "mass_inertia.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["design", "V_star", "IN625_fraction", "mass_g", "inertia_g_mm2"])
        writer.writerows(rows)

    print(f"blade volume {volume.sum():.1f} mm^3")
    print(f"{'design':12s} {'V*':>4s} {'IN625':>7s} {'mass [g]':>9s} {'I [g mm^2]':>11s}")
    for design, vf, fraction, mass, inertia in rows:
        print(f"{design:12s} {vf:4.1f} {fraction:7.4f} {mass:9.3f} {inertia:11.1f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
