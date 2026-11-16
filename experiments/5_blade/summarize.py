"""Table of the optimization runs of a study: results/5_blade/<study>/<case>/V<bound>/.

For every run (last evaluated design) one row in results/5_blade/<study>/summary.csv:

    IN625_used, bound_active   IN625 fraction of the filtered design, and whether it is at its bound
    J_uniform, J, gain         creep compliance of the starting blend and of the design [mJ], 1 - J / J_uniform
    spanwise                   share of the variance of the IN625 field explained by the span position alone
    chordwise, through         the same for the position along the chord and across the thickness
    interface                  total variation of the IN625 field between neighbouring cells per unit blade
                               surface: 1 = one sharp spanwise cut, larger = more interface (speckle)
    graded                     volume share of cells with an IN625 fraction between 0.1 and 0.9
    in625_span_centroid        mean spanwise position of the IN625 / span
    suction_half, leading_half share of the IN625 in the suction-side half and in the leading half
    capture_ratio, gain_noexp  from heuristic_design.py, if it has been run
    J_homogeneous              J of the uniform blend with the same IN625 content, if
                               evaluate_homogeneous_blends.py has been run for the case

No analysis is run; needs numpy, scipy and meshio only.

    python summarize.py --study explore
"""
import argparse
import csv
import json
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "5_blade"
SPAN = 50.0
_geometry = {}


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


def geometry(vtu):
    """Per mesh (keyed by the number of cells): centroids, volumes, face neighbours with face areas,
    and section coordinates of every cell (chordwise and through-thickness rank within its layer)."""
    mesh = meshio.read(vtu)
    cells = next(block.data for block in mesh.cells if block.type == "hexahedron")
    if len(cells) in _geometry:
        return mesh, _geometry[len(cells)]
    points = np.asarray(mesh.points, dtype=float)
    centroids = points[cells].mean(axis=1)
    volume = hex8_volumes(points, cells)
    # face neighbours: cells sharing 4 nodes
    faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]])
    keys = np.sort(cells[:, faces], axis=-1).reshape(-1, 4)
    owner = np.repeat(np.arange(len(cells)), 6)
    order = np.lexsort(keys.T[::-1])
    keys, owner = keys[order], owner[order]
    same = np.all(keys[1:] == keys[:-1], axis=1)
    pairs = np.stack((owner[:-1][same], owner[1:][same]), axis=1)
    face_nodes = points[keys[:-1][same]]            # (pairs, 4, 3), sorted node order: area from the centroid fan
    c = face_nodes.mean(axis=1, keepdims=True)
    d = face_nodes - c
    # area of a (near-planar) quad independent of the node order: half the sum of |d_i x d_j| over the angular order
    normal = np.cross(d[:, 0], d[:, 1])
    normal = np.where(np.linalg.norm(normal, axis=1, keepdims=True) > 1e-14, normal, np.cross(d[:, 0], d[:, 2]))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    e1 = d[:, 0] / np.linalg.norm(d[:, 0], axis=1, keepdims=True)
    e2 = np.cross(normal, e1)
    angle = np.arctan2(np.einsum("pni,pi->pn", d, e2), np.einsum("pni,pi->pn", d, e1))
    idx = np.argsort(angle, axis=1)
    ds = np.take_along_axis(d, idx[:, :, None], axis=1)
    area = 0.5 * np.linalg.norm(np.sum(np.cross(ds, np.roll(ds, -1, axis=1)), axis=1), axis=1)
    # section coordinates: within a layer of elements, the first principal axis is the chord
    _, layer = np.unique(np.round(centroids[:, 1], 3), return_inverse=True)
    chord, through = np.zeros(len(cells)), np.zeros(len(cells))
    for l in range(layer.max() + 1):
        m = layer == l
        xz = centroids[m][:, [0, 2]]
        xz = xz - xz.mean(axis=0)
        _, _, vt = np.linalg.svd(xz, full_matrices=False)
        chord[m], through[m] = xz @ vt[0], xz @ vt[1]
    # the blade surface area the interface measure is normalized with: one spanwise cut = the section area
    geo = dict(centroids=centroids, volume=volume, pairs=pairs, area=area, layer=layer, chord=chord, through=through,
               section_area=volume.sum() / SPAN)
    # halves from the mesh's named surfaces, when the .inp of this mesh is at hand
    geo['suction'] = geo['leading'] = None
    for inp in list((RESULTS / "mesh").glob("*.inp")) + [ROOT / "toga" / "mesh" / "blade" / "blade_050_000.inp"]:
        m = meshio.read(inp)
        if len(m.cells_dict.get("hexahedron", [])) != len(cells):
            continue
        p = np.asarray(m.points, dtype=float)
        dist = lambda names: cKDTree(p[np.unique(np.concatenate([m.point_sets[n] for n in names]))]).query(centroids)[0]
        geo['suction'] = dist(["suction_leading_edge", "suction_side", "suction_trailing_edge"]) \
            < dist(["pressure_leading_edge", "pressure_side", "pressure_trailing_edge"])
        geo['leading'] = dist(["suction_leading_edge", "pressure_leading_edge"]) \
            < dist(["suction_trailing_edge", "pressure_trailing_edge"])
        break
    _geometry[len(cells)] = geo
    return mesh, geo


def explained(x, v, bins):
    """Share of the volume-weighted variance of x explained by its mean within each bin."""
    total = np.sum(v * (x - np.sum(v * x) / v.sum()) ** 2)
    mean = (np.bincount(bins, weights=v * x) / np.maximum(np.bincount(bins, weights=v), 1e-300))[bins]
    return float(1.0 - np.sum(v * (x - mean) ** 2) / max(total, 1e-30))


def design_metrics(x, geo):
    v = geo['volume']
    quantile_bins = lambda c: np.searchsorted(np.quantile(c, np.linspace(0, 1, 21)[1:-1]), c)
    out = {
        'spanwise': explained(x, v, geo['layer']),
        'chordwise': explained(x, v, quantile_bins(geo['chord'])),
        'through': explained(x, v, quantile_bins(geo['through'])),
        'interface': float(np.sum(geo['area'] * np.abs(x[geo['pairs'][:, 0]] - x[geo['pairs'][:, 1]])) / geo['section_area']),
        'graded': float(v[(x > 0.1) & (x < 0.9)].sum() / v.sum()),
        'in625_span_centroid': float((x * v) @ geo['centroids'][:, 1] / max((x * v).sum(), 1e-30) / SPAN),
    }
    for name in ('suction', 'leading'):
        if geo[name] is not None:
            out[f'{name}_half'] = float((x * v)[geo[name]].sum() / max((x * v).sum(), 1e-30))
    return out


def summarize_run(run):
    scalars = list(csv.DictReader(open(run / "scalars.csv")))
    case = json.load(open(run / "case.json"))
    first, last = scalars[0], scalars[-1]
    row = {'case': run.parent.name, 'vf': case['vf'], 'root_stress': case['root_stress'], 'shroud': case['shroud'],
           'hours': case['hours'], 'radius_mm': case['radius_mm'], 'num_cells': case['num_cells'],
           'num_time_steps': case['num_time_steps'], 'expansion': case['expansion'], 'alpha_mode': case['alpha_mode'],
           'cte_mismatch_scale': case['cte_mismatch_scale'], 'init': case['init'], 'rpm': round(case['rpm']),
           'iterations': len(scalars) - 1,
           'status': 'done' if (run / 'DONE').exists() else ('failed' if (run / 'FAILED').exists() else 'running'),
           'IN625_used': float(last['IN625_fraction']),
           'bound_active': float(last['IN625_fraction']) > case['vf'] - 0.01,
           'J_uniform': float(first['J']), 'J': float(last['J']), 'gain': 1.0 - float(last['J']) / float(first['J']),
           'mass_g': float(last['mass_g']), 'inertia_g_mm2': float(last['inertia_g_mm2']),
           'vm0_mean': float(last['vm0_mean']), 'vm0_peak': float(last['vm0_max_cell']),
           'creep_peak': float(last['creep_eq_max_cell']), 'creep_peak_uniform': float(first['creep_eq_max_cell']),
           'w_sum_min': float(last['w_sum_min']), 'w_sum_max': float(last['w_sum_max'])}
    vtus = sorted((run / "vtk").glob("sol_*.vtu"))
    if vtus:
        mesh, geo = geometry(vtus[-1])
        row.update(design_metrics(np.asarray(mesh.cell_data["mat_filtered"][0], dtype=float)[:, 0], geo))
    for file in sorted(run.glob("heuristic*.json")):
        h = json.load(open(file))
        suffix = file.stem[len("heuristic"):]
        for k in ('capture_ratio', 'gain_optimized', 'gain_heuristic', 'gain_optimized_noexp', 'capture_ratio_noexp',
                  'layout_correlation'):
            if k in h:
                row[f'{k}{suffix}'] = h[k]
    homo = run.parent / "homogeneous_blends.csv"
    if homo.exists():
        blends = {round(float(r['V_star']), 2): float(r['J']) for r in csv.DictReader(open(homo))}
        row['J_homogeneous'] = blends.get(round(case['vf'], 2), float('nan'))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="explore")
    args = parser.parse_args()
    rows = [summarize_run(p.parent) for p in sorted((RESULTS / args.study).glob("*/V*/scalars.csv"))
            if (p.parent / "case.json").exists()]
    if not rows:
        raise SystemExit("no runs found")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = RESULTS / args.study / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'case':46s} {'V*':>4s} {'used':>5s} {'gain':>6s} {'span':>5s} {'chord':>5s} {'thru':>5s} {'intf':>5s} {'grad':>5s} {'capt':>5s} status")
    for r in rows:
        print(f"{r['case']:46s} {r['vf']:4.2f} {r['IN625_used']:5.3f} {r['gain']:6.1%} {r.get('spanwise', np.nan):5.2f} "
              f"{r.get('chordwise', np.nan):5.2f} {r.get('through', np.nan):5.2f} {r.get('interface', np.nan):5.2f} "
              f"{r.get('graded', np.nan):5.2f} {r.get('capture_ratio', np.nan):5.2f} {r['status']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
