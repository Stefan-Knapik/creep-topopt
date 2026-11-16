"""Contact sheets of the designs of a study (the last .vtu of every run under
results/5_blade/<study>/<case>/V<bound>/), into figures/5_blade_exploration/:

    designs_<study>[_<name>].png   IN625 fraction (filtered), cases (rows) x IN625 bound (columns); each panel
                                   shows the two faces of the blade and sections at 10 / 50 / 90 % span
    fields_<study>[_<name>].png    with --fields: temperature, von Mises stress at t = 0 and at the end, and
                                   equivalent creep strain of the runs at --fields-vf

    python render_designs.py --study explore
    python render_designs.py --study explore --match "S080*r1p4*" --name S080 --fields

Rendering is off screen with VTK. Run summarize.py for the numbers.
"""
import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import vtk
from matplotlib.colors import LinearSegmentedColormap
from vtk.util import numpy_support as vnp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "5_blade"
FIGURES = ROOT / "figures" / "5_blade_exploration"

# Composition is a balance between two alloys: two hues and a neutral middle.
COMPOSITION = LinearSegmentedColormap.from_list("composition", ["#2166ac", "#e4e4e4", "#b2182b"])
MAGNITUDE = plt.get_cmap("viridis")
INK, MUTED = "#222222", "#666666"


def von_mises(sigma):
    s = sigma.reshape(-1, 3, 3)
    s = s - onp.trace(s, axis1=1, axis2=2)[:, None, None] / 3.0 * onp.eye(3)
    return onp.sqrt(1.5 * onp.sum(s * s, axis=(1, 2)))


def equivalent_strain(eps):
    e = eps.reshape(-1, 3, 3)
    return onp.sqrt(2.0 / 3.0 * onp.sum(e * e, axis=(1, 2)))


def load_run(run):
    """Grid of the last iterate with the scalar cell fields used here, and the run's numbers."""
    grid = None
    for file in reversed(sorted((run / "vtk").glob("sol_*.vtu"))):   # the newest may still be being written
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(file))
        reader.Update()
        if reader.GetOutput().GetNumberOfCells() > 0:
            grid = reader.GetOutput()
            break
    if grid is None:
        return None
    cells = grid.GetCellData()
    cell = lambda name: vnp.vtk_to_numpy(cells.GetArray(name))
    connectivity = vnp.vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(grid.GetNumberOfCells(), -1)
    T = vnp.vtk_to_numpy(grid.GetPointData().GetArray("T")).reshape(-1)
    fields = {
        "IN625": cell("mat_filtered")[:, 0],
        "vm_0": von_mises(cell("stress_0")),
        "vm_end": von_mises(cell("stress_final")),
        "creep_eq": equivalent_strain(cell("creep_strain_final")),
        "T": T[connectivity].mean(axis=1),   # nodal, averaged to the cells
    }
    for name, values in fields.items():
        array = vnp.numpy_to_vtk(onp.ascontiguousarray(values, dtype=onp.float64), deep=True)
        array.SetName(name)
        cells.AddArray(array)
    scalars = list(csv.DictReader(open(run / "scalars.csv")))
    return {"grid": grid, "fields": fields, "case": json.load(open(run / "case.json")), "name": run.parent.name,
            "first": scalars[0], "last": scalars[-1], "iterations": len(scalars),
            "failed": (run / "FAILED").exists(), "done": (run / "DONE").exists()}


def add_cell_field(grid, name, values):
    """Attach a per-cell array (e.g. the indicator of heuristic.npz) to a loaded grid for rendering."""
    array = vnp.numpy_to_vtk(onp.ascontiguousarray(values, dtype=onp.float64), deep=True)
    array.SetName(name)
    grid.GetCellData().AddArray(array)


def lookup_table(cmap, vmin, vmax):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    for i in range(256):
        r, g, b, _ = cmap(i / 255.0)
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.SetRange(vmin, vmax)
    lut.Build()
    return lut


def render(polydata, field, lut, direction, view_up, size, zoom=1.35):
    """Off-screen image (H, W, 3) of a surface colored by a cell field, seen along -direction."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarModeToUseCellFieldData()
    mapper.SelectColorArray(field)
    mapper.SetLookupTable(lut)
    mapper.UseLookupTableScalarRangeOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(0.45)
    actor.GetProperty().SetDiffuse(0.6)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(*size)
    camera = renderer.GetActiveCamera()
    center = onp.array(polydata.GetCenter())
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + 500.0 * onp.array(direction) / onp.linalg.norm(direction)))
    camera.SetViewUp(*view_up)
    camera.ParallelProjectionOn()
    renderer.ResetCamera()
    camera.Zoom(zoom)
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    image = grab.GetOutput()
    w, h, _ = image.GetDimensions()
    return vnp.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]


def surface(grid):
    f = vtk.vtkDataSetSurfaceFilter()
    f.SetInputData(grid)
    f.Update()
    return f.GetOutput()


def sections(grid, fractions=(0.1, 0.5, 0.9), span=50.0, spacing=22.0):
    """The layers of elements at the given span fractions, laid out side by side (root section
    first) for a view down the span. (The mesh is an extrusion, so a layer is a flat section.)"""
    points = vnp.vtk_to_numpy(grid.GetPoints().GetData())
    connectivity = vnp.vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(grid.GetNumberOfCells(), -1)
    y = points[connectivity, 1].mean(axis=1)
    layers = onp.unique(onp.round(y, 3))
    out = vtk.vtkAppendPolyData()
    for i, fraction in enumerate(fractions):
        layer = layers[onp.argmin(onp.abs(layers - fraction * span))]
        ids = vtk.vtkIdList()
        for cell_id in onp.flatnonzero(onp.abs(y - layer) < 1e-2):
            ids.InsertNextId(int(cell_id))
        extract = vtk.vtkExtractCells()
        extract.SetInputData(grid)
        extract.SetCellList(ids)
        extract.Update()
        skin = surface(extract.GetOutput())
        center = onp.array(skin.GetCenter())
        move = vtk.vtkTransform()
        move.Translate(i * spacing - center[0], -center[1], -center[2])
        moved = vtk.vtkTransformPolyDataFilter()
        moved.SetInputData(skin)
        moved.SetTransform(move)
        moved.Update()
        out.AddInputData(moved.GetOutput())
    out.Update()
    return out.GetOutput()


def blade_panel(run, field, lut):
    """Both faces of the blade and the three sections, one image."""
    skin = surface(run["grid"])
    faces = [render(skin, field, lut, direction, (0.0, 1.0, 0.0), (330, 440))
             for direction in ((1.0, 0.25, 0.35), (-1.0, 0.25, 0.35))]
    cuts = render(sections(run["grid"]), field, lut, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (300, 440))
    return onp.concatenate(faces + [cuts], axis=1)


def colorbar(fig, cmap, vmin, vmax, label, rect):
    axis = fig.add_axes(rect)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), cax=axis, orientation="horizontal")
    bar.set_label(label, color=INK, fontsize=10)
    bar.outline.set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=9, length=3)


def case_label(run):
    c = run["case"]
    expansion = c["alpha_mode"] + (f" x{c['cte_mismatch_scale']:g}" if c["cte_mismatch_scale"] != 1.0 else "") if c["expansion"] else "off"
    return (f"{run['name']}:  root stress {c['root_stress']:.0f} MPa ({c['shroud']:.0%} from a shroud), {c['rpm']:.0f} rpm, "
            f"{c['hours']:g} h in {c['num_time_steps']} steps; filter {c['radius_mm']:g} mm; expansion {expansion}; "
            f"{c['num_cells']} cells" + (f"; start {c['init']}" if c["init"] != "uniform" else ""))


def design_sheet(runs, out, title):
    names = sorted({r["name"] for r in runs})
    bounds = sorted({r["case"]["vf"] for r in runs})
    by_key = {(r["name"], r["case"]["vf"]): r for r in runs}
    lut = lookup_table(COMPOSITION, 0.0, 1.0)
    fig, axes = plt.subplots(len(names), len(bounds), figsize=(4.4 * len(bounds), 2.75 * len(names) + 2.0), squeeze=False)
    for i, name in enumerate(names):
        for j, bound in enumerate(bounds):
            axis = axes[i, j]
            axis.axis("off")
            run = by_key.get((name, bound))
            if run is None:
                continue
            axis.imshow(blade_panel(run, "IN625", lut))
            used = float(run["last"]["IN625_fraction"])
            state = "" if run["done"] else ("  FAILED" if run["failed"] else f"  (iteration {run['iterations'] - 1})")
            axis.set_title(f"IN625 ≤ {bound:.0%}: uses {used:.0%} ({'bound active' if used > bound - 0.01 else 'bound NOT active'}){state}\n"
                           f"J {float(run['last']['J']) / float(run['first']['J']) - 1.0:+.0%} vs uniform, "
                           f"peak creep strain {float(run['last']['creep_eq_max_cell']):.1e}", fontsize=8.5, color=INK, loc="left")
        axes[i, 0].text(0.0, 1.3, case_label(by_key[next(k for k in by_key if k[0] == name)]),
                        transform=axes[i, 0].transAxes, fontsize=10, color=INK, fontweight="bold", ha="left", va="bottom")
    height = fig.get_figheight()
    fig.suptitle(f"{title}: IN625 fraction (filtered) of the designs for each bound on the IN625 content\n"
                 "each panel: the two faces of the blade (root at the bottom) and sections at 10 / 50 / 90 % span (bottom to top)",
                 fontsize=12, color=INK, x=0.005, ha="left")
    fig.subplots_adjust(left=0.005, right=0.995, top=1.0 - 1.55 / height, bottom=0.8 / height, wspace=0.02, hspace=0.5)
    colorbar(fig, COMPOSITION, 0.0, 1.0, "IN625 fraction (0 = SS316, 1 = IN625)", [0.3, 0.4 / height, 0.4, 0.13 / height])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"{len(runs)} runs drawn -> {out}")


def field_sheet(runs, out, title):
    columns = [("IN625", "IN625 fraction"), ("T", "temperature [°C]"), ("vm_0", "von Mises stress at t = 0 [MPa]"),
               ("vm_end", "von Mises stress at the end [MPa]"), ("creep_eq", "equivalent creep strain at the end")]
    fig, axes = plt.subplots(len(runs), len(columns), figsize=(5.2 * len(columns), 3.4 * len(runs) + 0.8), squeeze=False)
    for i, run in enumerate(runs):
        for j, (field, label) in enumerate(columns):
            values = run["fields"][field]
            if field == "IN625":
                cmap, vmin, vmax = COMPOSITION, 0.0, 1.0
            else:
                cmap = MAGNITUDE
                vmin = float(values.min()) if field == "T" else 0.0
                vmax = float(values.max()) if field == "T" else float(onp.percentile(values, 99.5))
                vmax = max(vmax, vmin + 1e-12)
            axis = axes[i, j]
            axis.axis("off")
            axis.imshow(blade_panel(run, field, lookup_table(cmap, vmin, vmax)))
            axis.set_title(f"{label}\n{run['name']} V{round(100 * run['case']['vf'])}; scale {vmin:.3g} to {vmax:.3g}, peak {values.max():.3g}",
                           fontsize=8.5, color=INK, loc="left")
    fig.suptitle(f"{title}: fields of the optimized blades (viridis, dark = low; stress and strain scales end at the 99.5th percentile)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0 - 0.75 / fig.get_figheight(), bottom=0.01, wspace=0.02, hspace=0.2)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"fields of {len(runs)} runs -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="explore")
    parser.add_argument("--match", default="*", help="glob on the case folder name")
    parser.add_argument("--name", default=None, help="suffix of the output files")
    parser.add_argument("--fields", action="store_true", help="also draw the field sheet")
    parser.add_argument("--fields-vf", type=float, default=0.3)
    args = parser.parse_args()
    paths = [p.parent for p in sorted((RESULTS / args.study).glob("*/V*/scalars.csv"))
             if fnmatch.fnmatch(p.parent.parent.name, args.match) and (p.parent / "case.json").exists()]
    runs = [r for r in (load_run(p) for p in paths) if r is not None]
    if not runs:
        sys.exit("no results found")
    FIGURES.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.name}" if args.name else ""
    design_sheet(runs, FIGURES / f"designs_{args.study}{suffix}.png", f"Study '{args.study}'")
    if args.fields:
        chosen = [r for r in runs if abs(r["case"]["vf"] - args.fields_vf) < 1e-9]
        if chosen:
            field_sheet(chosen, FIGURES / f"fields_{args.study}{suffix}.png", f"Study '{args.study}'")


if __name__ == "__main__":
    main()
