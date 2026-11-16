"""Blade figure of Sec. 4.5 (figures/BC_and_blade_mat_design.pdf): boundary conditions, the IN625
fraction of a graded design seen from the pressure and the suction side, and the stress field of
the uniform blend that explains the layout, with the outline of the IN625 region drawn on it.

    python visualization/plot_blade_designs.py results/5_blade/deploy/<case>/V30

The design is the filtered composition (`mat_filtered`) of the last .vtu of the run. The driving
field is the von Mises stress at t = 0 of the uniform blend at the same IN625 content, read from
<run>/heuristic*.npz (experiments/5_blade/heuristic_design.py <run>). Without that file the stress
at t = 0 of the graded design itself is shown and the panel title says so.

Rendering is off screen with VTK and needs no display or GPU. A PNG preview goes to
figures/5_blade_exploration/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Arc, FancyArrowPatch
from vtk.util import numpy_support as vnp

FIGURES = Path(__file__).resolve().parents[1] / "figures"
# end and mid colours of ParaView's "Cool to Warm", as in the other blade renders
COOL_WARM = LinearSegmentedColormap.from_list(
    "cool_to_warm", [(0.2314, 0.2980, 0.7529), (0.865, 0.865, 0.865), (0.7059, 0.0157, 0.1490)])
STRESS = plt.get_cmap("viridis")
INK, MUTED = "#222222", "#555555"
GREY = (0.80, 0.81, 0.84)
# Vectors from the blade to the camera. The convex (suction) face of the blade looks toward -x.
# The camera sits slightly above the tip so that the tip face is visible.
VIEWS = {"pressure": (1.0, 0.3, 0.2), "suction": (-1.0, 0.3, 0.2)}
SIZE = (900, 1100)     # pixels of one render (width, height)
FS, FS_SMALL = 8.0, 7.0


def von_mises(sigma):
    s = sigma.reshape(-1, 3, 3)
    s = s - np.trace(s, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
    return np.sqrt(1.5 * np.sum(s * s, axis=(1, 2)))


def load_run(run):
    """Grid of the last iterate with the cell fields `IN625` and `stress`; says where the stress is from."""
    grid = None
    for file in reversed(sorted((run / "vtk").glob("sol_*.vtu"))):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(file))
        reader.Update()
        if reader.GetOutput().GetNumberOfCells() > 0:
            grid = reader.GetOutput()
            break
    if grid is None:
        raise SystemExit(f"no readable .vtu under {run / 'vtk'}")
    cells = grid.GetCellData()
    in625 = vnp.vtk_to_numpy(cells.GetArray("mat_filtered"))[:, 0]
    uniform = True
    files = sorted(run.glob("heuristic*.npz"), key=lambda p: p.stat().st_mtime)
    if files and len(np.load(files[-1])["vm_0_uniform"]) == grid.GetNumberOfCells():
        stress = np.load(files[-1])["vm_0_uniform"]
    else:
        stress, uniform = von_mises(vnp.vtk_to_numpy(cells.GetArray("stress_0"))), False
    for name, values in (("IN625", in625), ("stress", stress)):
        array = vnp.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
        array.SetName(name)
        cells.AddArray(array)
    return grid, in625, stress, uniform


def surface(grid):
    f = vtk.vtkDataSetSurfaceFilter()
    f.SetInputData(grid)
    f.Update()
    return f.GetOutput()


def lookup_table(cmap, vmin, vmax):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    for i in range(256):
        r, g, b, _ = cmap(i / 255.0)
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.SetRange(vmin, vmax)
    lut.Build()
    return lut


def outline(skin, field, level):
    """Lines on the surface where the (point-averaged) cell field crosses `level`."""
    to_points = vtk.vtkCellDataToPointData()
    to_points.SetInputData(skin)
    to_points.Update()
    data = to_points.GetOutput()
    data.GetPointData().SetActiveScalars(field)
    contour = vtk.vtkContourFilter()
    contour.SetInputData(data)
    contour.SetValue(0, level)
    contour.Update()
    return contour.GetOutput()


def render(skin, direction, field=None, lut=None, lines=None, line_color=(1.0, 1.0, 1.0)):
    """Off-screen image (H, W, 3) of the surface seen from `direction`, span upward, and a function
    that maps world points (n, 3) to pixel coordinates (n, 2) of that image (origin at the top left)."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(skin)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if field is None:
        mapper.ScalarVisibilityOff()
        actor.GetProperty().SetColor(*GREY)
    else:
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray(field)
        mapper.SetLookupTable(lut)
        mapper.UseLookupTableScalarRangeOn()
    actor.GetProperty().SetAmbient(0.35)
    actor.GetProperty().SetDiffuse(0.7)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(actor)
    if lines is not None:
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputData(lines)
        line_mapper.ScalarVisibilityOff()
        line_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(*line_color)
        line_actor.GetProperty().SetLineWidth(5.0)
        line_actor.GetProperty().LightingOff()
        renderer.AddActor(line_actor)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(8)
    window.SetSize(*SIZE)
    window.AddRenderer(renderer)
    bounds = skin.GetBounds()
    centre = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*centre)
    camera.SetPosition(*(centre + 300.0 * np.array(direction)))
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.ParallelProjectionOn()
    renderer.ResetCamera()
    camera.Zoom(1.25)
    renderer.ResetCameraClippingRange()
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    image = grab.GetOutput()
    w, h, _ = image.GetDimensions()
    pixels = vnp.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3].copy()

    def project(points):
        out = []
        for p in np.atleast_2d(points):
            renderer.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            out.append((x, h - y))
        return np.array(out)

    return pixels, project, window    # the window is returned so that `project` stays valid


def boundary_panel(ax, grid, skin, case):
    """Grey blade with the supports, temperatures and the centrifugal load drawn on it."""
    image, project, _window = render(skin, VIEWS["pressure"])
    ax.imshow(image)
    points = vnp.vtk_to_numpy(grid.GetPoints().GetData())
    y0, y1 = points[:, 1].min(), points[:, 1].max()
    z0, z1 = points[:, 2].min(), points[:, 2].max()
    root = points[points[:, 1] < y0 + 1e-3]
    tip = points[points[:, 1] > y1 - 1e-3]
    root_c, tip_c = root.mean(0), tip.mean(0)
    span_px = project(tip_c)[0] - project(root_c)[0]            # root to tip in pixels
    up = span_px / np.linalg.norm(span_px)
    h, w = image.shape[:2]

    # Clamp: hatched ground under the root edge that faces the camera.
    order = np.argsort(root[:, 2])
    edge = project(root[order][[0, len(order) // 4, len(order) // 2, 3 * len(order) // 4, -1]])
    lo = edge[:, 1].max() + 6
    xs = np.linspace(edge[:, 0].min() - 10, edge[:, 0].max() + 10, 15)
    ax.plot([xs[0], xs[-1]], [lo, lo], color=INK, lw=1.0, solid_capstyle="butt")
    for x in xs[:-1]:
        ax.plot([x + (xs[1] - xs[0]), x], [lo, lo + 30], color=INK, lw=0.6)

    # Centrifugal body force: arrows along the span, longer toward the tip (the force grows with the radius).
    for fz in (0.2, 0.5, 0.8):
        for fy in (0.22, 0.52, 0.80):
            base = project([np.interp(fy, [0, 1], [root_c[0], tip_c[0]]), y0 + fy * (y1 - y0), z0 + fz * (z1 - z0)])[0]
            length = 62 * (case["r_hub"] + fy * (y1 - y0)) / case["r_hub"]
            ax.add_patch(FancyArrowPatch(base - 0.5 * length * up, base + 0.5 * length * up, arrowstyle="-|>",
                                         mutation_scale=7, lw=1.0, color="#1f4e79", shrinkA=0, shrinkB=0))
    mid = project([0.5 * (root_c[0] + tip_c[0]), y0 + 0.52 * (y1 - y0), z0 + 0.5 * (z1 - z0)])[0]
    ax.text(mid[0] + 26, mid[1] + 74, r"$\mathbf{b}=\rho\,\omega^{2}\mathbf{r}_\perp$", color="#1f4e79", fontsize=FS,
            ha="center", va="center")

    # Temperatures and the insulated faces.
    top = project(tip_c)[0]
    ax.annotate(r"tip: $T_\mathrm{tip}=%g\,^{\circ}$C" % case["t_hot"], xy=(top[0], top[1] + 4), xytext=(w * 0.5, 14),
                fontsize=FS, color=INK, ha="center", va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(w * 0.5, lo + 40, r"root: clamped, $T_\mathrm{root}=%g\,^{\circ}$C" % case["t_root"], fontsize=FS, color=INK,
            ha="center", va="top")
    side = project([0.5 * (root_c[0] + tip_c[0]), y0 + 0.665 * (y1 - y0), z0 + 0.5 * (z1 - z0)])[0]
    ax.text(side[0], side[1], "airfoil faces insulated", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")

    # Axis of rotation below the root (not to scale) and the span dimension.
    axis_y = lo + 172
    ax.plot([w * 0.16, w * 0.84], [axis_y, axis_y], color=INK, lw=0.8, dashes=(8, 2, 1.5, 2))
    ax.add_patch(Arc((w * 0.84, axis_y), 44, 80, theta1=35, theta2=325, color=INK, lw=0.8))
    ax.add_patch(FancyArrowPatch((w * 0.84 + 17, axis_y + 24), (w * 0.84 + 19, axis_y + 12), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=INK, shrinkA=0, shrinkB=0))
    ax.text(w * 0.84 + 34, axis_y, r"$\omega$", fontsize=FS, color=INK, ha="left", va="center")
    ax.text(w * 0.5, axis_y + 34, "axis of rotation, %.0f rpm\n(%g mm below the root)" % (case["rpm"], case["r_hub"]),
            fontsize=FS_SMALL, color=INK, ha="center", va="top", linespacing=1.15)
    x_dim = edge[:, 0].min() - 34
    y_root, y_tip = project(root[order][0])[0][1], project(tip[np.argsort(tip[:, 2])][0])[0][1]
    ax.annotate("", xy=(x_dim, y_tip), xytext=(x_dim, y_root), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.6,
                                                                                 shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(x_dim - 8, 0.5 * (y_root + y_tip), "%g mm" % round(y1 - y0), fontsize=FS_SMALL, color=MUTED, rotation=90,
            ha="right", va="center")
    ax.set_xlim(-130, w + 110)
    ax.set_ylim(axis_y + 150, -10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run folder, e.g. results/5_blade/deploy/<case>/V30")
    parser.add_argument("--out", default="BC_and_blade_mat_design")
    parser.add_argument("--stress-side", choices=("auto", *VIEWS), default="auto",
                        help="side shown in the stress panel (auto: the side with more IN625)")
    args = parser.parse_args()
    case = json.load(open(args.run / "case.json"))
    grid, in625, stress, uniform = load_run(args.run)
    skin = surface(grid)

    composition = lookup_table(COOL_WARM, 0.0, 1.0)
    designs = {side: render(skin, direction, "IN625", composition)[0] for side, direction in VIEWS.items()}
    if args.stress_side == "auto":   # the face with more red than blue
        redness = {side: float(np.mean(img[..., 0].astype(float) - img[..., 2])) for side, img in designs.items()}
        stress_side = max(redness, key=redness.get)
    else:
        stress_side = args.stress_side
    vmax = float(np.ceil(np.percentile(stress, 99.5) / 10.0) * 10.0)
    stress_image = render(skin, VIEWS[stress_side], "stress", lookup_table(STRESS, 0.0, vmax),
                          lines=outline(skin, "IN625", 0.5))[0]

    fig = plt.figure(figsize=(7.2, 2.75))
    left, bottom, height = 0.0, 0.02, 0.87
    widths = (0.275, 0.18, 0.18, 0.18)
    gaps = (0.012, 0.0, 0.088)
    axes, x = [], left
    for k, width in enumerate(widths):
        axes.append(fig.add_axes([x, bottom, width, height]))
        x += width + (gaps[k] if k < len(gaps) else 0.0)
    for ax in axes:
        ax.set_axis_off()

    boundary_panel(axes[0], grid, skin, case)
    crop = slice(40, SIZE[1] - 40), slice(60, SIZE[0] - 60)
    axes[1].imshow(designs["pressure"][crop])
    axes[2].imshow(designs["suction"][crop])
    axes[3].imshow(stress_image[crop])
    for ax in axes[1:]:
        ax.set_anchor("N")
    used = float(np.sum(in625 * vnp.vtk_to_numpy(cell_volumes(grid))) / np.sum(vnp.vtk_to_numpy(cell_volumes(grid))))
    titles = ("(a) Boundary conditions",
              "(b) Pressure side",
              "(c) Suction side",
              "(d) Initial stress")
    for ax, title in zip(axes, titles):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), 0.955, title, fontsize=FS + 0.5, color=INK, ha="center", va="center")

    def bar(cmap, vmin, vmax, label, after, ticks=None):
        box = after.get_position()
        cax = fig.add_axes([box.x1 + 0.004, box.y0 + 0.2 * box.height, 0.011, 0.6 * box.height])
        b = fig.colorbar(plt.cm.ScalarMappable(Normalize(vmin, vmax), cmap), cax=cax, ticks=ticks)
        b.set_label(label, fontsize=FS, color=INK, labelpad=2)
        b.outline.set_linewidth(0.4)
        b.ax.tick_params(labelsize=FS_SMALL, length=2, width=0.4, colors=INK)

    bar(COOL_WARM, 0.0, 1.0, "IN625 fraction", axes[2], ticks=(0, 0.5, 1))
    bar(STRESS, 0.0, vmax, "von Mises stress [MPa]", axes[3])
    axes[3].text(0.5, -0.045, ("uniform blend, " if uniform else "graded design, ") + stress_side + " side\nwhite line: IN625 fraction 0.5", transform=axes[3].transAxes, fontsize=FS_SMALL, color=MUTED,
                 ha="center", va="top", linespacing=1.25)

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{args.out}.pdf", dpi=400)
    preview = FIGURES / "5_blade_exploration"
    preview.mkdir(exist_ok=True)
    fig.savefig(preview / f"{args.out}.png", dpi=300)
    plt.close(fig)
    print(f"written to {FIGURES / (args.out + '.pdf')} (preview {preview / (args.out + '.png')})")
    print(f"run {args.run}: IN625 bound {case['vf']:g}, used {used:.3f}; stress panel: {stress_side} side, "
          f"{'uniform blend' if uniform else 'graded design'}, scale 0 to {vmax:g} MPa (peak {stress.max():.0f} MPa)")


def cell_volumes(grid):
    size = vtk.vtkCellSizeFilter()
    size.SetInputData(grid)
    size.SetComputeVolume(True)
    size.SetComputeArea(False)
    size.SetComputeLength(False)
    size.SetComputeVertexCount(False)
    size.Update()
    return size.GetOutput().GetCellData().GetArray("Volume")


if __name__ == "__main__":
    main()
