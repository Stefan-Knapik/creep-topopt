"""Helpers shared by the figure notebooks. ``load_design`` reads a run's stored design, applies the
optimization's density filter and thresholds it so exactly ``vf`` of the cells are solid."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# The filter is a small sparse product, so run it on the CPU. Must be set before importing jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from toga.opt.filtering import applyDensityFilter, compute_filter_kd_tree

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PANELS = FIGURES / "panels"

# Filter radius in element widths: `rmin` in every experiments/*/optimize_*.py.
RMIN = 1.5

# Text sizes for the multi-row field figures.
TITLE_FS = 22
ROWLABEL_FS = 20
CBAR_LABEL_FS = 20
CBAR_TICK_FS = 18

# Void (blue) to solid (orange).
specimen_cmap = LinearSegmentedColormap.from_list(
    "specimen", np.array([[173, 216, 230], [255, 255, 255], [255, 127, 14]]) / 255
)


@dataclass(frozen=True)
class Design:
    """An optimized design, filtered and thresholded for plotting."""

    mesh: meshio.Mesh       # the run's .vtu, for reading other fields
    points: np.ndarray      # (num_nodes, 2)
    cells: np.ndarray       # (num_cells, 4)
    solid: np.ndarray       # (num_cells,) bool

    @property
    def binary(self) -> np.ndarray:
        return self.solid.astype(float)

    @property
    def void_mask(self) -> np.ndarray:
        return ~self.solid


def filter_density(points: np.ndarray, cells: np.ndarray, geom: np.ndarray,
                   rmin: float = RMIN) -> np.ndarray:
    """Apply the optimization's density filter to a per-cell field."""
    x, y = points[cells, 0], points[cells, 1]
    areas = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))
    fe = SimpleNamespace(points=points, cells=cells, flex_inds=np.arange(len(cells)), dim=2,
                         num_cells=len(cells), JxW=areas)
    H, Hs = compute_filter_kd_tree(fe, rmin)
    filtered = applyDensityFilter({"H": H, "Hs": Hs}, jnp.asarray(geom).reshape(-1, 1))
    return np.asarray(filtered).ravel()


def threshold_to_vf(rho: np.ndarray, vf: float) -> np.ndarray:
    """Mark the densest cells solid, so that exactly ``vf`` of the cells are solid."""
    solid = np.zeros(rho.size, dtype=bool)
    solid[np.argsort(-rho, kind="stable")[:round(vf * rho.size)]] = True
    return solid


def load_design(vtu_path: str | Path, vf: float) -> Design:
    """Read a design from a run's ``.vtu``, then filter it and threshold it to ``vf``."""
    mesh = meshio.read(vtu_path)
    points = np.asarray(mesh.points, dtype=float)[:, :2]
    cells = next(np.asarray(block.data, dtype=int) for block in mesh.cells if block.type == "quad")
    geom = np.asarray(mesh.cell_data["geom"][0], dtype=float).reshape(-1)
    return Design(mesh, points, cells, threshold_to_vf(filter_density(points, cells, geom), vf))


def von_mises_stress(sig: np.ndarray) -> np.ndarray:
    """Plane-stress von Mises stress. ``sig``: (num_cells, 4), ordered [s11, s12, s21, s22]."""
    s11, s12, _, s22 = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3]
    return np.sqrt(s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2)


def equivalent_strain(eps: np.ndarray) -> np.ndarray:
    """Von Mises equivalent creep strain, assuming incompressibility (e33 = -(e11 + e22)).

    ``eps``: (num_cells, 4), ordered [e11, e12, e21, e22].
    """
    e11, e12, _, e22 = eps[:, 0], eps[:, 1], eps[:, 2], eps[:, 3]
    e33 = -(e11 + e22)
    return np.sqrt((2.0 / 3.0) * (e11**2 + e22**2 + e33**2 + 2.0 * e12**2))


def solid_bounds(design: Design, values: np.ndarray, positive_only: bool = False) -> tuple[float, float]:
    """Min and max over solid cells. ``positive_only`` drops values a log scale can't show."""
    solid = np.asarray(values)[design.solid]
    if positive_only:
        solid = solid[solid > 0.0]
    return float(solid.min()), float(solid.max())


def plot_field(ax, design: Design, values, cmap="viridis", mask=None, vmin=None, vmax=None,
               log=False, border=True):
    """Plot a per-cell field. Cells where ``mask`` is True are transparent."""
    ax.set_aspect("equal")
    ax.set_axis_off()
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
    coll = PolyCollection(design.points[design.cells], array=np.asarray(values), cmap=cmap,
                          edgecolors="none", norm=norm)
    if not log:
        coll.set_clim(vmin, vmax)
    if mask is not None:
        coll.set_alpha(np.where(mask, 0.0, 1.0))
    ax.add_collection(coll)
    ax.autoscale_view()
    if border:
        (xmin, ymin), (xmax, ymax) = design.points.min(axis=0), design.points.max(axis=0)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                   edgecolor="black", linewidth=1.0, zorder=4))
    return coll


def label_rows(axes, labels) -> None:
    """Write horizontal row labels to the left of the first column."""
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label, fontsize=ROWLABEL_FS, rotation=0, ha="center", va="center", labelpad=60)
        # set_axis_off hides the ylabel, so re-enable the axis and hide everything else.
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def draw_cantilever_bcs(ax, points: np.ndarray, scale=None, inside=False, arrow_below=0.0,
                        arrowhead=10) -> None:
    """Draw the cantilever's domain border, clamped left edge and load on the lower right edge.

    The load is on the lower 20% of the right edge, as in the 1_creep_elastic and 2_pareto_vf runs.

    ``scale`` sizes the symbols. If None, it follows the height of the plotted structure, with 1
    for a 2.1 inch structure. ``inside=True`` pads the view to keep the symbols in the axes, for
    small insets. ``arrow_below`` extends the load arrow below the domain, as a fraction of its
    height.
    """
    (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)
    width, height = xmax - xmin, ymax - ymin
    arrow_tip = ymin - arrow_below * height
    if inside:
        ax.set_xlim(xmin - 0.05 * width, xmax + 0.08 * width)
        ax.set_ylim(arrow_tip - 0.05 * height, ymax + 0.05 * height)
    if scale is None:
        ax.figure.canvas.draw()
        (_, y0), (_, y1) = ax.transData.transform([(xmin, ymin), (xmin, ymax)])
        scale = abs(y1 - y0) / ax.figure.dpi / 2.1

    ax.add_patch(plt.Rectangle((xmin, ymin), width, height, fill=False, edgecolor="black",
                               linewidth=1.0 * scale, zorder=4))
    y_supports = np.linspace(ymin, ymax, 8)
    ax.plot(np.full_like(y_supports, xmin - 0.015 * width), y_supports, marker=">",
            markersize=8 * scale, color="red", linestyle="none", clip_on=inside, zorder=5)
    y_load_top = ymin + 0.2 * height
    ax.plot([xmax, xmax], [ymin, y_load_top], color="blue", linewidth=4.0 * scale,
            solid_capstyle="butt", clip_on=inside, zorder=5)
    arrow_x = xmax + 0.03 * width
    ax.annotate("", xy=(arrow_x, arrow_tip), xytext=(arrow_x, y_load_top),
                arrowprops=dict(arrowstyle="-|>", color="blue", linewidth=2.0 * scale,
                                mutation_scale=arrowhead * scale, joinstyle="miter"),
                annotation_clip=False, zorder=6)
