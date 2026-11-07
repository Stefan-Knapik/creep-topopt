"""Animate the Sec. 4.2 V*=0.5 cantilever design over its MMA iterations -> figures/readme/design_evolution.gif."""
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from plot_utils import FIGURES, RESULTS, RMIN, draw_cantilever_bcs

from toga.opt.filtering import applyDensityFilter, compute_filter_kd_tree

RUN = RESULTS / "2_pareto_vf" / "data_creep_50"
OUT = FIGURES / "readme" / "design_evolution.gif"
FPS = 20
V_STAR = 0.5
# The paper's void-blue / solid-orange, without the white midpoint so grey regions stay visible.
CMAP = LinearSegmentedColormap.from_list("void_solid", np.array([[173, 216, 230], [255, 127, 14]]) / 255)


def read_geom(i):
    return np.asarray(meshio.read(RUN / "vtk" / f"sol_{i:03d}.vtu").cell_data["geom"][0], dtype=float).ravel()


def main():
    mesh = meshio.read(RUN / "vtk" / "sol_000.vtu")
    points = np.asarray(mesh.points, dtype=float)[:, :2]
    cells = next(np.asarray(b.data, dtype=int) for b in mesh.cells if b.type == "quad")
    x, y = points[cells, 0], points[cells, 1]
    areas = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))
    fe = SimpleNamespace(points=points, cells=cells, flex_inds=np.arange(len(cells)), dim=2,
                         num_cells=len(cells), JxW=areas)
    H, Hs = compute_filter_kd_tree(fe, RMIN)

    def filtered(i):
        return np.asarray(applyDensityFilter({"H": H, "Hs": Hs}, jnp.asarray(read_geom(i)).reshape(-1, 1))).ravel()

    history = pd.read_csv(RUN / "history.csv", sep=r",\s*", engine="python")
    last = int(history["iter"].iloc[-1])
    # The design changes fastest early on, so show every early iteration and fewer later ones.
    frames = list(range(0, 40)) + list(range(40, 100, 2)) + list(range(100, last + 1, 5))
    frames += [last] * FPS * 2  # hold the final design for 2 s
    rhos = {i: filtered(i) for i in sorted(set(frames))}

    its, J = history["iter"].to_numpy(), history["J"].to_numpy()
    V = V_STAR * (1.0 + history["vf"].to_numpy())  # history stores the constraint V / V* - 1

    fig = plt.figure(figsize=(8.4, 3.2), dpi=100)
    ax = fig.add_axes([0.01, 0.02, 0.6, 0.84])
    ax_j = fig.add_axes([0.71, 0.56, 0.27, 0.36])
    ax_v = fig.add_axes([0.71, 0.13, 0.27, 0.36], sharex=ax_j)

    ax.set_aspect("equal")
    ax.set_axis_off()
    coll = PolyCollection(points[cells], array=rhos[0], cmap=CMAP, edgecolors="none")
    coll.set_clim(0, 1)
    ax.add_collection(coll)
    draw_cantilever_bcs(ax, points, inside=True, arrow_below=0.1, arrowhead=12)
    title = ax.set_title("", fontsize=11)

    ax_v.axhline(V_STAR, color="k", ls="--", lw=1)
    ax_v.text(last, V_STAR + 0.04, f"$V^* = {V_STAR}$", ha="right", va="bottom", fontsize=9)
    artists = []
    for a, data, label, log in ((ax_j, J, "creep compliance $J$", True), (ax_v, V, "volume fraction", False)):
        a.plot(its, data, color="0.75", lw=1.2)
        artists.append(a.plot([], [], color="tab:orange", lw=2)[0])
        artists.append(a.plot([], [], "o", color="tab:orange", ms=5)[0])
        if log:
            a.set_yscale("log")
        a.set_ylabel(label, fontsize=9)
        a.tick_params(labelsize=8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax_j.set_xlim(0, last)
    ax_j.tick_params(labelbottom=False)
    ax_v.set_ylim(0.4, 1.05)
    ax_v.set_xlabel("MMA iteration", fontsize=9)

    def update(i):
        coll.set_array(rhos[i])
        title.set_text(f"iteration {i:3d}")
        for (trace, dot), data in zip((artists[0:2], artists[2:4]), (J, V)):
            trace.set_data(its[: i + 1], data[: i + 1])
            dot.set_data([its[i]], [data[i]])
        return coll, title, *artists

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=FPS))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
