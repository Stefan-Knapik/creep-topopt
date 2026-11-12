"""Assemble the renders of visualization/render_blade_design.py into one figure:
IN625 concentration of the graded blade at V* = 0.3 under the tip traction and under centrifugal
loading, seen from the pressure and the suction side. Writes figures/blade_designs_vf30.pdf.

    python visualization/render_blade_design.py results/5_blade/data_creep_30/vtk/sol_050.vtu figures/renders/blade_tip_traction_30
    python visualization/render_blade_design.py results/6_blade_centrifugal/data_creep_30/vtk/sol_050.vtu figures/renders/blade_centrifugal_30
    python visualization/plot_blade_designs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RENDERS = FIGURES / "renders"
# end and mid colours of ParaView's "Cool to Warm", for the colour bar
COOL_WARM = LinearSegmentedColormap.from_list(
    "cool_to_warm", [(0.2314, 0.2980, 0.7529), (0.865, 0.865, 0.865), (0.7059, 0.0157, 0.1490)])

rows = (("blade_tip_traction_30", "Tip traction"), ("blade_centrifugal_30", "Centrifugal load"))
cols = (("pressure", "Pressure side"), ("suction", "Suction side"))

fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.7))
for j, ((prefix, load), (side, side_title)) in enumerate((r, c) for r in rows for c in cols):
    ax = axes[j]
    ax.imshow(plt.imread(RENDERS / f"{prefix}_{side}.png"))
    ax.set_axis_off()
    ax.set_title(f"{load}\n{side_title.lower()}", fontsize=9)
fig.subplots_adjust(left=0.0, right=0.9, bottom=0.02, top=0.86, wspace=0.02)
cax = fig.add_axes([0.915, 0.12, 0.012, 0.66])
bar = fig.colorbar(plt.cm.ScalarMappable(Normalize(0.0, 1.0), COOL_WARM), cax=cax)
bar.set_label("IN625 concentration", fontsize=9)
bar.ax.tick_params(labelsize=8)
fig.savefig(FIGURES / "blade_designs_vf30.pdf", dpi=300)
print(f"written to {FIGURES / 'blade_designs_vf30.pdf'}")
