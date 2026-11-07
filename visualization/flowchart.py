"""Flow chart of the creep topology-optimization framework (Sec. 3.5, Fig. 2).
Writes figures/creep_optimization_flowchart.{pdf,png}; the paper uses a manually cropped copy."""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path

FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# ----------------------------------------------------------------------
# colors
C_OPT = "#eef6ee"      # outer optimization-loop group
C_FWD = "#eaf1fb"      # forward-solve (time-stepping) group
C_CREEP = "#fdeeee"    # innermost creep return-mapping group
C_BLOCK = "white"
C_TERM = "#ececec"
EDGE = "#222222"

# vertical compression: <1 makes boxes/diamonds/arrows slightly shorter
SY = 0.85

fig, ax = plt.subplots(figsize=(8.0, 14.5 * SY))
ax.set_xlim(0, 10)
ax.set_ylim(0, 23.0 * SY)
ax.axis("off")

W = 5.2          # block width
XC = 5.4         # center x of the main column
boxes = {}


def block(key, y, text, h=1.0, kind="block", w=W, fontsize=9.0):
    y *= SY
    h *= SY
    color = {"block": C_BLOCK, "term": C_TERM}[kind]
    style = "round,pad=0.02,rounding_size=0.18" if kind == "block" else \
            "round,pad=0.02,rounding_size=0.38"
    p = FancyBboxPatch((XC - w / 2, y - h / 2), w, h,
                       boxstyle=style, linewidth=1.4,
                       edgecolor=EDGE, facecolor=color, zorder=3)
    ax.add_patch(p)
    ax.text(XC, y, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, linespacing=1.25)
    boxes[key] = dict(x=XC, y=y, w=w, h=h)
    return boxes[key]


def diamond(key, y, text, w=2.7, h=1.1):
    y *= SY
    h *= SY
    verts = [(XC - w / 2, y), (XC, y + h / 2), (XC + w / 2, y),
             (XC, y - h / 2), (XC - w / 2, y)]
    ax.add_patch(PathPatch(Path(verts), linewidth=1.4, edgecolor=EDGE,
                           facecolor=C_BLOCK, zorder=3))
    ax.text(XC, y, text, ha="center", va="center", fontsize=9, zorder=4)
    boxes[key] = dict(x=XC, y=y, w=w, h=h)
    return boxes[key]


def arrow(a, b, label=None, lx=0.25):
    ya = boxes[a]["y"] - boxes[a]["h"] / 2
    yb = boxes[b]["y"] + boxes[b]["h"] / 2
    ax.add_patch(FancyArrowPatch((XC, ya), (XC, yb),
                 arrowstyle="-|>", mutation_scale=14,
                 linewidth=1.4, color=EDGE, zorder=5))
    if label:
        ax.text(XC + lx, (ya + yb) / 2, label, ha="left", va="center",
                fontsize=8, zorder=6)


# ---------------------------------------------------------------- nodes
block("start", 22.1, "Initialize design variables $\\rho^{0}$\n(uniform density $\\rho=V_{*}$)",
      h=0.85, kind="term")
block("filter", 20.5,
      "Density filter: $\\rho\\rightarrow\\tilde\\rho$\npenalize $E(\\tilde\\rho)$ and creep prefactor $A(\\tilde\\rho)$",
      h=0.95)
block("thermal", 19.1,
      "Solve Thermal FEM equilibrium\n"
      "$\\nabla\\!\\cdot\\!(\\kappa(\\tilde\\rho,T)\\,\\nabla T)=0$,   "
      "temperature field $T$ sets A(T)", h=0.95)
block("init", 17.5, "Initialize history state\n$\\sigma=0,\\ \\varepsilon^{cr}=0,\\ t=0$", h=0.9)
block("step", 16.1, "Advance time step $s$:  $t\\leftarrow t+\\Delta t_s$\n($s{=}0$: elastic, $\\Delta t_0{=}0$)", h=0.95)
block("equil", 14.55,
      "Solve Structural FEM equilibrium (global Newton)\n$R(u_s)=\\int_\\Omega \\sigma\\!:\\!\\nabla v\\,d\\Omega-f_{ext}=0$", h=1.05)
block("creep", 12.5,
      "Implicit creep update (local Newton)\nRoot-find for $\\Delta\\varepsilon^{cr}$ at each quadrature point:\n"
      "$\\varepsilon^{el}=\\varepsilon(u_s)-\\varepsilon^{cr}_{s-1}-\\Delta\\varepsilon^{cr}-\\varepsilon^{th},\\ \\ \\sigma=\\mathbb{E}\\!:\\!\\varepsilon^{el}$\n"
      "$\\dot\\varepsilon^{cr}=A\\,\\sigma_{vm}^{\\,n}\\,N_{ij}$   ($n$: Norton exponent)\n"
      "$r(\\Delta\\varepsilon^{cr})=\\Delta\\varepsilon^{cr}-\\Delta t_s\\,\\dot\\varepsilon^{cr}$", h=1.95, fontsize=8.4)
block("upd", 10.6,
      "Update history\n$\\sigma_s,\\ \\varepsilon^{cr}_{s}\\leftarrow\\varepsilon^{cr}_{s-1}+\\Delta\\varepsilon^{cr}$;  store $u_s$", h=0.95)
diamond("tloop", 9.2, "$s<N_t$ ?")
block("obj", 7.6,
      "Objective & constraint evaluation\ncreep compliance, $J$\nvolume fraction, $g_v$", h=1.25)
block("sens", 5.95,
      "Sensitivities via automatic differentiation\n$\\partial J/\\partial\\rho,\\ \\partial g_v/\\partial\\rho$  (adjoint through time scan)", h=1.05)
block("mma", 4.5, "MMA update:  $\\rho^{i}\\rightarrow\\rho^{i+1}$", h=0.8)
diamond("conv", 3.1, "Converged?")
block("end", 1.5, "Optimized creep-resistant design", h=0.85, kind="term")

# ---------------------------------------------------------------- arrows
arrow("start", "filter")
arrow("filter", "thermal")
arrow("thermal", "init")
arrow("init", "step")
arrow("step", "equil")
arrow("equil", "creep")
arrow("creep", "upd")
arrow("upd", "tloop")
arrow("tloop", "obj", label="no")
arrow("obj", "sens")
arrow("sens", "mma")
arrow("mma", "conv")
arrow("conv", "end", label="yes")

# feedback: global-local solve alternation (creep update -> equilibrium),
# inner loop on the left, iterate until convergence
xgl = XC - W / 2 - 0.75
ax.add_patch(FancyArrowPatch(
    path=Path([(boxes["creep"]["x"] - boxes["creep"]["w"] / 2, boxes["creep"]["y"]),
               (xgl, boxes["creep"]["y"]),
               (xgl, boxes["equil"]["y"]),
               (boxes["equil"]["x"] - boxes["equil"]["w"] / 2, boxes["equil"]["y"])],
              [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]),
    arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=EDGE, zorder=5))
ax.text(xgl - 0.1, (boxes["creep"]["y"] + boxes["equil"]["y"]) / 2,
        "iterate until converged", ha="center", va="center", rotation=90,
        fontsize=8, color=EDGE)

# feedback: time loop (tloop yes -> step), on the right
xr = XC + W / 2 + 1.45
ax.add_patch(FancyArrowPatch(
    path=Path([(boxes["tloop"]["x"] + boxes["tloop"]["w"] / 2, boxes["tloop"]["y"]),
               (xr, boxes["tloop"]["y"]),
               (xr, boxes["step"]["y"]),
               (boxes["step"]["x"] + boxes["step"]["w"] / 2, boxes["step"]["y"])],
              [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]),
    arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=EDGE, zorder=5))
ax.text(xr + 0.1, (boxes["tloop"]["y"] + boxes["step"]["y"]) / 2,
        "yes:  $s\\leftarrow s+1$", ha="left", va="center", rotation=90, fontsize=8, color=EDGE)

# feedback: optimization loop (conv no -> filter), on the left
xl = XC - W / 2 - 1.55
ax.add_patch(FancyArrowPatch(
    path=Path([(boxes["conv"]["x"] - boxes["conv"]["w"] / 2, boxes["conv"]["y"]),
               (xl, boxes["conv"]["y"]),
               (xl, boxes["filter"]["y"]),
               (boxes["filter"]["x"] - boxes["filter"]["w"] / 2, boxes["filter"]["y"])],
              [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]),
    arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=EDGE, zorder=5))
ax.text(xl - 0.1, (boxes["conv"]["y"] + boxes["filter"]["y"]) / 2,
        "no:  $i\\leftarrow i+1$ (next design iteration)", ha="center", va="center", rotation=90,
        fontsize=8, color=EDGE)

# ---------------------------------------------------------------- group boxes
def group(x0, y0, x1, y1, color, z=1):
    y0 *= SY
    y1 *= SY
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.02,rounding_size=0.2",
                 linewidth=1.2, edgecolor="#888888", facecolor=color, zorder=z))


# innermost creep group (label in a header strip above the first block)
group(XC - W / 2 - 0.25, 9.95, XC + W / 2 + 0.25, 13.87, C_CREEP, z=2)
ax.text(XC - W / 2 - 0.13, 13.68 * SY, "Creep return mapping", ha="left", va="center",
        fontsize=8, style="italic", color="#9b2222", zorder=6)
# forward-solve / time-stepping group
group(XC - W / 2 - 0.55, 8.5, XC + W / 2 + 0.55, 18.3, C_FWD, z=1)
ax.text(XC - W / 2 - 0.5, 18.1 * SY, "Forward solve (time-stepping)",
        ha="left", va="center", fontsize=8.2, style="italic", color="#234f8c", zorder=6)
# outer optimization-loop group
group(XC - W / 2 - 0.95, 2.45, XC + W / 2 + 0.95, 21.35, C_OPT, z=0)
ax.text(XC - W / 2 - 0.9, 21.17 * SY, "Optimization loop (MMA)",
        ha="left", va="center", fontsize=8.6, style="italic", color="#2c6e2c", zorder=6)

plt.tight_layout()
pdf = os.path.join(FIGS, "creep_optimization_flowchart.pdf")
png = os.path.join(FIGS, "creep_optimization_flowchart.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=200, bbox_inches="tight")
print("wrote", pdf)
print("wrote", png)
