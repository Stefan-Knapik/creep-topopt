"""Appendix B.1, Fig. B1: TOGA vs Abaqus vs closed form for Norton stress relaxation. Writes
figures/A1_creep_validation.{pdf,png} and results/appendix/A1_abaqus_validation/validation_errors*.csv."""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]  # repository root
RESULTS = ROOT / "results" / "appendix" / "A1_abaqus_validation"
TOGA = RESULTS / "toga"
ABAQUS = ROOT / "experiments" / "appendix" / "A1_abaqus_validation" / "abaqus"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Must match experiments/appendix/A1_abaqus_validation/run_simulation.py and the two Abaqus drivers.
E_MOD, A_CR, N_CR = 160e3, 1e-21, 3.5
LX = LY = 1.0
U_BAR = 0.002
NUM_TIME_STEPS = 32       # matches the 32 fixed Abaqus increments
NUM_TIME_STEPS_REF = 1024  # near-converged reference for the closed-form comparison

_reddish_orange = np.array([255, 127, 14]) / 255
COMP_COLOR = {"xx": "tab:blue", "yy": "tab:orange", "xy": "tab:green"}
COMP_MARKER = {"xx": "o", "yy": "s", "xy": "^"}
METHOD_STYLE = {"explicit": dict(ls="-"), "implicit": dict(ls=":")}
METHOD_LW = 2.0

# Text sizing.  Every piece of text in the figure is sized through fs(), so the
# whole figure can be scaled up with FONT_SCALE (or the FIG_FONT_SCALE env var,
# e.g. FIG_FONT_SCALE=1.4 python plot_A1_abaqus_validation.py) without touching the layout.
# Individual roles can still be retuned in FONT_SIZE.
FONT_SCALE = float(os.environ.get("FIG_FONT_SCALE", 1.0))
FONT_SIZE = {
    "tick": 10,        # axis tick labels
    "label": 11,       # axis labels
    "title": 11,       # panel titles
    "legend": 8,       # legend entries
    "annotation": 9,   # "sampled quadrature point"
    "bc": 11,          # boundary-condition labels in the schematics
}


def fs(role, scale=None):
    """Font size for a text role, scaled by FONT_SCALE."""
    return FONT_SIZE[role] * (FONT_SCALE if scale is None else scale)


def apply_font_scale(scale=None):
    """Push the scaled sizes into rcParams (ticks, offset text, defaults)."""
    global FONT_SCALE
    if scale is not None:
        FONT_SCALE = float(scale)
    plt.rcParams.update({
        "font.size": fs("tick"),
        "axes.labelsize": fs("label"),
        "axes.titlesize": fs("title"),
        "xtick.labelsize": fs("tick"),
        "ytick.labelsize": fs("tick"),
        "legend.fontsize": fs("legend"),
    })


def analytic_relaxation(t, e_total=U_BAR / LX, E=E_MOD, A=A_CR, n=N_CR):
    """Exact uniaxial Norton relaxation at constant total strain; see
    experiments/appendix/A1_abaqus_validation/run_simulation.py for the derivation."""
    t = np.asarray(t, dtype=float)
    sigma_0 = E * e_total
    sigma = (sigma_0 ** (1.0 - n) + (n - 1.0) * E * A * t) ** (1.0 / (1.0 - n))
    return sigma, e_total - sigma / E


# --------------------------------------------------------------------------
# Setup schematics
# --------------------------------------------------------------------------

def plot_validation_setup(ax, case, L=LX, n_ele=2, u_bar=U_BAR):
    """Draw the geometry, mesh, quadrature points and BCs of one coupon.

    Square plate, n_ele x n_ele QUAD4 / CPS4 mesh, plane stress.  Element edges
    and the 2x2 Gauss points of every element are drawn; the bottom-left Gauss
    point of the bottom-left element -- where the history is sampled -- is
    highlighted.  Red glyphs are homogeneous Dirichlet constraints, blue glyphs
    are the prescribed non-zero displacement.
    """
    ax.set_aspect("equal")
    ax.set_axis_off()

    xs = np.linspace(0.0, L, n_ele + 1)
    polygons = [
        [(xs[i], xs[j]), (xs[i + 1], xs[j]), (xs[i + 1], xs[j + 1]), (xs[i], xs[j + 1])]
        for j in range(n_ele)
        for i in range(n_ele)
    ]
    ax.add_collection(PolyCollection(
        polygons, facecolors="lightgray", edgecolors="black", linewidths=0.8,
    ))
    ax.add_patch(plt.Rectangle(
        (0.0, 0.0), L, L, fill=False, edgecolor="black", linewidth=1.4, zorder=4,
    ))

    # 2x2 Gauss points per element
    h = L / n_ele
    g = 1.0 / np.sqrt(3.0)
    off = 0.5 * h * np.array([1.0 - g, 1.0 + g])
    qx = np.concatenate([xs[i] + off for i in range(n_ele)])
    qy = np.concatenate([xs[j] + off for j in range(n_ele)])
    QX, QY = np.meshgrid(qx, qy)
    ax.plot(QX.ravel(), QY.ravel(), marker="o", markersize=4.0, color="0.25",
            linestyle="none", zorder=5)

    x_pt, y_pt = off[0], off[0]
    ax.plot([x_pt], [y_pt], marker="o", markersize=10, color=_reddish_orange,
            markeredgecolor="black", markeredgewidth=1.0, linestyle="none", zorder=7)
    # Sits just below the plate, clear of the u_y = 0 glyphs and label that the
    # tension case carries along the bottom edge.
    xytext = (0.60 * L, -0.17 * L) if case == "tension" else (0.52 * L, -0.15 * L)
    ax.annotate(
        "sampled\nquadrature point",
        xy=(x_pt, y_pt), xycoords="data",
        xytext=xytext, textcoords="data",
        color=_reddish_orange, fontsize=fs("annotation"), ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>", color=_reddish_orange, linewidth=1.4,
                        shrinkA=4, shrinkB=6, connectionstyle="arc3,rad=0.15"),
        annotation_clip=False, zorder=7,
    )

    tri_shift = 0.025 * L
    y_tri = np.linspace(0.0, L, 3)

    if case == "tension":
        # ux = 0 on the left, uy = 0 on the bottom (roller supports).
        ax.plot(np.full_like(y_tri, -tri_shift), y_tri, marker=">", markersize=7,
                color="red", linestyle="none", clip_on=False, zorder=5)
        ax.text(-tri_shift - 0.13 * L, 0.5 * L, r"$u_x = 0$", color="red",
                ha="center", va="center", rotation=90, fontsize=fs("bc"),
                clip_on=False, zorder=6)
        ax.plot(y_tri, np.full_like(y_tri, -tri_shift), marker="^", markersize=7,
                color="red", linestyle="none", clip_on=False, zorder=5)
        ax.text(0.18 * L, -tri_shift - 0.10 * L, r"$u_y = 0$", color="red",
                ha="center", va="top", fontsize=fs("bc"), clip_on=False, zorder=6)
        arrow_dx, arrow_dy = 0.10 * L, 0.0
        bc_label = "prescribed\n" rf"$u_x = {u_bar:g}$"
    elif case == "mixed":
        # Fully fixed left edge, fully driven right edge.
        ax.plot(np.full_like(y_tri, -tri_shift), y_tri, marker=">", markersize=7,
                color="red", linestyle="none", clip_on=False, zorder=5)
        ax.text(-tri_shift - 0.13 * L, 0.5 * L, r"$u_x = u_y = 0$", color="red",
                ha="center", va="center", rotation=90, fontsize=fs("bc"),
                clip_on=False, zorder=6)
        arrow_dx = arrow_dy = 0.075 * L
        bc_label = "prescribed\n" rf"$u_x = u_y = {u_bar:g}$"
    else:
        raise ValueError(f"Unknown case: {case!r}")

    ax.plot(np.full_like(y_tri, L + tri_shift), y_tri, marker="<", markersize=7,
            color="blue", linestyle="none", clip_on=False, zorder=5)
    x_tail = L + 0.05 * L
    for y0 in np.linspace(0.25 * L, 0.75 * L, 2):
        ax.annotate("", xy=(x_tail + arrow_dx, y0 + arrow_dy), xytext=(x_tail, y0),
                    arrowprops=dict(arrowstyle="-|>", color="blue", linewidth=1.8,
                                    joinstyle="miter"),
                    annotation_clip=False, zorder=6)
    ax.text(x_tail + arrow_dx + 0.09 * L, 0.5 * L, bc_label, color="blue",
            ha="center", va="center", rotation=90, fontsize=fs("bc"),
            clip_on=False, zorder=6)

    ax.set_xlim(-0.02 * L, 1.02 * L)
    ax.set_ylim(-0.02 * L, 1.02 * L)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_toga(case, method, n=NUM_TIME_STEPS):
    sig = pd.read_csv(TOGA / f"stress_{case}_{method}_{n}.csv")
    eps = pd.read_csv(TOGA / f"creep_strain_{case}_{method}_{n}.csv")
    return sig, eps


def load_abaqus(case):
    """Abaqus creep-step history at element 1, integration point 1.

    The creep step starts at TotalTime = 1 s (after the 1 s load step), so the
    load-step duration is subtracted to align t = 0 with the start of creep.
    CEE12 is the engineering shear strain (gamma_xy = 2 eps_xy) and is halved
    to match the tensorial component TOGA stores.
    """
    suffix = "_mixedBC" if case == "mixed" else ""
    sig = pd.read_csv(ABAQUS / f"stresses{suffix}.csv")
    eps = pd.read_csv(ABAQUS / f"strains{suffix}.csv")

    def _filter(df):
        m = (df["Step"] == "Creep") & (df["Element"] == 1) & (df["IP"] == 1)
        return df.loc[m].sort_values("TotalTime_s")

    sig, eps = _filter(sig), _filter(eps)
    sig_t = sig["TotalTime_s"].to_numpy() - 1.0
    eps_t = eps["TotalTime_s"].to_numpy() - 1.0
    sig_out = {"time": sig_t, "xx": sig["S11"].to_numpy(),
               "yy": sig["S22"].to_numpy(), "xy": sig["S12"].to_numpy()}
    eps_out = {"time": eps_t, "xx": eps["CEE11"].to_numpy(),
               "yy": eps["CEE22"].to_numpy(), "xy": eps["CEE12"].to_numpy() / 2.0}
    return sig_out, eps_out


def rel_dev(a, b, scale=None):
    """Max |a-b| normalised by the peak magnitude of the reference (so that
    components that are identically zero do not blow the metric up)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.max(np.abs(b)) if scale is None else scale
    if denom == 0.0:
        return np.max(np.abs(a - b)), np.nan
    return np.max(np.abs(a - b)), np.max(np.abs(a - b)) / denom


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def main(methods=("explicit", "implicit"), stem="A1_creep_validation",
         font_scale=None, figsize=(15.0, 8.0)):
    """Build the validation figure for the given time-integration schemes.

    `methods` may be restricted to a single scheme: the like-for-like comparison
    with Abaqus is against the explicit scheme (Abaqus's VISCO step with
    cetol = 1e30 integrates creep explicitly), so an explicit-only version is the
    cleaner statement of the code-to-code validation.

    `font_scale` multiplies every font size (default FONT_SCALE / the
    FIG_FONT_SCALE env var).  Enlarge `figsize` along with it if the text starts
    to crowd the panels.
    """
    apply_font_scale(font_scale)
    cases = [("tension", ["xx", "yy"]), ("mixed", ["xx", "yy", "xy"])]
    fig, axes = plt.subplots(2, 3, figsize=figsize,
                             gridspec_kw={"width_ratios": [0.85, 1.0, 1.0]})
    error_rows = []

    for row, (case, comps) in enumerate(cases):
        ax_sch, ax_eps, ax_sig = axes[row]
        plot_validation_setup(ax_sch, case)

        abq_sig, abq_eps = load_abaqus(case)
        t_ref = None

        for method in methods:
            sig, eps = load_toga(case, method)
            t = sig["time"].to_numpy()
            t_ref = t
            style = METHOD_STYLE[method]
            for comp in comps:
                ax_eps.plot(t, eps[f"creep_strain_{comp}"],
                            color=COMP_COLOR[comp], lw=METHOD_LW, **style)
                ax_sig.plot(t, sig[f"sigma_{comp}"],
                            color=COMP_COLOR[comp], lw=METHOD_LW, **style)

                a_eps, r_eps = rel_dev(np.interp(abq_eps["time"], t, eps[f"creep_strain_{comp}"]),
                                       abq_eps[comp], scale=np.max(np.abs(abq_eps["xx"])))
                a_sig, r_sig = rel_dev(np.interp(abq_sig["time"], t, sig[f"sigma_{comp}"]),
                                       abq_sig[comp], scale=np.max(np.abs(abq_sig["xx"])))
                error_rows += [
                    dict(case=case, method=method, num_time_steps=NUM_TIME_STEPS,
                         quantity=f"creep_strain_{comp}", reference="abaqus",
                         max_abs_err=a_eps, max_rel_err=r_eps),
                    dict(case=case, method=method, num_time_steps=NUM_TIME_STEPS,
                         quantity=f"sigma_{comp}", reference="abaqus",
                         max_abs_err=a_sig, max_rel_err=r_sig),
                ]

        for comp in comps:
            ax_eps.plot(abq_eps["time"], abq_eps[comp], color=COMP_COLOR[comp],
                        ls="none", marker=COMP_MARKER[comp], mfc="none", ms=7)
            ax_sig.plot(abq_sig["time"], abq_sig[comp], color=COMP_COLOR[comp],
                        ls="none", marker=COMP_MARKER[comp], mfc="none", ms=7)

        # The exact solution for the uniaxial case.  The biaxial case has no
        # closed form and is validated against Abaqus alone.
        if case == "tension":
            t_fine = np.linspace(0.0, t_ref[-1], 400)
            sig_ex, eps_ex = analytic_relaxation(t_fine)
            ax_sig.plot(t_fine, sig_ex, color="k", ls="--", lw=1.6)
            # Uniaxial tension is traction free in y: sigma_yy vanishes exactly.
            ax_sig.plot(t_fine, np.zeros_like(t_fine), color="k", ls="--", lw=1.6)
            ax_eps.plot(t_fine, eps_ex, color="k", ls="--", lw=1.6)
            ax_eps.plot(t_fine, -0.5 * eps_ex, color="k", ls="--", lw=1.6)

        if case == "tension":
            # Deviation from the closed form, at the Abaqus increment size and
            # at a temporally converged step count.
            for n_steps in (NUM_TIME_STEPS, NUM_TIME_STEPS_REF):
                for method in methods:
                    try:
                        sig_n, eps_n = load_toga(case, method, n_steps)
                    except FileNotFoundError:
                        continue
                    t_n = sig_n["time"].to_numpy()
                    s_ex, e_ex = analytic_relaxation(t_n)
                    a_s, r_s = rel_dev(sig_n["sigma_xx"], s_ex)
                    a_e, r_e = rel_dev(eps_n["creep_strain_xx"], e_ex)
                    error_rows += [
                        dict(case=case, method=method, num_time_steps=n_steps,
                             quantity="sigma_xx", reference="analytical",
                             max_abs_err=a_s, max_rel_err=r_s),
                        dict(case=case, method=method, num_time_steps=n_steps,
                             quantity="creep_strain_xx", reference="analytical",
                             max_abs_err=a_e, max_rel_err=r_e),
                    ]

        ax_eps.set_ylabel(r"creep strain $\varepsilon^c$")
        ax_sig.set_ylabel(r"stress $\sigma$ [MPa]")
        # One legend per axes, listing only what that panel actually shows.
        # Coloured entries are the tensor components, grey entries the data
        # sources; spelling out every (component, source) pair would take 8-13
        # entries, and a single shared legend under the figure sat too far from
        # the curves to read.
        comp_handles = [Line2D([], [], color=COMP_COLOR[c], lw=2.2,
                               marker=COMP_MARKER[c], mfc="none", ms=6,
                               label=rf"${c}$") for c in comps]
        src_handles = []
        if case == "tension":
            src_handles.append(Line2D([], [], color="k", lw=1.6, ls="--",
                                      label="closed form"))
        src_handles += [Line2D([], [], color="0.3", lw=METHOD_LW,
                               label=rf"JAX-FEM {m}, $N={NUM_TIME_STEPS}$",
                               **METHOD_STYLE[m]) for m in methods]
        src_handles.append(Line2D([], [], color="0.3", lw=0, marker="o",
                                  mfc="none", ms=7,
                                  label=rf"Abaqus explicit, $N={NUM_TIME_STEPS}$"))
        handles = comp_handles + src_handles
        for ax in (ax_eps, ax_sig):
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("time [s]")
            # Headroom scaled to the legend height, so it sits above the curves
            # instead of on top of them.
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.115 * math.ceil(len(handles) / 2) * (hi - lo))
            ax.legend(handles=handles, fontsize=fs("legend"), loc="upper left",
                      framealpha=0.92, ncol=2, columnspacing=1.0,
                      handlelength=2.2)

    axes[0, 0].set_title("(a) uniaxial tension: setup", fontsize=fs("title"))
    axes[0, 1].set_title("(b) uniaxial tension: creep strain", fontsize=fs("title"))
    axes[0, 2].set_title("(c) uniaxial tension: stress", fontsize=fs("title"))
    axes[1, 0].set_title("(d) mixed BC: setup", fontsize=fs("title"))
    axes[1, 1].set_title("(e) mixed BC: creep strain", fontsize=fs("title"))
    axes[1, 2].set_title("(f) mixed BC: stress", fontsize=fs("title"))

    fig.tight_layout()
    for ext, kw in ((".png", dict(dpi=200)), (".pdf", {})):
        out = FIGS / f"{stem}{ext}"
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"saved {out}")

    df = pd.DataFrame(error_rows)
    out = RESULTS / f"validation_errors{'' if len(methods) > 1 else '_' + methods[0]}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    # Explicit only: Abaqus integrates creep explicitly here, so this is the like-for-like comparison.
    main(methods=("explicit",), stem="A1_creep_validation")
