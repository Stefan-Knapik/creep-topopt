"""Appendix B.4, Fig. B4: mesh convergence figure and Richardson tables. Writes figures/A4_mesh_convergence.{pdf,png}."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
RESULTS = ROOT / "results" / "appendix" / "A4_mesh_convergence"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

METHOD_COLOR = {"explicit": "tab:blue", "implicit": "tab:red"}
METHOD_MARKER = {"explicit": "x", "implicit": "o"}

PAPER_H = 1.0          # mm, the element size used throughout the paper

# Integral quantities, which converge under refinement.
CONVERGENT = [
    ("J", r"$J = C(t_{end}) - C(0)$", "tab:red", "o"),
    ("C0", r"$C(0)$", "tab:purple", "s"),
    ("probe_tip_end", r"$u_y$ at $(L_x, 0)$", "tab:blue", "^"),
    ("probe_topright_end", r"$u_y$ at $(L_x, L_y)$", "tab:cyan", "v"),
    ("vm_avg_end", r"$\langle\sigma_{eq}\rangle$", "tab:green", "D"),
    ("eps_eq_avg_end", r"$\langle\varepsilon^c_{eq}\rangle$", "tab:olive", "P"),
]
# Pointwise peaks, which do not: the traction jump at (Lx, 0.2 Ly) and the
# clamped corners are stress singularities.
SINGULAR = [
    ("vm_max_end", r"$\max\,\sigma_{eq}$", "tab:red", "o"),
    ("eps_eq_max_end", r"$\max\,\varepsilon^c_{eq}$", "tab:orange", "s"),
]
REGULARISED = [
    ("vm_avg_end", r"$\langle\sigma_{eq}\rangle$", "tab:blue", "^"),
]


def save(fig, stem):
    for ext, kw in ((".png", dict(dpi=200)), (".pdf", {})):
        out = FIGS / f"{stem}{ext}"
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"saved {out}")


def richardson(h, q):
    """Observed order, extrapolated value and a GCI-style band from the three
    finest levels of a factor-two refinement ladder.

    p = log((q_4h - q_2h) / (q_2h - q_h)) / log 2, which unlike an error plot
    against the finest run does not consume a level as the reference.
    """
    order = np.argsort(h)          # fine -> coarse
    h, q = np.asarray(h)[order], np.asarray(q)[order]
    if len(q) < 3:
        return np.nan, np.nan, np.nan
    q_h, q_2h, q_4h = q[0], q[1], q[2]
    num, den = q_4h - q_2h, q_2h - q_h
    if den == 0 or num / den <= 0:
        return np.nan, np.nan, np.nan
    p = float(np.log(num / den) / np.log(h[1] / h[0]))
    q_ext = float(q_h + (q_h - q_2h) / (2.0**p - 1.0))
    gci = float(1.25 * abs((q_h - q_2h) / q_h) / (2.0**p - 1.0))
    return p, q_ext, gci


def _ladder_ticks(ax, h):
    """Label the rungs of the refinement ladder, not the decade minor ticks."""
    h = np.unique(np.asarray(h, float))
    ax.set_xticks(h)
    ax.set_xticklabels([f"{v:g}" for v in h])
    ax.minorticks_off()


def plot_mesh(ref_method="implicit"):
    df = pd.read_csv(RESULTS / "mesh_convergence.csv")
    if not df["finite_all_steps"].all():
        bad = df[~df["finite_all_steps"]][["method", "nx", "ny"]]
        print("WARNING: non-finite runs present:\n" + bad.to_string(index=False))
    methods = [m for m in ("implicit", "explicit") if (df["method"] == m).any()]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # --- (a) self-convergence of the integral quantities -------------------
    ax = axes[0]
    d = df[df["method"] == ref_method].sort_values("h_mm")
    h = d["h_mm"].to_numpy()
    for key, label, color, marker in CONVERGENT:
        if key not in d:
            continue
        q = d[key].to_numpy()
        # Relative difference from the finest level.  The finest point is the
        # reference (zero by construction) and the second finest is contaminated
        # by it, so neither enters the fit.
        err = np.abs(q - q[0]) / abs(q[0])
        ax.loglog(h[1:], err[1:], color=color, marker=marker, ms=6, lw=1.6,
                  label=label)
    ax.axvline(PAPER_H, color="0.4", ls=":", lw=1.2)
    _ladder_ticks(ax, h[1:])
    ax.set_xlabel("element size $h$ [mm]")
    ax.set_ylabel(rf"relative difference from $h = {h.min():g}$ mm")
    ax.set_title("(a) converged quantities", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    # --- (b) the objective, with Richardson extrapolation ------------------
    ax = axes[1]
    rows = []
    for method in methods:
        d = df[df["method"] == method].sort_values("h_mm")
        hh, q = d["h_mm"].to_numpy(), d["J"].to_numpy()
        p, q_ext, gci = richardson(hh, q)
        ax.plot(hh, q, color=METHOD_COLOR[method], marker=METHOD_MARKER[method],
                ms=7, lw=1.8, label=rf"{method}")
        if np.isfinite(q_ext):
            ax.axhline(q_ext, color=METHOD_COLOR[method], ls="--", lw=1.2)
        for key, label in (("J", "J"), ("C0", "C(0)"),
                           ("probe_tip_end", "uy_tip_end"),
                           ("vm_avg_end", "vm_avg_end"),
                           ("eps_eq_avg_end", "eps_eq_avg_end")):
            if key not in d:
                continue
            pk, qk, gk = richardson(d["h_mm"].to_numpy(), d[key].to_numpy())
            at_paper = d[np.isclose(d["h_mm"], PAPER_H)]
            v_paper = float(at_paper[key].iloc[0]) if len(at_paper) else np.nan
            rows.append({"method": method, "quantity": label,
                         "value_at_h1mm": v_paper,
                         "finest_h_mm": float(hh.min()),
                         "finest_value": float(d.sort_values("h_mm")[key].iloc[0]),
                         "richardson_p": pk, "richardson_value": qk, "gci": gk,
                         "rel_err_at_h1mm": abs(v_paper - qk) / abs(qk)
                         if np.isfinite(qk) else np.nan})
    ax.axvline(PAPER_H, color="0.4", ls=":", lw=1.2)
    ref = [r for r in rows if r["method"] == ref_method and r["quantity"] == "J"]
    if ref and np.isfinite(ref[0]["rel_err_at_h1mm"]):
        ax.annotate(f"at $h = 1$ mm, $J$ is within "
                    f"{100*ref[0]['rel_err_at_h1mm']:.2f}%\n"
                    f"of the extrapolated value",
                    xy=(0.04, 0.08), xycoords="axes fraction", fontsize=8.5,
                    color="0.25",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6",
                              alpha=0.9))
    ax.set_xscale("log")
    _ladder_ticks(ax, df["h_mm"].unique())
    ax.set_xlabel("element size $h$ [mm]")
    ax.set_ylabel(r"creep compliance $J$")
    ax.set_title("(b) objective vs mesh; Richardson extrapolation (dashed)",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # --- (c) singular vs regularised --------------------------------------
    ax = axes[2]
    d = df[df["method"] == ref_method].sort_values("h_mm")
    hh = d["h_mm"].to_numpy()
    for key, label, color, marker in SINGULAR:
        v = d[key].to_numpy()
        ax.loglog(hh, v / v[-1], color=color, marker=marker, ms=7, lw=1.8,
                  label=f"{label} (does not converge)")
    for key, label, color, marker in REGULARISED:
        v = d[key].to_numpy()
        ax.loglog(hh, v / v[-1], color=color, marker=marker, ms=7, lw=1.8, ls="--",
                  label=f"{label} (converges)")
    ax.axvline(PAPER_H, color="0.4", ls=":", lw=1.2)
    _ladder_ticks(ax, hh)
    ax.set_xlabel("element size $h$ [mm]")
    ax.set_ylabel("value, normalised by the coarsest mesh")
    ax.set_title("(c) pointwise peaks do not converge:\n"
                 "the clamped corners are stress singularities", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    save(fig, "A4_mesh_convergence")

    tab = pd.DataFrame(rows)
    out = RESULTS / "mesh_richardson.csv"
    tab.to_csv(out, index=False)
    print(f"saved {out}")

    print("\nMesh ladder:")
    cols = ["method", "nx", "ny", "h_mm", "num_cells", "num_dofs", "C0", "J",
            "probe_tip_end", "vm_avg_end", "vm_max_end", "finite_all_steps",
            "wall_time_s"]
    print(df[[c for c in cols if c in df]].sort_values(["method", "h_mm"])
          .to_string(index=False))
    print("\nRichardson extrapolation:")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    plot_mesh()
