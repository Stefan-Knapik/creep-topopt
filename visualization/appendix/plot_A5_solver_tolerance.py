"""Appendix B.5, Fig. B5: one figure and LaTeX table per design field (`solid`, `grey`; the paper shows `grey`).
Writes figures/A5_solver_tolerance_<design>.{pdf,png} and results/appendix/A5_solver_tolerance/tolerance_table_<design>.tex."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
RESULTS = ROOT / "results" / "appendix" / "A5_solver_tolerance"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

DEFAULT_GLOBAL_TOL = 1e-6
DEFAULT_LOCAL_ATOL = 1e-10
FLOOR = 1e-16               # a deviation of exactly zero has no place on a log axis

SERIES = [
    ("rel_dJ", r"objective $|\Delta J| / J_{\mathrm{ref}}$", "tab:red", "o"),
    ("rel_dgrad", r"sensitivity $\|\Delta \nabla J\| / \|\nabla J_{\mathrm{ref}}\|$",
     "tab:blue", "s"),
]


def save(fig, stem):
    for ext, kw in ((".png", dict(dpi=200)), (".pdf", {})):
        out = FIGS / f"{stem}{ext}"
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"saved {out}")


def _plot(ax, df, xcol, default, xlabel):
    df = df.sort_values(xcol)
    for col, label, color, marker in SERIES:
        if col not in df:
            continue
        y = df[col].to_numpy(float)
        if np.all(np.isnan(y)):
            continue
        # Points that landed bit-for-bit on the reference are drawn on the
        # floor; the table carries the exact zeros.
        ax.plot(df[xcol], np.maximum(y, FLOOR), marker=marker, color=color,
                lw=1.4, ms=6, label=label)

    ax.axvline(default, color="0.35", ls="--", lw=1.2)
    ax.annotate("selected as default", xy=(default, 0.96),
                xycoords=("data", "axes fraction"), ha="right", va="top",
                rotation=90, fontsize=8, color="0.35")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # A decade of padding at each end, so the default marker never lands on the
    # frame -- in both panels the default is at or near one end of the ladder.
    xs = df[xcol].to_numpy(float)
    ax.set_xlim(xs.max() * 6.0, xs.min() / 6.0)   # inverted: tightening rightwards
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", alpha=0.3)


def main(design):
    df = pd.read_csv(RESULTS / f"tolerance_study_{design}.csv")
    has_grad = "rel_dgrad" in df.columns
    prod = df[df["sweep"] == "production"]
    prod = prod.iloc[0] if not prod.empty else None

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)

    g = df[df["sweep"].isin(["global", "reference"])]
    _plot(axes[0], g, "global_tol", DEFAULT_GLOBAL_TOL,
          "global equilibrium tolerance\n"
          r"(with $\mathrm{relative\_tol} = 10^{-2} \times$ tolerance)")
    axes[0].set_title("(a) global equilibrium solver", fontsize=11)
    axes[0].set_ylabel("deviation from the tightest solve")

    l = df[df["sweep"].isin(["local", "reference"])]
    _plot(axes[1], l, "local_atol", DEFAULT_LOCAL_ATOL,
          "local return-map tolerance\n"
          r"(with $\mathrm{relative\_tol} = 10^{-4}$)")
    axes[1].set_title("(b) local return map", fontsize=11)

    ymin = FLOOR / 3
    axes[0].set_ylim(ymin, 3.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10), fontsize=9)
    fig.tight_layout()
    save(fig, f"A5_solver_tolerance_{design}")

    # --- table --------------------------------------------------------------
    def sci(v):
        if pd.isna(v):
            return "--"
        if v == 0.0:
            return r"$0$"
        e = int(np.floor(np.log10(abs(v))))
        m = v / 10.0**e
        return rf"${m:.2f}\times 10^{{{e}}}$"

    LABEL = {"reference": "reference (tightest)",
             "production": r"\textbf{selected as default}",
             "global": "global ladder",
             "local": "local ladder",
             "local_rtol": r"local \texttt{rtol} check"}
    rows, last = [], None
    for _, r in df.iterrows():
        if last is not None and r["sweep"] != last:
            rows.append(r"\midrule")
        last = r["sweep"]
        rows.append(
            f"{LABEL.get(r['sweep'], r['sweep'])} & {sci(r['global_tol'])} & "
            f"{sci(r['local_atol'])} & {sci(r['local_rtol'])} & "
            f"${r['J']:.10g}$ & {sci(r['rel_dJ'])} & "
            + (sci(r['rel_dgrad']) if has_grad else "--")
            + rf" & {r['wall_time_s']:.0f} \\")
    tex = "\n".join([
        r"\begin{tabular}{llllrrrr}", r"\toprule",
        r"case & global \texttt{tol} & local \texttt{atol} & local \texttt{rtol} & "
        r"$J$ & $|\Delta J| / J_{\mathrm{ref}}$ & "
        r"$\|\Delta \nabla J\| / \|\nabla J_{\mathrm{ref}}\|$ & wall (s) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    out = RESULTS / f"tolerance_table_{design}.tex"
    out.write_text(tex + "\n")
    print(f"saved {out}")

    if prod is not None:
        print(f"\nproduction defaults: rel dJ = {prod['rel_dJ']:.3e}"
              + (f", rel dgrad = {prod['rel_dgrad']:.3e}" if has_grad else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", nargs="+", default=None,
                        help="designs to plot; default is whatever is in results/")
    args = parser.parse_args()

    designs = args.design or [p.stem.split("tolerance_study_")[1]
                              for p in sorted(RESULTS.glob("tolerance_study_*.csv"))]
    for d in designs:
        print(f"--- {d} ---")
        main(d)
