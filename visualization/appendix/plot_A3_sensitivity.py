"""Appendix B.3, Fig. B3: sensitivity verification figure and summary table. Writes figures/A3_sensitivity_verify.{pdf,png}."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
RESULTS = ROOT / "results" / "appendix" / "A3_sensitivity"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

METHOD_COLOR = {"explicit": "tab:blue", "implicit": "tab:red"}
METHOD_MARKER = {"explicit": "x", "implicit": "o"}
METHODS = ("explicit", "implicit")


def load(method):
    df = pd.read_csv(RESULTS / f"sensitivity_{method}.csv")
    meta = pd.read_csv(RESULTS / f"sensitivity_{method}_meta.csv").iloc[0]
    return df, meta


def stats(ad, fd):
    ad, fd = np.asarray(ad, float), np.asarray(fd, float)
    err = np.abs(ad - fd)
    scale = np.max(np.abs(fd))
    ss_res = np.sum((ad - fd) ** 2)
    ss_tot = np.sum((fd - fd.mean()) ** 2)
    return {
        "n": len(ad),
        # Per-entry relative error.  The max is dominated by the handful of
        # design variables whose gradient is near zero, where a relative measure
        # is not meaningful, so the error normalised by the largest gradient in
        # the vector is reported alongside it.
        "median_rel": float(np.median(err / np.abs(fd))),
        "max_rel": float(np.max(err / np.abs(fd))),
        "max_err_over_max_grad": float(np.max(err) / scale),
        "R2": float(1.0 - ss_res / ss_tot),
    }


def format_box(s, meta):
    return "\n".join([
        f"n           = {s['n']}",
        f"median rel. = {s['median_rel']:.2e}",
        f"max rel.    = {s['max_rel']:.2e}",
        rf"max err / max|$\nabla J$| = {s['max_err_over_max_grad']:.1e}",
        f"$R^2$          = {1 - s['R2']:.1e} from 1" if s["R2"] < 1
        else "$R^2$          = 1.000000",
        f"h           = {meta['fd_step']:.0e}",
    ])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))

    # (a), (b) parity plots
    for ax, method in zip(axes[:2], METHODS):
        df, meta = load(method)
        s = stats(df["dJ_ad"], df["dJ_fd"])
        lim = [min(df["dJ_fd"].min(), df["dJ_ad"].min()),
               max(df["dJ_fd"].max(), df["dJ_ad"].max())]
        pad = 0.05 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, ls="--", color="0.4", lw=1.2, zorder=1)
        ax.scatter(df["dJ_fd"], df["dJ_ad"], s=22, color=METHOD_COLOR[method],
                   edgecolor="black", lw=0.3, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"central difference  $\partial J/\partial\rho_e$")
        ax.set_ylabel(r"AD and adjoint  $\partial J/\partial\rho_e$")
        ax.set_title(f"({'ab'[METHODS.index(method)]}) {method} scheme, "
                     f"all {int(meta['num_cells'])} design variables", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.annotate(format_box(s, meta), xy=(0.04, 0.96), xycoords="axes fraction",
                    va="top", ha="left", fontsize=8.5, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6",
                              alpha=0.92))

    # (c) finite-difference step-size study
    ax = axes[2]
    for method in METHODS:
        path = RESULTS / f"fd_step_study_{method}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        g = d.groupby("h")["rel_err"]
        h = np.array(sorted(d["h"].unique()))
        med = g.median().reindex(h).to_numpy()
        # Interquartile band, not min-max: the spread is dominated by the probe
        # cells whose gradient is near zero, where a relative measure says little.
        lo = g.quantile(0.25).reindex(h).to_numpy()
        hi = g.quantile(0.75).reindex(h).to_numpy()
        ax.fill_between(h, lo, hi, color=METHOD_COLOR[method], alpha=0.15)
        ax.loglog(h, med, color=METHOD_COLOR[method], marker=METHOD_MARKER[method],
                  ms=7, lw=1.8, label=f"{method} (median over probe cells)")
        med_ref, h_ref = med, h

    # Guides: central differences truncate as O(h^2) and lose to round-off as
    # O(eps/h), so the two branches bracket the optimum.  Anchored on the
    # extreme data points rather than on hand-picked constants.
    ax.loglog(h_ref, med_ref[-1] * (h_ref / h_ref[-1]) ** 2,
              color="0.4", ls="--", lw=1.1)
    ax.annotate(r"$\mathcal{O}(h^2)$ truncation error",
                xy=(h_ref[-1], med_ref[-1]), xytext=(-6, 8),
                textcoords="offset points", color="0.4", fontsize=9, ha="right")
    ax.loglog(h_ref, med_ref[0] * (h_ref[0] / h_ref),
              color="0.4", ls="-.", lw=1.1)
    ax.annotate(r"$\mathcal{O}(\epsilon/h)$ round-off error",
                xy=(h_ref[0], med_ref[0]), xytext=(22, 6),
                textcoords="offset points", color="0.4", fontsize=9)
    ax.set_ylim(min(np.nanmin(med_ref) * 0.2, 1e-9), np.nanmax(hi) * 5)

    ax.set_xlabel(r"finite-difference step $h$")
    ax.set_ylabel("relative error vs the AD gradient")
    ax.set_title("(c) step-size study, per-cell derivatives",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper center")

    fig.tight_layout()
    for ext, kw in ((".png", dict(dpi=200)), (".pdf", {})):
        out = FIGS / f"A3_sensitivity_verify{ext}"
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"saved {out}")

    # --- summary tables ---------------------------------------------------
    print("\nPer-cell parity (coarse mesh, every design variable):")
    rows = []
    for method in METHODS:
        df, meta = load(method)
        s = stats(df["dJ_ad"], df["dJ_fd"])
        s.update(method=method, nx=int(meta["nx"]), ny=int(meta["ny"]),
                 num_time_steps=int(meta["num_time_steps"]), fd_step=meta["fd_step"])
        rows.append(s)
    summary = pd.DataFrame(rows)[
        ["method", "nx", "ny", "num_time_steps", "fd_step", "n",
         "median_rel", "max_rel", "max_err_over_max_grad", "R2"]]
    print(summary.to_string(index=False))
    out = RESULTS / "parity_summary.csv"
    summary.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
