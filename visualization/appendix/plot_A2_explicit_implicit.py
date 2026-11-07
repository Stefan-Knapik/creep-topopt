"""Appendix B.2, Fig. B2: time-step refinement study. Writes figures/A2_scheme_convergence.{pdf,png}."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]  # repository root
RESULTS = ROOT / "results" / "appendix" / "A2_explicit_implicit"
TRAJ = RESULTS / "traj"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

METHOD_COLOR = {"explicit": "tab:blue", "implicit": "tab:red"}
METHOD_MARKER = {"explicit": "x", "implicit": "o"}
TRAJ_STEPS = [1, 4, 16]
TRAJ_STYLE = {1: dict(ls=":", lw=1.6), 4: dict(ls="--", lw=1.8), 16: dict(ls="-", lw=2.0)}

E_MOD, A_CR, N_CR = 160e3, 1e-21, 3.5
U_BAR, LX = 0.002, 1.0
END_TIME = 1e10

SIGMA_0 = E_MOD * U_BAR / LX
# Past the explicit stability limit the state does not merely lose accuracy, it
# grows without bound and then turns non-finite.  Such samples carry no
# information and would either blank out a curve (nan) or flatten the axes by
# stretching them over tens of decades, so they are withheld from the data
# series and reported as off-scale instead.
SIGMA_BLOWUP = 10.0 * SIGMA_0   # |sigma| beyond this is a diverged state
# Lower limit of the panel (a) stress axis.  A diverging explicit run swings far
# negative before it blows up, which auto-scaling then pays for by squashing the
# physical band; pinning the floor at 0 keeps the relaxation curves legible and
# lets the excursion run off the bottom.  None auto-scales to the plotted data.
TRAJ_Y_FLOOR = 0.0
ERR_BLOWUP = 10.0               # relative error beyond this is a diverged state
# The slope-1 reference is drawn parallel to the data but offset upwards by this
# factor, so that it reads as a guide rather than disappearing behind the curves.
SLOPE_GUIDE_LIFT = 4.0


def analytic_relaxation(t):
    t = np.asarray(t, dtype=float)
    sigma_0 = E_MOD * U_BAR / LX
    return (sigma_0 ** (1.0 - N_CR) + (N_CR - 1.0) * E_MOD * A_CR * t) ** (1.0 / (1.0 - N_CR))


def stable_prefix(y):
    """Number of leading samples of `y` that are finite and bounded."""
    y = np.asarray(y, float)
    bad = ~(np.isfinite(y) & (np.abs(y) <= SIGMA_BLOWUP))
    return int(np.argmax(bad)) if bad.any() else len(y)


def main(traj_floor=TRAJ_Y_FLOOR):
    df = pd.read_csv(RESULTS / "scheme_convergence.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    ax_traj, ax_err_t, ax_err_m = axes

    # (a) trajectories, tension coupon.  A run that leaves the bounded regime is
    # drawn up to its last physical sample and flagged there; the divergent tail
    # is not plotted.
    shown, diverged = [], []
    for method in ("explicit", "implicit"):
        for n in TRAJ_STEPS:
            path = TRAJ / f"tension_{method}_{n}.csv"
            if not path.exists():
                continue
            d = pd.read_csv(path)
            t = d["time"].to_numpy(float)
            s = d["sigma_xx"].to_numpy(float)
            k = stable_prefix(s)
            if k:
                ax_traj.plot(t[:k], s[:k], color=METHOD_COLOR[method],
                             marker=METHOD_MARKER[method], ms=5, mfc="none",
                             **TRAJ_STYLE[n])
                shown.append(s[:k])
            if k < len(s):
                diverged.append((method, n))
                if k:
                    ax_traj.plot(t[k - 1], s[k - 1], color=METHOD_COLOR[method],
                                 marker="*", ms=13, ls="none", zorder=5)
    t_fine = np.linspace(0.0, END_TIME, 400)
    sig_fine = analytic_relaxation(t_fine)
    ax_traj.plot(t_fine, sig_fine, color="k", ls="-", lw=2.2, zorder=1, alpha=0.8)
    shown.append(sig_fine)

    vals = np.concatenate(shown)
    if traj_floor is None:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(hi - lo, 1.0)
        ax_traj.set_ylim(min(lo, 0.0) - pad, hi + pad)
    else:
        # Scale the top to what is visible above the floor, not to the excursion
        # that is being cut off.
        above = vals[vals >= traj_floor]
        hi = float(np.max(above)) if above.size else float(np.max(vals))
        ax_traj.set_ylim(traj_floor, hi + 0.08 * max(hi - traj_floor, 1.0))
    ax_traj.set_xlabel("time [s]")
    ax_traj.set_ylabel(r"$\sigma_{xx}$ [MPa]")
    ax_traj.set_title("(a) uniaxial coupon: relaxation history", fontsize=11)
    ax_traj.grid(True, alpha=0.3)
    handles = ([Line2D([], [], color="k", lw=2.2, label="closed form")]
               + [Line2D([], [], color="0.3", label=rf"$N = {n}$", **TRAJ_STYLE[n])
                  for n in TRAJ_STEPS]
               + [Line2D([], [], color=METHOD_COLOR[m], lw=2, marker=METHOD_MARKER[m],
                         mfc="none", label=m) for m in ("explicit", "implicit")])
    if diverged:
        handles.append(Line2D([], [], color="0.3", marker="*", ms=11, ls="none",
                              label="divergence (tail omitted)"))
        note = ", ".join(f"{m} $N = {n}$" for m, n in diverged)
        ax_traj.text(0.40, 0.04, f"diverged: {note}", transform=ax_traj.transAxes,
                     fontsize=8, color="0.35")
    ax_traj.legend(handles=handles, fontsize=8.5, ncol=2, loc="upper right")

    # (b), (c) error vs step size
    panels = [
        (ax_err_t, "tension", "err_vs_exact",
         "(b) uniaxial coupon: error vs closed form",
         r"$|\sigma_{xx}(t_{end}) - \sigma^{exact}_{xx}| \,/\, \sigma^{exact}_{xx}$"),
        (ax_err_m, "mixed", "err_vs_ref",
         "(c) mixed-BC coupon: error vs converged reference",
         r"$|\sigma_{xx}(t_{end}) - \sigma^{ref}_{xx}| \,/\, \sigma^{ref}_{xx}$"),
    ]
    n_ref = int(df["num_time_steps"].max())
    for ax, case, err_col, title, ylabel in panels:
        series = {}
        for method in ("explicit", "implicit"):
            d = df[(df["case"] == case) & (df["method"] == method)
                   & (df["num_time_steps"] < n_ref)].sort_values("dt_s")
            if d.empty:
                continue
            dt, err = d["dt_s"].to_numpy(float), d[err_col].to_numpy(float)
            stable = (d["stable"].to_numpy(bool) if "stable" in d.columns
                      else np.ones(len(d), bool))
            # A run counts as usable only if the solver stayed finite *and* the
            # resulting error is a number that can sit on a log axis.
            ok = stable & np.isfinite(err) & (err > 0.0) & (err <= ERR_BLOWUP)
            series[method] = (dt, err, ok)

        # Head-room above the largest usable error so the off-scale carets sit
        # inside the frame instead of on top of the data.
        finite = [e[o] for _, e, o in series.values() if o.any()]
        if finite:
            allerr = np.concatenate(finite)
            ax.set_ylim(0.4 * float(allerr.min()), 8.0 * float(allerr.max()))
        y_caret = ax.get_ylim()[1] / 2.5

        for method, (dt, err, ok) in series.items():
            if ok.any():
                ax.loglog(dt[ok], err[ok], color=METHOD_COLOR[method], lw=1.8,
                          label=method)
            # Every usable sample carries the same marker.  Steps beyond the
            # explicit stability limit produce no finite error at all -- the
            # response there is not merely inaccurate but qualitatively wrong --
            # so they are drawn as carets on the top edge rather than given a
            # spurious ordinate.
            ax.loglog(dt[ok], err[ok], color=METHOD_COLOR[method],
                      marker=METHOD_MARKER[method], ms=7, ls="none")
            if (~ok).any():
                lbl = (f"{method}: diverged" if ok.any()
                       else f"{method}: diverged at every $\\Delta t$")
                ax.loglog(dt[~ok], np.full((~ok).sum(), y_caret),
                          color=METHOD_COLOR[method], marker="^", ms=9, mfc="none",
                          ls="none", label=lbl)

        # First-order reference line, anchored on the finest implicit point but
        # lifted by SLOPE_GUIDE_LIFT decades so it runs alongside the data
        # instead of underneath it.
        d = df[(df["case"] == case) & (df["method"] == "implicit")
               & (df["num_time_steps"] < n_ref)].sort_values("dt_s")
        if not d.empty and np.isfinite(d[err_col].iloc[0]) and d[err_col].iloc[0] > 0:
            dt0 = float(d["dt_s"].iloc[0])
            e0 = float(d[err_col].iloc[0]) * SLOPE_GUIDE_LIFT
            dts = np.array([dt0, dt0 * 20.0])
            ax.loglog(dts, e0 * (dts / dt0), color="0.4", ls="--", lw=1.2, zorder=1)
            ax.annotate("slope 1", xy=(dts[1], e0 * 20.0), xytext=(-2, 6),
                        textcoords="offset points", color="0.4", fontsize=9,
                        ha="right", va="bottom")

        ax.set_xlabel(r"$\Delta t$ [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    for ext, kw in ((".png", dict(dpi=200)), (".pdf", {})):
        out = FIGS / f"A2_scheme_convergence{ext}"
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"saved {out}")

    # Summary table for the appendix text.
    print()
    cols = ["case", "method", "num_time_steps", "dt_s", "sigma_xx_end",
            "err_vs_exact", "err_vs_ref", "stable"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-floor", default=str(TRAJ_Y_FLOOR),
                        help="lower limit of the panel (a) stress axis [MPa], "
                             "or 'auto' to fit the plotted data "
                             f"(default: {TRAJ_Y_FLOOR})")
    args = parser.parse_args()
    floor = None if args.traj_floor.lower() == "auto" else float(args.traj_floor)
    main(traj_floor=floor)
