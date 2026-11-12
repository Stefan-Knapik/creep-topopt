"""Rotating blade: figures of the centrifugal-loading study, in the style of visualization/5_blade.ipynb.

    figures/pareto_J_vs_vf_blade_centrifugal.pdf     creep compliance against the IN625 volume fraction,
                                                     uniform blends and graded designs
    figures/pareto_J_vs_mass_blade_centrifugal.pdf   creep compliance against the blade mass, uniform blends
                                                     and the graded designs bounded in IN625 content, in mass
                                                     and in moment of inertia

Reads homogeneous_blends.csv, graded_vf.csv and graded_mass.csv in results/6_blade_centrifugal/
(written by evaluate_homogeneous_blends.py and summarize.py).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "results" / "6_blade_centrifugal"
FIGURES = ROOT / "figures"

homo = pd.read_csv(RUN_DIR / "homogeneous_blends.csv").sort_values("V_star")
graded = pd.read_csv(RUN_DIR / "graded_vf.csv").sort_values("V_star")

# At vf = 0 the design is pure SS316, so both curves share that endpoint. The constraint is an upper
# bound on the IN625 fraction, so at vf = 1 the graded design need not be pure IN625 and no point is added.
cga_vf_ext = np.concatenate(([homo["V_star"].iloc[0]], graded["V_star"]))
cga_J_ext = np.concatenate(([homo["J"].iloc[0]], graded["J"]))

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(homo["V_star"], homo["J"], "-o", label="Homogeneous Material", zorder=3)
ax.plot(cga_vf_ext, cga_J_ext, "-s", label="Continuously Graded Alloy Designs", zorder=2)
ax.set_xlabel("Volume fraction of IN625 (upper bound $V_*$ for graded designs)")
ax.set_ylabel("Creep compliance [mJ]")
ax.set_title("Homogeneous and graded alloy blades under centrifugal loading")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "pareto_J_vs_vf_blade_centrifugal.pdf")
plt.close(fig)
print(f"written to {FIGURES / 'pareto_J_vs_vf_blade_centrifugal.pdf'}")

# Creep compliance against mass. The mass is linear in the IN625 content (m = m_SS316 + V (m_IN625 - m_SS316)),
# so the V* sweep is at the same time a sweep of the mass bound; the runs bounded in mass are overlaid.
mass_file = RUN_DIR / "graded_mass.csv"
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(homo["mass_g"], homo["J"], "-o", label="Homogeneous Material", zorder=3)
ax.plot(np.concatenate(([homo["mass_g"].iloc[0]], graded["mass_g"])), cga_J_ext, "-s",
        label="Graded, IN625 content bounded", zorder=2)
if mass_file.exists():
    runs = pd.read_csv(mass_file)
    only_mass = runs[runs["inertia_bound_fraction"].isna()]
    with_inertia = runs[runs["inertia_bound_fraction"].notna()]
    ax.plot(only_mass["mass_g"], only_mass["J"], "x", color="k", ms=9, mew=1.8, ls="none",
            label="Graded, mass bounded", zorder=4)
    ax.plot(with_inertia["mass_g"], with_inertia["J"], "^", color="C3", ms=8, ls="none",
            label="Graded, moment of inertia bounded", zorder=4)
ax.set_xlabel("Blade mass [g]")
ax.set_ylabel("Creep compliance [mJ]")
ax.set_title("Creep compliance against mass under centrifugal loading")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "pareto_J_vs_mass_blade_centrifugal.pdf")
plt.close(fig)
print(f"written to {FIGURES / 'pareto_J_vs_mass_blade_centrifugal.pdf'}")

# Where the IN625 goes along the span, under centrifugal loading and under the tip traction of
# experiments/5_blade (design_distribution.csv, written by summarize.py).
dist_file = RUN_DIR / "design_distribution.csv"
if dist_file.exists():
    dist = pd.read_csv(dist_file).set_index(["study", "run"])
    span_cols = [c for c in dist.columns if c.startswith("span_")]
    centres = [np.mean([float(v) for v in c[len("span_"):-len("mm")].split("_")]) for c in span_cols]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    panels = (("tip_traction", "Tip traction"), ("centrifugal", "Centrifugal load"))
    for ax, (study, title) in zip(axes, panels):
        for run, label, style in (("data_creep_10", "$V_* = 0.1$", "-o"), ("data_creep_30", "$V_* = 0.3$", "-s"),
                                  ("data_creep_90", "bound inactive", "-^")):
            if (study, run) in dist.index:
                ax.plot(centres, dist.loc[(study, run), span_cols].to_numpy(dtype=float), style, ms=4, label=label)
        ax.set_title(title)
        ax.set_xlabel("Distance from the root [mm]")
        ax.set_xlim(0, 50)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, ls=":", alpha=0.5)
    axes[0].set_ylabel("Mean IN625 fraction of the section")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "blade_in625_spanwise.pdf")
    plt.close(fig)
    print(f"written to {FIGURES / 'blade_in625_spanwise.pdf'}")
