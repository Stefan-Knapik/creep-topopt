"""Sec. 4.5: creep compliance against the IN625 volume fraction, uniform blends and graded designs
(figures/pareto_J_vs_vf_blade.pdf and figures/readme/pareto_J_vs_vf_blade.png).

Reads <case>/homogeneous_blends.csv (evaluate_homogeneous_blends.py) and the last row of every
<case>/V<bound>/scalars.csv (optimize_graded_blade.py).

    python plot_pareto.py results/5_blade/deploy/<case>
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", default="pareto_J_vs_vf_blade")
    args = parser.parse_args()

    homo = sorted((float(r['V_star']), float(r['J'])) for r in csv.DictReader(open(args.case / "homogeneous_blends.csv")))
    graded = []
    for run in sorted(args.case.glob("V*")):
        if (run / "scalars.csv").exists():
            last = list(csv.DictReader(open(run / "scalars.csv")))[-1]
            graded.append((json.load(open(run / "case.json"))['vf'], float(last['J']), float(last['IN625_fraction'])))
    graded.sort()
    homo_vf, homo_J = np.array(homo).T
    vf, J, used = np.array(graded).T

    # At V* = 0 the design is pure SS316, so both curves share that endpoint. The constraint is an upper
    # bound on the IN625 fraction, so at V* = 1 the graded design need not be pure IN625 and no point is added.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(homo_vf, homo_J, "-o", label="Homogeneous Material", zorder=3)
    ax.plot(np.concatenate(([homo_vf[0]], vf)), np.concatenate(([homo_J[0]], J)), "-s",
            label="Continuously Graded Alloy Designs", zorder=2)
    ax.set_xlabel("Volume fraction of IN625 (upper bound $V_*$ for graded designs)")
    ax.set_ylabel("Creep compliance [mJ]")
    ax.set_title("Pareto frontiers of homogeneous and graded alloy blades")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{args.out}.pdf")
    plt.close(fig)

    # Smaller version for the README, shown at about half width, so the text is larger.
    (FIGURES / "readme").mkdir(exist_ok=True)
    with plt.rc_context({"font.size": 13}):
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        ax.plot(homo_vf, homo_J, "-o", ms=5, label="Uniform blend", zorder=3)
        ax.plot(np.concatenate(([homo_vf[0]], vf)), np.concatenate(([homo_J[0]], J)), "-s", ms=5,
                label="Graded (optimized)", zorder=2)
        ax.set_xlabel("Volume fraction of IN625")
        ax.set_ylabel("Creep compliance [mJ]")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "readme" / f"{args.out}.png", dpi=200)
        plt.close(fig)
    print(f"written to {FIGURES / (args.out + '.pdf')}")
    print(f"{'V*':>4s} {'graded J':>10s} {'IN625 used':>10s} {'homogeneous J':>14s}")
    for v, j, u in graded:
        print(f"{v:4.1f} {j:10.5g} {u:10.4f} {dict(homo).get(round(v, 1), float('nan')):14.5g}")


if __name__ == "__main__":
    main()
