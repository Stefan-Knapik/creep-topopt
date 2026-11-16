#!/usr/bin/env bash
# Copy the figures used by the paper from figures/ into the paper source folder.
#
#   visualization/sync_paper_figures.sh /path/to/creep-top-opt-paper/paper
#
# Run ./run_all.sh figures first. method_domain.pdf is drawn by hand and is copied as it is.
# BC_and_blade_mat_design.pdf is written by visualization/plot_blade_designs.py.
# creep_optimization_flowchart_cropped.pdf needs pdfcrop: run_all.sh skips the crop if it is
# not installed.
set -euo pipefail

dest=${1:?usage: sync_paper_figures.sh <paper directory>}
src="$(cd "$(dirname "$0")/.." && pwd)/figures"

figures=(
    method_domain.pdf
    creep_optimization_flowchart_cropped.pdf
    elastic_vs_creep.pdf
    pareto_vf_structures.pdf
    thermal_creep_shared_cbar.pdf
    thermal_creep_tdep.pdf
    load_magnitude.pdf
    BC_and_blade_mat_design.pdf
    pareto_J_vs_vf_blade.pdf
    A1_creep_validation.pdf
    A2_scheme_convergence.pdf
    A3_sensitivity_verify.pdf
    A4_mesh_convergence.pdf
    A5_solver_tolerance_grey.pdf
)

for f in "${figures[@]}"; do
    cp -v "$src/$f" "$dest/$f"
done
