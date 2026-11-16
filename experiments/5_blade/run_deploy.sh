#!/usr/bin/env bash
# Sec. 4.5 at the paper settings: 50,336 HEX8, implicit, 10 time steps, 50 MMA iterations.
# IN625 bounds 0.1 ... 0.9, the uniform blends, then the tables and the Pareto figure.
#
#   ROOT_STRESS=100 SHROUD=0 HOURS=100 RADIUS_MM=1.4 ./run_deploy.sh
#
#   VFS="0.1 ... 0.9"   IN625 bounds to run (runs that already have a DONE marker are skipped)
#   EXTRA=""            further arguments of the case, e.g. "--alpha-mode secant"
#   STUDY=deploy        results/5_blade/<STUDY>/
#   GPUS="0 1" PER_GPU=3 THREADS=12   three runs fit on a 96 GB card with the JAX memory capped at 32 % each
#                                     (XLA_PYTHON_CLIENT_MEM_FRACTION); uncapped, a run grows to 42 GiB and only two fit
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
E=experiments/5_blade
PYTHON="${PYTHON:-$HOME/miniconda3/envs/creep-topopt/bin/python}"
ROOT_STRESS="${ROOT_STRESS:-100}"; SHROUD="${SHROUD:-0}"; HOURS="${HOURS:-100}"; RADIUS_MM="${RADIUS_MM:-1.4}"
VFS="${VFS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
STUDY="${STUDY:-deploy}"
EXTRA="${EXTRA:-}"
CASE_ARGS="--fine --study $STUDY --root-stress $ROOT_STRESS --shroud $SHROUD --hours $HOURS --radius-mm $RADIUS_MM $EXTRA"
export GPUS="${GPUS:-0 1}" PER_GPU="${PER_GPU:-3}" THREADS="${THREADS:-12}" NAME="${NAME:-deploy}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.32}"
export TOGA_MAX_ITER="${TOGA_MAX_ITER:-50}"

CASE_DIR="$("$PYTHON" - <<PY
import argparse, sys
sys.path.insert(0, "$E")
from _common import bp, case_kwargs
from optimize_graded_blade import add_case_arguments
p = argparse.ArgumentParser(); add_case_arguments(p)
a = p.parse_args("$CASE_ARGS".split()); k = case_kwargs(a)
print(bp.RESULTS / a.study / bp.case_name(a.root_stress, a.shroud, a.hours, radius_mm=k['radius_mm'], mesh=bp.mesh_name(k['elem_size'], a.fine),
      num_time_steps=k['num_time_steps'], alpha_mode=a.alpha_mode, expansion=k['expansion'], cte_mismatch_scale=a.cte_mismatch_scale))
PY
)" || exit 1
echo "case folder: $CASE_DIR"

jobs="$(mktemp)"
[ -f "$CASE_DIR/homogeneous_blends.csv" ] || echo "evaluate_homogeneous_blends.py $CASE_ARGS" >> "$jobs"
for vf in $VFS; do
    [ -f "$CASE_DIR/V$(printf '%02.0f' "$(echo "$vf * 100" | bc)")/DONE" ] || echo "optimize_graded_blade.py $CASE_ARGS --vf $vf" >> "$jobs"
done
"$E/run_exploration.sh" "$jobs"
rm -f "$jobs"

"$PYTHON" "$E/mass_inertia.py" "$CASE_DIR"
"$PYTHON" "$E/plot_pareto.py" "$CASE_DIR"
"$PYTHON" "$E/summarize.py" --study "$STUDY"
