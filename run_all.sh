#!/usr/bin/env bash
# Reproduce all results and figures of the paper.
#
#   ./run_all.sh                         # everything, in dependency order, on GPU 0
#   GPU=2 ./run_all.sh 5_blade figures   # selected steps on GPU 2
#   GPUS="1 2 3 4 5" ./run_all.sh        # everything, independent steps in parallel on GPUs 1-5
#
# Steps: 2_pareto_vf 1_creep_elastic 3_thermal_creep 4_load_magnitude 5_blade appendix figures
# (1_creep_elastic reuses the V*=0.5 design from 2_pareto_vf, so run that first.)
# Each script's output goes to logs/<step>.log; wall time, peak host RSS and peak GPU memory (whole
# device, sampled each second into logs/<step>_gpu.csv) are appended to results/timings.csv.
# TOGA_MAX_ITER=2 caps the MMA iterations of the optimization scripts (quick smoke test).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -n "${GPUS:-}" ]; then
    # Parallel mode: one stream per GPU (cycling if fewer GPUs than streams), then figures.
    # PARDISO factorizes on the CPU, so the cores are split between streams to avoid oversubscription.
    STREAMS=("5_blade" "2_pareto_vf 1_creep_elastic" "3_thermal_creep" "4_load_magnitude" "appendix")
    read -ra GPU_LIST <<< "$GPUS"
    THREADS=$(( $(nproc) / ${#STREAMS[@]} ))
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
    pids=()
    for i in "${!STREAMS[@]}"; do
        # shellcheck disable=SC2086
        GPU=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]} GPUS= "$ROOT/run_all.sh" ${STREAMS[$i]} & pids+=($!)
    done
    code=0
    for p in "${pids[@]}"; do wait "$p" || code=1; done
    GPU=${GPU_LIST[0]} GPUS= "$ROOT/run_all.sh" figures || code=1
    exit "$code"
fi

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
# numpy's OpenBLAS threads and MKL's OpenMP threads (Pardiso) oversubscribe the CPU,
# and MKL threads spin for 200 ms after each call by default.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
# Pardiso with cached symbolic analysis (toga/fem/sparse_linear_solvers.py); TOGA_PARDISO=none for pypardiso.spsolve.
export TOGA_PARDISO="${TOGA_PARDISO:-cached}"
PYTHON="${PYTHON:-python}"
mkdir -p logs results
TIMINGS_HEADER="step,command,gpu,start,end,seconds,exit_code,peak_rss_mb,peak_gpu_mb"
if [ -f results/timings.csv ] && [ "$(head -n1 results/timings.csv)" != "$TIMINGS_HEADER" ]; then
    mv results/timings.csv "results/timings_old_$(date +%Y%m%d_%H%M%S).csv"  # older format, keep it aside
fi
[ -f results/timings.csv ] || echo "$TIMINGS_HEADER" > results/timings.csv

run() {  # run <step> <command...>: log output, wall time, peak host RSS and peak GPU memory
    local step=$1; shift
    local start end code=0 rss_file gpu_file smi_pid="" peak_rss="" peak_gpu=""
    rss_file=$(mktemp)
    gpu_file="logs/${step}_gpu.csv"  # GPU memory.used (MiB, whole device) sampled once per second
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=timestamp,memory.used \
            --format=csv,noheader,nounits -lms 1000 >> "$gpu_file" 2> /dev/null & smi_pid=$!
    fi
    start=$(date +%s)
    echo "[$(date '+%F %T')] $step: $*" | tee -a "logs/$step.log"
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -f "%M" -o "$rss_file" "$@" >> "logs/$step.log" 2>&1 || code=$?
        peak_rss=$(( $(tail -n1 "$rss_file" | grep -Eo '^[0-9]+$' || echo 0) / 1024 ))
    else
        "$@" >> "logs/$step.log" 2>&1 || code=$?
    fi
    end=$(date +%s)
    if [ -n "$smi_pid" ]; then
        kill "$smi_pid" 2> /dev/null; wait "$smi_pid" 2> /dev/null || true
        peak_gpu=$(awk -F', *' -v s="$(date -d @"$start" '+%Y/%m/%d %H:%M:%S')" \
            '$1 >= s && $2+0 > m {m = $2+0} END {print m+0}' "$gpu_file")
    fi
    rm -f "$rss_file"
    echo "$step,\"$*\",$CUDA_VISIBLE_DEVICES,$(date -d @"$start" '+%F %T'),$(date -d @"$end" '+%F %T'),$((end - start)),$code,$peak_rss,$peak_gpu" >> results/timings.csv
    echo "  done in $((end - start)) s, peak RSS ${peak_rss:-?} MB, peak GPU ${peak_gpu:-?} MB" | tee -a "logs/$step.log"
    if [ "$code" -ne 0 ]; then echo "  failed (exit $code), see logs/$step.log"; return "$code"; fi
}

step_2_pareto_vf()      { run 2_pareto_vf "$PYTHON" -u experiments/2_pareto_vf/optimize_volume_fraction_sweep.py; }
step_1_creep_elastic()  {
    run 1_creep_elastic "$PYTHON" -u experiments/1_creep_elastic/optimize_elastic_baseline.py
    run 1_creep_elastic "$PYTHON" -u experiments/1_creep_elastic/evaluate_designs_under_creep.py
    run 1_creep_elastic "$PYTHON" -u experiments/1_creep_elastic/reconstruct_stress_strain_fields.py
}
step_3_thermal_creep()  { run 3_thermal_creep "$PYTHON" -u experiments/3_thermal_creep/optimize_thermal_boundary_conditions.py; }
step_4_load_magnitude() { run 4_load_magnitude "$PYTHON" -u experiments/4_load_magnitude/optimize_load_magnitude_sweep.py; }
step_5_blade() {
    run 5_blade "$PYTHON" -u experiments/5_blade/optimize_graded_blade.py
    run 5_blade "$PYTHON" -u experiments/5_blade/evaluate_homogeneous_blends.py
}
step_appendix() {
    local A=experiments/appendix
    run appendix "$PYTHON" -u $A/A1_abaqus_validation/run_all.py
    run appendix "$PYTHON" -u $A/A2_explicit_implicit/run_all.py
    run appendix "$PYTHON" -u $A/A3_sensitivity/run_all.py
    rm -f results/appendix/A4_mesh_convergence/mesh_convergence.csv  # run_all.py resumes from this file
    TOGA_PLATFORM=gpu run appendix "$PYTHON" -u $A/A4_mesh_convergence/run_all.py
    MKL_NUM_THREADS=1 run appendix "$PYTHON" -u $A/A5_solver_tolerance/run_simulation.py --rtol-check --design solid grey --fresh
}
step_figures() {
    for nb in visualization/*.ipynb; do
        run figures "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace "$nb"
    done
    for f in visualization/appendix/plot_A*.py; do run figures "$PYTHON" "$f"; done
    run figures "$PYTHON" visualization/flowchart.py
    run figures "$PYTHON" visualization/animate_design.py
    if command -v pdftoppm > /dev/null; then  # README copies of two paper figures, at 2x resolution
        for f in elastic_vs_creep BC_and_blade_mat_design; do
            run figures pdftoppm -png -r 220 -singlefile "figures/$f.pdf" "figures/readme/$f"
        done
    fi
    if command -v pdfcrop > /dev/null; then
        run figures pdfcrop figures/creep_optimization_flowchart.pdf figures/creep_optimization_flowchart_cropped.pdf
    fi
}

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(2_pareto_vf 1_creep_elastic 3_thermal_creep 4_load_magnitude 5_blade appendix figures)
for s in "${STEPS[@]}"; do
    declare -F "step_$s" > /dev/null || { echo "unknown step: $s"; exit 1; }
    "step_$s"
done
