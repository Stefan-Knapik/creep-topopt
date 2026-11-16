#!/usr/bin/env bash
# Runs a list of jobs of experiments/5_blade on the GPUs, a bounded number at a time on each.
#
#   ./run_exploration.sh cases.txt            # one job per line
#   ./run_exploration.sh - <<'JOBS'           # or from standard input
#   optimize_graded_blade.py --steps 2 --root-stress 80 --hours 30 --vf 0.3
#   optimize_graded_blade.py --steps 2 --root-stress 80 --shroud 0.3 --hours 30 --vf 0.3 0.6
#   screen_forward.py --root-stress 60 80 --shroud 0 0.3 --hours 100 --check-steps 40
#   JOBS
#
# A line is a script of this folder with its arguments ('#' lines and blank lines are skipped). Jobs are
# dealt to the GPUs in turn and each GPU runs PER_GPU of them at a time.
#
#   GPUS="0 1"        GPUs to use
#   PER_GPU=12        concurrent jobs per GPU (25k mesh). On the 50k mesh: 3, with XLA_PYTHON_CLIENT_MEM_FRACTION=0.32
#                     (a run needs one 16 GiB allocation; uncapped it grows to 42 GiB and only 2 fit on 96 GB)
#   THREADS=4         MKL / OpenMP threads per job
#   TOGA_MAX_ITER=15  MMA iterations of optimize_graded_blade.py (exported if set; the script's default is 50)
#   SKIP_DONE=1       skip optimize_graded_blade.py lines when LOG (below) of the same line ends with success
#   NAME=explore      prefix of the log files logs/5_<NAME>_<line number>.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
E=experiments/5_blade
PYTHON="${PYTHON:-$HOME/miniconda3/envs/creep-topopt/bin/python}"
PER_GPU="${PER_GPU:-12}"
THREADS="${THREADS:-4}"
NAME="${NAME:-explore}"
read -ra GPU_LIST <<< "${GPUS:-0 1}"
mkdir -p logs

export MKL_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS=1
export TOGA_PARDISO="${TOGA_PARDISO:-cached}"
[ -n "${TOGA_MAX_ITER:-}" ] && export TOGA_MAX_ITER

INPUT="${1:--}"; [ "$INPUT" = "-" ] && INPUT=/dev/stdin
jobs_file="$(mktemp)"
i=0
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    [ -z "${line// }" ] && continue
    i=$((i + 1))
    gpu=${GPU_LIST[$(( (i - 1) % ${#GPU_LIST[@]} ))]}
    log="logs/5_${NAME}_$(printf '%03d' "$i").log"
    if [ -n "${SKIP_DONE:-}" ] && [ -f "$log" ] && grep -q "^JOB OK: $line\$" "$log"; then
        continue
    fi
    echo "CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $E/$line > $log 2>&1 && echo 'JOB OK: $line' >> $log || { echo 'JOB FAILED: $line' >> $log; echo FAILED $log: $line; }" >> "$jobs_file"
done < "$INPUT"

echo "$(wc -l < "$jobs_file") jobs, $PER_GPU at a time on each of GPUs ${GPU_LIST[*]}, $THREADS threads each; logs/5_${NAME}_*.log"
for gpu in "${GPU_LIST[@]}"; do
    grep "^CUDA_VISIBLE_DEVICES=$gpu " "$jobs_file" | xargs -P "$PER_GPU" -d '\n' -I{} bash -c '{}' &
done
wait
rm -f "$jobs_file"
echo "finished: $(grep -l '^JOB OK' logs/5_${NAME}_*.log 2>/dev/null | wc -l) ok, $(grep -l '^JOB FAILED' logs/5_${NAME}_*.log 2>/dev/null | wc -l) failed"
