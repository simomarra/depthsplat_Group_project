#!/bin/bash
#SBATCH --job-name=ds-aggregation
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/3274075/DepthSplat_view_selection/logs/aggregation-%A_%a.log

set -euo pipefail
module load cuda/12.4

PROJECT=/home/3274075/DepthSplat_view_selection
BASELINE=/home/3274075/DepthSplat_baseline
CODE="$PROJECT/code_validity_matching"
PYTHON="$BASELINE/envs/depthsplat/bin/python"
MANIFEST="${1:?Pass the frozen selection manifest as the first argument}"

export TORCH_HOME="$BASELINE/cache/torch"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
cd "$CODE"
git diff --exit-code HEAD

mapfile -t SCENE_INFO < <(
    "$PYTHON" - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

entry = json.loads(Path(sys.argv[1]).read_text())[int(sys.argv[2])]
plan = Path(entry["selection_plan"])
assert json.loads(plan.read_text())["scene"] == entry["scene"]
print(entry["scene"])
print(plan)
PY
)
if [ "${#SCENE_INFO[@]}" -ne 2 ]; then
    echo "Failed to validate manifest entry." >&2
    exit 1
fi
SCENE="${SCENE_INFO[0]}"
PLAN="${SCENE_INFO[1]}"
RUN_ROOT="$PROJECT/outputs/weighting_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/provenance"
JOB_START_SECONDS=$SECONDS
trap 'job_exit_code=$?; printf "{\"shell_job_wall_seconds\": %s, \"exit_code\": %s}\n" "$((SECONDS-JOB_START_SECONDS))" "$job_exit_code" > "$RUN_ROOT/provenance/shell_job_wall.json"' EXIT

cp "$MANIFEST" "$RUN_ROOT/provenance/manifest.json"
cp "$PLAN" "$RUN_ROOT/provenance/input_selection.json"
cp jobs/compare_aggregation.sbatch "$RUN_ROOT/provenance/"
git rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
git archive HEAD | gzip > "$RUN_ROOT/provenance/source.tar.gz"
nvidia-smi > "$RUN_ROOT/provenance/nvidia-smi.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/provenance/pip_freeze.txt"

ARGS=(
    --config "$BASELINE/outputs/baseline19_749444/resolved_config.yaml"
    --checkpoint "$BASELINE/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth"
    --dataset-root "$BASELINE/data/dl3dv_subset20"
    --selection-plan "$PLAN"
    --reference-root "$BASELINE/code"
    --out "$RUN_ROOT/comparison"
    --warmups 3
    --render-chunk 8
)
printf '%q ' "$PYTHON" tools/run_aggregation_comparison.py "${ARGS[@]}" \
    > "$RUN_ROOT/provenance/command.sh"
"$PYTHON" tools/run_aggregation_comparison.py "${ARGS[@]}"
echo "RESULTS: $RUN_ROOT/comparison"
