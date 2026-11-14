#!/bin/bash
#SBATCH --job-name=ds-primary-attn
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/3274075/DepthSplat_view_selection/logs/primary-attn-%A_%a.log

set -euo pipefail

module load cuda/12.4

PROJECT=/home/3274075/DepthSplat_view_selection
BASELINE=/home/3274075/DepthSplat_baseline
CODE="$PROJECT/code_primary_attention"
PYTHON="$BASELINE/envs/depthsplat/bin/python"
MANIFEST="${1:?Pass the selection manifest path as the first argument}"

export TORCH_HOME="$BASELINE/cache/torch"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$CODE"

# Require the tracked source to match its recorded commit.
git diff --exit-code
git diff --cached --exit-code

mapfile -t SCENE_INFO < <(
    "$PYTHON" - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    entries = json.load(file)
entry = entries[int(sys.argv[2])]
print(entry["scene"])
print(entry["selection_plan"])
print(entry["expected_frames"])
PY
)

if [ "${#SCENE_INFO[@]}" -ne 3 ]; then
    echo "Failed to read scene from manifest." >&2
    exit 1
fi

SCENE="${SCENE_INFO[0]}"
PLAN="${SCENE_INFO[1]}"
EXPECTED_FRAMES="${SCENE_INFO[2]}"
RUN_ROOT="$PROJECT/outputs/primary_attention_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/provenance"

cp "$MANIFEST" "$RUN_ROOT/provenance/manifest.json"
cp "$PLAN" "$RUN_ROOT/provenance/input_selection.json"
cp jobs/compare_primary_attention.sbatch "$RUN_ROOT/provenance/"
cp "$BASELINE/outputs/baseline19_749444/resolved_config.yaml" \
   "$RUN_ROOT/provenance/baseline_resolved_config.yaml"

git rev-parse HEAD > "$RUN_ROOT/provenance/git_commit.txt"
git archive HEAD | gzip > "$RUN_ROOT/provenance/source.tar.gz"
nvidia-smi > "$RUN_ROOT/provenance/nvidia-smi.txt"

JOB_START_SECONDS=$SECONDS
trap 'job_exit_code=$?; printf "{\"shell_job_wall_seconds\": %s, \"exit_code\": %s}\n" "$((SECONDS-JOB_START_SECONDS))" "$job_exit_code" > "$RUN_ROOT/provenance/shell_job_wall.json"' EXIT

"$PYTHON" -c \
  'import torch; print("Torch:", torch.__version__); assert torch.cuda.is_available(), "No CUDA GPU"'

"$PYTHON" tools/test_selection_protocol.py
"$PYTHON" tools/test_first_loop.py
"$PYTHON" tools/test_primary_support.py

ARGS=(
    --config "$BASELINE/outputs/baseline19_749444/resolved_config.yaml"
    --checkpoint "$BASELINE/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth"
    --dataset-root "$BASELINE/data/dl3dv_subset20"
    --scene "$SCENE"
    --selection-plan "$PLAN"
    --expected-frames "$EXPECTED_FRAMES"
    --num-primary 12
    --num-targets 8
    --support-offset 0.25
    --render-chunk 8
    --warmups 3
)

printf '%q ' "$PYTHON" tools/run_one_scene_comparison.py \
    "${ARGS[@]}" --out "$RUN_ROOT/comparison" --save-gaussians \
    > "$RUN_ROOT/provenance/command.sh"

"$PYTHON" tools/run_one_scene_comparison.py \
    "${ARGS[@]}" --out "$RUN_ROOT/plan_check" --plan-only

"$PYTHON" tools/run_one_scene_comparison.py \
    "${ARGS[@]}" --out "$RUN_ROOT/comparison" --save-gaussians

echo "COMPLETE: $SCENE"
echo "Results: $RUN_ROOT/comparison"