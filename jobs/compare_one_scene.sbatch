#!/bin/bash
#SBATCH --job-name=ds-one-scene
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/3274075/DepthSplat_view_selection/logs/compare-%j.log

set -euo pipefail
module load miniconda3 cuda/12.4
source /software/miniconda3/etc/profile.d/conda.sh
conda activate /home/3274075/DepthSplat_baseline/envs/depthsplat
export TORCH_HOME=/home/3274075/DepthSplat_baseline/cache/torch
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

PROJECT=/home/3274075/DepthSplat_view_selection
BASELINE=/home/3274075/DepthSplat_baseline
cd "$PROJECT/code_one_scene_v3"
RUN_ROOT="$PROJECT/outputs/compare_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT/provenance"
JOB_START_SECONDS=$SECONDS
trap 'job_exit_code=$?; printf "{\"shell_job_wall_seconds\": %s, \"exit_code\": %s}\n" "$((SECONDS-JOB_START_SECONDS))" "$job_exit_code" > "$RUN_ROOT/provenance/shell_job_wall.json"' EXIT
cp jobs/compare_one_scene.sbatch "$RUN_ROOT/provenance/"
cp "$BASELINE/outputs/baseline19_749444/resolved_config.yaml" "$RUN_ROOT/provenance/baseline_resolved_config.yaml"
if [ -f "$PROJECT/depthsplat_hpc_base.zip" ]; then
    unzip -z "$PROJECT/depthsplat_hpc_base.zip" > "$RUN_ROOT/provenance/base_archive_commit.txt"
fi
nvidia-smi > "$RUN_ROOT/provenance/nvidia-smi.txt"
python -m pip freeze > "$RUN_ROOT/provenance/pip_freeze.txt"
# Keep the exact source used, including uncommitted overlay changes.
tar -czf "$RUN_ROOT/provenance/source.tar.gz" --exclude='__pycache__' --exclude='*.pyc' --exclude='*.before*' src tools config jobs LICENSE
python -c 'import torch; print("Torch:", torch.__version__); assert torch.cuda.is_available(), "No CUDA GPU"'
python tools/test_selection_protocol.py
python tools/test_primary_support.py

ARGS=(
    --config "$BASELINE/outputs/baseline19_749444/resolved_config.yaml"
    --checkpoint "$BASELINE/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth"
    --dataset-root "$BASELINE/data/dl3dv_subset20"
    --scene 15ff83e2531668d27c92091c97d31401ce323e24ee7c844cb32d5109ab9335f7
    --expected-frames 348
    --num-primary 6
    --num-targets 8
    --support-offset 0.25
    --render-chunk 8
    --warmups 3
)
printf '%q ' python tools/run_one_scene_comparison.py "${ARGS[@]}" --out "$RUN_ROOT/comparison" --save-gaussians > "$RUN_ROOT/provenance/command.sh"
printf '\n' >> "$RUN_ROOT/provenance/command.sh"
# Fail before constructing the model if converted scene coverage is incomplete.
python tools/run_one_scene_comparison.py "${ARGS[@]}" --out "$RUN_ROOT/plan" --plan-only
python tools/run_one_scene_comparison.py "${ARGS[@]}" --out "$RUN_ROOT/comparison" --save-gaussians
printf 'COMPARISON COMPLETE: %s\n' "$RUN_ROOT/comparison"
