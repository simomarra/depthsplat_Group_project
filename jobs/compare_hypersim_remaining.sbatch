#!/bin/bash
#SBATCH --job-name=ds-hypersim-9
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=01:00:00
#SBATCH --array=1-9%2
#SBATCH --output=/home/3274075/DepthSplat_view_selection/logs/hypersim-%A_%a.log

set -euo pipefail
module load miniconda3 cuda/12.4
source /software/miniconda3/etc/profile.d/conda.sh
conda activate /home/3274075/DepthSplat_baseline/envs/depthsplat
export TORCH_HOME=/home/3274075/DepthSplat_baseline/cache/torch
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

PROJECT=/home/3274075/DepthSplat_view_selection
BASELINE=/home/3274075/DepthSplat_baseline
cd "$PROJECT/code_primary_attention"
SCENE=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path('plans/hypersim_view_plan_seed2026.json').read_text())
remaining = [scene for scene in plan['scene_selection']['scene_ids']
             if scene != 'ai_007_005']
index = int(sys.argv[1]) - 1
if len(remaining) != 9 or index not in range(9):
    raise ValueError('Unexpected frozen scene plan or Slurm array index')
print(remaining[index])
PY
)
RUN_ROOT="$PROJECT/outputs/hypersim_${SCENE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN_ROOT/provenance"
cp jobs/compare_hypersim_remaining.sbatch "$RUN_ROOT/provenance/"
cp "$PROJECT/hypersim_config/resolved_config.yaml" "$RUN_ROOT/provenance/"
git rev-parse HEAD > "$RUN_ROOT/provenance/repository_commit.txt"
sha256sum plans/hypersim_view_plan_seed2026.json \
  "$PROJECT/hypersim_prepared/$SCENE/preprocessing.json" \
  "$BASELINE/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth" \
  > "$RUN_ROOT/provenance/input_sha256.txt"
python -c 'import torch; assert torch.cuda.is_available(); print("GPU:",torch.cuda.get_device_name())'

python tools/run_one_scene_comparison.py \
  --config "$PROJECT/hypersim_config/resolved_config.yaml" \
  --checkpoint "$BASELINE/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth" \
  --hypersim-plan plans/hypersim_view_plan_seed2026.json \
  --prepared-root "$PROJECT/hypersim_prepared" \
  --scene "$SCENE" \
  --out "$RUN_ROOT/comparison" \
  --render-chunk 2 \
  --warmups 1 \
  --save-gaussians

printf 'COMPARISON COMPLETE: %s\n' "$RUN_ROOT/comparison"
