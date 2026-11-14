#!/bin/bash
#SBATCH --job-name=ds-export3d
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/export3d-%j.out
#SBATCH --error=logs/export3d-%j.err

set -e
module load miniconda3 cuda/12.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate depthsplat
cd "${DEPTHSPLAT_REPO:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1

# First validate unchanged baseline behaviour after installing the patch.
# Each variant receives exactly the same primary and held-out target views.
for variant in baseline support; do
  if [ "$variant" = baseline ]; then
    selection=assets/dl3dv_subset_6p_8t.json
    primary_count=0
    context_count=6
    sort_context=true
  else
    selection=assets/dl3dv_subset_6p_12s_8t.json
    primary_count=6
    context_count=18
    sort_context=false
  fi
  out="outputs/${variant}_gap_${SLURM_JOB_ID}"
  mkdir -p "$out"
  cp "$selection" "$out/selection.json"
  cp assets/dl3dv_subset_support_manifest.json "$out/support_manifest.json"
  git rev-parse HEAD > "$out/base_commit.txt"
  echo "Running $variant: context_count=$context_count, primary_count=$primary_count"

  python -u -m src.main +experiment=dl3dv \
    mode=test wandb.mode=disabled use_plugins=true \
    dataset.roots='[datasets/dl3dv_960p_test_subset]' \
    dataset.ori_image_shape='[540,960]' dataset.image_shape='[256,448]' \
    dataset/view_sampler=evaluation \
    dataset.sort_context_index="$sort_context" \
    dataset.view_sampler.num_context_views="$context_count" \
    dataset.view_sampler.index_path="$selection" \
    model.encoder.num_primary_views="$primary_count" \
    model.encoder.num_scales=2 model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 model.encoder.monodepth_vit_type=vitb \
    checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
    data_loader.test.num_workers=2 test.compute_scores=true \
    test.render_chunk_size=4 test.save_image=true test.save_gt_image=true \
    test.save_input_images=true test.save_gaussian=true output_dir="$out"

  if [ "$variant" = baseline ]; then
    python - "$out" <<'PY'
import json
import sys
from pathlib import Path

scores = json.loads((Path(sys.argv[1]) / "metrics/scores_all_avg.json").read_text())
reference = {"psnr": 24.542296409606934, "ssim": 0.8174033761024475,
             "lpips": 0.12623437121510506}
for key, value in reference.items():
    if abs(scores[key] - value) > 1e-3:
        raise RuntimeError(f"Baseline drift for {key}: {scores[key]} vs {value}; investigate before support test.")
print("Baseline matches job 763386 within 0.001 for all three metrics.")
PY
  fi
done

python - "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

job = sys.argv[1]
results = {
    name: json.loads(Path(f"outputs/{name}_gap_{job}/metrics/scores_all_avg.json").read_text())
    for name in ("baseline", "support")
}
print("\nMetric       baseline      support     difference")
for metric in ("psnr", "ssim", "lpips"):
    a, b = results["baseline"][metric], results["support"][metric]
    print(f"{metric:8} {a:12.6f} {b:12.6f} {b-a:+12.6f}")
PY
