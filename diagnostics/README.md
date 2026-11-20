# Fixed-primary secondary-view diagnostics

Question: which outputs of the secondary-view depth pass help or harm Gaussian reconstruction when the six Gaussian-generating photographs stay fixed?

This suite consolidates the HPC experiments into one intervention runner and three analysis commands. It is based on `experiment/all-view-depth-attention`, commit `f5c372157edc494e85783bb0a7cd7b11c2be6b71`. The model, its reference encoder, `primary_support.py`, configurations and tracked selection utilities/tests already belong to that commit and must remain in the checkout. No model weight or architecture change is introduced here. The only change outside `diagnostics/` and `.gitignore` is the exact-input export in `tools/run_one_scene_comparison.py`.

## What is held fixed

Each method (FPS or trajectory/uniform) is a separate paired experiment. Within a pair, primary RGB, primary order, normalized intrinsics, camera-to-world poses, near/far bounds, preprocessing, checkpoint and held-out targets are identical. Only primaries generate Gaussians: 6 × 256 × 448 = 688,128. Baseline depth prediction uses the six primaries, with the two nearest other primaries as matching neighbors. The secondary pass processes those same primaries plus 12 distinct assigned secondary photographs; each primary uses its assigned two sources in multiview attention and cost-volume construction. Secondary rows use their own nearest neighbors. Both neighbor budgets are two.

**This branch uses all-view depth attention.** Supports also undergo depth prediction and can affect transformer representations. It is not a matching-only or computation-saving implementation, and the fixed Gaussian count does not imply fixed compute. Target RGB is used only for evaluation and plots; targets must be absent from the whole context pool.

## Conditions and exact model boundary

B = six-primary baseline depth-predictor output. S = secondary-pass output, gathered back into the same primary order. Features means **all three groups entering `encoder.feature_upsampler`**, not rendered RGB.

| Condition | CNN + mono + MV features | Final primary depth | Matching confidence |
|---|---|---|---|
| `baseline` | B | B | B |
| `secondary_depth_only` | B | S | B |
| `secondary_depth_confidence` | B | S | S |
| `secondary_features_only` | S | B | B |
| `secondary_features_depth` | S | S | B |
| `full_secondary` | S | S | S |

Supplementary conditions `secondary_mv_features_only`, `secondary_cnn_features_only`, `secondary_mono_features_only` swap just the named feature group; all other groups, depth and confidence stay B. For comparison with the historical feature-group ablations, `secondary_depth_mv_features`, `secondary_depth_cnn_features` and `secondary_depth_mono_features` instead keep depth S and confidence B. `--feature-ablations` includes both sets. The all-features-only and supplementary MV features-only conditions were missing from the archived experiments and are new here. Historical `hybrid_support_features` and `hybrid_support_mv` already used **secondary depth**, so they must not be relabelled as features-only.

`features_cnn_all_scales`, `features_mono_intermediate` and `features_mv` are feature pyramids with tensors `[B*P,C,h,w]`. They feed the ordinary feature upsampler. Final depth is `[B,P,H,W]`. Confidence is `max(match_probs[-1], dim=candidate)`, then nearest-neighbor upsampling to `[B*P,1,H,W]`: a peak over inverse-depth candidates, **not a verified probability of correctness**. The complete `match_probs` tensor is selected; the ordinary encoder computes the peak itself.

In `src/model/encoder/encoder_depthsplat.py`, the Gaussian regressor receives primary RGB, selected depth, selected confidence and upsampled features. The Gaussian head also receives selected confidence and features. The selected depth is used again in the Gaussian adapter to position Gaussians using primary cameras and predicted image-plane offsets. RGB initialization of spherical harmonics, when enabled, uses the unchanged primary images. Swapping predictor outputs and executing the ordinary encoder covers these paths together; replacing just a visualization depth would not.

Mixed-source conditions are **diagnostic interventions**: their confidence may describe a different depth prediction than the depth supplied to the Gaussian branch. They are not independently trained models. Caching predictor outputs isolates Gaussian-branch effects; timings from this runner are not full inference benchmarks.

## File catalogue and consolidation

| Entry point / files | Preserved functionality |
|---|---|
| `python -m diagnostics.infer` → retained `tools/run_one_scene_comparison.py` | Ordinary baseline/full-secondary inference for both methods; strict loading, timing, exact preprocessed NPZ export |
| `python -m diagnostics.run` | Consolidates `run_depth_hybrid.py`, `run_component_ablation.py`, `run_feature_ablation.py`; all six conditions, feature-group ablations, sink checks, rendering and raw per-target metrics |
| `conditions.py`, `io.py` | Explicit source table, shape/provenance checks, RGB/camera alignment, leakage prevention |
| `python -m diagnostics.reproject` + `reprojection.py` | Replaces `diagnose_support_depths.py`; per-primary/combined RGB reprojections, coverage, common-mask errors, identity check |
| `python -m diagnostics.inspect` + `feature_maps.py` | Replaces `inspect_matching_confidence.py` and `inspect_mv_features.py`; confidence maps/histograms/statistics and native-resolution MV feature comparisons before/after upsampling |
| `python -m diagnostics.plot_renderings` | GT, six rendered views with identical crops, common-scale RGB error maps, per-target metrics, aggregate tables |
| `make_config.py`, `plans/`, `jobs/` | Resolved architecture settings, two exact filename-based selection plans, generic Slurm launchers |
| `tests/` | CPU routing/alignment/projection tests and temporary synthetic plotting smoke test; no synthetic results are retained |
| `results/legacy/` | Small historical measured CSVs, provenance and representative figure; not new six-condition results |
| `VALIDATION.md` | Completed checks, HPC checks still required and limitations |

Existing tracked helpers and tests are retained unchanged. The two exported repositories' common helpers/tests were byte-identical. Calibration runners and launchers, first-loop-only added diagnostics, raw logs, data, checkpoints, large arrays and generated bulk figures are excluded. Their original HPC files remain untouched.

## Environment and exact inputs

Use the repository installation instructions and the existing working `depthsplat` environment, including its CUDA rasterizer, xformers, DINOv2/monocular caches and LPIPS weights. Recorded HPC environment: Python 3.10.22, PyTorch 2.4.0+cu124, CUDA 12.4, A100 MIG 4g.40gb. `requirements.txt` is the upstream dependency list, not an exact lockfile. Offline plots/tests require NumPy, Matplotlib and Pillow; inference additionally requires the full original model environment. No automatic download or environment repair is performed by the suite.

Checkpoint: `depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth`; SHA256 `02c7b19d8d038f71b512ef434d42b49698c3288d8ed65b748c04bd43d298f37a`. DL3DV converted dataset must contain `test/index.json`, full scene chunks and the original filenames/timestamps expected by the plans. Architecture: Base/vitb, two scales, 128 initial inverse-depth candidates, `local_mv_match=2`, `upsample_factor=4`, `lowest_feature_resolution=8`, image 256×448, near 0.5, far 200, no baseline rescaling. Loading is strict.

Scene 1: `0bfdd020cf475b9c68e4b469d1d1a2d0cad303eefe8b78fb2307855afdaac8be`; targets 17,35,52,70,87,105,122,140. Scene 2: `ff592398657b3dfe94153332861985194a3e3c9d199c4a3a27a0ce4038e81ade`; targets 23,46,70,93,116,139,163,186. The JSON plans store full filename catalogues, primary and secondary selections. Row numbers refer to the converted scene catalogue, not necessarily filename suffixes. These are exploratory scenes, not a final held-out benchmark.

## Reproduction on HPC

Run from the **isolated checkout**; original experiment directories are read-only inputs. Adjust only the paths below. Run model computation through Slurm, not on the login node. The launcher inherits an activated environment; optional `DIAG_ENV=/path/to/conda/env` activates the recorded module setup, whose availability must match your cluster.

```bash
conda activate depthsplat
export SUITE_REPO="$PWD"
export DATA_ROOT=/home/3188473/depthsplat_support_diagnostics/datasets/dl3dv_960p_test_subset
export CHECKPOINT=/home/3188473/depthsplat_support_diagnostics/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
export SCENE=0bfdd020cf475b9c68e4b469d1d1a2d0cad303eefe8b78fb2307855afdaac8be
python -m diagnostics.make_config --dataset-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" --out diagnostics/local_config.yaml
sbatch diagnostics/jobs/infer.sbatch --config diagnostics/local_config.yaml --dataset-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" --scene "$SCENE" --selection-plan "diagnostics/plans/$SCENE.json" --out outputs/focused_export_scene1
```

Wait for the export job to finish successfully. For existing exports, skip that job and set `EXPORTED` to `/home/3188473/depthsplat_support_diagnostics/outputs/support_diag_771900_0`. Those inputs must include `diagnostic_inputs.npz`, `environment.json`, `selection.json` and `resolved_config.yaml`; regenerate if missing, never guess preprocessing.

```bash
export EXPORTED="$PWD/outputs/focused_export_scene1"
sbatch diagnostics/jobs/run_suite.sbatch --run "$EXPORTED" --method trajectory --checkpoint "$CHECKPOINT" --out outputs/focused_scene1_uniform --feature-ablations
```

Repeat with `--method fps` and a new output directory. For scene 2, change `SCENE` to its full ID, use that plan, and choose distinct export/result directories. The runners refuse to overwrite existing inference outputs. Each completed run has `status.json` = `complete`, `checks.json`, source/input/checkpoint hashes, per-target CSV, scene summary, saved renders, selected depths and inspection arrays. Keep the arrays on HPC, outside Git. Do not treat partially written summaries as completed runs.

After the GPU job completes, regenerate selected plots on HPC or a machine with its copied outputs:

```bash
python -m diagnostics.plot_renderings --run outputs/focused_scene1_uniform --out outputs/focused_scene1_uniform/figures/renderings --targets 35 70 --crop 144 64 160 160
python -m diagnostics.inspect --run outputs/focused_scene1_uniform --out outputs/focused_scene1_uniform/figures/inputs --primaries 0 101
python -m diagnostics.reproject --run "$EXPORTED" --method trajectory --out outputs/focused_scene1_uniform/figures/reprojection --targets 35 70 --primaries 0 37
```

Plot defaults deliberately select one target/primary. `--all-figures` regenerates every panel; numerical tables always cover all targets/primaries. Crops use x,y,width,height in saved image pixels and are shared by all conditions. Run plotting in a suitable CPU allocation if cluster policy requires it.

## Reading the outputs

* Rendering PSNR/SSIM increase and LPIPS decreases indicate improvement on the identical target images. Inspect per-target changes before interpreting the scene mean. Saved PNG error maps are channel-mean absolute RGB differences on [0,1], after PNG quantization; tabulated metrics use unquantized model tensors and repository metric functions.
* Confidence histograms use identical bins from 0 to 1, shared axes and density normalization (area = 1). A rightward shift means sharper candidate probabilities, not proven accuracy. Spatial maps show the associated primary RGB and secondary-minus-baseline confidence with a fixed ±1 scale.
* Feature magnitude is channel RMS. Relative difference at each pixel is channel RMS of the difference divided by the baseline RMS over the entire primary map, protected by 1e-8. Cosine similarity compares channel-vector direction; zero-vector cases are undefined/grey. Before maps are native MV levels; after maps are the fused feature-upsample output, compared with baseline. Both magnitude maps share a stage-wide scale; magnitude/difference limits clip at the group 99th percentile, while numerical tables are unclipped. None of these maps measures reconstruction error or confidence.
* Reprojection uses primary RGB + primary camera-Z depth, normalized pixel centers and c2w cameras to project into a target. Nearest-pixel z-buffering keeps the closest projected sample. Coverage is the fraction of occupied target pixels. Both photometric errors are scored on their intersection, so missing coverage is not mistaken for zero error. Grey pixels are unscored. Combined panels z-buffer all six primaries; individual panels isolate one. This is a photometric alignment proxy, not ground-truth depth accuracy; viewpoint-dependent appearance, occlusions, collisions and sparse sampling affect it. No target-depth visibility test is available.

Depth-only historical results largely preserve reconstruction quality, while confidence and multiview-feature substitutions degrade the tested scene. The new baseline-depth feature-only conditions test whether those effects persist without the depth change. Their results remain pending; no generalization claim follows from two scenes or mixed-source interventions.
