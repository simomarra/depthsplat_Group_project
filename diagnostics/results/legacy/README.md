# Historical results, not a fresh evaluation

These are measured scores from the uploaded HPC review archive. `provenance.json` hashes the source JSON files. `import_legacy.py` verifies scene averages against the per-target scores where available. The two ordinary-secondary controls have aggregate scores only in those ablation directories; their absent per-target files are explicitly listed, not fabricated.

* `support_diag_771900_0` / `_1`: two scenes, FPS and trajectory, ordinary baseline and full secondary.
* `ablation_uniform_scene1_772082`: scene 1 trajectory, five intervention configurations plus ordinary-secondary control.
* `feature_ablation_uniform_scene1_772128`: scene 1 trajectory, feature-group substitutions **with secondary depth**.

The CSV `source_run` distinguishes experiments: repeated baselines and their small numerical differences are preserved. Historical names stay in `original_condition`; `condition` maps only source-equivalent conditions. In particular, `hybrid_support_features` maps to `secondary_features_depth`, not `secondary_features_only`. Historical `hybrid_support_cnn/mono/mv` names stay unchanged because they use secondary depth, unlike the new baseline-depth group controls.

The new `secondary_features_only` and `secondary_mv_features_only` have no completed measurements. Do not describe the tracked figure as a six-condition result. No calibration results are included in this focused suite.

Regenerate the representative figure:

```bash
python -m diagnostics.plot_legacy --out diagnostics/results/legacy/per_target_metrics.png
```

Reimport tables from the original, unchanged HPC output directory if needed:

```bash
python -m diagnostics.import_legacy --source-outputs /home/3188473/depthsplat_support_diagnostics/outputs --out diagnostics/results/legacy
```
