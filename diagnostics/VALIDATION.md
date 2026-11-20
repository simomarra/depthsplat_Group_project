# Validation status — 20 November 2026

## Completed in the review environment

* Inspected the actual uploaded HPC sources, model routing, saved plans, configurations, launchers and small measured results. Original HPC working directories were not accessed or changed.
* Confirmed common tracked selection helpers/tests in the two exports are identical. Retained them, and retained the support branch model/reference encoder and full configuration tree by basing the patch on `f5c372157edc494e85783bb0a7cd7b11c2be6b71`.
* Reconstructed the baseline exporter from the recorded Git diff; its Git blob matches `7b858309ca8f1b18455892f030e4ecec074c8d53`. The 37-line exact-input export is preserved.
* 25 new CPU tests passed: every intervention's input provenance, incompatible tensors, RGB/camera/bound alignment, role/target exclusion, both exact real plans including reversed dataset mapping, identity/translation/z-buffer/invalid-depth cases, feature magnitude versus direction, and temporary synthetic smoke runs of all three plotting commands.
* Existing available CPU tests: 26 passed (10 first-loop, 8 selection protocol, 8 role validation). Four original tensor-routing tests were skipped because PyTorch is absent.
* Eight original saved-plan tests were attempted but blocked: the uploaded archive omitted their fixture `assets/first_loop_scene_selection.json`. The tests were not changed, deleted or given an invented replacement. Run them in the complete real checkout; the two current diagnostic plans have separate passing tests.
* The binary patch applied cleanly to a separate clean reconstruction of the uploaded base snapshot, and all 25 new tests passed again there. This checks patch packaging, not independent access to upstream history.
* Staged diff/whitespace and file-size review passed: 30 selected files, about 407 KB total; largest file is the one representative figure. No excluded scripts, logs, checkpoints, arrays or datasets were staged.
* Python compilation and Slurm shell syntax checks passed. Historical scene means were checked against the 160 imported per-target rows wherever the source files existed. Missing ordinary-control per-target files are listed in the legacy provenance JSON.
* One representative measured historical figure was regenerated. The six-condition plot generator was exercised only on temporary synthetic data; no synthetic figures or scores are presented as experiment results.

Commands:

```bash
python -m unittest discover -s diagnostics/tests -v
python -m unittest discover -s tools -p 'test_*.py' -v
python -m compileall -q diagnostics tools
bash -n diagnostics/jobs/infer.sbatch diagnostics/jobs/run_suite.sbatch
```

## Implemented runtime checks, not yet executed here

The review environment has no PyTorch/CUDA, dataset or checkpoint. Config composition, model imports, strict checkpoint loading, inference equivalence and the new-condition measurements remain pending on HPC. Runtime checks in `diagnostics.run` fail rather than silently accepting:

1. Changed checkpoint hash, architecture/preprocessing configuration, primary RGB/order/cameras/bounds, targets, overlapping roles or target leakage.
2. Baseline neighbor selection differing from the export; support neighbor matrix differing from the export or the assigned two sources.
3. Ordinary baseline differing from the reference encoder; recomputed baseline/secondary depths differing from saved depths.
4. Any feature pyramid supplied to the feature upsampler differing from the selected source; any Gaussian-regressor depth/RGB/confidence, Gaussian-head confidence, or Gaussian-adapter depth/primary cameras differing from the selected source.
5. Diagnostic baseline Gaussians differing from ordinary baseline; mixed full-secondary Gaussians differing from ordinary secondary inference. Every Gaussian field is compared with default rtol=1e-5, atol=1e-6. Exact source routing uses zero tolerance. Probability normalization is also checked.
6. Non-primary Gaussian counts or unexpected module call counts.

These are checks that execute in a GPU run, **not claims that the run has passed**. A completed six-condition run and selected figures still need to be regenerated, ideally first on scene 1 trajectory and then on scene 2. Inspect `status.json` and `checks.json`. If numerical equivalence fails, investigate before changing tolerance or interpreting scores.

## Scope and limitations

This is an inference-only intervention suite for the recorded six-primary, two-source DL3DV setup, not a training pipeline. It deliberately retains all-view attention and support-depth computation. Reprojection is a photometric proxy; feature magnitude and probability sharpness do not establish geometric accuracy. Legacy results and new-condition results are kept distinct.

GitHub access from this review environment failed with HTTP 403: `Forbidden. Calls to this URL are not allowed.` No GitHub branch was pushed. The delivery contains a patch for a new isolated branch in the actual upstream repository; it must not replace the repository with the review snapshot or its reconstructed history. GPU validation and the actual remote push can be completed through the existing HPC and local VS Code checkout.

## HPC validation update
All 25 diagnostic tests and 38 existing tests passed.
Environment: NumPy 1.24.4; PyTorch 2.4.0+cu124.
Previously blocked fixture and tensor-routing tests now pass.
New six-condition GPU inference and equivalence checks remain pending.
