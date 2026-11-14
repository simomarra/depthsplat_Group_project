# Support views inside DepthSplat's primary sampling interval

We tested whether additional images from the same sampling interval improve
novel-view rendering while keeping the six Gaussian-generating views fixed.

## What was tested

- Frozen checkpoint: `depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth`.
- Data: the official two-scene DL3DV 960p test subset, resized/cropped to 256x448.
- Interval: frame indices 0 through 49, inclusive, within each scene.
- Primaries: the six views already selected in DepthSplat's released
  `assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json`, unchanged between runs.
- Targets: eight temporally spread frames per scene, selected from that interval
  after excluding the primary views. Targets were then held fixed and excluded
  from support selection.
- Baseline: six input views; all six predict depth and generate Gaussians.
- Support variant: the same six primaries plus 12 distinct support images.
  Support selection uses the existing trajectory-distance helper: two frames
  per primary, an offset of 0.25 of nominal primary spacing, and inward offsets
  at the endpoints. All supports stay inside the interval and exclude targets.

The support pairs define the candidate pool, not compulsory matching pairs.
Matching retains the original limit of **two nearest other camera positions**,
now selected from the primary-plus-support pool. In the observed run, every
primary's two neighbours were support images, replacing its original primary
neighbours. This experiment does **not** retain two original primary matches
and append additional support matches, and does not use reliability weighting.

All input images supply CNN and multi-view Transformer features. Only the first
six context views enter monocular feature extraction, depth regression and
Gaussian prediction. The support context order must therefore remain
`[six primaries, twelve supports]`; the job disables context sorting for this
variant. Model weights were not fine-tuned.

## Observed result

These numbers were transcribed from the user's HPC logs for Slurm job 763429.
Both scenes and all 16 held-out target renders completed. The patched baseline
reproduced the original baseline job 763386's reported scores exactly.

| Metric | Six-primary baseline | Six primaries + 12 supports | Support minus baseline |
| --- | ---: | ---: | ---: |
| PSNR (higher is better) | 24.542296 | 18.085376 | -6.456921 |
| SSIM (higher is better) | 0.817403 | 0.696823 | -0.120581 |
| LPIPS (lower is better) | 0.126234 | 0.239254 | +0.113020 |
| Generated Gaussians per scene | 688,128 | 688,128 | 0 |

This particular frozen-checkpoint configuration worsened all three rendering
metrics. Two scenes are a pilot, not evidence about the full benchmark or all
support-view methods. The run does not establish why performance declined;
replacing the primary matching neighbours with nearby support images is one
possible explanation to investigate. Keeping primary neighbours and adding
support matches would be a separate, untested ablation.

Encoder timings from job 763429 were about 0.808 s versus 0.868 s per scene.
Each mean contains only two calls, with no dedicated warm-up protocol, so these
are diagnostic timings rather than a reliable efficiency comparison.

## Files and reproduction

`results.json` records the aggregate scores and provenance. The three selection
files under `assets/` record the fixed primary/target indices, expanded context
indices and support-selection manifest. The original checkpoint, dataset,
rendered images and exported PLY files are external artifacts, not included here.

After following the repository installation instructions, download the checkpoint
to `pretrained/` and extract the official test subset to:

```text
datasets/dl3dv_960p_test_subset/test/000000.torch
datasets/dl3dv_960p_test_subset/test/000001.torch
datasets/dl3dv_960p_test_subset/test/index.json
```

From the repository root, run:

```bash
python tools/test_primary_support.py
python tools/prepare_gap_support.py
mkdir -p logs
sbatch tools/support_gap.sbatch
```

The preparation command regenerates and validates the saved support selections
using the downloaded camera poses. It refuses to overwrite different existing
selections. The Slurm script runs the baseline first, checks all three metrics
against job 763386 within an absolute tolerance of 0.001, then runs the support
variant and prints the differences. This reference check is specific to this
two-scene setup; it is not suitable unchanged for other evaluation sets.

The supplied Slurm jobs use partition `stud`, an A100 MIG `4g.40gb` GPU,
four CPUs, 32 GB host memory, CUDA 12.4 and the `depthsplat` Conda environment.
Adapt these cluster-specific settings as necessary. Submit from the repository
root; `DEPTHSPLAT_REPO` may override that directory.

For interactive 3D viewing, run:

```bash
sbatch tools/support_gap_3d.sbatch
```

This repeats both evaluations with Gaussian PLY export enabled. Slurm job 763445
produced four PLY files (two scenes per method), each 42,301,856 bytes, according
to the user's file listings. PLY export trims an eight-pixel border using the
existing exporter. The exporter now uses the primary depth layout and primary
camera rotations rather than treating support images as Gaussian generators.

Outputs are written to `outputs/baseline_gap_JOBID/` and
`outputs/support_gap_JOBID/`, with `metrics/`, `images/` and, for the export job,
`gaussians/` subdirectories. Scene IDs and target image filenames match between
methods. Use the saved target renders for score comparisons; freely navigated
viewer screenshots do not reproduce the evaluation cameras automatically.

The helper tests cover batch-major feature slicing, agreement with original
matching batches, actual support-source gathering and self-match rejection.
Full GPU evaluation was performed on the HPC; this branch records the supplied
logs' results rather than claiming a separate local GPU rerun.
