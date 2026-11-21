"""Shared CPU selection protocol. No DepthSplat imports or torch required."""
from pathlib import Path
import runpy
import numpy as np

SAMPLERS = Path(__file__).resolve().parents[1] / 'src/dataset/view_sampler'
trajectory = runpy.run_path(str(SAMPLERS / 'trajectory_spacing.py'))['uniform_trajectory_indices']
secondary = runpy.run_path(str(SAMPLERS / 'secondary_selection.py'))['select_secondary_indices']


def fps_numpy(positions, count):
    """Float32 translation of original FPS; ties can vary numerically vs Torch.

    The actual HPC runner calls the repository's Torch implementation.
    """
    xyz = np.asarray(positions, dtype=np.float32)
    distance = np.full(len(xyz), 1e10, dtype=np.float32)
    center = xyz.sum(axis=0) / len(xyz)
    farthest = int(np.argmax(np.sum((xyz - center) ** 2, axis=-1)))
    selected = []
    for _ in range(count):
        selected.append(farthest)
        dist = np.sum((xyz - xyz[farthest]) ** 2, axis=-1)
        distance = np.minimum(distance, dist)
        farthest = int(np.argmax(distance))
    if len(set(selected)) != count:
        raise ValueError('Original FPS repeated indices on this degenerate trajectory.')
    return np.sort(np.asarray(selected, dtype=np.int64))


def make_plan(
    positions,
    baseline_indices,
    count=6,
    target_count=10,
    offset=0.25,
    seed=0,
    input_indices=None,
):
    """Select uniform inputs, then random targets unused by any condition.

    All indices are rows of positions. input_indices defines one chronological
    camera trajectory; targets are sampled across all released scene frames.
    baseline_indices contains every input frame used by the separate baseline.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if positions.shape != (n, 3) or not np.isfinite(positions).all():
        raise ValueError("Expected finite positions [N, 3].")
    if count != 6 or target_count != 10:
        raise ValueError("This experiment requires 6 primaries and 10 targets.")

    baseline = np.asarray(baseline_indices)
    if (baseline.ndim != 1 or not np.issubdtype(baseline.dtype, np.integer)
            or len(baseline) == 0
            or np.any(baseline < 0) or np.any(baseline >= n)
            or len(np.unique(baseline)) != len(baseline)):
        raise ValueError("baseline_indices must be unique, valid frame rows.")

    input_rows = np.arange(n, dtype=np.int64) if input_indices is None else np.asarray(input_indices)
    if (input_rows.ndim != 1 or not np.issubdtype(input_rows.dtype, np.integer)
            or np.any(input_rows < 0) or np.any(input_rows >= n)
            or len(np.unique(input_rows)) != len(input_rows)
            or len(input_rows) < 3 * count):
        raise ValueError('input_indices must be unique, ordered trajectory rows.')
    primary_local = trajectory(positions[input_rows], count)
    supports_local = secondary(positions[input_rows], primary_local, offset_fraction=offset)
    primary = input_rows[primary_local]
    supports = input_rows[supports_local]
    inputs = np.concatenate((primary, supports.ravel()))
    if len(np.unique(inputs)) != 3 * count:
        raise ValueError("Primary and secondary frames must be distinct.")

    used = np.unique(np.concatenate((inputs, baseline)))
    eligible_targets = np.setdiff1d(np.arange(n), used)
    if len(eligible_targets) < target_count:
        raise ValueError("Too few unused frames for 10 held-out targets.")

    targets = np.sort(
        np.random.default_rng(seed).choice(
            eligible_targets, size=target_count, replace=False
        )
    )
    return {
        "frame_count": n,
        "num_primary": count,
        "num_secondary": 2 * count,
        "target_indices": targets.tolist(),
        "baseline_input_indices": baseline.tolist(),
        "support_offset": offset,
        "target_seed": seed,
        "input_trajectory_indices": input_rows.tolist(),
        "target_rule": "seeded random sample from all frames unused by any condition",
        "methods": {
            "trajectory": {
                "primary": primary.tolist(),
                "secondary": supports.tolist(),
            }
        },
    }
