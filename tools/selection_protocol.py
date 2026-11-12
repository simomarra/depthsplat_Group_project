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


def make_plan(positions, count=6, target_count=8, offset=0.25, fps=fps_numpy):
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if positions.shape != (n, 3) or not np.isfinite(positions).all():
        raise ValueError('Expected finite positions [N,3].')
    if not isinstance(count, int) or count < 2 or not isinstance(target_count, int) or target_count < 1:
        raise ValueError('Require integer primary count >=2 and target count >=1.')
    if n < 3 * count + target_count:
        raise ValueError('Too few frames for disjoint primaries, secondaries and targets.')
    # Interior quantiles: spread target frames temporally; reserve before selection.
    targets = np.linspace(0, n - 1, target_count + 2)[1:-1].round().astype(np.int64)
    if len(np.unique(targets)) != target_count:
        raise ValueError('Target quantiles collapsed onto repeated frames.')
    candidates = np.setdiff1d(np.arange(n), targets)
    fps_rows = candidates[np.asarray(fps(positions[candidates], count), dtype=np.int64)]
    primaries = {
        'fps': np.sort(fps_rows),
        'trajectory': trajectory(positions, count, eligible_indices=candidates),
    }
    methods = {}
    for name, primary in primaries.items():
        supports = secondary(positions, primary, offset_fraction=offset, excluded_indices=targets)
        all_inputs = np.concatenate((primary, supports.ravel()))
        if len(np.unique(all_inputs)) != 3 * count or np.intersect1d(all_inputs, targets).size:
            raise ValueError('Selection uniqueness or target separation failed.')
        methods[name] = {'primary': primary.tolist(), 'secondary': supports.tolist()}
    return {
        'frame_count': n, 'num_primary': count, 'num_secondary': 2 * count,
        'target_indices': targets.tolist(), 'support_offset': offset,
        'target_rule': 'interior uniformly spaced frame-index quantiles; fixed before input selection',
        'methods': methods,
    }
