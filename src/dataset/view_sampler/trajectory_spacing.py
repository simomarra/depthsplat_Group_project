"""Select frames approximately uniformly along a 3D camera trajectory."""

import numpy as np


def uniform_trajectory_indices(
    positions: np.ndarray,
    num_views: int,
    eligible_indices=None,
) -> np.ndarray:
    """
    Select unique frame indices by accumulated camera travel distance.

    Parameters
    ----------
    positions:
        Camera centers with shape (N, 3), in chronological order.
    num_views:
        Number of frames to select, between 2 and N.

    Returns
    -------
    np.ndarray:
        Selected row indices, in chronological order.
        The first and last eligible frames are always included.

    eligible_indices:
        Optional allowed frame rows. Distances still follow the full path.

    Notes
    -----
    Selection is deterministic and approximately uniform.
    Repeated camera positions may still be selected, but frame indices
    are always unique. A completely stationary trajectory is rejected.
    """
    # Use floating-point coordinates for distance calculations.
    positions = np.asarray(positions, dtype=np.float64)

    # Validate the input before computing distances.
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")

    if not np.isfinite(positions).all():
        raise ValueError("Camera positions must be finite.")

    n = len(positions)

    if isinstance(num_views, (bool, np.bool_)) or not isinstance(
        num_views, (int, np.integer)
    ):
        raise ValueError("num_views must be an integer.")

    if not 2 <= num_views <= n:
        raise ValueError("Require 2 <= num_views <= number of frames.")

    # Distance travelled between each pair of consecutive frames.
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    # Distance travelled from the first frame to each frame.
    distance = np.concatenate(([0.0], np.cumsum(steps)))

    if not np.isfinite(distance).all():
        raise ValueError("Trajectory distance overflowed.")

    if distance[-1] == 0:
        raise ValueError(
            "No camera translation: trajectory spacing is undefined."
        )

    # Keep the original path lengths, even when held-out frames are excluded.
    eligible = np.arange(n) if eligible_indices is None else np.asarray(eligible_indices)
    if (eligible.ndim != 1 or not np.issubdtype(eligible.dtype, np.integer)
            or np.any(eligible < 0) or np.any(eligible >= n)
            or len(np.unique(eligible)) != len(eligible)):
        raise ValueError("eligible_indices must contain unique valid integer rows.")
    eligible = np.sort(eligible)
    if len(eligible) < num_views:
        raise ValueError("Not enough eligible primary frames.")
    targets = np.linspace(distance[eligible[0]], distance[eligible[-1]], num_views)
    selected = np.empty(num_views, dtype=np.int64)
    selected[0] = 0
    selected[-1] = len(eligible) - 1
    for j in range(1, num_views - 1):
        lower = selected[j - 1] + 1
        upper = len(eligible) - num_views + j
        errors = np.abs(distance[eligible[lower:upper + 1]] - targets[j])
        selected[j] = lower + np.argmin(errors)
    return eligible[selected]
