"""Select two unique secondary frames per primary using trajectory distance."""

import numpy as np


def select_secondary_indices(
    positions,
    primary_indices,
    offset_fraction=0.25,
    excluded_indices=None,
):
    """
    Return an array shaped (P, 2).

    Each row contains the two secondary indices assigned to the
    corresponding primary in primary_indices.

    All returned indices are globally unique and exclude all primaries.
    Indices refer to rows of positions, which must be chronological.
    """
    positions = np.asarray(positions, dtype=np.float64)
    primary = np.asarray(primary_indices)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")

    if not np.isfinite(positions).all():
        raise ValueError("Camera positions must be finite.")

    if primary.ndim != 1 or not np.issubdtype(primary.dtype, np.integer):
        raise ValueError("primary_indices must be a 1D integer array.")

    n = len(positions)
    p = len(primary)

    if p < 2:
        raise ValueError("At least two primaries are required.")

    if np.any(primary < 0) or np.any(primary >= n):
        raise ValueError("Primary indices are outside the trajectory.")

    if len(np.unique(primary)) != p:
        raise ValueError("Primary indices must be unique.")

    if n - p < 2 * p:
        raise ValueError(
            f"{p} primaries need {2 * p} unique secondary frames, "
            f"but only {n - p} non-primary frames are available."
        )

    if not np.isfinite(offset_fraction) or not 0 < offset_fraction < 0.5:
        raise ValueError("offset_fraction must be between 0 and 0.5.")

    # Same accumulated travel distance used by the primary selector.
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(steps)))

    if not np.isfinite(distance).all() or distance[-1] <= 0:
        raise ValueError("Trajectory must have finite, positive travel distance.")

    # Nominal primary spacing and desired support offset.
    spacing = distance[-1] / (p - 1)
    offset = offset_fraction * spacing

    # A frame becomes unavailable immediately after selection.
    available = np.ones(n, dtype=bool)
    if excluded_indices is not None:
        excluded = np.asarray(excluded_indices)
        if (excluded.ndim != 1 or not np.issubdtype(excluded.dtype, np.integer)
                or np.any(excluded < 0) or np.any(excluded >= n)):
            raise ValueError("excluded_indices must be valid integer rows.")
        available[excluded] = False
    available[primary] = False
    if np.count_nonzero(available) < 2 * p:
        raise ValueError("Not enough secondary candidates after exclusions.")

    secondary = np.empty((p, 2), dtype=np.int64)
    frame_rows = np.arange(n)

    # Process chronologically, but preserve the input primary row mapping.
    processing_order = np.argsort(primary)

    for rank, primary_slot in enumerate(processing_order):
        primary_row = int(primary[primary_slot])
        location = distance[primary_row]

        if rank == 0:
            # First primary: two supports farther along the trajectory.
            desired = [location + offset, location + 2 * offset]
            sides = [1, 1]

        elif rank == p - 1:
            # Last primary: two supports earlier along the trajectory.
            desired = [location - offset, location - 2 * offset]
            sides = [-1, -1]

        else:
            desired = [location - offset, location + offset]
            sides = [-1, 1]

        for support_slot, (target, side) in enumerate(zip(desired, sides)):
            # Prefer the requested chronological side of this primary.
            side_mask = (
                frame_rows < primary_row
                if side == -1
                else frame_rows > primary_row
            )

            candidates = np.flatnonzero(available & side_mask)

            # Fall back if previous assignments exhausted that side.
            if candidates.size == 0:
                candidates = np.flatnonzero(available)

            # Choose the available frame nearest the desired travel distance.
            errors = np.abs(distance[candidates] - target)
            chosen = int(candidates[np.argmin(errors)])

            secondary[primary_slot, support_slot] = chosen
            available[chosen] = False

    # Verify the constraints independently of the selection loop.
    assert np.unique(secondary).size == 2 * p
    assert np.intersect1d(primary, secondary).size == 0

    return secondary