"""Validity-aware source aggregation. No learned parameters."""

import torch
import math


def aggregate_valid_sources(correlations, validity, reliability=None):
    """Aggregate source correlations.

    correlations: [N, K, D, H, W]
    validity: boolean [N, K, D, H, W]
    reliability: optional nonnegative [N, K, H, W]

    N is the flattened scene/reference-view batch.
    K is the number of matching sources.
    D is the number of candidate depths.

    If every source is invalid at a candidate, retain the original mean.
    """
    if correlations.ndim != 5:
        raise ValueError("Expected correlations [N, K, D, H, W].")
    if validity.shape != correlations.shape or validity.dtype != torch.bool:
        raise ValueError("Validity must be boolean and match correlations.")
    if validity.device != correlations.device:
        raise ValueError("Validity and correlations must share a device.")

    baseline = correlations.mean(dim=1)
    weights = validity.to(dtype=correlations.dtype)

    if reliability is not None:
        expected = (
            correlations.shape[0],
            correlations.shape[1],
            correlations.shape[3],
            correlations.shape[4],
        )
        if reliability.shape != expected:
            raise ValueError("Expected reliability [N, K, H, W].")
        if reliability.device != correlations.device:
            raise ValueError("Reliability must share the correlations device.")
        if not torch.isfinite(reliability).all() or (reliability < 0).any():
            raise ValueError("Reliability must be finite and nonnegative.")
        weights = weights * reliability.to(correlations.dtype).unsqueeze(2)

    # Remove invalid contributions explicitly, rather than multiplying
    # potentially nonfinite invalid values by zero.
    usable = correlations.masked_fill(weights == 0, 0)
    denominator = weights.sum(dim=1)
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    aggregated = (usable * weights).sum(dim=1) / safe_denominator

    return torch.where(denominator > 0, aggregated, baseline)
@torch.no_grad()
def source_peak_reliability(
    correlations,
    validity,
    exclusion_radius=1,
    reliability_floor=0.05,
):
    """Return source reliability [N, K, H, W].

    Reliability = floor + gap between the strongest valid match
    and its strongest valid competitor outside the exclusion window.

    Sources without a usable competing candidate receive only the floor.
    """
    if correlations.ndim != 5:
        raise ValueError("Expected correlations [N, K, D, H, W].")
    if validity.shape != correlations.shape or validity.dtype != torch.bool:
        raise ValueError("Validity must be boolean and match correlations.")
    if validity.device != correlations.device:
        raise ValueError("Validity and correlations must share a device.")
    if type(exclusion_radius) is not int or exclusion_radius < 0:
        raise ValueError("Exclusion radius must be a nonnegative integer.")
    if not math.isfinite(reliability_floor) or reliability_floor <= 0:
        raise ValueError("Reliability floor must be finite and positive.")
    if (validity & ~torch.isfinite(correlations)).any():
        raise ValueError("Nonfinite correlation at a valid projection.")

    scores = correlations.masked_fill(~validity, float("-inf"))
    best_score, best_index = scores.max(dim=2)

    depth_index = torch.arange(
        correlations.shape[2], device=correlations.device
    ).reshape(1, 1, -1, 1, 1)

    competing = validity & (
        (depth_index - best_index.unsqueeze(2)).abs() > exclusion_radius
    )
    competing_score = scores.masked_fill(
        ~competing, float("-inf")
    ).max(dim=2).values

    has_comparison = validity.any(dim=2) & competing.any(dim=2)

    # Avoid subtracting infinities when no comparison exists.
    safe_best = torch.where(
        has_comparison, best_score, torch.zeros_like(best_score)
    )
    safe_competing = torch.where(
        has_comparison, competing_score, torch.zeros_like(competing_score)
    )
    gap = (safe_best - safe_competing).clamp_min(0)

    return gap + reliability_floor
