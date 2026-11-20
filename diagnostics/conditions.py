"""Input provenance at the depth-predictor -> Gaussian-branch boundary.

Pure Python on purpose: the condition table can be tested without CUDA/PyTorch.
"""
from dataclasses import asdict, dataclass

FEATURE_KEYS = ("features_mono_intermediate", "features_cnn_all_scales", "features_mv")


@dataclass(frozen=True)
class Condition:
    features: str
    depth: str
    confidence: str
    feature_keys: tuple = ()


CONDITIONS = {
    "baseline": Condition("baseline", "baseline", "baseline"),
    "secondary_depth_only": Condition("baseline", "secondary", "baseline"),
    "secondary_depth_confidence": Condition("baseline", "secondary", "secondary"),
    "secondary_features_only": Condition("secondary", "baseline", "baseline", FEATURE_KEYS),
    "secondary_features_depth": Condition("secondary", "secondary", "baseline", FEATURE_KEYS),
    "full_secondary": Condition("secondary", "secondary", "secondary", FEATURE_KEYS),
    # Supplementary ablations deliberately keep BASELINE depth and confidence.
    "secondary_mv_features_only": Condition("secondary_mv", "baseline", "baseline", ("features_mv",)),
    "secondary_cnn_features_only": Condition("secondary_cnn", "baseline", "baseline", ("features_cnn_all_scales",)),
    "secondary_mono_features_only": Condition("secondary_mono", "baseline", "baseline", ("features_mono_intermediate",)),
    # Preserve the original feature-group controls, which used secondary depth.
    "secondary_depth_mv_features": Condition("secondary_mv", "secondary", "baseline", ("features_mv",)),
    "secondary_depth_cnn_features": Condition("secondary_cnn", "secondary", "baseline", ("features_cnn_all_scales",)),
    "secondary_depth_mono_features": Condition("secondary_mono", "secondary", "baseline", ("features_mono_intermediate",)),
}
MAIN_CONDITIONS = tuple(CONDITIONS)[:6]


def condition_table():
    return {name: asdict(spec) for name, spec in CONDITIONS.items()}


def _check_lists(a, b, key):
    if not isinstance(a[key], (list, tuple)) or len(a[key]) != len(b[key]):
        raise ValueError(f"Incompatible pyramid: {key}")
    for x, y in zip(a[key], b[key]):
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Shape/dtype mismatch for {key}: {x.shape} vs {y.shape}")


def mix_results(baseline, secondary, name):
    """Select whole predictor outputs; no input mutation or scalar-probability hacks.

    depth_preds: list of [B,P,H,W]; features/probs: lists of [B*P,C,H,W].
    Full match_probs is selected so the ordinary encoder takes max/interpolates
    and supplies confidence to BOTH Gaussian modules exactly as before.
    """
    spec = CONDITIONS[name]
    if len(baseline["depth_preds"]) != 1 or len(secondary["depth_preds"]) != 1:
        raise ValueError("Inference-only suite requires one final depth prediction.")
    for key in ("depth_preds", *FEATURE_KEYS, "match_probs"):
        _check_lists(baseline, secondary, key)
    result = dict(baseline)
    if spec.depth == "secondary":
        result["depth_preds"] = secondary["depth_preds"]
    for key in spec.feature_keys:
        result[key] = secondary[key]
    if spec.confidence == "secondary":
        result["match_probs"] = secondary["match_probs"]
    return result
