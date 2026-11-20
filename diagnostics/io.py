"""Small JSON/CSV and exact preprocessed-input validation helpers."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

VIEW_KEYS = ("image", "extrinsics", "intrinsics", "near", "far", "index")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def pair_inputs(run, method):
    root = Path(run)
    a = load_npz(root / f"{method}_primary_only/diagnostic_inputs.npz")
    b = load_npz(root / f"{method}_support/diagnostic_inputs.npz")
    validate_pair(a, b)
    return a, b


def depth_array(value):
    if value.ndim == 6 and value.shape[-2:] == (1, 1):
        value = value[..., 0, 0]
    if value.ndim != 4 or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError("Expected finite positive primary depth [1,P,H,W].")
    return value


def validate_pair(a, b):
    """Verify ordering, all camera/RGB/bound fields, roles and held-out targets."""
    np.testing.assert_array_equal(a["primary_indices"], b["primary_indices"])
    pids = a["primary_indices"]
    if pids.ndim != 1 or len(set(pids.tolist())) != len(pids):
        raise ValueError("Primary IDs must be unique 1D indices.")
    for data in (a, b):
        if data["context_image"].ndim != 5 or data["context_image"].shape[0] != 1:
            raise ValueError("Only single-scene batches [1,V,3,H,W] supported.")
        v = data["context_image"].shape[1]
        rows = data["primary_rows"]
        if rows.shape != (1, len(pids)) or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("Invalid primary row metadata.")
        if (rows < 0).any() or (rows >= v).any():
            raise ValueError("Primary row out of bounds.")
        np.testing.assert_array_equal(data["context_index"][0, rows[0]], pids)
        ids = data["context_index"][0]
        targets = data["target_index"][0]
        if len(set(ids.tolist())) != len(ids) or len(set(targets.tolist())) != len(targets):
            raise ValueError("Duplicate context or target IDs.")
        if set(ids.tolist()) & set(targets.tolist()):
            raise ValueError("Target leakage into the input pool.")
        depth = depth_array(data["primary_depth"])
        if depth.shape != (1, len(pids), *data["context_image"].shape[-2:]):
            raise ValueError("Primary depth/image alignment failed.")
        for group in ("context", "target"):
            image = data[f"{group}_image"]
            n = image.shape[1]
            if image.shape[:1] != (1,) or image.shape[2] != 3:
                raise ValueError("Expected RGB image batches.")
            expected = {"intrinsics": (1,n,3,3), "extrinsics": (1,n,4,4),
                        "near": (1,n), "far": (1,n), "index": (1,n)}
            for key in VIEW_KEYS:
                value = data[f"{group}_{key}"]
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite {group}/{key}")
                if key in expected and value.shape != expected[key]:
                    raise ValueError(f"Invalid shape {group}/{key}: {value.shape}")
            if (data[f"{group}_near"] <= 0).any() or not (
                data[f"{group}_far"] > data[f"{group}_near"]
            ).all():
                raise ValueError("Invalid depth bounds.")
    if a["context_image"].shape[1] != len(pids):
        raise ValueError("Baseline must contain exactly the primary views.")
    np.testing.assert_array_equal(a["primary_rows"], np.arange(len(pids))[None])
    sources = b["source_rows"]
    if sources.shape != (1, len(pids), 2) or not np.issubdtype(sources.dtype, np.integer):
        raise ValueError("Expected two assigned secondaries per primary.")
    v = b["context_image"].shape[1]
    if (sources < 0).any() or (sources >= v).any():
        raise ValueError("Secondary row out of bounds.")
    all_rows = np.r_[b["primary_rows"].ravel(), sources.ravel()]
    if len(all_rows) != v or len(np.unique(all_rows)) != v:
        raise ValueError("Roles must be disjoint and cover every context view.")
    for key in VIEW_KEYS:
        aa, bb = a[f"target_{key}"], b[f"target_{key}"]
        np.testing.assert_allclose(aa, bb, rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            a[f"context_{key}"][0, a["primary_rows"][0]],
            b[f"context_{key}"][0, b["primary_rows"][0]], rtol=0, atol=1e-7
        )
