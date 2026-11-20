"""Feature changes are activation diagnostics, not confidence or reconstruction error."""
import numpy as np


def feature_comparison(a, b):
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError('Expected matching feature maps [P,C,H,W].')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite features.')
    rms_a = np.sqrt(np.mean(a*a, axis=1))
    rms_b = np.sqrt(np.mean(b*b, axis=1))
    diff = np.sqrt(np.mean((b-a)**2, axis=1))
    scales = np.sqrt(np.mean(a*a, axis=(1,2,3)))
    norms = np.sqrt(np.sum(a*a,axis=1) * np.sum(b*b,axis=1))
    valid = norms > 1e-12
    cosine = np.full(norms.shape, np.nan, dtype=np.float32)
    cosine[valid] = np.sum(a*b,axis=1)[valid] / norms[valid]
    cosine = np.clip(cosine,-1,1)
    maps = dict(rms_baseline=rms_a, rms_secondary=rms_b,
                relative_difference=diff/np.maximum(scales[:,None,None],1e-8), cosine=cosine)
    stats = []
    for j in range(len(a)):
        stats.append(dict(primary_slot=j, channels=a.shape[1], height=a.shape[2], width=a.shape[3],
            rms_baseline=float(scales[j]), rms_secondary=float(np.sqrt(np.mean(b[j]**2))),
            relative_rms_change=float(np.sqrt(np.mean((b[j]-a[j])**2))/max(float(scales[j]),1e-8)),
            mean_spatial_cosine=float(cosine[j][valid[j]].mean()) if valid[j].any() else None,
            valid_cosine_fraction=float(valid[j].mean())))
    return maps, stats
