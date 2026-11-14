"""Inference-only primary/support routing. No learned parameters.

All depth views are retained internally. Only primary tensors reach the
Gaussian branch. Metadata indices are LOCAL rows in the loaded context.
"""
import numpy as np


def validate_roles(primary, sources, num_views, local_mv_match):
    primary = np.asarray(primary)
    sources = np.asarray(sources)
    if primary.ndim != 2 or sources.ndim != 3:
        raise ValueError('Require primary_rows [B,P] and source_rows [B,P,K].')
    if not np.issubdtype(primary.dtype, np.integer) or not np.issubdtype(sources.dtype, np.integer):
        raise ValueError('Role indices must be integers.')
    b, p = primary.shape
    if p < 2 or sources.shape[:2] != (b, p):
        raise ValueError('Require at least two primaries and matching batch/primary dimensions.')
    k = sources.shape[-1]
    if k != local_mv_match or k < 1 or k >= num_views:
        raise ValueError('Sources per primary must equal local_mv_match, between 1 and V-1.')
    if np.any(primary < 0) or np.any(primary >= num_views) or np.any(sources < 0) or np.any(sources >= num_views):
        raise ValueError('Role index outside loaded context.')
    for i in range(b):
        if len(np.unique(primary[i])) != p:
            raise ValueError('Repeated primary row.')
        if len(np.unique(sources[i])) != p * k:
            raise ValueError('Secondary views must be globally unique within each scene.')
        if np.intersect1d(primary[i], sources[i]).size:
            raise ValueError('A primary cannot be a secondary.')
        if p + p * k != num_views:
            raise ValueError('Every loaded context view must have exactly one role.')
    return primary, sources


def gather_views(value, rows):
    """Gather [B,V,...] with independent row indices for each batch item."""
    import torch
    batch = torch.arange(value.shape[0], device=value.device)[:, None]
    return value[batch, rows.to(value.device)]


def prepare_roles(context, local_mv_match):
    """Return primary rows and self-first neighbor matrix, or (None,None)."""
    import torch
    has_primary = 'primary_rows' in context
    has_sources = 'source_rows' in context
    if not has_primary and not has_sources:
        return None, None
    if has_primary != has_sources:
        raise ValueError('Provide both primary_rows and source_rows.')
    b, v = context['image'].shape[:2]
    primary = context['primary_rows']
    sources = context['source_rows']
    if not isinstance(primary, torch.Tensor) or not isinstance(sources, torch.Tensor):
        raise ValueError('Role metadata must be batched integer tensors.')
    validate_roles(primary.detach().cpu().numpy(), sources.detach().cpu().numpy(), v, local_mv_match)
    if primary.shape[0] != b:
        raise ValueError('Role metadata batch size does not match images.')
    device = context['image'].device
    primary = primary.to(device=device, dtype=torch.long)
    sources = sources.to(device=device, dtype=torch.long)
    # Secondary feature updates and depth estimation use their K nearest
    # other cameras in the full loaded pool. Explicitly exclude self,
    # including when distinct frames have identical camera positions.
    xyz = context['extrinsics'][:, :, :3, 3].detach()
    if not torch.isfinite(xyz).all():
        raise ValueError('Nonfinite camera positions.')
    distances = torch.cdist(xyz.float(), xyz.float())
    eye = torch.eye(v, dtype=torch.bool, device=device)[None]
    distances = distances.masked_fill(eye, float('inf'))
    neighbors = torch.argsort(distances, dim=-1, stable=True)[..., :local_mv_match]
    self_rows = torch.arange(v, device=device)[None, :, None].expand(b, -1, -1)
    matrix = torch.cat((self_rows, neighbors), dim=-1)
    batch = torch.arange(b, device=device)[:, None]
    matrix[batch, primary, 1:] = sources
    return primary, matrix


def primary_results(results, rows, num_views):
    """Select Gaussian-branch inputs after the ALL-view depth forward pass."""
    b = rows.shape[0]
    selected = dict(results)
    selected['depth_preds'] = [gather_views(x, rows) for x in results['depth_preds']]
    for key in ('features_mono_intermediate', 'features_cnn_all_scales',
                'features_mv', 'match_probs'):
        selected[key] = []
        for value in results[key]:
            if value.shape[0] != b * num_views:
                raise ValueError(f'Unexpected flattened view dimension in {key}.')
            viewed = value.reshape(b, num_views, *value.shape[1:])
            selected[key].append(gather_views(viewed, rows).flatten(0, 1))
    return selected


def primary_context(context, rows):
    """Copy only known per-view fields; never guess from coincident shapes."""
    selected = dict(context)
    for key in ('image', 'extrinsics', 'intrinsics', 'near', 'far', 'index', 'depth'):
        if key in context:
            selected[key] = gather_views(context[key], rows)
    selected.pop('primary_rows', None)
    selected.pop('source_rows', None)
    return selected


def pack_primary_support_context(full_context, primary_rows, secondary_rows):
    """Bridge from the NumPy selectors to a PREPROCESSED single-scene context.

    full_context has [1,N,...] tensors in the same chronological order used
    by the selectors. This function does NOT convert raw Nerfstudio poses,
    intrinsics, images or scene scale. Use the existing dataset preprocessing.
    """
    import torch
    if full_context['image'].shape[0] != 1:
        raise ValueError('Packing helper currently supports one scene at a time.')
    p = np.asarray(primary_rows)
    s = np.asarray(secondary_rows)
    if p.ndim != 1 or s.ndim != 2 or s.shape[0] != len(p):
        raise ValueError('Require primary_rows [P], secondary_rows [P,K].')
    # Validate original rows, then pack without requiring every full-scene
    # frame to be used.
    order = np.concatenate((p, s.ravel()))
    if not np.issubdtype(order.dtype, np.integer):
        raise ValueError('Selection indices must be integers.')
    n = full_context['image'].shape[1]
    if np.any(order < 0) or np.any(order >= n) or len(np.unique(order)) != len(order):
        raise ValueError('Selections must be in range, unique and disjoint.')
    device = full_context['image'].device
    rows = torch.as_tensor(order, dtype=torch.long, device=device)[None]
    packed = primary_context(full_context, rows)
    packed['primary_rows'] = torch.arange(len(p), device=device)[None]
    packed['source_rows'] = torch.arange(len(p), len(order), device=device).reshape(1, *s.shape)
    return packed
