"""Parameter-free routing of preserved secondary descriptors for inference.

All flattened view indices are scene-major, matching MultiViewUniMatch.
The existing feature pyramid supplies its own frozen parameters; this module
adds no parameters and never changes the multiview attention inputs.
"""
import torch


def view_role_indices(primary_rows, batch_size, num_views, device):
    rows = primary_rows.to(device=device)
    if rows.dtype != torch.long or rows.ndim != 2 or rows.shape[0] != batch_size:
        raise ValueError('Expected primary_rows [B,P] with torch.long dtype.')
    if rows.shape[1] < 2 or rows.shape[1] >= num_views:
        raise ValueError('CNN-secondary matching requires primaries and secondaries.')
    if ((rows < 0) | (rows >= num_views)).any():
        raise ValueError('Primary row outside context.')
    mask = torch.zeros(batch_size, num_views, dtype=torch.bool, device=device)
    mask.scatter_(1, rows, True)
    if not torch.all(mask.sum(1) == rows.shape[1]):
        raise ValueError('Primary rows must be unique.')
    offsets = torch.arange(batch_size, device=device)[:, None] * num_views
    primary = (rows + offsets).reshape(-1)
    secondary = (~mask).reshape(-1).nonzero(as_tuple=True)[0]
    return primary, secondary


def preserve_secondary_cnn(coarse_cnn, primary_rows, batch_size, num_views):
    if coarse_cnn.ndim != 4 or coarse_cnn.shape[0] != batch_size * num_views:
        raise ValueError('Expected coarse CNN features [B*V,C,H,W].')
    _, secondary = view_role_indices(
        primary_rows, batch_size, num_views, coarse_cnn.device
    )
    # index_select creates independent storage before positional encoding
    # and attention. The original CNN tensor is not overwritten.
    return coarse_cnn.index_select(0, secondary), secondary


def replace_primary_secondary_sources(
    normal_sources, cnn_secondary, secondary_indices,
    primary_rows, nn_matrix, batch_size, num_views,
):
    """Replace only source slots for primary references; preserve all others.

    normal_sources: [B*V,K,C,H,W] after normal camera/source batching.
    cnn_secondary: [S,C,H,W], in secondary_indices order, at this scale.
    Cameras and reference descriptors are deliberately not arguments to modify.
    """
    if normal_sources.ndim != 5 or normal_sources.shape[0] != batch_size * num_views:
        raise ValueError('Expected normal source features [B*V,K,C,H,W].')
    if cnn_secondary.ndim != 4 or cnn_secondary.shape[1:] != normal_sources.shape[2:]:
        raise ValueError('CNN-pyramid and multiview descriptors differ in C/H/W.')
    if cnn_secondary.dtype != normal_sources.dtype or cnn_secondary.device != normal_sources.device:
        raise ValueError('Source descriptor dtype/device mismatch.')
    primary, expected_secondary = view_role_indices(
        primary_rows, batch_size, num_views, normal_sources.device
    )
    if not torch.equal(secondary_indices, expected_secondary):
        raise ValueError('Preserved secondary order disagrees with roles.')
    if cnn_secondary.shape[0] != secondary_indices.numel():
        raise ValueError('Wrong secondary feature count.')
    k = normal_sources.shape[1]
    if nn_matrix is None or nn_matrix.shape != (batch_size, num_views, k + 1):
        raise ValueError('Explicit self-first neighbor matrix required.')
    if nn_matrix.dtype != torch.long or nn_matrix.device != normal_sources.device:
        raise ValueError('Neighbor indices must be long tensors on the feature device.')
    if ((nn_matrix < 0) | (nn_matrix >= num_views)).any():
        raise ValueError('Neighbor row outside context.')
    offsets = torch.arange(batch_size, device=normal_sources.device)[:, None, None] * num_views
    source_global = (nn_matrix[:, :, 1:] + offsets).reshape(batch_size * num_views, k)
    primary_sources = source_global.index_select(0, primary)
    lookup = torch.full((batch_size * num_views,), -1, dtype=torch.long,
                        device=normal_sources.device)
    lookup[secondary_indices] = torch.arange(secondary_indices.numel(), device=lookup.device)
    slots = lookup[primary_sources]
    if (slots < 0).any():
        raise ValueError('Every primary matching source must be an assigned secondary.')
    replacements = cnn_secondary.index_select(0, slots.reshape(-1))
    replacements = replacements.reshape(primary.numel(), k, *normal_sources.shape[2:])
    result = normal_sources.clone()
    result.index_copy_(0, primary, replacements)
    return result
