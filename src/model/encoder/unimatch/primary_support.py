"""Tensor operations for primary-only depth regression with support sources."""
import torch


def select_primary_features(features, batch_size, input_views, primary_count):
    """Select the first P views in batch-major flattened [B*V,...] tensors."""
    return features.reshape(batch_size, input_views, *features.shape[1:])[
        :, :primary_count
    ].reshape(batch_size * primary_count, *features.shape[1:])


def batch_primary_matching(features, intrinsics, extrinsics, nn_matrix):
    """Gather K source views for each primary reference from all input views."""
    features = torch.stack(features, dim=1)
    intrinsics = torch.stack(intrinsics, dim=1)
    extrinsics = torch.stack(extrinsics, dim=1)
    batch, total_views = features.shape[:2]
    primaries = nn_matrix.shape[1]
    neighbours = nn_matrix[..., 1:]
    expected = torch.arange(primaries, device=nn_matrix.device)[None].expand(batch, -1)
    if not torch.equal(nn_matrix[..., 0], expected):
        raise ValueError("Neighbour row must begin with its primary reference.")
    if (neighbours == expected[..., None]).any():
        raise ValueError("A reference cannot match against itself.")
    if (neighbours < 0).any() or (neighbours >= total_views).any():
        raise ValueError("Matching source index out of range.")
    batch_indices = torch.arange(batch, device=features.device)[:, None, None]

    def reference(tensor):
        return tensor[:, :primaries].reshape(batch * primaries, *tensor.shape[2:])

    def sources(tensor):
        return tensor[batch_indices, neighbours].reshape(
            batch * primaries, neighbours.shape[-1], *tensor.shape[2:]
        )

    return (reference(features), reference(intrinsics), reference(extrinsics),
            sources(features), sources(intrinsics), sources(extrinsics))
