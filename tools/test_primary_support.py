"""Run with python tools/test_primary_support.py; no model weights needed."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'src/model/encoder/primary_support.py'
spec = importlib.util.spec_from_file_location('primary_support', path)
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)
try:
    import torch
except ImportError:
    torch = None


class RoleValidation(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[0, 1]])
        self.s = np.array([[[2, 3], [4, 5]]])

    def check_invalid(self, p=None, s=None, v=6, k=2):
        with self.assertRaises(ValueError):
            routing.validate_roles(self.p if p is None else p, self.s if s is None else s, v, k)

    def test_valid(self):
        routing.validate_roles(self.p, self.s, 6, 2)

    def test_primary_overlap(self):
        self.check_invalid(s=np.array([[[0, 3], [4, 5]]]))

    def test_shared_secondary_rejected(self):
        self.check_invalid(s=np.array([[[2, 3], [3, 5]]]))

    def test_repeated_primary_rejected(self):
        self.check_invalid(p=np.array([[0, 0]]))

    def test_out_of_bounds(self):
        self.check_invalid(s=np.array([[[2, 3], [4, 6]]]))

    def test_float_indices_rejected(self):
        self.check_invalid(p=self.p.astype(float))

    def test_wrong_source_count(self):
        self.check_invalid(k=3)

    def test_unassigned_inputs_rejected(self):
        self.check_invalid(v=7)


@unittest.skipIf(torch is None, 'PyTorch required for tensor routing tests')
class TensorRouting(unittest.TestCase):
    def context(self):
        return {
            'image': torch.zeros(1, 6, 3, 2, 2),
            'extrinsics': torch.eye(4).repeat(1, 6, 1, 1),
            'intrinsics': torch.eye(3).repeat(1, 6, 1, 1),
            'near': torch.ones(1, 6), 'far': torch.ones(1, 6) * 10,
            'index': torch.arange(100, 106)[None],
        }

    def test_legacy_no_roles(self):
        self.assertEqual(routing.prepare_roles(self.context(), 2), (None, None))

    def test_pack_neighbors_and_self_exclusion(self):
        context = routing.pack_primary_support_context(self.context(), [4, 1], [[0, 5], [2, 3]])
        self.assertEqual(context['index'].tolist(), [[104, 101, 100, 105, 102, 103]])
        rows, matrix = routing.prepare_roles(context, 2)
        self.assertEqual(matrix[0, :2].tolist(), [[0, 2, 3], [1, 4, 5]])
        for i in range(6):
            self.assertEqual(matrix[0, i, 0].item(), i)
            self.assertNotIn(i, matrix[0, i, 1:].tolist())
        selected = routing.primary_context(context, rows)
        self.assertEqual(selected['index'].tolist(), [[104, 101]])
        self.assertEqual(context['image'].shape[1], 6)  # no mutation

    def test_batch_selection_and_results(self):
        rows = torch.tensor([[4, 1], [0, 5]])
        depth = torch.arange(12).reshape(2, 6, 1, 1)
        features = depth.reshape(12, 1, 1, 1)
        result = {'depth_preds': [depth]}
        for key in ('features_mono_intermediate', 'features_cnn_all_scales', 'features_mv', 'match_probs'):
            result[key] = [features]
        selected = routing.primary_results(result, rows, 6)
        self.assertEqual(selected['depth_preds'][0].flatten().tolist(), [4, 1, 6, 11])
        self.assertEqual(selected['features_mv'][0].flatten().tolist(), [4, 1, 6, 11])
        self.assertEqual(result['depth_preds'][0].shape[1], 6)

    def test_partial_metadata_rejected(self):
        context = self.context()
        context['primary_rows'] = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            routing.prepare_roles(context, 2)


if __name__ == '__main__':
    unittest.main()
