"""CPU tensor checks; requires PyTorch, no checkpoint/GPU/model downloads.

Run: python tools/test_cnn_secondary_matching.py
"""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
UNIMATCH = ROOT / 'src/model/encoder/unimatch'


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, UNIMATCH / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


routing = load_module('secondary_matching')
Pyramid = load_module('vit_fpn').ViTFeaturePyramid
CNN = load_module('backbone').CNNEncoder


class SourceRouting(unittest.TestCase):
    def setUp(self):
        self.b, self.v = 2, 6
        self.rows = torch.tensor([[4, 1], [0, 5]])
        self.coarse = torch.arange(12.).view(12, 1, 1, 1).expand(12, 3, 2, 2).clone()
        self.saved, self.indices = routing.preserve_secondary_cnn(
            self.coarse, self.rows, self.b, self.v)
        self.normal = torch.arange(12.).view(12, 1, 1, 1, 1).expand(12, 2, 3, 2, 2).clone() + 100
        self.nn = torch.tensor([[[i, (i+1)%6, (i+2)%6] for i in range(6)]]).repeat(2, 1, 1)
        self.nn[0, 4, 1:] = torch.tensor([0, 2])
        self.nn[0, 1, 1:] = torch.tensor([3, 5])
        self.nn[1, 0, 1:] = torch.tensor([1, 2])
        self.nn[1, 5, 1:] = torch.tensor([3, 4])

    def replace(self, saved=None, nn=None):
        return routing.replace_primary_secondary_sources(
            self.normal, self.saved if saved is None else saved,
            self.indices, self.rows, self.nn if nn is None else nn, self.b, self.v)

    def test_capture_has_independent_storage(self):
        before = self.saved.clone()
        self.coarse.add_(1000)
        torch.testing.assert_close(self.saved, before, rtol=0, atol=0)

    def test_batch_source_order_and_unchanged_secondary_references(self):
        normal_before = self.normal.clone()
        nn_before = self.nn.clone()
        result = self.replace()
        expected = {4: [0., 2.], 1: [3., 5.], 6: [7., 8.], 11: [9., 10.]}
        for ref in range(12):
            if ref in expected:
                for slot, value in enumerate(expected[ref]):
                    torch.testing.assert_close(result[ref, slot], torch.full_like(result[ref, slot], value), rtol=0, atol=0)
            else:
                torch.testing.assert_close(result[ref], self.normal[ref], rtol=0, atol=0)
        torch.testing.assert_close(self.normal, normal_before, rtol=0, atol=0)
        torch.testing.assert_close(self.nn, nn_before, rtol=0, atol=0)

    def test_channel_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'C/H/W'):
            self.replace(saved=self.saved[:, :2])

    def test_primary_source_rejected(self):
        changed = self.nn.clone()
        changed[0, 4, 1] = 1
        with self.assertRaisesRegex(ValueError, 'assigned secondary'):
            self.replace(nn=changed)

    def test_reordered_secondary_bank_rejected(self):
        with self.assertRaisesRegex(ValueError, 'order'):
            routing.replace_primary_secondary_sources(
                self.normal, self.saved.flip(0), self.indices.flip(0),
                self.rows, self.nn, self.b, self.v)


class FeatureScales(unittest.TestCase):
    def test_two_scale_cnn_vs_pyramid_and_no_new_parameters(self):
        torch.manual_seed(3)
        cnn = CNN(num_output_scales=2, lowest_scale=8, return_all_scales=True).eval()
        pyramid = Pyramid(128, [1, 2]).eval()
        before = {k: v.clone() for k, v in pyramid.state_dict().items()}
        with torch.inference_mode():
            fine_to_coarse = cnn(torch.randn(6, 3, 32, 48))
            coarse = fine_to_coarse[-1]
            saved, _ = routing.preserve_secondary_cnn(coarse, torch.tensor([[0, 1]]), 1, 6)
            outputs = pyramid(saved)
        self.assertEqual(tuple(fine_to_coarse[-2].shape[1:]), (96, 8, 12))
        self.assertEqual(tuple(outputs[0].shape), (4, 128, 4, 6))
        self.assertEqual(tuple(outputs[1].shape), (4, 64, 8, 12))
        torch.testing.assert_close(outputs[0], saved, rtol=0, atol=0)
        self.assertEqual(before.keys(), pyramid.state_dict().keys())
        for name, value in pyramid.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
