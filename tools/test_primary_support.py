"""CPU checks for batching, matching references and baseline equivalence."""
import ast
from pathlib import Path
import runpy
import unittest

import torch
from einops import repeat

ROOT = Path(__file__).resolve().parents[1]
OPS = runpy.run_path(str(ROOT / "src/model/encoder/unimatch/primary_support.py"))
select = OPS["select_primary_features"]
batch_primary = OPS["batch_primary_matching"]
# Load the unchanged original batching function without model initialisation.
tree = ast.parse((ROOT / "src/model/encoder/unimatch/mv_transformer.py").read_text())
fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
          and x.name == "batch_features_camera_parameters")
namespace = {"torch": torch, "repeat": repeat}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "original_batch", "exec"), namespace)
original = namespace["batch_features_camera_parameters"]


class PrimarySupportTest(unittest.TestCase):
    def test_multi_batch_feature_selection(self):
        x = torch.arange(2 * 9 * 3).reshape(18, 3)
        expected = torch.cat((x[:6], x[9:15]))
        torch.testing.assert_close(select(x, 2, 9, 6), expected)

    def test_matching_equals_original_for_selected_references(self):
        for batch_size in (1, 2):
            total, primary = 9, 6
            features = [torch.randn(batch_size, 3, 2, 4) for _ in range(total)]
            intrinsic = [torch.eye(3).repeat(batch_size, 1, 1) for _ in range(total)]
            extrinsic = [torch.eye(4).repeat(batch_size, 1, 1) for _ in range(total)]
            nn = torch.tensor([[i, (i + 1) % total, (i + 2) % total]
                               for i in range(total)])[None].repeat(batch_size, 1, 1)
            all_outputs = original(features, intrinsic, extrinsic, nn_matrix=nn)
            selected = batch_primary(features, intrinsic, extrinsic, nn[:, :primary])
            for full, part in zip(all_outputs, selected):
                torch.testing.assert_close(part, select(full, batch_size, total, primary))
            # A support source really appears in the primary matching tensor.
            torch.testing.assert_close(selected[3][4, 1], features[6][0])

    def test_all_primary_reproduces_original_batching(self):
        features = [torch.randn(2, 3, 2, 4) for _ in range(6)]
        intrinsic = [torch.eye(3).repeat(2, 1, 1) for _ in features]
        extrinsic = [torch.eye(4).repeat(2, 1, 1) for _ in features]
        nn = torch.tensor([[i, (i + 1) % 6, (i + 2) % 6]
                           for i in range(6)])[None].repeat(2, 1, 1)
        for a, b in zip(original(features, intrinsic, extrinsic, nn_matrix=nn),
                        batch_primary(features, intrinsic, extrinsic, nn)):
            torch.testing.assert_close(a, b)

    def test_self_matching_rejected(self):
        f = [torch.zeros(1, 2, 2, 2) for _ in range(3)]
        k = [torch.eye(3)[None] for _ in f]
        e = [torch.eye(4)[None] for _ in f]
        with self.assertRaises(ValueError):
            batch_primary(f, k, e, torch.tensor([[[0, 0, 1], [1, 0, 2]]]))


if __name__ == "__main__":
    unittest.main()
