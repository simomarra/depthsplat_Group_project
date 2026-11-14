"""Save support selections without changing the completed six-view baseline."""
import json
import runpy
from pathlib import Path

import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "datasets/dl3dv_960p_test_subset/test"
    baseline_path = root / "assets/dl3dv_subset_6p_8t.json"
    baseline = json.loads(baseline_path.read_text())
    chunk_index = json.loads((data / "index.json").read_text())
    choose = runpy.run_path(str(root / "src/dataset/view_sampler/secondary_selection.py"))[
        "select_secondary_indices"
    ]
    selected, manifest = {}, {}
    for scene, entry in baseline.items():
        chunk = torch.load(data / chunk_index[scene], map_location="cpu", weights_only=False)
        example = next(x for x in chunk if x["key"] == scene)
        primary = sorted(entry["context"])
        targets = sorted(entry["target"])
        assert len(primary) == 6 and len(set(primary)) == 6
        assert len(targets) == 8 and len(set(targets)) == 8
        assert len(example["images"]) >= 50
        assert all(0 <= x < 50 for x in primary + targets)
        assert not set(primary) & set(targets)
        cameras = example["cameras"][:50]
        w2c = torch.eye(4).repeat(50, 1, 1)
        w2c[:, :3] = cameras[:, 6:].reshape(50, 3, 4)
        positions = w2c.inverse()[:, :3, 3].numpy()
        pairs = choose(positions, np.asarray(primary), excluded_indices=np.asarray(targets))
        support = pairs.reshape(-1).tolist()
        context = primary + support
        assert len(set(context)) == 18
        assert not set(context) & set(targets)
        assert all(0 <= x < 50 for x in context)
        selected[scene] = {"context": context, "target": targets}
        manifest[scene] = {
            "interval": [0, 49], "primary": primary, "support": support,
            "support_pairs": pairs.tolist(), "target": targets,
            "selection": "two unique frames per primary; trajectory offset 0.25; inward endpoints",
            "matching": "two nearest other cameras from primary + support pool",
            "note": "Pairs select the pool; they do not force matching assignments.",
        }
        print(scene, "\n  Primary:", primary, "\n  Support:", support,
              "\n  Target:", targets, flush=True)
        del chunk, example
    for path, value in (
        (root / "assets/dl3dv_subset_6p_12s_8t.json", selected),
        (root / "assets/dl3dv_subset_support_manifest.json", manifest),
    ):
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f"Refusing to replace different selections: {path}")
        path.write_text(json.dumps(value, indent=2) + "\n")
        print("Saved", path)


if __name__ == "__main__":
    main()
