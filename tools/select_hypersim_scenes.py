"""Freeze a random subset of released Hypersim scenes before image downloads."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--num-plots", type=int, default=3)
    args = parser.parse_args()

    if not 0 < args.num_plots <= args.num_scenes:
        raise ValueError("Require 0 < num-plots <= num-scenes.")

    frames = defaultdict(set)
    with args.metadata.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected = {"scene_name", "camera_name", "frame_id", "included_in_public_release"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing metadata columns: {expected - set(reader.fieldnames or [])}")
        for row in reader:
            if row["included_in_public_release"].strip().lower() == "true":
                frames[(row["scene_name"], row["camera_name"])].add(row["frame_id"])

    # 6 uniform inputs + 12 supports + up to 6 different baseline inputs
    # + 10 held-out targets = at most 34 distinct frames needed.
    minimum_frames = 34
    eligible = defaultdict(dict)
    for (scene, camera), ids in frames.items():
        if len(ids) >= minimum_frames:
            eligible[scene][camera] = len(ids)
    population = sorted(eligible)
    if len(population) < args.num_scenes:
        raise ValueError(f"Only {len(population)} eligible scenes; need {args.num_scenes}.")

    scenes = random.Random(args.seed).sample(population, args.num_scenes)
    plots = random.Random(args.seed + 1).sample(scenes, args.num_plots)
    result = {
        "dataset": "hypersim",
        "selection_seed": args.seed,
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "minimum_released_frames_in_one_camera_path": minimum_frames,
        "eligible_scene_count": len(population),
        "scene_ids": scenes,
        "plot_scene_ids": plots,
        "eligible_camera_paths": {scene: eligible[scene] for scene in scenes},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"Saved {len(scenes)} of {len(population)} eligible scenes to {args.out}")
    print("Scene IDs:", ", ".join(scenes))
    print("Plot scene IDs:", ", ".join(plots))


if __name__ == "__main__":
    main()
