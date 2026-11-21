"""Freeze the same inputs and held-out targets for all three conditions.

Run using only Hypersim's public image CSV and small camera metadata files.
This writes a manifest; it never downloads images or constructs a model.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from selection_protocol import fps_numpy, make_plan


def released_frames(path, scenes):
    wanted = set(scenes)
    released = defaultdict(set)
    with path.open(newline='', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            scene = row['scene_name']
            if scene in wanted and row['included_in_public_release'].strip().lower() == 'true':
                released[(scene, row['camera_name'])].add(int(row['frame_id']))
    return released


def camera_data(camera_dir):
    arrays = []
    for name in ('camera_keyframe_frame_indices', 'camera_keyframe_positions',
                 'camera_keyframe_orientations'):
        with h5py.File(camera_dir / f'{name}.hdf5', 'r') as source:
            arrays.append(np.asarray(source['dataset'][:]))
    ids, positions, rotations = arrays
    if (ids.ndim != 1 or positions.shape != (len(ids), 3)
            or rotations.shape != (len(ids), 3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
            or not np.allclose(ids, np.rint(ids))
            or len(set(ids.astype(int))) != len(ids)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
        raise ValueError(f'Invalid camera arrays in {camera_dir}')
    return {int(frame): positions[i] for i, frame in enumerate(ids)}


def scale_for(scene_dir):
    with (scene_dir / '_detail' / 'metadata_scene.csv').open(newline='') as source:
        scales = [float(row['parameter_value']) for row in csv.DictReader(source)
                  if row['parameter_name'] == 'meters_per_asset_unit']
    if len(scales) != 1 or not np.isfinite(scales[0]) or scales[0] <= 0:
        raise ValueError(f'Invalid meters_per_asset_unit in {scene_dir}')
    return scales[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--metadata-images', type=Path, required=True)
    parser.add_argument('--camera-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    chosen = json.loads(args.scenes.read_text(encoding='utf-8'))
    if hashlib.sha256(args.metadata_images.read_bytes()).hexdigest() != chosen['metadata_sha256']:
        raise ValueError('Image metadata differs from the frozen scene-selection input.')
    scenes = chosen['scene_ids']
    released = released_frames(args.metadata_images, scenes)
    output = {'dataset': 'hypersim', 'scene_selection': chosen,
              'baseline_rule': '6 farthest-point primaries from the same input camera trajectory',
              'target_rule': '10 random released frames across every path in the scene, excluding all inputs',
              'scenes': {}}

    for scene in scenes:
        scene_dir = args.camera_root / scene
        scale = scale_for(scene_dir)
        paths = chosen['eligible_camera_paths'][scene]
        input_camera = sorted(paths, key=lambda cam: (-paths[cam], cam))[0]
        # All released paths, including those too short to serve as input trajectories.
        keys = sorted((cam, frame) for (name, cam), frames in released.items()
                      if name == scene for frame in frames)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f'No unique released frames in {scene}')
        cameras = {cam: camera_data(scene_dir / '_detail' / cam)
                   for cam in {cam for cam, _ in keys}}
        try:
            positions = np.stack([cameras[cam][frame] * scale for cam, frame in keys])
        except KeyError as error:
            raise ValueError(f'No pose for a released frame in {scene}: {error}') from error
        input_rows = np.array([i for i, (cam, _) in enumerate(keys)
                               if cam == input_camera], dtype=np.int64)
        if len(input_rows) != paths[input_camera]:
            raise ValueError(f'Input path count changed for {scene}.')
        baseline_local = fps_numpy(positions[input_rows], 6)
        baseline_rows = input_rows[baseline_local]
        scene_seed = int.from_bytes(hashlib.sha256(
            f'{chosen["selection_seed"]}:{scene}'.encode()).digest()[:8], 'big')
        plan = make_plan(positions, baseline_rows, seed=scene_seed,
                         input_indices=input_rows)
        frame_keys = [f'{cam}/frame.{frame:04d}' for cam, frame in keys]
        primary = plan['methods']['trajectory']['primary']
        supports = [i for pair in plan['methods']['trajectory']['secondary'] for i in pair]
        required = sorted(set(primary) | set(supports) |
                          set(baseline_rows) | set(plan['target_indices']))
        plan.update(input_camera=input_camera, scene=scene,
                    meters_per_asset_unit=scale,
                    frame_keys=frame_keys,
                    required_image_frame_keys=[frame_keys[i] for i in required],
                    primary_frame_keys=[frame_keys[i] for i in primary],
                    secondary_frame_keys=[[frame_keys[i] for i in pair]
                                          for pair in plan['methods']['trajectory']['secondary']],
                    baseline_input_frame_keys=[frame_keys[i] for i in baseline_rows],
                    target_frame_keys=[frame_keys[i] for i in plan['target_indices']])
        output['scenes'][scene] = plan
        print(f'{scene}: input {input_camera} ({len(input_rows)} frames), '
              f'{len(keys)} scene frames, 10 targets; baseline and support inputs excluded')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as destination:
        json.dump(output, destination, indent=2)
        destination.write('\n')
    print(f'Saved view plan to {args.out}')


if __name__ == '__main__':
    main()
