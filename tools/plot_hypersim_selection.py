"""Save two-condition 3D camera-path graphs for three frozen random scenes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prepare_hypersim_views import camera_data, scale_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--camera-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads(args.plan.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in saved['scene_selection']['plot_scene_ids']:
        selection = saved['scenes'][scene]
        keys = selection['frame_keys']
        scale = scale_for(args.camera_root / scene)
        camera_positions = {}
        for camera in {key.split('/')[0] for key in keys}:
            camera_positions[camera] = camera_data(
                args.camera_root / scene / '_detail' / camera)
        positions = np.stack([
            camera_positions[cam][int(frame.removeprefix('frame.'))] * scale
            for cam, frame in (key.split('/') for key in keys)
        ])
        input_rows = np.asarray(selection['input_trajectory_indices'])
        primaries = np.asarray(selection['methods']['trajectory']['primary'])
        supports = np.asarray(selection['methods']['trajectory']['secondary']).ravel()
        targets = np.asarray(selection['target_indices'])
        center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
        radius = max((positions.max(axis=0) - positions.min(axis=0)).max() * 0.55, 0.5)

        destination = args.out / f'{scene}_uniform_vs_support.png'
        if destination.exists():
            raise FileExistsError(f'Plot already exists: {destination}')
        fig = plt.figure(figsize=(14, 6))
        for panel, with_support in enumerate((False, True), start=1):
            ax = fig.add_subplot(1, 2, panel, projection='3d')
            for cam in sorted(camera_positions):
                rows = [i for i, key in enumerate(keys) if key.startswith(cam + '/')]
                ax.plot(*positions[rows].T, color='0.82', linewidth=1)
            ax.plot(*positions[input_rows].T, color='black', linewidth=1.4,
                    label='input trajectory')
            ax.scatter(*positions[primaries].T, c='#d94a36', s=55,
                       depthshade=False, label='6 primaries')
            if with_support:
                ax.scatter(*positions[supports].T, c='#2582c2', s=42,
                           depthshade=False, label='12 secondaries')
            ax.scatter(*positions[targets].T, c='#8b41ba', marker='^', s=44,
                       depthshade=False, label='10 held-out targets')
            for dimension, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                setter(center[dimension] - radius, center[dimension] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Uniform: 6 primaries' + (' + 12 secondaries' if with_support else ''))
            ax.legend(loc='upper right', fontsize=7)
        fig.suptitle(f'{scene}: same primaries and target cameras in both conditions')
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        print(destination)


if __name__ == '__main__':
    main()
