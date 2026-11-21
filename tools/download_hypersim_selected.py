"""Download only the raw Hypersim RGB frames in a frozen view plan.

Uses the Hypersim contributed range-request downloader already installed by
the user. Dry-run by default; --download performs the transfers. Running the
same command again skips complete files but deliberately retries empty files.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


KEY = re.compile(r"^(cam_\d\d)/frame\.(\d{4})$")


def expected_files(plan, scene):
    if plan.get('dataset') != 'hypersim' or scene not in plan['scenes']:
        raise ValueError(f'Not a scene in the Hypersim plan: {scene}')
    record = plan['scenes'][scene]
    primary = record['primary_frame_keys']
    support = [key for pair in record['secondary_frame_keys'] for key in pair]
    baseline = record['baseline_input_frame_keys']
    targets = record['target_frame_keys']
    if (len(primary), len(support), len(baseline), len(targets)) != (6, 12, 6, 10):
        raise ValueError(f'Unexpected input/target counts for {scene}')
    inputs = set(primary + support + baseline)
    if len(inputs.intersection(targets)) or len(set(targets)) != 10:
        raise ValueError(f'Target/input overlap in {scene}')
    required = record['required_image_frame_keys']
    if len(required) != len(set(required)) or set(required) != inputs | set(targets):
        raise ValueError(f'Required images do not match the plan for {scene}')
    names = []
    for key in required:
        match = KEY.fullmatch(key)
        if match is None or key not in record['frame_keys']:
            raise ValueError(f'Invalid or non-public frame key: {key}')
        camera, number = match.groups()
        names.append(f'{scene}/images/scene_{camera}_final_hdf5/frame.{number}.color.hdf5')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--downloader', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Download instead of listing paths')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    names = expected_files(plan, args.scene)
    print(f'{args.scene}: {len(names)} distinct raw RGB files are required', flush=True)
    for name in names:
        destination = args.directory / name
        if not args.download:
            print(name)
            continue
        if destination.is_file() and destination.stat().st_size > 1000:
            print(f'Present: {name}', flush=True)
            continue
        camera_name = name.split('/')[2].removeprefix('scene_').removesuffix('_final_hdf5')
        number = name.split('/')[-1].split('.')[1]
        base = [sys.executable, str(args.downloader), '--scene', args.scene,
                '--contains', f'scene_{camera_name}_final_hdf5',
                '--contains', f'frame.{number}.color.hdf5']
        listed = subprocess.run(base + ['--list'], text=True, capture_output=True, check=True)
        paths = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        if paths != [name]:
            raise ValueError(f'Downloader did not uniquely match {name}: {paths!r}')
        subprocess.run(base + ['--directory', str(args.directory), '--silent'], check=True)
        if not destination.is_file() or destination.stat().st_size <= 1000:
            raise RuntimeError(f'File missing or empty after download: {destination}')
    print(f'Finished {args.scene}: {len(names)} expected image files.', flush=True)


if __name__ == '__main__':
    main()
