"""Convert frozen Hypersim HDR frames into consistent centered-camera RGB.

Choose one exposure per scene using the *input frames only*, then apply that
exposure to every input and held-out target. Camera matrices and source keys
are saved alongside the images for downstream inference and auditing.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from download_hypersim_selected import expected_files
from hypersim_camera_model import rectification


def read_rgb(path, expected_shape):
    with h5py.File(path, 'r') as source:
        rgb = np.asarray(source['dataset'][:], dtype=np.float32)
    if rgb.shape != expected_shape or not np.isfinite(rgb).all() or (rgb < 0).any():
        raise ValueError(f'Invalid HDR color image: {path}; shape {rgb.shape}')
    return rgb


def camera_poses(camera_root, scene, keys, rectified, scale):
    cameras = {}
    rotations = np.asarray(rectified['hypersim_to_centered_rotation'], dtype=np.float64)
    for key in keys:
        camera, number = key.split('/')
        if camera not in cameras:
            folder = camera_root / scene / '_detail' / camera
            values = []
            for field in ('frame_indices', 'positions', 'orientations'):
                path = folder / f'camera_keyframe_{field}.hdf5'
                with h5py.File(path, 'r') as source:
                    values.append(np.asarray(source['dataset'][:]))
            ids, positions, orientations = values
            if (positions.shape != (len(ids), 3)
                    or orientations.shape != (len(ids), 3, 3)
                    or len(set(ids.astype(int))) != len(ids)):
                raise ValueError(f'Invalid camera trajectory: {folder}')
            cameras[camera] = {int(i): (p, r) for i, p, r in
                               zip(ids, positions, orientations)}
        position, orientation = cameras[camera][int(number.removeprefix('frame.'))]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = orientation @ rotations
        pose[:3, 3] = position * scale
        if not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3):
            raise ValueError(f'Invalid orientation for {scene}/{key}')
        yield pose


def convert_one_scene(plan, scene, raw_root, camera_root, camera_csv, output_root):
    record = plan['scenes'][scene]
    files = expected_files(plan, scene)
    paths = [raw_root / f for f in files]
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size <= 1000]
    if missing:
        raise FileNotFoundError(f'{len(missing)} selected RGB files missing; first: {missing[0]}')
    camera = rectification(camera_csv, scene)
    shape = (camera['height'], camera['width'], 3)
    input_keys = set(record['primary_frame_keys'] + record['baseline_input_frame_keys'])
    input_keys.update(key for pair in record['secondary_frame_keys'] for key in pair)
    assert input_keys.isdisjoint(record['target_frame_keys'])
    # Fixed exposure from input images; targets do not determine preprocessing.
    percentiles = []
    for key, path in zip(record['required_image_frame_keys'], paths):
        if key in input_keys:
            rgb = read_rgb(path, shape)
            luminance = rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722
            percentiles.append(float(np.percentile(luminance, 90)))
    scene_p90 = float(np.median(percentiles))
    if not np.isfinite(scene_p90) or scene_p90 <= 0:
        raise ValueError(f'Invalid scene exposure estimate: {scene}')
    exposure = 0.8 / scene_p90

    if output_root.exists():
        raise FileExistsError(f'Refusing to overwrite prepared images: {output_root}')
    keys = record['required_image_frame_keys']
    poses = np.stack(list(camera_poses(camera_root, scene, keys, camera,
                                        record['meters_per_asset_unit'])))
    intrinsics = np.repeat(camera['intrinsics'][None].astype(np.float32), len(keys), axis=0)
    height, width = shape[:2]
    pixel_scale = np.diag([width, height, 1.0])
    transform = pixel_scale @ camera['original_from_centered'] @ np.linalg.inv(pixel_scale)
    y, x = np.mgrid[:height, :width]
    source = transform @ np.stack((x.ravel() + .5, y.ravel() + .5, np.ones(x.size)))
    source_x = (source[0] / source[2] - .5).reshape(height, width)
    source_y = (source[1] / source[2] - .5).reshape(height, width)
    output_root.mkdir(parents=True)
    records = []
    for key, path in zip(keys, paths):
        hdr = read_rgb(path, shape)
        aligned = np.stack([
            map_coordinates(hdr[..., channel], [source_y, source_x],
                            order=1, mode='nearest')
            for channel in range(3)], axis=-1)
        encoded = (255 * np.power(np.clip(aligned * exposure, 0, 1), 1 / 2.2)).round().astype('uint8')
        out = output_root / 'images' / f'{key}.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(encoded, 'RGB').save(out)
        records.append({'frame_key': key, 'source': str(path),
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'prepared_image': str(out.relative_to(output_root))})
        print(f'Prepared {scene}/{key}', flush=True)
    np.savez(output_root / 'selected_cameras.npz',
             frame_keys=np.asarray(keys), extrinsics=poses, intrinsics=intrinsics)
    summary = {'scene': scene, 'source_plan': str(plan['_plan_path']),
               'source_plan_sha256': plan['_plan_sha256'],
               'input_luminance_p90_median': scene_p90,
               'exposure': exposure, 'gamma': 2.2,
               'target_luminance': 0.8, 'intrinsics': camera['intrinsics'].tolist(),
               'camera_zoom': camera['zoom'], 'image_count': len(keys),
               'images': records}
    (output_root / 'preprocessing.json').write_text(json.dumps(summary, indent=2) + '\n',
                                                     encoding='utf-8')
    print(f'Finished {scene}: {len(keys)} RGB files, fixed scene exposure {exposure:.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--camera-root', type=Path, required=True)
    parser.add_argument('--camera-parameters', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan['_plan_path'] = str(args.plan)
    plan['_plan_sha256'] = hashlib.sha256(plan_bytes).hexdigest()
    convert_one_scene(plan, args.scene, args.raw_root, args.camera_root,
                      args.camera_parameters, args.out)


if __name__ == '__main__':
    main()
