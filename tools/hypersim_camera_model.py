"""Rectify Hypersim rays to the centered pinhole camera used by DepthSplat.

Hypersim stores OpenGL-style camera rotations and scene-specific image-plane
transforms, sometimes with tilt/shift. DepthSplat's rasterizer uses only FoV
and requires a centered pinhole image. These matrices map a rectified pixel
back into the published source image while retaining correct world rays.
"""

import csv
import numpy as np


def rectification(camera_csv, scene, margin=0.01):
    with open(camera_csv, newline='', encoding='utf-8') as file:
        rows = [row for row in csv.DictReader(file) if row['scene_name'] == scene]
    if len(rows) != 1:
        raise ValueError(f'Expected one camera-parameter row for {scene}.')
    row = rows[0]
    width = int(float(row['settings_output_img_width']))
    height = int(float(row['settings_output_img_height']))
    m = np.array([[float(row[f'M_cam_from_uv_{i}{j}']) for j in range(3)]
                  for i in range(3)], dtype=np.float64)
    # Normalized image xy (right, down) to Hypersim's UV (right, up).
    uv_from_xy = np.array([[2, 0, -1], [0, -2, 1], [0, 0, 1]], dtype=np.float64)
    cv_from_hypersim = np.diag([1, -1, -1])
    ray_from_xy = cv_from_hypersim @ m @ uv_from_xy
    camera_rotation, triangular = np.linalg.qr(ray_from_xy)
    signs = np.where(np.diag(triangular) >= 0, 1, -1)
    camera_rotation = camera_rotation @ np.diag(signs)
    triangular = np.diag(signs) @ triangular
    if np.linalg.det(camera_rotation) < 0.999 or np.any(np.diag(triangular) <= 0):
        raise ValueError(f'Camera cannot be converted to a proper pinhole: {scene}')
    original_k = np.linalg.inv(triangular)
    original_k /= original_k[2, 2]
    fx, fy = original_k[0, 0], original_k[1, 1]
    cx, cy = original_k[0, 2], original_k[1, 2]
    skew = original_k[0, 1]
    if (not np.isfinite(original_k).all() or fx <= 0 or fy <= 0
            or min(cx, cy, 1 - cx, 1 - cy) <= margin):
        raise ValueError(f'Invalid or unsupported scene intrinsics: {scene}')

    # Aim the new optical axis at the center of the *source image*. A shift in
    # the original principal point is mostly a camera rotation, not a reason
    # to discard the majority of the source field of view.
    z_axis = np.linalg.solve(original_k, np.array([0.5, 0.5, 1.0]))
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.array([1.0, 0.0, 0.0])
    x_axis -= z_axis * np.dot(x_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    center_rotation = np.column_stack((x_axis, y_axis, z_axis))

    corners = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).T

    def projection(zoom):
        centered = np.array([[zoom * fx, 0, 0.5],
                             [0, zoom * fy, 0.5],
                             [0, 0, 1]], dtype=np.float64)
        homography = original_k @ center_rotation @ np.linalg.inv(centered)
        mapped = homography @ corners
        forward = np.all(mapped[2] > 0)
        mapped /= mapped[2]
        safe = (forward
                and np.all(mapped[:2] >= margin)
                and np.all(mapped[:2] <= 1 - margin))
        return safe, centered, homography

    low, high = 0.5, 1.0
    while not projection(high)[0]:
        high *= 2
        if high > 128:
            raise ValueError(f'Unable to rectify camera without border: {scene}')
    for _ in range(50):
        mid = (low + high) / 2
        if projection(mid)[0]:
            high = mid
        else:
            low = mid
    zoom = 1.01 * high
    centered_k = np.array([[zoom * fx, 0, 0.5],
                           [0, zoom * fy, 0.5],
                           [0, 0, 1]], dtype=np.float64)
    original_from_centered = original_k @ center_rotation @ np.linalg.inv(centered_k)
    projected = original_from_centered @ corners
    projected /= projected[2]
    if not (np.all(projected[:2] >= 0) and np.all(projected[:2] <= 1)):
        raise ValueError(f'Rectified image would sample outside {scene}')
    return {
        'width': width, 'height': height, 'zoom': float(zoom),
        'original_intrinsics': original_k,
        'intrinsics': centered_k,
        'original_from_centered': original_from_centered,
        'hypersim_to_centered_rotation': cv_from_hypersim @ camera_rotation @ center_rotation,
    }


def rectify_image(image, camera):
    """Return (RGB image, centered normalized intrinsics) at source resolution."""
    import cv2
    h, w = image.shape[:2]
    if (w, h) != (camera['width'], camera['height']):
        raise ValueError(f'Image shape {(w, h)} differs from camera metadata.')
    pixel_scale = np.diag([w, h, 1.0])
    source_from_rectified = (pixel_scale @ camera['original_from_centered']
                             @ np.linalg.inv(pixel_scale))
    rectified = cv2.warpPerspective(
        image, source_from_rectified, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return rectified, camera['intrinsics']
