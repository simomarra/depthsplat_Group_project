"""Heuristic first spatial-loop detection and closed-path view selection.

This detects a return near the starting camera center, not a 360-degree
camera rotation. Nonzero enclosed projected area rejects simple retracing.
All thresholds are experimental and must be checked on the plotted scene.
"""
import numpy as np


def path_data(positions):
    xyz = np.asarray(positions, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError('Require at least three finite camera centers shaped [N,3].')
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if not np.isfinite(travel).all() or scale <= 0:
        raise ValueError('No usable translational trajectory.')
    return xyz, travel, scale


def detect_first_loop(positions, return_fraction=0.08, departure_fraction=0.30,
                      min_travel_fraction=1.5, min_frames=12,
                      min_area_ratio=0.01, rotations=None, max_rotation_deg=None):
    xyz, travel, scale = path_data(positions)
    if not (0 < return_fraction < departure_fraction <= 1):
        raise ValueError('Require 0 < return_fraction < departure_fraction <= 1.')
    if not np.isfinite(min_travel_fraction) or min_travel_fraction <= 0:
        raise ValueError('min_travel_fraction must be positive.')
    if not isinstance(min_frames, (int, np.integer)) or min_frames < 3:
        raise ValueError('min_frames must be an integer >=3.')
    if not np.isfinite(min_area_ratio) or min_area_ratio <= 0:
        raise ValueError('min_area_ratio must be positive.')
    shifted = xyz - xyz[0]
    distance = np.linalg.norm(shifted, axis=1)
    # Closing to the origin contributes zero to this area-vector sum.
    area_vector = np.vstack((np.zeros(3), np.cumsum(np.cross(shifted[:-1], shifted[1:]), axis=0))) / 2
    perimeter = travel + distance
    area_ratio = np.linalg.norm(area_vector, axis=1) / np.maximum(perimeter ** 2, np.finfo(float).tiny)
    angle = np.zeros(len(xyz))
    if max_rotation_deg is not None:
        if not np.isfinite(max_rotation_deg) or not 0 < max_rotation_deg <= 180:
            raise ValueError('max_rotation_deg must lie in (0,180].')
        r = np.asarray(rotations, dtype=float)
        if r.shape != (len(xyz), 3, 3) or not np.isfinite(r).all():
            raise ValueError('Orientation filtering requires rotations [N,3,3].')
        if not np.allclose(r.transpose(0,2,1) @ r, np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(r), 1, atol=1e-3):
            raise ValueError('Camera rotations are not valid rotation matrices.')
        relative = r[0].T @ r
        angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2,-1,1)))
    away = np.flatnonzero(distance >= departure_fraction * scale)
    if not len(away):
        raise ValueError('No loop detected: camera never departed far enough from the start.')
    i = int(away[0]) + 1
    while i < len(xyz):
        if distance[i] > return_fraction * scale:
            i += 1
            continue
        # First near-start visit: choose its closest qualifying frame, not
        # its first threshold crossing, to avoid cutting the lap too early.
        right = i
        while right + 1 < len(xyz) and distance[right + 1] <= return_fraction * scale:
            right += 1
        candidates = np.arange(i, right + 1)
        valid = ((candidates >= min_frames - 1)
                 & (travel[candidates] >= min_travel_fraction * scale)
                 & (area_ratio[candidates] >= min_area_ratio))
        if max_rotation_deg is not None:
            valid &= angle[candidates] <= max_rotation_deg
        candidates = candidates[valid]
        if len(candidates):
            end = int(candidates[np.argmin(distance[candidates])])
            return {'start_row': 0, 'end_row': end, 'scale': scale,
                    'return_tolerance': return_fraction * scale,
                    'closure_distance': float(distance[end]), 'travel_distance': float(travel[end]),
                    'loop_length': float(perimeter[end]), 'area_ratio': float(area_ratio[end]),
                    'rotation_difference_deg': float(angle[end]) if max_rotation_deg is not None else None,
                    'method': 'first qualifying near-start visit, closest frame within visit',
                    'thresholds': {'return_fraction': return_fraction, 'departure_fraction': departure_fraction,
                                   'min_travel_fraction': min_travel_fraction, 'min_frames': min_frames,
                                   'min_area_ratio': min_area_ratio, 'max_rotation_deg': max_rotation_deg}}
        i = right + 1
    raise ValueError('No loop detected: no return passed departure, travel, area and optional orientation checks. No full-sequence fallback was used.')


def closed_trajectory_indices(positions, end_row, num_views, excluded_indices=()):
    xyz, travel, _ = path_data(positions)
    if not isinstance(end_row, (int,np.integer)) or not 1 < end_row < len(xyz):
        raise ValueError('Invalid closing frame.')
    if not isinstance(num_views, (int,np.integer)) or num_views < 2:
        raise ValueError('Require integer num_views >=2.')
    eligible = np.setdiff1d(np.arange(end_row), np.asarray(excluded_indices, dtype=np.int64))
    if len(eligible) < num_views:
        raise ValueError('Not enough eligible frames inside the first loop.')
    length = travel[end_row] + np.linalg.norm(xyz[end_row]-xyz[0])
    targets = np.arange(num_views) * length / num_views
    selected = []
    lower = 0
    for j, target in enumerate(targets):
        upper = len(eligible) - (num_views-j)
        candidates = eligible[lower:upper+1]
        slot = lower + int(np.argmin(np.abs(travel[candidates]-target)))
        selected.append(int(eligible[slot]))
        lower = slot + 1
    return np.asarray(selected, dtype=np.int64)


def closed_secondary_indices(positions, end_row, primary, offset_fraction=0.25, excluded_indices=()):
    xyz, travel, _ = path_data(positions)
    primary = np.asarray(primary)
    if (primary.ndim != 1 or not np.issubdtype(primary.dtype,np.integer)
            or len(primary)<2 or len(np.unique(primary))!=len(primary)
            or np.any(primary<0) or np.any(primary>=end_row)):
        raise ValueError('Primaries must be unique integer rows before the closing frame.')
    if not np.isfinite(offset_fraction) or not 0 < offset_fraction < 0.5:
        raise ValueError('Require support offset in (0,0.5).')
    if not 1 < end_row < len(xyz):
        raise ValueError('Invalid closing frame.')
    available = np.ones(end_row,dtype=bool)
    excluded = np.asarray(excluded_indices,dtype=np.int64)
    if np.any(excluded < 0) or np.any(excluded >= end_row):
        raise ValueError('Excluded frames outside loop.')
    available[excluded] = False
    available[primary] = False
    if available.sum() < 2*len(primary):
        raise ValueError('First loop has too few unused frames for two unique supports per primary.')
    length = travel[end_row] + np.linalg.norm(xyz[end_row]-xyz[0])
    offset = offset_fraction * length / len(primary)
    result = np.empty((len(primary),2),dtype=np.int64)
    for slot in np.argsort(primary):
        for side, sign in enumerate((-1,1)):
            target = (travel[primary[slot]] + sign*offset) % length
            candidates = np.flatnonzero(available)
            delta = np.abs(travel[candidates]-target)
            errors = np.minimum(delta,length-delta)
            chosen = int(candidates[np.argmin(errors)])
            result[slot,side] = chosen
            available[chosen] = False
    return result
