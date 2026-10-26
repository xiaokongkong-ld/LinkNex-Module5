#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
All-in-one organelle pipeline
=============================

Pipeline order
--------------
1. Reviewed *_mask.tif -> *_voxel.txt + *_mesh.obj + *_pcd.txt
2. Instance segmentation from *_pcd.txt + *_voxel.txt
3. Dynamic organelle tracking + temporal network analysis
4. Tracking timelapse video generation

IMPORTANT
---------
Image -> mask generation is intentionally NOT included here.
Review/correct the masks manually first, then run this script.

The three stages were merged from:
- segmentation_others_public2_clean
- dynamic_organelle_tracking_network_v18_pcd_voxel_only
- make_tracking_timelapse_v18

Edit only the USER SETTINGS section for normal use.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx
try:
    import open3d as o3d
except ImportError:
    o3d = None
import numpy as np
import pandas as pd
import tifffile
import imageio.v2 as imageio
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from tqdm import tqdm
from skimage import measure


# =============================================================================
# USER SETTINGS -- normally edit only this section
# =============================================================================

# Root for one cell/time-series, or a root containing multiple cell folders.
BASE_ROOT_DIR = Path("data")

# Empty = automatically process BASE_ROOT_DIR itself when it contains
# processed_output, otherwise process all immediate child folders that do.
CELL_FOLDER_NAMES: List[str] = []

# Run stages in this order.+
# Image -> mask is intentionally separate so masks can be manually reviewed.
RUN_MASK_TO_PCD = True
RUN_INSTANCE_SEGMENTATION = True
RUN_TRACKING_NETWORK = True
RUN_VIDEO_GENERATION = False

SEGMENTATION_DIR_NAME = "processed_output"
VIDEO_DIR_NAME = "tracking_videos"


# -----------------------------------------------------------------------------
# Stage 1: reviewed mask -> voxel + mesh + PCD
# -----------------------------------------------------------------------------

# Reviewed masks must be inside processed_output and named:
#     <frame_name>_mask.tif
#
# Outputs:
#     <frame_name>_voxel.txt   x,y,z,intensity
#     <frame_name>_mesh.obj
#     <frame_name>_pcd.txt     x,y,z,nx,ny,nz
MASK_SUFFIX = "_mask.tif"

# Keep the same geometry settings as the original mask->PCD code.
MASK_MESH_VOXEL_SIZE_ZYX = (1, 1, 1)
MASK_MESH_SMOOTH_ITERATIONS = 6
MASK_POST_READ_LAPLACIAN_ITERATIONS = 1

# True is safest after manual mask correction: OBJ/PCD will always reflect
# the current reviewed mask instead of an older mesh from a previous run.
FORCE_REBUILD_MASK_GEOMETRY = True

# -----------------------------------------------------------------------------
# Stage 2: instance segmentation + morphology
# -----------------------------------------------------------------------------

DATASET_NAME = "peroxi"

# Keep the segmentation morphology calibration exactly as in the source script.
# This is separate from the physical calibration used by tracking/network.
SEGMENTATION_MORPH_VOXEL_SIZE_ZYX = (1, 1, 1)
SEGMENTATION_COORDS_ARE_PHYSICAL = False

OVERWRITE_MORPHOLOGY = True
PCD_TO_VOXEL_MAX_ASSIGN_DISTANCE = None

CONE_HALF_ANGLE_DEG = 15.0
CONE_MAX_DISTANCE = 5.0
N_DIRECTION_BINS = 32
SCORE_THRESHOLD = 2
MIN_CENTER_SCORE = 2
MIN_PEAK_DISTANCE = 4
MERGE_CENTER_DISTANCE = 2.0
CENTER_GAUSSIAN_SIGMA = 1.0
CENTER_CONNECTIVITY = 1
INSTANCE_COLOR_SEED = 42

# -----------------------------------------------------------------------------
# Stage 2: dynamic tracking + temporal network
# -----------------------------------------------------------------------------

# Physical calibration, order = (z, y, x), in micrometres.
VOXEL_SIZE_ZYX = (0.20, 0.057292, 0.057292)
INPUT_COORDS_ARE_PHYSICAL = False
DT_SECONDS = 2.0

TRACKING_OBJECT_TYPE = "puncta"   # Module 5 default; GUI also enforces puncta

TRACKING_PROFILES = {
    "mitochondria": {
        "search_radius_um": 1.50,
        "max_gap_frames": 1,
        "max_assignment_cost": 0.92,
        "ambiguity_margin": 0.10,
        "require_mutual_best": True,
        "min_size_ratio": 0.20,
        "max_size_ratio": 5.00,
        "weights": {
            "distance": 0.80,
            "size": 0.08,
            "direction": 0.07,
            "shape": 0.05,
        },
        "kalman_process_noise": 0.10,
        "kalman_measurement_noise": 0.20,
        "direction_min_step_um": 0.08,
        "shape_max_points": 250,
    },
    "puncta": {
        "search_radius_um": 1.00,
        "max_gap_frames": 2,
        "max_assignment_cost": 0.90,
        "ambiguity_margin": 0.12,
        "require_mutual_best": True,
        "min_size_ratio": 0.35,
        "max_size_ratio": 2.85,
        "weights": {
            "distance": 0.72,
            "size": 0.12,
            "direction": 0.08,
            "shape": 0.08,
        },
        "kalman_process_noise": 0.08,
        "kalman_measurement_noise": 0.15,
        "direction_min_step_um": 0.05,
        "shape_max_points": 200,
    },
}

MIN_INSTANCE_POINTS = 10
USE_PCD_SHAPE_TIEBREAKER = True
MAX_VOXEL_PCD_PAIR_DISTANCE_UM = 0.75

NETWORK_DISTANCE_UM = 0.20
PAIR_DISTANCE_SAVE_LIMIT_UM = 0.50
MAX_SURFACE_POINTS_PER_INSTANCE = 8000
CALCULATE_BETWEENNESS = False
TRANSIENT_MAX_FRAMES = 3

SAVE_TRACKED_POINTCLOUDS = True
SAVE_FRAME_NETWORK_PNGS = True
SAVE_TRAJECTORY_PNGS = True
SAVE_NETWORK_TIME_SERIES = True
PNG_DPI = 180
RANDOM_SEED = 42

TRACKING_ALGORITHM_VERSION = "pcd_voxel_only_kalman_sphere_hungarian_ambiguity_v18"
NETWORK_ALGORITHM_VERSION = "physical_surface_distance_temporal_network_v18"

PCD_INSTANCE_SUFFIXES = (
    "_pcd_instance.txt",
    "_pcd_instance_coarse.txt",
    "_pcd_instance_corase.txt",
)
VOXEL_INSTANCE_SUFFIXES = (
    "_voxel_instance.txt",
    "_voxel_instance_coarse.txt",
    "_voxel_instance_corase.txt",
)
ALL_INSTANCE_SUFFIXES = PCD_INSTANCE_SUFFIXES + VOXEL_INSTANCE_SUFFIXES

# -----------------------------------------------------------------------------
# Stage 4: tracking videos
# -----------------------------------------------------------------------------

# None = each cell folder itself.
BACKGROUND_TIF_DIR: Optional[Path] = None

MAKE_CENTROID_TRACKS = False
MAKE_CENTROID_TRACKS_NETWORK = False
MAKE_TIF_CENTROID_TRACKS = False
MAKE_TIF_CENTROID_TRACKS_NETWORK = False
MAKE_TRACKED_POINTCLOUD = False
MAKE_TRACKED_POINTCLOUD_TRACKS_NETWORK = False
MAKE_3D_CENTROID_TRACKS = False
MAKE_TIF_INSTANCE_OVERLAY = False

TRAIL_LENGTH = 10
SHOW_CURRENT_CENTROIDS = True
SHOW_TRACK_ID = True
MAX_TRACK_LABELS_PER_FRAME = 100

CENTROID_SIZE = 8
TRAIL_LINEWIDTH = 1.0
NETWORK_LINEWIDTH = 0.8
POINTCLOUD_POINT_SIZE = 0.45
INSTANCE_OVERLAY_ALPHA = 0.40

FPS = 5
DPI = 160
FIGSIZE = (8.5, 8.5)

# Keep the direct-pixel overlay behavior from the latest video script.
ORIGINAL_WIDTH = 2400
ORIGINAL_HEIGHT = 2400
BACKGROUND_PERCENTILES = (1.0, 99.7)
DISPLAY_RANGE_SAMPLE_FRAMES = 12
INVERT_Y = True
FIXED_XLIM = None
FIXED_YLIM = None
VIEW_ELEV = 24
VIEW_AZIM = -65
PREFER_MP4 = True
SAVE_RENDERED_PNG_FRAMES = False


if TRACKING_OBJECT_TYPE not in TRACKING_PROFILES:
    raise ValueError(f"Unknown TRACKING_OBJECT_TYPE: {TRACKING_OBJECT_TYPE}")


# =============================================================================
# Shared numeric TXT helpers
# =============================================================================

def read_numeric_txt(path: Path) -> np.ndarray:
    """Read comma-, whitespace-, or tab-separated numeric TXT."""
    last_error = None
    for delimiter in (",", None, "\t"):
        try:
            arr = np.loadtxt(path, delimiter=delimiter)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return np.asarray(arr)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Cannot read numeric point cloud: {path}\n{last_error}")


def load_point_cloud(path: Path) -> np.ndarray:
    """Standalone replacement for the old pcd2skelenton.load_point_cloud helper."""
    return read_numeric_txt(Path(path))


def save_pcd2txt(points: np.ndarray, save_path: Path) -> None:
    """Standalone replacement for the old img2pcd.save_pcd2txt helper."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(save_path, np.asarray(points), fmt="%.3f", delimiter=",")




def _fibonacci_sphere_directions(n_directions):
    """Create approximately uniform unit directions on a sphere."""
    if n_directions < 1:
        raise ValueError("n_directions must be at least 1.")

    i = np.arange(n_directions, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n_directions
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = i * golden_angle

    directions = np.column_stack((
        radius * np.cos(phi),
        radius * np.sin(phi),
        z,
    ))
    return directions.astype(np.float32)


def score_voxels_by_inward_normal_cones(
    pcd,
    voxel,
    cone_half_angle_deg=15.0,
    max_distance=20.0,
    n_direction_bins=32,
    min_forward_distance=0.0,
    surface_batch_size=2000,
    workers=-1,
    return_debug=False,
):
    """
    Score interior voxels using cones emitted along the inward surface normals.

    Parameters
    ----------
    pcd : array, shape (N, >=6)
        Columns 0:3 are x,y,z and columns 3:6 are outward nx,ny,nz.
    voxel : array, shape (M, >=3)
        Columns 0:3 are x,y,z. Extra columns (for example intensity) are kept.
    cone_half_angle_deg : float
        Half opening angle of each inward cone in degrees.
    max_distance : float
        Maximum cone length, in the same coordinate unit as pcd and voxel.
    n_direction_bins : int
        Number of spherical direction bins. Repeated hits from the same bin
        contribute only one point to a voxel's score. Must be <= 64.
    min_forward_distance : float
        Ignore voxels this close to or behind the cone origin.
    surface_batch_size : int
        Number of surface points processed per KD-tree batch.
    workers : int
        Worker count passed to scipy.spatial.cKDTree.query_ball_point.
    return_debug : bool
        If True, also return the direction bit mask and bin-center directions.

    Returns
    -------
    voxel_with_score : array, shape (M, voxel.shape[1] + 1)
        Original voxel columns followed by the unique-direction score (0..bins).
    """
    pcd = np.asarray(pcd)
    voxel = np.asarray(voxel)

    if pcd.ndim != 2 or pcd.shape[1] < 6:
        raise ValueError("pcd must be N x >=6: x,y,z,nx,ny,nz.")
    if voxel.ndim != 2 or voxel.shape[1] < 3:
        raise ValueError("voxel must be M x >=3: x,y,z[,intensity,...].")
    if not 0.0 < cone_half_angle_deg < 90.0:
        raise ValueError("cone_half_angle_deg must be between 0 and 90.")
    if max_distance <= 0:
        raise ValueError("max_distance must be positive.")
    if not 1 <= n_direction_bins <= 64:
        raise ValueError("n_direction_bins must be between 1 and 64.")
    if min_forward_distance < 0:
        raise ValueError("min_forward_distance must be non-negative.")
    if surface_batch_size < 1:
        raise ValueError("surface_batch_size must be at least 1.")

    n_voxels = len(voxel)
    direction_masks = np.zeros(n_voxels, dtype=np.uint64)

    if len(pcd) == 0 or n_voxels == 0:
        scores = np.zeros(n_voxels, dtype=np.uint8)
        result = np.column_stack((voxel, scores))
        if return_debug:
            return result, direction_masks, _fibonacci_sphere_directions(n_direction_bins)
        return result

    surface_valid = (
        np.isfinite(pcd[:, :3]).all(axis=1)
        & np.isfinite(pcd[:, 3:6]).all(axis=1)
    )
    voxel_valid = np.isfinite(voxel[:, :3]).all(axis=1)

    surface_xyz = pcd[surface_valid, :3].astype(np.float32, copy=False)
    outward_normals = pcd[surface_valid, 3:6].astype(np.float32, copy=False)
    voxel_valid_indices = np.flatnonzero(voxel_valid)
    voxel_xyz = voxel[voxel_valid, :3].astype(np.float32, copy=False)

    normal_lengths = np.linalg.norm(outward_normals, axis=1)
    normal_valid = normal_lengths > 1e-8
    surface_xyz = surface_xyz[normal_valid]
    outward_normals = outward_normals[normal_valid]
    normal_lengths = normal_lengths[normal_valid]

    direction_centers = _fibonacci_sphere_directions(n_direction_bins)

    if len(surface_xyz) == 0 or len(voxel_xyz) == 0:
        scores = np.zeros(n_voxels, dtype=np.uint8)
        result = np.column_stack((voxel, scores))
        if return_debug:
            return result, direction_masks, direction_centers
        return result

    inward_directions = -outward_normals / normal_lengths[:, None]
    direction_bin_ids = np.argmax(
        inward_directions @ direction_centers.T,
        axis=1,
    ).astype(np.int64)

    voxel_tree = cKDTree(voxel_xyz)
    cos_angle_sq = np.cos(np.deg2rad(cone_half_angle_deg)) ** 2
    valid_direction_masks = np.zeros(len(voxel_xyz), dtype=np.uint64)

    for start in tqdm(
        range(0, len(surface_xyz), surface_batch_size),
        desc="Scoring voxels by inward cones",
    ):
        stop = min(start + surface_batch_size, len(surface_xyz))
        origins_batch = surface_xyz[start:stop]
        directions_batch = inward_directions[start:stop]
        bins_batch = direction_bin_ids[start:stop]

        neighbor_lists = voxel_tree.query_ball_point(
            origins_batch,
            r=max_distance,
            workers=workers,
        )
        counts = np.fromiter(
            (len(indices) for indices in neighbor_lists),
            dtype=np.int64,
            count=len(neighbor_lists),
        )
        total_candidates = int(counts.sum())
        if total_candidates == 0:
            continue

        source_ids = np.repeat(np.arange(len(origins_batch)), counts)
        voxel_ids = np.concatenate(neighbor_lists).astype(np.int64, copy=False)

        offsets = voxel_xyz[voxel_ids] - origins_batch[source_ids]
        forward = np.einsum(
            "ij,ij->i",
            offsets,
            directions_batch[source_ids],
        )
        distance_sq = np.einsum("ij,ij->i", offsets, offsets)

        inside_cone = (
            (forward > min_forward_distance)
            & (forward * forward >= distance_sq * cos_angle_sq)
        )
        if not np.any(inside_cone):
            continue

        hit_voxel_ids = voxel_ids[inside_cone]
        hit_bin_ids = bins_batch[source_ids[inside_cone]]
        hit_bits = np.left_shift(np.uint64(1), hit_bin_ids.astype(np.uint64))

        np.bitwise_or.at(valid_direction_masks, hit_voxel_ids, hit_bits)

    direction_masks[voxel_valid_indices] = valid_direction_masks

    byte_view = direction_masks.view(np.uint8).reshape(n_voxels, 8)
    popcount_table = np.array(
        [int(value).bit_count() for value in range(256)],
        dtype=np.uint8,
    )
    scores = popcount_table[byte_view].sum(axis=1, dtype=np.uint8)
    result = np.column_stack((voxel, scores))

    if return_debug:
        return result, direction_masks, direction_centers
    return result


def filter_points_by_column_threshold(
    pointcloud,
    column_index,
    threshold,
):
    """Keep points whose selected column value is greater than or equal to threshold."""
    pointcloud = np.asarray(pointcloud)

    if pointcloud.ndim != 2:
        raise ValueError("pointcloud must be a 2D array.")
    if not isinstance(column_index, (int, np.integer)):
        raise TypeError("column_index must be an integer.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    n_columns = pointcloud.shape[1]
    normalized_index = int(column_index)
    if normalized_index < 0:
        normalized_index += n_columns

    if not 0 <= normalized_index < n_columns:
        raise IndexError(
            f"column_index {column_index} is out of range for "
            f"a point cloud with {n_columns} columns."
        )

    keep_mask = (
        np.isfinite(pointcloud[:, normalized_index])
        & (pointcloud[:, normalized_index] >= threshold)
    )
    filtered_pointcloud = pointcloud[keep_mask].copy()

    print(
        f"Column filter: kept {len(filtered_pointcloud)}/{len(pointcloud)} points "
        f"with column {column_index} >= {threshold}."
    )

    return filtered_pointcloud


def segment_voxels_by_nearest_score_center(
    voxel,
    scored_voxel,
    score_column=-1,
    min_center_score=2.0,
    min_peak_distance=4,
    merge_center_distance=4.0,
    gaussian_sigma=1.0,
    connectivity=1,
    color_seed=42,
    return_centers=False,
):
    """
    Find high-score centers and assign each voxel to its nearest center.

    Parameters
    ----------
    voxel : array, shape (N, >=3)
        Complete foreground voxel cloud. The first three columns are x,y,z.
    scored_voxel : array, shape (K, >=4)
        Scored center candidates. The first three columns are x,y,z and the
        selected score column contains the inward-direction score.
    score_column : int
        Column containing the score in scored_voxel; -1 means the last column.
    min_center_score : float
        Minimum smoothed score required for a local maximum to become a marker.
    min_peak_distance : int
        Neighborhood radius used to detect candidate score peaks.
    merge_center_distance : float
        After peak detection, suppress candidate centers closer than this
        Euclidean voxel distance within the same connected component.
    gaussian_sigma : float
        Gaussian smoothing sigma applied to the sparse score volume.
    connectivity : int
        1, 2, or 3 for 6-, 18-, or 26-neighbor 3D connectivity.
    color_seed : int
        Seed for deterministic, unique instance RGB colors.
    return_centers : bool
        If True, return (voxel_xyzrgb, centers_xyzrgb).

    Returns
    -------
    voxel_xyzrgb : array, shape (N, 6)
        Columns are x,y,z,r,g,b. Directions, scores, and instance IDs are
        used internally only and are not included in the output.
    """
    voxel = np.asarray(voxel)
    scored_voxel = np.asarray(scored_voxel)

    if voxel.ndim != 2 or voxel.shape[1] < 3:
        raise ValueError("voxel must be N x >=3 with x,y,z in the first columns.")
    if scored_voxel.ndim != 2 or scored_voxel.shape[1] < 4:
        raise ValueError(
            "scored_voxel must be K x >=4 with x,y,z and a score column."
        )
    if len(voxel) == 0:
        empty_result = np.empty((0, 6), dtype=np.float32)
        if return_centers:
            return empty_result, np.empty((0, 6), dtype=np.float32)
        return empty_result
    if min_peak_distance < 1:
        raise ValueError("min_peak_distance must be at least 1.")
    if merge_center_distance < 0:
        raise ValueError("merge_center_distance must be non-negative.")
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be non-negative.")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2, or 3.")

    n_score_columns = scored_voxel.shape[1]
    normalized_score_column = int(score_column)
    if normalized_score_column < 0:
        normalized_score_column += n_score_columns
    if not 0 <= normalized_score_column < n_score_columns:
        raise IndexError(
            f"score_column {score_column} is out of range for "
            f"scored_voxel with {n_score_columns} columns."
        )

    voxel_xyz_float = voxel[:, :3].astype(np.float64, copy=False)
    if not np.isfinite(voxel_xyz_float).all():
        raise ValueError("voxel coordinates contain NaN or inf.")

    voxel_xyz = np.rint(voxel_xyz_float).astype(np.int64)
    if not np.allclose(voxel_xyz_float, voxel_xyz, atol=1e-4):
        raise ValueError("voxel x,y,z coordinates must be integer voxel indices.")

    xyz_min = voxel_xyz.min(axis=0)
    xyz_max = voxel_xyz.max(axis=0)
    shape_xyz = xyz_max - xyz_min + 1
    shape_zyx = tuple(shape_xyz[[2, 1, 0]].astype(int))

    local_xyz = voxel_xyz - xyz_min
    vx = local_xyz[:, 0]
    vy = local_xyz[:, 1]
    vz = local_xyz[:, 2]

    voxel_mask = np.zeros(shape_zyx, dtype=bool)
    voxel_mask[vz, vy, vx] = True
    score_volume = np.zeros(shape_zyx, dtype=np.float32)

    if len(scored_voxel) > 0:
        scored_xyz_float = scored_voxel[:, :3].astype(np.float64, copy=False)
        scored_values = scored_voxel[:, normalized_score_column].astype(
            np.float32,
            copy=False,
        )
        scored_valid = (
            np.isfinite(scored_xyz_float).all(axis=1)
            & np.isfinite(scored_values)
        )
        scored_xyz = np.rint(scored_xyz_float[scored_valid]).astype(np.int64)
        scored_values = scored_values[scored_valid]
        scored_local_xyz = scored_xyz - xyz_min

        in_bounds = (
            (scored_local_xyz[:, 0] >= 0)
            & (scored_local_xyz[:, 0] < shape_xyz[0])
            & (scored_local_xyz[:, 1] >= 0)
            & (scored_local_xyz[:, 1] < shape_xyz[1])
            & (scored_local_xyz[:, 2] >= 0)
            & (scored_local_xyz[:, 2] < shape_xyz[2])
        )
        scored_local_xyz = scored_local_xyz[in_bounds]
        scored_values = scored_values[in_bounds]

        if len(scored_local_xyz) > 0:
            sx = scored_local_xyz[:, 0]
            sy = scored_local_xyz[:, 1]
            sz = scored_local_xyz[:, 2]
            inside_mask = voxel_mask[sz, sy, sx]
            sx, sy, sz = sx[inside_mask], sy[inside_mask], sz[inside_mask]
            scored_values = scored_values[inside_mask]
            np.maximum.at(score_volume, (sz, sy, sx), scored_values)

    if gaussian_sigma > 0:
        smoothed_score = ndi.gaussian_filter(
            score_volume,
            sigma=gaussian_sigma,
            mode="constant",
            cval=0.0,
        )
    else:
        smoothed_score = score_volume.copy()
    smoothed_score[~voxel_mask] = 0.0

    center_candidate_mask = (
        voxel_mask
        & (score_volume >= float(min_center_score))
    )
    local_maximum = smoothed_score == ndi.maximum_filter(
        smoothed_score,
        size=2 * int(min_peak_distance) + 1,
        mode="constant",
        cval=-np.inf,
    )
    peak_mask = center_candidate_mask & local_maximum

    structure = ndi.generate_binary_structure(3, connectivity)
    peak_plateaus, n_peak_plateaus = ndi.label(peak_mask, structure=structure)
    if n_peak_plateaus > 0:
        peak_positions = ndi.maximum_position(
            score_volume,
            labels=peak_plateaus,
            index=np.arange(1, n_peak_plateaus + 1),
        )
        peak_zyx = np.asarray(peak_positions, dtype=np.int64).reshape(-1, 3)
    else:
        peak_zyx = np.empty((0, 3), dtype=np.int64)

    component_labels, n_components = ndi.label(voxel_mask, structure=structure)

    if merge_center_distance > 0 and len(peak_zyx) > 1:
        peak_scores = score_volume[
            peak_zyx[:, 0],
            peak_zyx[:, 1],
            peak_zyx[:, 2],
        ]
        peak_smoothed_scores = smoothed_score[
            peak_zyx[:, 0],
            peak_zyx[:, 1],
            peak_zyx[:, 2],
        ]
        peak_component_ids = component_labels[
            peak_zyx[:, 0],
            peak_zyx[:, 1],
            peak_zyx[:, 2],
        ]

        peak_order = sorted(
            range(len(peak_zyx)),
            key=lambda index: (
                float(peak_scores[index]),
                float(peak_smoothed_scores[index]),
            ),
            reverse=True,
        )
        kept_peak_indices = []

        for candidate_index in peak_order:
            candidate_component = peak_component_ids[candidate_index]
            same_component_kept = [
                kept_index
                for kept_index in kept_peak_indices
                if peak_component_ids[kept_index] == candidate_component
            ]

            if same_component_kept:
                offsets = (
                    peak_zyx[same_component_kept]
                    - peak_zyx[candidate_index]
                )
                distances = np.linalg.norm(offsets, axis=1)
                if np.any(distances <= merge_center_distance):
                    continue

            kept_peak_indices.append(candidate_index)

        n_merged_centers = len(peak_zyx) - len(kept_peak_indices)
        peak_zyx = peak_zyx[kept_peak_indices]
        print(
            f"Center merge: removed {n_merged_centers} candidates within "
            f"distance <= {merge_center_distance}."
        )

    markers = np.zeros(shape_zyx, dtype=np.int32)

    next_marker_id = 1
    for z, y, x in peak_zyx:
        if markers[z, y, x] == 0:
            markers[z, y, x] = next_marker_id
            next_marker_id += 1

    distance_volume = ndi.distance_transform_edt(voxel_mask).astype(np.float32)

    components_with_markers = set(
        component_labels[markers > 0].astype(int).tolist()
    )
    for component_id in range(1, n_components + 1):
        if component_id in components_with_markers:
            continue

        component_mask = component_labels == component_id
        component_scores = np.where(component_mask, smoothed_score, -np.inf)
        if np.max(component_scores) > 0:
            fallback_flat_index = int(np.argmax(component_scores))
        else:
            fallback_distance = np.where(component_mask, distance_volume, -1.0)
            fallback_flat_index = int(np.argmax(fallback_distance))

        fallback_zyx = np.unravel_index(fallback_flat_index, shape_zyx)
        markers[fallback_zyx] = next_marker_id
        next_marker_id += 1

    marker_zyx = np.argwhere(markers > 0)
    marker_ids = markers[
        marker_zyx[:, 0],
        marker_zyx[:, 1],
        marker_zyx[:, 2],
    ]
    marker_xyz = marker_zyx[:, [2, 1, 0]] + xyz_min
    marker_component_ids = component_labels[
        marker_zyx[:, 0],
        marker_zyx[:, 1],
        marker_zyx[:, 2],
    ]
    voxel_component_ids = component_labels[vz, vy, vx]
    instance_ids = np.zeros(len(voxel), dtype=np.int32)

    for component_id in range(1, n_components + 1):
        voxel_row_indices = np.flatnonzero(voxel_component_ids == component_id)
        center_row_indices = np.flatnonzero(marker_component_ids == component_id)

        if len(voxel_row_indices) == 0:
            continue
        if len(center_row_indices) == 0:
            raise RuntimeError(
                f"Connected component {component_id} has no center marker."
            )

        center_tree = cKDTree(marker_xyz[center_row_indices])
        _, nearest_center_indices = center_tree.query(
            voxel_xyz_float[voxel_row_indices],
            k=1,
            workers=-1,
        )
        instance_ids[voxel_row_indices] = marker_ids[
            center_row_indices[np.asarray(nearest_center_indices, dtype=np.int64)]
        ]

    unique_instance_ids = np.unique(instance_ids)
    unique_instance_ids = unique_instance_ids[unique_instance_ids > 0]
    max_unique_colors = (256 - 40) ** 3
    if len(unique_instance_ids) > max_unique_colors:
        raise ValueError(
            f"Too many instances ({len(unique_instance_ids)}) for the "
            f"available unique RGB colors ({max_unique_colors})."
        )

    rng = np.random.default_rng(color_seed)
    rgb_palette = np.zeros((int(instance_ids.max()) + 1, 3), dtype=np.uint8)
    used_colors = set()

    for instance_id in unique_instance_ids:
        while True:
            color = tuple(
                int(value)
                for value in rng.integers(40, 256, size=3)
            )
            if color not in used_colors:
                used_colors.add(color)
                rgb_palette[int(instance_id)] = color
                break

    marker_rgb = rgb_palette[marker_ids]
    centers = np.column_stack((marker_xyz, marker_rgb))
    centers = centers[np.argsort(marker_ids)]

    voxel_rgb = rgb_palette[instance_ids]
    voxel_xyzrgb = np.column_stack((
        voxel[:, :3],
        voxel_rgb.astype(np.float32),
    )).astype(np.float32)

    print(
        f"Nearest-center assignment: found {len(centers)} centers and "
        f"{int(instance_ids.max())} instances."
    )

    if return_centers:
        return voxel_xyzrgb, centers
    return voxel_xyzrgb


def segment_pcd_by_voxel_xyzrgb(
    pcd,
    voxel_xyzrgb,
    max_assign_distance=None,
    batch_size=300000,
    workers=-1,
):
    """
    Assign each PCD surface point the RGB of its nearest segmented voxel.

    Parameters
    ----------
    pcd : array, shape (N, >=6)
        Columns 0:6 are x,y,z,nx,ny,nz.
    voxel_xyzrgb : array, shape (M, >=6)
        Columns 0:6 are x,y,z,r,g,b.
    max_assign_distance : float or None
        Optional maximum nearest-voxel distance. Points farther away are black.
    batch_size : int
        Number of PCD points queried per KD-tree batch.
    workers : int
        Worker count passed to scipy.spatial.cKDTree.query.

    Returns
    -------
    pcd_xyznormal_rgb : array, shape (N, 9)
        Columns are x,y,z,nx,ny,nz,r,g,b.
    """
    pcd = np.asarray(pcd)
    voxel_xyzrgb = np.asarray(voxel_xyzrgb)

    if pcd.ndim != 2 or pcd.shape[1] < 6:
        raise ValueError("pcd must be N x >=6: x,y,z,nx,ny,nz.")
    if voxel_xyzrgb.ndim != 2 or voxel_xyzrgb.shape[1] < 6:
        raise ValueError("voxel_xyzrgb must be M x >=6: x,y,z,r,g,b.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if max_assign_distance is not None and max_assign_distance < 0:
        raise ValueError("max_assign_distance must be non-negative or None.")

    pcd_xyz = pcd[:, :3].astype(np.float32, copy=False)
    pcd_normals = pcd[:, 3:6].astype(np.float32, copy=False)
    voxel_xyz = voxel_xyzrgb[:, :3].astype(np.float32, copy=False)
    voxel_rgb = np.rint(voxel_xyzrgb[:, 3:6]).astype(np.int32)

    valid_voxels = (
        np.isfinite(voxel_xyz).all(axis=1)
        & np.isfinite(voxel_rgb).all(axis=1)
        & np.any(voxel_rgb != 0, axis=1)
    )
    if not np.any(valid_voxels):
        raise ValueError("voxel_xyzrgb contains no valid colored voxels.")

    valid_pcd = (
        np.isfinite(pcd_xyz).all(axis=1)
        & np.isfinite(pcd_normals).all(axis=1)
    )
    pcd_rgb = np.zeros((len(pcd), 3), dtype=np.uint8)
    valid_pcd_indices = np.flatnonzero(valid_pcd)

    voxel_tree = cKDTree(voxel_xyz[valid_voxels])
    valid_voxel_rgb = np.clip(voxel_rgb[valid_voxels], 0, 255).astype(np.uint8)

    for start in tqdm(
        range(0, len(valid_pcd_indices), batch_size),
        desc="Coloring PCD from segmented voxels",
    ):
        stop = min(start + batch_size, len(valid_pcd_indices))
        batch_indices = valid_pcd_indices[start:stop]
        distances, nearest_indices = voxel_tree.query(
            pcd_xyz[batch_indices],
            k=1,
            workers=workers,
        )

        batch_rgb = valid_voxel_rgb[
            np.asarray(nearest_indices, dtype=np.int64)
        ].copy()
        if max_assign_distance is not None:
            batch_rgb[np.asarray(distances) > max_assign_distance] = 0

        pcd_rgb[batch_indices] = batch_rgb

    pcd_xyznormal_rgb = np.column_stack((
        pcd_xyz,
        pcd_normals,
        pcd_rgb.astype(np.float32),
    )).astype(np.float32)

    print(
        f"PCD color assignment: colored {np.sum(np.any(pcd_rgb != 0, axis=1))}/"
        f"{len(pcd_rgb)} surface points."
    )

    return pcd_xyznormal_rgb


def compute_surface_area_from_voxels_xyz(coords_xyz_idx, voxel_size_zyx=(1, 1, 1)):
    z_um, y_um, x_um = voxel_size_zyx

    area_x_face = y_um * z_um
    area_y_face = x_um * z_um
    area_z_face = x_um * y_um

    voxel_set = set(map(tuple, coords_xyz_idx.astype(np.int64)))

    surface_area = 0.0

    neighbor_defs = [
        ((1, 0, 0), area_x_face),
        ((-1, 0, 0), area_x_face),
        ((0, 1, 0), area_y_face),
        ((0, -1, 0), area_y_face),
        ((0, 0, 1), area_z_face),
        ((0, 0, -1), area_z_face),
    ]

    for x, y, z in voxel_set:
        for (dx, dy, dz), face_area in neighbor_defs:
            nb = (x + dx, y + dy, z + dz)

            if nb not in voxel_set:
                surface_area += face_area

    return float(surface_area)


def compute_instance_morphology_from_voxel_xyzrgb(
    voxel_xyzrgb,
    voxel_size_zyx=(1, 1, 1),
    coords_are_physical=False,
    ignore_background_color=True,
    background_color=(0, 0, 0),
    compute_surface_area=True,
    intensity_image=None,
    sort_by_size=True,
):
    voxel_xyzrgb = np.asarray(voxel_xyzrgb)

    if voxel_xyzrgb.ndim != 2 or voxel_xyzrgb.shape[1] < 6:
        raise ValueError("voxel_xyzrgb must be N x 6: x,y,z,r,g,b")

    xyz = voxel_xyzrgb[:, :3].astype(np.float32)
    rgb = np.rint(voxel_xyzrgb[:, 3:6]).astype(np.int32)

    z_um, y_um, x_um = voxel_size_zyx
    scale_xyz = np.asarray([x_um, y_um, z_um], dtype=np.float32)

    if coords_are_physical:
        coords_idx = np.zeros_like(xyz, dtype=np.int64)
        coords_idx[:, 0] = np.rint(xyz[:, 0] / x_um).astype(np.int64)
        coords_idx[:, 1] = np.rint(xyz[:, 1] / y_um).astype(np.int64)
        coords_idx[:, 2] = np.rint(xyz[:, 2] / z_um).astype(np.int64)

        coords_phys = xyz.astype(np.float32)
    else:
        coords_idx = np.rint(xyz).astype(np.int64)
        coords_phys = coords_idx.astype(np.float32) * scale_xyz

    if ignore_background_color:
        bg = np.asarray(background_color, dtype=np.int32)
        keep = ~np.all(rgb == bg[None, :], axis=1)

        coords_idx = coords_idx[keep]
        coords_phys = coords_phys[keep]
        rgb = rgb[keep]

    if len(coords_idx) == 0:
        raise ValueError("No foreground colored voxels found after filtering.")

    unique_rgb, inverse = np.unique(rgb, axis=0, return_inverse=True)

    groups = []

    for color_idx, color in enumerate(unique_rgb):
        idx = np.where(inverse == color_idx)[0]
        groups.append((color_idx, color, idx, len(idx)))

    if sort_by_size:
        groups = sorted(groups, key=lambda x: x[3], reverse=True)

    voxel_volume = float(x_um * y_um * z_um)

    results = []

    for new_instance_id, (color_idx, color, idx, n_voxels) in enumerate(groups, start=1):

        coords_i_idx = coords_idx[idx]
        coords_i_phys = coords_phys[idx]

        r, g, b = color.tolist()

        volume = float(n_voxels * voxel_volume)

        centroid = coords_i_phys.mean(axis=0)

        bbox_min = coords_i_phys.min(axis=0)
        bbox_max = coords_i_phys.max(axis=0)

        bbox_size = (bbox_max - bbox_min) + scale_xyz

        bbox_size_x = float(bbox_size[0])
        bbox_size_y = float(bbox_size[1])
        bbox_size_z = float(bbox_size[2])

        bbox_volume = float(bbox_size_x * bbox_size_y * bbox_size_z)

        if bbox_volume > 1e-8:
            extent_volume_bbox = float(volume / bbox_volume)
        else:
            extent_volume_bbox = np.nan

        bbox_aspect_ratio_xy = float(bbox_size_x / (bbox_size_y + 1e-8))
        bbox_aspect_ratio_xz = float(bbox_size_x / (bbox_size_z + 1e-8))
        bbox_aspect_ratio_yz = float(bbox_size_y / (bbox_size_z + 1e-8))

        if n_voxels >= 3:
            centered = coords_i_phys - centroid[None, :]

            try:
                cov = np.cov(centered.T)

                eigvals, eigvecs = np.linalg.eigh(cov)

                order = np.argsort(eigvals)[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]

                eigvals = np.clip(eigvals, 0, None)

                proj = centered @ eigvecs

                axis_lengths = proj.max(axis=0) - proj.min(axis=0)

                major_axis_length = float(axis_lengths[0])
                middle_axis_length = float(axis_lengths[1])
                minor_axis_length = float(axis_lengths[2])

                eig_major = float(eigvals[0])
                eig_middle = float(eigvals[1])
                eig_minor = float(eigvals[2])

                aspect_ratio_major_minor = float(
                    major_axis_length / (minor_axis_length + 1e-8)
                )

                mean_width_pca = float((middle_axis_length + minor_axis_length) / 2.0)

                length_width_ratio = float(
                    major_axis_length / (mean_width_pca + 1e-8)
                )

                elongation_major_middle = float(
                    major_axis_length / (middle_axis_length + 1e-8)
                )

                flatness_middle_minor = float(
                    middle_axis_length / (minor_axis_length + 1e-8)
                )

                radius_of_gyration = float(
                    np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
                )

                lambda1 = eig_major
                lambda2 = eig_middle
                lambda3 = eig_minor

                pca_linearity = float((lambda1 - lambda2) / (lambda1 + 1e-8))
                pca_planarity = float((lambda2 - lambda3) / (lambda1 + 1e-8))
                pca_sphericity_like = float(lambda3 / (lambda1 + 1e-8))
                pca_anisotropy = float((lambda1 - lambda3) / (lambda1 + 1e-8))

            except np.linalg.LinAlgError:
                major_axis_length = np.nan
                middle_axis_length = np.nan
                minor_axis_length = np.nan
                mean_width_pca = np.nan

                eig_major = np.nan
                eig_middle = np.nan
                eig_minor = np.nan

                aspect_ratio_major_minor = np.nan
                length_width_ratio = np.nan
                elongation_major_middle = np.nan
                flatness_middle_minor = np.nan

                radius_of_gyration = np.nan

                pca_linearity = np.nan
                pca_planarity = np.nan
                pca_sphericity_like = np.nan
                pca_anisotropy = np.nan
        else:
            major_axis_length = np.nan
            middle_axis_length = np.nan
            minor_axis_length = np.nan
            mean_width_pca = np.nan

            eig_major = np.nan
            eig_middle = np.nan
            eig_minor = np.nan

            aspect_ratio_major_minor = np.nan
            length_width_ratio = np.nan
            elongation_major_middle = np.nan
            flatness_middle_minor = np.nan

            radius_of_gyration = np.nan

            pca_linearity = np.nan
            pca_planarity = np.nan
            pca_sphericity_like = np.nan
            pca_anisotropy = np.nan

        if volume > 0:
            equivalent_sphere_diameter = float((6.0 * volume / np.pi) ** (1.0 / 3.0))
            equivalent_sphere_radius = float(equivalent_sphere_diameter / 2.0)
        else:
            equivalent_sphere_diameter = np.nan
            equivalent_sphere_radius = np.nan

        if compute_surface_area:
            surface_area = compute_surface_area_from_voxels_xyz(
                coords_i_idx,
                voxel_size_zyx=voxel_size_zyx,
            )
        else:
            surface_area = np.nan

        if compute_surface_area and surface_area > 1e-8 and volume > 0:
            sphericity = float(
                (np.pi ** (1.0 / 3.0))
                * ((6.0 * volume) ** (2.0 / 3.0))
                / surface_area
            )

            compactness_3d = float(
                36.0 * np.pi * (volume ** 2) / ((surface_area ** 3) + 1e-8)
            )

            surface_volume_ratio = float(surface_area / (volume + 1e-8))
        else:
            sphericity = np.nan
            compactness_3d = np.nan
            surface_volume_ratio = np.nan

        row = {
            "instance_id": int(new_instance_id),

            "r": int(r),
            "g": int(g),
            "b": int(b),

            "voxel_count": int(n_voxels),
            "volume": float(volume),
            "equivalent_sphere_diameter": float(equivalent_sphere_diameter),
            "equivalent_sphere_radius": float(equivalent_sphere_radius),

            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),

            "bbox_min_x": float(bbox_min[0]),
            "bbox_min_y": float(bbox_min[1]),
            "bbox_min_z": float(bbox_min[2]),

            "bbox_max_x": float(bbox_max[0]),
            "bbox_max_y": float(bbox_max[1]),
            "bbox_max_z": float(bbox_max[2]),

            "bbox_size_x": float(bbox_size_x),
            "bbox_size_y": float(bbox_size_y),
            "bbox_size_z": float(bbox_size_z),
            "bbox_volume": float(bbox_volume),
            "extent_volume_bbox": float(extent_volume_bbox),

            "bbox_aspect_ratio_x_over_y": float(bbox_aspect_ratio_xy),
            "bbox_aspect_ratio_x_over_z": float(bbox_aspect_ratio_xz),
            "bbox_aspect_ratio_y_over_z": float(bbox_aspect_ratio_yz),

            "major_axis_length_pca": float(major_axis_length),
            "middle_axis_length_pca": float(middle_axis_length),
            "minor_axis_length_pca": float(minor_axis_length),
            "mean_width_pca": float(mean_width_pca),

            "aspect_ratio_major_minor": float(aspect_ratio_major_minor),
            "length_width_ratio": float(length_width_ratio),
            "elongation_major_middle": float(elongation_major_middle),
            "flatness_middle_minor": float(flatness_middle_minor),

            "pca_eig_major": float(eig_major),
            "pca_eig_middle": float(eig_middle),
            "pca_eig_minor": float(eig_minor),
            "pca_linearity": float(pca_linearity),
            "pca_planarity": float(pca_planarity),
            "pca_sphericity_like": float(pca_sphericity_like),
            "pca_anisotropy": float(pca_anisotropy),

            "radius_of_gyration": float(radius_of_gyration),

            "surface_area_voxel_faces": float(surface_area),
            "surface_volume_ratio": float(surface_volume_ratio),
            "sphericity": float(sphericity),
            "compactness_3d": float(compactness_3d),
        }

        if intensity_image is not None:
            intensity_image = np.asarray(intensity_image)

            x_idx = coords_i_idx[:, 0]
            y_idx = coords_i_idx[:, 1]
            z_idx = coords_i_idx[:, 2]

            valid = (
                (z_idx >= 0) & (z_idx < intensity_image.shape[0]) &
                (y_idx >= 0) & (y_idx < intensity_image.shape[1]) &
                (x_idx >= 0) & (x_idx < intensity_image.shape[2])
            )

            if np.any(valid):
                vals = intensity_image[
                    z_idx[valid],
                    y_idx[valid],
                    x_idx[valid]
                ]

                row["intensity_mean"] = float(np.mean(vals))
                row["intensity_median"] = float(np.median(vals))
                row["intensity_max"] = float(np.max(vals))
                row["intensity_min"] = float(np.min(vals))
                row["intensity_sum"] = float(np.sum(vals))
                row["intensity_std"] = float(np.std(vals))
            else:
                row["intensity_mean"] = np.nan
                row["intensity_median"] = np.nan
                row["intensity_max"] = np.nan
                row["intensity_min"] = np.nan
                row["intensity_sum"] = np.nan
                row["intensity_std"] = np.nan

        results.append(row)

    df = pd.DataFrame(results)

    print("Instances:", len(df))
    print("Total colored voxels:", int(df["voxel_count"].sum()))

    return df


def collect_input_base_names(input_path):
    """
    Collect image base names from raw and/or existing instance files.

    This keeps the original main-loop rhythm, but allows morphology to run
    even when only *_voxel_instance.txt or *_pcd_instance.txt already exists.
    """
    input_path = Path(input_path)

    suffixes = [
        "_voxel_instance.txt",
        "_pcd_instance.txt",
        "_pcd.txt",
        "_voxel.txt",
    ]

    base_names = []
    seen = set()

    for file_name in os.listdir(input_path):
        lower_name = file_name.lower()

        for suffix in suffixes:
            if lower_name.endswith(suffix):
                base_name = file_name[: -len(suffix)]
                if base_name not in seen:
                    seen.add(base_name)
                    base_names.append(base_name)
                break

    return base_names


def color_voxels_from_existing_pcd_instance(
    voxel,
    pcd_instance,
    max_assign_distance=None,
    batch_size=300000,
    workers=-1,
):
    """
    Rebuild voxel_instance from an existing colored pcd_instance and raw voxel.txt.

    This does NOT rerun instance segmentation. It only transfers the existing
    instance RGB labels from the nearest colored PCD surface point to each raw
    foreground voxel from *_voxel.txt.

    Parameters
    ----------
    voxel : array, shape (N, >=3)
        Raw foreground voxel point cloud from *_voxel.txt.
    pcd_instance : array, shape (M, >=6 or >=9)
        Existing colored PCD instance file.
        Accepted formats:
            x,y,z,r,g,b
            x,y,z,nx,ny,nz,r,g,b
    max_assign_distance : float or None
        If not None, voxels farther than this distance are assigned background RGB.
    batch_size : int
        KD-tree query batch size.
    workers : int
        Worker count for cKDTree.query.

    Returns
    -------
    voxel_xyzrgb : array, shape (N, 6)
        x,y,z,r,g,b; coordinates come directly from *_voxel.txt.
    """
    voxel = np.asarray(voxel)
    pcd_instance = np.asarray(pcd_instance)

    if voxel.ndim != 2 or voxel.shape[1] < 3:
        raise ValueError("voxel must be N x >=3 with x,y,z in the first columns.")
    if pcd_instance.ndim != 2 or pcd_instance.shape[1] < 6:
        raise ValueError(
            "pcd_instance must be x,y,z,r,g,b or x,y,z,nx,ny,nz,r,g,b."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if max_assign_distance is not None and max_assign_distance < 0:
        raise ValueError("max_assign_distance must be non-negative or None.")

    voxel_xyz = voxel[:, :3].astype(np.float32, copy=False)
    pcd_xyz = pcd_instance[:, :3].astype(np.float32, copy=False)

    # Your current *_pcd_instance.txt is x,y,z,nx,ny,nz,r,g,b.
    # This also supports a simpler x,y,z,r,g,b file.
    if pcd_instance.shape[1] >= 9:
        pcd_rgb = pcd_instance[:, -3:]
    else:
        pcd_rgb = pcd_instance[:, 3:6]

    pcd_rgb = np.rint(pcd_rgb).astype(np.int32)

    valid_pcd = (
        np.isfinite(pcd_xyz).all(axis=1)
        & np.isfinite(pcd_rgb).all(axis=1)
        & np.any(pcd_rgb != 0, axis=1)
    )
    valid_voxel = np.isfinite(voxel_xyz).all(axis=1)

    if not np.any(valid_pcd):
        raise ValueError("Existing pcd_instance contains no valid colored points.")
    if not np.any(valid_voxel):
        raise ValueError("voxel contains no valid xyz coordinates.")

    pcd_tree = cKDTree(pcd_xyz[valid_pcd])
    valid_pcd_rgb = np.clip(pcd_rgb[valid_pcd], 0, 255).astype(np.uint8)

    voxel_rgb = np.zeros((len(voxel), 3), dtype=np.uint8)
    valid_voxel_indices = np.flatnonzero(valid_voxel)

    for start in tqdm(
        range(0, len(valid_voxel_indices), batch_size),
        desc="Coloring voxels from existing PCD instance",
    ):
        stop = min(start + batch_size, len(valid_voxel_indices))
        batch_indices = valid_voxel_indices[start:stop]

        distances, nearest_indices = pcd_tree.query(
            voxel_xyz[batch_indices],
            k=1,
            workers=workers,
        )

        batch_rgb = valid_pcd_rgb[np.asarray(nearest_indices, dtype=np.int64)].copy()

        if max_assign_distance is not None:
            batch_rgb[np.asarray(distances) > max_assign_distance] = 0

        voxel_rgb[batch_indices] = batch_rgb

    voxel_xyzrgb = np.column_stack([
        voxel_xyz,
        voxel_rgb.astype(np.float32),
    ]).astype(np.float32)

    print(
        f"Voxel coloring from existing PCD instance: colored "
        f"{np.sum(np.any(voxel_rgb != 0, axis=1))}/{len(voxel_rgb)} voxels."
    )

    return voxel_xyzrgb


def add_point_intensity_stats_from_voxel_txt(
    morphology_df,
    voxel_xyzrgb,
    raw_voxel,
    intensity_column=3,
    background_color=(0, 0, 0),
):
    """
    Optional: add intensity statistics when *_voxel.txt has an intensity column.

    This assumes voxel_xyzrgb and raw_voxel use the same voxel coordinates.
    It matches by x,y,z, so it still works if row order changed.
    """
    morphology_df = morphology_df.copy()

    raw_voxel = np.asarray(raw_voxel)
    voxel_xyzrgb = np.asarray(voxel_xyzrgb)

    if raw_voxel.ndim != 2 or raw_voxel.shape[1] <= intensity_column:
        return morphology_df
    if voxel_xyzrgb.ndim != 2 or voxel_xyzrgb.shape[1] < 6:
        return morphology_df

    raw_xyz = np.rint(raw_voxel[:, :3]).astype(np.int64)
    raw_intensity = raw_voxel[:, intensity_column].astype(np.float32)

    intensity_map = {
        tuple(xyz): val
        for xyz, val in zip(raw_xyz, raw_intensity)
        if np.isfinite(val)
    }

    inst_xyz = np.rint(voxel_xyzrgb[:, :3]).astype(np.int64)
    inst_rgb = np.rint(voxel_xyzrgb[:, 3:6]).astype(np.int32)

    bg = np.asarray(background_color, dtype=np.int32)
    keep = ~np.all(inst_rgb == bg[None, :], axis=1)

    inst_xyz = inst_xyz[keep]
    inst_rgb = inst_rgb[keep]

    if len(inst_xyz) == 0:
        return morphology_df

    values = np.array(
        [intensity_map.get(tuple(xyz), np.nan) for xyz in inst_xyz],
        dtype=np.float32,
    )

    unique_rgb = morphology_df[["r", "g", "b"]].to_numpy(dtype=np.int32)

    stats_rows = []
    for color in unique_rgb:
        color_keep = np.all(inst_rgb == color[None, :], axis=1)
        vals = values[color_keep]
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            stats_rows.append({
                "voxel_txt_intensity_mean": np.nan,
                "voxel_txt_intensity_median": np.nan,
                "voxel_txt_intensity_max": np.nan,
                "voxel_txt_intensity_min": np.nan,
                "voxel_txt_intensity_sum": np.nan,
                "voxel_txt_intensity_std": np.nan,
            })
        else:
            stats_rows.append({
                "voxel_txt_intensity_mean": float(np.mean(vals)),
                "voxel_txt_intensity_median": float(np.median(vals)),
                "voxel_txt_intensity_max": float(np.max(vals)),
                "voxel_txt_intensity_min": float(np.min(vals)),
                "voxel_txt_intensity_sum": float(np.sum(vals)),
                "voxel_txt_intensity_std": float(np.std(vals)),
            })

    stats_df = pd.DataFrame(stats_rows)
    morphology_df = pd.concat(
        [morphology_df.reset_index(drop=True), stats_df.reset_index(drop=True)],
        axis=1,
    )

    return morphology_df


def summarize_instance_morphology_to_cell_mean(
    morph_df,
    image_name=None,
    dataset_name=None,
    organelle_name=None,
    morphology_source=None,
):
    """
    Convert per-instance morphology into one per-cell / per-image mean row.

    The instance-level table is used only internally. The output row summarizes
    each cell by averaging morphology metrics across all segmented instances.
    """
    if morph_df is None or len(morph_df) == 0:
        raise ValueError("morph_df is empty; cannot compute cell-level morphology mean.")

    morph_df = morph_df.copy()

    # These columns identify instances or spatial positions. They are not useful
    # as averaged morphology features, so they are excluded from mean aggregation.
    exclude_numeric_cols = {
        "instance_id",
        "r", "g", "b",
        "centroid_x", "centroid_y", "centroid_z",
        "bbox_min_x", "bbox_min_y", "bbox_min_z",
        "bbox_max_x", "bbox_max_y", "bbox_max_z",
    }

    numeric_cols = morph_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [
        col for col in numeric_cols
        if col not in exclude_numeric_cols
    ]

    row = {
        "image_name": image_name,
        "dataset_name": dataset_name,
        "organelle": organelle_name,
        "morphology_source": morphology_source,

        # Keep these as cell-level descriptors.
        "instance_count": int(len(morph_df)),
        "total_voxel_count": float(morph_df["voxel_count"].sum())
            if "voxel_count" in morph_df.columns else np.nan,
        "total_volume": float(morph_df["volume"].sum())
            if "volume" in morph_df.columns else np.nan,
    }

    # Mean morphology per cell: each instance contributes one value.
    for col in metric_cols:
        values = pd.to_numeric(morph_df[col], errors="coerce")
        row[f"mean_{col}"] = float(values.mean(skipna=True))

    return pd.DataFrame([row])


def compute_cell_mean_morphology_from_voxel_instance(
    voxel_xyzrgb,
    voxel_size_zyx=(1, 1, 1),
    coords_are_physical=False,
    image_name=None,
    dataset_name=None,
    organelle_name=None,
    morphology_source=None,
    raw_voxel=None,
    save_cell_mean_path=None,
):
    """
    Compute morphology from colored voxel instances, average across instances,
    and return one row per cell / image.

    No per-instance CSV is saved. The per-instance morphology table is created
    only in memory so that each cell can be summarized as mean morphology.
    """
    morph_instance_df = compute_instance_morphology_from_voxel_xyzrgb(
        voxel_xyzrgb=voxel_xyzrgb,
        voxel_size_zyx=voxel_size_zyx,
        coords_are_physical=coords_are_physical,
        ignore_background_color=True,
        background_color=(0, 0, 0),
        compute_surface_area=True,
        intensity_image=None,
        sort_by_size=True,
    )

    # Optional intensity stats from *_voxel.txt if it contains an intensity column.
    if raw_voxel is not None:
        morph_instance_df = add_point_intensity_stats_from_voxel_txt(
            morphology_df=morph_instance_df,
            voxel_xyzrgb=voxel_xyzrgb,
            raw_voxel=raw_voxel,
            intensity_column=3,
            background_color=(0, 0, 0),
        )

    cell_mean_df = summarize_instance_morphology_to_cell_mean(
        morph_df=morph_instance_df,
        image_name=image_name,
        dataset_name=dataset_name,
        organelle_name=organelle_name,
        morphology_source=morphology_source,
    )

    if save_cell_mean_path is not None:
        cell_mean_df.to_csv(save_cell_mean_path, index=False)
        print(f"[SAVED CELL-MEAN MORPHOLOGY] {save_cell_mean_path}")

    print(
        f"[CELL MEAN MORPHOLOGY] {image_name}: "
        f"{int(cell_mean_df.loc[0, 'instance_count'])} instances averaged."
    )

    return cell_mean_df


@dataclass
class FrameRecord:
    frame: int
    frame_name: str
    voxel_path: Optional[Path]
    pcd_path: Optional[Path]


@dataclass
class LocalInstance:
    frame: int
    frame_name: str
    local_instance_id: int
    centroid_raw: np.ndarray
    centroid_um: np.ndarray
    size_count: int
    voxel_rgb: Optional[Tuple[int, int, int]] = None
    pcd_rgb: Optional[Tuple[int, int, int]] = None
    voxel_points_raw: Optional[np.ndarray] = None
    pcd_points_raw: Optional[np.ndarray] = None
    pcd_normals_raw: Optional[np.ndarray] = None
    shape_points_centered_um: Optional[np.ndarray] = None


@dataclass
class KalmanState:
    x: np.ndarray       # [x, y, z, vx, vy, vz]
    P: np.ndarray


@dataclass
class TrackState:
    track_id: int
    last_frame: int
    last_local_instance_id: int
    kalman: KalmanState
    last_position_um: np.ndarray
    last_size_count: int
    last_shape_centered_um: Optional[np.ndarray]
    observed_positions: List[np.ndarray] = field(default_factory=list)
    observed_frames: List[int] = field(default_factory=list)
    missed_frames: int = 0
    retired: bool = False


def natural_key(text: str) -> List[object]:
    return [int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", str(text))]


def frame_number_from_name(name: str) -> Optional[int]:
    stem = Path(name).stem
    explicit = re.findall(
        r"(?:^|[_\-])(frame|time|t)[_\-]?(\d+)(?:$|[_\-])",
        stem, flags=re.IGNORECASE,
    )
    if explicit:
        return int(explicit[-1][1])
    numbers = re.findall(r"\d+", stem)
    return int(numbers[-1]) if numbers else None


def save_numeric_txt(path: Path, array: np.ndarray, fmt: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array, delimiter=",", fmt=list(fmt))


def rgb_tuple(values: Sequence[float]) -> Tuple[int, int, int]:
    a = np.rint(np.asarray(values)).astype(int)
    return int(a[0]), int(a[1]), int(a[2])


def deterministic_track_rgb(track_id: int) -> Tuple[int, int, int]:
    hue = (int(track_id) * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return tuple(int(round(v * 255)) for v in rgb)


def xyz_scale_um() -> np.ndarray:
    z_um, y_um, x_um = VOXEL_SIZE_ZYX
    return np.array([x_um, y_um, z_um], dtype=float)


def to_physical(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    if INPUT_COORDS_ARE_PHYSICAL:
        return xyz.copy()
    return xyz * xyz_scale_um()[None, :]


def group_indices_by_rgb(arr: np.ndarray, rgb_slice: Tuple[int, int]) -> Dict[Tuple[int, int, int], np.ndarray]:
    rgb = np.rint(arr[:, rgb_slice[0]:rgb_slice[1]]).astype(int)
    unique, inverse = np.unique(rgb, axis=0, return_inverse=True)
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for i, color in enumerate(unique):
        key = rgb_tuple(color)
        if key == (0, 0, 0):
            continue
        out[key] = np.flatnonzero(inverse == i)
    return out


def frame_name_from_instance_filename(filename: str) -> Optional[str]:
    lower = filename.lower()
    for suffix in sorted(ALL_INSTANCE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix.lower()):
            return filename[:-len(suffix)]
    return None


def first_existing(seg_dir: Path, frame_name: str, suffixes: Sequence[str]) -> Optional[Path]:
    for suffix in suffixes:
        p = seg_dir / f"{frame_name}{suffix}"
        if p.exists():
            return p
    return None


def coefficient_of_variation(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.mean(values) == 0:
        return np.nan
    return float(np.std(values, ddof=0) / np.mean(values))


def discover_frames(cell_dir: Path) -> List[FrameRecord]:
    seg_dir = Path(cell_dir) / SEGMENTATION_DIR_NAME
    if not seg_dir.exists():
        if Path(cell_dir).name == SEGMENTATION_DIR_NAME:
            seg_dir = Path(cell_dir)
        else:
            raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

    frame_names: Set[str] = set()
    for path in seg_dir.iterdir():
        if not path.is_file():
            continue
        name = frame_name_from_instance_filename(path.name)
        if name:
            frame_names.add(name)

    if not frame_names:
        raise FileNotFoundError(f"No PCD/voxel instance files found in {seg_dir}")

    temp = []
    for frame_name in frame_names:
        voxel = first_existing(seg_dir, frame_name, VOXEL_INSTANCE_SUFFIXES)
        pcd = first_existing(seg_dir, frame_name, PCD_INSTANCE_SUFFIXES)
        if voxel is None and pcd is None:
            continue
        temp.append((frame_name, voxel, pcd, frame_number_from_name(frame_name)))

    numbers = [x[3] for x in temp]
    if all(v is not None for v in numbers) and len(set(numbers)) == len(numbers):
        temp.sort(key=lambda x: (int(x[3]), natural_key(x[0])))
    else:
        temp.sort(key=lambda x: natural_key(x[0]))

    frames = [
        FrameRecord(frame=i, frame_name=name, voxel_path=voxel, pcd_path=pcd)
        for i, (name, voxel, pcd, _n) in enumerate(temp)
    ]
    return frames


def _centroid_raw(points: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _subsample_centered_shape(points_raw: Optional[np.ndarray], max_points: int, seed: int) -> Optional[np.ndarray]:
    if points_raw is None or len(points_raw) == 0:
        return None
    pts_um = to_physical(np.asarray(points_raw, dtype=float))
    center = np.mean(pts_um, axis=0)
    pts = pts_um - center[None, :]
    if max_points > 0 and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[np.sort(idx)]
    return pts.astype(np.float32)


def _extract_groups(path: Optional[Path], pcd: bool) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    if path is None or not Path(path).exists():
        return {}
    arr = read_numeric_txt(Path(path))
    required = 9 if pcd else 6
    if arr.shape[1] < required:
        raise ValueError(f"Expected >= {required} columns in {path}; got {arr.shape[1]}")
    groups = group_indices_by_rgb(arr, (6, 9) if pcd else (3, 6))
    out = {}
    for color, idx in groups.items():
        points = np.asarray(arr[idx, :3], dtype=float)
        item = {"points": points, "centroid_raw": _centroid_raw(points)}
        if pcd:
            item["normals"] = np.asarray(arr[idx, 3:6], dtype=float)
        out[color] = item
    return out


def _pair_voxel_and_pcd_groups(
    voxel_groups: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    pcd_groups: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
) -> List[Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]]:
    """Pair the two representations WITHIN ONE FRAME only.

    RGB is used only when the same RGB exists in both representations in that frame.
    Remaining groups are paired by centroid geometry, never by temporal identity.
    """
    pairs: List[Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]] = []
    voxel_unmatched = set(voxel_groups)
    pcd_unmatched = set(pcd_groups)

    # Safe frame-local exact correspondence if both files retained the same segmentation color.
    for color in sorted(voxel_unmatched & pcd_unmatched):
        pairs.append((color, color))
    voxel_unmatched -= (set(voxel_groups) & set(pcd_groups))
    pcd_unmatched -= (set(voxel_groups) & set(pcd_groups))

    if voxel_unmatched and pcd_unmatched:
        vg = sorted(voxel_unmatched)
        pg = sorted(pcd_unmatched)
        vc = to_physical(np.vstack([voxel_groups[c]["centroid_raw"] for c in vg]))
        pc = to_physical(np.vstack([pcd_groups[c]["centroid_raw"] for c in pg]))
        dist = np.linalg.norm(vc[:, None, :] - pc[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(dist)
        used_v, used_p = set(), set()
        for r, c in zip(rows, cols):
            if dist[r, c] <= MAX_VOXEL_PCD_PAIR_DISTANCE_UM:
                pairs.append((vg[r], pg[c]))
                used_v.add(vg[r])
                used_p.add(pg[c])
        voxel_unmatched -= used_v
        pcd_unmatched -= used_p

    # Keep unmatched representations too; tracking can operate from either one.
    pairs.extend((c, None) for c in sorted(voxel_unmatched))
    pairs.extend((None, c) for c in sorted(pcd_unmatched))
    return pairs


def reconstruct_local_instances(record: FrameRecord) -> Tuple[List[LocalInstance], pd.DataFrame]:
    voxel_groups = _extract_groups(record.voxel_path, pcd=False)
    pcd_groups = _extract_groups(record.pcd_path, pcd=True)
    pairs = _pair_voxel_and_pcd_groups(voxel_groups, pcd_groups)

    instances: List[LocalInstance] = []
    rows = []
    profile = TRACKING_PROFILES[TRACKING_OBJECT_TYPE]

    for local_id, (voxel_rgb, pcd_rgb) in enumerate(pairs, start=1):
        vg = voxel_groups.get(voxel_rgb) if voxel_rgb is not None else None
        pg = pcd_groups.get(pcd_rgb) if pcd_rgb is not None else None

        # Prefer voxel centroid/count because voxel data represent the full instance.
        if vg is not None:
            centroid_raw = np.asarray(vg["centroid_raw"], dtype=float)
            size_count = int(len(vg["points"]))
        elif pg is not None:
            centroid_raw = np.asarray(pg["centroid_raw"], dtype=float)
            size_count = int(len(pg["points"]))
        else:
            continue

        if size_count < int(MIN_INSTANCE_POINTS):
            continue

        voxel_points = None if vg is None else np.asarray(vg["points"], dtype=float)
        pcd_points = None if pg is None else np.asarray(pg["points"], dtype=float)
        pcd_normals = None if pg is None else np.asarray(pg["normals"], dtype=float)
        shape_source = pcd_points if pcd_points is not None else voxel_points
        shape = _subsample_centered_shape(
            shape_source,
            max_points=int(profile["shape_max_points"]),
            seed=RANDOM_SEED + 10007 * record.frame + local_id,
        )

        centroid_um = to_physical(centroid_raw[None, :])[0]
        inst = LocalInstance(
            frame=record.frame,
            frame_name=record.frame_name,
            local_instance_id=local_id,
            centroid_raw=centroid_raw,
            centroid_um=centroid_um,
            size_count=size_count,
            voxel_rgb=voxel_rgb,
            pcd_rgb=pcd_rgb,
            voxel_points_raw=voxel_points,
            pcd_points_raw=pcd_points,
            pcd_normals_raw=pcd_normals,
            shape_points_centered_um=shape,
        )
        instances.append(inst)
        rows.append({
            "frame": record.frame,
            "frame_name": record.frame_name,
            "time_seconds": record.frame * DT_SECONDS,
            "local_instance_id": local_id,
            "centroid_x": centroid_raw[0],
            "centroid_y": centroid_raw[1],
            "centroid_z": centroid_raw[2],
            "centroid_x_um": centroid_um[0],
            "centroid_y_um": centroid_um[1],
            "centroid_z_um": centroid_um[2],
            "size_count": size_count,
            "voxel_point_count": len(voxel_points) if voxel_points is not None else 0,
            "pcd_point_count": len(pcd_points) if pcd_points is not None else 0,
            "voxel_r": voxel_rgb[0] if voxel_rgb else np.nan,
            "voxel_g": voxel_rgb[1] if voxel_rgb else np.nan,
            "voxel_b": voxel_rgb[2] if voxel_rgb else np.nan,
            "pcd_r": pcd_rgb[0] if pcd_rgb else np.nan,
            "pcd_g": pcd_rgb[1] if pcd_rgb else np.nan,
            "pcd_b": pcd_rgb[2] if pcd_rgb else np.nan,
        })

    # Re-number after filtering to guarantee dense frame-local IDs.
    for new_id, inst in enumerate(instances, start=1):
        old_id = inst.local_instance_id
        inst.local_instance_id = new_id
        for row in rows:
            if row["local_instance_id"] == old_id:
                row["local_instance_id"] = new_id
                break

    return instances, pd.DataFrame(rows)


def kalman_initialise(position_um: np.ndarray) -> KalmanState:
    x = np.zeros(6, dtype=float)
    x[:3] = np.asarray(position_um, dtype=float)
    P = np.diag([0.20, 0.20, 0.20, 1.0, 1.0, 1.0]).astype(float)
    return KalmanState(x=x, P=P)


def kalman_matrices(dt_frames: float, process_noise: float, measurement_noise: float):
    dt = float(dt_frames)
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    H = np.zeros((3, 6), dtype=float)
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    # Constant-velocity process model.
    q = float(process_noise) ** 2
    Q = np.zeros((6, 6), dtype=float)
    for axis in range(3):
        p = axis
        v = axis + 3
        Q[p, p] = 0.25 * dt**4 * q
        Q[p, v] = 0.5 * dt**3 * q
        Q[v, p] = 0.5 * dt**3 * q
        Q[v, v] = dt**2 * q

    R = np.eye(3, dtype=float) * float(measurement_noise) ** 2
    return F, H, Q, R


def kalman_predict_copy(kalman: KalmanState, dt_frames: float) -> KalmanState:
    profile = TRACKING_PROFILES[TRACKING_OBJECT_TYPE]
    F, _H, Q, _R = kalman_matrices(
        dt_frames,
        profile["kalman_process_noise"],
        profile["kalman_measurement_noise"],
    )
    return KalmanState(x=F @ kalman.x, P=F @ kalman.P @ F.T + Q)


def kalman_update(predicted: KalmanState, measurement_um: np.ndarray) -> KalmanState:
    profile = TRACKING_PROFILES[TRACKING_OBJECT_TYPE]
    _F, H, _Q, R = kalman_matrices(
        1.0,
        profile["kalman_process_noise"],
        profile["kalman_measurement_noise"],
    )
    z = np.asarray(measurement_um, dtype=float)
    y = z - H @ predicted.x
    S = H @ predicted.P @ H.T + R
    K = predicted.P @ H.T @ np.linalg.pinv(S)
    x = predicted.x + K @ y
    P = (np.eye(6) - K @ H) @ predicted.P
    return KalmanState(x=x, P=P)


def symmetric_centered_shape_distance_um(
    a: Optional[np.ndarray], b: Optional[np.ndarray]
) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.nan
    tree_b = cKDTree(np.asarray(b, dtype=float))
    da, _ = tree_b.query(np.asarray(a, dtype=float), k=1)
    tree_a = cKDTree(np.asarray(a, dtype=float))
    db, _ = tree_a.query(np.asarray(b, dtype=float), k=1)
    return float(0.5 * (np.mean(da) + np.mean(db)))


def tracking_pair_cost(
    track: TrackState,
    detection: LocalInstance,
    frame: int,
) -> Tuple[float, Dict[str, float]]:
    profile = TRACKING_PROFILES[TRACKING_OBJECT_TYPE]
    dt_frames = max(1, int(frame - track.last_frame))
    predicted = kalman_predict_copy(track.kalman, dt_frames)
    predicted_position = predicted.x[:3]
    detection_position = detection.centroid_um
    distance_um = float(np.linalg.norm(detection_position - predicted_position))

    # Hard physical sphere. Gap links get only a modest radius expansion.
    allowed_radius = float(profile["search_radius_um"]) * (1.0 + 0.35 * (dt_frames - 1))
    if distance_um > allowed_radius:
        return np.inf, {"reason": "outside_sphere", "distance_um": distance_um, "allowed_radius_um": allowed_radius}

    size_ratio = max(float(detection.size_count), 1.0) / max(float(track.last_size_count), 1.0)
    if not (float(profile["min_size_ratio"]) <= size_ratio <= float(profile["max_size_ratio"])):
        return np.inf, {"reason": "size_gate", "distance_um": distance_um, "allowed_radius_um": allowed_radius, "size_ratio": size_ratio}

    distance_cost = distance_um / max(allowed_radius, 1e-12)
    size_cost = abs(math.log(max(size_ratio, 1e-12)))

    velocity = predicted.x[3:6]
    observed = detection_position - track.last_position_um
    vnorm = float(np.linalg.norm(velocity))
    onorm = float(np.linalg.norm(observed))
    direction_cost = 0.0
    if vnorm >= float(profile["direction_min_step_um"]) and onorm >= float(profile["direction_min_step_um"]):
        cosine = float(np.dot(velocity, observed) / (vnorm * onorm))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        direction_cost = 0.5 * (1.0 - cosine)

    shape_distance = np.nan
    shape_cost = 0.0
    if USE_PCD_SHAPE_TIEBREAKER and float(profile["weights"]["shape"]) > 0:
        shape_distance = symmetric_centered_shape_distance_um(
            track.last_shape_centered_um,
            detection.shape_points_centered_um,
        )
        if np.isfinite(shape_distance):
            shape_cost = min(shape_distance / max(allowed_radius, 1e-12), 2.0)

    w = profile["weights"]
    total = (
        float(w["distance"]) * distance_cost
        + float(w["size"]) * size_cost
        + float(w["direction"]) * direction_cost
        + float(w["shape"]) * shape_cost
    )

    return float(total), {
        "reason": "candidate",
        "predicted_x_um": float(predicted_position[0]),
        "predicted_y_um": float(predicted_position[1]),
        "predicted_z_um": float(predicted_position[2]),
        "distance_um": distance_um,
        "allowed_radius_um": allowed_radius,
        "distance_cost": distance_cost,
        "size_ratio": size_ratio,
        "size_cost": size_cost,
        "direction_cost": direction_cost,
        "shape_distance_um": shape_distance,
        "shape_cost": shape_cost,
        "total_cost": float(total),
    }


def second_best_margin(cost_matrix: np.ndarray, row: int, col: int) -> Tuple[float, float, float, float]:
    """Return row second-best, row margin, column second-best, column margin."""
    assigned = float(cost_matrix[row, col])
    row_vals = cost_matrix[row, np.isfinite(cost_matrix[row])]
    row_vals = np.sort(row_vals)
    row_second = float(row_vals[1]) if len(row_vals) > 1 else np.inf
    row_margin = row_second - assigned if np.isfinite(row_second) else np.inf

    col_vals = cost_matrix[np.isfinite(cost_matrix[:, col]), col]
    col_vals = np.sort(col_vals)
    col_second = float(col_vals[1]) if len(col_vals) > 1 else np.inf
    col_margin = col_second - assigned if np.isfinite(col_second) else np.inf
    return row_second, row_margin, col_second, col_margin


def track_instances(
    frame_instances: List[List[LocalInstance]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile = TRACKING_PROFILES[TRACKING_OBJECT_TYPE]
    next_track_id = 1
    tracks: Dict[int, TrackState] = {}
    output_rows: List[Dict[str, object]] = []
    link_rows: List[Dict[str, object]] = []
    rejection_rows: List[Dict[str, object]] = []

    for frame, detections in enumerate(tqdm(frame_instances, desc=f"Tracking {TRACKING_OBJECT_TYPE}")):
        # Tracks are eligible if they were seen recently and were not retired.
        active_ids = [
            tid for tid, tr in tracks.items()
            if (not tr.retired)
            and 1 <= frame - tr.last_frame <= int(profile["max_gap_frames"]) + 1
        ]
        active_ids.sort()

        matched_tracks: Set[int] = set()
        matched_detections: Set[int] = set()

        if active_ids and detections:
            large = 1e12
            cost = np.full((len(active_ids), len(detections)), large, dtype=float)
            parts: Dict[Tuple[int, int], Dict[str, float]] = {}

            for r, tid in enumerate(active_ids):
                tr = tracks[tid]
                for c, det in enumerate(detections):
                    value, diagnostics = tracking_pair_cost(tr, det, frame)
                    if np.isfinite(value) and value <= float(profile["max_assignment_cost"]):
                        cost[r, c] = value
                        parts[(r, c)] = diagnostics

            rows, cols = linear_sum_assignment(cost)
            accepted = []

            for r, c in zip(rows, cols):
                assigned = float(cost[r, c])
                if not np.isfinite(assigned) or assigned >= large / 2:
                    continue

                row_second, row_margin, col_second, col_margin = second_best_margin(cost, r, c)
                ambiguous = (
                    row_margin < float(profile["ambiguity_margin"])
                    or col_margin < float(profile["ambiguity_margin"])
                )

                if bool(profile["require_mutual_best"]):
                    row_best_col = int(np.argmin(cost[r]))
                    col_best_row = int(np.argmin(cost[:, c]))
                    mutual_best = row_best_col == c and col_best_row == r
                else:
                    mutual_best = True

                tid = active_ids[r]
                det = detections[c]
                diag = parts[(r, c)]

                if ambiguous or not mutual_best:
                    rejection_rows.append({
                        "frame": frame,
                        "time_seconds": frame * DT_SECONDS,
                        "track_id": tid,
                        "local_instance_id": det.local_instance_id,
                        "assigned_cost": assigned,
                        "row_second_best_cost": row_second,
                        "row_margin": row_margin,
                        "column_second_best_cost": col_second,
                        "column_margin": col_margin,
                        "mutual_best": mutual_best,
                        "reason": "ambiguous" if ambiguous else "not_mutual_best",
                        **diag,
                    })
                    continue

                accepted.append((r, c, tid, det, diag, row_second, row_margin, col_second, col_margin))

            # Update accepted links only after all ambiguity checks.
            for r, c, tid, det, diag, row_second, row_margin, col_second, col_margin in accepted:
                tr = tracks[tid]
                frame_delta = frame - tr.last_frame
                previous_position = tr.last_position_um.copy()
                previous_local = tr.last_local_instance_id
                predicted = kalman_predict_copy(tr.kalman, frame_delta)
                tr.kalman = kalman_update(predicted, det.centroid_um)
                tr.last_frame = frame
                tr.last_local_instance_id = det.local_instance_id
                tr.last_position_um = det.centroid_um.copy()
                tr.last_size_count = int(det.size_count)
                tr.last_shape_centered_um = det.shape_points_centered_um
                tr.observed_positions.append(det.centroid_um.copy())
                tr.observed_frames.append(frame)
                tr.missed_frames = 0

                matched_tracks.add(tid)
                matched_detections.add(c)

                output_rows.append(_tracked_row(det, tid, "linked", diag["total_cost"], False))
                link_rows.append({
                    "track_id": tid,
                    "previous_frame": frame - frame_delta,
                    "current_frame": frame,
                    "frame_delta": frame_delta,
                    "previous_local_instance_id": previous_local,
                    "current_local_instance_id": det.local_instance_id,
                    "previous_x_um": previous_position[0],
                    "previous_y_um": previous_position[1],
                    "previous_z_um": previous_position[2],
                    "current_x_um": det.centroid_um[0],
                    "current_y_um": det.centroid_um[1],
                    "current_z_um": det.centroid_um[2],
                    "displacement_um": float(np.linalg.norm(det.centroid_um - previous_position)),
                    "row_second_best_cost": row_second,
                    "row_margin": row_margin,
                    "column_second_best_cost": col_second,
                    "column_margin": col_margin,
                    **diag,
                })

        # New IDs for every detection that was not confidently linked.
        for c, det in enumerate(detections):
            if c in matched_detections:
                continue
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = TrackState(
                track_id=tid,
                last_frame=frame,
                last_local_instance_id=det.local_instance_id,
                kalman=kalman_initialise(det.centroid_um),
                last_position_um=det.centroid_um.copy(),
                last_size_count=int(det.size_count),
                last_shape_centered_um=det.shape_points_centered_um,
                observed_positions=[det.centroid_um.copy()],
                observed_frames=[frame],
            )
            output_rows.append(_tracked_row(det, tid, "new", 0.0, False))

        # Age unmatched tracks. They are allowed to survive short gaps but never
        # steal an already accepted detection.
        for tid in active_ids:
            if tid not in matched_tracks and tid in tracks:
                tracks[tid].missed_frames = frame - tracks[tid].last_frame

    tracked = pd.DataFrame(output_rows)
    if not tracked.empty:
        tracked["track_id"] = tracked["track_id"].astype(int)
        tracked = tracked.sort_values(["frame", "track_id"]).reset_index(drop=True)
    return tracked, pd.DataFrame(link_rows), pd.DataFrame(rejection_rows)


def _tracked_row(det: LocalInstance, track_id: int, status: str, cost: float, ambiguous: bool) -> Dict[str, object]:
    return {
        "frame": det.frame,
        "frame_name": det.frame_name,
        "time_seconds": det.frame * DT_SECONDS,
        "local_instance_id": det.local_instance_id,
        "track_id": int(track_id),
        "tracking_status": status,
        "tracking_cost": float(cost),
        "tracking_confidence": float(np.exp(-max(0.0, cost))),
        "tracking_ambiguous": bool(ambiguous),
        "centroid_x": det.centroid_raw[0],
        "centroid_y": det.centroid_raw[1],
        "centroid_z": det.centroid_raw[2],
        "centroid_x_um": det.centroid_um[0],
        "centroid_y_um": det.centroid_um[1],
        "centroid_z_um": det.centroid_um[2],
        "size_count": det.size_count,
        "voxel_r": det.voxel_rgb[0] if det.voxel_rgb else np.nan,
        "voxel_g": det.voxel_rgb[1] if det.voxel_rgb else np.nan,
        "voxel_b": det.voxel_rgb[2] if det.voxel_rgb else np.nan,
        "pcd_r": det.pcd_rgb[0] if det.pcd_rgb else np.nan,
        "pcd_g": det.pcd_rgb[1] if det.pcd_rgb else np.nan,
        "pcd_b": det.pcd_rgb[2] if det.pcd_rgb else np.nan,
    }


def summarize_tracks(tracked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if tracked.empty:
        return pd.DataFrame()
    for tid, g in tracked.groupby("track_id", sort=True):
        g = g.sort_values("frame")
        xyz = g[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy(float)
        frames = g["frame"].to_numpy(int)
        if len(g) >= 2:
            dxyz = np.diff(xyz, axis=0)
            df = np.diff(frames)
            steps = np.linalg.norm(dxyz, axis=1)
            speeds = steps / np.clip(df * DT_SECONDS, 1e-12, None)
            path_length = float(np.sum(steps))
            net = float(np.linalg.norm(xyz[-1] - xyz[0]))
            directionality = net / path_length if path_length > 0 else 0.0
            mean_speed = float(np.mean(speeds))
            max_speed = float(np.max(speeds))
        else:
            path_length = net = directionality = mean_speed = max_speed = 0.0
        span = int(frames[-1] - frames[0] + 1)
        rows.append({
            "track_id": int(tid),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "observed_frame_count": int(len(g)),
            "span_frame_count": span,
            "missing_frame_count": int(span - len(g)),
            "lifetime_seconds": float((frames[-1] - frames[0]) * DT_SECONDS),
            "path_length_um": path_length,
            "net_displacement_um": net,
            "directionality": directionality,
            "mean_speed_um_per_s": mean_speed,
            "max_speed_um_per_s": max_speed,
            "mean_size_count": float(g["size_count"].mean()),
            "size_cv": coefficient_of_variation(g["size_count"]),
        })
    return pd.DataFrame(rows)


def _frame_rgb_track_maps(frame_tracked: pd.DataFrame):
    voxel_map = {}
    pcd_map = {}
    for row in frame_tracked.itertuples(index=False):
        tid = int(row.track_id)
        local_id = int(row.local_instance_id)
        if np.isfinite(row.voxel_r) and np.isfinite(row.voxel_g) and np.isfinite(row.voxel_b):
            voxel_map[(int(row.voxel_r), int(row.voxel_g), int(row.voxel_b))] = (tid, local_id)
        if np.isfinite(row.pcd_r) and np.isfinite(row.pcd_g) and np.isfinite(row.pcd_b):
            pcd_map[(int(row.pcd_r), int(row.pcd_g), int(row.pcd_b))] = (tid, local_id)
    return voxel_map, pcd_map


def relabel_pointcloud(path: Path, output: Path, frame_tracked: pd.DataFrame, pcd_mode: bool) -> None:
    arr = read_numeric_txt(path)
    rgb_cols = (6, 9) if pcd_mode else (3, 6)
    voxel_map, pcd_map = _frame_rgb_track_maps(frame_tracked)
    mapping = pcd_map if pcd_mode else voxel_map
    rgb = np.rint(arr[:, rgb_cols[0]:rgb_cols[1]]).astype(int)

    keep = np.zeros(len(arr), dtype=bool)
    new_rgb = np.zeros((len(arr), 3), dtype=int)
    track_ids = np.zeros(len(arr), dtype=int)
    local_ids = np.zeros(len(arr), dtype=int)
    unique, inverse = np.unique(rgb, axis=0, return_inverse=True)
    for i, color in enumerate(unique):
        key = rgb_tuple(color)
        if key not in mapping:
            continue
        tid, lid = mapping[key]
        mask = inverse == i
        keep[mask] = True
        new_rgb[mask] = deterministic_track_rgb(tid)
        track_ids[mask] = tid
        local_ids[mask] = lid

    arr = arr[keep]
    new_rgb = new_rgb[keep]
    track_ids = track_ids[keep]
    local_ids = local_ids[keep]

    if pcd_mode:
        out = np.column_stack([arr[:, :6], new_rgb, track_ids, local_ids])
        fmt = ["%.6f"] * 6 + ["%d"] * 5
    else:
        out = np.column_stack([arr[:, :3], new_rgb, track_ids, local_ids])
        fmt = ["%.6f"] * 3 + ["%d"] * 5
    save_numeric_txt(output, out, fmt)


def save_tracked_pointclouds(frames: List[FrameRecord], tracked: pd.DataFrame, out_dir: Path) -> None:
    """Save only persistent-ID surface PCDs for LinkNex v3.

    Tracked voxel copies are intentionally omitted to avoid duplicate output.
    Original *_voxel_instance.txt files remain available for GUI overlays.
    """
    pcd_dir = out_dir / "tracked_surface_pointclouds"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(frames, desc="Saving persistent-ID surface point clouds"):
        ft = tracked[tracked["frame"] == record.frame]
        if ft.empty or record.pcd_path is None:
            continue
        relabel_pointcloud(
            record.pcd_path,
            pcd_dir / f"{record.frame_name}_pcd_tracked.txt",
            ft,
            pcd_mode=True,
        )


def pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def surface_points_by_track(record: FrameRecord, frame_tracked: pd.DataFrame) -> Dict[int, np.ndarray]:
    result = {}
    voxel_map, pcd_map = _frame_rgb_track_maps(frame_tracked)
    rng = np.random.default_rng(RANDOM_SEED + record.frame)

    if record.pcd_path is not None:
        arr = read_numeric_txt(record.pcd_path)
        groups = group_indices_by_rgb(arr, (6, 9))
        mapping = pcd_map
        for color, idx in groups.items():
            if color not in mapping:
                continue
            tid, _lid = mapping[color]
            points = to_physical(np.asarray(arr[idx, :3], dtype=float))
            if MAX_SURFACE_POINTS_PER_INSTANCE > 0 and len(points) > MAX_SURFACE_POINTS_PER_INSTANCE:
                chosen = rng.choice(len(points), size=MAX_SURFACE_POINTS_PER_INSTANCE, replace=False)
                points = points[np.sort(chosen)]
            result[int(tid)] = points
        return result

    if record.voxel_path is not None:
        arr = read_numeric_txt(record.voxel_path)
        groups = group_indices_by_rgb(arr, (3, 6))
        mapping = voxel_map
        for color, idx in groups.items():
            if color not in mapping:
                continue
            tid, _lid = mapping[color]
            points = to_physical(np.asarray(arr[idx, :3], dtype=float))
            if MAX_SURFACE_POINTS_PER_INSTANCE > 0 and len(points) > MAX_SURFACE_POINTS_PER_INSTANCE:
                chosen = rng.choice(len(points), size=MAX_SURFACE_POINTS_PER_INSTANCE, replace=False)
                points = points[np.sort(chosen)]
            result[int(tid)] = points
    return result


def minimum_surface_distance_um(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.full(3, np.nan), np.full(3, np.nan)
    swapped = False
    aa, bb = a, b
    if len(aa) > len(bb):
        aa, bb = bb, aa
        swapped = True
    tree = cKDTree(bb)
    dist, idx_b = tree.query(aa, k=1)
    i = int(np.argmin(dist))
    j = int(idx_b[i])
    pa, pb = aa[i], bb[j]
    if swapped:
        pa, pb = pb, pa
    return float(dist[i]), np.asarray(pa), np.asarray(pb)


def graph_metrics(graph: nx.Graph) -> Dict[str, float]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n == 0:
        return {
            "node_count": 0, "edge_count": 0, "density": np.nan,
            "mean_degree": np.nan, "average_clustering": np.nan,
            "transitivity": np.nan, "global_efficiency": np.nan,
            "local_efficiency": np.nan, "component_count": 0,
            "largest_component_fraction": np.nan, "isolated_node_fraction": np.nan,
            "modularity": np.nan,
        }
    degrees = [d for _, d in graph.degree()]
    comps = list(nx.connected_components(graph))
    largest = max((len(c) for c in comps), default=0)
    try:
        global_eff = float(nx.global_efficiency(graph)) if n > 1 else 0.0
    except Exception:
        global_eff = np.nan
    try:
        local_eff = float(nx.local_efficiency(graph)) if n > 1 else 0.0
    except Exception:
        local_eff = np.nan
    if m > 0 and n > 1:
        try:
            comm = list(nx.community.greedy_modularity_communities(graph))
            modularity = float(nx.community.modularity(graph, comm)) if comm else 0.0
        except Exception:
            modularity = np.nan
    else:
        modularity = 0.0
    return {
        "node_count": n,
        "edge_count": m,
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "average_clustering": float(nx.average_clustering(graph)) if n else 0.0,
        "transitivity": float(nx.transitivity(graph)) if n else 0.0,
        "global_efficiency": global_eff,
        "local_efficiency": local_eff,
        "component_count": len(comps),
        "largest_component_fraction": largest / n if n else np.nan,
        "isolated_node_fraction": float(sum(d == 0 for d in degrees) / n),
        "modularity": modularity,
    }


def build_temporal_network(frames: List[FrameRecord], tracked: pd.DataFrame):
    edge_rows = []
    pair_rows = []
    metric_rows = []
    node_rows = []
    previous_edges: Set[Tuple[int, int]] = set()
    edge_age: Dict[Tuple[int, int], int] = {}

    for record in tqdm(frames, desc="Building temporal network"):
        ft = tracked[tracked["frame"] == record.frame].sort_values("track_id")
        tids = ft["track_id"].astype(int).to_numpy()
        graph = nx.Graph()
        for row in ft.itertuples(index=False):
            graph.add_node(
                int(row.track_id),
                x=float(row.centroid_x_um), y=float(row.centroid_y_um), z=float(row.centroid_z_um),
            )

        surfaces = surface_points_by_track(record, ft)
        current_edges: Set[Tuple[int, int]] = set()

        # Broad centroid gate only avoids obviously distant pairs. Exact edge is
        # always decided by minimum surface distance.
        if len(ft) >= 2:
            positions = ft[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy(float)
            max_extent = 0.0
            for pts in surfaces.values():
                if len(pts):
                    c = np.mean(pts, axis=0)
                    max_extent = max(max_extent, float(np.max(np.linalg.norm(pts - c[None, :], axis=1))))
            broad = max(2 * max_extent + PAIR_DISTANCE_SAVE_LIMIT_UM, NETWORK_DISTANCE_UM * 4)
            pairs = cKDTree(positions).query_pairs(r=broad, output_type="set")
        else:
            pairs = set()

        tid_to_index = {int(t): i for i, t in enumerate(tids)}
        for old in previous_edges:
            if old[0] in tid_to_index and old[1] in tid_to_index:
                pairs.add((tid_to_index[old[0]], tid_to_index[old[1]]))

        for i, j in pairs:
            ti, tj = int(tids[i]), int(tids[j])
            key = pair_key(ti, tj)
            if ti not in surfaces or tj not in surfaces:
                continue
            d, pa, pb = minimum_surface_distance_um(surfaces[ti], surfaces[tj])
            was = key in previous_edges
            is_edge = np.isfinite(d) and d <= NETWORK_DISTANCE_UM
            if np.isfinite(d) and (d <= PAIR_DISTANCE_SAVE_LIMIT_UM or was or is_edge):
                pair_rows.append({
                    "frame": record.frame,
                    "frame_name": record.frame_name,
                    "time_seconds": record.frame * DT_SECONDS,
                    "track_i": key[0], "track_j": key[1],
                    "surface_distance_um": d,
                    "was_edge_previous_frame": was,
                    "is_edge": is_edge,
                })
            if is_edge:
                current_edges.add(key)
                age = edge_age.get(key, 0) + 1 if was else 1
                edge_age[key] = age
                graph.add_edge(key[0], key[1], distance_um=d)
                edge_rows.append({
                    "frame": record.frame,
                    "frame_name": record.frame_name,
                    "time_seconds": record.frame * DT_SECONDS,
                    "track_i": key[0], "track_j": key[1],
                    "surface_distance_um": d,
                    "edge_age_frames": age,
                    "edge_age_seconds": (age - 1) * DT_SECONDS,
                    "is_new_edge": not was,
                    "is_persistent_edge": was,
                    "nearest_i_x_um": pa[0], "nearest_i_y_um": pa[1], "nearest_i_z_um": pa[2],
                    "nearest_j_x_um": pb[0], "nearest_j_y_um": pb[1], "nearest_j_z_um": pb[2],
                })

        new_edges = current_edges - previous_edges
        lost_edges = previous_edges - current_edges
        persistent = current_edges & previous_edges
        union = current_edges | previous_edges
        jaccard = len(persistent) / len(union) if union else 1.0
        for lost in lost_edges:
            edge_age.pop(lost, None)

        m = graph_metrics(graph)
        m.update({
            "frame": record.frame,
            "frame_name": record.frame_name,
            "time_seconds": record.frame * DT_SECONDS,
            "new_edge_count": len(new_edges),
            "lost_edge_count": len(lost_edges),
            "persistent_edge_count": len(persistent),
            "edge_jaccard_previous": jaccard,
            "edge_turnover_previous": 1.0 - jaccard,
        })
        metric_rows.append(m)

        clustering = nx.clustering(graph)
        betweenness = nx.betweenness_centrality(graph) if CALCULATE_BETWEENNESS and graph.number_of_nodes() else {}
        for node in graph.nodes:
            node_rows.append({
                "frame": record.frame,
                "frame_name": record.frame_name,
                "time_seconds": record.frame * DT_SECONDS,
                "track_id": int(node),
                "degree": int(graph.degree(node)),
                "clustering": float(clustering.get(node, 0.0)),
                "betweenness": float(betweenness.get(node, np.nan)),
                "is_isolated": graph.degree(node) == 0,
            })

        previous_edges = current_edges

    return pd.DataFrame(edge_rows), pd.DataFrame(pair_rows), pd.DataFrame(metric_rows), pd.DataFrame(node_rows)


def extract_edge_events(edges: pd.DataFrame, pair_distances: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    lookup = {}
    for row in pair_distances.itertuples(index=False):
        lookup[(int(row.track_i), int(row.track_j), int(row.frame))] = float(row.surface_distance_um)

    out = []
    event_id = 1
    for (ti, tj), g in edges.groupby(["track_i", "track_j"], sort=True):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy(int)
        blocks = []
        start = 0
        for k in range(1, len(frames)):
            if frames[k] != frames[k - 1] + 1:
                blocks.append(g.iloc[start:k])
                start = k
        blocks.append(g.iloc[start:])
        for bi, block in enumerate(blocks, start=1):
            s = int(block["frame"].iloc[0])
            e = int(block["frame"].iloc[-1])
            duration = e - s + 1
            minimum = float(block["surface_distance_um"].min())
            before = lookup.get((int(ti), int(tj), s - 1), np.nan)
            after = lookup.get((int(ti), int(tj), e + 1), np.nan)
            approach_leave = np.isfinite(before) and np.isfinite(after) and before > minimum and after > minimum
            if duration <= TRANSIENT_MAX_FRAMES and approach_leave:
                event_class = "collision_like_transient"
            elif duration <= TRANSIENT_MAX_FRAMES:
                event_class = "transient_encounter"
            else:
                event_class = "stable_association"
            out.append({
                "event_id": event_id,
                "track_i": int(ti), "track_j": int(tj),
                "pair_event_index": bi,
                "pair_total_event_count": len(blocks),
                "is_recurrent_pair": len(blocks) > 1,
                "start_frame": s, "end_frame": e,
                "start_time_seconds": s * DT_SECONDS,
                "end_time_seconds": e * DT_SECONDS,
                "duration_frames": duration,
                "duration_seconds": duration * DT_SECONDS,
                "minimum_surface_distance_um": minimum,
                "distance_before_event_um": before,
                "distance_after_event_um": after,
                "event_class": event_class,
            })
            event_id += 1
    return pd.DataFrame(out)


def summarize_dynamic_network(metrics: pd.DataFrame, events: pd.DataFrame, tracked: pd.DataFrame, node_roles: pd.DataFrame) -> pd.DataFrame:
    duration_min = max(
        ((metrics["frame"].max() - metrics["frame"].min()) * DT_SECONDS / 60.0) if len(metrics) > 1 else DT_SECONDS / 60.0,
        DT_SECONDS / 60.0,
    )
    row = {
        "frame_count": int(tracked["frame"].nunique()) if not tracked.empty else 0,
        "track_count": int(tracked["track_id"].nunique()) if not tracked.empty else 0,
        "encounter_event_count": int(len(events)),
        "encounter_frequency_per_min": float(len(events) / duration_min),
    }
    if not events.empty:
        row["median_edge_lifetime_seconds"] = float(events["duration_seconds"].median())
        row["mean_edge_lifetime_seconds"] = float(events["duration_seconds"].mean())
        row["recurrent_pair_fraction"] = float(
            events.loc[events["is_recurrent_pair"], ["track_i", "track_j"]].drop_duplicates().shape[0]
            / max(1, events[["track_i", "track_j"]].drop_duplicates().shape[0])
        )
    for col in [
        "node_count", "edge_count", "density", "mean_degree", "average_clustering",
        "transitivity", "global_efficiency", "local_efficiency", "component_count",
        "largest_component_fraction", "isolated_node_fraction", "modularity",
        "edge_turnover_previous", "new_edge_count", "lost_edge_count",
    ]:
        if col in metrics.columns and len(metrics):
            vals = pd.to_numeric(metrics[col], errors="coerce").dropna()
            if len(vals):
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_sd"] = float(vals.std(ddof=0))
                row[f"{col}_cv"] = coefficient_of_variation(vals)
    if not node_roles.empty:
        mean_degree_per_track = node_roles.groupby("track_id")["degree"].mean()
        row["mean_temporal_degree_per_track"] = float(mean_degree_per_track.mean())
    return pd.DataFrame([row])


def plot_trajectories(tracked: pd.DataFrame, fig_dir: Path) -> None:
    if tracked.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    for plane, cols in {
        "xy": ("centroid_x_um", "centroid_y_um"),
        "xz": ("centroid_x_um", "centroid_z_um"),
        "yz": ("centroid_y_um", "centroid_z_um"),
    }.items():
        fig, ax = plt.subplots(figsize=(8, 8), dpi=PNG_DPI)
        for tid, g in tracked.groupby("track_id"):
            g = g.sort_values("frame")
            color = np.asarray(deterministic_track_rgb(int(tid))) / 255.0
            ax.plot(g[cols[0]], g[cols[1]], marker="o", markersize=2, linewidth=0.8, color=color, alpha=0.8)
        ax.set_xlabel(cols[0].replace("centroid_", "").replace("_um", " (µm)"))
        ax.set_ylabel(cols[1].replace("centroid_", "").replace("_um", " (µm)"))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Persistent organelle trajectories ({plane.upper()})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"trajectories_{plane}.png", bbox_inches="tight")
        plt.close(fig)


def plot_network_timeseries(metrics: pd.DataFrame, fig_dir: Path) -> None:
    if metrics.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["density", "global_efficiency", "average_clustering", "largest_component_fraction", "edge_turnover_previous"] if c in metrics]
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.4 * max(1, len(cols))), dpi=PNG_DPI, sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(metrics["time_seconds"], metrics[col], marker="o", markersize=2, linewidth=1)
        ax.set_ylabel(col.replace("_", " "))
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "network_metrics_time_series.png", bbox_inches="tight")
    plt.close(fig)


def save_frame_network_pngs(frames: List[FrameRecord], tracked: pd.DataFrame, edges: pd.DataFrame, fig_dir: Path) -> None:
    out = fig_dir / "frame_network_xy"
    out.mkdir(parents=True, exist_ok=True)
    for record in tqdm(frames, desc="Saving frame network PNGs"):
        ft = tracked[tracked["frame"] == record.frame]
        if ft.empty:
            continue
        fe = edges[edges["frame"] == record.frame] if not edges.empty else pd.DataFrame()
        pos = {int(r.track_id): (float(r.centroid_x_um), float(r.centroid_y_um)) for r in ft.itertuples(index=False)}
        fig, ax = plt.subplots(figsize=(8, 8), dpi=PNG_DPI)
        if not fe.empty:
            for e in fe.itertuples(index=False):
                if int(e.track_i) in pos and int(e.track_j) in pos:
                    a, b = pos[int(e.track_i)], pos[int(e.track_j)]
                    ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=0.8, alpha=0.55)
        for r in ft.itertuples(index=False):
            color = np.asarray(deterministic_track_rgb(int(r.track_id))) / 255.0
            ax.scatter([r.centroid_x_um], [r.centroid_y_um], s=18, color=[color], zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_title(f"frame {record.frame:04d} | {record.frame_name}")
        fig.tight_layout()
        fig.savefig(out / f"frame_{record.frame:04d}.png", bbox_inches="tight")
        plt.close(fig)


def process_cell(cell_dir: Path) -> None:
    cell_dir = Path(cell_dir)
    seg_dir = cell_dir / SEGMENTATION_DIR_NAME
    out_dir = seg_dir
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"CELL: {cell_dir}")
    print(f"MODE: {TRACKING_OBJECT_TYPE}")
    print("Cross-frame RGB use: DISABLED")
    print("Morphology CSV required: NO")
    print("=" * 80)

    frames = discover_frames(cell_dir)
    print(f"Discovered {len(frames)} frames")

    frame_instances: List[List[LocalInstance]] = []
    local_tables = []
    manifest_rows = []
    for record in tqdm(frames, desc="Reconstructing local instances"):
        instances, table = reconstruct_local_instances(record)
        frame_instances.append(instances)
        if not table.empty:
            local_tables.append(table)
        manifest_rows.append({
            "frame": record.frame,
            "frame_name": record.frame_name,
            "voxel_path": str(record.voxel_path) if record.voxel_path else "",
            "pcd_path": str(record.pcd_path) if record.pcd_path else "",
            "local_instance_count": len(instances),
        })

    local_df = pd.concat(local_tables, ignore_index=True) if local_tables else pd.DataFrame()
    local_df.to_csv(tables_dir / "instances_reconstructed_from_pointclouds.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(tables_dir / "frame_manifest.csv", index=False)

    tracked, links, rejected = track_instances(frame_instances)
    tracked.to_csv(tables_dir / "instances_tracked_all_frames.csv", index=False)
    links.to_csv(tables_dir / "tracking_links.csv", index=False)
    rejected.to_csv(tables_dir / "tracking_ambiguous_rejections.csv", index=False)
    summarize_tracks(tracked).to_csv(tables_dir / "tracks_summary.csv", index=False)

    if SAVE_TRACKED_POINTCLOUDS:
        save_tracked_pointclouds(frames, tracked, out_dir)

    edges, pair_distances, metrics, node_roles = build_temporal_network(frames, tracked)
    edges.to_csv(tables_dir / "network_edges_by_frame.csv", index=False)
    pair_distances.to_csv(tables_dir / "near_pair_distances_by_frame.csv", index=False)
    metrics.to_csv(tables_dir / "network_metrics_by_frame.csv", index=False)
    node_roles.to_csv(tables_dir / "node_network_roles_by_frame.csv", index=False)

    events = extract_edge_events(edges, pair_distances)
    events.to_csv(tables_dir / "edge_events.csv", index=False)
    summarize_dynamic_network(metrics, events, tracked, node_roles).to_csv(
        tables_dir / "dynamic_network_summary.csv", index=False
    )

    if SAVE_TRAJECTORY_PNGS:
        plot_trajectories(tracked, fig_dir)
    if SAVE_NETWORK_TIME_SERIES:
        plot_network_timeseries(metrics, fig_dir)
    if SAVE_FRAME_NETWORK_PNGS:
        save_frame_network_pngs(frames, tracked, edges, fig_dir)

    config = {
        "tracking_algorithm_version": TRACKING_ALGORITHM_VERSION,
        "network_algorithm_version": NETWORK_ALGORITHM_VERSION,
        "tracking_object_type": TRACKING_OBJECT_TYPE,
        "voxel_size_zyx_um": VOXEL_SIZE_ZYX,
        "input_coords_are_physical": INPUT_COORDS_ARE_PHYSICAL,
        "dt_seconds": DT_SECONDS,
        "tracking_profile": TRACKING_PROFILES[TRACKING_OBJECT_TYPE],
        "min_instance_points": MIN_INSTANCE_POINTS,
        "use_pcd_shape_tiebreaker": USE_PCD_SHAPE_TIEBREAKER,
        "network_distance_um": NETWORK_DISTANCE_UM,
        "rgb_cross_frame_matching": False,
        "morphology_csv_required": False,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"DONE: {cell_dir.name}")
    print(f"Output: {out_dir}")


def resolve_cell_dirs() -> List[Path]:
    root = Path(BASE_ROOT_DIR)
    if CELL_FOLDER_NAMES:
        return [root / name for name in CELL_FOLDER_NAMES]
    # If the root itself looks like one cell, process it directly.
    if (root / SEGMENTATION_DIR_NAME).exists():
        return [root]
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / SEGMENTATION_DIR_NAME).exists()],
        key=lambda p: natural_key(p.name),
    )


def _synthetic_crossing_test() -> pd.DataFrame:
    """Internal sanity test: two tracks cross while moving in opposite directions."""
    frames: List[List[LocalInstance]] = []
    xs_a = [0.0, 0.4, 0.8, 1.2, 1.6]
    xs_b = [2.0, 1.6, 1.2, 0.8, 0.4]
    for f, (xa, xb) in enumerate(zip(xs_a, xs_b)):
        dets = []
        for lid, x in enumerate((xa, xb), start=1):
            p = np.array([x, 0.0, 0.0], dtype=float)
            dets.append(LocalInstance(
                frame=f, frame_name=f"frame_{f:02d}", local_instance_id=lid,
                centroid_raw=p.copy(), centroid_um=p.copy(), size_count=100,
                voxel_rgb=(lid, 0, 0), pcd_rgb=(lid, 0, 0),
                shape_points_centered_um=np.array([[0.0, 0.0, 0.0]], dtype=float),
            ))
        frames.append(dets)
    old = globals()["VOXEL_SIZE_ZYX"]
    try:
        globals()["VOXEL_SIZE_ZYX"] = (1.0, 1.0, 1.0)
        tracked, _links, _rej = track_instances(frames)
    finally:
        globals()["VOXEL_SIZE_ZYX"] = old
    return tracked


def track_color_float(track_id: int) -> np.ndarray:
    return np.asarray(deterministic_track_rgb(track_id), dtype=float) / 255.0




def find_tif_for_frame(tif_dir: Path, frame_name: str) -> Optional[Path]:
    """Find exact frame_name.tif/tiff first, then a conservative stem match."""
    direct = [
        tif_dir / f"{frame_name}.tif",
        tif_dir / f"{frame_name}.tiff",
        tif_dir / f"{frame_name}.TIF",
        tif_dir / f"{frame_name}.TIFF",
    ]
    for p in direct:
        if p.exists():
            return p

    # Conservative fallback: same stem ignoring case.
    target = frame_name.lower()
    candidates = []
    if tif_dir.exists():
        for p in tif_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
                if p.stem.lower() == target:
                    candidates.append(p)
    if candidates:
        return sorted(candidates, key=lambda p: natural_key(p.name))[0]
    return None


def read_tif_as_2d(path: Path) -> np.ndarray:
    """Read one TIFF and reduce it to a 2D array without per-frame normalization."""
    img = np.asarray(tifffile.imread(path))
    img = np.squeeze(img)

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3].astype(float)
        img = np.mean(rgb, axis=-1)

    while img.ndim > 2:
        img = np.max(img, axis=0)

    return img.astype(float)


def tif_to_display_image(path: Path, display_range=None) -> np.ndarray:
    """Read TIFF and convert to a display-ready 2D image using a FIXED display range."""
    img = read_tif_as_2d(path)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=float)

    if display_range is None:
        lo_p, hi_p = BACKGROUND_PERCENTILES
        lo = float(np.nanpercentile(img[finite], lo_p))
        hi = float(np.nanpercentile(img[finite], hi_p))
    else:
        lo, hi = map(float, display_range)

    if not np.isfinite(lo):
        lo = float(np.nanmin(img[finite]))
    if not np.isfinite(hi):
        hi = float(np.nanmax(img[finite]))
    if hi <= lo:
        hi = lo + 1.0

    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return img


def compute_fixed_display_range(tif_dir: Path, frame_names) -> Optional[Tuple[float, float]]:
    paths = []
    for name in frame_names:
        p = find_tif_for_frame(tif_dir, name)
        if p is not None:
            paths.append(p)
    if not paths:
        return None

    if len(paths) > DISPLAY_RANGE_SAMPLE_FRAMES and DISPLAY_RANGE_SAMPLE_FRAMES > 0:
        idx = np.linspace(0, len(paths)-1, DISPLAY_RANGE_SAMPLE_FRAMES, dtype=int)
        paths = [paths[i] for i in idx]

    values = []
    lo_p, hi_p = BACKGROUND_PERCENTILES
    for p in paths:
        try:
            img = read_tif_as_2d(p)
            finite = np.isfinite(img)
            if np.any(finite):
                values.append(float(np.nanpercentile(img[finite], lo_p)))
                values.append(float(np.nanpercentile(img[finite], hi_p)))
        except Exception:
            pass

    if not values:
        return None

    # Aggregate low/high robustly across sampled frames.
    lows = values[0::2]
    highs = values[1::2]
    lo = float(np.median(lows))
    hi = float(np.median(highs))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def choose_video_writer():
    """
    Returns:
        extension, writer_factory
    """
    ffmpeg_available = animation.writers.is_available("ffmpeg") or shutil.which("ffmpeg") is not None
    if PREFER_MP4 and ffmpeg_available:
        def make_writer():
            return animation.FFMpegWriter(
                fps=FPS,
                codec="libx264",
                bitrate=5000,
                extra_args=["-pix_fmt", "yuv420p"],
            )
        return ".mp4", make_writer

    def make_writer():
        return animation.PillowWriter(fps=FPS)

    return ".gif", make_writer


def frame_subset(tracked: pd.DataFrame, frame: int) -> pd.DataFrame:
    return tracked[tracked["frame"].astype(int) == int(frame)].copy()


def history_subset(group: pd.DataFrame, frame: int) -> pd.DataFrame:
    g = group[group["frame"].astype(int) <= int(frame)].sort_values("frame")
    if TRAIL_LENGTH and TRAIL_LENGTH > 0:
        min_frame = int(frame) - int(TRAIL_LENGTH) + 1
        g = g[g["frame"].astype(int) >= min_frame]
    return g


def compute_raw_xy_limits(tracked: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x = pd.to_numeric(tracked["centroid_x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(tracked["centroid_y"], errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return (0.0, 1.0), (1.0, 0.0)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    xpad = max(2.0, 0.05 * max(1.0, xmax - xmin))
    ypad = max(2.0, 0.05 * max(1.0, ymax - ymin))

    xlim = (xmin - xpad, xmax + xpad)
    if INVERT_Y:
        ylim = (ymax + ypad, ymin - ypad)
    else:
        ylim = (ymin - ypad, ymax + ypad)

    return xlim, ylim


def apply_xy_limits(ax, default_xlim, default_ylim, background_shape=None):
    if FIXED_XLIM is not None:
        ax.set_xlim(FIXED_XLIM)
    elif background_shape is not None:
        ax.set_xlim(0, background_shape[1] - 1)
    else:
        ax.set_xlim(default_xlim)

    if FIXED_YLIM is not None:
        ax.set_ylim(FIXED_YLIM)
    elif background_shape is not None:
        if INVERT_Y:
            ax.set_ylim(background_shape[0] - 1, 0)
        else:
            ax.set_ylim(0, background_shape[0] - 1)
    else:
        ax.set_ylim(default_ylim)


def draw_track_tails_2d(
    ax,
    tracked: pd.DataFrame,
    frame: int,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    draw_current: bool = True,
    draw_labels: bool = True,
):
    """Draw history tail and current centroid for every persistent track."""
    current = frame_subset(tracked, frame)
    current_ids = set(current["track_id"].astype(int).tolist())

    for tid in sorted(current_ids):
        g = tracked[tracked["track_id"].astype(int) == int(tid)]
        h = history_subset(g, frame)
        if h.empty:
            continue

        color = track_color_float(tid)
        xx = pd.to_numeric(h[x_col], errors="coerce").to_numpy(float)
        yy = pd.to_numeric(h[y_col], errors="coerce").to_numpy(float)

        valid = np.isfinite(xx) & np.isfinite(yy)
        xx, yy = xx[valid], yy[valid]
        if len(xx) == 0:
            continue

        # Draw older -> newer segments with increasing opacity.
        if len(xx) >= 2:
            for i in range(len(xx) - 1):
                frac = (i + 1) / max(1, len(xx) - 1)
                alpha = 0.20 + 0.70 * frac
                ax.plot(
                    xx[i:i+2], yy[i:i+2],
                    color=color,
                    linewidth=TRAIL_LINEWIDTH,
                    alpha=alpha,
                    zorder=3,
                )

        if draw_current:
            ax.scatter(
                [xx[-1]], [yy[-1]],
                s=CENTROID_SIZE,
                color=[color],
                edgecolors="white",
                linewidths=0.20,
                zorder=6,
            )

    if draw_labels and SHOW_TRACK_ID and len(current) <= MAX_TRACK_LABELS_PER_FRAME:
        for r in current.itertuples(index=False):
            tid = int(r.track_id)
            x = float(getattr(r, x_col))
            y = float(getattr(r, y_col))
            ax.text(
                x + 2.0,
                y - 2.0,
                str(tid),
                fontsize=6,
                color=track_color_float(tid),
                zorder=7,
            )


def draw_network_edges_2d(
    ax,
    tracked: pd.DataFrame,
    edges: pd.DataFrame,
    frame: int,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
):
    if edges.empty:
        return

    ft = frame_subset(tracked, frame)
    if ft.empty:
        return

    pos = {}
    for r in ft.itertuples(index=False):
        pos[int(r.track_id)] = (
            float(getattr(r, x_col)),
            float(getattr(r, y_col)),
        )

    fe = edges[edges["frame"].astype(int) == int(frame)]
    for e in fe.itertuples(index=False):
        a = int(e.track_i)
        b = int(e.track_j)
        if a not in pos or b not in pos:
            continue
        xa, ya = pos[a]
        xb, yb = pos[b]

        # Persistent edges slightly darker; new edges slightly stronger.
        is_new = bool(getattr(e, "is_new_edge", False))
        alpha = 0.78 if is_new else 0.45
        lw = NETWORK_LINEWIDTH * (1.35 if is_new else 1.0)

        ax.plot(
            [xa, xb], [ya, yb],
            linewidth=lw,
            alpha=alpha,
            color="white",
            zorder=4,
        )


def load_tracked_pointcloud_for_frame(
    output_dir: Path,
    frame_name: str,
) -> Optional[np.ndarray]:
    """
    Returns array:
        x, y, z, r, g, b, track_id

    Priority:
        tracked PCD
        tracked voxel
    """
    pcd = output_dir / "tracked_surface_pointclouds" / f"{frame_name}_pcd_tracked.txt"
    if pcd.exists() and pcd.stat().st_size > 0:
        arr = read_numeric_txt(pcd)
        if arr.shape[1] >= 10:
            return np.column_stack([
                arr[:, 0:3],
                arr[:, 6:9],
                arr[:, 9],
            ]).astype(float)

    voxel = output_dir / "tracked_voxel_pointclouds" / f"{frame_name}_voxel_tracked.txt"
    if voxel.exists() and voxel.stat().st_size > 0:
        arr = read_numeric_txt(voxel)
        if arr.shape[1] >= 7:
            return np.column_stack([
                arr[:, 0:3],
                arr[:, 3:6],
                arr[:, 6],
            ]).astype(float)

    return None


def save_raw_tif_instance_overlay_video(
    tracked: pd.DataFrame,
    frame_manifest: pd.DataFrame,
    output_dir: Path,
    video_dir: Path,
    basename: str,
    tif_dir: Path,
):
    """
    Create the clean TIFF + instance-overlay video directly in pixel space.

    IMPORTANT:
    - No matplotlib figure.
    - No axes, title, padding, or canvas outside the original image.
    - Every output frame is EXACTLY ORIGINAL_HEIGHT x ORIGINAL_WIDTH pixels.
    - Missing TIFFs raise an error instead of producing blank black/white frames.
    - The tracked PCD/voxel XY coordinates are painted directly into the image.
    """
    frames = sorted(tracked["frame"].astype(int).unique().tolist())
    if not frames:
        return None

    frame_name_lookup = {
        int(r.frame): str(r.frame_name)
        for r in frame_manifest.itertuples(index=False)
    }
    frame_names = [
        frame_name_lookup.get(int(frame), f"frame_{int(frame):04d}")
        for frame in frames
    ]

    display_range = compute_fixed_display_range(tif_dir, frame_names)
    if display_range is None:
        raise RuntimeError("Could not determine a fixed TIFF display range.")

    ext = ".mp4"
    out_path = video_dir / f"{basename}{ext}"

    # imageio-ffmpeg writes the supplied RGB arrays directly. macro_block_size=1
    # prevents automatic resizing/padding of 2400x2400 frames.
    try:
        writer = imageio.get_writer(
            out_path,
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
    except Exception:
        writer = imageio.get_writer(
            out_path,
            fps=FPS,
            macro_block_size=1,
        )

    png_dir = video_dir / f"{basename}_frames"
    if SAVE_RENDERED_PNG_FRAMES:
        png_dir.mkdir(parents=True, exist_ok=True)

    try:
        for frame in frames:
            frame = int(frame)
            frame_name = frame_name_lookup.get(frame, f"frame_{frame:04d}")
            tif_path = find_tif_for_frame(tif_dir, frame_name)
            if tif_path is None:
                raise FileNotFoundError(
                    f"Missing TIFF for frame '{frame_name}'. "
                    "The clean overlay video will not substitute a blank frame."
                )

            gray = tif_to_display_image(
                tif_path,
                display_range=display_range,
            )

            if gray.shape != (ORIGINAL_HEIGHT, ORIGINAL_WIDTH):
                raise ValueError(
                    f"Frame '{frame_name}' has image shape {gray.shape}, "
                    f"but this video is configured for exactly "
                    f"{ORIGINAL_HEIGHT}x{ORIGINAL_WIDTH}. "
                    "No resizing/padding is performed."
                )

            base_u8 = np.clip(np.rint(gray * 255.0), 0, 255).astype(np.uint8)
            rgb_img = np.repeat(base_u8[..., None], 3, axis=2)

            cloud = load_tracked_pointcloud_for_frame(output_dir, frame_name)
            if cloud is not None and len(cloud):
                x = np.rint(cloud[:, 0]).astype(np.int64)
                y = np.rint(cloud[:, 1]).astype(np.int64)
                colors = np.clip(np.rint(cloud[:, 3:6]), 0, 255).astype(np.uint8)

                valid = (
                    np.isfinite(cloud[:, 0])
                    & np.isfinite(cloud[:, 1])
                    & (x >= 0)
                    & (x < ORIGINAL_WIDTH)
                    & (y >= 0)
                    & (y < ORIGINAL_HEIGHT)
                )

                x = x[valid]
                y = y[valid]
                colors = colors[valid]

                if len(x):
                    # If several Z points project onto the same XY pixel, the last
                    # one wins; because all points of a persistent instance have
                    # the same track color this preserves the projected instance.
                    overlay = rgb_img.copy()
                    overlay[y, x] = colors
                    mask = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH), dtype=bool)
                    mask[y, x] = True

                    alpha = float(np.clip(INSTANCE_OVERLAY_ALPHA, 0.0, 1.0))
                    blended = rgb_img[mask].astype(np.float32) * (1.0 - alpha) \
                              + overlay[mask].astype(np.float32) * alpha
                    rgb_img[mask] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

            if SAVE_RENDERED_PNG_FRAMES:
                imageio.imwrite(
                    png_dir / f"frame_{frame:04d}.png",
                    rgb_img,
                )

            writer.append_data(rgb_img)
    finally:
        writer.close()

    return out_path


def save_2d_video(
    tracked: pd.DataFrame,
    edges: pd.DataFrame,
    frame_manifest: pd.DataFrame,
    output_dir: Path,
    video_dir: Path,
    basename: str,
    show_network: bool,
    use_tif_background: bool,
    use_pointcloud_background: bool,
    tif_dir: Path,
    draw_tracks: bool = True,
    draw_centroids: bool = True,
    draw_labels: bool = True,
):
    """
    Generate a 2D overlay video with an EXACT 2400x2400 canvas and NO extra
    axes/title/padding for every frame.
    """
    frames = sorted(tracked["frame"].astype(int).unique().tolist())
    if not frames:
        return None

    ext, writer_factory = choose_video_writer()
    out_path = video_dir / f"{basename}{ext}"

    png_dir = video_dir / f"{basename}_frames"
    if SAVE_RENDERED_PNG_FRAMES:
        png_dir.mkdir(parents=True, exist_ok=True)

    frame_name_lookup = {
        int(r.frame): str(r.frame_name)
        for r in frame_manifest.itertuples(index=False)
    }
    frame_names_all = [frame_name_lookup.get(int(frame), f"frame_{int(frame):04d}") for frame in frames]

    reference_display_range = None
    if use_tif_background:
        reference_display_range = compute_fixed_display_range(tif_dir, frame_names_all)

    fig = plt.figure(
        figsize=(ORIGINAL_WIDTH / DPI, ORIGINAL_HEIGHT / DPI),
        dpi=DPI,
        frameon=False,
        facecolor="black",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_facecolor("black")

    def render(frame_index: int):
        frame = int(frames[frame_index])
        frame_name = frame_name_lookup.get(frame, f"frame_{frame:04d}")
        ax.clear()
        ax.set_axis_off()
        ax.set_facecolor("black")

        if use_tif_background:
            tif_path = find_tif_for_frame(tif_dir, frame_name)
            if tif_path is None:
                raise FileNotFoundError(f"Missing TIFF for frame '{frame_name}'")
            img = tif_to_display_image(tif_path, display_range=reference_display_range)
            if img.shape != (ORIGINAL_HEIGHT, ORIGINAL_WIDTH):
                raise ValueError(
                    f"Frame '{frame_name}' has image shape {img.shape}, expected exactly "
                    f"({ORIGINAL_HEIGHT}, {ORIGINAL_WIDTH})"
                )
        else:
            img = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH), dtype=float)

        ax.imshow(
            img,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="upper" if INVERT_Y else "lower",
            interpolation="nearest",
            extent=(0, ORIGINAL_WIDTH, ORIGINAL_HEIGHT, 0) if INVERT_Y else (0, ORIGINAL_WIDTH, 0, ORIGINAL_HEIGHT),
            aspect="equal",
            zorder=0,
        )

        if use_pointcloud_background:
            cloud = load_tracked_pointcloud_for_frame(output_dir, frame_name)
            if cloud is not None and len(cloud):
                xyz = cloud[:, 0:3]
                rgb = np.clip(cloud[:, 3:6] / 255.0, 0, 1)
                valid = (
                    np.isfinite(xyz[:, 0])
                    & np.isfinite(xyz[:, 1])
                    & (xyz[:, 0] >= 0) & (xyz[:, 0] <= ORIGINAL_WIDTH)
                    & (xyz[:, 1] >= 0) & (xyz[:, 1] <= ORIGINAL_HEIGHT)
                )
                ax.scatter(
                    xyz[valid, 0],
                    xyz[valid, 1],
                    s=POINTCLOUD_POINT_SIZE,
                    c=rgb[valid],
                    linewidths=0,
                    alpha=INSTANCE_OVERLAY_ALPHA,
                    zorder=2,
                )

        if show_network:
            draw_network_edges_2d(
                ax,
                tracked=tracked,
                edges=edges,
                frame=frame,
                x_col="centroid_x",
                y_col="centroid_y",
            )

        if draw_tracks:
            draw_track_tails_2d(
                ax,
                tracked=tracked,
                frame=frame,
                x_col="centroid_x",
                y_col="centroid_y",
                draw_current=draw_centroids,
                draw_labels=draw_labels,
            )

        ax.set_xlim(0, ORIGINAL_WIDTH)
        ax.set_ylim(ORIGINAL_HEIGHT, 0) if INVERT_Y else ax.set_ylim(0, ORIGINAL_HEIGHT)
        ax.set_aspect("equal", adjustable="box")
        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()

        if SAVE_RENDERED_PNG_FRAMES:
            fig.savefig(
                png_dir / f"frame_{frame:04d}.png",
                dpi=DPI,
                bbox_inches=None,
                pad_inches=0,
                facecolor='black',
            )
        return []

    ani = animation.FuncAnimation(
        fig,
        render,
        frames=len(frames),
        interval=1000.0 / max(FPS, 1),
        blit=False,
        repeat=False,
    )

    writer = writer_factory()
    ani.save(out_path, writer=writer, dpi=DPI)
    plt.close(fig)
    return out_path


def save_3d_video(
    tracked: pd.DataFrame,
    edges: pd.DataFrame,
    video_dir: Path,
    basename: str,
):
    frames = sorted(tracked["frame"].astype(int).unique().tolist())
    if not frames:
        return None

    z = pd.to_numeric(tracked["centroid_z_um"], errors="coerce").to_numpy(float)
    z = z[np.isfinite(z)]
    if len(z) == 0 or float(np.ptp(z)) < 1e-9:
        print("[SKIP 3D] All Z values are identical; dataset is effectively 2D.")
        return None

    xall = pd.to_numeric(tracked["centroid_x_um"], errors="coerce").to_numpy(float)
    yall = pd.to_numeric(tracked["centroid_y_um"], errors="coerce").to_numpy(float)
    zall = pd.to_numeric(tracked["centroid_z_um"], errors="coerce").to_numpy(float)

    def lim(values):
        values = values[np.isfinite(values)]
        lo, hi = float(np.min(values)), float(np.max(values))
        pad = max(0.1, 0.05 * max(0.1, hi - lo))
        return lo - pad, hi + pad

    xlim = lim(xall)
    ylim = lim(yall)
    zlim = lim(zall)

    ext, writer_factory = choose_video_writer()
    out_path = video_dir / f"{basename}{ext}"

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    def render(frame_index: int):
        frame = int(frames[frame_index])
        ax.clear()
        ax.set_facecolor("black")

        current = frame_subset(tracked, frame)
        current_ids = set(current["track_id"].astype(int).tolist())

        for tid in sorted(current_ids):
            g = tracked[tracked["track_id"].astype(int) == int(tid)]
            h = history_subset(g, frame)
            if h.empty:
                continue

            color = track_color_float(tid)
            xx = h["centroid_x_um"].to_numpy(float)
            yy = h["centroid_y_um"].to_numpy(float)
            zz = h["centroid_z_um"].to_numpy(float)

            ax.plot(
                xx, yy, zz,
                color=color,
                linewidth=TRAIL_LINEWIDTH,
                alpha=0.80,
            )
            ax.scatter(
                [xx[-1]], [yy[-1]], [zz[-1]],
                s=CENTROID_SIZE,
                color=[color],
                edgecolors="white",
                linewidths=0.20,
            )

            if SHOW_TRACK_ID and len(current) <= MAX_TRACK_LABELS_PER_FRAME:
                ax.text(
                    xx[-1], yy[-1], zz[-1],
                    str(tid),
                    fontsize=6,
                    color=color,
                )

        if not edges.empty:
            pos = {
                int(r.track_id): (
                    float(r.centroid_x_um),
                    float(r.centroid_y_um),
                    float(r.centroid_z_um),
                )
                for r in current.itertuples(index=False)
            }
            fe = edges[edges["frame"].astype(int) == frame]
            for e in fe.itertuples(index=False):
                a, b = int(e.track_i), int(e.track_j)
                if a not in pos or b not in pos:
                    continue
                pa, pb = pos[a], pos[b]
                ax.plot(
                    [pa[0], pb[0]],
                    [pa[1], pb[1]],
                    [pa[2], pb[2]],
                    color="white",
                    alpha=0.35,
                    linewidth=NETWORK_LINEWIDTH,
                )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X (µm)", color="white")
        ax.set_ylabel("Y (µm)", color="white")
        ax.set_zlabel("Z (µm)", color="white")
        ax.tick_params(colors="white")
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        ax.set_title(
            f"3D persistent tracks | frame {frame}",
            color="white",
        )
        fig.tight_layout()
        return []

    ani = animation.FuncAnimation(
        fig,
        render,
        frames=len(frames),
        interval=1000.0 / max(FPS, 1),
        blit=False,
        repeat=False,
    )

    ani.save(out_path, writer=writer_factory(), dpi=DPI)
    plt.close(fig)
    return out_path




# =============================================================================
# Reviewed mask -> voxel / mesh / PCD
# =============================================================================

def mask_to_mesh_marching_cubes_zyx(
        mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        level=0.5,
        pad=True,
        pre_smooth_sigma_um=0.0,
):
    """
    Convert 3D binary mask to mesh using marching cubes.

    Input:
        mask: shape = (Z, Y, X)
        voxel_size_zyx: physical voxel size, e.g. (0.16, 0.04, 0.04)

    Output:
        verts_zyx: vertex coordinates, order = (z, y, x), physical unit
        faces: triangle indices
    """
    mask = (mask != 0)

    z_um, y_um, x_um = voxel_size_zyx

    if pad:
        mask_work = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    else:
        mask_work = mask.copy()

    volume = mask_work.astype(np.float32)

    # Optional: very light volume smoothing before marching cubes.
    # Do not use too large sigma, otherwise thin mitochondria may shrink or merge.
    if pre_smooth_sigma_um is not None and pre_smooth_sigma_um > 0:
        sigma_zyx = (
            pre_smooth_sigma_um / z_um,
            pre_smooth_sigma_um / y_um,
            pre_smooth_sigma_um / x_um,
        )
        volume = ndi.gaussian_filter(volume, sigma=sigma_zyx)

    verts_zyx, faces, normals_zyx, values = measure.marching_cubes(
        volume,
        level=level,
        spacing=voxel_size_zyx,
        allow_degenerate=False,
    )

    # If padded by 1 voxel, subtract the physical padding offset.
    if pad:
        offset = np.asarray(voxel_size_zyx, dtype=np.float32)
        verts_zyx = verts_zyx - offset

    return verts_zyx.astype(np.float32), faces.astype(np.int32)


def build_vertex_adjacency(n_vertices, faces):
    """
    Build vertex adjacency list from triangle faces.
    """
    adjacency = [set() for _ in range(n_vertices)]

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

        adjacency[a].add(b)
        adjacency[a].add(c)

        adjacency[b].add(a)
        adjacency[b].add(c)

        adjacency[c].add(a)
        adjacency[c].add(b)

    adjacency = [np.array(list(nei), dtype=np.int32) for nei in adjacency]
    return adjacency


def taubin_smooth_mesh(
        verts,
        faces,
        iterations=10,
        lamb=0.5,
        mu=-0.53,
):
    """
    Taubin smoothing.
    Better than simple Laplacian smoothing because it reduces shrinkage.

    verts are in ZYX physical coordinates.
    """
    verts = verts.astype(np.float32).copy()
    adjacency = build_vertex_adjacency(len(verts), faces)

    def laplacian_step(v, step):
        new_v = v.copy()

        for i, nei in enumerate(adjacency):
            if len(nei) == 0:
                continue

            mean_neighbor = v[nei].mean(axis=0)
            new_v[i] = v[i] + step * (mean_neighbor - v[i])

        return new_v

    for _ in range(iterations):
        verts = laplacian_step(verts, lamb)
        verts = laplacian_step(verts, mu)

    return verts.astype(np.float32)


def voxel_mask_to_smoothed_mesh(
        mask,
        voxel_size_zyx=(0.16, 0.04, 0.04),
        mesh_smooth_iterations=6,
        pre_smooth_sigma_um=0.0,

):
    """
    Full pipeline:
        voxel mask -> marching cubes mesh -> Taubin smoothing -> surface point cloud + normals

    Output coordinate order:
        z, y, x, nz, ny, nx
    """
    mask = (mask != 0)

    verts_zyx, faces = mask_to_mesh_marching_cubes_zyx(
        mask,
        voxel_size_zyx=voxel_size_zyx,
        level=0.5,
        pad=True,
        pre_smooth_sigma_um=pre_smooth_sigma_um,
    )

    print("Raw mesh vertices:", verts_zyx.shape)
    print("Raw mesh faces:", faces.shape)

    if mesh_smooth_iterations is not None and mesh_smooth_iterations > 0:
        verts_smooth_zyx = taubin_smooth_mesh(
            verts_zyx,
            faces,
            iterations=mesh_smooth_iterations,
            lamb=0.5,
            mu=-0.53,
        )
    else:
        verts_smooth_zyx = verts_zyx

    return {
        "verts_zyx": verts_zyx,
        "verts_smooth_zyx": verts_smooth_zyx,
        "faces": faces,
    }


def read_tif(img_path):
    try:
        with tifffile.TiffFile(img_path) as tif:
            data = tif.asarray()
    except AttributeError as exc:
        if "newbyteorder" not in str(exc):
            raise

        from PIL import Image, ImageSequence

        with Image.open(img_path) as img:
            frames = [np.asarray(frame) for frame in ImageSequence.Iterator(img)]

        if len(frames) == 0:
            raise ValueError(f"No frames found in TIFF: {img_path}") from exc

        data = np.stack(frames, axis=0) if len(frames) > 1 else frames[0]
    print("************** read tif *******************")
    print(f"Original Shape: {data.shape}")
    print(f"Original Data Type: {data.dtype}")

        # 如果是2D图像，添加一个前置维度
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
        print("Converted to 3D by adding leading dimension.")

    print(f"Processed Shape: {data.shape}")
    print("************** read tif *******************")
    return data


def _laplacian_smooth_vertices_xyz(vertices, faces, iterations=1, step=0.5):
    """Small NumPy fallback for Open3D Laplacian smoothing."""
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)

    adjacency = [set() for _ in range(len(vertices))]
    for a, b, c in faces:
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    for _ in range(max(0, int(iterations))):
        old_vertices = vertices.copy()
        for i, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            idx = np.fromiter(neighbors, dtype=np.int64)
            mean_neighbor = old_vertices[idx].mean(axis=0)
            vertices[i] = old_vertices[i] + float(step) * (mean_neighbor - old_vertices[i])

    return vertices


def _compute_vertex_normals_xyz(vertices, faces):
    """Compute per-vertex normals from OBJ triangle winding."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / (lengths + 1e-12)

    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for col in range(3):
        np.add.at(vertex_normals, faces[:, col], face_normals)

    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (lengths + 1e-12)
    return vertex_normals


def _read_obj_numpy(mesh_path):
    """Read vertices/faces from the simple OBJ files written by this script."""
    vertices = []
    faces = []

    with open(mesh_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()[1:4]
                # Support both "f 1 2 3" and "f 1/... 2/... 3/..." forms.
                face = [int(item.split('/')[0]) - 1 for item in parts]
                faces.append(face)

    return {
        'vertices': np.asarray(vertices, dtype=np.float64),
        'faces': np.asarray(faces, dtype=np.int64),
    }


def read_mesh(mesh_path, iterations=3):
    """
    Read OBJ and apply the original post-read Laplacian smoothing.

    Uses Open3D when available. If Open3D is not installed, a NumPy fallback
    keeps the pipeline runnable without changing the file interface.
    """
    if o3d is not None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        return mesh

    mesh = _read_obj_numpy(mesh_path)
    mesh['vertices'] = _laplacian_smooth_vertices_xyz(
        mesh['vertices'],
        mesh['faces'],
        iterations=iterations,
    )
    return mesh


def save_obj_from_zyx_mesh(save_path, verts_zyx, faces):
    """
    Save mesh to OBJ for Blender/MeshLab.

    Internal verts are ZYX.
    OBJ writes XYZ.
    """
    verts_xyz = verts_zyx[:, [2, 1, 0]]

    with open(save_path, "w", encoding="utf-8") as f:
        for v in verts_xyz:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")

    print(f"OBJ saved at: {save_path}")


def mesh_2_pcd(mesh):
    """Convert a triangle mesh to x,y,z,nx,ny,nz surface points."""
    if o3d is not None and not isinstance(mesh, dict):
        vertices = np.asarray(mesh.vertices)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()
        return np.hstack((np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)))

    vertices = np.asarray(mesh['vertices'], dtype=np.float64)
    faces = np.asarray(mesh['faces'], dtype=np.int64)
    normals = _compute_vertex_normals_xyz(vertices, faces)
    return np.hstack((vertices, normals))


def reviewed_mask_to_voxel_point_cloud(mask_image: np.ndarray) -> np.ndarray:
    """
    Convert a manually reviewed binary/intensity mask to raw foreground voxels.

    Output columns:
        x, y, z, intensity

    Coordinates remain voxel indices, matching the downstream segmentation code.
    """
    arr = np.asarray(mask_image)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(
            f"Reviewed mask must be 2D or 3D, got shape={arr.shape}"
        )

    foreground = arr != 0
    z_idx, y_idx, x_idx = np.where(foreground)

    # Keep the reviewed mask intensity when available.
    # For a binary/bool mask, use 255 for foreground.
    if arr.dtype == np.bool_:
        intensity = np.full(len(x_idx), 255, dtype=np.float32)
    else:
        intensity = arr[z_idx, y_idx, x_idx].astype(np.float32)
        if intensity.size and np.nanmax(intensity) <= 1:
            intensity = intensity * 255.0

    points = np.column_stack((
        x_idx.astype(np.float32),
        y_idx.astype(np.float32),
        z_idx.astype(np.float32),
        intensity.astype(np.float32),
    ))

    return points.astype(np.float32)


def run_reviewed_mask_to_pcd(cell_dir: Path) -> None:
    """
    Stage 1.

    Read ONLY manually reviewed *_mask.tif files from processed_output and create:
        *_voxel.txt
        *_mesh.obj
        *_pcd.txt

    This function never creates or modifies the mask itself.
    """
    cell_dir = Path(cell_dir)
    processed_dir = cell_dir / SEGMENTATION_DIR_NAME

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"processed_output not found: {processed_dir}"
        )

    mask_paths = sorted(
        [
            p for p in processed_dir.iterdir()
            if p.is_file() and p.name.lower().endswith(MASK_SUFFIX.lower())
        ],
        key=lambda p: natural_key(p.name),
    )

    if not mask_paths:
        print(f"[MASK -> PCD] No reviewed {MASK_SUFFIX} files found in {processed_dir}")
        return

    print(f"[MASK -> PCD] {cell_dir.name}: {len(mask_paths)} reviewed mask(s)")

    for i, mask_path in enumerate(mask_paths, start=1):
        frame_name = mask_path.name[:-len(MASK_SUFFIX)]

        voxel_path = processed_dir / f"{frame_name}_voxel.txt"
        obj_path = processed_dir / f"{frame_name}_mesh.obj"
        pcd_path = processed_dir / f"{frame_name}_pcd.txt"

        print("-" * 73)
        print(f"[MASK -> PCD] {i}/{len(mask_paths)}  {frame_name}")
        print("-" * 73)
        print(f"Reviewed mask: {mask_path}")

        mask_image = read_tif(mask_path)
        mask_bool = mask_image != 0

        if not np.any(mask_bool):
            print(f"[SKIP EMPTY MASK] {mask_path}")
            continue

        # Always regenerate raw foreground voxels from the REVIEWED mask.
        # This prevents stale *_voxel.txt files after manual mask correction.
        voxel_points = reviewed_mask_to_voxel_point_cloud(mask_image)
        save_pcd2txt(voxel_points, voxel_path)
        print(f"[SAVED VOXEL] {voxel_path}")

        rebuild_obj = FORCE_REBUILD_MASK_GEOMETRY or (not obj_path.exists())

        if rebuild_obj:
            mesh_data = voxel_mask_to_smoothed_mesh(
                mask_bool,
                voxel_size_zyx=MASK_MESH_VOXEL_SIZE_ZYX,
                mesh_smooth_iterations=MASK_MESH_SMOOTH_ITERATIONS,
            )

            save_obj_from_zyx_mesh(
                obj_path,
                mesh_data["verts_smooth_zyx"],
                mesh_data["faces"],
            )
        else:
            print(f"[USE EXISTING OBJ] {obj_path}")

        mesh = read_mesh(
            obj_path,
            iterations=MASK_POST_READ_LAPLACIAN_ITERATIONS,
        )

        pcd = mesh_2_pcd(mesh)
        save_pcd2txt(pcd, pcd_path)
        print(f"[SAVED PCD] {pcd_path}")



# =============================================================================
# Stage wrappers
# =============================================================================

def run_instance_segmentation(cell_dir: Path) -> None:
    """Stage 2: create/reuse instance point clouds and calculate morphology."""
    cell_dir = Path(cell_dir)
    input_path = cell_dir / SEGMENTATION_DIR_NAME
    output_path = input_path
    output_path.mkdir(parents=True, exist_ok=True)

    save_all_morphology_path = output_path / "morphology_cell_mean_all.csv"

    if save_all_morphology_path.exists() and not OVERWRITE_MORPHOLOGY:
        print(f"[SKIP CELL-MEAN MORPHOLOGY TABLE EXISTS] {save_all_morphology_path}")
        return

    imgs_list = collect_input_base_names(input_path)
    print(f"[INSTANCE] {cell_dir.name}: {len(imgs_list)} frame(s)")

    morphology_tables_this_folder = []

    for i, img_name in enumerate(imgs_list, start=1):
        print("-" * 73)
        print(f"[INSTANCE] {i}/{len(imgs_list)}  {img_name}")
        print("-" * 73)

        img_name = img_name.split(".")[0]

        pcd_path = input_path / f"{img_name}_pcd.txt"
        voxel_path = input_path / f"{img_name}_voxel.txt"

        save_voxel_instance_path = output_path / f"{img_name}_voxel_instance.txt"
        save_centers_path = output_path / f"{img_name}_centers.txt"
        save_pcd_instance_path = output_path / f"{img_name}_pcd_instance.txt"

        voxel_xyzrgb = None
        raw_voxel = None
        morphology_source = None

        # A. Existing voxel instance: reuse directly.
        if save_voxel_instance_path.exists():
            print(f"[USE EXISTING VOXEL INSTANCE] {save_voxel_instance_path}")
            voxel_xyzrgb = load_point_cloud(save_voxel_instance_path)
            morphology_source = "existing_voxel_instance"
            if voxel_path.exists():
                raw_voxel = load_point_cloud(voxel_path)

        # B. Existing PCD instance only: transfer colors back to raw voxels.
        elif save_pcd_instance_path.exists():
            print(f"[USE EXISTING PCD INSTANCE] {save_pcd_instance_path}")

            if not voxel_path.exists():
                print(f"[SKIP] Missing raw voxel: {voxel_path}")
                continue

            raw_voxel = load_point_cloud(voxel_path)
            pcd_instance = load_point_cloud(save_pcd_instance_path)

            voxel_xyzrgb = color_voxels_from_existing_pcd_instance(
                voxel=raw_voxel,
                pcd_instance=pcd_instance,
                max_assign_distance=PCD_TO_VOXEL_MAX_ASSIGN_DISTANCE,
                batch_size=300000,
                workers=-1,
            )

            save_pcd2txt(voxel_xyzrgb, save_voxel_instance_path)
            print(f"[SAVED VOXEL INSTANCE FROM EXISTING PCD INSTANCE] {save_voxel_instance_path}")
            morphology_source = "existing_pcd_instance_to_voxel_instance"

        # C. No instance files: run full instance segmentation.
        else:
            if not pcd_path.exists():
                print(f"[SKIP] Missing PCD: {pcd_path}")
                continue
            if not voxel_path.exists():
                print(f"[SKIP] Missing voxel: {voxel_path}")
                continue

            pcd = load_point_cloud(pcd_path)
            raw_voxel = load_point_cloud(voxel_path)

            voxel_score = score_voxels_by_inward_normal_cones(
                pcd=pcd,
                voxel=raw_voxel,
                cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
                max_distance=CONE_MAX_DISTANCE,
                n_direction_bins=N_DIRECTION_BINS,
                min_forward_distance=0.0,
                surface_batch_size=2000,
                workers=-1,
            )

            voxel_score = filter_points_by_column_threshold(
                pointcloud=voxel_score,
                column_index=4,
                threshold=SCORE_THRESHOLD,
            )

            voxel_xyzrgb, score_centers = segment_voxels_by_nearest_score_center(
                voxel=raw_voxel,
                scored_voxel=voxel_score,
                score_column=-1,
                min_center_score=MIN_CENTER_SCORE,
                min_peak_distance=MIN_PEAK_DISTANCE,
                merge_center_distance=MERGE_CENTER_DISTANCE,
                gaussian_sigma=CENTER_GAUSSIAN_SIGMA,
                connectivity=CENTER_CONNECTIVITY,
                color_seed=INSTANCE_COLOR_SEED,
                return_centers=True,
            )

            save_pcd2txt(voxel_xyzrgb, save_voxel_instance_path)
            save_pcd2txt(score_centers, save_centers_path)
            print(f"[SAVED VOXEL INSTANCE] {save_voxel_instance_path}")
            print(f"[SAVED CENTERS] {save_centers_path}")

            pcd_xyznormal_rgb = segment_pcd_by_voxel_xyzrgb(
                pcd=pcd,
                voxel_xyzrgb=voxel_xyzrgb,
                max_assign_distance=None,
                batch_size=300000,
                workers=-1,
            )

            save_pcd2txt(pcd_xyznormal_rgb, save_pcd_instance_path)
            print(f"[SAVED PCD INSTANCE] {save_pcd_instance_path}")
            morphology_source = "new_segmentation_voxel_instance"

        # D. Morphology from colored voxel instances.
        try:
            cell_mean_df = compute_cell_mean_morphology_from_voxel_instance(
                voxel_xyzrgb=voxel_xyzrgb,
                voxel_size_zyx=SEGMENTATION_MORPH_VOXEL_SIZE_ZYX,
                coords_are_physical=SEGMENTATION_COORDS_ARE_PHYSICAL,
                image_name=img_name,
                dataset_name=DATASET_NAME,
                organelle_name=cell_dir.name,
                morphology_source=morphology_source,
                raw_voxel=raw_voxel,
                save_cell_mean_path=None,
            )
            morphology_tables_this_folder.append(cell_mean_df)
        except Exception as exc:
            print(f"[FAILED MORPHOLOGY] {img_name}: {exc}")

    if morphology_tables_this_folder:
        all_cell_mean_morphology_df = pd.concat(
            morphology_tables_this_folder,
            ignore_index=True,
        )
        all_cell_mean_morphology_df.to_csv(save_all_morphology_path, index=False)
        print(f"[SAVED CELL-MEAN MORPHOLOGY TABLE] {save_all_morphology_path}")


def generate_tracking_videos(cell_dir: Path) -> pd.DataFrame:
    """Stage 4: generate all enabled tracking/network videos for one cell."""
    cell_dir = Path(cell_dir)
    output_dir = cell_dir / SEGMENTATION_DIR_NAME
    tables_dir = output_dir / "tables"
    video_dir = output_dir / VIDEO_DIR_NAME
    video_dir.mkdir(parents=True, exist_ok=True)

    tracked_path = tables_dir / "instances_tracked_all_frames.csv"
    edges_path = tables_dir / "network_edges_by_frame.csv"
    manifest_path = tables_dir / "frame_manifest.csv"

    if not tracked_path.exists():
        raise FileNotFoundError(f"Required tracking table not found:\n{tracked_path}")

    tracked = pd.read_csv(tracked_path)

    required_columns = [
        "frame",
        "frame_name",
        "track_id",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "centroid_x_um",
        "centroid_y_um",
        "centroid_z_um",
    ]
    missing = [c for c in required_columns if c not in tracked.columns]
    if missing:
        raise ValueError(
            "instances_tracked_all_frames.csv is missing columns:\n"
            + "\n".join(missing)
        )

    if edges_path.exists():
        try:
            edges = pd.read_csv(edges_path)
        except pd.errors.EmptyDataError:
            edges = pd.DataFrame()
    else:
        edges = pd.DataFrame()

    if manifest_path.exists():
        frame_manifest = pd.read_csv(manifest_path)
    else:
        frame_manifest = (
            tracked[["frame", "frame_name"]]
            .drop_duplicates()
            .sort_values("frame")
            .reset_index(drop=True)
        )

    if BACKGROUND_TIF_DIR is None:
        tif_dir = cell_dir
    else:
        tif_dir = Path(BACKGROUND_TIF_DIR)

    print("=" * 80)
    print("TRACKING VIDEO GENERATOR")
    print("=" * 80)
    print(f"Cell:       {cell_dir}")
    print(f"Output:     {output_dir}")
    print(f"Video dir:  {video_dir}")
    print(f"Frames:     {tracked['frame'].nunique()}")
    print(f"Tracks:     {tracked['track_id'].nunique()}")
    print(f"TIFF dir:   {tif_dir}")
    print(f"Network:    {'YES' if not edges.empty else 'NO'}")
    print()

    results = []

    def run_video(label, enabled, func):
        if not enabled:
            results.append((label, "disabled", ""))
            return
        try:
            path = func()
            if path is None:
                results.append((label, "skipped", ""))
            else:
                results.append((label, "saved", str(path)))
                print(f"[SAVED] {path}")
        except Exception as exc:
            results.append((label, f"ERROR: {exc}", ""))
            print(f"[ERROR] {label}: {exc}")

    run_video(
        "01_centroid_tracks",
        MAKE_CENTROID_TRACKS,
        lambda: save_2d_video(
            tracked, edges, frame_manifest, output_dir, video_dir,
            basename="01_centroid_tracks",
            show_network=False,
            use_tif_background=False,
            use_pointcloud_background=False,
            tif_dir=tif_dir,
        ),
    )

    run_video(
        "02_centroid_tracks_network",
        MAKE_CENTROID_TRACKS_NETWORK,
        lambda: save_2d_video(
            tracked, edges, frame_manifest, output_dir, video_dir,
            basename="02_centroid_tracks_network",
            show_network=True,
            use_tif_background=False,
            use_pointcloud_background=False,
            tif_dir=tif_dir,
        ),
    )

    run_video(
        "03_tif_centroid_tracks",
        MAKE_TIF_CENTROID_TRACKS,
        lambda: save_2d_video(
            tracked, edges, frame_manifest, output_dir, video_dir,
            basename="03_tif_centroid_tracks",
            show_network=False,
            use_tif_background=True,
            use_pointcloud_background=False,
            tif_dir=tif_dir,
        ),
    )

    run_video(
        "04_tif_centroid_tracks_network",
        MAKE_TIF_CENTROID_TRACKS_NETWORK,
        lambda: save_2d_video(
            tracked, edges, frame_manifest, output_dir, video_dir,
            basename="04_tif_centroid_tracks_network",
            show_network=True,
            use_tif_background=True,
            use_pointcloud_background=False,
            tif_dir=tif_dir,
        ),
    )

    run_video(
        "05_tracked_pointcloud",
        MAKE_TRACKED_POINTCLOUD,
        lambda: save_2d_video(
            tracked, edges, frame_manifest, output_dir, video_dir,
            basename="05_tracked_pointcloud",
            show_network=False,
            use_tif_background=False,
            use_pointcloud_background=True,
            tif_dir=tif_dir,
        ),
    )

    run_video(
        "06_tracked_pointcloud_tracks_network",
        MAKE_TRACKED_POINTCLOUD_TRACKS_NETWORK,
        lambda: save_2d_video(
            tracked, edges, frame_manifest, output_dir, video_dir,
            basename="06_tracked_pointcloud_tracks_network",
            show_network=True,
            use_tif_background=False,
            use_pointcloud_background=True,
            tif_dir=tif_dir,
        ),
    )

    run_video(
        "06b_tif_instance_overlay",
        MAKE_TIF_INSTANCE_OVERLAY,
        lambda: save_raw_tif_instance_overlay_video(
            tracked=tracked,
            frame_manifest=frame_manifest,
            output_dir=output_dir,
            video_dir=video_dir,
            basename="06b_tif_instance_overlay",
            tif_dir=tif_dir,
        ),
    )

    run_video(
        "07_centroid_tracks_3d",
        MAKE_3D_CENTROID_TRACKS,
        lambda: save_3d_video(
            tracked=tracked,
            edges=edges,
            video_dir=video_dir,
            basename="07_centroid_tracks_3d",
        ),
    )

    result_df = pd.DataFrame(results, columns=["video", "status", "path"])
    result_df.to_csv(video_dir / "video_generation_manifest.csv", index=False)

    print()
    print("=" * 80)
    print("VIDEO GENERATION DONE")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print()
    print(f"Videos saved under:\n{video_dir}")

    return result_df


def main() -> None:
    cells = resolve_cell_dirs()
    if not cells:
        raise FileNotFoundError(
            f"No cell folders found under {BASE_ROOT_DIR}. "
            f"Each cell should contain {SEGMENTATION_DIR_NAME}/"
        )

    print("=" * 80)
    print("ALL-IN-ONE ORGANELLE PIPELINE")
    print("=" * 80)
    print(f"Cells to process: {len(cells)}")
    print(f"Stage 1 - reviewed mask -> PCD:  {'ON' if RUN_MASK_TO_PCD else 'OFF'}")
    print(f"Stage 2 - instance segmentation: {'ON' if RUN_INSTANCE_SEGMENTATION else 'OFF'}")
    print(f"Stage 3 - tracking/network:      {'ON' if RUN_TRACKING_NETWORK else 'OFF'}")
    print(f"Stage 4 - videos:                {'ON' if RUN_VIDEO_GENERATION else 'OFF'}")
    print("=" * 80)

    for cell in cells:
        print("\n" + "#" * 80)
        print(f"CELL: {cell}")
        print("#" * 80)

        if RUN_MASK_TO_PCD:
            print("\n[STAGE 1/4] REVIEWED MASK -> VOXEL / MESH / PCD")
            run_reviewed_mask_to_pcd(cell)

        if RUN_INSTANCE_SEGMENTATION:
            print("\n[STAGE 2/4] INSTANCE SEGMENTATION + MORPHOLOGY")
            run_instance_segmentation(cell)

        if RUN_TRACKING_NETWORK:
            print("\n[STAGE 3/4] DYNAMIC TRACKING + TEMPORAL NETWORK")
            process_cell(cell)

        if RUN_VIDEO_GENERATION:
            print("\n[STAGE 4/4] TRACKING VIDEO")
            generate_tracking_videos(cell)

    print("\n" + "=" * 80)
    print("ALL-IN-ONE PIPELINE FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()
