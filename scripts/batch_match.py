#!/usr/bin/env python3
"""Offline batch satellite matching.

Run ONCE before AIRSLAM.  Processes all drone images in a directory,
saves results to a JSON file keyed by timestamp.

Usage:
  python batch_match.py \\
    --image-dir /path/to/exported_images \\
    --map-db /path/to/satellite/tiles \\
    --tile-index /path/to/tile_index.json \\
    --sat-features /path/to/sat_features.npz \\
    --output /path/to/pre_match_results.json

Output format:
  {
    "1703738785.891234": {"lat": ..., "lon": ..., "inliers": 4, "keypoints": [...]},
    ...
  }
"""

import argparse
import bisect
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import LightGlueMatcher, SuperGlueMatcher
from svl.keypoint_pipeline.typing import (
    ImageKeyPoints,
    LightGlueConfig,
    SuperGlueConfig,
    SuperPointConfig,
)
from svl.localization.geometry import (
    CameraCalibration,
    estimate_planar_pnp,
    metric_xy_to_gps,
    satellite_pixels_to_metric,
)


def load_tile_index(json_path):
    with open(json_path) as f:
        return json.load(f)


def load_sat_features(npz_path):
    if not npz_path:
        return {}
    data = np.load(npz_path, allow_pickle=True)
    features = {}
    for name in data["__tile_names__"]:
        prefix = str(name).replace(".", "_")
        features[str(name)] = {
            "keypoints": data[f"{prefix}_keypoints"],
            "descriptors": data[f"{prefix}_descriptors"],
            "scores": data[f"{prefix}_scores"],
            "image_size": data[f"{prefix}_image_size"],
        }
    return features


def preprocess_image(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        return cv2.resize(
            img, (max_width, int(max_width * h / w)), interpolation=cv2.INTER_AREA
        )
    return img


def resize_long_side(img, target_size):
    if target_size <= 0:
        return img
    h, w = img.shape[:2]
    scale = float(target_size) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_tile_keypoints(
    name, sp, sat_features, sat_feature_cache, map_db, sat_resize_size
):
    if sat_resize_size <= 0 and sat_features and name in sat_features:
        sf = sat_features[name]
        tile_kpts = ImageKeyPoints(
            keypoints=sf["keypoints"],
            descriptors=sf["descriptors"],
            scores=sf["scores"],
            image_size=tuple(sf["image_size"]),
        )
        th, tw = sf["image_size"]
        return tile_kpts, int(th), int(tw)

    cache_key = (name, sat_resize_size)
    if cache_key in sat_feature_cache:
        return sat_feature_cache[cache_key]

    timg = cv2.imread(str(Path(map_db) / name), cv2.IMREAD_GRAYSCALE)
    if timg is None:
        return None, None, None
    if sat_resize_size > 0:
        timg = resize_long_side(timg, sat_resize_size)
    else:
        timg = preprocess_image(timg)

    tile_kpts = sp.detect_and_describe_keypoints(timg)
    th, tw = timg.shape[:2]
    sat_feature_cache[cache_key] = (tile_kpts, th, tw)
    return tile_kpts, th, tw


def pixel_to_gps(px, py, tl_lat, tl_lon, br_lat, br_lon, th, tw):
    lat = tl_lat + (py / th) * (br_lat - tl_lat)
    lon = tl_lon + (px / tw) * (br_lon - tl_lon)
    return lat, lon


def gps_to_tile_pixel(lat, lon, tl_lat, tl_lon, br_lat, br_lon, th, tw):
    px = (lon - tl_lon) / (br_lon - tl_lon) * tw
    py = (lat - tl_lat) / (br_lat - tl_lat) * th
    return px, py


METERS_PER_DEG_LAT = 111320.0


def gps_to_metric_xy(lat, lon, ref_lat, ref_lon):
    lat_mid = (lat + ref_lat) / 2.0
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(lat_mid))
    return np.array(
        [
            (lon - ref_lon) * meters_per_deg_lon,
            (lat - ref_lat) * METERS_PER_DEG_LAT,
        ],
        dtype=np.float64,
    )


def tile_center_and_radius(tile, ref_lat, ref_lon):
    tl = gps_to_metric_xy(tile["top_left_lat"], tile["top_left_lon"], ref_lat, ref_lon)
    br = gps_to_metric_xy(
        tile["bottom_right_lat"], tile["bottom_right_lon"], ref_lat, ref_lon
    )
    center = 0.5 * (tl + br)
    radius = 0.5 * float(np.linalg.norm(tl - br))
    return center, radius


def filter_tiles_by_prior(tile_index, prior_xy, ref_lat, ref_lon, search_radius_m):
    if prior_xy is None or search_radius_m <= 0:
        return tile_index
    out = []
    for tile in tile_index:
        center, half_diag = tile_center_and_radius(tile, ref_lat, ref_lon)
        if np.linalg.norm(center - prior_xy) <= search_radius_m + half_diag:
            out.append(tile)
    return out


def load_rtk_csv(path, ref_lat, ref_lon):
    if not path:
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 8:
                continue
            try:
                ts = float(row[2]) * 1e-9
                lat = float(row[6])
                lon = float(row[7])
            except ValueError:
                continue
            rows.append((ts, lat, lon, gps_to_metric_xy(lat, lon, ref_lat, ref_lon)))
    rows.sort(key=lambda x: x[0])
    return rows


def load_image_timestamps(image_dir):
    metadata_path = Path(image_dir) / "metadata.csv"
    if not metadata_path.exists():
        return {}

    timestamps = {}
    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("Filename") or row.get("filename")
            timestamp = row.get("timestamp") or row.get("Timestamp")
            if not filename or not timestamp:
                continue
            try:
                timestamps[Path(filename).stem] = float(timestamp)
            except ValueError:
                continue
    return timestamps


def nearest_rtk(rtk_rows, timestamp, max_dt):
    if not rtk_rows:
        return None
    times = [r[0] for r in rtk_rows]
    idx = bisect.bisect_left(times, timestamp)
    cand = []
    if idx < len(rtk_rows):
        cand.append(rtk_rows[idx])
    if idx > 0:
        cand.append(rtk_rows[idx - 1])
    if not cand:
        return None
    best = min(cand, key=lambda r: abs(r[0] - timestamp))
    return best if abs(best[0] - timestamp) <= max_dt else None


def make_rotation_list(yaw_prior_deg, yaw_window_deg, yaw_step_deg, full_step_deg):
    if yaw_prior_deg is not None:
        values = []
        n = int(round(yaw_window_deg / yaw_step_deg))
        for i in range(-n, n + 1):
            values.append((yaw_prior_deg + i * yaw_step_deg) % 360.0)
        return sorted(set(round(v, 6) for v in values))
    return [float(v) for v in np.arange(0.0, 360.0, full_step_deg)]


def parse_rotation_ranges(spec):
    rotations = {}
    if not spec:
        return rotations
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        frame_range, angle = item.split(":", 1)
        angle = float(angle)
        if "-" in frame_range:
            start, end = (int(v) for v in frame_range.split("-", 1))
        else:
            start = end = int(frame_range)
        for idx in range(start, end + 1):
            rotations[f"{idx:06d}"] = angle
    return rotations


def rotate_image_bound(img, clockwise_deg):
    rot_norm = clockwise_deg % 360.0
    if abs(rot_norm) < 1e-9 or abs(rot_norm - 360.0) < 1e-9:
        return img, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    h, w = img.shape[:2]
    if abs(rot_norm - 90.0) < 1e-9:
        mat = np.array([[0.0, -1.0, h - 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), mat
    if abs(rot_norm - 180.0) < 1e-9:
        mat = np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0]], dtype=np.float64)
        return cv2.rotate(img, cv2.ROTATE_180), mat
    if abs(rot_norm - 270.0) < 1e-9:
        mat = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w - 1.0]], dtype=np.float64)
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), mat

    center = (w / 2.0, h / 2.0)
    # OpenCV positive angle is counter-clockwise; our search angle is clockwise.
    mat = cv2.getRotationMatrix2D(center, -rot_norm, 1.0)
    cos_a = abs(mat[0, 0])
    sin_a = abs(mat[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    mat[0, 2] += new_w / 2.0 - center[0]
    mat[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(
        img,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return rotated, mat


def transform_points_affine(points, mat):
    pts = np.asarray(points, dtype=np.float64)
    homo = np.c_[pts, np.ones(len(pts))]
    return homo @ mat.T


def homography_reprojection_error(H, pts0, pts1, inlier_mask):
    if H is None or not np.any(inlier_mask):
        return float("inf")
    src = pts0[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    pred = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    dst = pts1[inlier_mask]
    return float(np.mean(np.linalg.norm(pred - dst, axis=1)))


def draw_match_viz(image_path, map_db, result, out_path, show_rotated_query=False):
    if not result or "tile_name" not in result:
        return False
    drone = cv2.imread(image_path, cv2.IMREAD_COLOR)
    sat = cv2.imread(str(Path(map_db) / result["tile_name"]), cv2.IMREAD_COLOR)
    if drone is None or sat is None:
        return False
    sat_size = result.get("sat_image_size")
    if sat_size and len(sat_size) == 2:
        sat_w, sat_h = int(sat_size[0]), int(sat_size[1])
        if sat.shape[1] != sat_w or sat.shape[0] != sat_h:
            sat = cv2.resize(sat, (sat_w, sat_h), interpolation=cv2.INTER_AREA)
    point_mat = None
    if show_rotated_query:
        drone, point_mat = rotate_image_bound(
            drone, float(result.get("rotation_used", 0.0))
        )

    max_h = 720
    dh, dw = drone.shape[:2]
    sh, sw = sat.shape[:2]
    d_scale = min(1.0, max_h / float(dh))
    s_scale = min(3.0, max_h / float(sh))
    drone_v = cv2.resize(
        drone, (int(dw * d_scale), int(dh * d_scale)), interpolation=cv2.INTER_AREA
    )
    sat_v = cv2.resize(
        sat, (int(sw * s_scale), int(sh * s_scale)), interpolation=cv2.INTER_NEAREST
    )

    h = max(drone_v.shape[0], sat_v.shape[0])
    gap = 24
    canvas = np.zeros((h, drone_v.shape[1] + gap + sat_v.shape[1], 3), dtype=np.uint8)
    canvas[: drone_v.shape[0], : drone_v.shape[1]] = drone_v
    sat_x0 = drone_v.shape[1] + gap
    canvas[: sat_v.shape[0], sat_x0 : sat_x0 + sat_v.shape[1]] = sat_v

    rng = np.random.default_rng(7)
    for i, kp in enumerate(result.get("keypoints", [])):
        dx_raw, dy_raw = kp["drone_x"], kp["drone_y"]
        if point_mat is not None:
            dx_raw, dy_raw = transform_points_affine([[dx_raw, dy_raw]], point_mat)[0]
        dx = int(round(dx_raw * d_scale))
        dy = int(round(dy_raw * d_scale))
        sx_raw = kp.get("sat_x")
        sy_raw = kp.get("sat_y")
        if sx_raw is None or sy_raw is None:
            continue
        sx = sat_x0 + int(round(sx_raw * s_scale))
        sy = int(round(sy_raw * s_scale))
        color = tuple(int(c) for c in rng.integers(64, 255, size=3))
        cv2.circle(canvas, (dx, dy), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (dx, dy), (sx, sy), color, 1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(i),
            (dx + 6, dy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(i),
            (sx + 6, sy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    view_name = "rotated_query" if show_rotated_query else "original_query"
    label = (
        f"view={view_name}  rot={result.get('rotation_used')}  inliers={result.get('inliers')}  "
        f"pnp_err={result.get('position_error_m')}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), canvas)


def match_one_image(
    image_path,
    sp,
    matcher,
    tile_index,
    sat_features,
    map_db,
    rotations_to_try,
    prior_xy,
    ref_lat,
    ref_lon,
    tile_search_radius_m,
    max_center_error_m,
    min_inliers,
    max_reproj_error,
    sat_resize_size,
    sat_feature_cache,
    camera_calibration,
    pnp_ransac_threshold_px,
    pnp_min_bbox_area_ratio,
    pnp_max_optical_tilt_deg,
    pnp_tilt_regularization_px_per_deg,
):
    """Match a single drone image. Returns result dict or None."""
    drone_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if drone_img is None:
        return None
    # Keep original resolution — pixel coords must match AIRSLAM's feature extraction
    # AIRSLAM processes at original resolution (typically 1280px), do NOT resize
    orig_h, orig_w = drone_img.shape[:2]

    best_result = None
    best_score = -1e18
    candidate_tiles = filter_tiles_by_prior(
        tile_index, prior_xy, ref_lat, ref_lon, tile_search_radius_m
    )

    for rot in rotations_to_try:
        img_r, rot_mat = rotate_image_bound(drone_img, rot)
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)

        kpts_d = sp.detect_and_describe_keypoints(img_r)
        rh, rw = img_r.shape[:2]

        for tile in candidate_tiles:
            name = tile["name"]
            tile_kpts, th, tw = get_tile_keypoints(
                name, sp, sat_features, sat_feature_cache, map_db, sat_resize_size
            )
            if tile_kpts is None:
                continue

            matches, _ = matcher.match_keypoints(kpts_d, tile_kpts)
            valid = matches > -1
            mk0 = kpts_d.keypoints[valid]
            mk1 = tile_kpts.keypoints[matches[valid]]
            if len(mk0) < 4:
                continue

            try:
                H, mask = cv2.findHomography(
                    mk0, mk1, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.995
                )
            except Exception:
                continue
            if H is None or mask is None:
                continue

            n = int(np.sum(mask))
            if n < min_inliers:
                continue
            inlier_mask = mask.ravel().astype(bool)
            reproj_error = homography_reprojection_error(H, mk0, mk1, inlier_mask)
            if reproj_error > max_reproj_error:
                continue

            tl_lat, tl_lon = tile["top_left_lat"], tile["top_left_lon"]
            br_lat, br_lon = tile["bottom_right_lat"], tile["bottom_right_lon"]

            original_points = transform_points_affine(mk0, inv_rot_mat)
            original_points[:, 0] *= (camera_calibration.image_width - 1.0) / max(
                orig_w - 1.0, 1.0
            )
            original_points[:, 1] *= (camera_calibration.image_height - 1.0) / max(
                orig_h - 1.0, 1.0
            )
            ground_points = satellite_pixels_to_metric(
                mk1,
                (th, tw),
                tl_lat,
                tl_lon,
                br_lat,
                br_lon,
                ref_lat,
                ref_lon,
            )
            planar_pose = estimate_planar_pnp(
                original_points,
                ground_points,
                camera_calibration,
                min_inliers=min_inliers,
                ransac_threshold_px=pnp_ransac_threshold_px,
                min_bbox_area_ratio=pnp_min_bbox_area_ratio,
                max_optical_tilt_deg=pnp_max_optical_tilt_deg,
                tilt_regularization_px_per_deg=(pnp_tilt_regularization_px_per_deg),
            )
            if planar_pose is None:
                continue
            if planar_pose.reprojection_mean_px > max_reproj_error:
                continue

            original_center = np.array([[orig_w / 2.0, orig_h / 2.0]], dtype=np.float64)
            rotated_center = transform_points_affine(original_center, rot_mat)
            cp = rotated_center.reshape(1, 1, 2).astype(np.float32)
            spx, spy = cv2.perspectiveTransform(cp, H)[0][0]
            clat, clon = pixel_to_gps(spx, spy, tl_lat, tl_lon, br_lat, br_lon, th, tw)
            look_at_error_m = None
            position_error_m = None
            if prior_xy is not None:
                center_xy = gps_to_metric_xy(clat, clon, ref_lat, ref_lon)
                look_at_error_m = float(np.linalg.norm(center_xy - prior_xy))
                position_error_m = float(
                    np.linalg.norm(planar_pose.camera_position_enu[:2] - prior_xy)
                )
                if max_center_error_m > 0 and position_error_m > max_center_error_m:
                    continue

            pnp_inlier_mask = planar_pose.inlier_mask

            kps = []
            for i in range(len(mk0)):
                if not pnp_inlier_mask[i]:
                    continue
                px, py = original_points[i]
                sx, sy = mk1[i]
                klat, klon = pixel_to_gps(sx, sy, tl_lat, tl_lon, br_lat, br_lon, th, tw)
                kps.append(
                    {
                        "drone_x": float(px),
                        "drone_y": float(py),
                        "sat_lat": klat,
                        "sat_lon": klon,
                        "sat_x": float(sx),
                        "sat_y": float(sy),
                        "is_pnp_inlier": True,
                    }
                )

            spread = float(np.linalg.norm(np.ptp(mk1[pnp_inlier_mask], axis=0)))
            position_penalty = (
                0.0 if position_error_m is None else 0.05 * position_error_m
            )
            score = (
                10.0 * planar_pose.inlier_count
                + 2.0 * planar_pose.inlier_ratio
                + 0.02 * spread
                - 2.0 * planar_pose.reprojection_mean_px
                - position_penalty
            )
            if score <= best_score:
                continue

            pose_dict = planar_pose.to_dict(ref_lat, ref_lon)
            camera_lat, camera_lon = metric_xy_to_gps(
                planar_pose.camera_position_enu[0],
                planar_pose.camera_position_enu[1],
                ref_lat,
                ref_lon,
            )
            best_score = score
            best_result = {
                "lat": camera_lat,
                "lon": camera_lon,
                "inliers": planar_pose.inlier_count,
                "keypoints": kps,
            }
            best_result.update(
                {
                    "rotation_used": float(rot),
                    "tile_name": name,
                    "sat_image_size": [int(tw), int(th)],
                    "position_method": "planar_pnp",
                    "planar_pose": pose_dict,
                    "look_at_lat": clat,
                    "look_at_lon": clon,
                    "homography_inliers": n,
                    "homography_reproj_error": reproj_error,
                    "reproj_error": planar_pose.reprojection_mean_px,
                    "position_error_m": position_error_m,
                    "look_at_error_m": look_at_error_m,
                    "score": score,
                }
            )

    return best_result


def main():
    parser = argparse.ArgumentParser(description="Offline batch satellite matching")
    parser.add_argument("--image-dir", required=True, help="Directory of drone images")
    parser.add_argument("--map-db", required=True)
    parser.add_argument("--tile-index", required=True)
    parser.add_argument(
        "--sat-features",
        default="",
        help="Precomputed .npz file; ignored when --sat-resize-size > 0",
    )
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--rtk-path", default="", help="RTK CSV for position prior")
    parser.add_argument("--rtk-max-dt", type=float, default=0.5)
    parser.add_argument(
        "--tile-search-radius",
        type=float,
        default=250.0,
        help="Search only tiles near RTK prior; <=0 disables",
    )
    parser.add_argument(
        "--max-center-error",
        type=float,
        default=120.0,
        help="Reject matches whose projected center is too far from RTK",
    )
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-reproj-error", type=float, default=4.0)
    parser.add_argument(
        "--camera-calib",
        default="/media/sy/30C2CCDEC2CCAA04/datasets/a_low/winter/nav_left.yaml",
        help="OpenCV camera calibration YAML for the original unrotated image",
    )
    parser.add_argument("--reference-lat", type=float, default=31.8325939032)
    parser.add_argument("--reference-lon", type=float, default=118.71416416)
    parser.add_argument("--pnp-ransac-threshold", type=float, default=6.0)
    parser.add_argument("--pnp-min-bbox-ratio", type=float, default=0.03)
    parser.add_argument("--pnp-max-tilt", type=float, default=30.0)
    parser.add_argument("--pnp-tilt-regularization", type=float, default=0.05)
    parser.add_argument(
        "--sat-resize-size",
        type=int,
        default=800,
        help="Resize each satellite tile long side before feature extraction; 800 matches scripts/main.py",
    )
    parser.add_argument("--keypoint-threshold", type=float, default=0.01)
    parser.add_argument("--max-keypoints", type=int, default=-1)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument(
        "--rotation-step",
        type=float,
        default=30.0,
        help="Full 0..360 clockwise search step when no yaw prior is available",
    )
    parser.add_argument(
        "--yaw-prior",
        type=float,
        default=None,
        help="Optional fixed clockwise image-to-north rotation prior",
    )
    parser.add_argument("--yaw-window", type=float, default=45.0)
    parser.add_argument("--yaw-step", type=float, default=15.0)
    parser.add_argument(
        "--rotation-ranges",
        default="",
        help="Per-image fixed clockwise rotations, e.g. '0-48:90,49-67:180'",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process only the first N images after sorting; 0 means all",
    )
    parser.add_argument(
        "--viz-dir",
        default="",
        help="If set, save side-by-side match visualizations here",
    )
    parser.add_argument(
        "--viz-rotated-query",
        action="store_true",
        help="Draw the rotated query image used by matching instead of the original",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--matcher", default="superglue", choices=["superglue", "lightglue"]
    )
    args = parser.parse_args()

    # Init models ONCE
    sp_cfg = SuperPointConfig(
        device=args.device,
        nms_radius=4,
        keypoint_threshold=args.keypoint_threshold,
        max_keypoints=args.max_keypoints,
    )
    sp = SuperPointAlgorithm(sp_cfg)

    if args.matcher == "lightglue":
        mc = LightGlueConfig(
            device=args.device, features="superpoint", n_layers=9, filter_threshold=0.5
        )
        matcher = LightGlueMatcher(mc)
    else:
        mc = SuperGlueConfig(
            device=args.device,
            weights="outdoor",
            sinkhorn_iterations=20,
            match_threshold=args.match_threshold,
        )
        matcher = SuperGlueMatcher(mc)

    tile_index = load_tile_index(args.tile_index)
    sat_features = (
        {} if args.sat_resize_size > 0 else load_sat_features(args.sat_features)
    )
    sat_feature_cache = {}
    ref_lat = args.reference_lat
    ref_lon = args.reference_lon
    camera_calibration = CameraCalibration.from_opencv_yaml(args.camera_calib)
    rtk_rows = load_rtk_csv(args.rtk_path, ref_lat, ref_lon)
    rotations_to_try = make_rotation_list(
        args.yaw_prior, args.yaw_window, args.yaw_step, args.rotation_step
    )
    rotation_by_stem = parse_rotation_ranges(args.rotation_ranges)
    print(
        f"Using {len(rotations_to_try)} rotations: {rotations_to_try}", file=sys.stderr
    )
    if rotation_by_stem:
        print(
            f"Loaded fixed rotations for {len(rotation_by_stem)} image indices",
            file=sys.stderr,
        )
    if rtk_rows:
        print(f"Loaded {len(rtk_rows)} RTK priors from {args.rtk_path}", file=sys.stderr)

    # Process all images
    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        print(f"No images found in {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(images)} images...", file=sys.stderr)
    image_timestamps = load_image_timestamps(image_dir)
    if image_timestamps:
        print(
            f"Loaded {len(image_timestamps)} image timestamps from metadata.csv",
            file=sys.stderr,
        )
    results = {}
    t0 = time.time()
    n_success = 0

    for i, img_path in enumerate(images):
        # Extract timestamp from filename: <timestamp>.jpg or airslam_sat_match_<fid>.jpg
        stem = img_path.stem
        image_rotations = (
            [rotation_by_stem[stem]] if stem in rotation_by_stem else rotations_to_try
        )
        if stem in image_timestamps:
            ts_float = image_timestamps[stem]
            ts_key = f"{ts_float:.6f}"
        else:
            # Try parsing as numeric timestamp first
            try:
                ts_float = float(stem)
                ts_key = str(ts_float)
            except ValueError:
                ts_float = None
                ts_key = stem  # Use full stem as fallback key

        prior_xy = None
        if ts_float is not None and rtk_rows:
            rtk = nearest_rtk(rtk_rows, ts_float, args.rtk_max_dt)
            if rtk:
                prior_xy = rtk[3]

        result = match_one_image(
            str(img_path),
            sp,
            matcher,
            tile_index,
            sat_features,
            args.map_db,
            image_rotations,
            prior_xy,
            ref_lat,
            ref_lon,
            args.tile_search_radius,
            args.max_center_error,
            args.min_inliers,
            args.max_reproj_error,
            args.sat_resize_size,
            sat_feature_cache,
            camera_calibration,
            args.pnp_ransac_threshold,
            args.pnp_min_bbox_ratio,
            args.pnp_max_tilt,
            args.pnp_tilt_regularization,
        )
        if result:
            results[ts_key] = result
            n_success += 1
            if args.viz_dir:
                viz_path = Path(args.viz_dir) / f"{img_path.stem}_match.jpg"
                draw_match_viz(
                    str(img_path), args.map_db, result, viz_path, args.viz_rotated_query
                )
            print(
                f"  [{i+1}/{len(images)}] {img_path.name}: "
                f"{result['inliers']} inliers rot={result.get('rotation_used')} "
                f"pnp_err={result.get('position_error_m')}",
                file=sys.stderr,
            )
        else:
            print(f"  [{i+1}/{len(images)}] {img_path.name}: FAILED", file=sys.stderr)

    elapsed = time.time() - t0
    print(
        f"\nDone: {n_success}/{len(images)} matched in {elapsed:.1f}s "
        f"({elapsed/len(images):.1f}s avg)",
        file=sys.stderr,
    )

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
