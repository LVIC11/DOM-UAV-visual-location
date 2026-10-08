#!/usr/bin/env python3
"""Match one drone image and estimate its camera pose on the ground plane.

Called by AIRSLAM's ``SatelliteMatcher`` via ``popen``.  The last stdout line
is a JSON object; diagnostics are written to stderr.
"""

import argparse
import json
import sys
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


def load_tile_index(json_path: str) -> list:
    with open(json_path, "r") as f:
        return json.load(f)


def load_sat_features(npz_path: str) -> dict:
    """{tile_name: {keypoints, descriptors, scores, image_size}}"""
    data = np.load(npz_path, allow_pickle=True)
    tile_names = data["__tile_names__"]
    features = {}
    for name in tile_names:
        prefix = str(name).replace(".", "_")
        features[str(name)] = {
            "keypoints": data[f"{prefix}_keypoints"],
            "descriptors": data[f"{prefix}_descriptors"],
            "scores": data[f"{prefix}_scores"],
            "image_size": data[f"{prefix}_image_size"],
        }
    print(f"Loaded {len(features)} satellite feature sets", file=sys.stderr)
    return features


def load_airslam_features(bin_path: str):
    """Binary format: int32 N, int32 W, int32 H, then N × (float32 x, float32 y, float32[256] desc)"""
    import struct

    with open(bin_path, "rb") as f:
        data = f.read()
    offset = 0
    N = struct.unpack_from("i", data, offset)[0]
    offset += 4
    img_w = struct.unpack_from("i", data, offset)[0]
    offset += 4
    img_h = struct.unpack_from("i", data, offset)[0]
    offset += 4
    kpts = np.zeros((N, 2), dtype=np.float32)
    descs = np.zeros((N, 256), dtype=np.float32)
    scores = np.ones(N, dtype=np.float32)
    for i in range(N):
        kpts[i, 0] = struct.unpack_from("f", data, offset)[0]
        offset += 4
        kpts[i, 1] = struct.unpack_from("f", data, offset)[0]
        offset += 4
        for j in range(256):
            descs[i, j] = struct.unpack_from("f", data, offset)[0]
            offset += 4
    from svl.keypoint_pipeline.typing import ImageKeyPoints

    drone_kpts = ImageKeyPoints(
        keypoints=kpts, descriptors=descs, scores=scores, image_size=(img_h, img_w)
    )
    print(f"Loaded {N} AIRSLAM features (img={img_w}x{img_h})", file=sys.stderr)
    return drone_kpts, img_w, img_h


def resize_long_side(img: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0:
        return img
    height, width = img.shape[:2]
    scale = float(target_size) / float(max(height, width))
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def rotate_image_bound(img: np.ndarray, clockwise_deg: float):
    """Rotate an image and return the original-to-rotated affine matrix."""

    rotation = float(clockwise_deg) % 360.0
    height, width = img.shape[:2]
    if np.isclose(rotation, 0.0):
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return img, matrix
    if np.isclose(rotation, 90.0):
        matrix = np.array([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), matrix
    if np.isclose(rotation, 180.0):
        matrix = np.array(
            [[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0]],
            dtype=np.float64,
        )
        return cv2.rotate(img, cv2.ROTATE_180), matrix
    if np.isclose(rotation, 270.0):
        matrix = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, width - 1.0]], dtype=np.float64)
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), matrix
    raise ValueError("--rotate must be one of 0, 90, 180, or 270")


def transform_points_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return homogeneous @ matrix.T


def homography_reprojection_error(H, source, destination, mask) -> float:
    inliers = np.asarray(mask).ravel().astype(bool)
    if H is None or not np.any(inliers):
        return float("inf")
    projected = cv2.perspectiveTransform(
        np.asarray(source[inliers], dtype=np.float32).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - destination[inliers], axis=1)))


def pixel_to_gps(px, py, tl_lat, tl_lon, br_lat, br_lon, tile_h, tile_w):
    """Convert pixel coordinate on tile image to GPS."""
    lat = tl_lat + (py / tile_h) * (br_lat - tl_lat)
    lon = tl_lon + (px / tile_w) * (br_lon - tl_lon)
    return lat, lon


def match_single_image(
    image_path: str,
    map_db: str,
    tile_index_path: str,
    device: str = "cuda",
    matcher_name: str = "superglue",
    rotate_cw: int = 0,
    sat_features_path: str = "",
    drone_features_path: str = "",
    camera_calibration: CameraCalibration = None,
    ref_lat: float = 31.8325939032,
    ref_lon: float = 118.71416416,
    min_inliers: int = 20,
    max_reprojection_error_px: float = 4.0,
    satellite_resize_size: int = 0,
    pnp_ransac_threshold_px: float = 6.0,
    pnp_min_bbox_area_ratio: float = 0.03,
    pnp_max_optical_tilt_deg: float = 30.0,
    pnp_tilt_regularization_px_per_deg: float = 0.05,
) -> dict:
    """Match a drone image and return planar-PnP camera pose and inliers."""

    tile_index = load_tile_index(tile_index_path)
    if not tile_index:
        return {"success": False, "error": "Empty tile index"}
    if camera_calibration is None:
        return {"success": False, "error": "Camera calibration is required"}

    sat_features = (
        load_sat_features(sat_features_path)
        if sat_features_path and satellite_resize_size <= 0
        else None
    )

    # Init models (loaded once, reused across tiles)
    sp_config = SuperPointConfig(
        device=device, nms_radius=4, keypoint_threshold=0.01, max_keypoints=-1
    )
    sp = SuperPointAlgorithm(sp_config)

    if matcher_name == "lightglue":
        mc = LightGlueConfig(
            device=device, features="superpoint", n_layers=9, filter_threshold=0.5
        )
        matcher = LightGlueMatcher(mc)
    else:
        mc = SuperGlueConfig(
            device=device,
            weights="outdoor",
            sinkhorn_iterations=20,
            match_threshold=0.5,
        )
        matcher = SuperGlueMatcher(mc)

    drone_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if drone_img is None:
        return {"success": False, "error": f"Cannot read {image_path}"}
    orig_h, orig_w = drone_img.shape[:2]

    if drone_features_path:
        drone_kpts_orig, orig_w, orig_h = load_airslam_features(drone_features_path)
        rotations_to_try = [0]
        if rotate_cw:
            print(
                "Ignoring --rotate for pre-extracted AIRSLAM descriptors; "
                "rotate the image before feature extraction instead",
                file=sys.stderr,
            )
    else:
        rotations_to_try = [rotate_cw] if rotate_cw != 0 else [0, 90, 180, 270]

    best_result = None
    best_score = -float("inf")
    satellite_cache = {}

    for rot in rotations_to_try:
        if drone_features_path:
            kpts_rotated = drone_kpts_orig
            rot_h, rot_w = orig_h, orig_w
            rotation_matrix = np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
            )
        else:
            rotated_image, rotation_matrix = rotate_image_bound(drone_img, rot)
            kpts_rotated = sp.detect_and_describe_keypoints(rotated_image)
            rot_h, rot_w = rotated_image.shape[:2]
        inverse_rotation = cv2.invertAffineTransform(rotation_matrix)

        for tile in tile_index:
            name = tile["name"]
            if sat_features and name in sat_features:
                sf = sat_features[name]
                tile_kpts = ImageKeyPoints(
                    keypoints=sf["keypoints"],
                    descriptors=sf["descriptors"],
                    scores=sf["scores"],
                    image_size=tuple(sf["image_size"]),
                )
                th, tw = sf["image_size"]
            else:
                if name not in satellite_cache:
                    tile_image = cv2.imread(
                        str(Path(map_db) / name), cv2.IMREAD_GRAYSCALE
                    )
                    if tile_image is not None:
                        tile_image = resize_long_side(tile_image, satellite_resize_size)
                        satellite_cache[name] = (
                            sp.detect_and_describe_keypoints(tile_image),
                            *tile_image.shape[:2],
                        )
                if name not in satellite_cache:
                    continue
                tile_kpts, th, tw = satellite_cache[name]

            matches, _ = matcher.match_keypoints(kpts_rotated, tile_kpts)
            valid = matches > -1
            mkpts0 = kpts_rotated.keypoints[valid]
            mkpts1 = tile_kpts.keypoints[matches[valid]]
            if len(mkpts0) < 4:
                continue

            try:
                H, mask = cv2.findHomography(
                    mkpts0,
                    mkpts1,
                    cv2.RANSAC,
                    5.0,
                    maxIters=2000,
                    confidence=0.995,
                )
            except cv2.error:
                continue
            if H is None or mask is None:
                continue

            n_inliers = int(np.sum(mask))
            if n_inliers < min_inliers:
                continue
            homography_error = homography_reprojection_error(H, mkpts0, mkpts1, mask)
            if homography_error > max_reprojection_error_px:
                continue

            tl_lat, tl_lon = tile["top_left_lat"], tile["top_left_lon"]
            br_lat, br_lon = tile["bottom_right_lat"], tile["bottom_right_lon"]

            camera_pixels = transform_points_affine(mkpts0, inverse_rotation)
            camera_pixels[:, 0] *= (camera_calibration.image_width - 1.0) / max(
                orig_w - 1.0, 1.0
            )
            camera_pixels[:, 1] *= (camera_calibration.image_height - 1.0) / max(
                orig_h - 1.0, 1.0
            )
            ground_points = satellite_pixels_to_metric(
                mkpts1,
                (int(th), int(tw)),
                tl_lat,
                tl_lon,
                br_lat,
                br_lon,
                ref_lat,
                ref_lon,
            )
            pose = estimate_planar_pnp(
                camera_pixels,
                ground_points,
                camera_calibration,
                min_inliers=min_inliers,
                ransac_threshold_px=pnp_ransac_threshold_px,
                min_bbox_area_ratio=pnp_min_bbox_area_ratio,
                max_optical_tilt_deg=pnp_max_optical_tilt_deg,
                tilt_regularization_px_per_deg=(pnp_tilt_regularization_px_per_deg),
            )
            if pose is None or pose.reprojection_mean_px > max_reprojection_error_px:
                continue

            center_original = np.array([[orig_w / 2.0, orig_h / 2.0]], dtype=np.float64)
            center_rotated = transform_points_affine(
                center_original, rotation_matrix
            ).astype(np.float32)
            center_satellite = cv2.perspectiveTransform(
                center_rotated.reshape(1, 1, 2), H
            )[0, 0]
            look_at_lat, look_at_lon = pixel_to_gps(
                center_satellite[0],
                center_satellite[1],
                tl_lat,
                tl_lon,
                br_lat,
                br_lon,
                th,
                tw,
            )
            camera_lat, camera_lon = metric_xy_to_gps(
                pose.camera_position_enu[0],
                pose.camera_position_enu[1],
                ref_lat,
                ref_lon,
            )

            valid_indices = np.flatnonzero(valid)
            keypoints_gps = []
            for i in range(len(mkpts0)):
                if not pose.inlier_mask[i]:
                    continue
                satellite_x, satellite_y = mkpts1[i]
                kp_lat, kp_lon = pixel_to_gps(
                    satellite_x,
                    satellite_y,
                    tl_lat,
                    tl_lon,
                    br_lat,
                    br_lon,
                    th,
                    tw,
                )
                keypoint = {
                    "drone_x": float(camera_pixels[i, 0]),
                    "drone_y": float(camera_pixels[i, 1]),
                    "sat_x": float(satellite_x),
                    "sat_y": float(satellite_y),
                    "sat_lat": float(kp_lat),
                    "sat_lon": float(kp_lon),
                    "is_pnp_inlier": True,
                }
                if drone_features_path:
                    keypoint["keypoint_idx"] = int(valid_indices[i])
                keypoints_gps.append(keypoint)

            score = (
                10.0 * pose.inlier_count
                + 2.0 * pose.inlier_ratio
                - 2.0 * pose.reprojection_mean_px
            )
            if score <= best_score:
                continue
            best_score = score
            best_result = {
                "success": True,
                "lat": float(camera_lat),
                "lon": float(camera_lon),
                "inliers": int(pose.inlier_count),
                "keypoints": keypoints_gps,
                "rotation_used": int(rot),
                "tile_name": name,
                "position_method": "planar_pnp",
                "planar_pose": pose.to_dict(ref_lat, ref_lon),
                "look_at_lat": float(look_at_lat),
                "look_at_lon": float(look_at_lon),
                "homography_inliers": n_inliers,
                "homography_reprojection_error": homography_error,
                "reproj_error": pose.reprojection_mean_px,
                "query_coordinate_frame": (
                    "airslam_feature_pixels"
                    if drone_features_path
                    else "unrotated_original"
                ),
            }

    if best_result is None:
        return {
            "success": False,
            "lat": 0,
            "lon": 0,
            "inliers": 0,
            "error": "No geometrically valid planar-PnP match found",
        }
    return best_result


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Single-image satellite matching with planar PnP"
    )
    p.add_argument("--image", required=True)
    p.add_argument("--map-db", required=True)
    p.add_argument("--tile-index", required=True)
    p.add_argument("--no-gt", action="store_true")
    p.add_argument("--ref-lat", type=float, default=31.8325939032)
    p.add_argument("--ref-lon", type=float, default=118.71416416)
    p.add_argument("--rotate", type=int, default=0)
    p.add_argument("--sat-features", type=str, default="")
    p.add_argument("--drone-features", type=str, default="")
    p.add_argument(
        "--camera-calib",
        default="/media/sy/30C2CCDEC2CCAA04/datasets/a_low/winter/nav_left.yaml",
    )
    p.add_argument("--min-inliers", type=int, default=20)
    p.add_argument("--max-reproj-error", type=float, default=4.0)
    p.add_argument(
        "--sat-resize-size",
        type=int,
        default=0,
        help="Re-extract satellite features at this long-side size; 0 uses the NPZ when supplied",
    )
    p.add_argument("--pnp-ransac-threshold", type=float, default=6.0)
    p.add_argument("--pnp-min-bbox-ratio", type=float, default=0.03)
    p.add_argument("--pnp-max-tilt", type=float, default=30.0)
    p.add_argument("--pnp-tilt-regularization", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--matcher",
        type=str,
        default="superglue",
        choices=["superglue", "lightglue"],
    )
    args = p.parse_args()

    calibration = CameraCalibration.from_opencv_yaml(args.camera_calib)
    result = match_single_image(
        image_path=args.image,
        map_db=args.map_db,
        tile_index_path=args.tile_index,
        device=args.device,
        matcher_name=args.matcher,
        rotate_cw=args.rotate,
        sat_features_path=args.sat_features,
        drone_features_path=args.drone_features,
        camera_calibration=calibration,
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
        min_inliers=args.min_inliers,
        max_reprojection_error_px=args.max_reproj_error,
        satellite_resize_size=args.sat_resize_size,
        pnp_ransac_threshold_px=args.pnp_ransac_threshold,
        pnp_min_bbox_area_ratio=args.pnp_min_bbox_ratio,
        pnp_max_optical_tilt_deg=args.pnp_max_tilt,
        pnp_tilt_regularization_px_per_deg=args.pnp_tilt_regularization,
    )
    print(json.dumps(result))
