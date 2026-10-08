#!/usr/bin/env python3
"""Convert visual-localization results to AIRSLAM offline pre-match JSON."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def _load_yaml(path):
    text = Path(path).read_text()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("%YAML:"))
    return yaml.safe_load(text)


def _camera_matrix(intrinsics):
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _camera_transform(camera):
    transform = np.asarray(camera["T"], dtype=np.float64)
    if int(camera.get("T_type", 0)):
        transform = np.linalg.inv(transform)
    return transform


def load_left_rectification(camera_config):
    """Reproduce AIRSLAM Camera's left-image stereo rectification."""
    config = _load_yaml(camera_config)
    width = int(config["image_width"])
    height = int(config["image_height"])
    distortion_type = int(config["distortion_type"])

    cam0 = config["cam0"]
    cam1 = config["cam1"]
    k0 = _camera_matrix(cam0["intrinsics"])
    k1 = _camera_matrix(cam1["intrinsics"])
    d0 = np.asarray(cam0["distortion_coeffs"], dtype=np.float64)
    d1 = np.asarray(cam1["distortion_coeffs"], dtype=np.float64)

    if distortion_type == 0:
        return width, height, distortion_type, k0, d0, np.eye(3), k0
    if distortion_type != 1:
        raise ValueError(
            "Only AIRSLAM pinhole/radial-tangential rectification is supported"
        )

    tbc0 = _camera_transform(cam0)
    tbc1 = _camera_transform(cam1)
    tc1c0 = np.linalg.inv(tbc1) @ tbc0
    r10 = tc1c0[:3, :3]
    t10 = tc1c0[:3, 3]
    size = (width, height)
    r0, _, p0, _, _, _, _ = cv2.stereoRectify(
        k0,
        d0,
        k1,
        d1,
        size,
        r10,
        t10,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1,
        newImageSize=size,
    )
    return width, height, distortion_type, k0, d0, r0, p0


def rectify_points(points, k0, d0, r0, p0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, k0, d0, R=r0, P=p0).reshape(-1, 2)


def convert_record(record, rectification, min_inliers):
    width, height, _, k0, d0, r0, p0 = rectification
    source_points = record.get("keypoints", [])
    if not source_points:
        return None, 0

    raw_points = np.asarray(
        [[point["query_x"], point["query_y"]] for point in source_points],
        dtype=np.float64,
    )
    rectified_points = rectify_points(raw_points, k0, d0, r0, p0)

    keypoints = []
    outside_count = 0
    for source, raw, rectified in zip(source_points, raw_points, rectified_points):
        rectified_x, rectified_y = (float(value) for value in rectified)
        if not (0.0 <= rectified_x < width and 0.0 <= rectified_y < height):
            outside_count += 1
            continue

        point = {
            "drone_x": rectified_x,
            "drone_y": rectified_y,
            "drone_x_raw": float(raw[0]),
            "drone_y_raw": float(raw[1]),
            "sat_lat": float(source["sat_lat"]),
            "sat_lon": float(source["sat_lon"]),
            "is_pnp_inlier": True,
        }
        for source_name, output_name in (
            ("sat_x", "sat_x"),
            ("sat_y", "sat_y"),
            ("confidence", "confidence"),
            ("query_idx", "query_idx"),
            ("sat_idx", "sat_idx"),
        ):
            if source_name in source:
                value = source[source_name]
                point[output_name] = (
                    int(value) if source_name.endswith("_idx") else float(value)
                )
        keypoints.append(point)

    if len(keypoints) < min_inliers:
        return None, outside_count

    planar_pose = record.get("planar_pose") or {}
    output = {
        "lat": float(record["predicted_lat"]),
        "lon": float(record["predicted_lon"]),
        "inliers": len(keypoints),
        "keypoints": keypoints,
        "frame": str(record["frame"]),
        "rotation_used": float(record.get("query_rotation_cw", 0.0)),
        "tile_name": str(record.get("matched_image", "")),
        "position_method": "planar_pnp",
        "pixel_coordinate_frame": "airslam_rectified_left",
        "source_pixel_coordinate_frame": "unrotated_original_distorted",
        "planar_pose": planar_pose,
        "reproj_error": planar_pose.get("pnp_reprojection_mean_px"),
    }
    if record.get("look_at_lat") is not None:
        output["look_at_lat"] = float(record["look_at_lat"])
        output["look_at_lon"] = float(record["look_at_lon"])
    return output, outside_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="matched_keypoints.json")
    parser.add_argument("--output", required=True, help="AIRSLAM pre-match JSON")
    parser.add_argument(
        "--airslam-camera-config",
        required=True,
        help="AIRSLAM stereo camera YAML used during bag playback",
    )
    parser.add_argument("--min-inliers", type=int, default=20)
    args = parser.parse_args()

    with open(args.input) as stream:
        source = json.load(stream)
    if not isinstance(source, list):
        raise ValueError("Input must be the list written by scripts/main.py")

    rectification = load_left_rectification(args.airslam_camera_config)
    output = {}
    outside_count = 0
    rejected_after_rectification = 0
    for record in source:
        converted, outside = convert_record(record, rectification, args.min_inliers)
        outside_count += outside
        if converted is None:
            rejected_after_rectification += 1
            continue
        timestamp = f"{float(record['timestamp']):.6f}"
        if timestamp in output:
            raise ValueError(f"Duplicate timestamp after formatting: {timestamp}")
        output[timestamp] = converted

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        json.dump(output, stream, separators=(",", ":"))

    inlier_counts = [record["inliers"] for record in output.values()]
    summary = {
        "input_records": len(source),
        "output_records": len(output),
        "rejected_after_rectification": rejected_after_rectification,
        "points_outside_rectified_image": outside_count,
        "total_keypoints": sum(inlier_counts),
        "min_keypoints_per_record": min(inlier_counts) if inlier_counts else 0,
        "max_keypoints_per_record": max(inlier_counts) if inlier_counts else 0,
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
