#!/usr/bin/env python3
"""Replace homography-center positions in a pre-match JSON with planar PnP poses."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from svl.localization.geometry import (
    CameraCalibration,
    estimate_planar_pnp,
    gps_to_metric_xy,
    metric_xy_to_gps,
)


def load_metadata(path):
    if not path:
        return {}
    rows = {}
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = row.get("timestamp") or row.get("Timestamp")
            latitude = row.get("latitude") or row.get("Latitude")
            longitude = row.get("longitude") or row.get("Longitude")
            if not timestamp or not latitude or not longitude:
                continue
            rows[f"{float(timestamp):.6f}"] = (float(latitude), float(longitude))
    return rows


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--camera-calib",
        default="/media/sy/30C2CCDEC2CCAA04/datasets/a_low/winter/nav_left.yaml",
    )
    parser.add_argument("--reference-lat", type=float, default=31.8325939032)
    parser.add_argument("--reference-lon", type=float, default=118.71416416)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--ransac-threshold", type=float, default=6.0)
    parser.add_argument("--min-bbox-ratio", type=float, default=0.03)
    parser.add_argument("--max-tilt", type=float, default=30.0)
    parser.add_argument("--tilt-regularization", type=float, default=0.05)
    parser.add_argument(
        "--metadata",
        default="",
        help="Optional metadata.csv used only to report position error",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep records whose PnP pose is invalid, retaining their old position",
    )
    args = parser.parse_args()

    with open(args.input) as stream:
        source = json.load(stream)
    calibration = CameraCalibration.from_opencv_yaml(args.camera_calib)
    metadata = load_metadata(args.metadata)

    output = {}
    pnp_errors = []
    look_at_errors = []
    failure_reasons = {"too_few_points": 0, "pnp_rejected": 0}

    for timestamp, record in source.items():
        keypoints = record.get("keypoints", [])
        usable = [
            point
            for point in keypoints
            if "sat_lat" in point
            and "sat_lon" in point
            and (
                ("drone_x_raw" in point and "drone_y_raw" in point)
                or ("drone_x" in point and "drone_y" in point)
            )
        ]
        if len(usable) < args.min_inliers:
            failure_reasons["too_few_points"] += 1
            if args.keep_failed:
                output[timestamp] = record
            continue

        image_points = np.array(
            [
                [
                    point.get("drone_x_raw", point["drone_x"]),
                    point.get("drone_y_raw", point["drone_y"]),
                ]
                for point in usable
            ],
            dtype=np.float64,
        )
        ground_points = np.array(
            [
                gps_to_metric_xy(
                    point["sat_lat"],
                    point["sat_lon"],
                    args.reference_lat,
                    args.reference_lon,
                )
                for point in usable
            ],
            dtype=np.float64,
        )
        pose = estimate_planar_pnp(
            image_points,
            ground_points,
            calibration,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            ransac_threshold_px=args.ransac_threshold,
            min_bbox_area_ratio=args.min_bbox_ratio,
            max_optical_tilt_deg=args.max_tilt,
            tilt_regularization_px_per_deg=args.tilt_regularization,
        )
        if pose is None:
            failure_reasons["pnp_rejected"] += 1
            if args.keep_failed:
                output[timestamp] = record
            continue

        refined = dict(record)
        refined["look_at_lat"] = float(record["lat"])
        refined["look_at_lon"] = float(record["lon"])
        camera_lat, camera_lon = metric_xy_to_gps(
            pose.camera_position_enu[0],
            pose.camera_position_enu[1],
            args.reference_lat,
            args.reference_lon,
        )
        refined["lat"] = camera_lat
        refined["lon"] = camera_lon
        refined["inliers"] = pose.inlier_count
        refined["position_method"] = "planar_pnp"
        refined["planar_pose"] = pose.to_dict(args.reference_lat, args.reference_lon)
        refined["reproj_error"] = pose.reprojection_mean_px
        refined["keypoints"] = []
        for point, is_inlier in zip(usable, pose.inlier_mask):
            if not is_inlier:
                continue
            kept = dict(point)
            kept["is_pnp_inlier"] = True
            refined["keypoints"].append(kept)
        output[timestamp] = refined

        metadata_row = metadata.get(f"{float(timestamp):.6f}")
        if metadata_row:
            gt_xy = np.asarray(
                gps_to_metric_xy(
                    metadata_row[0],
                    metadata_row[1],
                    args.reference_lat,
                    args.reference_lon,
                )
            )
            pnp_errors.append(
                float(np.linalg.norm(pose.camera_position_enu[:2] - gt_xy))
            )
            old_xy = np.asarray(
                gps_to_metric_xy(
                    refined["look_at_lat"],
                    refined["look_at_lon"],
                    args.reference_lat,
                    args.reference_lon,
                )
            )
            look_at_errors.append(float(np.linalg.norm(old_xy - gt_xy)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        json.dump(output, stream)

    summary = {
        "input_records": len(source),
        "output_records": len(output),
        "planar_pnp_records": sum(
            record.get("position_method") == "planar_pnp" for record in output.values()
        ),
        "failures": failure_reasons,
        "pnp_horizontal_error_m": summarize(pnp_errors),
        "old_look_at_horizontal_error_m": summarize(look_at_errors),
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
