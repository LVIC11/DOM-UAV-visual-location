#!/usr/bin/env python3
"""Rotate original queries before SP/SG, refine angles, and export raw pixels."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import SuperPointConfig, SuperGlueConfig
from svl.localization.base import PipelineConfig
from svl.localization.geometry import CameraCalibration
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.rotation_search import export_match, match_rotation, rotate_query


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_fixed_rotations(spec):
    mapping = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        interval, value = item.strip().split(":")
        angle = float(value)
        if not np.isfinite(angle):
            raise ValueError("Rotation must be finite")
        limits = interval.split("-")
        start, end = int(limits[0]), int(limits[-1])
        if len(limits) > 2 or not 0 <= start <= end:
            raise ValueError("Invalid frame rotation interval")
        for number in range(start, end + 1):
            key = f"{number:06d}"
            if key in mapping:
                raise ValueError("Overlapping frame rotation intervals")
            mapping[key] = angle % 360.0
    return mapping


def draw_matches(original, candidate, path, rotated):
    if rotated:
        left, points = candidate.image, candidate.query_points
    else:
        left, points = original, candidate.raw_points
    right = candidate.tile.image
    canvas = np.full(
        (max(left.shape[0], right.shape[0]), left.shape[1] + right.shape[1] + 12, 3),
        245,
        dtype=np.uint8,
    )
    canvas[: left.shape[0], : left.shape[1]] = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    offset = left.shape[1] + 12
    canvas[: right.shape[0], offset:] = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    mask = (
        candidate.pose.inlier_mask
        if candidate.pose is not None
        else candidate.homography_mask
    )
    indices = np.flatnonzero(mask)
    if len(indices) > 120:
        indices = indices[np.linspace(0, len(indices) - 1, 120, dtype=int)]
    for i in indices:
        a = tuple(np.round(points[i]).astype(int))
        b = tuple(np.round(candidate.satellite_points[i] + [offset, 0]).astype(int))
        cv2.line(canvas, a, b, (50, 210, 70), 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, (20, 230, 240), -1)
        cv2.circle(canvas, b, 3, (20, 230, 240), -1)
    status = (
        "PNP VERIFIED"
        if candidate.pose is not None
        else (
            "H IMAGE CENTER - NOT PNP"
            if candidate.center_prediction is not None
            else "TENTATIVE ONLY - NO CONSTRAINT"
        )
    )
    label = f"CW {candidate.angle:g} deg | {candidate.tile.name} | {status} | {len(np.flatnonzero(mask))} inliers"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 35), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError("Cannot write match visualization")


def run(args):
    if args.output.exists():
        raise FileExistsError(str(args.output))
    with (args.image_folder / "metadata.csv").open(newline="") as stream:
        rows = [
            {str(k).lower(): v for k, v in row.items()} for row in csv.DictReader(stream)
        ]
    rows.sort(key=lambda row: float(row["timestamp"]))
    if args.frames:
        names = set(args.frames.split(","))
        rows = [r for r in rows if Path(r["filename"]).stem in names]
    if args.max_images:
        rows = rows[: args.max_images]
    if not rows:
        raise ValueError("No selected queries")
    if len({r["filename"] for r in rows}) != len(rows) or len(
        {float(r["timestamp"]) for r in rows}
    ) != len(rows):
        raise ValueError("Duplicate image identity or timestamp")
    fixed = parse_fixed_rotations(args.rotation_ranges)
    if args.position_method == "homography_center" and not fixed:
        raise ValueError(
            "Homography-center comparison requires fixed physical rotations"
        )
    if args.homography_min_inliers < 4:
        raise ValueError("Homography needs at least four inliers")
    if args.position_method == "homography_center" and args.pnp_homography_prefilter:
        raise ValueError("PnP prefilter is not applicable when PnP is disabled")
    if args.rotation_ranges and any(Path(r["filename"]).stem not in fixed for r in rows):
        raise ValueError("Every selected frame needs an explicit fixed rotation")
    calibration = CameraCalibration.from_opencv_yaml(str(args.camera_calib))
    config = PipelineConfig(camera_calibration=calibration)
    config.pnp_homography_prefilter = args.pnp_homography_prefilter
    config.position_method = args.position_method
    config.homography_min_inliers = args.homography_min_inliers
    scales = [int(x) for x in args.query_sizes.split(",")]
    angles = [float(x) % 360 for x in args.angles.split(",")]
    torch.manual_seed(0)
    np.random.seed(0)
    cv2.setRNGSeed(0)
    detector = SuperPointAlgorithm(
        SuperPointConfig(
            device=args.device,
            nms_radius=4,
            keypoint_threshold=args.keypoint_threshold,
            max_keypoints=-1,
        )
    )
    matcher = SuperGlueMatcher(
        SuperGlueConfig(
            device=args.device,
            weights="outdoor",
            sinkhorn_iterations=20,
            match_threshold=0.5,
        )
    )
    reader = SatelliteMapReader(
        args.map_db, logging.getLogger("tiles"), resize_size=(800,)
    )
    reader.initialize_db()
    reader.setup_db()
    reader.resize_db_images()
    reader.describe_db_images(detector)
    args.output.mkdir(parents=True)
    for name in (
        "images_rotated",
        "visualizations_rotated",
        "visualizations_original",
        "source_archive",
    ):
        (args.output / name).mkdir()
    paths = [
        args.image_folder / "metadata.csv",
        args.camera_calib,
        Path(__file__),
        Path(__file__).parents[1] / "src/svl/localization/rotation_search.py",
        Path(__file__).parents[1] / "src/svl/localization/geometry.py",
    ]
    paths += sorted(args.map_db.glob("*.png")) + sorted(args.map_db.glob("*.csv"))
    paths += [
        Path(__file__).parents[1] / "src/superglue_lib/models/weights" / name
        for name in ("superpoint_v1.pth", "superglue_outdoor.pth")
    ]
    paths += [args.image_folder / row["filename"] for row in rows]
    hashes = {str(p.resolve()): sha(p) for p in paths}
    for path in paths:
        if path.suffix == ".py":
            shutil.copy2(path, args.output / "source_archive" / path.name)
    result = dict(
        status="running",
        query_folder=str(args.image_folder.resolve()),
        total_images=len(rows),
        frames=[],
        input_sha256=hashes,
        ground_truth_used_for_matching=False,
        rtk_xy_used=False,
        camera_calibration=str(args.camera_calib),
        angles_cw=angles,
        query_sizes=scales,
        fine_step_deg=args.fine_step,
        rotation_policy="user_fixed_no_angle_search"
        if fixed
        else "automatic_angle_search",
        user_rotation_ranges=args.rotation_ranges,
        all_tiles_searched=len(reader),
        superpoint_keypoint_threshold=args.keypoint_threshold,
        point_coordinate_frame="unrotated_original_distorted",
        position_method=args.position_method,
        candidate_ranking="raw_superglue_match_count"
        if args.position_method == "homography_center"
        else "pnp_inliers_ratio_reprojection",
        geometric_thresholds=config.__dict__.copy(),
    )
    result["geometric_thresholds"].pop("camera_calibration")
    matched = []
    try:
        for index, row in enumerate(rows):
            start = time.monotonic()
            path = args.image_folder / row["filename"]
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (
                calibration.image_height,
                calibration.image_width,
            ):
                raise ValueError(
                    "Raw query differs from original camera calibration: " + str(path)
                )
            best = None
            attempts = []
            seen = set()

            def attempt(angle, size):
                nonlocal best
                key = (round(angle % 360, 6), size)
                if key in seen:
                    return
                seen.add(key)
                candidate, audit = match_rotation(
                    image, key[0], size, detector, matcher, reader, calibration, config
                )
                attempts.extend(audit)
                if candidate is not None and (
                    best is None or candidate.score > best.score
                ):
                    best = candidate
                print(
                    "ATTEMPT",
                    path.stem,
                    "CW",
                    key[0],
                    "size",
                    size,
                    "raw_max",
                    max(a["raw_matches"] for a in audit),
                    "H_max",
                    max(a["homography_inliers"] for a in audit),
                    "PnP",
                    candidate.pose.inlier_count if candidate and candidate.pose else 0,
                    "predicted",
                    bool(candidate and candidate.has_prediction),
                    flush=True,
                )

            if fixed:
                for size in scales:
                    attempt(fixed[path.stem], size)
            else:
                for angle in angles:
                    attempt(angle, scales[0])
                if best is not None and args.fine_step > 0 and best.score[1] >= 8:
                    center = best.angle
                    fine = [
                        center + offset
                        for offset in np.arange(
                            -args.fine_radius, args.fine_radius + 0.1, args.fine_step
                        )
                    ]
                    for angle in fine:
                        attempt(angle, scales[0])
                if best is not None:
                    for size in scales[1:]:
                        attempt(best.angle, size)
                if (
                    best is not None
                    and best.pose is None
                    and best.score[1] >= 20
                    and args.refine_step > 0
                ):
                    center, size = best.angle, best.long_side
                    for offset in np.arange(-5.0, 5.1, args.refine_step):
                        attempt(center + offset, size)
            frame = dict(
                frame=path.stem,
                filename=path.name,
                timestamp=float(row["timestamp"]),
                matched=bool(best is not None and best.has_prediction),
                position_method=args.position_method,
                attempts=attempts,
                stages=dict(Counter(a["stage"] for a in attempts)),
                elapsed_s=time.monotonic() - start,
            )
            selected_angle = (
                fixed[path.stem] if fixed else (best.angle if best else None)
            )
            if selected_angle is not None:
                rotated, affine = rotate_query(image, selected_angle)
                target = args.output / "images_rotated" / (path.stem + ".png")
                if not cv2.imwrite(str(target), rotated):
                    raise RuntimeError("Cannot write rotated image")
                frame.update(
                    rotation_cw=selected_angle,
                    rotation_source="user_provided" if fixed else "automatic_candidate",
                    rotation_verified_by_pnp=bool(
                        best is not None and best.pose is not None
                    ),
                    rotated_image=str(target.resolve()),
                    original_to_rotated_pixels=affine.tolist(),
                    rotated_to_original_pixels=np.linalg.inv(affine).tolist(),
                    best_homography_inliers=int(best.homography_mask.sum())
                    if best
                    else 0,
                    best_tile=best.tile.name if best else None,
                )
            if best is not None:
                draw_matches(
                    image,
                    best,
                    args.output / "visualizations_rotated" / (path.stem + "_viz.jpg"),
                    True,
                )
                draw_matches(
                    image,
                    best,
                    args.output / "visualizations_original" / (path.stem + "_viz.jpg"),
                    False,
                )
                if best.has_prediction:
                    record = export_match(
                        best, path.stem, float(row["timestamp"]), calibration, config
                    )
                    matched.append(record)
                    frame["prediction_inliers"] = record["num_inliers"]
                    if best.pose is not None:
                        frame.update(
                            pnp_inliers=best.pose.inlier_count,
                            reprojection_mean_px=best.pose.reprojection_mean_px,
                        )
                    else:
                        frame["homography_center"] = best.center_prediction
            result["frames"].append(frame)
            write_json(args.output / "rotation_manifest.json", result)
            write_json(args.output / "matched_keypoints.json", matched)
            print(
                "FRAME",
                index + 1,
                "/",
                len(rows),
                path.stem,
                "CW",
                frame.get("rotation_cw"),
                "PnP",
                frame.get("pnp_inliers", 0),
                "prediction_inliers",
                frame.get("prediction_inliers", 0),
                "seconds",
                round(frame["elapsed_s"], 1),
                flush=True,
            )
        if any(sha(p) != h for p, h in hashes.items()):
            raise ValueError("Input changed during matching")
        result.update(
            status="complete",
            matched_frames=len(matched),
            total_inlier_points=sum(len(r["keypoints"]) for r in matched),
        )
        with (args.output / "images_rotated/metadata.csv").open(
            "w", newline=""
        ) as stream:
            columns = list(rows[0]) + [
                "rotation_cw",
                "rotation_source",
                "rotation_verified_by_pnp",
            ]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            originals = {Path(r["filename"]).stem: r for r in rows}
            for frame in result["frames"]:
                if "rotated_image" in frame:
                    writer.writerow(
                        dict(
                            originals[frame["frame"]],
                            filename=Path(frame["rotated_image"]).name,
                            rotation_cw=frame["rotation_cw"],
                            rotation_source=frame["rotation_source"],
                            rotation_verified_by_pnp=frame["rotation_verified_by_pnp"],
                        )
                    )
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "rotation_manifest.json", result)
        write_json(args.output / "matched_keypoints.json", matched)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "total_images",
                    "matched_frames",
                    "total_inlier_points",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image-folder", "map-db", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--camera-calib",
        type=Path,
        default=Path("/media/sy/30C2CCDEC2CCAA04/datasets/a_low/winter/nav_left.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--position-method",
        choices=["planar_pnp", "homography_center"],
        default="planar_pnp",
    )
    parser.add_argument(
        "--homography-min-inliers",
        type=int,
        default=4,
        help="Legacy homography image-center minimum; NOT a PnP geometry certificate",
    )
    parser.add_argument("--keypoint-threshold", type=float, default=0.01)
    parser.add_argument(
        "--pnp-homography-prefilter",
        action="store_true",
        help="Feed homography consensus to PnP; ratio denominator is explicitly recorded",
    )
    parser.add_argument("--query-sizes", default="480,800,1280")
    parser.add_argument("--angles", default="0,90,180,270,30,60,120,150,210,240,300,330")
    parser.add_argument(
        "--rotation-ranges",
        default="",
        help="Fixed physical clockwise rotations by file number, e.g. '4-5:270,6-15:90'; disables angle search",
    )
    parser.add_argument("--fine-radius", type=float, default=15.0)
    parser.add_argument("--fine-step", type=float, default=5.0)
    parser.add_argument("--refine-step", type=float, default=1.0)
    parser.add_argument(
        "--frames",
        default="",
        help="Comma-separated filename stems for a diagnostic subset",
    )
    parser.add_argument("--max-images", type=int)
    logging.basicConfig(level=logging.ERROR)
    run(parser.parse_args())
