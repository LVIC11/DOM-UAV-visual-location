#!/usr/bin/env python3
"""Summarize physical-rotation searches without turning tentative matches into fixes."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from svl.localization.rotation_search import transform_points
from svl.tms.geo import haversine_distance
from svl.tms.schemas import GpsCoordinate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unrotated_count(frame):
    return max(
        (a["raw_matches"] for a in frame["attempts"] if a["rotation_cw"] == 0),
        default=None,
    )


def report(root):
    target = root / "summary.json"
    if target.exists():
        raise FileExistsError(str(target))
    source = json.loads((root / "rotation_manifest.json").read_text())
    matches = json.loads((root / "matched_keypoints.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed matching required")
    fixed = source.get("rotation_policy") == "user_fixed_no_angle_search"
    for path, expected in source["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Input changed: " + path)
    with (Path(source["query_folder"]) / "metadata.csv").open(newline="") as stream:
        metadata = {Path(r["filename"]).stem: r for r in csv.DictReader(stream)}
    frames = {r["frame"]: r for r in source["frames"]}
    errors = []
    error_rows = []
    max_inverse_error = 0.0
    point_count = 0
    for match in matches:
        frame = frames[match["frame"]]
        if not frame["matched"] or match["timestamp"] != frame["timestamp"]:
            raise ValueError("Native timestamp/frame identity mismatch")
        points = match["keypoints"]
        raw = np.array([[p["query_x"], p["query_y"]] for p in points])
        processed = np.array(
            [[p["query_rotated_x"], p["query_rotated_y"]] for p in points]
        )
        width, height = (
            points[0]["query_unrotated_width"],
            points[0]["query_unrotated_height"],
        )
        if (
            not np.isfinite(raw).all()
            or np.any(raw < 0)
            or np.any(raw[:, 0] >= width)
            or np.any(raw[:, 1] >= height)
        ):
            raise ValueError("Exported points are not original camera pixels")
        recovered = transform_points(
            processed, np.array(match["matching_to_original_pixels"])
        )
        max_inverse_error = max(max_inverse_error, float(abs(recovered - raw).max()))
        if abs(recovered - raw).max() > 1e-6 or any(
            not p["is_pnp_inlier"] for p in points
        ):
            raise ValueError("Invalid inverse mapping or unverified point")
        point_count += len(points)
        row = metadata[match["frame"]]
        distance = 1000 * haversine_distance(
            GpsCoordinate(lat=float(row["latitude"]), long=float(row["longitude"])),
            GpsCoordinate(lat=match["predicted_lat"], long=match["predicted_lon"]),
        )
        errors.append(distance)
        error_rows.append(
            dict(
                frame=match["frame"],
                timestamp=match["timestamp"],
                horizontal_reference_distance_m=distance,
            )
        )
    with (root / "frame_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame",
                "timestamp",
                "rotation_cw",
                "pnp_verified",
                "max_unrotated_raw_matches",
                "max_searched_raw_matches",
                "best_homography_inliers",
                "pnp_inliers",
                "reprojection_mean_px",
            ]
        )
        for f in source["frames"]:
            baseline = unrotated_count(f)
            writer.writerow(
                [
                    f["frame"],
                    f["timestamp"],
                    f.get("rotation_cw"),
                    f["matched"],
                    baseline,
                    max(a["raw_matches"] for a in f["attempts"]),
                    f.get("best_homography_inliers", 0),
                    f.get("pnp_inliers", 0),
                    f.get("reprojection_mean_px"),
                ]
            )
    stage_counts = Counter(a["stage"] for f in source["frames"] for a in f["attempts"])
    reasons = Counter(
        a["rejection_detail"]
        for f in source["frames"]
        for a in f["attempts"]
        if "rejection_detail" in a
    )
    minimum = source["geometric_thresholds"].get("pnp_min_inliers", 20)
    frame_diagnostics = dict(
        best_raw_count_below_required=sum(
            max(a["raw_matches"] for a in f["attempts"]) < minimum
            for f in source["frames"]
        ),
        best_homography_count_below_required=sum(
            max(a["homography_inliers"] for a in f["attempts"]) < minimum
            for f in source["frames"]
        ),
        any_pnp_geometry_attempt=sum(
            any(
                a["stage"] in ("pnp_passed", "pnp_geometry_rejected")
                for a in f["attempts"]
            )
            for f in source["frames"]
        ),
        unrotated_frames_tested=sum(
            unrotated_count(f) is not None for f in source["frames"]
        ),
        unrotated_pnp_passes=(
            sum(
                any(
                    a["rotation_cw"] == 0 and a["stage"] == "pnp_passed"
                    for a in f["attempts"]
                )
                for f in source["frames"]
            )
            if any(unrotated_count(f) is not None for f in source["frames"])
            else None
        ),
    )
    timestamps = np.array([r["timestamp"] for r in source["frames"]])
    seconds = timestamps - timestamps[0]
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    valid = np.array([r["matched"] for r in source["frames"]])
    for accepted, color, label in (
        (
            False,
            "#92979c",
            "User-prescribed angle; PnP rejected"
            if fixed
            else "Tentative angle, NOT verified",
        ),
        (True, "#218b66", "PnP-verified rotation"),
    ):
        rows = [
            (t, r)
            for t, r in zip(seconds, source["frames"])
            if r["matched"] == accepted and "rotation_cw" in r
        ]
        if rows:
            axes[0].scatter(
                [t for t, _ in rows],
                [r["rotation_cw"] for _, r in rows],
                c=color,
                s=22,
                label=label,
            )
    baseline = [unrotated_count(f) for f in source["frames"]]
    searched = [max(a["raw_matches"] for a in f["attempts"]) for f in source["frames"]]
    if any(count is not None for count in baseline):
        axes[1].plot(
            seconds,
            [np.nan if count is None else count for count in baseline],
            color="#737a80",
            label="Unrotated, best tested query scale",
        )
    axes[1].plot(
        seconds,
        searched,
        color="#347bb8",
        label="Best raw count at user angle across scales (not necessarily valid)"
        if fixed
        else "Best raw count across searched angles/scales (not necessarily valid)",
    )
    axes[2].bar(
        seconds,
        [r.get("pnp_inliers", 0) for r in source["frames"]],
        width=1.5,
        color="#218b66",
    )
    axes[0].set(
        ylabel="Clockwise angle (deg)", yticks=[0, 90, 180, 270, 360], ylim=(-10, 370)
    )
    axes[1].set(ylabel="Raw SuperGlue matches")
    axes[2].set(ylabel="Verified PnP inliers", xlabel="Native seconds since first image")
    for axis in axes:
        axis.grid(alpha=0.2)
    for axis in axes[:2]:
        axis.legend(fontsize=8)
    figure.suptitle(
        f"{Path(source['query_folder']).name} physical rotation search | {len(matches)}/{len(source['frames'])} PnP frames | {point_count} points"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(root / "rotation_summary.png", dpi=140)
    plt.close(figure)
    candidates = [r for r in source["frames"] if "rotated_image" in r]
    comparison_frames = []
    if candidates:
        ids = (
            [
                i
                for i, frame in enumerate(candidates)
                if i == 0 or frame["rotation_cw"] != candidates[i - 1]["rotation_cw"]
            ]
            if fixed
            else np.linspace(0, len(candidates) - 1, min(4, len(candidates)), dtype=int)
        )
        comparison_frames = [candidates[i]["frame"] for i in ids]
        figure, axes = plt.subplots(
            len(ids), 2, figsize=(12, 3.7 * len(ids)), squeeze=False
        )
        for row, idx in enumerate(ids):
            f = candidates[idx]
            before = cv2.imread(
                str(Path(source["query_folder"]) / f["filename"]), cv2.IMREAD_GRAYSCALE
            )
            after = cv2.imread(f["rotated_image"], cv2.IMREAD_GRAYSCALE)
            axes[row, 0].imshow(before, cmap="gray")
            axes[row, 0].set_title(f"{f['frame']} | Original")
            axes[row, 1].imshow(after, cmap="gray")
            status = (
                "PnP VERIFIED"
                if f["matched"]
                else (
                    "USER ROTATION; PnP REJECTED" if fixed else "TENTATIVE, NOT VERIFIED"
                )
            )
            axes[row, 1].set_title(f"Actual CW {f['rotation_cw']:g} deg | {status}")
            for axis in axes[row]:
                axis.axis("off")
        figure.tight_layout()
        figure.savefig(root / "before_after.png", dpi=120)
        plt.close(figure)
    value = dict(
        status="complete",
        total_images=len(source["frames"]),
        physically_rotated_images=len(candidates),
        pnp_verified_frames=len(matches),
        rotation_policy=source.get("rotation_policy", "automatic_angle_search"),
        user_fixed_rotation_frames=len(candidates) if fixed else 0,
        comparison_frames=comparison_frames,
        superpoint_keypoint_threshold=source["superpoint_keypoint_threshold"],
        pnp_homography_prefilter=source["geometric_thresholds"].get(
            "pnp_homography_prefilter", False
        ),
        exported_inlier_points=point_count,
        tentative_frames=0 if fixed else len(candidates) - len(matches),
        unverified_geometry_frames=len(source["frames"]) - len(matches),
        unresolved_frames=len(source["frames"]) - len(candidates),
        stage_counts=dict(stage_counts),
        pnp_rejection_details=dict(reasons),
        per_frame_diagnostics=frame_diagnostics,
        max_inverse_pixel_error=max_inverse_error,
        input_hash_checks=len(source["input_sha256"]),
        GPS_reference_used_for_selection=False,
        evaluation_scope="Metadata RTK versus PnP camera center; lever arm unmodelled; no alignment or filtering",
        reference_distance_m=(
            dict(
                count=len(errors),
                rmse=float(np.sqrt(np.mean(np.square(errors)))),
                median=float(np.median(errors)),
                max=float(max(errors)),
            )
            if errors
            else None
        ),
        limitations=[
            "More raw matches do not certify a correct rotation, tile or position.",
            "Tentative rotated images must not be used as verified satellite constraints.",
            "Only query scale is searched at user-prescribed angles; no automatic angle override."
            if fixed
            else "Angle and query scale are searched together; this is not a rotation-only ablation.",
            "Detector threshold and optional PnP homography prefilter are recorded policy changes, not evidence that rotation alone fixes the dataset.",
            "No AIRSLAM BA insertion or bag replay performed.",
        ],
        input_sha256={
            str(root / p): sha(root / p)
            for p in ("rotation_manifest.json", "matched_keypoints.json")
        },
    )
    (root / "position_reference_audit.json").write_text(
        json.dumps(error_rows, indent=2) + "\n"
    )
    target.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output.resolve())
