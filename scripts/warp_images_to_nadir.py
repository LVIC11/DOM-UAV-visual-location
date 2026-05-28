#!/usr/bin/env python3
"""Warp oblique drone images to nadir (top-down) view for satellite matching."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def rotation_matrix_from_angles(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Compute rotation matrix from roll, pitch, yaw angles (in degrees)."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    return r


def get_intrinsics(focal_length_px: float, width: int, height: int) -> np.ndarray:
    """Build camera intrinsics matrix."""
    return np.array([
        [focal_length_px, 0, width / 2],
        [0, focal_length_px, height / 2],
        [0, 0, 1],
    ], dtype=np.float64)


def compute_headings_from_trajectory(metadata: list) -> dict:
    """Compute heading for each image from RTK trajectory.

    Parameters
    ----------
    metadata : list
        list of dicts with 'filename', 'latitude', 'longitude' keys

    Returns
    -------
    dict
        mapping from filename to heading in degrees
    """
    headings = {}
    for i in range(len(metadata)):
        if i == 0:
            # First image: use direction to next
            if len(metadata) > 1:
                dlat = metadata[i+1]['latitude'] - metadata[i]['latitude']
                dlon = metadata[i+1]['longitude'] - metadata[i]['longitude']
                headings[metadata[i]['filename']] = math.degrees(math.atan2(dlon, dlat))
            else:
                headings[metadata[i]['filename']] = 0.0
        elif i == len(metadata) - 1:
            # Last image: use direction from previous
            dlat = metadata[i]['latitude'] - metadata[i-1]['latitude']
            dlon = metadata[i]['longitude'] - metadata[i-1]['longitude']
            headings[metadata[i]['filename']] = math.degrees(math.atan2(dlon, dlat))
        else:
            # Middle images: average of前后 directions
            dlat1 = metadata[i]['latitude'] - metadata[i-1]['latitude']
            dlon1 = metadata[i]['longitude'] - metadata[i-1]['longitude']
            dlat2 = metadata[i+1]['latitude'] - metadata[i]['latitude']
            dlon2 = metadata[i+1]['longitude'] - metadata[i]['longitude']
            heading1 = math.atan2(dlon1, dlat1)
            heading2 = math.atan2(dlon2, dlat2)
            # Average angles properly
            avg_heading = math.atan2(
                math.sin(heading1) + math.sin(heading2),
                math.cos(heading1) + math.cos(heading2)
            )
            headings[metadata[i]['filename']] = math.degrees(avg_heading)

    return headings


def warp_image_to_nadir(
    image: np.ndarray,
    heading: float,
    pitch: float,
    focal_length_px: float,
    altitude: float = 100.0,
) -> np.ndarray:
    """Warp an oblique drone image to nadir (top-down) view.

    Parameters
    ----------
    image : np.ndarray
        input oblique image
    heading : float
        drone heading in degrees (yaw)
    pitch : float
        camera pitch angle in degrees (0=nadir, 90=horizon)
    focal_length_px : float
        focal length in pixels
    altitude : float
        flight altitude in meters (affects scale)

    Returns
    -------
    np.ndarray
        warped image in nadir view
    """
    height, width = image.shape[:2]

    # Camera intrinsics
    K = get_intrinsics(focal_length_px, width, height)

    # The camera orientation in world frame:
    # 1. Drone yaw rotation
    # 2. Gimbal pitch rotation (camera tilted from nadir)
    # Combined: R_cam = R_yaw @ R_pitch

    R_yaw = rotation_matrix_from_angles(roll=0, pitch=0, yaw=heading)
    R_pitch = rotation_matrix_from_angles(roll=0, pitch=pitch, yaw=0)

    # Full camera rotation in world frame
    R_cam = R_yaw @ R_pitch

    # To transform from oblique view to nadir view:
    # We need to apply the inverse of the camera rotation
    # T = K @ R_nadir @ inv(R_cam) @ inv(K)
    # Where R_nadir = I (identity, since nadir is the target)

    R_target = np.eye(3)  # Identity for nadir view

    # Transformation matrix
    transformation_matrix = K @ R_target @ np.linalg.inv(R_cam) @ np.linalg.inv(K)

    # Define image corners
    corners = np.array([
        [0, 0, 1],
        [width - 1, 0, 1],
        [width - 1, height - 1, 1],
        [0, height - 1, 1],
    ], dtype=np.float64).T

    # Apply transformation to corners
    warped_corners = transformation_matrix @ corners

    # Normalize homogeneous coordinates
    warped_corners /= warped_corners[2]

    # Remove third component
    warped_corners = warped_corners[:2].T

    # Shift to origin (so minimum corner is at 0,0)
    warped_corners -= warped_corners.min(axis=0)

    # Original corners (in pixel coordinates)
    orig_corners = corners[:2].T.astype(np.float32)
    warped_corners = warped_corners.astype(np.float32)

    # Compute perspective transform
    dst = cv2.getPerspectiveTransform(orig_corners, warped_corners)

    # Compute output size
    new_size = warped_corners.max(axis=0).astype(np.int32)
    # Ensure minimum size
    new_size = np.maximum(new_size, np.array([width, height]))

    # Warp image
    warped_image = cv2.warpPerspective(image, dst, tuple(new_size))

    return warped_image


def main():
    parser = argparse.ArgumentParser(description="Warp oblique drone images to nadir view")
    parser.add_argument("--query-dir", type=str, required=True, help="directory with query images and metadata.csv")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory for warped images")
    parser.add_argument("--pitch", type=float, default=45.0, help="camera pitch angle in degrees (default: 45)")
    parser.add_argument("--focal-length-px", type=float, default=None, help="focal length in pixels (default: compute from HFOV)")
    parser.add_argument("--hfov-deg", type=float, default=82.9, help="horizontal field of view in degrees")
    parser.add_argument("--image-width", type=int, default=4056, help="image width in pixels")
    parser.add_argument("--image-height", type=int, default=3040, help="image height in pixels")

    args = parser.parse_args()

    query_dir = Path(args.query_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute focal length in pixels if not provided
    if args.focal_length_px is None:
        hfov_rad = math.radians(args.hfov_deg)
        focal_length_px = args.image_width / (2 * math.tan(hfov_rad / 2))
    else:
        focal_length_px = args.focal_length_px

    print(f"Camera parameters:")
    print(f"  Image size: {args.image_width}x{args.image_height}")
    print(f"  HFOV: {args.hfov_deg}°")
    print(f"  Focal length: {focal_length_px:.1f} px")
    print(f"  Pitch angle: {args.pitch}°")

    # Load metadata
    metadata_csv = query_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found in {query_dir}")

    metadata = []
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append({
                'filename': row['filename'],
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'altitude': float(row['altitude']),
                'timestamp': float(row['timestamp']),
            })

    print(f"\nLoaded {len(metadata)} images from metadata.csv")

    # Compute headings from trajectory
    headings = compute_headings_from_trajectory(metadata)

    # Process each image
    print(f"\nWarping images to nadir view...")
    for i, meta in enumerate(metadata):
        filename = meta['filename']
        input_path = query_dir / filename
        output_path = output_dir / filename

        if not input_path.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue

        # Read image
        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

        # Get heading
        heading = headings[filename]

        # Warp to nadir
        warped = warp_image_to_nadir(
            image=image,
            heading=heading,
            pitch=args.pitch,
            focal_length_px=focal_length_px,
            altitude=meta['altitude'],
        )

        # Save warped image
        cv2.imwrite(str(output_path), warped)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(metadata)} images")

    # Save output metadata
    output_metadata = output_dir / "metadata.csv"
    with open(output_metadata, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'latitude', 'longitude', 'altitude', 'timestamp', 'heading'])
        writer.writeheader()
        for meta in metadata:
            meta['heading'] = headings[meta['filename']]
            writer.writerow(meta)

    print(f"\nDone! Warped {len(metadata)} images to {output_dir}")
    print(f"Metadata saved to {output_metadata}")


if __name__ == "__main__":
    main()
