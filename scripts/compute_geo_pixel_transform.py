#!/usr/bin/env python3
"""Compute transformation between geographic plane and pixel coordinates using prior pose."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def get_camera_intrinsics(fx, fy, cx, cy):
    """Build camera intrinsics matrix."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def rotation_matrix_from_euler(roll, pitch, yaw):
    """Compute rotation matrix from roll, pitch, yaw (degrees)."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    return r


def compute_ground_to_pixel_homography(
    lat, lon, alt, heading, pitch,
    fx, fy, cx, cy,
    ref_lat, ref_lon, meters_per_pixel
):
    """Compute homography from ground plane (meters) to pixel coordinates.

    Parameters
    ----------
    lat, lon, alt : float
        Drone position (GPS + altitude)
    heading, pitch : float
        Drone orientation (degrees)
    camera_focal_length_px : float
        Camera focal length in pixels
    image_width, image_height : int
        Image dimensions
    ref_lat, ref_lon : float
        Reference point for local coordinate system (top-left of satellite tile)
    meters_per_pixel : float
        Ground resolution of the local coordinate system

    Returns
    -------
    np.ndarray
        3x3 homography matrix H that maps ground meters to pixel coordinates
    """
    # Camera intrinsics
    K = get_camera_intrinsics(fx, fy, cx, cy)

    # Camera rotation: R = R_yaw @ R_pitch
    R_yaw = rotation_matrix_from_euler(0, 0, heading)
    R_pitch = rotation_matrix_from_euler(0, pitch, 0)
    R_cam = R_yaw @ R_pitch

    # Project ground plane to image: H = K @ R_cam
    # Ground plane basis vectors (in meters relative to ref point)
    # e1 = [1, 0, 0] (east), e2 = [0, 1, 0] (north)
    # Camera projects these to image plane

    # For a camera looking down at pitch angle from altitude h:
    # The ground-to-image homography is: H = K @ [R*e1, R*e2, t]
    # where t = R @ [0, 0, altitude]^T

    e1 = np.array([1, 0, 0])  # east
    e2 = np.array([0, 1, 0])  # north

    R_e1 = R_cam @ e1
    R_e2 = R_cam @ e2
    t = R_cam @ np.array([0, 0, alt])

    H = np.column_stack([R_e1, R_e2, t])
    H = K @ H

    return H


def gps_to_local_meters(lat, lon, ref_lat, ref_lon):
    """Convert GPS coordinates to local meters relative to reference point."""
    # Approximate conversion: 1 degree latitude ≈ 111320 meters
    # 1 degree longitude ≈ 111320 * cos(latitude) meters
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * math.cos(math.radians(ref_lat))

    x_meters = (lon - ref_lon) * lon_to_m
    y_meters = (lat - ref_lat) * lat_to_m

    return x_meters, y_meters


def main():
    parser = argparse.ArgumentParser(description="Compute geo-pixel transformation")
    parser.add_argument("--poses-csv", type=str, required=True, help="RTK poses CSV")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV with transformations")
    parser.add_argument("--fx", type=float, default=820.216716801767, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=823.9122506176146, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=642.5062817866961, help="Principal point x")
    parser.add_argument("--cy", type=float, default=370.5784082465089, help="Principal point y")
    parser.add_argument("--image-width", type=int, default=1280, help="Image width")
    parser.add_argument("--image-height", type=int, default=720, help="Image height")
    parser.add_argument("--pitch", type=float, default=30.0, help="Camera pitch angle (degrees)")

    args = parser.parse_args()

    # Load poses
    poses = []
    with open(args.poses_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                'frame': int(row['frame']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'altitude': float(row['altitude']),
                'heading': float(row['heading']),
            })

    print(f"Loaded {len(poses)} poses")

    # Compute transformations
    results = []
    for pose in poses:
        H = compute_ground_to_pixel_homography(
            lat=pose['latitude'],
            lon=pose['longitude'],
            alt=pose['altitude'],
            heading=pose['heading'],
            pitch=args.pitch,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            ref_lat=pose['latitude'],
            ref_lon=pose['longitude'],
            meters_per_pixel=1.0,
        )

        # Flatten homography for CSV storage
        H_flat = H.flatten()
        results.append({
            'frame': pose['frame'],
            'latitude': pose['latitude'],
            'longitude': pose['longitude'],
            'altitude': pose['altitude'],
            'heading': pose['heading'],
            'H': ','.join(f'{h:.6f}' for h in H_flat),
        })

    # Save results
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'latitude', 'longitude', 'altitude', 'heading', 'H'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved transformations to {args.output_csv}")

    # Print example
    print("\nExample transformation for frame 0:")
    H = np.array([float(x) for x in results[0]['H'].split(',')]).reshape(3, 3)
    print(H)


if __name__ == "__main__":
    main()
