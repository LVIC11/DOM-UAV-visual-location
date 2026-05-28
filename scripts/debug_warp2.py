#!/usr/bin/env python3
"""Debug script to test different coordinate conventions for warping."""

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def get_lat_long_from_tile_xy(x, y, zoom):
    """Get lat/lon from tile coordinates."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def main():
    # Camera intrinsics from nav_left.yaml
    fx, fy = 820.22, 823.91
    cx, cy = 642.51, 370.58
    pitch_deg = 30.0

    # Load poses
    poses = {}
    with open('data/my_experiment/rtk_poses.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            poses[frame] = {
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'altitude': float(row['altitude']),
                'heading': float(row['heading']),
            }

    # Test with frame 14 (stable altitude ~173m)
    frame = 14
    pose = poses[frame]
    drone_lat = pose['latitude']
    drone_lon = pose['longitude']
    drone_alt = pose['altitude']
    heading_deg = pose['heading']

    print(f"Frame {frame}: GPS=({drone_lat:.6f}, {drone_lon:.6f}), Alt={drone_alt:.1f}m, Heading={heading_deg:.1f}°")

    # Compute tile coordinates
    zoom = 18
    n = 2.0 ** zoom
    tile_x = int((drone_lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(drone_lat)
    tile_y = int((1.0 - math.log(math.tan(lat_rad) + 1.0/math.cos(lat_rad)) / math.pi) / 2.0 * n)
    print(f"Tile: ({tile_x}, {tile_y})")

    # Load satellite image
    sat_path = Path(f'data/my_experiment/map/tiles/{tile_x}_{tile_y}_{zoom}.png')
    if not sat_path.exists():
        for p in Path('data/my_experiment/map/tiles').glob(f'*{tile_x}*{tile_y}*'):
            sat_path = p
            break
    print(f"Satellite path: {sat_path}")
    sat_img = cv2.imread(str(sat_path))
    if sat_img is None:
        print("Failed to load satellite image")
        return

    print(f"Satellite image size: {sat_img.shape[:2]}")

    # Get tile corners in GPS
    tile_north, tile_west = get_lat_long_from_tile_xy(tile_x, tile_y, zoom)
    tile_south, tile_east = get_lat_long_from_tile_xy(tile_x + 1, tile_y + 1, zoom)
    print(f"Tile GPS: N={tile_north:.6f}, S={tile_south:.6f}, W={tile_west:.6f}, E={tile_east:.6f}")

    # Convert GPS to local meters relative to drone
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * math.cos(math.radians(drone_lat))

    tile_corners_m = np.array([
        [(tile_west - drone_lon) * lon_to_m, (tile_north - drone_lat) * lat_to_m],
        [(tile_east - drone_lon) * lon_to_m, (tile_north - drone_lat) * lat_to_m],
        [(tile_east - drone_lon) * lon_to_m, (tile_south - drone_lat) * lat_to_m],
        [(tile_west - drone_lon) * lon_to_m, (tile_south - drone_lat) * lat_to_m],
    ], dtype=np.float32)

    print(f"Tile corners (meters relative to drone):")
    for i, name in enumerate(['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']):
        print(f"  {name}: ({tile_corners_m[i, 0]:.1f}, {tile_corners_m[i, 1]:.1f})")

    # Camera intrinsics
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Try different rotation conventions
    heading = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    # Convention 1: Original (z-rotation for heading, y-rotation for pitch)
    R_h1 = Rotation.from_euler('z', -heading_deg, degrees=True).as_matrix()
    R_p1 = Rotation.from_euler('y', pitch_deg, degrees=True).as_matrix()
    R1 = R_p1 @ R_h1

    # Convention 2: Pitch first, then heading
    R2 = R_h1 @ R_p1

    # Convention 3: Different axis order (x for pitch, z for heading)
    R_p3 = Rotation.from_euler('x', pitch_deg, degrees=True).as_matrix()
    R3 = R_p3 @ R_h1

    # Convention 4: No negative heading
    R_h4 = Rotation.from_euler('z', heading_deg, degrees=True).as_matrix()
    R4 = R_p1 @ R_h4

    conventions = [
        ("Original (z=-heading, y=pitch)", R1),
        ("Pitch then heading", R2),
        ("x-pitch, z-heading", R3),
        ("z=+heading, y=pitch", R4),
    ]

    output_dir = Path('data/my_experiment/debug')
    output_dir.mkdir(exist_ok=True)

    for name, R_cam in conventions:
        print(f"\n--- {name} ---")
        print(f"R_cam:\n{R_cam}")

        # Ground to image homography
        H = np.column_stack([R_cam[:, 0], R_cam[:, 1], -drone_alt * R_cam[:, 2]])
        H = K @ H

        print(f"H:\n{H}")

        # Transform tile corners to image
        tile_corners_h = np.column_stack([tile_corners_m, np.ones(4)])
        img_corners = (H @ tile_corners_h.T).T
        img_corners[:, :2] /= img_corners[:, 2:3]

        print(f"Image corners:")
        for i, cname in enumerate(['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']):
            print(f"  {cname}: ({img_corners[i, 0]:.1f}, {img_corners[i, 1]:.1f})")

        # Warp
        sat_h, sat_w = sat_img.shape[:2]
        sat_corners = np.array([
            [0, 0],
            [sat_w-1, 0],
            [sat_w-1, sat_h-1],
            [0, sat_h-1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(sat_corners, img_corners[:, :2].astype(np.float32))
        img_w, img_h = 1280, 720
        warped = cv2.warpPerspective(sat_img, M, (img_w, img_h))

        # Save
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')
        cv2.imwrite(str(output_dir / f'frame_{frame:06d}_{safe_name}.jpg'), warped)

        # Check if center is in image
        center_img = H @ np.array([0, 0, 1])
        center_img /= center_img[2]
        print(f"Ground origin (0,0) maps to image: ({center_img[0]:.1f}, {center_img[1]:.1f})")

    # Also load drone image for comparison
    drone_path = Path(f'data/my_experiment/query/{frame:06d}.jpg')
    if drone_path.exists():
        drone_img = cv2.imread(str(drone_path))
        cv2.imwrite(str(output_dir / f'frame_{frame:06d}_drone.jpg'), drone_img)

    print(f"\nSaved debug images to {output_dir}")


if __name__ == "__main__":
    main()
