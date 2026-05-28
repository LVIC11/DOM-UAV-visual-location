#!/usr/bin/env python3
"""Debug script to visualize satellite-to-drone warping."""

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
    pitch = 30.0

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
    heading = pose['heading']

    print(f"Frame {frame}: GPS=({drone_lat:.6f}, {drone_lon:.6f}), Alt={drone_alt:.1f}m, Heading={heading:.1f}°")

    # Compute tile coordinates
    zoom = 18
    n = 2.0 ** zoom
    tile_x = int((drone_lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(drone_lat)
    tile_y = int((1.0 - math.log(math.tan(lat_rad) + 1.0/math.cos(lat_rad)) / math.pi) / 2.0 * n)
    print(f"Tile: ({tile_x}, {tile_y})")

    # Load satellite image
    sat_path = Path(f'data/my_experiment/map/tiles/{tile_x}_{tile_y}_{zoom}.jpg')
    if not sat_path.exists():
        # Try other naming patterns
        for p in Path('data/my_experiment/map/tiles').glob(f'*{tile_x}*{tile_y}*'):
            sat_path = p
            break
    print(f"Satellite path: {sat_path}")
    sat_img = cv2.imread(str(sat_path))
    if sat_img is None:
        print("Failed to load satellite image")
        return

    print(f"Satellite image size: {sat_img.shape[:2]}")

    # Get tile corners in GPS (top-left and bottom-right)
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

    # Rotation
    R_heading = Rotation.from_euler('z', -heading, degrees=True).as_matrix()
    R_pitch = Rotation.from_euler('y', pitch, degrees=True).as_matrix()
    R_cam = R_pitch @ R_heading

    print(f"\nHeading: {heading:.1f}°, Pitch: {pitch:.1f}°")
    print(f"R_heading:\n{R_heading}")
    print(f"R_pitch:\n{R_pitch}")
    print(f"R_cam:\n{R_cam}")

    # Ground to image homography: H = K @ [R[:,0], R[:,1], -h*R[:,2]]
    R_col0 = R_cam[:, 0]
    R_col1 = R_cam[:, 1]
    R_col2 = R_cam[:, 2]

    H = np.column_stack([R_col0, R_col1, -drone_alt * R_col2])
    H = K @ H

    print(f"\nHomography H (ground to image):")
    print(H)

    # Transform tile corners to image
    tile_corners_h = np.column_stack([tile_corners_m, np.ones(4)])
    img_corners = (H @ tile_corners_h.T).T
    img_corners[:, :2] /= img_corners[:, 2:3]

    print(f"\nTile corners in image coordinates:")
    for i, name in enumerate(['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']):
        print(f"  {name}: ({img_corners[i, 0]:.1f}, {img_corners[i, 1]:.1f})")

    # Satellite pixel corners
    sat_h, sat_w = sat_img.shape[:2]
    sat_corners = np.array([
        [0, 0],
        [sat_w-1, 0],
        [sat_w-1, sat_h-1],
        [0, sat_h-1]
    ], dtype=np.float32)

    print(f"\nSatellite corners: {sat_corners}")

    # Perspective transform
    M = cv2.getPerspectiveTransform(sat_corners, img_corners[:, :2].astype(np.float32))
    print(f"\nPerspective matrix M:")
    print(M)

    # Output size
    img_w, img_h = 1280, 720
    warped = cv2.warpPerspective(sat_img, M, (img_w, img_h))

    # Save debug images
    output_dir = Path('data/my_experiment/debug')
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / f'frame_{frame:06d}_satellite.jpg'), sat_img)
    cv2.imwrite(str(output_dir / f'frame_{frame:06d}_warped.jpg'), warped)

    # Also load drone image for comparison
    drone_path = Path(f'data/my_experiment/query/{frame:06d}.jpg')
    if drone_path.exists():
        drone_img = cv2.imread(str(drone_path))
        cv2.imwrite(str(output_dir / f'frame_{frame:06d}_drone.jpg'), drone_img)

    print(f"\nSaved debug images to {output_dir}")

    # Create side-by-side comparison
    if drone_path.exists():
        drone_img = cv2.imread(str(drone_path))
        # Resize to same height
        h = max(drone_img.shape[0], warped.shape[0])
        drone_resized = cv2.resize(drone_img, (int(drone_img.shape[1] * h / drone_img.shape[0]), h))
        warped_resized = cv2.resize(warped, (int(warped.shape[1] * h / warped.shape[0]), h))

        comparison = np.hstack([drone_resized, warped_resized])
        cv2.imwrite(str(output_dir / f'frame_{frame:06d}_comparison.jpg'), comparison)
        print(f"Saved comparison image")


if __name__ == "__main__":
    main()
