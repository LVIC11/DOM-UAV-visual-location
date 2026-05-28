#!/usr/bin/env python3
"""Warp drone images to satellite coordinate system using SLAM poses."""

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def get_lat_long_from_tile_xy(x, y, zoom):
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def main():
    # Camera intrinsics
    fx, fy = 820.22, 823.91
    cx, cy = 642.51, 370.58
    img_w, img_h = 1280, 720
    pitch_deg = 30.0

    # Load SLAM GPS poses
    poses = {}
    with open('data/my_experiment/slam_gps_poses.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            poses[i] = {
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'altitude': float(row['altitude']),
                'heading': float(row['heading']),
            }

    zoom = 18
    output_dir = Path('data/my_experiment/warped_slam')
    output_dir.mkdir(parents=True, exist_ok=True)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Only process frames 0-49 (matching query images)
    for frame in range(min(50, len(poses))):
        if frame not in poses:
            continue

        pose = poses[frame]
        drone_lat = pose['latitude']
        drone_lon = pose['longitude']
        drone_alt = pose['altitude']
        heading_deg = pose['heading']

        # Tile coordinates
        n = 2.0 ** zoom
        tile_x = int((drone_lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(drone_lat)
        tile_y = int((1.0 - math.log(math.tan(lat_rad) + 1.0/math.cos(lat_rad)) / math.pi) / 2.0 * n)

        # Get tile GPS bounds
        tile_north, tile_west = get_lat_long_from_tile_xy(tile_x, tile_y, zoom)
        tile_south, tile_east = get_lat_long_from_tile_xy(tile_x + 1, tile_y + 1, zoom)

        # Convert GPS to meters
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * math.cos(math.radians(drone_lat))
        tile_w_m = (tile_east - tile_west) * lon_to_m
        tile_h_m = (tile_north - tile_south) * lat_to_m

        # Drone position in meters relative to tile top-left
        drone_x_m = (drone_lon - tile_west) * lon_to_m
        drone_y_m = (tile_north - drone_lat) * lat_to_m

        # Camera rotation
        R_heading = Rotation.from_euler('z', -heading_deg, degrees=True).as_matrix()
        R_pitch = Rotation.from_euler('y', pitch_deg, degrees=True).as_matrix()
        R_cam = R_heading @ R_pitch
        R_world = R_cam.T

        # Image corners
        img_corners = np.array([
            [0, 0, 1],
            [img_w-1, 0, 1],
            [img_w-1, img_h-1, 1],
            [0, img_h-1, 1]
        ], dtype=np.float64).T

        # Undistort to camera rays
        rays = K_inv @ img_corners

        # Rotate to world frame
        rays_world = R_world @ rays

        # Intersect with ground plane
        t = -drone_alt / rays_world[2, :]

        # Ground points in meters relative to tile top-left
        ground_pts = t * rays_world
        ground_pts[0, :] += drone_x_m
        ground_pts[1, :] += drone_y_m

        # Convert to satellite tile pixel coordinates
        meters_per_pixel_x = tile_w_m / 256.0
        meters_per_pixel_y = tile_h_m / 256.0

        sat_corners = np.array([
            [ground_pts[0, 0] / meters_per_pixel_x, ground_pts[1, 0] / meters_per_pixel_y],
            [ground_pts[0, 1] / meters_per_pixel_x, ground_pts[1, 1] / meters_per_pixel_y],
            [ground_pts[0, 2] / meters_per_pixel_x, ground_pts[1, 2] / meters_per_pixel_y],
            [ground_pts[0, 3] / meters_per_pixel_x, ground_pts[1, 3] / meters_per_pixel_y],
        ], dtype=np.float32)

        src_corners = np.array([
            [0, 0],
            [img_w-1, 0],
            [img_w-1, img_h-1],
            [0, img_h-1]
        ], dtype=np.float32)

        M, _ = cv2.findHomography(src_corners, sat_corners)
        if M is None:
            print(f"Frame {frame}: homography failed")
            continue

        # Load drone image
        drone_path = Path(f'data/my_experiment/query/{frame:06d}.jpg')
        if not drone_path.exists():
            print(f"Frame {frame}: drone image not found")
            continue

        drone_img = cv2.imread(str(drone_path))

        # Warp
        warped = cv2.warpPerspective(drone_img, M, (256, 256))

        # Save
        cv2.imwrite(str(output_dir / f'{frame:06d}_warped.jpg'), warped)

        # Satellite for comparison
        sat_path = None
        for ext in ['.png', '.jpg']:
            p = Path(f'data/my_experiment/map/tiles/{tile_x}_{tile_y}_{zoom}{ext}')
            if p.exists():
                sat_path = p
                break
        if sat_path:
            sat_img = cv2.imread(str(sat_path))
            comparison = np.hstack([warped, sat_img])
            cv2.imwrite(str(output_dir / f'{frame:06d}_comparison.jpg'), comparison)

        print(f"Frame {frame}: Heading={heading_deg:.1f}°, Alt={drone_alt:.1f}m, GPS=({drone_lat:.6f}, {drone_lon:.6f})")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
