#!/usr/bin/env python3
"""Warp drone images to satellite coordinate system (top-down view)."""

import csv
import math
import os
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
    # Camera intrinsics from nav_left.yaml
    fx, fy = 820.22, 823.91
    cx, cy = 642.51, 370.58
    img_w, img_h = 1280, 720

    # Load poses
    poses = {}
    with open('data/my_experiment/rtk_poses_with_heading.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            poses[i] = {
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'altitude': float(row['altitude']),
                'heading': float(row['heading']),
            }

    zoom = 18
    output_dir = Path('data/my_experiment/warped_to_sat')
    output_dir.mkdir(parents=True, exist_ok=True)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    pitch_deg = 30.0

    for frame, pose in sorted(poses.items()):
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

        # Convert GPS to meters relative to tile top-left corner
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * math.cos(math.radians(drone_lat))
        tile_w_m = (tile_east - tile_west) * lon_to_m
        tile_h_m = (tile_north - tile_south) * lat_to_m

        # Drone position in meters relative to tile top-left
        drone_x_m = (drone_lon - tile_west) * lon_to_m
        drone_y_m = (tile_north - drone_lat) * lat_to_m

        # Camera rotation: heading then pitch
        R_heading = Rotation.from_euler('z', -heading_deg, degrees=True).as_matrix()
        R_pitch = Rotation.from_euler('y', pitch_deg, degrees=True).as_matrix()
        R_cam = R_heading @ R_pitch

        # Image to ground homography
        # From paper: pixel (u,v) -> ground (x,y)
        # 1. Undistort pixel to camera ray: d = K_inv @ [u,v,1]
        # 2. Rotate to world frame: R_cam^T @ d
        # 3. Intersect with ground plane z=0 at altitude h
        # Ground point = -h * (R^T @ d) / (R^T @ d)_z + drone_pos

        # For homography: we need to map image corners to ground plane
        # Image corners in pixel coordinates
        img_corners = np.array([
            [0, 0, 1],
            [img_w-1, 0, 1],
            [img_w-1, img_h-1, 1],
            [0, img_h-1, 1]
        ], dtype=np.float64).T  # 3x4

        # Undistort to camera rays
        rays = K_inv @ img_corners  # 3x4

        # Rotate to world frame
        R_world = R_cam.T  # camera to world
        rays_world = R_world @ rays  # 3x4

        # Intersect with ground plane (z = -altitude in camera frame, or z=0 in world)
        # The ground plane in world coordinates is at z=0
        # Camera is at (0, 0, altitude) in world frame
        # Ray: P = cam_pos + t * ray_dir
        # Ground intersection: cam_z + t * ray_z = 0 => t = -cam_z / ray_z
        # cam_pos in world = [0, 0, altitude] (above ground)
        t = -drone_alt / rays_world[2, :]  # 4 intersection depths

        # Ground points in world frame (meters relative to drone)
        ground_pts = t * rays_world  # 3x4

        # Add drone position to get meters relative to tile top-left
        ground_pts[0, :] += drone_x_m
        ground_pts[1, :] += drone_y_m

        # The ground plane in satellite tile pixel coordinates
        # satellite pixel = ground_meters / meters_per_pixel
        meters_per_pixel_x = tile_w_m / 256.0
        meters_per_pixel_y = tile_h_m / 256.0

        sat_corners = np.array([
            [ground_pts[0, 0] / meters_per_pixel_x, ground_pts[1, 0] / meters_per_pixel_y],
            [ground_pts[0, 1] / meters_per_pixel_x, ground_pts[1, 1] / meters_per_pixel_y],
            [ground_pts[0, 2] / meters_per_pixel_x, ground_pts[1, 2] / meters_per_pixel_y],
            [ground_pts[0, 3] / meters_per_pixel_x, ground_pts[1, 3] / meters_per_pixel_y],
        ], dtype=np.float32)

        # Image corners (source)
        src_corners = np.array([
            [0, 0],
            [img_w-1, 0],
            [img_w-1, img_h-1],
            [0, img_h-1]
        ], dtype=np.float32)

        # Compute homography from drone image to satellite tile
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

        # Warp drone image to satellite tile coordinates
        warped = cv2.warpPerspective(drone_img, M, (256, 256))

        # Save
        cv2.imwrite(str(output_dir / f'{frame:06d}_warped_to_sat.jpg'), warped)

        # Also save satellite tile for comparison
        sat_path = None
        for ext in ['.png', '.jpg']:
            p = Path(f'data/my_experiment/map/tiles/{tile_x}_{tile_y}_{zoom}{ext}')
            if p.exists():
                sat_path = p
                break
        if sat_path:
            sat_img = cv2.imread(str(sat_path))
            cv2.imwrite(str(output_dir / f'{frame:06d}_satellite.jpg'), sat_img)

        # Create side-by-side comparison
        if sat_path:
            comparison = np.hstack([warped, sat_img])
            cv2.imwrite(str(output_dir / f'{frame:06d}_comparison.jpg'), comparison)

        print(f"Frame {frame}: Heading={heading_deg:.1f}°, Alt={drone_alt:.1f}m")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
