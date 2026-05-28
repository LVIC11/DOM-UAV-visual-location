#!/usr/bin/env python3
"""Warp satellite images to drone viewpoint and save for inspection."""

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

    zoom = 18
    output_dir = Path('data/my_experiment/warped')
    output_dir.mkdir(parents=True, exist_ok=True)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

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

        # Load satellite
        sat_path = None
        for ext in ['.png', '.jpg']:
            p = Path(f'data/my_experiment/map/tiles/{tile_x}_{tile_y}_{zoom}{ext}')
            if p.exists():
                sat_path = p
                break
        if sat_path is None:
            print(f"Frame {frame}: satellite tile not found")
            continue

        sat_img = cv2.imread(str(sat_path))
        if sat_img is None:
            continue

        # Tile corners in meters relative to drone
        tile_north, tile_west = get_lat_long_from_tile_xy(tile_x, tile_y, zoom)
        tile_south, tile_east = get_lat_long_from_tile_xy(tile_x + 1, tile_y + 1, zoom)
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * math.cos(math.radians(drone_lat))

        tile_corners_m = np.array([
            [(tile_west - drone_lon) * lon_to_m, (tile_north - drone_lat) * lat_to_m],
            [(tile_east - drone_lon) * lon_to_m, (tile_north - drone_lat) * lat_to_m],
            [(tile_east - drone_lon) * lon_to_m, (tile_south - drone_lat) * lat_to_m],
            [(tile_west - drone_lon) * lon_to_m, (tile_south - drone_lat) * lat_to_m],
        ], dtype=np.float32)

        # Rotation: heading then pitch
        R_heading = Rotation.from_euler('z', -heading_deg, degrees=True).as_matrix()
        R_pitch = Rotation.from_euler('y', pitch_deg, degrees=True).as_matrix()
        R_cam = R_heading @ R_pitch

        # Homography
        H = np.column_stack([R_cam[:, 0], R_cam[:, 1], -drone_alt * R_cam[:, 2]])
        H = K @ H

        # Project to image
        tile_h = np.column_stack([tile_corners_m, np.ones(4)])
        img_corners = (H @ tile_h.T).T
        img_corners[:, :2] /= img_corners[:, 2:3]

        # Warp
        sat_h, sat_w = sat_img.shape[:2]
        sat_corners = np.array([[0, 0], [sat_w-1, 0], [sat_w-1, sat_h-1], [0, sat_h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(sat_corners, img_corners[:, :2].astype(np.float32))
        warped = cv2.warpPerspective(sat_img, M, (1280, 720))

        # Save warped
        cv2.imwrite(str(output_dir / f'{frame:06d}_warped.jpg'), warped)

        # Save drone for comparison
        drone_path = Path(f'data/my_experiment/query/{frame:06d}.jpg')
        if drone_path.exists():
            cv2.imwrite(str(output_dir / f'{frame:06d}_drone.jpg'), cv2.imread(str(drone_path)))

        print(f"Frame {frame}: GPS=({drone_lat:.6f}, {drone_lon:.6f}), Alt={drone_alt:.1f}m, Heading={heading_deg:.1f}°")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
