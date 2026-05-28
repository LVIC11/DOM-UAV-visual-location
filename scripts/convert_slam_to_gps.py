#!/usr/bin/env python3
"""Convert SLAM poses to GPS coordinates using RTK reference."""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_slam_poses(filepath):
    """Load SLAM poses: timestamp x y z qw qx qy qz"""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                poses.append({
                    'timestamp': float(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3]),
                    'qw': float(parts[4]),
                    'qx': float(parts[5]),
                    'qy': float(parts[6]),
                    'qz': float(parts[7]),
                })
    return poses


def load_rtk_poses(filepath):
    """Load RTK poses from metadata.csv: filename,latitude,longitude,altitude,timestamp"""
    poses = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            poses[i] = {
                'timestamp': float(row['timestamp']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'altitude': float(row['altitude']),
                'heading': 0,  # will compute from consecutive positions
            }
    # Compute heading from consecutive positions with smoothing
    headings = [0] * len(poses)
    for i in range(1, len(poses)):
        lat1, lon1 = poses[i-1]['latitude'], poses[i-1]['longitude']
        lat2, lon2 = poses[i]['latitude'], poses[i]['longitude']
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * math.cos(math.radians(lat1))
        dx = (lon2 - lon1) * lon_to_m
        dy = (lat2 - lat1) * lat_to_m
        headings[i] = math.degrees(math.atan2(dx, dy))

    # Smooth headings using moving average
    window = 3
    for i in range(len(poses)):
        start = max(0, i - window)
        end = min(len(poses), i + window + 1)
        # Handle angle wrapping
        sins = [math.sin(math.radians(h)) for h in headings[start:end]]
        coss = [math.cos(math.radians(h)) for h in headings[start:end]]
        avg_sin = sum(sins) / len(sins)
        avg_cos = sum(coss) / len(coss)
        poses[i]['heading'] = math.degrees(math.atan2(avg_sin, avg_cos))

    return poses


def gps_to_ecef(lat, lon, alt):
    """Convert GPS to ECEF coordinates."""
    a = 6378137.0  # WGS84 semi-major axis
    e2 = 0.00669437999014  # WGS84 eccentricity squared

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)

    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)

    return np.array([x, y, z])


def ecef_to_gps(ecef):
    """Convert ECEF to GPS coordinates (simplified)."""
    a = 6378137.0
    e2 = 0.00669437999014

    x, y, z = ecef
    lon = math.atan2(y, x)

    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - e2))

    for _ in range(10):
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        lat = math.atan2(z + e2 * N * math.sin(lat), p)

    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    alt = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), alt


def main():
    slam_poses = load_slam_poses('/home/sy/sy/AIRSLAM/result/a_low/winter/150/BA_loop.txt')
    rtk_poses = load_rtk_poses('data/my_experiment/query/metadata.csv')

    print(f"Loaded {len(slam_poses)} SLAM poses")
    print(f"Loaded {len(rtk_poses)} RTK poses")

    # Find matching timestamps between SLAM and RTK
    # RTK timestamps correspond to frame numbers
    # SLAM timestamps are continuous

    # Get RTK timestamps in order
    rtk_timestamps = []
    for frame in sorted(rtk_poses.keys()):
        rtk_timestamps.append(rtk_poses[frame]['timestamp'])

    print(f"RTK time range: {rtk_timestamps[0]:.3f} to {rtk_timestamps[-1]:.3f}")
    print(f"SLAM time range: {slam_poses[0]['timestamp']:.3f} to {slam_poses[-1]['timestamp']:.3f}")

    # Find closest SLAM pose for each RTK frame
    matches = []
    for frame in sorted(rtk_poses.keys()):
        rtk_ts = rtk_poses[frame]['timestamp']

        # Find closest SLAM pose
        min_diff = float('inf')
        best_slam = None
        for slam_pose in slam_poses:
            diff = abs(slam_pose['timestamp'] - rtk_ts)
            if diff < min_diff:
                min_diff = diff
                best_slam = slam_pose

        if best_slam and min_diff < 1.0:  # within 1 second
            matches.append({
                'frame': frame,
                'slam': best_slam,
                'rtk': rtk_poses[frame],
                'time_diff': min_diff,
            })

    print(f"\nFound {len(matches)} timestamp matches")

    if len(matches) < 3:
        print("Not enough matches to compute transformation")
        return

    # Compute transformation: SLAM -> GPS (similarity transform with scale)
    # GPS_pos = s * R @ SLAM_pos + t

    # Stack all matched SLAM positions and GPS positions
    slam_positions = []
    gps_positions = []

    # Use first RTK pose as reference for local meters
    ref_lat = rtk_poses[0]['latitude']
    ref_lon = rtk_poses[0]['longitude']
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * math.cos(math.radians(ref_lat))

    for m in matches:
        slam_pos = np.array([m['slam']['x'], m['slam']['y'], m['slam']['z']])

        gps_x = (m['rtk']['longitude'] - ref_lon) * lon_to_m
        gps_y = (m['rtk']['latitude'] - ref_lat) * lat_to_m
        gps_z = m['rtk']['altitude'] - rtk_poses[0]['altitude']

        slam_positions.append(slam_pos)
        gps_positions.append(np.array([gps_x, gps_y, gps_z]))

    slam_positions = np.array(slam_positions)
    gps_positions = np.array(gps_positions)

    # Compute centroids
    slam_centroid = slam_positions.mean(axis=0)
    gps_centroid = gps_positions.mean(axis=0)

    # Center the points
    slam_centered = slam_positions - slam_centroid
    gps_centered = gps_positions - gps_centroid

    # Compute scale: s = ||gps_centered|| / ||slam_centered||
    # Using the ratio of norms
    slam_norm = np.linalg.norm(slam_centered, axis=1)
    gps_norm = np.linalg.norm(gps_centered, axis=1)
    s = np.mean(gps_norm) / np.mean(slam_norm)

    # Compute rotation using SVD (on normalized points)
    H = slam_centered.T @ gps_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = gps_centroid - s * R @ slam_centroid

    print(f"\nTransformation (SLAM -> local meters):")
    print(f"  Scale: {s:.4f}")
    print(f"  Rotation:\n{R}")
    print(f"  Translation: {t}")

    # Compute residuals
    residuals = []
    for i in range(len(matches)):
        pred = s * R @ slam_positions[i] + t
        error = np.linalg.norm(pred - gps_positions[i])
        residuals.append(error)

    print(f"\nResiduals (meters):")
    print(f"  Mean: {np.mean(residuals):.2f}")
    print(f"  Max: {np.max(residuals):.2f}")
    print(f"  Min: {np.min(residuals):.2f}")

    # Convert all SLAM poses to GPS
    ref_lat = rtk_poses[0]['latitude']
    ref_lon = rtk_poses[0]['longitude']
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * math.cos(math.radians(ref_lat))

    output_poses = []
    for slam_pose in slam_poses:
        slam_pos = np.array([slam_pose['x'], slam_pose['y'], slam_pose['z']])
        local_pos = s * R @ slam_pos + t

        # Convert local meters to GPS
        lat = ref_lat + local_pos[1] / lat_to_m
        lon = ref_lon + local_pos[0] / lon_to_m
        alt = local_pos[2] + rtk_poses[0]['altitude']

        # Compute heading from quaternion
        q = [slam_pose['qx'], slam_pose['qy'], slam_pose['qz'], slam_pose['qw']]
        rot = Rotation.from_quat(q)
        euler = rot.as_euler('xyz', degrees=True)
        heading = euler[2]  # yaw

        output_poses.append({
            'timestamp': slam_pose['timestamp'],
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'heading': heading,
        })

    # Save converted poses
    output_path = Path('data/my_experiment/slam_gps_poses.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'latitude', 'longitude', 'altitude', 'heading'])
        writer.writeheader()
        writer.writerows(output_poses)

    print(f"\nSaved {len(output_poses)} poses to {output_path}")

    # Print first few
    print("\nFirst 5 poses:")
    for p in output_poses[:5]:
        print(f"  ts={p['timestamp']:.3f}, GPS=({p['latitude']:.6f}, {p['longitude']:.6f}), "
              f"Alt={p['altitude']:.1f}m, Heading={p['heading']:.1f}°")


if __name__ == "__main__":
    main()
