#!/usr/bin/env python3
"""Compute heading from RTK GPS positions and save poses."""

import csv
import math
from pathlib import Path


def main():
    # Load metadata
    poses = []
    with open('data/my_experiment/query/metadata.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                'filename': row['filename'],
                'timestamp': float(row['timestamp']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'altitude': float(row['altitude']),
            })

    print(f"Loaded {len(poses)} poses")

    # Compute heading from consecutive positions
    headings = [0.0] * len(poses)
    for i in range(1, len(poses)):
        lat1, lon1 = poses[i-1]['latitude'], poses[i-1]['longitude']
        lat2, lon2 = poses[i]['latitude'], poses[i]['longitude']
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * math.cos(math.radians(lat1))
        dx = (lon2 - lon1) * lon_to_m
        dy = (lat2 - lat1) * lat_to_m
        headings[i] = math.degrees(math.atan2(dx, dy))

    # Smooth headings
    window = 3
    smoothed = []
    for i in range(len(poses)):
        start = max(0, i - window)
        end = min(len(poses), i + window + 1)
        sins = [math.sin(math.radians(h)) for h in headings[start:end]]
        coss = [math.cos(math.radians(h)) for h in headings[start:end]]
        avg_sin = sum(sins) / len(sins)
        avg_cos = sum(coss) / len(coss)
        smoothed.append(math.degrees(math.atan2(avg_sin, avg_cos)))

    # Add headings to poses
    for i, pose in enumerate(poses):
        pose['heading'] = smoothed[i]

    # Save
    output_path = Path('data/my_experiment/rtk_poses_with_heading.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'timestamp', 'latitude', 'longitude', 'altitude', 'heading'])
        writer.writeheader()
        writer.writerows(poses)

    print(f"Saved to {output_path}")

    # Print first 10
    print("\nFirst 10 poses:")
    for p in poses[:10]:
        print(f"  {p['filename']}: GPS=({p['latitude']:.6f}, {p['longitude']:.6f}), "
              f"Alt={p['altitude']:.1f}m, Heading={p['heading']:.1f}°")


if __name__ == "__main__":
    main()
