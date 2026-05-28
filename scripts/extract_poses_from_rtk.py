#!/usr/bin/env python3
"""Extract pose (lat, lon, alt, heading) for each image from RTK trajectory."""

import argparse
import csv
import math
from pathlib import Path

import rosbag


def compute_heading(lat1, lon1, lat2, lon2):
    """Compute heading from two GPS points (degrees)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return math.degrees(math.atan2(dlon, dlat))


def interpolate_rtk(rtk_data, target_time):
    """Find closest RTK data point to target time."""
    closest = None
    min_diff = float('inf')
    for t, lat, lon, alt in rtk_data:
        diff = abs(t - target_time)
        if diff < min_diff:
            min_diff = diff
            closest = (lat, lon, alt)
    return closest


def extract_poses(bag_path, image_topic, rtk_topic, output_csv, skip_images, interval, max_images):
    """Extract poses for each image timestamp."""
    bag = rosbag.Bag(bag_path, "r")

    # Collect all RTK data
    print("Collecting RTK data...")
    rtk_data = []
    for topic, msg, t in bag.read_messages(topics=[rtk_topic]):
        rtk_data.append((t.to_sec(), msg.latitude, msg.longitude, msg.altitude))
    print(f"Collected {len(rtk_data)} RTK messages")

    # Compute headings from RTK trajectory
    headings = []
    for i in range(len(rtk_data)):
        if i == 0 and len(rtk_data) > 1:
            h = compute_heading(rtk_data[i][1], rtk_data[i][2],
                              rtk_data[i+1][1], rtk_data[i+1][2])
        elif i == len(rtk_data) - 1:
            h = compute_heading(rtk_data[i-1][1], rtk_data[i-1][2],
                              rtk_data[i][1], rtk_data[i][2])
        else:
            h1 = compute_heading(rtk_data[i-1][1], rtk_data[i-1][2],
                               rtk_data[i][1], rtk_data[i][2])
            h2 = compute_heading(rtk_data[i][1], rtk_data[i][2],
                               rtk_data[i+1][1], rtk_data[i+1][2])
            # Average angles properly
            h = math.degrees(math.atan2(
                math.sin(math.radians(h1)) + math.sin(math.radians(h2)),
                math.cos(math.radians(h1)) + math.cos(math.radians(h2))
            ))
        headings.append(h)

    # Extract images with interpolated poses
    print("Extracting images with poses...")
    count = 0
    extracted = 0
    metadata = []

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        count += 1
        if count <= skip_images:
            continue
        if max_images and extracted >= max_images:
            break
        if (count - skip_images) % interval != 0:
            continue

        t_sec = t.to_sec()

        # Find closest RTK and heading
        closest_idx = 0
        min_diff = float('inf')
        for i, (rtk_t, lat, lon, alt) in enumerate(rtk_data):
            diff = abs(rtk_t - t_sec)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        lat, lon, alt = rtk_data[closest_idx][1], rtk_data[closest_idx][2], rtk_data[closest_idx][3]
        heading = headings[closest_idx]

        metadata.append({
            'frame': extracted,
            'timestamp': t_sec,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'heading': heading,
        })

        extracted += 1
        if extracted % 10 == 0:
            print(f"Processed {extracted} frames...")

    bag.close()

    # Save metadata
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'latitude', 'longitude', 'altitude', 'heading'])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nSaved {len(metadata)} poses to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract poses from RTK trajectory")
    parser.add_argument("bag_path", type=str, help="path to ROS bag file")
    parser.add_argument("--image-topic", type=str, default="/camera/left/image_raw")
    parser.add_argument("--rtk-topic", type=str, default="/rtk")
    parser.add_argument("--output-csv", type=str, default="data/my_experiment/rtk_poses.csv")
    parser.add_argument("--skip-images", type=int, default=1500)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=50)

    args = parser.parse_args()
    extract_poses(
        args.bag_path, args.image_topic, args.rtk_topic,
        args.output_csv, args.skip_images, args.interval, args.max_images
    )
