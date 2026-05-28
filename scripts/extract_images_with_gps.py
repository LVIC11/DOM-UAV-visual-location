#!/usr/bin/env python3
"""Extract images from ROS bag file with GPS coordinates."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge


def extract_images_with_gps(bag_path: str, output_dir: str, topic: str = "/camera/left/image_raw",
                            max_images: int = None, skip_images: int = 0, interval: int = 1):
    """Extract images from a ROS bag file with GPS coordinates.

    Parameters
    ----------
    bag_path : str
        path to the ROS bag file
    output_dir : str
        path to the output directory
    topic : str
        topic to extract images from
    max_images : int, optional
        maximum number of images to extract, by default None (extract all)
    skip_images : int, optional
        number of images to skip from the beginning, by default 0
    interval : int, optional
        extract every N-th image, by default 1
    """
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found at {bag_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting images from {bag_path}")
    print(f"Topic: {topic}")
    print(f"Output directory: {output_dir}")

    bridge = CvBridge()
    bag = rosbag.Bag(str(bag_path), "r")

    # First pass: collect all RTK data
    print("Collecting RTK data...")
    rtk_data = []
    for topic_name, msg, t in bag.read_messages(topics=['/rtk']):
        rtk_data.append((t.to_sec(), msg.latitude, msg.longitude, msg.altitude))
    print(f"Collected {len(rtk_data)} RTK messages")

    # Second pass: extract images with interpolated GPS
    print("Extracting images...")
    count = 0
    extracted = 0
    metadata = []

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        count += 1
        if count <= skip_images:
            continue

        if max_images and extracted >= max_images:
            break

        # Only extract every interval-th image
        if (count - skip_images) % interval != 0:
            continue

        # Find the closest RTK data point
        t_sec = t.to_sec()
        closest_rtk = None
        min_diff = float('inf')
        for rtk_time, lat, lon, alt in rtk_data:
            diff = abs(rtk_time - t_sec)
            if diff < min_diff:
                min_diff = diff
                closest_rtk = (lat, lon, alt)

        # Convert image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        filename = f"{extracted:06d}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), cv_image)

        # Store metadata
        if closest_rtk:
            metadata.append({
                'filename': filename,
                'latitude': closest_rtk[0],
                'longitude': closest_rtk[1],
                'altitude': closest_rtk[2],
                'timestamp': t_sec,
            })

        extracted += 1
        if extracted % 10 == 0:
            print(f"Extracted {extracted} images...")

    bag.close()

    # Save metadata CSV
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'latitude', 'longitude', 'altitude', 'timestamp'])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Total images extracted: {extracted}")
    print(f"Metadata saved to {csv_path}")
    print(f"Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS bag file with GPS")
    parser.add_argument("bag_path", type=str, help="path to the ROS bag file")
    parser.add_argument("--output-dir", type=str, default="data/query/", help="output directory")
    parser.add_argument("--topic", type=str, default="/camera/left/image_raw", help="topic to extract")
    parser.add_argument("--max-images", type=int, default=None, help="maximum number of images")
    parser.add_argument("--skip-images", type=int, default=0, help="skip first N images")
    parser.add_argument("--interval", type=int, default=1, help="extract every N-th image")

    args = parser.parse_args()

    extract_images_with_gps(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        topic=args.topic,
        max_images=args.max_images,
        skip_images=args.skip_images,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
