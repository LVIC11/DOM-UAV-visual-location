#!/usr/bin/env python3
"""Extract images from ROS bag file."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge


def extract_images_from_bag(bag_path: str, output_dir: str, topic: str = "/camera/left/image_raw",
                            max_images: int = None, skip_images: int = 0, interval: int = 1,
                            rtk_topic: str = None, min_altitude: float = None):
    """Extract images from a ROS bag file.

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
        extract every N-th image, by default 1 (extract every image)
    rtk_topic : str, optional
        RTK GPS topic for altitude filtering and metadata, e.g. "/rtk"
    min_altitude : float, optional
        minimum altitude (meters) to keep an image, by default None (no filter)
    """
    import csv

    bag_path = Path(bag_path)
    output_dir = Path(output_dir)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found at {bag_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting images from {bag_path}")
    print(f"Topic: {topic}")
    print(f"Output directory: {output_dir}")
    print(f"Skipping first {skip_images} images")
    print(f"Extracting every {interval} image(s)")
    if rtk_topic:
        print(f"RTK topic: {rtk_topic}")
    if min_altitude is not None:
        print(f"Min altitude filter: {min_altitude}m")

    bridge = CvBridge()
    bag = rosbag.Bag(str(bag_path), "r")

    # Pre-load RTK messages into a list of (timestamp, lat, lon, alt) for fast lookup
    rtk_data = None
    if rtk_topic:
        rtk_data = []
        print("Loading RTK messages...")
        for _, msg, t in bag.read_messages(topics=[rtk_topic]):
            rtk_data.append((t.to_sec(), msg.latitude, msg.longitude, msg.altitude))
        print(f"  Loaded {len(rtk_data)} RTK messages")

    def get_rtk_at(timestamp_sec):
        """Find nearest RTK message by timestamp."""
        if not rtk_data:
            return None
        nearest = min(rtk_data, key=lambda r: abs(r[0] - timestamp_sec))
        return nearest

    # Metadata records for images that pass filter
    metadata = []

    count = 0
    extracted = 0
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        count += 1
        if count <= skip_images:
            continue

        if max_images and extracted >= max_images:
            break

        # Only extract every interval-th image
        if (count - skip_images) % interval != 0:
            continue

        ts_sec = t.to_sec()

        # Altitude filter
        if min_altitude is not None:
            rtk = get_rtk_at(ts_sec)
            if rtk is None:
                print(f"  WARNING: No RTK data for image at t={ts_sec:.3f}, skipping")
                continue
            alt = rtk[3]
            if alt < min_altitude:
                continue

        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        filename = f"{extracted:06d}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), cv_image)

        # Record metadata
        if rtk_data:
            rtk = get_rtk_at(ts_sec)
            if rtk:
                metadata.append({
                    'filename': filename,
                    'latitude': rtk[1],
                    'longitude': rtk[2],
                    'altitude': rtk[3],
                    'timestamp': ts_sec,
                })

        extracted += 1
        if extracted % 10 == 0:
            print(f"Extracted {extracted} images...")

    bag.close()

    # Write metadata.csv
    if metadata:
        csv_path = output_dir / "metadata.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'latitude', 'longitude', 'altitude', 'timestamp'])
            writer.writeheader()
            writer.writerows(metadata)
        print(f"Metadata saved to {csv_path}")

    skipped_msg = f" (altitude < {min_altitude}m filtered)" if min_altitude else ""
    print(f"Total images extracted: {extracted}{skipped_msg}")
    print(f"Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS bag file")
    parser.add_argument("bag_path", type=str, help="path to the ROS bag file")
    parser.add_argument("--output-dir", type=str, default="data/query/", help="output directory for extracted images")
    parser.add_argument("--topic", type=str, default="/camera/left/image_raw", help="topic to extract images from")
    parser.add_argument("--max-images", type=int, default=None, help="maximum number of images to extract")
    parser.add_argument("--skip-images", type=int, default=0, help="number of images to skip from the beginning")
    parser.add_argument("--interval", type=int, default=1, help="extract every N-th image")
    parser.add_argument("--rtk-topic", type=str, default=None, help="RTK GPS topic for altitude filtering and metadata")
    parser.add_argument("--min-altitude", type=float, default=None, help="minimum altitude (meters) filter")

    args = parser.parse_args()

    extract_images_from_bag(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        topic=args.topic,
        max_images=args.max_images,
        skip_images=args.skip_images,
        interval=args.interval,
        rtk_topic=args.rtk_topic,
        min_altitude=args.min_altitude,
    )


if __name__ == "__main__":
    main()
