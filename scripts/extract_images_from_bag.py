#!/usr/bin/env python3
"""Extract images from ROS bag file."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge


def extract_images_from_bag(bag_path: str, output_dir: str, topic: str = "/camera/left/image_raw", max_images: int = None, skip_images: int = 0, interval: int = 1):
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
    """
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found at {bag_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting images from {bag_path}")
    print(f"Topic: {topic}")
    print(f"Output directory: {output_dir}")
    print(f"Skipping first {skip_images} images")
    print(f"Extracting every {interval} images")

    bridge = CvBridge()
    bag = rosbag.Bag(str(bag_path), "r")

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

        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        filename = output_dir / f"{extracted:06d}.jpg"
        cv2.imwrite(str(filename), cv_image)

        extracted += 1
        if extracted % 10 == 0:
            print(f"Extracted {extracted} images...")

    bag.close()
    print(f"Total images extracted: {extracted}")
    print(f"Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS bag file")
    parser.add_argument("bag_path", type=str, help="path to the ROS bag file")
    parser.add_argument("--output-dir", type=str, default="data/query/", help="output directory for extracted images")
    parser.add_argument("--topic", type=str, default="/camera/left/image_raw", help="topic to extract images from")
    parser.add_argument("--max-images", type=int, default=None, help="maximum number of images to extract")
    parser.add_argument("--skip-images", type=int, default=0, help="number of images to skip from the beginning")
    parser.add_argument("--interval", type=int, default=1, help="extract every N-th image")

    args = parser.parse_args()

    extract_images_from_bag(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        topic=args.topic,
        max_images=args.max_images,
        skip_images=args.skip_images,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
