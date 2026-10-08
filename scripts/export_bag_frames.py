#!/usr/bin/env python3
"""Export left-camera images from a rosbag, named by exact timestamp.

Usage:
  python export_bag_frames.py \
    --bag /path/to/bag.bag \
    --topic /camera/left/image_raw \
    --out /path/to/images

Output files:  /path/to/images/1703738785.891234.jpg
The timestamp string is used as the key for offline matching lookup.
"""

import argparse
import os
import sys

import cv2
import rosbag
from cv_bridge import CvBridge
from genpy import Time as RosTime


def main():
    parser = argparse.ArgumentParser(description="Export bag images with timestamps")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", default="/camera/left/image_raw")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="Start time offset (s)")
    parser.add_argument("--duration", type=float, default=-1, help="Duration to extract (s), -1=all")
    parser.add_argument("--stride", type=int, default=1, help="Extract every Nth frame (default: 1=all)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bridge = CvBridge()
    bag = rosbag.Bag(args.bag, 'r')

    t0 = bag.get_start_time() + args.start
    t1 = t0 + args.duration if args.duration > 0 else bag.get_end_time()

    frame_idx = 0
    exported = 0
    for topic, msg, t in bag.read_messages(topics=[args.topic],
                                            start_time=RosTime.from_sec(t0),
                                            end_time=RosTime.from_sec(t1)):
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ts = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else t.to_sec()
        fname = f"{ts:.6f}.jpg"
        cv2.imwrite(os.path.join(args.out, fname), img)
        exported += 1
        if exported % 100 == 0:
            print(f"  Exported {exported} frames...", file=sys.stderr)

    bag.close()
    print(f"Done: {exported} frames exported (stride={args.stride}) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
