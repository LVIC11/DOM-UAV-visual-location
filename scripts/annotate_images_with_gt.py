#!/usr/bin/env python3
"""Annotate drone images from a rosbag with GT (GNSS) points.

For a nadir camera, the drone's GPS position projects to the image center.
Draws a green crosshair at the GT point + GPS text overlay on each image.

Usage:
    python scripts/annotate_images_with_gt.py \
        --bag /path/to/TD_03.bag \
        --output data/output/annotated \
        --max-images 50 \
        --rotate-cw
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

logger = logging.getLogger("annotate_gt")


def annotate_images(
    bag_path: Path,
    output_path: Path,
    image_topic: str = "/camera_array/cam0/image_raw/compressed",
    gnss_topic: str = "/gnss/data",
    max_images: Optional[int] = None,
    skip_frames: int = 1,
    rotate_cw: bool = False,
    realtime_factor: float = 0,
):
    """Annotate drone images from a rosbag with GT points.

    Parameters
    ----------
    bag_path : Path
        Path to the ROS bag file.
    output_path : Path
        Directory to save annotated images.
    image_topic : str
        ROS topic for compressed images.
    gnss_topic : str
        ROS topic for GNSS/GPS messages.
    max_images : int, optional
        Maximum number of images to process.
    skip_frames : int
        Process every N-th image frame.
    rotate_cw : bool
        Rotate images 90 degrees clockwise.
    realtime_factor : float
        Playback speed multiplier (0 = as fast as possible).
    """
    output_path.mkdir(parents=True, exist_ok=True)

    with AnyReader([bag_path]) as reader:
        # --- Pre-load all GNSS data ---
        gnss_conn = next(c for c in reader.connections if c.topic == gnss_topic)
        gnss_list = []
        for conn, ts, raw in reader.messages(connections=[gnss_conn]):
            msg = reader.deserialize(raw, conn.msgtype)
            gnss_list.append((ts / 1e9, msg.latitude, msg.longitude, msg.altitude))
        logger.info(f"Loaded {len(gnss_list)} GNSS messages")

        img_conn = next(c for c in reader.connections if c.topic == image_topic)

        # Timing
        bag_start_ts = None
        wall_start_time = None
        frame_count = 0
        saved_count = 0

        for conn, ts, raw in reader.messages(connections=[img_conn]):
            frame_count += 1

            if (frame_count - 1) % skip_frames != 0:
                continue

            msg = reader.deserialize(raw, conn.msgtype)
            img_ts = ts / 1e9

            # Init timing
            if bag_start_ts is None:
                bag_start_ts = img_ts
                wall_start_time = time.time()

            # Real-time playback
            if realtime_factor > 0:
                expected = (img_ts - bag_start_ts) / realtime_factor
                actual = time.time() - wall_start_time
                if expected > actual:
                    time.sleep(expected - actual)

            # Find nearest GNSS fix
            nearest = min(gnss_list, key=lambda g: abs(g[0] - img_ts))
            dt = abs(nearest[0] - img_ts)
            if dt > 1.0:
                continue
            lat, lon, alt = nearest[1], nearest[2], nearest[3]

            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if rotate_cw:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            h, w = bgr.shape[:2]

            # GT point = image center (for nadir camera)
            cx, cy = w // 2, h // 2

            # --- Draw GT crosshair ---
            cross_size = max(20, min(w, h) // 30)
            color_gt = (0, 255, 0)  # green
            cv2.drawMarker(bgr, (cx, cy), color_gt, cv2.MARKER_CROSS,
                           cross_size, 2, cv2.LINE_AA)
            cv2.circle(bgr, (cx, cy), cross_size // 2, color_gt, 2, cv2.LINE_AA)

            # --- Draw grid lines from center ---
            grid_color = (0, 255, 0)
            dash_len = 8
            for x in range(0, w, 80):
                cv2.line(bgr, (x, cy - dash_len), (x, cy + dash_len), grid_color, 1, cv2.LINE_AA)
            for y in range(0, h, 80):
                cv2.line(bgr, (cx - dash_len, y), (cx + dash_len, y), grid_color, 1, cv2.LINE_AA)

            # --- Scale bar (bottom-right corner) ---
            # Compute ground resolution at nadir
            # GSD = altitude * pixel_size / focal_length
            # pixel_size ≈ sensor_width / resolution_width
            # For a typical drone camera with hfov=82.9°, focal_length=4.5mm:
            # GSD ≈ altitude * (sensor_width / resolution_width) / focal_length
            # sensor_width ≈ 2 * focal_length * tan(hfov/2) = 2*4.5*tan(41.45°) = 7.96mm
            # GSD ≈ alt * 7.96mm/3040 / 4.5mm ≈ alt * 0.000582
            # At 195m: GSD ≈ 0.113m/px → 100px ≈ 11.3m
            fov_rad = 82.9 * np.pi / 180
            sensor_w_mm = 2 * 4.5 * np.tan(fov_rad / 2)
            gsd = alt * sensor_w_mm / 3040 / 4.5  # meters per pixel
            bar_len_px = int(50 / gsd)  # 50 meter scale bar
            bar_len_px = max(50, min(bar_len_px, w // 3))
            bar_x0 = w - bar_len_px - 30
            bar_y = h - 50
            cv2.line(bgr, (bar_x0, bar_y), (bar_x0 + bar_len_px, bar_y),
                     (0, 255, 0), 4, cv2.LINE_AA)
            bar_label = f"{bar_len_px * gsd:.0f}m"
            cv2.putText(bgr, bar_label, (bar_x0, bar_y - 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            # --- Text overlay ---
            text_lines = [
                f"GT: {lat:.7f}, {lon:.7f}",
                f"Alt: {alt:.1f}m  GSD: {gsd:.3f}m/px",
                f"Frame: {frame_count}  t={img_ts - bag_start_ts:.1f}s",
            ]
            for i, t in enumerate(text_lines):
                y = 35 + i * 30
                cv2.putText(bgr, t, (15, y), cv2.FONT_HERSHEY_DUPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(bgr, t, (15, y), cv2.FONT_HERSHEY_DUPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Save
            out_name = f"frame_{frame_count:06d}_gt.jpg"
            cv2.imwrite(str(output_path / out_name), bgr)
            saved_count += 1

            if saved_count % 10 == 0:
                logger.info(f"Annotated {saved_count} images...")

            if max_images and saved_count >= max_images:
                break

    logger.info(f"Done. Annotated {saved_count} images, saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate drone images from rosbag with GT (GNSS) points"
    )
    parser.add_argument("--bag", type=str, required=True, help="Path to ROS bag file")
    parser.add_argument("--output", type=str, default="data/output/annotated",
                        help="Output directory for annotated images")
    parser.add_argument("--image-topic", type=str,
                        default="/camera_array/cam0/image_raw/compressed")
    parser.add_argument("--gnss-topic", type=str, default="/gnss/data")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--rotate-cw", action="store_true",
                        help="Rotate images 90 degrees clockwise")
    parser.add_argument("--realtime-factor", type=float, default=0,
                        help="Playback speed (0=fast as possible)")

    args = parser.parse_args()

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, datefmt="%H:%M:%S")

    annotate_images(
        bag_path=Path(args.bag),
        output_path=Path(args.output),
        image_topic=args.image_topic,
        gnss_topic=args.gnss_topic,
        max_images=args.max_images,
        skip_frames=args.skip_frames,
        rotate_cw=args.rotate_cw,
        realtime_factor=args.realtime_factor,
    )


if __name__ == "__main__":
    main()
