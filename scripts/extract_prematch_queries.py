#!/usr/bin/env python3
"""Sample raw left images on native header time for independent SVL prematching.

Run with the system ROS Python. PNG pixels are checked against the original
mono8 message. RTK coordinates are metadata for later evaluation only; they
never select images. This creates a new directory and never overwrites one.
"""

import argparse
from bisect import bisect_left
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decimal_stamp(ns):
    return f"{ns // 1000000000}.{ns % 1000000000:09d}"


def extract(bag_path, output, period, topic):
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Sampling period must be positive and finite")
    bag_path = bag_path.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    stat = bag_path.stat()
    manifest = dict(
        status="running", bag=str(bag_path), bag_size_bytes=stat.st_size,
        bag_mtime_ns=stat.st_mtime_ns,
        bag_identity_scope="path, size and mtime; no whole-bag digest",
        image_topic=topic, sampling_period_s=period,
        selection="first image at/after each native-header grid point, starting at first image",
        timestamp_semantics="camera_header", native_resolution=True,
        encoding="mono8", image_format="lossless PNG", source_images_rotated=False,
        rtk_xy_used_for_selection=False, altitude_filter=False,
        rtk_scope="nearest native-header reference metadata only; not used for image selection or matching",
        map_keyframe_association="not performed; these are uniformly sampled raw images",
        frames=[], source_sha256=sha(__file__),
    )
    manifest_path = output / "extraction_manifest.json"
    bridge = CvBridge()
    rows = []
    try:
        with rosbag.Bag(str(bag_path)) as bag:
            references = sorted(
                (msg.header.stamp.to_nsec(), float(msg.latitude), float(msg.longitude), float(msg.altitude))
                for _, msg, _ in bag.read_messages(topics=["/rtk"])
                if msg.header.stamp.to_nsec() > 0
            )
            rt = [r[0] for r in references]
            first_ns = previous_ns = None
            period_ns = round(period * 1e9)
            sample_index = 0
            source_count = 0
            for _, msg, record in bag.read_messages(topics=[topic]):
                source_count += 1
                stamp = msg.header.stamp.to_nsec()
                if stamp <= 0 or (previous_ns is not None and stamp <= previous_ns):
                    raise ValueError("Camera header time is missing, duplicate or nonmonotonic")
                previous_ns = stamp
                if first_ns is None:
                    first_ns = stamp
                if stamp < first_ns + sample_index * period_ns:
                    continue
                sample_index += 1
                while first_ns + sample_index * period_ns <= stamp:
                    sample_index += 1
                if msg.encoding != "mono8":
                    raise ValueError("Expected native mono8; no implicit pixel conversion")
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if image.dtype != np.uint8 or image.shape != (msg.height, msg.width):
                    raise ValueError("Unexpected native image layout")
                filename = f"{len(rows):06d}.png"
                path = output / filename
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError("Failed to save image")
                if not np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), image):
                    raise ValueError("PNG/raw pixel mismatch")
                row = dict(filename=filename, timestamp=decimal_stamp(stamp),
                    header_timestamp_ns=stamp, bag_record_timestamp=decimal_stamp(record.to_nsec()),
                    bag_record_timestamp_ns=record.to_nsec(), source_image_index=source_count - 1,
                    camera_sequence=msg.header.seq)
                if references:
                    i = bisect_left(rt, stamp)
                    candidates = [j for j in (i - 1, i) if 0 <= j < len(rt)]
                    ref = references[min(candidates, key=lambda j: abs(rt[j] - stamp))]
                    row.update(latitude=ref[1], longitude=ref[2], altitude=ref[3],
                        rtk_header_timestamp=decimal_stamp(ref[0]), rtk_time_offset_s=(ref[0]-stamp)/1e9)
                rows.append(row)
                manifest["frames"].append(dict(row, width=msg.width, height=msg.height,
                    image_sha256=sha(path), raw_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    png_raw_pixels_equal=True))
                print("EXTRACT",len(rows),filename,row["timestamp"],flush=True)
            if not rows:
                raise ValueError("No images extracted")
            manifest.update(source_image_count=source_count, source_rtk_count=len(references),
                source_first_header_ns=first_ns, source_last_header_ns=previous_ns)
        if bag_path.stat().st_size != stat.st_size or bag_path.stat().st_mtime_ns != stat.st_mtime_ns:
            raise ValueError("Bag identity changed during extraction")
        with (output / "metadata.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        shutil.copy2(__file__,output / "extract_prematch_queries.py")
        manifest.update(status="complete", extracted_images=len(rows),
            metadata_sha256=sha(output / "metadata.csv"),
            max_record_header_offset_s=max(abs(r["bag_record_timestamp_ns"]-r["header_timestamp_ns"])/1e9 for r in rows))
    except BaseException as error:
        manifest.update(status="failed",error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:manifest[k] for k in ("status","extracted_images","source_image_count","max_record_header_offset_s")}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--period",type=float,default=4.0)
    parser.add_argument("--topic",default="/camera/left/image_raw")
    args=parser.parse_args()
    extract(args.bag,args.output,args.period,args.topic)
