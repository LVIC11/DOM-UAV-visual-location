#!/usr/bin/env python3
"""Associate JPEG queries by exact bag record time, preserving native image headers."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def associate(folder, bagpath, output):
    import rosbag
    import rospy
    from cv_bridge import CvBridge

    if output.exists():
        raise FileExistsError(str(output))
    with (folder / "metadata.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {round(float(r["timestamp"]) * 1e6): r for r in rows}
    if not rows or len(lookup) != len(rows):
        raise ValueError("Missing/ambiguous record timestamps")
    before = bagpath.stat()
    bridge = CvBridge()
    hashes = {
        str(folder / "metadata.csv"): sha(folder / "metadata.csv"),
        str(Path(__file__).resolve()): sha(__file__),
    }
    found = {}
    start, end = min(float(r["timestamp"]) for r in rows), max(
        float(r["timestamp"]) for r in rows
    )
    with rosbag.Bag(str(bagpath)) as bag:
        for _, message, record in bag.read_messages(
            topics=["/camera/left/image_raw"],
            start_time=rospy.Time.from_sec(start - 0.1),
            end_time=rospy.Time.from_sec(end + 0.1),
        ):
            stamp = record.to_sec()
            row = lookup.get(round(stamp * 1e6))
            if row is None:
                continue
            filename = row["filename"]
            if filename in found or abs(stamp - float(row["timestamp"])) > 1e-6:
                raise ValueError("Non-unique exact bag record association")
            original = cv2.imread(str(folder / filename), cv2.IMREAD_GRAYSCALE)
            raw = bridge.imgmsg_to_cv2(message, desired_encoding="mono8")
            if original is None or raw.shape != original.shape:
                raise ValueError("Original JPEG dimensions differ from bag image")
            delta = abs(original.astype(float) - raw)
            mean, p95 = float(delta.mean()), float(np.percentile(delta, 95))
            if mean > 4 or p95 > 12:
                raise ValueError(
                    "Query may be geometrically altered/not the raw source frame: "
                    + filename
                )
            hashes[str(folder / filename)] = sha(folder / filename)
            found[filename] = dict(
                frame=Path(filename).stem,
                filename=filename,
                metadata_timestamp=float(row["timestamp"]),
                bag_record_timestamp=stamp,
                image_header_timestamp=message.header.stamp.to_sec(),
                image_sequence=int(message.header.seq),
                metadata_minus_header_s=stamp - message.header.stamp.to_sec(),
                mean_jpeg_raw_abs_pixel_error=mean,
                p95_abs_pixel_error=p95,
                image_sha256=hashes[str(folder / filename)],
                raw_array_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            )
    after = bagpath.stat()
    if (
        set(found) != {r["filename"] for r in rows}
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("Bag changed or some exact source records were not found")
    if any(sha(p) != h for p, h in hashes.items()):
        raise ValueError("Original inputs changed during association")
    result = dict(
        status="complete",
        frames=sorted(found.values(), key=lambda r: r["image_header_timestamp"]),
        bag=str(bagpath),
        bag_size_bytes=before.st_size,
        bag_mtime_ns=before.st_mtime_ns,
        bag_identity_scope="stat + exact selected record/header/content associations, not whole-bag SHA256",
        association_method="exact bag record timestamp plus raw/JPEG content check; not nearest SLAM keyframe",
        input_sha256=hashes,
    )
    output.mkdir(parents=True)
    (output / "associations.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "metadata_header_time.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "timestamp", "bag_record_timestamp"])
        for row in result["frames"]:
            writer.writerow(
                [
                    row["filename"],
                    row["image_header_timestamp"],
                    row["bag_record_timestamp"],
                ]
            )
    print(
        json.dumps(
            dict(
                frames=len(found),
                maximum_record_header_offset_s=max(
                    abs(r["metadata_minus_header_s"]) for r in found.values()
                ),
                maximum_jpeg_raw_mean_pixel_error=max(
                    r["mean_jpeg_raw_abs_pixel_error"] for r in found.values()
                ),
            ),
            indent=2,
        )
    )


def retime_records(records, association):
    mapping = {r["frame"]: r for r in association["frames"]}
    if association["status"] != "complete" or len(mapping) != len(association["frames"]):
        raise ValueError("Complete unambiguous header associations required")
    result = []
    for row in records:
        identity = mapping[row["frame"]]
        if abs(row["timestamp"] - identity["metadata_timestamp"]) > 1e-6:
            raise ValueError("Match timestamp is not the associated source record")
        result.append(
            dict(
                row,
                timestamp=identity["image_header_timestamp"],
                metadata_timestamp=identity["metadata_timestamp"],
                bag_record_timestamp=identity["bag_record_timestamp"],
                image_header_sequence=identity["image_sequence"],
                timestamp_frame="native_camera_image_header",
            )
        )
    times = [r["timestamp"] for r in result]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate source image headers")
    return result


def retime(input_path, association_path, output):
    if output.exists():
        raise FileExistsError(str(output))
    source = json.loads(input_path.read_text())
    association = json.loads(association_path.read_text())
    for p, h in association["input_sha256"].items():
        if sha(p) != h:
            raise ValueError("Source association input changed: " + p)
    records = retime_records(source, association)
    output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".provenance.json").write_text(
        json.dumps(
            dict(
                status="complete",
                records=len(records),
                input_sha256={
                    str(p): sha(p)
                    for p in (input_path, association_path, Path(__file__).resolve())
                },
                output_sha256=sha(output),
                pixels_and_geographic_points_unchanged=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(records=len(records), output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("associate")
    for name in ("image-folder", "bag", "output"):
        first.add_argument("--" + name, type=Path, required=True)
    second = commands.add_parser("retime")
    for name in ("input", "associations", "output"):
        second.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "associate":
        associate(args.image_folder.resolve(), args.bag.resolve(), args.output.resolve())
    else:
        retime(args.input.resolve(), args.associations.resolve(), args.output.resolve())
