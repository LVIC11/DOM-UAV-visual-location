"""Rotate drone query images based on BA_loop yaw angles.

For each query image:
1. Match its timestamp to the nearest BA_loop pose
2. Extract yaw from the body orientation quaternion
3. Rotate to align with the first image's yaw (relative alignment)
4. Then rotate additional 90° clockwise (standardize orientation)
5. Save rotated image to output directory
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_ba_loop(ba_loop_path: Path) -> np.ndarray:
    """Load BA_loop.txt, return array of [timestamp, tx, ty, tz, qw, qx, qy, qz]."""
    data = np.loadtxt(ba_loop_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def quat_to_yaw_deg(qw, qx, qy, qz) -> float:
    """Compute yaw angle (degrees, clockwise from north-ish) from quaternion.

    Uses scipy Rotation to robustly extract Euler angles from the body-to-world
    rotation. Assumes VINS convention: world=ENU, body=FLU (Forward-Left-Up).
    Yaw is the rotation around the world Z (Up) axis.
    """
    r = R.from_quat([qx, qy, qz, qw])  # scipy uses xyzw order
    # In ENU world: yaw = atan2(R[1,0], R[0,0]) = rotation around Z
    yaw_rad = r.as_euler('ZYX', degrees=False)[0]  # first angle is around Z
    return np.degrees(yaw_rad)


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around its center by angle_deg (positive = CCW in OpenCV)."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Compute new bounds to avoid cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rotated


def main():
    parser = argparse.ArgumentParser(description="Rotate query images by BA_loop yaw")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="path to query images")
    parser.add_argument("--ba-loop", type=str, required=True,
                        help="path to BA_loop.txt")
    parser.add_argument("--output", type=str, required=True,
                        help="output directory for rotated images")
    parser.add_argument("--flip-yaw", action="store_true",
                        help="negate yaw if images appear rotated the wrong way")
    args = parser.parse_args()

    image_folder = Path(args.image_folder)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BA_loop
    ba_data = load_ba_loop(Path(args.ba_loop))
    ba_ts = ba_data[:, 0]
    ba_quats = ba_data[:, 4:8]  # qw, qx, qy, qz

    # Load image timestamps from metadata.csv
    metadata_path = image_folder / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {image_folder}")

    image_files = []
    image_timestamps = []
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('Filename') or row.get('filename')
            ts = row.get('timestamp')
            image_files.append(fname)
            image_timestamps.append(float(ts))

    print(f"Found {len(image_files)} images")
    print(f"BA_loop has {len(ba_data)} poses")
    print(f"Image time range: {image_timestamps[0]:.3f} - {image_timestamps[-1]:.3f}")
    print(f"BA_loop time range: {ba_ts[0]:.3f} - {ba_ts[-1]:.3f}")

    # Compute first image's yaw as reference
    first_idx = np.argmin(np.abs(ba_ts - image_timestamps[0]))
    fqw, fqx, fqy, fqz = ba_quats[first_idx]
    first_yaw_deg = quat_to_yaw_deg(fqw, fqx, fqy, fqz)
    print(f"First image yaw (reference): {first_yaw_deg:.2f}°")

    # For each image, find nearest BA_loop pose and compute yaw
    processed = 0
    for fname, img_ts in zip(image_files, image_timestamps):
        # Find nearest BA_loop timestamp
        idx = np.argmin(np.abs(ba_ts - img_ts))

        qw, qx, qy, qz = ba_quats[idx]
        yaw_deg = quat_to_yaw_deg(qw, qx, qy, qz)

        if args.flip_yaw:
            yaw_deg = -yaw_deg

        # Align to first image's yaw, then rotate 90° CW
        relative_yaw = yaw_deg - first_yaw_deg
        opencv_angle = relative_yaw - 90.0

        # Read image
        img_path = image_folder / fname
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Cannot read {img_path}, skipping")
            continue

        rotated = rotate_image(img, opencv_angle)

        # Save
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), rotated)
        processed += 1

        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(image_files)} images...")

    print(f"Done. {processed} images saved to {output_dir}")


if __name__ == "__main__":
    main()
