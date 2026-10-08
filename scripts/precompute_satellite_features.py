#!/usr/bin/env python3
"""Pre-compute SuperPoint features for all satellite tiles and save to a .npz file.

Run ONCE offline. The output .npz file is loaded by run_single_match.py at runtime.
Eliminates the need to run SuperPoint on satellite tiles during live matching.

Usage:
    python precompute_satellite_features.py \
        --tile-dir /path/to/tiles \
        --tile-index /path/to/tile_index.json \
        --output /path/to/sat_features.npz \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.typing import SuperPointConfig


def resize_long_side(img: np.ndarray, target_size: int = 800) -> np.ndarray:
    """Resize both small and large tiles to the feature extraction scale."""

    if target_size <= 0:
        return img
    height, width = img.shape[:2]
    scale = float(target_size) / float(max(height, width))
    if np.isclose(scale, 1.0):
        return img
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute satellite tile features")
    parser.add_argument("--tile-dir", type=str, required=True)
    parser.add_argument("--tile-index", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resize-size",
        "--max-width",
        dest="resize_size",
        type=int,
        default=800,
        help="Target tile long-side size used for feature extraction",
    )
    args = parser.parse_args()

    # Load tile index
    with open(args.tile_index, "r") as f:
        tile_index = json.load(f)
    print(f"Loaded tile index with {len(tile_index)} tiles")

    # Initialize SuperPoint
    sp_config = SuperPointConfig(
        device=args.device,
        nms_radius=4,
        keypoint_threshold=0.01,
        max_keypoints=-1,
    )
    superpoint = SuperPointAlgorithm(sp_config)
    print("Loaded SuperPoint model")

    # Pre-compute features for each tile
    all_features = {}
    total_kpts = 0

    for i, tile_entry in enumerate(tile_index):
        tile_name = tile_entry["name"]
        tile_path = str(Path(args.tile_dir) / tile_name)

        img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(
                f"  [{i+1}/{len(tile_index)}] SKIP {tile_name} — not found",
                file=sys.stderr,
            )
            continue

        img = resize_long_side(img, target_size=args.resize_size)
        kpts = superpoint.detect_and_describe_keypoints(img)

        # Store as numpy arrays
        all_features[tile_name] = {
            "keypoints": kpts.keypoints.astype(np.float32),  # (N, 2)
            "descriptors": kpts.descriptors.astype(np.float32),  # (N, 256)
            "scores": kpts.scores.astype(np.float32),  # (N,)
            "image_size": np.array(img.shape[:2], dtype=np.int32),  # (H, W)
        }
        total_kpts += len(kpts.keypoints)
        print(
            f"  [{i+1}/{len(tile_index)}] {tile_name}: {len(kpts.keypoints)} keypoints"
        )

    # Save to .npz
    save_dict = {}
    for tile_name, feats in all_features.items():
        prefix = tile_name.replace(".", "_")  # npz key can't have dots
        save_dict[f"{prefix}_keypoints"] = feats["keypoints"]
        save_dict[f"{prefix}_descriptors"] = feats["descriptors"]
        save_dict[f"{prefix}_scores"] = feats["scores"]
        save_dict[f"{prefix}_image_size"] = feats["image_size"]

    # Also save the tile index and tile name list for easy lookup
    save_dict["__tile_names__"] = np.array(list(all_features.keys()))
    save_dict["__num_tiles__"] = np.array([len(all_features)])
    save_dict["__resize_size__"] = np.array([args.resize_size], dtype=np.int32)

    np.savez_compressed(args.output, **save_dict)

    # Report
    file_size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nDone: {len(all_features)} tiles, {total_kpts} total keypoints")
    print(f"Output: {args.output} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
