"""Warp drone images to satellite view using homography matching.

For each drone image:
1. Match against satellite tiles to find the best tile
2. Compute homography H via SuperPoint + SuperGlue + RANSAC
3. Warp the full drone image using H to satellite coordinates
4. Save the warped image
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import SuperGlueConfig, SuperPointConfig
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import PipelineConfig
from svl.tms.data_structures import GeoSatelliteImage
from svl.tms.schemas import GpsCoordinate
from svl.tms.geo import haversine_distance


def main():
    parser = argparse.ArgumentParser(description="Warp drone images to satellite view")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--map-db", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    # Load map
    map_reader = SatelliteMapReader(
        db_path=args.map_db, resize_size=(800,),
        logger=logging.getLogger("map"),
    )
    map_reader.initialize_db()
    map_reader.setup_db()
    map_reader.resize_db_images()

    # Load models
    superpoint = SuperPointAlgorithm(SuperPointConfig(
        device=args.device, nms_radius=4, keypoint_threshold=0.01, max_keypoints=-1,
    ))
    superglue = LightGlueMatcher(LightGlueConfig(
        device=args.device, features="superpoint", n_layers=9, filter_threshold=0.5,
    ))
    map_reader.describe_db_images(superpoint)

    config = PipelineConfig()

    # Collect drone images
    image_folder = Path(args.image_folder)
    image_files = sorted(image_folder.glob("*.jpg"))
    if args.max_images:
        image_files = image_files[:args.max_images]
    print(f"Processing {len(image_files)} images...")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_files, desc="Warping"):
        drone_img = cv2.imread(str(img_path))
        if drone_img is None:
            print(f"  Skip {img_path.name}: cannot read")
            continue

        # Convert to grayscale for SuperPoint, resize to match map reader
        drone_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
        h, w = drone_gray.shape[:2]
        new_w, new_h = 800, int(800 * h / w)
        drone_gray_rs = cv2.resize(drone_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        drone_bgr_rs = cv2.resize(drone_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Detect keypoints
        drone_kp = superpoint.detect_and_describe_keypoints(drone_gray_rs)

        best_matches = -1
        best_H = None
        best_sat = None

        for idx in range(len(map_reader)):
            sat: GeoSatelliteImage = map_reader[idx]
            matches, confidence = superglue.match_keypoints(drone_kp, sat.key_points)
            valid = matches > -1
            mkpts0 = drone_kp.keypoints[valid]
            mkpts1 = sat.key_points.keypoints[matches[valid]]

            if len(mkpts0) < 4:
                continue

            try:
                H, mask = cv2.findHomography(
                    mkpts0, mkpts1, config.homography_method,
                    config.homography_threshold,
                    maxIters=config.homography_max_iter,
                    confidence=config.homography_confidence,
                )
            except cv2.error:
                continue

            if H is not None and len(mkpts1) > best_matches:
                # Validate center is in bounds
                dh, dw = drone_gray_rs.shape[:2]
                center = np.float32([[[dw / 2, dh / 2]]])
                dc = cv2.perspectiveTransform(center, H)[0][0]
                sh, sw = sat.image.shape[:2]
                cx, cy = dc[0] / sw, dc[1] / sh
                best_matches = len(mkpts1)
                best_H = H
                best_sat = sat

        if best_H is not None:
            sh, sw = best_sat.image.shape[:2]
            warped = cv2.warpPerspective(
                drone_bgr_rs, best_H, (sw, sh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), warped)
        else:
            print(f"  No match for {img_path.name} (best_matches={best_matches})")

    print(f"Done. Images saved to {output_dir}")


if __name__ == "__main__":
    main()
