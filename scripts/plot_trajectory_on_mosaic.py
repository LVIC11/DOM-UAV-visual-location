"""Plot predicted and GT trajectories on the satellite mosaic."""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import SuperGlueConfig, SuperPointConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel


def gps_to_mosaic_pixel(lat, lon, mosaic_bounds, mosaic_shape):
    """Convert GPS (lat, lon) to pixel (x, y) on the mosaic image."""
    lat_min, lat_max, lon_min, lon_max = mosaic_bounds
    h, w = mosaic_shape[:2]
    x = int((lon - lon_min) / (lon_max - lon_min) * (w - 1))
    y = int((lat_max - lat) / (lat_max - lat_min) * (h - 1))
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--map-db", type=str, required=True)
    parser.add_argument("--mosaic", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/my_experiment/trajectory.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=logging.WARNING)

    # Mosaic bounds (from map.csv tile coverage)
    mosaic_bounds = (31.829732, 31.833233, 118.708649, 118.716888)
    mosaic = cv2.imread(args.mosaic)
    if mosaic is None:
        raise FileNotFoundError(f"Mosaic not found: {args.mosaic}")
    print(f"Mosaic: {mosaic.shape[1]}x{mosaic.shape[0]}")

    # Setup pipeline
    superpoint = SuperPointAlgorithm(SuperPointConfig(
        device=args.device, nms_radius=4, keypoint_threshold=0.01, max_keypoints=-1,
    ))
    superglue = SuperGlueMatcher(SuperGlueConfig(
        device=args.device, weights="outdoor", sinkhorn_iterations=20, match_threshold=0.5,
    ))
    map_reader = SatelliteMapReader(
        db_path=args.map_db, resize_size=(800,),
        logger=logging.getLogger("map_reader"),
    )
    map_reader.initialize_db()
    map_reader.setup_db()
    map_reader.resize_db_images()
    map_reader.describe_db_images(superpoint)

    streamer = DroneImageStreamer(
        image_folder=args.image_folder, has_gt=True,
        logger=logging.getLogger("streamer"),
    )
    if args.max_images:
        streamer._num_images = min(args.max_images, streamer._num_images)

    pipeline = Pipeline(
        map_reader=map_reader, drone_streamer=streamer,
        detector=superpoint, matcher=superglue,
        config=PipelineConfig(),
        query_processor=QueryProcessor(
            processings=None,
            camera_model=CameraModel(
                focal_length=4.5 / 1000, resolution_height=4056,
                resolution_width=3040, hfov_deg=82.9,
            ),
            satellite_resolution=None, size=(800,),
        ),
        logger=logging.getLogger("pipeline"),
    )

    print(f"Running on {len(streamer)} images...")
    preds = pipeline.run(output_path=None)

    # Collect coordinates
    gt_points = []
    pred_points = []
    for p in preds:
        if p["is_match"] and p["gt_coordinate"] is not None:
            gt = p["gt_coordinate"]
            pred = p["predicted_coordinate"]
            gt_xy = gps_to_mosaic_pixel(gt.lat, gt.long, mosaic_bounds, mosaic.shape)
            pred_xy = gps_to_mosaic_pixel(pred.lat, pred.long, mosaic_bounds, mosaic.shape)
            # Filter outliers (points outside mosaic)
            if 0 <= gt_xy[0] < mosaic.shape[1] and 0 <= gt_xy[1] < mosaic.shape[0]:
                gt_points.append(gt_xy)
                pred_points.append(pred_xy)

    print(f"Plotted {len(gt_points)} points on mosaic")

    # Draw on mosaic
    vis = mosaic.copy()
    # GT trajectory: green circles + green line
    for i, pt in enumerate(gt_points):
        cv2.circle(vis, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
        if i > 0:
            cv2.line(vis, gt_points[i-1], pt, (0, 255, 0), 2, cv2.LINE_AA)

    # Pred trajectory: magenta circles + magenta line
    for i, pt in enumerate(pred_points):
        cv2.circle(vis, pt, 4, (255, 0, 255), -1, cv2.LINE_AA)
        if i > 0:
            cv2.line(vis, pred_points[i-1], pt, (255, 0, 255), 2, cv2.LINE_AA)

    # Start points: larger circles
    if gt_points:
        cv2.circle(vis, gt_points[0], 8, (0, 255, 0), 3, cv2.LINE_AA)
    if pred_points:
        cv2.circle(vis, pred_points[0], 8, (255, 0, 255), 3, cv2.LINE_AA)

    # Legend
    cv2.putText(vis, "GT (green)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, "GT (green)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Pred (magenta)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, "Pred (magenta)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 255), 2, cv2.LINE_AA)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    print(f"Saved to {args.output}")

    # Compute stats
    dists = []
    for (gx, gy), (px, py) in zip(gt_points, pred_points):
        dists.append(np.sqrt((gx - px)**2 + (gy - py)**2))
    if dists:
        print(f"Pixel error: mean={np.mean(dists):.1f}px, max={np.max(dists):.1f}px, min={np.min(dists):.1f}px")


if __name__ == "__main__":
    main()
