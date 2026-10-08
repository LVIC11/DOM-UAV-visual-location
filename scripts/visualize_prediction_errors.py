"""Visualize prediction error vectors on the satellite mosaic.

For each matched image, draws an arrow from the predicted look-at point to the
ground truth point, showing the direction and magnitude of the error.
"""
import argparse
import csv
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


MOSAIC_BOUNDS = (31.829732, 31.833233, 118.708649, 118.716888)  # lat_min, lat_max, lon_min, lon_max


def gps_to_px(lat, lon, mosaic_w=1536, mosaic_h=768):
    lat_min, lat_max, lon_min, lon_max = MOSAIC_BOUNDS
    x = int((lon - lon_min) / (lon_max - lon_min) * (mosaic_w - 1))
    y = int((lat_max - lat) / (lat_max - lat_min) * (mosaic_h - 1))
    return x, y


def draw_arrow(img, pt_from, pt_to, color, thickness=2, tip_length=8):
    """Draw an arrow from pt_from to pt_to."""
    cv2.arrowedLine(img, pt_from, pt_to, color, thickness,
                    line_type=cv2.LINE_AA, tipLength=tip_length / max(np.linalg.norm(np.array(pt_to) - np.array(pt_from)), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--map-db", type=str, required=True)
    parser.add_argument("--mosaic", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/my_experiment/error_vectors.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    # Setup
    superpoint = SuperPointAlgorithm(SuperPointConfig(device=args.device))
    superglue = SuperGlueMatcher(SuperGlueConfig(device=args.device))
    map_reader = SatelliteMapReader(db_path=args.map_db, resize_size=(800,), logger=logging.getLogger("m"))
    map_reader.initialize_db()
    map_reader.setup_db()
    map_reader.resize_db_images()
    map_reader.describe_db_images(superpoint)

    streamer = DroneImageStreamer(image_folder=args.image_folder, has_gt=True, logger=logging.getLogger("s"))
    if args.max_images:
        streamer._num_images = min(args.max_images, streamer._num_images)

    pipeline = Pipeline(
        map_reader=map_reader, drone_streamer=streamer,
        detector=superpoint, matcher=superglue,
        config=PipelineConfig(),
        query_processor=QueryProcessor(
            processings=None,
            camera_model=CameraModel(focal_length=4.5 / 1000, resolution_height=4056,
                                     resolution_width=3040, hfov_deg=82.9),
            satellite_resolution=None, size=(800,),
        ),
        logger=logging.getLogger("p"),
    )

    print(f"Running pipeline on {len(streamer)} images...")
    preds = pipeline.run(output_path=None)

    # Load mosaic
    mosaic = cv2.imread(args.mosaic)
    vis = mosaic.copy()
    mh, mw = vis.shape[:2]

    # Load timestamps
    ts_map = {}
    meta_path = Path(args.image_folder) / "metadata.csv"
    if meta_path.exists():
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            fname_col = 'Filename' if 'Filename' in reader.fieldnames else 'filename'
            for row in reader:
                ts_map[row[fname_col]] = float(row['timestamp'])

    # Draw error vectors
    colors = {
        'good': (0, 255, 0),    # green: <10m
        'warn': (0, 255, 255),  # yellow: 10-30m
        'bad':  (0, 0, 255),    # red: >30m
    }

    good_count = warn_count = bad_count = 0
    for p in preds:
        if not p['is_match'] or p['gt_coordinate'] is None:
            continue

        gt = p['gt_coordinate']
        pred = p['predicted_coordinate']
        dist = p['distance'] or 0

        pred_px = gps_to_px(pred.lat, pred.long, mw, mh)
        gt_px = gps_to_px(gt.lat, gt.long, mw, mh)

        if not (0 <= pred_px[0] < mw and 0 <= pred_px[1] < mh):
            continue
        if not (0 <= gt_px[0] < mw and 0 <= gt_px[1] < mh):
            continue

        if dist < 10:
            color = colors['good']
            good_count += 1
        elif dist < 30:
            color = colors['warn']
            warn_count += 1
        else:
            color = colors['bad']
            bad_count += 1

        # Draw arrow from predicted to GT
        draw_arrow(vis, pred_px, gt_px, color, thickness=2)

        # Predicted point: small filled circle
        cv2.circle(vis, pred_px, 3, (255, 0, 255), -1, cv2.LINE_AA)

    # Legend
    y0 = 25
    for label, clr in [("Pred→GT (<10m)", colors['good']),
                        ("Pred→GT (10-30m)", colors['warn']),
                        ("Pred→GT (>30m)", colors['bad']),
                        ("Predicted (look-at)", (255, 0, 255))]:
        cv2.putText(vis, label, (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)
        y0 += 22

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    print(f"Saved to {args.output}")
    print(f"Good (<10m): {good_count}, Warn (10-30m): {warn_count}, Bad (>30m): {bad_count}")


if __name__ == "__main__":
    main()
