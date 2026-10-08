import math
import argparse
import logging
from pathlib import Path
from pprint import pprint

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import LightGlueMatcher, SuperGlueMatcher
from svl.keypoint_pipeline.typing import (
    LightGlueConfig,
    SuperGlueConfig,
    SuperPointConfig,
)
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.geometry import CameraCalibration
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel


def parse_rotation_ranges(spec: str):
    rotations = {}
    if not spec:
        return rotations
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        frame_range, angle = item.split(":")
        angle = float(angle)
        if "-" in frame_range:
            start, end = (int(v) for v in frame_range.split("-", 1))
        else:
            start = end = int(frame_range)
        for idx in range(start, end + 1):
            rotations[f"{idx:06d}"] = angle
    return rotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run visual localization pipeline")
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="path to drone query images folder",
    )
    parser.add_argument(
        "--map-db", type=str, required=True, help="path to satellite map tiles folder"
    )
    parser.add_argument(
        "--output-path",
        "--output",
        dest="output_path",
        type=str,
        default="data/output",
        help="output folder for visualizations and results",
    )
    parser.add_argument(
        "--no-gt", action="store_true", help="run without ground truth metadata"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device for models, e.g. cuda or cpu"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="maximum number of query images to process",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="superglue",
        choices=["superglue", "lightglue"],
        help="feature matcher: superglue or lightglue",
    )
    parser.add_argument(
        "--use-centroid",
        action="store_true",
        help="use keypoint centroid instead of image center for geo-pose",
    )
    parser.add_argument(
        "--query-rotation-cw",
        type=float,
        default=0.0,
        help="clockwise rotation already applied to query images; exported query_x/query_y are mapped back",
    )
    parser.add_argument(
        "--query-rotation-ranges",
        default="",
        help="per-frame rotations, e.g. '0-13:90,14-18:180,19-36:270'",
    )
    parser.add_argument(
        "--camera-calib",
        default="/media/sy/30C2CCDEC2CCAA04/datasets/a_low/winter/nav_left.yaml",
        help="OpenCV camera calibration YAML for the original unrotated query image",
    )
    parser.add_argument("--reference-lat", type=float, default=31.8325939032)
    parser.add_argument("--reference-lon", type=float, default=118.71416416)
    parser.add_argument("--pnp-min-inliers", type=int, default=20)
    parser.add_argument("--pnp-ransac-threshold", type=float, default=6.0)
    parser.add_argument("--pnp-min-bbox-ratio", type=float, default=0.03)
    parser.add_argument("--pnp-max-tilt", type=float, default=30.0)
    parser.add_argument("--pnp-tilt-regularization", type=float, default=0.05)
    parser.add_argument(
        "--use-gt-tile-prior",
        action="store_true",
        help="Restrict candidate tiles using metadata GPS (debug/evaluation only)",
    )
    parser.add_argument("--gt-tile-prior-radius", type=float, default=150.0)
    args = parser.parse_args()
    query_rotation_by_name = parse_rotation_ranges(args.query_rotation_ranges)

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # Initialize the keypoint detector
    superpoint_config = SuperPointConfig(
        device=args.device,
        nms_radius=4,
        keypoint_threshold=0.01,
        max_keypoints=-1,
    )
    superpoint_algorithm = SuperPointAlgorithm(superpoint_config)

    # Initialize the keypoint matcher
    if args.matcher == "lightglue":
        matcher_config = LightGlueConfig(
            device=args.device,
            features="superpoint",
            n_layers=9,
            filter_threshold=0.5,
        )
        matcher = LightGlueMatcher(matcher_config)
    else:
        matcher_config = SuperGlueConfig(
            device=args.device,
            weights="outdoor",
            sinkhorn_iterations=20,
            match_threshold=0.5,
        )
        matcher = SuperGlueMatcher(matcher_config)

    # Initialize the map reader
    map_reader = SatelliteMapReader(
        db_path=args.map_db,
        resize_size=(800,),
        logger=logging.getLogger("%s.SatelliteMapReader" % __name__),
    )
    map_reader.initialize_db()
    map_reader.setup_db()
    map_reader.resize_db_images()
    map_reader.describe_db_images(superpoint_algorithm)

    # Initialize the drone image streamer
    streamer = DroneImageStreamer(
        image_folder=args.image_folder,
        has_gt=not args.no_gt,
        logger=logging.getLogger("%s.DroneImageStreamer" % __name__),
    )
    print(f"Number of query images: {len(streamer)}")

    if args.max_images:
        streamer._num_images = min(args.max_images, streamer._num_images)

    # Initialize the query processor
    camera_calibration = CameraCalibration.from_opencv_yaml(args.camera_calib)
    hfov_deg = math.degrees(
        2.0
        * math.atan(
            camera_calibration.image_width
            / (2.0 * camera_calibration.camera_matrix[0, 0])
        )
    )
    camera_model = CameraModel(
        focal_length=1.0,
        resolution_height=camera_calibration.image_height,
        resolution_width=camera_calibration.image_width,
        hfov_deg=hfov_deg,
        principal_point_x=camera_calibration.camera_matrix[0, 2],
        principal_point_y=camera_calibration.camera_matrix[1, 2],
    )
    query_processor = QueryProcessor(
        processings=None,
        camera_model=camera_model,
        satellite_resolution=None,
        size=(800,),
    )

    # Initialize the pipeline
    logger = logging.getLogger("%s.Pipeline" % __name__)
    logger.setLevel(logging.DEBUG)
    pipeline = Pipeline(
        map_reader=map_reader,
        drone_streamer=streamer,
        detector=superpoint_algorithm,
        matcher=matcher,
        query_processor=query_processor,
        config=PipelineConfig(
            use_centroid=args.use_centroid,
            query_rotation_cw=args.query_rotation_cw,
            query_rotation_by_name=query_rotation_by_name,
            camera_calibration=camera_calibration,
            reference_latitude=args.reference_lat,
            reference_longitude=args.reference_lon,
            pnp_min_inliers=args.pnp_min_inliers,
            pnp_ransac_threshold_px=args.pnp_ransac_threshold,
            pnp_min_bbox_area_ratio=args.pnp_min_bbox_ratio,
            pnp_max_optical_tilt_deg=args.pnp_max_tilt,
            pnp_tilt_regularization_px_per_deg=args.pnp_tilt_regularization,
            use_gt_tile_prior=args.use_gt_tile_prior,
            gt_tile_prior_radius_m=args.gt_tile_prior_radius,
        ),
        logger=logger,
    )
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    preds = pipeline.run(
        output_path=output_path,
    )
    metrics = pipeline.compute_metrics(preds)
    pprint(metrics)
