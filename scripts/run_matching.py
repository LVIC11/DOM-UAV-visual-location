#!/usr/bin/env python3
"""Run visual localization pipeline on custom dataset."""

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
from svl.localization.map_reader import TileSatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel


if __name__ == "__main__":

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Run visual localization pipeline")
    parser.add_argument("--query-dir", type=str, required=True, help="path to drone query images folder")
    parser.add_argument("--map-dir", type=str, required=True, help="path to satellite map tiles folder")
    parser.add_argument("--output-path", type=str, default="data/output", help="output folder for visualizations and results")
    parser.add_argument("--device", type=str, default="cuda", help="device for models, e.g. cuda or cpu")
    parser.add_argument("--zoom-level", type=int, default=19, help="zoom level of the tile images")
    parser.add_argument("--max-images", type=int, default=None, help="maximum number of query images to process")
    parser.add_argument("--matcher", type=str, default="superglue", choices=["superglue", "lightglue"],
                        help="feature matcher: superglue or lightglue")
    args = parser.parse_args()

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
    map_reader = TileSatelliteMapReader(
        db_path=Path(args.map_dir),
        zoom_level=args.zoom_level,
        logger=logging.getLogger("%s.TileSatelliteMapReader" % __name__),
    )
    map_reader.initialize_db()
    map_reader.setup_db()
    map_reader.resize_db_images()
    map_reader.describe_db_images(superpoint_algorithm)

    # Initialize the drone image streamer
    streamer = DroneImageStreamer(
        image_folder=Path(args.query_dir),
        has_gt=False,
        logger=logging.getLogger("%s.DroneImageStreamer" % __name__),
    )
    print(f"Number of query images: {len(streamer)}")

    # Initialize the query processor
    camera_model = CameraModel(
        focal_length=4.5 / 1000,  # 4.5mm
        resolution_height=3040,
        resolution_width=4056,
        hfov_deg=82.9,
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
        config=PipelineConfig(),
        logger=logger,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    preds = pipeline.run(
        output_path=output_path,
    )

    print("Pipeline completed!")
    print(f"Results saved to {output_path}")
