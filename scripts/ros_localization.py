#!/usr/bin/env python3
"""
Real-time visual localization node that reads from a ROS bag file.

This script reads drone images and GNSS data from a rosbag, runs the
visual localization pipeline against a pre-tiled satellite map, and
outputs predictions with visualizations.

Usage:
    # First, tile the GeoTIFF map (one-time step):
    python scripts/prepare_geotiff_map.py \
        --geotiff data/geovins/map/lingshan.tif \
        --output data/geovins/map/tiles

    # Then run real-time localization:
    python scripts/ros_localization.py \
        --bag /path/to/TD_03.bag \
        --map-db data/geovins/map/tiles \
        --output data/output/td03 \
        --device cuda
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
from rosbags.highlevel import AnyReader

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import LightGlueMatcher, SuperGlueMatcher
from svl.keypoint_pipeline.typing import (
    LightGlueConfig,
    SuperGlueConfig,
    SuperPointConfig,
)
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel, DroneImage
from svl.tms.schemas import GeoPoint


logger = logging.getLogger("ros_localization")


class RosbagLocalizer:
    """Read images and GNSS from a rosbag and run visual localization in real-time.

    Parameters
    ----------
    bag_path : Path
        Path to the ROS bag file.
    pipeline : Pipeline
        Initialized localization pipeline.
    query_processor : QueryProcessor
        Query image preprocessor.
    image_topic : str
        ROS topic for compressed images.
    gnss_topic : str
        ROS topic for GNSS/GPS messages.
    output_path : Path
        Directory to save visualizations and results.
    max_images : int, optional
        Maximum number of images to process.
    realtime_factor : float
        Playback speed multiplier (1.0 = real-time).
    skip_frames : int
        Process every N-th image frame (1 = all frames).
    """

    def __init__(
        self,
        bag_path: Path,
        pipeline: Pipeline,
        query_processor: QueryProcessor,
        image_topic: str = "/camera_array/cam0/image_raw/compressed",
        gnss_topic: str = "/gnss/data",
        output_path: Path = None,
        max_images: Optional[int] = None,
        realtime_factor: float = 1.0,
        skip_frames: int = 1,
        gps_search_radius_m: float = 500.0,
        no_gps_filter: bool = False,
        rotate_cw: bool = False,
        full_map_path: Path = None,
        max_distance_m: float = None,
    ):
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {self.bag_path}")

        self.pipeline = pipeline
        self.query_processor = query_processor
        self.image_topic = image_topic
        self.gnss_topic = gnss_topic
        self.output_path = Path(output_path) if output_path else None
        self.max_images = max_images
        self.realtime_factor = realtime_factor
        self.skip_frames = skip_frames
        self.gps_search_radius_m = gps_search_radius_m
        self.no_gps_filter = no_gps_filter
        self.rotate_cw = rotate_cw
        self.full_map_path = Path(full_map_path) if full_map_path else None
        self.max_distance_m = max_distance_m

        if self.output_path:
            self.output_path.mkdir(parents=True, exist_ok=True)

        # Pre-loaded GNSS data: list of (timestamp_sec, latitude, longitude, altitude)
        self._gnss_data: List[Tuple[float, float, float, float]] = []

        # Full-map metadata for trajectory drawing
        self._map_image: Optional[np.ndarray] = None
        self._map_geo_bounds: Optional[Dict[str, float]] = None

    def _load_gnss_data(self, reader: AnyReader) -> None:
        """Pre-load all GNSS messages for fast timestamp-based lookup."""
        logger.info("Loading GNSS data from bag...")
        gnss_conn = next(
            c for c in reader.connections if c.topic == self.gnss_topic
        )

        for conn, ts, raw in reader.messages(connections=[gnss_conn]):
            msg = reader.deserialize(raw, conn.msgtype)
            self._gnss_data.append((
                ts / 1e9,  # nanoseconds to seconds
                msg.latitude,
                msg.longitude,
                msg.altitude,
            ))

        logger.info(f"Loaded {len(self._gnss_data)} GNSS messages")

    def _get_gnss_at(self, timestamp_sec: float) -> Optional[Tuple[float, float, float]]:
        """Find the nearest GNSS fix by timestamp.

        Returns (latitude, longitude, altitude) or None.
        """
        if not self._gnss_data:
            return None
        nearest = min(self._gnss_data, key=lambda r: abs(r[0] - timestamp_sec))
        # Reject if time difference > 1.0 second
        if abs(nearest[0] - timestamp_sec) > 1.0:
            return None
        return (nearest[1], nearest[2], nearest[3])

    def _decode_image(self, msg) -> np.ndarray:
        """Decode a CompressedImage message to a grayscale numpy array."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Decode JPEG (BGR)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode JPEG image from bag")
        # Rotate 90° clockwise if requested
        if self.rotate_cw:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        # Convert to grayscale (matching DroneImageStreamer's cv2.IMREAD_GRAYSCALE)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray

    def run(self) -> List[Dict]:
        """Run the localization pipeline on images from the bag.

        Returns
        -------
        List[Dict]
            Prediction results for each processed image.
        """
        logger.info(f"Opening bag: {self.bag_path}")

        # Load full map for trajectory drawing (if enabled)
        self._load_full_map()

        all_preds = []

        with AnyReader([self.bag_path]) as reader:
            # Pre-load GNSS data
            self._load_gnss_data(reader)

            img_conn = next(
                c for c in reader.connections if c.topic == self.image_topic
            )
            logger.info(f"Image topic: {self.image_topic} (type: {img_conn.msgtype})")

            # Timing
            bag_start_ts = None  # bag timestamp of first image (seconds)
            wall_start_time = None  # wall clock when we started processing
            frame_count = 0
            processed_count = 0

            for conn, ts, raw in reader.messages(connections=[img_conn]):
                frame_count += 1

                if (frame_count - 1) % self.skip_frames != 0:
                    continue

                msg = reader.deserialize(raw, conn.msgtype)
                img_ts = ts / 1e9  # nanoseconds to seconds

                # Initialize timing on first frame
                if bag_start_ts is None:
                    bag_start_ts = img_ts
                    wall_start_time = time.time()

                # Real-time playback: sleep until this frame's turn
                if self.realtime_factor > 0:
                    expected_elapsed = (img_ts - bag_start_ts) / self.realtime_factor
                    actual_elapsed = time.time() - wall_start_time
                    sleep_duration = expected_elapsed - actual_elapsed
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

                # Decode image
                try:
                    image = self._decode_image(msg)
                except Exception as e:
                    logger.warning(f"Failed to decode frame {frame_count}: {e}")
                    continue

                # Get nearest GNSS fix
                gnss = self._get_gnss_at(img_ts)
                geo_point = None
                if gnss is not None:
                    geo_point = GeoPoint(
                        latitude=gnss[0],
                        longitude=gnss[1],
                        altitude=gnss[2],
                    )

                # Create DroneImage
                drone_image = DroneImage(
                    image_path=Path(f"frame_{frame_count:06d}.jpg"),
                    geo_point=geo_point,
                    timestamp=img_ts,
                    image=image,
                )

                # Preprocess query image
                query = self.query_processor(drone_image)

                # Find nearby tile indices from GPS
                candidate_indices = None
                if not self.no_gps_filter and query.geo_point is not None:
                    from svl.tms.schemas import GpsCoordinate
                    drone_gps = GpsCoordinate(
                        lat=query.geo_point.latitude,
                        long=query.geo_point.longitude,
                    )
                    candidate_indices = self.pipeline._find_nearby_tile_indices(
                        drone_gps, radius_m=self.gps_search_radius_m
                    )
                    logger.debug(
                        f"Frame {frame_count}: {len(candidate_indices)} nearby tiles"
                    )

                # Run pipeline on this image
                logger.info(
                    f"Processing frame {frame_count} "
                    f"(ts={img_ts - bag_start_ts:.1f}s, "
                    f"GPS={'yes' if gnss else 'no'})"
                )
                pred = self.pipeline.run_on_image(
                    query,
                    output_path=self.output_path,
                    candidate_indices=candidate_indices,
                )

                # Store matched image name as string (not object) for CSV output
                pred["matched_image"] = (
                    pred["matched_image"].name if pred["matched_image"] else None
                )
                pred["frame"] = frame_count
                pred["timestamp"] = img_ts
                all_preds.append(pred)

                processed_count += 1

                # Reject matches exceeding max distance threshold
                if pred["is_match"] and self.max_distance_m is not None:
                    if pred.get("distance") is not None and pred["distance"] > self.max_distance_m:
                        pred["is_match"] = False
                        logger.warning(
                            f"  REJECTED (dist={pred['distance']:.1f}m > {self.max_distance_m}m)"
                        )

                # Status
                if pred["is_match"]:
                    dist_str = (
                        f"dist={pred['distance']:.2f}m" if pred.get("distance") else ""
                    )
                    logger.info(
                        f"  MATCH: {pred['predicted_coordinate']} {dist_str}"
                    )
                else:
                    logger.warning(f"  NO MATCH for frame {frame_count}")

                if self.max_images and processed_count >= self.max_images:
                    logger.info(f"Reached max_images limit ({self.max_images})")
                    break

            # Save aggregate results CSV
            if self.output_path and all_preds:
                self._save_results_csv(all_preds)

        # Draw trajectory on full map
        if self.full_map_path and all_preds:
            self._draw_trajectory_on_map(all_preds)

        logger.info(
            f"Done. Processed {processed_count}/{frame_count} frames, "
            f"matched {sum(1 for p in all_preds if p['is_match'])}"
        )
        return all_preds

    def _load_full_map(self) -> None:
        """Load the full GeoTIFF map and its geographic bounds for trajectory drawing."""
        if self.full_map_path is None:
            return

        logger.info(f"Loading full map from: {self.full_map_path}")
        img = tifffile.imread(str(self.full_map_path))
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Parse geographic bounds from companion .txt file
        txt_path = self.full_map_path.with_suffix(".txt")
        bounds = self._parse_map_txt(txt_path)

        self._map_image = img
        self._map_geo_bounds = bounds
        logger.info(f"Full map loaded: {w}x{h}, bounds: {bounds}")

    @staticmethod
    def _parse_map_txt(txt_path: Path) -> Dict[str, float]:
        """Parse the lingshan.txt-style metadata file for WGS84 bounds."""
        if not txt_path.exists():
            raise FileNotFoundError(f"Map metadata not found: {txt_path}")

        content = None
        for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(txt_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if content is None:
            raise RuntimeError(f"Could not decode {txt_path}")

        bounds = {}
        in_wgs84 = False
        for line in content.split("\n"):
            line = line.strip()
            if "WGS 84 geographic coordinates" in line:
                in_wgs84 = True
                continue
            if in_wgs84:
                if "Upper-left coordinate:" in line:
                    parts = line.split(":")[1].strip().split(",")
                    bounds["top_left_lon"] = float(parts[0].strip())
                    bounds["top_left_lat"] = float(parts[1].strip())
                elif "Lower-right coordinate:" in line:
                    parts = line.split(":")[1].strip().split(",")
                    bounds["bottom_right_lon"] = float(parts[0].strip())
                    bounds["bottom_right_lat"] = float(parts[1].strip())

        required = ["top_left_lat", "top_left_lon", "bottom_right_lat", "bottom_right_lon"]
        for k in required:
            if k not in bounds:
                raise RuntimeError(f"Could not parse {k} from {txt_path}. Found: {bounds}")

        return bounds

    def _gps_to_full_map_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert GPS coordinate to pixel on the full map image."""
        b = self._map_geo_bounds
        h, w = self._map_image.shape[:2]
        lat_range = b["top_left_lat"] - b["bottom_right_lat"]
        lon_range = b["bottom_right_lon"] - b["top_left_lon"]
        px = int((lon - b["top_left_lon"]) / lon_range * w)
        py = int((b["top_left_lat"] - lat) / lat_range * h)
        return px, py

    def _draw_trajectory_on_map(self, preds: List[Dict]) -> None:
        """Draw GT and predicted trajectories on the full satellite map."""
        if self.full_map_path is None or self._map_image is None:
            return
        if self.output_path is None:
            return

        logger.info("Drawing trajectory on full map...")
        canvas = self._map_image.copy()

        gt_points = []    # green: ground truth GPS
        pred_points = []  # red: predicted GPS

        for p in preds:
            if p.get("gt_coordinate") is not None:
                gt = p["gt_coordinate"]
                px, py = self._gps_to_full_map_pixel(gt.lat, gt.long)
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    gt_points.append((px, py))
            if p.get("predicted_coordinate") is not None:
                pr = p["predicted_coordinate"]
                px, py = self._gps_to_full_map_pixel(pr.lat, pr.long)
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    pred_points.append((px, py))

        # Draw GT trajectory (green, thicker)
        for i in range(1, len(gt_points)):
            cv2.line(canvas, gt_points[i - 1], gt_points[i], (0, 255, 0), 3, cv2.LINE_AA)
        # Draw predicted trajectory (red)
        for i in range(1, len(pred_points)):
            cv2.line(canvas, pred_points[i - 1], pred_points[i], (0, 0, 255), 2, cv2.LINE_AA)
        # Draw start/end markers
        if gt_points:
            cv2.circle(canvas, gt_points[0], 8, (0, 255, 0), -1, cv2.LINE_AA)   # start
            cv2.circle(canvas, gt_points[-1], 8, (0, 128, 0), -1, cv2.LINE_AA)  # end
        if pred_points:
            cv2.circle(canvas, pred_points[0], 8, (0, 0, 255), -1, cv2.LINE_AA)   # start
            cv2.circle(canvas, pred_points[-1], 8, (128, 0, 0), -1, cv2.LINE_AA)  # end

        # Legend
        legend = [
            ("GT trajectory", (0, 255, 0)),
            ("Pred trajectory", (0, 0, 255)),
        ]
        y0 = 30
        for i, (label, color) in enumerate(legend):
            cv2.putText(canvas, label, (20, y0 + i * 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (20, y0 + i * 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)

        out_path = self.output_path / "trajectory_map.jpg"
        cv2.imwrite(str(out_path), canvas)
        logger.info(f"Trajectory map saved to {out_path}")

    def _save_results_csv(self, preds: List[Dict]) -> None:
        """Save prediction results to a CSV file."""
        csv_path = self.output_path / "localization_results.csv"
        fieldnames = [
            "frame", "timestamp", "is_match",
            "gt_lat", "gt_lon",
            "pred_lat", "pred_lon",
            "distance_m", "matched_image", "num_inliers",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for p in preds:
                row = dict(p)
                # Flatten GPS coordinates
                if p.get("gt_coordinate"):
                    row["gt_lat"] = p["gt_coordinate"].lat
                    row["gt_lon"] = p["gt_coordinate"].long
                else:
                    row["gt_lat"] = None
                    row["gt_lon"] = None
                if p.get("predicted_coordinate"):
                    row["pred_lat"] = p["predicted_coordinate"].lat
                    row["pred_lon"] = p["predicted_coordinate"].long
                else:
                    row["pred_lat"] = None
                    row["pred_lon"] = None
                row["distance_m"] = p.get("distance", None)
                writer.writerow(row)
        logger.info(f"Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time visual localization from ROS bag"
    )
    # Bag and map
    parser.add_argument("--bag", type=str, required=True,
                        help="Path to the ROS bag file")
    parser.add_argument("--map-db", type=str, required=True,
                        help="Path to the pre-tiled satellite map folder (with map.csv)")
    parser.add_argument("--output", type=str, default="data/output/ros_localization",
                        help="Output folder for visualizations and results")

    # Pipeline configuration
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for models: cuda or cpu")
    parser.add_argument("--matcher", type=str, default="superglue",
                        choices=["superglue", "lightglue"],
                        help="Feature matcher: superglue or lightglue")

    # Playback control
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to process")
    parser.add_argument("--realtime-factor", type=float, default=1.0,
                        help="Playback speed: 1.0=real-time, 0=as-fast-as-possible")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Process every N-th frame (1=all frames)")
    parser.add_argument("--gps-search-radius", type=float, default=500.0,
                        help="Search radius in meters for nearby tiles (default: 500m)")
    parser.add_argument("--no-gps-filter", action="store_true",
                        help="Match against all tiles instead of only GPS-nearby ones")
    parser.add_argument("--rotate-cw", action="store_true",
                        help="Rotate images 90 degrees clockwise before processing")
    parser.add_argument("--use-centroid", action="store_true",
                        help="Use keypoint centroid instead of image center for geo-pose")
    parser.add_argument("--max-distance", type=float, default=None,
                        help="Reject matches with haversine distance > this value (meters)")
    parser.add_argument("--draw-trajectory", action="store_true",
                        help="Draw GT and predicted trajectory on the full satellite map")
    parser.add_argument("--full-map", type=str,
                        default="data/geovins/map/lingshan.tif",
                        help="Path to the full GeoTIFF map for trajectory drawing")

    # Topics
    parser.add_argument("--image-topic", type=str,
                        default="/camera_array/cam0/image_raw/compressed",
                        help="ROS topic for compressed images")
    parser.add_argument("--gnss-topic", type=str, default="/gnss/data",
                        help="ROS topic for GNSS/GPS data")

    # Map reader
    parser.add_argument("--resize-size", type=int, default=800,
                        help="Resize map tiles so width equals this value")

    # Camera model
    parser.add_argument("--focal-length", type=float, default=4.5,
                        help="Camera focal length in mm")
    parser.add_argument("--resolution-width", type=int, default=3040,
                        help="Camera image width in pixels")
    parser.add_argument("--resolution-height", type=int, default=4056,
                        help="Camera image height in pixels")
    parser.add_argument("--hfov-deg", type=float, default=82.9,
                        help="Camera horizontal field of view in degrees")

    args = parser.parse_args()

    # --- Logging ---
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, datefmt="%H:%M:%S")

    logger.info("=" * 60)
    logger.info("Visual Localization from ROS Bag (Real-Time)")
    logger.info("=" * 60)

    # --- Initialize detector ---
    logger.info("Initializing SuperPoint detector...")
    superpoint_config = SuperPointConfig(
        device=args.device,
        nms_radius=4,
        keypoint_threshold=0.01,
        max_keypoints=-1,
    )
    detector = SuperPointAlgorithm(superpoint_config)

    # --- Initialize matcher ---
    logger.info(f"Initializing {args.matcher} matcher...")
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

    # --- Initialize map reader ---
    logger.info(f"Loading satellite map from: {args.map_db}")
    map_reader = SatelliteMapReader(
        db_path=Path(args.map_db),
        resize_size=(args.resize_size,),
        logger=logging.getLogger("SatelliteMapReader"),
    )
    map_reader.initialize_db()
    map_reader.setup_db()
    map_reader.resize_db_images()
    map_reader.describe_db_images(detector)
    logger.info(f"Map loaded: {len(map_reader)} tiles")

    # --- Initialize query processor ---
    camera_model = CameraModel(
        focal_length=args.focal_length / 1000,  # mm to m
        resolution_height=args.resolution_height,
        resolution_width=args.resolution_width,
        hfov_deg=args.hfov_deg,
    )
    query_processor = QueryProcessor(
        processings=None,
        camera_model=camera_model,
        satellite_resolution=None,
        size=(args.resize_size,),
    )

    # --- Initialize pipeline ---
    pipeline_logger = logging.getLogger("Pipeline")
    pipeline_logger.setLevel(logging.DEBUG)
    pipeline = Pipeline(
        map_reader=map_reader,
        drone_streamer=None,  # Not used in run_on_image mode
        detector=detector,
        matcher=matcher,
        query_processor=query_processor,
        config=PipelineConfig(use_centroid=args.use_centroid),
        logger=pipeline_logger,
    )

    # --- Run localization ---
    localizer = RosbagLocalizer(
        bag_path=Path(args.bag),
        pipeline=pipeline,
        query_processor=query_processor,
        image_topic=args.image_topic,
        gnss_topic=args.gnss_topic,
        output_path=Path(args.output),
        max_images=args.max_images,
        realtime_factor=args.realtime_factor,
        skip_frames=args.skip_frames,
        gps_search_radius_m=args.gps_search_radius,
        no_gps_filter=args.no_gps_filter,
        rotate_cw=args.rotate_cw,
        full_map_path=Path(args.full_map) if args.draw_trajectory else None,
        max_distance_m=args.max_distance,
    )

    preds = localizer.run()

    # --- Summary metrics ---
    metrics = pipeline.compute_metrics(preds)
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total processed: {len(preds)}")
    logger.info(f"  Matches: {metrics['num_matches']}")
    logger.info(f"  Match ratio: {metrics['ratio_matches']:.2%}")
    if metrics["num_matches"] > 0:
        logger.info(f"  MAE: {metrics['mae']:.2f} m")
        logger.info(f"  Max distance: {metrics['max_distance']:.2f} m")
        logger.info(f"  Min distance: {metrics['min_distance']:.2f} m")


if __name__ == "__main__":
    main()
