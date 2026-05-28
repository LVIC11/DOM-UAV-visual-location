import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

from superglue_lib.models.utils import make_matching_plot_fast
from svl.keypoint_pipeline.base import CombinedKeyPointAlgorithm
from svl.keypoint_pipeline.matcher import KeyPointMatcher
from svl.localization.base import BasePipeline, PipelineConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import DroneImage, GeoSatelliteImage
from svl.tms.geo import haversine_distance
from svl.tms.schemas import GpsCoordinate


class Pipeline(BasePipeline):
    """Class to run the localization pipeline based on keypoint matching.

    The pipeline consists of the following steps:
    1. Detect keypoints in the drone image
    2. Match the keypoints with the keypoints in the satellite image
    3. Estimate the affine transform between the matched keypoints
    4. Compute the center of the affine transform
    5. Compute the predicted GPS coordinates based on the center of the affine transform
    6. Compute the haversine distance between the predicted GPS coordinates and the
       ground truth GPS coordinates
    7. Save the visualization of the matching results


    Parameters
    ----------
    map_reader : SatelliteMapReader
        the map reader to read the satellite images, stored in a database
    drone_streamer : DroneImageStreamer
        the drone image streamer to read the drone images
    detector : CombinedKeyPointAlgorithm
        the keypoint detector to accomplish the detection and description of keypoints
    matcher : KeyPointMatcher
        the keypoint matcher to accomplish the matching of keypoints
    config : PipelineConfig
        the configuration of the pipeline
    query_processor : QueryProcessor
        the query processor to preprocess the query image (resize, warp, etc.)
    logger : logging.Logger
        the logger to use for logging
    """

    def __init__(
        self,
        map_reader: SatelliteMapReader,
        drone_streamer: DroneImageStreamer,
        detector: CombinedKeyPointAlgorithm,
        matcher: KeyPointMatcher,
        config: PipelineConfig,
        query_processor: QueryProcessor,
        logger: logging.Logger,
    ) -> None:

        super().__init__(
            map_reader=map_reader,
            drone_streamer=drone_streamer,
            detector=detector,
            matcher=matcher,
            config=config,
            query_processor=query_processor,
            logger=logger,
        )

    def run_on_image(
        self,
        drone_image: DroneImage,
        output_path: Union[str, Path] = None,
        candidate_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline on a single drone image.

        Parameters
        ----------
        drone_image : DroneImage
            the drone image to process
        output_path : Union[str, Path]
            the output path to save the visualization

        Returns
        -------
        Dict[str, Any]
            the prediction results with the following keys:
            - is_match: bool
                whether a match was found
            - gt_coordinate: GpsCoordinate
                the ground truth GPS coordinate
            - predicted_coordinate: GpsCoordinate
                the predicted GPS coordinate
            - center: Tuple[int, int]
                the center of the affine transform
            - best_dst: np.ndarray
                the affine transform matrix
            - matched_image: str
                the name of the matched satellite image
            - distance: float
                the haversine distance between the predicted and ground truth GPS
                coordinates
        """
        self.logger.info(f"Processing image {drone_image.name}")

        max_macthes = -1
        best_dst = None
        best_denormalized_center = None
        matched_image = None
        is_match = False
        predicted_coordinates = None
        center = None
        distance = None
        matched_kpts0 = None
        matched_kpts1 = None
        features_mean = None
        matched_confidence = None
        matched_valid = None
        matched_inliers = None
        saved_kpts1 = None

        drone_image.key_points = self.detector.detect_and_describe_keypoints(
            drone_image.image
        )
        gt_coordinates = None
        if drone_image.geo_point is not None:
            gt_coordinates = GpsCoordinate(
                lat=drone_image.geo_point.latitude,
                long=drone_image.geo_point.longitude,
            )

        match_indices = (
            candidate_indices
            if candidate_indices is not None
            else range(len(self.map_reader))
        )
        for idx in tqdm(
            match_indices,
            desc="Matching images",
            total=len(list(match_indices)),
        ):
            satellite_image: GeoSatelliteImage = self.map_reader[idx]

            # Match the keypoints
            matches, confidence = self.matcher.match_keypoints(
                drone_image.key_points, satellite_image.key_points
            )
            valid = matches > -1
            mkpts0 = drone_image.key_points.keypoints[valid]
            mkpts1 = satellite_image.key_points.keypoints[matches[valid]]

            if len(mkpts0) < 4:
                logging.debug(
                    f"Skipping image {satellite_image.name} not enough matches {len(mkpts0)}"
                )
                continue

            ret, num_inliers, dst, H = self.estimate_and_apply_geometric_transform(
                mkpts0, mkpts1, drone_image.image.shape[:2]
            )

            if ret and len(mkpts1) > max_macthes:
                h, w = drone_image.image.shape[:2]
                center_pt = np.float32([[[w / 2, h / 2]]])
                denormalized_center_pt = cv2.perspectiveTransform(center_pt, H)[0][0]
                denormalized_center = (
                    int(denormalized_center_pt[0]),
                    int(denormalized_center_pt[1]),
                )

                max_macthes = len(mkpts1)
                center = self.normalize_center(
                    denormalized_center, satellite_image.image.shape
                )
                if center[0] < 0 or center[0] > 1 or center[1] < 0 or center[1] > 1:
                    continue

                best_dst = dst
                best_denormalized_center = denormalized_center
                matched_image = satellite_image
                features_mean = np.mean(mkpts0, axis=0)
                matched_kpts0 = mkpts0
                matched_kpts1 = mkpts1
                matched_confidence = confidence
                matched_valid = valid
                matched_inliers = num_inliers
                saved_kpts1 = satellite_image.key_points.keypoints

        if best_dst is not None:
            predicted_coordinates = self.compute_geo_pose(matched_image, center)

            distance = haversine_distance(gt_coordinates, predicted_coordinates) if gt_coordinates else None
            is_match = True
            color = cm.jet(matched_confidence[matched_valid])
            if output_path:
                output_path = (
                    Path(output_path) if isinstance(output_path, str) else output_path
                )
                viz_path = output_path / f"{drone_image.name}_viz.jpg"

                # Draw white polygon on grayscale satellite image (white shows on gray)
                viz_satellite_image = matched_image.image.copy()
                viz_satellite_image = self.draw_transform_polygon_on_image(
                    viz_satellite_image, best_dst
                )

                # Build matching plot from raw grayscale images
                viz_drone_image = drone_image.image.copy()
                out = make_matching_plot_fast(
                    image0=viz_drone_image,
                    image1=viz_satellite_image,
                    kpts0=drone_image.key_points.keypoints,
                    kpts1=saved_kpts1,
                    mkpts0=matched_kpts0,
                    mkpts1=matched_kpts1,
                    color=color,
                    text="",
                    path=None,
                    show_keypoints=True,
                    small_text=[],
                )

                # Draw colored annotations on the BGR canvas AFTER make_matching_plot_fast
                margin = 10
                W0 = drone_image.image.shape[1]

                # Features mean on drone image (magenta ring)
                cx_d = int(features_mean[0])
                cy_d = int(features_mean[1])
                cv2.circle(out, (cx_d, cy_d), 10, (255, 0, 255), 5, lineType=cv2.LINE_AA)

                # Predicted look-at point on satellite image (magenta ring)
                cx_s, cy_s = best_denormalized_center
                cv2.circle(
                    out, (cx_s + margin + W0, cy_s), 10,
                    (255, 0, 255), 5, lineType=cv2.LINE_AA,
                )

                # Ground truth GPS point on satellite image (green ring)
                if gt_coordinates:
                    gt_pixel = self.gps_to_pixel(gt_coordinates, matched_image)
                    cv2.circle(
                        out, (gt_pixel[0] + margin + W0, gt_pixel[1]), 10,
                        (0, 255, 0), 5, lineType=cv2.LINE_AA,
                    )

                # Draw text overlay
                small_text = []
                if gt_coordinates:
                    small_text.append(f"GT: {gt_coordinates}")
                    small_text.append(f"Pred: {predicted_coordinates}")
                    small_text.append(f"Dist: {distance*1000:.2f}m")
                else:
                    small_text.append(f"Pred: {predicted_coordinates}")
                Ht = int(min(out.shape[0] / 640., 2.0) * 30)
                for i, t in enumerate(small_text):
                    cv2.putText(out, t, (8, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                                0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(out, t, (8, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                                0.8, (255, 255, 255), 1, cv2.LINE_AA)

                self.save_viz(out, viz_path)
            if gt_coordinates:
                self.logger.info(
                    f"Predicted coordinates: {predicted_coordinates}, GT coordinates: {gt_coordinates}"
                )
                self.logger.info(f"Haversine distance in meters: {distance * 1000}")
            else:
                self.logger.info(f"Predicted coordinates: {predicted_coordinates}")
        else:
            self.logger.warning(f"No match found for {drone_image.name}")

        return {
            "is_match": is_match,
            "gt_coordinate": gt_coordinates,
            "predicted_coordinate": predicted_coordinates,
            "center": center,
            "best_dst": best_dst,
            "num_inliers": matched_inliers,
            "matched_image": matched_image,
            "distance": distance * 1000 if distance else None,
        }

    def _find_nearby_tile_indices(
        self, drone_gps: GpsCoordinate, radius_m: float = 150.0
    ) -> List[int]:
        """Find tile indices whose center is within radius_m of drone_gps."""
        nearby = []
        for idx in range(len(self.map_reader)):
            tile = self.map_reader[idx]
            if hasattr(tile, "top_left") and hasattr(tile, "bottom_right"):
                center_lat = (tile.top_left.lat + tile.bottom_right.lat) / 2
                center_long = (tile.top_left.long + tile.bottom_right.long) / 2
            elif hasattr(tile, "tile"):
                from svl.tms.geo import get_lat_long_from_tile_xy

                tl_lat, tl_long = get_lat_long_from_tile_xy(
                    tile.tile.x, tile.tile.y, tile.tile.zoom_level
                )
                br_lat, br_long = get_lat_long_from_tile_xy(
                    tile.tile.x + 1, tile.tile.y + 1, tile.tile.zoom_level
                )
                center_lat = (tl_lat + br_lat) / 2
                center_long = (tl_long + br_long) / 2
            else:
                nearby.append(idx)
                continue
            tile_center = GpsCoordinate(lat=center_lat, long=center_long)
            dist = haversine_distance(drone_gps, tile_center)
            if dist * 1000 <= radius_m:
                nearby.append(idx)
        return nearby

    def run(self, output_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """Run the pipeline on all drone images.

        Parameters
        ----------
        output_path : Union[str, Path]
            the output path to save the visualization

        Returns
        -------
        List[Dict[str, Any]]
            the prediction results for all drone images
        """
        self.logger.info(f"Running the pipeline on {len(self.drone_streamer)} images")
        preds = []
        num_matches = 0
        for drone_image in self.drone_streamer:
            query = self.query_processor(drone_image)

            candidate_indices = None
            if query.geo_point is not None:
                drone_gps = GpsCoordinate(
                    lat=query.geo_point.latitude,
                    long=query.geo_point.longitude,
                )
                candidate_indices = self._find_nearby_tile_indices(drone_gps)
                self.logger.info(
                    f"Image {drone_image.name}: {len(candidate_indices)} nearby tiles"
                )

            pred = self.run_on_image(query, output_path, candidate_indices)
            pred["matched_image"] = (
                pred["matched_image"].name if pred["matched_image"] else None
            )
            preds.append(pred)
            num_matches += pred["is_match"]
            if pred["is_match"] and pred["distance"] is not None and pred["distance"] > 50:
                self.logger.warning(
                    f"Large distance: {pred['distance']} for {drone_image.name}"
                )
        self.logger.info(
            f"Number of matches: {num_matches} among {len(self.drone_streamer)} images"
        )
        return preds
