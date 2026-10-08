import logging
import json
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
from svl.localization.geometry import (
    estimate_planar_pnp,
    query_points_to_camera_pixels,
    satellite_pixels_to_metric,
)
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

    def _query_rotation_for_image(self, image_name: str) -> float:
        if image_name in self.config.query_rotation_by_name:
            return self.config.query_rotation_by_name[image_name]
        try:
            idx_name = f"{int(image_name):06d}"
        except ValueError:
            idx_name = image_name
        return self.config.query_rotation_by_name.get(
            idx_name, self.config.query_rotation_cw
        )

    def _query_point_to_unrotated(
        self, point: np.ndarray, image_shape: tuple, rotation_cw: float
    ) -> tuple:
        """Map a point from the current query image back to the unrotated image."""
        x, y = float(point[0]), float(point[1])
        h, w = image_shape[:2]
        rot = rotation_cw % 360.0
        if abs(rot) < 1e-9 or abs(rot - 360.0) < 1e-9:
            return x, y, w, h
        if abs(rot - 90.0) < 1e-9:
            return y, float(w - 1) - x, h, w
        if abs(rot - 180.0) < 1e-9:
            return float(w - 1) - x, float(h - 1) - y, w, h
        if abs(rot - 270.0) < 1e-9:
            return float(h - 1) - y, x, h, w
        raise ValueError(
            f"Unsupported query_rotation_cw={rotation_cw}; "
            "only 0, 90, 180, and 270 are supported for exact pixel remapping."
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

        calibration = self.config.camera_calibration
        if calibration is None:
            raise ValueError(
                "PipelineConfig.camera_calibration is required for planar PnP"
            )
        query_rotation_cw = self._query_rotation_for_image(drone_image.name)
        best_candidate_score = None
        best_dst = None
        best_H = None
        best_planar_pose = None
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
        matched_homography_inliers = None
        matched_inlier_mask = None
        matched_pnp_inlier_mask = None
        matched_camera_pixels = None
        matched_query_indices = None
        matched_sat_indices = None
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
            query_indices = np.nonzero(valid)[0]
            sat_indices = matches[valid]

            if len(mkpts0) < 4:
                logging.debug(
                    f"Skipping image {satellite_image.name} not enough matches {len(mkpts0)}"
                )
                continue

            (
                ret,
                num_inliers,
                dst,
                H,
                inlier_mask,
            ) = self.estimate_and_apply_geometric_transform(
                mkpts0, mkpts1, drone_image.image.shape[:2]
            )

            if not ret:
                continue

            camera_pixels = query_points_to_camera_pixels(
                mkpts0,
                drone_image.image.shape[:2],
                query_rotation_cw,
                calibration,
            )
            ground_points = satellite_pixels_to_metric(
                mkpts1,
                satellite_image.image.shape[:2],
                satellite_image.top_left.lat,
                satellite_image.top_left.long,
                satellite_image.bottom_right.lat,
                satellite_image.bottom_right.long,
                self.config.reference_latitude,
                self.config.reference_longitude,
            )
            planar_pose = estimate_planar_pnp(
                camera_pixels,
                ground_points,
                calibration,
                min_inliers=self.config.pnp_min_inliers,
                min_inlier_ratio=self.config.pnp_min_inlier_ratio,
                ransac_threshold_px=self.config.pnp_ransac_threshold_px,
                min_bbox_area_ratio=self.config.pnp_min_bbox_area_ratio,
                min_camera_height_m=self.config.pnp_min_camera_height_m,
                max_camera_height_m=self.config.pnp_max_camera_height_m,
                max_optical_tilt_deg=self.config.pnp_max_optical_tilt_deg,
                tilt_regularization_px_per_deg=(
                    self.config.pnp_tilt_regularization_px_per_deg
                ),
            )
            if planar_pose is None:
                continue

            candidate_score = (
                planar_pose.inlier_count,
                planar_pose.inlier_ratio,
                -planar_pose.reprojection_mean_px,
            )
            if (
                best_candidate_score is not None
                and candidate_score <= best_candidate_score
            ):
                continue

            h, w = drone_image.image.shape[:2]
            if self.config.use_centroid:
                center_pt = np.float32([[np.mean(mkpts0, axis=0)]])
            else:
                center_pt = np.float32([[[w / 2, h / 2]]])
            denormalized_center_pt = cv2.perspectiveTransform(center_pt, H)[0][0]
            denormalized_center = (
                int(denormalized_center_pt[0]),
                int(denormalized_center_pt[1]),
            )

            best_candidate_score = candidate_score
            center = self.normalize_center(
                denormalized_center, satellite_image.image.shape
            )
            best_dst = dst
            best_H = H
            best_planar_pose = planar_pose
            best_denormalized_center = denormalized_center
            matched_image = satellite_image
            features_mean = np.mean(mkpts0, axis=0)
            matched_kpts0 = mkpts0
            matched_kpts1 = mkpts1
            matched_confidence = confidence
            matched_valid = valid
            matched_inliers = planar_pose.inlier_count
            matched_homography_inliers = int(num_inliers)
            matched_inlier_mask = inlier_mask
            matched_pnp_inlier_mask = planar_pose.inlier_mask
            matched_camera_pixels = camera_pixels
            matched_query_indices = query_indices
            matched_sat_indices = sat_indices
            saved_kpts1 = satellite_image.key_points.keypoints

        matched_keypoints = []
        planar_pose_dict = None
        look_at_coordinates = None
        if best_dst is not None and best_planar_pose is not None:
            planar_pose_dict = best_planar_pose.to_dict(
                self.config.reference_latitude,
                self.config.reference_longitude,
            )
            predicted_coordinates = GpsCoordinate(
                lat=planar_pose_dict["camera_lat"],
                long=planar_pose_dict["camera_lon"],
            )
            look_at_coordinates = self.compute_geo_pose(matched_image, center)

            distance = (
                haversine_distance(gt_coordinates, predicted_coordinates)
                if gt_coordinates
                else None
            )
            is_match = True
            color = cm.jet(matched_confidence[matched_valid])
            if output_path:
                output_path = (
                    Path(output_path) if isinstance(output_path, str) else output_path
                )
                viz_path = output_path / f"{drone_image.name}_viz.jpg"

                # Warp drone image onto satellite image, convert to gray for matching plot
                viz_drone_image = drone_image.image.copy()
                viz_sat_bgr = self.warp_drone_to_satellite(
                    matched_image.image, drone_image.image, best_H, alpha=0.6
                )
                viz_sat_gray = cv2.cvtColor(viz_sat_bgr, cv2.COLOR_BGR2GRAY)
                out = make_matching_plot_fast(
                    image0=viz_drone_image,
                    image1=viz_sat_gray,
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

                # Features mean on drone image (magenta ring) — removed
                # cx_d = int(features_mean[0])
                # cy_d = int(features_mean[1])
                # cv2.circle(out, (cx_d, cy_d), 10, (255, 0, 255), 5, lineType=cv2.LINE_AA)

                # Predicted look-at point on satellite image (magenta ring)
                cx_s, cy_s = best_denormalized_center
                cv2.circle(
                    out,
                    (cx_s + margin + W0, cy_s),
                    10,
                    (255, 0, 255),
                    5,
                    lineType=cv2.LINE_AA,
                )

                # Ground truth GPS point on satellite image (green ring)
                if gt_coordinates:
                    gt_pixel = self.gps_to_pixel(gt_coordinates, matched_image)
                    cv2.circle(
                        out,
                        (gt_pixel[0] + margin + W0, gt_pixel[1]),
                        10,
                        (0, 255, 0),
                        5,
                        lineType=cv2.LINE_AA,
                    )

                # Camera ground position from planar PnP (cyan ring).
                pnp_pixel = self.gps_to_pixel(predicted_coordinates, matched_image)
                cv2.circle(
                    out,
                    (pnp_pixel[0] + margin + W0, pnp_pixel[1]),
                    10,
                    (255, 255, 0),
                    5,
                    lineType=cv2.LINE_AA,
                )

                # Draw text overlay
                small_text = []
                if gt_coordinates:
                    small_text.append(f"GT: {gt_coordinates}")
                    small_text.append(f"PnP camera: {predicted_coordinates}")
                    small_text.append(f"Dist: {distance*1000:.2f}m")
                else:
                    small_text.append(f"PnP camera: {predicted_coordinates}")
                small_text.append(
                    f"PnP: {matched_inliers} inliers, "
                    f"{best_planar_pose.reprojection_mean_px:.2f}px"
                )
                Ht = int(min(out.shape[0] / 640.0, 2.0) * 30)
                for i, t in enumerate(small_text):
                    cv2.putText(
                        out,
                        t,
                        (8, Ht * (i + 1)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        out,
                        t,
                        (8, Ht * (i + 1)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.8,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                self.save_viz(out, viz_path)
            if gt_coordinates:
                self.logger.info(
                    f"Predicted coordinates: {predicted_coordinates}, GT coordinates: {gt_coordinates}"
                )
                self.logger.info(f"Haversine distance in meters: {distance * 1000}")
            else:
                self.logger.info(f"Predicted coordinates: {predicted_coordinates}")

            h_sat, w_sat = matched_image.image.shape[:2]
            h_query, w_query = drone_image.image.shape[:2]
            reprojected = cv2.perspectiveTransform(
                matched_kpts0.reshape(-1, 1, 2).astype(np.float32), best_H
            ).reshape(-1, 2)
            reproj_errors = np.linalg.norm(reprojected - matched_kpts1, axis=1)
            inlier_flags = matched_inlier_mask.ravel().astype(bool)
            confidence_values = matched_confidence[matched_valid]
            for i, (q_pt, s_pt) in enumerate(zip(matched_kpts0, matched_kpts1)):
                if not matched_pnp_inlier_mask[i]:
                    continue
                query_x, query_y = matched_camera_pixels[i]
                sat_norm_x = float(s_pt[0]) / float(w_sat)
                sat_norm_y = float(s_pt[1]) / float(h_sat)
                sat_lat = matched_image.top_left.lat + sat_norm_y * (
                    matched_image.bottom_right.lat - matched_image.top_left.lat
                )
                sat_lon = matched_image.top_left.long + sat_norm_x * (
                    matched_image.bottom_right.long - matched_image.top_left.long
                )
                matched_keypoints.append(
                    {
                        "query_idx": int(matched_query_indices[i]),
                        "sat_idx": int(matched_sat_indices[i]),
                        "query_x": float(query_x),
                        "query_y": float(query_y),
                        "query_rotated_x": float(q_pt[0]),
                        "query_rotated_y": float(q_pt[1]),
                        "query_rotation_cw": float(query_rotation_cw),
                        "query_input_width": int(w_query),
                        "query_input_height": int(h_query),
                        "query_unrotated_width": int(calibration.image_width),
                        "query_unrotated_height": int(calibration.image_height),
                        "sat_x": float(s_pt[0]),
                        "sat_y": float(s_pt[1]),
                        "sat_lat": float(sat_lat),
                        "sat_lon": float(sat_lon),
                        "confidence": float(confidence_values[i]),
                        "homography_reprojection_error": float(reproj_errors[i]),
                        "is_homography_inlier": bool(inlier_flags[i]),
                        "is_pnp_inlier": True,
                    }
                )
        else:
            self.logger.warning(f"No match found for {drone_image.name}")

        return {
            "is_match": is_match,
            "gt_coordinate": gt_coordinates,
            "gt_altitude": (
                drone_image.geo_point.altitude if drone_image.geo_point else None
            ),
            "timestamp": drone_image.timestamp,
            "predicted_coordinate": predicted_coordinates,
            "center": center,
            "look_at_coordinate": look_at_coordinates,
            "best_dst": best_dst,
            "num_inliers": matched_inliers,
            "num_homography_inliers": matched_homography_inliers,
            "matched_image": matched_image,
            "matched_keypoints": matched_keypoints,
            "planar_pose": planar_pose_dict,
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
        match_records = []
        num_matches = 0
        for drone_image in self.drone_streamer:
            query = self.query_processor(drone_image)

            candidate_indices = None
            if self.config.use_gt_tile_prior and query.geo_point is not None:
                drone_gps = GpsCoordinate(
                    lat=query.geo_point.latitude,
                    long=query.geo_point.longitude,
                )
                candidate_indices = self._find_nearby_tile_indices(
                    drone_gps, self.config.gt_tile_prior_radius_m
                )
                self.logger.info(
                    f"Image {drone_image.name}: {len(candidate_indices)} nearby tiles"
                )

            pred = self.run_on_image(query, output_path, candidate_indices)
            matched_image_name = (
                pred["matched_image"].name if pred["matched_image"] else None
            )
            if pred["is_match"]:
                record = {
                    "frame": drone_image.name,
                    "timestamp": pred["timestamp"],
                    "matched_image": matched_image_name,
                    "query_rotation_cw": float(
                        self._query_rotation_for_image(drone_image.name)
                    ),
                    "query_coordinate_frame": "unrotated_original",
                    "num_matches": len(pred.get("matched_keypoints", [])),
                    "num_inliers": int(pred["num_inliers"])
                    if pred["num_inliers"] is not None
                    else None,
                    "num_homography_inliers": pred["num_homography_inliers"],
                    "predicted_lat": pred["predicted_coordinate"].lat
                    if pred["predicted_coordinate"]
                    else None,
                    "predicted_lon": pred["predicted_coordinate"].long
                    if pred["predicted_coordinate"]
                    else None,
                    "distance_m": pred["distance"],
                    "planar_pose": pred["planar_pose"],
                    "keypoints": pred.get("matched_keypoints", []),
                }
                if pred["look_at_coordinate"]:
                    record["look_at_lat"] = pred["look_at_coordinate"].lat
                    record["look_at_lon"] = pred["look_at_coordinate"].long
                if pred["gt_coordinate"]:
                    record["gt_lat"] = pred["gt_coordinate"].lat
                    record["gt_lon"] = pred["gt_coordinate"].long
                    record["gt_altitude"] = pred["gt_altitude"]
                match_records.append(record)
            pred["matched_image"] = matched_image_name
            preds.append(pred)
            num_matches += pred["is_match"]
            if (
                pred["is_match"]
                and pred["distance"] is not None
                and pred["distance"] > 50
            ):
                self.logger.warning(
                    f"Large distance: {pred['distance']} for {drone_image.name}"
                )
        self.logger.info(
            f"Number of matches: {num_matches} among {len(self.drone_streamer)} images"
        )
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            matches_path = output_path / "matched_keypoints.json"
            with open(matches_path, "w") as f:
                json.dump(match_records, f, indent=2)
            self.logger.info(f"Saved matched keypoints to {matches_path}")
        return preds
