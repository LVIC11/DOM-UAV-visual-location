"""Geometry helpers for metric satellite-image localization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


METERS_PER_DEGREE_LATITUDE = 111320.0


@dataclass
class CameraCalibration:
    """Pinhole calibration in the original, unrotated camera image frame."""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(
            3, 3
        )
        self.distortion_coeffs = np.asarray(
            self.distortion_coeffs, dtype=np.float64
        ).reshape(-1, 1)
        self.image_width = int(self.image_width)
        self.image_height = int(self.image_height)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Camera image dimensions must be positive")
        if not np.isfinite(self.camera_matrix).all():
            raise ValueError("Camera matrix contains non-finite values")
        if self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0:
            raise ValueError("Camera focal lengths must be positive")

    @classmethod
    def from_opencv_yaml(cls, yaml_path: str) -> "CameraCalibration":
        """Load the camera YAML format used by ``nav_left.yaml``."""

        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Camera calibration not found: {path}")

        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise ValueError(f"Could not open camera calibration: {path}")
        try:
            projection = storage.getNode("projection_parameters")
            distortion = storage.getNode("distortion_parameters")
            values = {
                name: projection.getNode(name).real()
                for name in ("fx", "fy", "cx", "cy")
            }
            dist = [distortion.getNode(name).real() for name in ("k1", "k2", "p1", "p2")]
            width = int(storage.getNode("image_width").real())
            height = int(storage.getNode("image_height").real())
        finally:
            storage.release()

        camera_matrix = np.array(
            [
                [values["fx"], 0.0, values["cx"]],
                [0.0, values["fy"], values["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # nav_left.yaml uses the radial-tangential order k1, k2, p1, p2.
        return cls(camera_matrix, np.array(dist + [0.0]), width, height)


@dataclass
class PlanarPoseEstimate:
    """Camera pose estimated from image-to-ground-plane correspondences."""

    camera_position_enu: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    inlier_ratio: float
    reprojection_mean_px: float
    reprojection_median_px: float
    reprojection_max_px: float
    bbox_area_ratio: float
    optical_axis_enu: np.ndarray
    optical_tilt_deg: float
    optical_azimuth_deg: float
    euler_enu_zyx_deg: np.ndarray
    quaternion_xyzw: np.ndarray

    @property
    def rotation_camera_to_world(self) -> np.ndarray:
        return self.rotation_world_to_camera.T

    def to_dict(self, ref_lat: float, ref_lon: float) -> Dict[str, Any]:
        lat, lon = metric_xy_to_gps(
            float(self.camera_position_enu[0]),
            float(self.camera_position_enu[1]),
            ref_lat,
            ref_lon,
        )
        return {
            "coordinate_frame": "ENU",
            "reference_latitude": float(ref_lat),
            "reference_longitude": float(ref_lon),
            "camera_lat": float(lat),
            "camera_lon": float(lon),
            "camera_position_enu": self.camera_position_enu.astype(float).tolist(),
            "rotation_world_to_camera": self.rotation_world_to_camera.astype(
                float
            ).tolist(),
            "rotation_camera_to_enu": self.rotation_camera_to_world.astype(
                float
            ).tolist(),
            "translation_world_to_camera": (
                self.translation_world_to_camera.astype(float).tolist()
            ),
            "quaternion_xyzw": self.quaternion_xyzw.astype(float).tolist(),
            "euler_enu_zyx_deg": self.euler_enu_zyx_deg.astype(float).tolist(),
            "optical_axis_enu": self.optical_axis_enu.astype(float).tolist(),
            "optical_tilt_deg": float(self.optical_tilt_deg),
            "optical_azimuth_deg": float(self.optical_azimuth_deg),
            "pnp_inliers": int(self.inlier_count),
            "pnp_inlier_ratio": float(self.inlier_ratio),
            "pnp_reprojection_mean_px": float(self.reprojection_mean_px),
            "pnp_reprojection_median_px": float(self.reprojection_median_px),
            "pnp_reprojection_max_px": float(self.reprojection_max_px),
            "pnp_bbox_area_ratio": float(self.bbox_area_ratio),
        }


def gps_to_metric_xy(
    lat: float, lon: float, ref_lat: float, ref_lon: float
) -> Tuple[float, float]:
    """Convert WGS84 latitude/longitude to a local ENU approximation in meters."""

    lat_mid = 0.5 * (lat + ref_lat)
    meters_per_degree_lon = METERS_PER_DEGREE_LATITUDE * np.cos(np.radians(lat_mid))
    east = (lon - ref_lon) * meters_per_degree_lon
    north = (lat - ref_lat) * METERS_PER_DEGREE_LATITUDE
    return float(east), float(north)


def metric_xy_to_gps(
    east: float, north: float, ref_lat: float, ref_lon: float
) -> Tuple[float, float]:
    """Convert local ENU east/north coordinates back to latitude/longitude."""

    lat = ref_lat + north / METERS_PER_DEGREE_LATITUDE
    lat_mid = 0.5 * (lat + ref_lat)
    meters_per_degree_lon = METERS_PER_DEGREE_LATITUDE * np.cos(np.radians(lat_mid))
    lon = ref_lon + east / meters_per_degree_lon
    return float(lat), float(lon)


def satellite_pixels_to_metric(
    pixels: np.ndarray,
    image_shape: Tuple[int, int],
    top_left_lat: float,
    top_left_lon: float,
    bottom_right_lat: float,
    bottom_right_lon: float,
    ref_lat: float,
    ref_lon: float,
) -> np.ndarray:
    """Map satellite-image pixels to local ENU east/north coordinates."""

    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Satellite image dimensions must be positive")

    lon = top_left_lon + points[:, 0] / float(width) * (bottom_right_lon - top_left_lon)
    lat = top_left_lat + points[:, 1] / float(height) * (bottom_right_lat - top_left_lat)
    north = (lat - ref_lat) * METERS_PER_DEGREE_LATITUDE
    lat_mid = 0.5 * (lat + ref_lat)
    east = (lon - ref_lon) * METERS_PER_DEGREE_LATITUDE * np.cos(np.radians(lat_mid))
    return np.column_stack([east, north]).astype(np.float64)


def query_points_to_camera_pixels(
    points: np.ndarray,
    query_image_shape: Tuple[int, int],
    rotation_cw: float,
    calibration: CameraCalibration,
) -> np.ndarray:
    """Undo a cardinal query rotation and map pixels to calibration resolution."""

    query_points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    height, width = query_image_shape[:2]
    rotation = float(rotation_cw) % 360.0

    if np.isclose(rotation, 0.0) or np.isclose(rotation, 360.0):
        unrotated = query_points.copy()
        unrotated_width, unrotated_height = width, height
    elif np.isclose(rotation, 90.0):
        unrotated = np.column_stack(
            [query_points[:, 1], (width - 1.0) - query_points[:, 0]]
        )
        unrotated_width, unrotated_height = height, width
    elif np.isclose(rotation, 180.0):
        unrotated = np.column_stack(
            [
                (width - 1.0) - query_points[:, 0],
                (height - 1.0) - query_points[:, 1],
            ]
        )
        unrotated_width, unrotated_height = width, height
    elif np.isclose(rotation, 270.0):
        unrotated = np.column_stack(
            [(height - 1.0) - query_points[:, 1], query_points[:, 0]]
        )
        unrotated_width, unrotated_height = height, width
    else:
        raise ValueError(
            "Planar PnP requires an exact inverse pixel mapping; query rotation "
            "must be one of 0, 90, 180, or 270 degrees"
        )

    scale_x = (calibration.image_width - 1.0) / max(unrotated_width - 1.0, 1.0)
    scale_y = (calibration.image_height - 1.0) / max(unrotated_height - 1.0, 1.0)
    unrotated[:, 0] *= scale_x
    unrotated[:, 1] *= scale_y
    return unrotated


def rotation_matrix_to_euler_zyx_deg(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw for ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    horizontal = np.hypot(matrix[0, 0], matrix[1, 0])
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arctan2(-matrix[2, 0], horizontal)
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.degrees([roll, pitch, yaw])


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to an ``[x, y, z, w]`` quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / root,
                (matrix[0, 2] - matrix[2, 0]) / root,
                (matrix[1, 0] - matrix[0, 1]) / root,
                0.25 * root,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            root = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    0.25 * root,
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    (matrix[0, 2] + matrix[2, 0]) / root,
                    (matrix[2, 1] - matrix[1, 2]) / root,
                ]
            )
        elif index == 1:
            root = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    0.25 * root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                    (matrix[0, 2] - matrix[2, 0]) / root,
                ]
            )
        else:
            root = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                    0.25 * root,
                    (matrix[1, 0] - matrix[0, 1]) / root,
                ]
            )
    return quat / np.linalg.norm(quat)


def estimate_planar_pnp(
    image_points: np.ndarray,
    ground_points_xy: np.ndarray,
    calibration: CameraCalibration,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.35,
    ransac_threshold_px: float = 6.0,
    min_bbox_area_ratio: float = 0.03,
    min_camera_height_m: float = 5.0,
    max_camera_height_m: float = 1000.0,
    max_optical_tilt_deg: float = 30.0,
    tilt_regularization_px_per_deg: float = 0.05,
) -> Optional[PlanarPoseEstimate]:
    """Estimate full camera pose from 2D pixels and coplanar ENU points.

    IPPE explicitly returns the two planar pose hypotheses. Candidates are refined
    with Levenberg-Marquardt and selected only from physically valid solutions.
    """

    pixels_all = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    ground_all = np.asarray(ground_points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pixels_all) != len(ground_all):
        raise ValueError("Image and ground correspondence counts differ")

    valid = np.isfinite(pixels_all).all(axis=1) & np.isfinite(ground_all).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    pixels = pixels_all[valid]
    ground = ground_all[valid]
    required = max(4, int(min_inliers))
    if len(pixels) < required:
        return None

    object_origin = np.mean(ground, axis=0)
    local_ground = ground - object_origin
    object_points = np.column_stack(
        [local_ground, np.zeros(len(local_ground), dtype=np.float64)]
    )

    try:
        _, homography_mask = cv2.findHomography(
            local_ground,
            pixels,
            cv2.RANSAC,
            float(ransac_threshold_px),
            maxIters=3000,
            confidence=0.999,
        )
    except cv2.error:
        return None
    if homography_mask is None:
        return None
    seed_mask = homography_mask.ravel().astype(bool)
    if int(seed_mask.sum()) < required or float(seed_mask.mean()) < float(
        min_inlier_ratio
    ):
        return None

    seed_objects = object_points[seed_mask]
    seed_pixels = pixels[seed_mask]
    try:
        pnp_output = cv2.solvePnPGeneric(
            seed_objects,
            seed_pixels,
            calibration.camera_matrix,
            calibration.distortion_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        solved = bool(pnp_output[0])
        rvecs = list(pnp_output[1]) if solved else []
        tvecs = list(pnp_output[2]) if solved else []
    except cv2.error:
        rvecs, tvecs = [], []

    if not rvecs:
        try:
            solved, rvec, tvec = cv2.solvePnP(
                seed_objects,
                seed_pixels,
                calibration.camera_matrix,
                calibration.distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            solved = False
        if solved:
            rvecs, tvecs = [rvec], [tvec]
    if not rvecs:
        return None

    best_estimate = None
    best_score = float("inf")
    for initial_rvec, initial_tvec in zip(rvecs, tvecs):
        rvec = np.asarray(initial_rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(initial_tvec, dtype=np.float64).reshape(3, 1).copy()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                seed_objects,
                seed_pixels,
                calibration.camera_matrix,
                calibration.distortion_coeffs,
                rvec,
                tvec,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    50,
                    1e-8,
                ),
            )
        except cv2.error:
            continue

        def collect_inliers() -> Tuple[np.ndarray, np.ndarray]:
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                calibration.camera_matrix,
                calibration.distortion_coeffs,
            )
            errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
            return errors <= float(ransac_threshold_px), errors

        try:
            pnp_inliers, reprojection_errors = collect_inliers()
        except cv2.error:
            continue
        if int(pnp_inliers.sum()) < required or float(pnp_inliers.mean()) < float(
            min_inlier_ratio
        ):
            continue

        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[pnp_inliers],
                pixels[pnp_inliers],
                calibration.camera_matrix,
                calibration.distortion_coeffs,
                rvec,
                tvec,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    100,
                    1e-9,
                ),
            )
            pnp_inliers, reprojection_errors = collect_inliers()
        except cv2.error:
            continue
        if int(pnp_inliers.sum()) < required:
            continue

        rotation_world_to_camera, _ = cv2.Rodrigues(rvec)
        camera_points = (
            rotation_world_to_camera @ object_points[pnp_inliers].T + tvec
        ).T
        if not np.all(camera_points[:, 2] > 0.0):
            continue

        camera_local = (-rotation_world_to_camera.T @ tvec).reshape(3)
        camera_position = camera_local.copy()
        camera_position[:2] += object_origin
        rotation_camera_to_world = rotation_world_to_camera.T
        optical_axis = rotation_camera_to_world[:, 2]
        if (
            not np.isfinite(camera_position).all()
            or camera_position[2] < float(min_camera_height_m)
            or camera_position[2] > float(max_camera_height_m)
            or optical_axis[2] >= -0.05
        ):
            continue

        optical_tilt = np.degrees(
            np.arccos(np.clip(np.dot(optical_axis, [0.0, 0.0, -1.0]), -1.0, 1.0))
        )
        if optical_tilt > float(max_optical_tilt_deg):
            continue

        inlier_pixels = pixels[pnp_inliers]
        extent = np.ptp(inlier_pixels, axis=0)
        bbox_ratio = float(
            extent[0]
            * extent[1]
            / max(1.0, calibration.image_width * calibration.image_height)
        )
        if bbox_ratio < float(min_bbox_area_ratio):
            continue

        errors = reprojection_errors[pnp_inliers]
        mean_error = float(np.mean(errors))
        median_error = float(np.median(errors))
        max_error = float(np.max(errors))
        # Coplanar points admit two IPPE pose hypotheses whose reprojection
        # errors can be almost identical.  For a downward-looking UAV camera,
        # a weak nadir prior prevents selecting the high-tilt mirror solution
        # solely because it is a fraction of a pixel better.
        score = (
            median_error
            + 0.25 * mean_error
            + 0.01 * max_error
            + float(tilt_regularization_px_per_deg) * float(optical_tilt)
        )
        if score >= best_score:
            continue

        full_inlier_mask = np.zeros(len(pixels_all), dtype=bool)
        full_inlier_mask[valid_indices[pnp_inliers]] = True
        euler = rotation_matrix_to_euler_zyx_deg(rotation_camera_to_world)
        quaternion = rotation_matrix_to_quaternion_xyzw(rotation_camera_to_world)
        optical_azimuth = np.degrees(np.arctan2(optical_axis[0], optical_axis[1]))
        best_estimate = PlanarPoseEstimate(
            camera_position_enu=camera_position,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=tvec.reshape(3),
            inlier_mask=full_inlier_mask,
            inlier_count=int(pnp_inliers.sum()),
            inlier_ratio=float(pnp_inliers.mean()),
            reprojection_mean_px=mean_error,
            reprojection_median_px=median_error,
            reprojection_max_px=max_error,
            bbox_area_ratio=bbox_ratio,
            optical_axis_enu=optical_axis,
            optical_tilt_deg=float(optical_tilt),
            optical_azimuth_deg=float(optical_azimuth),
            euler_enu_zyx_deg=euler,
            quaternion_xyzw=quaternion,
        )
        best_score = score

    return best_estimate
