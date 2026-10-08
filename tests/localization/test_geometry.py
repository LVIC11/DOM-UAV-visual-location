import cv2
import numpy as np

from svl.localization.geometry import (
    CameraCalibration,
    estimate_planar_pnp,
    gps_to_metric_xy,
    metric_xy_to_gps,
    query_points_to_camera_pixels,
)


def make_calibration():
    return CameraCalibration(
        camera_matrix=np.array(
            [[820.2, 0.0, 642.5], [0.0, 823.9, 370.6], [0.0, 0.0, 1.0]]
        ),
        distortion_coeffs=np.array([0.0287, -0.0560, 0.00218, 0.00096, 0.0]),
        image_width=1280,
        image_height=720,
    )


def test_query_rotation_is_mapped_back_to_camera_pixels():
    calibration = make_calibration()
    original = np.array([[100.0, 200.0], [1100.0, 600.0]])
    rotated_cw_90 = np.column_stack([719.0 - original[:, 1], original[:, 0]])

    recovered = query_points_to_camera_pixels(
        rotated_cw_90, (1280, 720), 90.0, calibration
    )

    np.testing.assert_allclose(recovered, original, atol=1e-9)


def test_planar_pnp_recovers_camera_pose_with_outliers():
    calibration = make_calibration()
    rng = np.random.default_rng(4)
    ground_xy = np.column_stack(
        [rng.uniform(-50.0, 70.0, 100), rng.uniform(-45.0, 65.0, 100)]
    )
    expected_position = np.array([8.0, 12.0, 130.0])
    rotation_camera_to_world = np.diag([1.0, -1.0, -1.0])
    rotation_world_to_camera = rotation_camera_to_world.T
    translation = -rotation_world_to_camera @ expected_position
    rvec, _ = cv2.Rodrigues(rotation_world_to_camera)
    object_points = np.column_stack([ground_xy, np.zeros(len(ground_xy))])
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        translation.reshape(3, 1),
        calibration.camera_matrix,
        calibration.distortion_coeffs,
    )
    image_points = image_points.reshape(-1, 2)
    image_points += rng.normal(0.0, 0.35, image_points.shape)
    outliers = rng.choice(len(image_points), 10, replace=False)
    image_points[outliers] = rng.uniform([0.0, 0.0], [1280.0, 720.0], (10, 2))

    pose = estimate_planar_pnp(
        image_points,
        ground_xy,
        calibration,
        min_inliers=20,
        min_bbox_area_ratio=0.03,
    )

    assert pose is not None
    assert pose.inlier_count >= 85
    assert pose.reprojection_mean_px < 1.0
    assert pose.optical_tilt_deg < 1.0
    assert np.linalg.norm(pose.camera_position_enu - expected_position) < 0.5


def test_local_metric_coordinate_round_trip():
    ref_lat, ref_lon = 31.8325939032, 118.71416416
    expected_lat, expected_lon = 31.8331, 118.7152
    east, north = gps_to_metric_xy(expected_lat, expected_lon, ref_lat, ref_lon)
    actual_lat, actual_lon = metric_xy_to_gps(east, north, ref_lat, ref_lon)

    assert abs(actual_lat - expected_lat) < 1e-12
    assert abs(actual_lon - expected_lon) < 1e-12
