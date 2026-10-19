from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from svl.localization.geometry import CameraCalibration, estimate_planar_pnp
from svl.localization.rotation_search import (
    RotationCandidate,
    export_match,
    fit_candidate_pose,
    rotate_query,
    rotation_geometry,
    transform_points,
)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_physical_cardinal_rotation_is_exact(angle):
    image = np.arange(6 * 9, dtype=np.uint8).reshape(6, 9)
    result, affine = rotate_query(image, angle)
    assert np.array_equal(result, np.rot90(image, -angle // 90))
    points = np.array([[1, 2], [6, 4]])
    pixels = transform_points(points, affine).round().astype(int)
    assert np.array_equal(
        result[pixels[:, 1], pixels[:, 0]], image[points[:, 1], points[:, 0]]
    )


@pytest.mark.parametrize("angle", [0, 25, 90, 137, 180, 270, 355])
def test_general_rotation_resize_returns_original_camera_pixels(angle):
    image = np.zeros((720, 1280), np.uint8)
    points = np.array([[0, 0], [1279, 719], [640.2, 370.4]])
    query, affine = rotate_query(image, angle, 480)
    processed = transform_points(points, affine)
    assert np.all(processed >= -1e-8)
    assert np.all(processed <= np.array([query.shape[1] - 1, query.shape[0] - 1]) + 1e-8)
    np.testing.assert_allclose(
        transform_points(processed, np.linalg.inv(affine)), points, atol=1e-9
    )


def test_no_match_and_invalid_geometry_are_explicit():
    assert transform_points([], np.eye(3)).shape == (0, 2)
    with pytest.raises(ValueError):
        rotation_geometry((1, 1280), 90)
    with pytest.raises(ValueError):
        rotate_query(np.zeros((8, 8)), float("nan"))
    with pytest.raises(ValueError, match="unverified"):
        export_match(None, "000004", 1.0, None, None)


def test_arbitrary_query_angle_does_not_change_planar_pnp_pose():
    rng = np.random.default_rng(4)
    k = np.array([[820.0, 0, 642.0], [0, 824.0, 370.0], [0, 0, 1.0]])
    calibration = CameraCalibration(k, np.zeros(5), 1280, 720)
    ground = rng.uniform([-35, -20], [35, 20], (70, 2))
    position = np.array([8.0, 12.0, 100.0])
    world_camera = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(world_camera)
    pixels, _ = cv2.projectPoints(
        np.column_stack([ground, np.zeros(len(ground))]),
        rvec,
        -world_camera @ position,
        k,
        np.zeros(5),
    )
    raw = pixels.reshape(-1, 2)
    _, affine = rotate_query(np.zeros((720, 1280), np.uint8), 73.0, 480)
    recovered = transform_points(transform_points(raw, affine), np.linalg.inv(affine))
    pose = estimate_planar_pnp(recovered, ground, calibration)
    assert pose is not None
    np.testing.assert_allclose(pose.camera_position_enu, position, atol=0.01)


def test_homography_prefilter_keeps_original_indices_and_expands_pnp_mask(monkeypatch):
    from svl.localization.base import PipelineConfig
    from svl.localization import rotation_search

    raw = np.array([[5, 5], [15, 6], [25, 5], [5, 15], [15, 25], [25, 20]], dtype=float)
    tile = SimpleNamespace(
        name="tile",
        image=np.zeros((40, 40), np.uint8),
        top_left=SimpleNamespace(lat=31.833, long=118.714),
        bottom_right=SimpleNamespace(lat=31.832, long=118.715),
    )
    candidate = RotationCandidate(
        0.0,
        40,
        tile,
        tile.image,
        np.eye(3),
        raw,
        raw,
        raw,
        np.array([11, 22, 33, 44, 55, 66]),
        np.arange(6),
        np.ones(6),
        np.eye(3),
        np.array([True, False, True, True, True, True]),
    )
    config = PipelineConfig(pnp_min_inliers=4)
    config.pnp_homography_prefilter = True

    def fake_pnp(pixels, ground, calibration, **kwargs):
        np.testing.assert_array_equal(pixels, raw[[0, 2, 3, 4, 5]])
        return SimpleNamespace(
            inlier_mask=np.array([True, False, True, True, True]),
            inlier_count=4,
            inlier_ratio=0.8,
            reprojection_mean_px=0.1,
            to_dict=lambda lat, lon: dict(
                camera_lat=lat, camera_lon=lon, pnp_inlier_ratio=0.8
            ),
        )

    monkeypatch.setattr(rotation_search, "estimate_planar_pnp", fake_pnp)
    calibration = CameraCalibration(
        np.array([[20.0, 0, 20], [0, 20, 20], [0, 0, 1]]), np.zeros(5), 40, 40
    )
    diagnostic = fit_candidate_pose(candidate, calibration, config)
    assert diagnostic["stage"] == "pnp_passed" and candidate.pnp_input_count == 5
    assert candidate.pose.inlier_mask.tolist() == [True, False, False, True, True, True]
    record = export_match(candidate, "000004", 1703474441.0, calibration, config)
    assert [p["query_idx"] for p in record["keypoints"]] == [11, 44, 55, 66]
    assert record["pnp_inlier_ratio_denominator"] == "homography_inliers"
    assert record["raw_superglue_pnp_inlier_ratio"] == pytest.approx(4 / 6)
    candidate.pose = None
    candidate.homography_mask[:] = False
    assert (
        fit_candidate_pose(candidate, calibration, config)["stage"]
        == "insufficient_homography_inliers_for_pnp"
    )
