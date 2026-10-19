import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from svl.localization.base import PipelineConfig
from svl.localization import rotation_search as module


class Features:
    def __init__(self, points):
        self.keypoints = np.asarray(points, dtype=np.float32)

    def __len__(self):
        return len(self.keypoints)


def candidate():
    query = np.array(
        [[10, 10], [70, 10], [70, 50], [10, 50], [20, 30], [50, 40]], dtype=float
    )
    transform = np.array([[1.0, 0, 100], [0, 1, 120], [0, 0, 1]])
    tile = SimpleNamespace(
        name="tile",
        image=np.zeros((800, 800), np.uint8),
        top_left=SimpleNamespace(lat=31.833, long=118.714),
        bottom_right=SimpleNamespace(lat=31.832, long=118.715),
    )
    return module.RotationCandidate(
        0,
        120,
        tile,
        np.zeros((80, 120), np.uint8),
        np.eye(3),
        query,
        query,
        module.transform_points(query, transform),
        np.arange(6),
        np.arange(6),
        np.ones(6),
        transform,
        np.array([True, True, True, True, False, False]),
    )


def test_center_prediction_uses_h_not_pnp_and_exports_h_inliers_only():
    item = candidate()
    assert module.fit_homography_center(item)["stage"] == "homography_center_passed"
    assert item.center_prediction["satellite_center_pixel"] == [160, 160]
    assert item.center_prediction["latitude"] == pytest.approx(31.8328)
    assert item.center_prediction["longitude"] == pytest.approx(118.7142)
    config = PipelineConfig()
    calibration = SimpleNamespace(image_width=120, image_height=80)
    record = module.export_match(item, "000004", 10, calibration, config)
    assert record["position_method"] == "homography_center"
    assert record["planar_pose"] is None
    assert record["num_inliers"] == len(record["keypoints"]) == 4
    assert record["num_pnp_input_matches"] == 0
    assert all(
        p["is_homography_inlier"] and not p["is_pnp_inlier"] for p in record["keypoints"]
    )
    assert record["pnp_inlier_ratio_denominator"] is None
    assert item.has_prediction and item.pose is None


@pytest.mark.parametrize("problem", ["insufficient", "outside", "at_infinity"])
def test_center_prediction_rejects_invalid_homographies(problem):
    item = candidate()
    if problem == "insufficient":
        item.homography_mask[:2] = False
    elif problem == "outside":
        item.homography[0, 2] = 1000
    else:
        item.homography[2] = 0
    assert module.fit_homography_center(item)["stage"] != "homography_center_passed"
    assert not item.has_prediction


def test_matching_homography_mode_never_calls_pnp(monkeypatch):
    item = candidate()
    query = Features(item.query_points)
    item.tile.key_points = Features(item.satellite_points)
    detector = SimpleNamespace(detect_and_describe_keypoints=lambda image: query)
    matcher = SimpleNamespace(match_keypoints=lambda a, b: (np.arange(6), np.ones(6)))
    config = PipelineConfig()
    config.position_method = "homography_center"
    config.homography_min_inliers = 4

    def forbidden(*args, **kwargs):
        raise AssertionError("PnP must not run in homography-center mode")

    monkeypatch.setattr(module, "fit_candidate_pose", forbidden)
    result, diagnostics = module.match_rotation(
        item.image, 0, 120, detector, matcher, [item.tile], None, config
    )
    assert result.has_prediction and result.pose is None
    assert diagnostics[0]["stage"] == "homography_center_passed"


def test_pnp_only_airslam_exporter_refuses_homography_labels():
    path = Path(__file__).resolve().parents[2] / "scripts/export_airslam_prematch.py"
    spec = importlib.util.spec_from_file_location("prematch_export_guard", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    with pytest.raises(ValueError, match="mislabeled"):
        exporter.convert_record({"position_method": "homography_center"}, None, 20)


@pytest.mark.parametrize("method", ["homography_center", "planar_pnp"])
def test_main_pipeline_method_switch_and_default(monkeypatch, tmp_path, method):
    from svl.localization import pipeline as pipeline_module

    item = candidate()
    query = Features(item.query_points)
    item.tile.key_points = Features(item.satellite_points)
    engine = object.__new__(pipeline_module.Pipeline)
    engine.detector = SimpleNamespace(detect_and_describe_keypoints=lambda image: query)
    engine.matcher = SimpleNamespace(
        match_keypoints=lambda a, b: (np.arange(6), np.ones(6))
    )
    engine.map_reader = [item.tile]
    engine.logger = logging.getLogger("pipeline-method-test")
    engine.config = PipelineConfig(
        camera_calibration=SimpleNamespace(image_width=120, image_height=80)
    )
    if method == "homography_center":
        engine.config.position_method = method
    calls = []

    def pnp(*args, **kwargs):
        calls.append(True)
        return None

    monkeypatch.setattr(pipeline_module, "estimate_planar_pnp", pnp)
    image = SimpleNamespace(
        name="000004", image=item.image, geo_point=None, timestamp=10.0
    )
    result = engine.run_on_image(image, output_path=tmp_path)
    if method == "homography_center":
        assert calls == []
        assert result["is_match"] and result["planar_pose"] is None
        assert result["predicted_coordinate"].lat == pytest.approx(31.8328)
        assert result["num_inliers"] == len(result["matched_keypoints"]) == 6
        assert all(not p["is_pnp_inlier"] for p in result["matched_keypoints"])
        assert (tmp_path / "000004_viz.jpg").exists()
    else:
        assert calls == [True]
        assert not result["is_match"]
