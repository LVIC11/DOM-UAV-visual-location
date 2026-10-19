import importlib.util
import csv
import json
from pathlib import Path


path = Path(__file__).resolve().parents[2] / "scripts/report_rotation_search.py"
spec = importlib.util.spec_from_file_location("report_rotation_search", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empty_verified_result_remains_explicitly_unresolved(tmp_path):
    folder = tmp_path / "queries"
    folder.mkdir()
    metadata = folder / "metadata.csv"
    metadata.write_text(
        "filename,timestamp,latitude,longitude\n000004.jpg,10,31.8,118.7\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    source = dict(
        status="complete",
        query_folder=str(folder),
        superpoint_keypoint_threshold=0.005,
        geometric_thresholds=dict(pnp_homography_prefilter=True),
        input_sha256={str(metadata): module.sha(metadata)},
        frames=[
            dict(
                frame="000004",
                filename="000004.jpg",
                timestamp=10.0,
                matched=False,
                attempts=[
                    dict(
                        rotation_cw=0.0,
                        raw_matches=0,
                        homography_inliers=0,
                        stage="insufficient_matches",
                    )
                ],
            )
        ],
    )
    (output / "rotation_manifest.json").write_text(json.dumps(source))
    (output / "matched_keypoints.json").write_text("[]")
    module.report(output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["unresolved_frames"] == 1
    assert summary["pnp_verified_frames"] == summary["exported_inlier_points"] == 0
    assert summary["reference_distance_m"] is None
    assert summary["GPS_reference_used_for_selection"] is False
    assert not (output / "before_after.png").exists()


def test_untested_unrotated_baseline_is_not_a_zero_result(tmp_path):
    folder = tmp_path / "queries"
    folder.mkdir()
    metadata = folder / "metadata.csv"
    metadata.write_text(
        "filename,timestamp,latitude,longitude\n000004.jpg,10,31.8,118.7\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    source = dict(
        status="complete",
        query_folder=str(folder),
        rotation_policy="user_fixed_no_angle_search",
        superpoint_keypoint_threshold=0.005,
        geometric_thresholds={},
        input_sha256={str(metadata): module.sha(metadata)},
        frames=[
            dict(
                frame="000004",
                filename="000004.jpg",
                timestamp=10.0,
                matched=False,
                rotation_cw=270.0,
                attempts=[
                    dict(
                        rotation_cw=270.0,
                        raw_matches=2,
                        homography_inliers=0,
                        stage="insufficient_matches",
                    )
                ],
            )
        ],
    )
    (output / "rotation_manifest.json").write_text(json.dumps(source))
    (output / "matched_keypoints.json").write_text("[]")
    module.report(output)
    summary = json.loads((output / "summary.json").read_text())
    diagnostics = summary["per_frame_diagnostics"]
    assert diagnostics["unrotated_frames_tested"] == 0
    assert diagnostics["unrotated_pnp_passes"] is None
    with (output / "frame_summary.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["max_unrotated_raw_matches"] == ""
    assert row["max_searched_raw_matches"] == "2"
