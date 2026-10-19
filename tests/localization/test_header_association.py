import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[2] / "scripts/associate_query_headers.py"
spec = importlib.util.spec_from_file_location("associate_query_headers", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def association():
    return dict(
        status="complete",
        frames=[
            dict(
                frame="000004",
                metadata_timestamp=10.003,
                bag_record_timestamp=10.003,
                image_header_timestamp=10.0,
                image_sequence=42,
            )
        ],
    )


def test_header_retiming_preserves_original_time_and_all_point_data():
    points = [dict(query_x=123.0, query_y=456.0, sat_lat=31.8, sat_lon=118.7)]
    original = dict(frame="000004", timestamp=10.003, keypoints=points)
    result = module.retime_records([original], association())
    assert result[0]["timestamp"] == 10.0
    assert result[0]["metadata_timestamp"] == 10.003
    assert result[0]["keypoints"] == points
    assert original["timestamp"] == 10.003
    assert result[0]["timestamp_frame"] == "native_camera_image_header"


def test_nonassociated_matches_are_not_silently_snapped_to_an_image():
    with pytest.raises(ValueError, match="not the associated source record"):
        module.retime_records([dict(frame="000004", timestamp=10.04)], association())


def test_ambiguous_header_identity_is_rejected():
    data = association()
    data["frames"].append(data["frames"][0].copy())
    with pytest.raises(ValueError, match="unambiguous"):
        module.retime_records([], data)
