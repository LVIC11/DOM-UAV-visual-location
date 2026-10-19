import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


path = Path(__file__).resolve().parents[2] / "scripts/plot_prematch_trajectory.py"
spec = importlib.util.spec_from_file_location("plot_prematch_trajectory", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prediction_lines_never_bridge_rejected_frames_or_long_gaps():
    rows = [
        dict(metadata_timestamp=t, predicted_latitude=lat)
        for t, lat in [(0, 31.0), (2, None), (4, 31.1), (6, 31.2), (20, 31.3)]
    ]
    assert module.prediction_edges(rows, 5.0) == [(2, 3)]


def test_header_time_records_associate_by_frame_and_original_timestamp(tmp_path):
    metadata, matches = tmp_path / "metadata.csv", tmp_path / "matches.json"
    metadata.write_text(
        "filename,latitude,longitude,timestamp\n000004.jpg,31.8,118.7,10\n"
    )
    record = dict(
        frame="000004",
        timestamp=9.99,
        metadata_timestamp=10.0,
        timestamp_frame="native_camera_image_header",
        predicted_lat=31.8,
        predicted_lon=118.7,
    )
    matches.write_text(json.dumps([record]))
    row = module.load_rows(metadata, matches)[0]
    assert row["image_header_timestamp"] == 9.99
    assert row["metadata_timestamp"] == 10.0
    assert row["horizontal_reference_distance_m"] == 0.0
    record["metadata_timestamp"] = 11.0
    matches.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="timestamp mismatch"):
        module.load_rows(metadata, matches)


def test_stitched_map_uses_matching_tile_geography(tmp_path):
    columns = [
        "Filename",
        "Top_left_lat",
        "Top_left_lon",
        "Bottom_right_lat",
        "Bottom_right_long",
    ]
    with (tmp_path / "map.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for x in range(2):
            for y in range(2):
                name = f"{x}_{y}.png"
                image = np.full((12, 12, 3), 30 + 50 * x + 100 * y, np.uint8)
                assert cv2.imwrite(str(tmp_path / name), image)
                writer.writerow([name, 32 - y, 118 + x, 31 - y, 119 + x])
    mosaic, bounds, _ = module.load_mosaic(tmp_path)
    assert bounds == (30.0, 32.0, 118.0, 120.0)
    assert mosaic.shape == (24, 24, 3)
    np.testing.assert_array_equal(
        module.gps_to_pixel(31.5, 118.5, bounds, mosaic.shape), [6, 6]
    )
    np.testing.assert_array_equal(
        module.gps_to_pixel(30.5, 119.5, bounds, mosaic.shape), [18, 18]
    )
    assert np.all(mosaic[:12, :12] == 30)
    assert np.all(mosaic[12:, 12:] == 180)


@pytest.mark.parametrize("method", ["planar_pnp", "homography_center"])
def test_end_to_end_plot_serializes_local_viewport(tmp_path, method):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    assert cv2.imwrite(str(tiles / "tile.png"), np.full((320, 640, 3), 70, np.uint8))
    (tiles / "map.csv").write_text(
        "Filename,Top_left_lat,Top_left_lon,Bottom_right_lat,Bottom_right_long\n"
        "tile.png,32,118,31,119\n"
    )
    metadata, matches = tmp_path / "metadata.csv", tmp_path / "matches.json"
    metadata.write_text(
        "filename,latitude,longitude,timestamp\n000004.jpg,31.5,118.5,10\n"
        "000005.jpg,31.51,118.51,12\n"
    )
    matches.write_text(
        json.dumps(
            [
                dict(
                    frame="000004",
                    timestamp=10,
                    predicted_lat=31.5,
                    predicted_lon=118.5,
                    position_method=method,
                ),
                dict(
                    frame="000005",
                    timestamp=12,
                    predicted_lat=31.51,
                    predicted_lon=118.51,
                    position_method=method,
                ),
            ]
        )
    )
    output = tmp_path / "plot"
    module.plot(metadata, matches, tiles, output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["reference_frames"] == summary["predicted_frames"] == 2
    assert summary["position_methods"] == [method]
    assert summary["prediction_segments"] == [["000004", "000005"]]
    assert summary["errors_m"]["rmse"] == 0.0
    assert summary["reran_matching_or_AIRSLAM"] is False
    assert cv2.imread(str(output / "trajectory.jpg")).shape == (320, 640, 3)
    local = cv2.imread(str(output / "trajectory_zoom.jpg"))
    assert local is not None and local.std() > 0
    with pytest.raises(FileExistsError):
        module.plot(metadata, matches, tiles, output)
