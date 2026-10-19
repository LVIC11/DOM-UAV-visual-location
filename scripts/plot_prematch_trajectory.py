"""Plot saved SVL predictions and metadata references, without rerunning matching."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from svl.tms.geo import haversine_distance
from svl.tms.schemas import GpsCoordinate


def load_mosaic(folder):
    metadata = folder / "map.csv"
    with metadata.open(newline="") as stream:
        tiles = list(csv.DictReader(stream))
    if not tiles:
        raise ValueError("Empty satellite tile metadata")
    west = sorted({float(r["Top_left_lon"]) for r in tiles})
    north = sorted({float(r["Top_left_lat"]) for r in tiles}, reverse=True)
    if len(tiles) != len(west) * len(north):
        raise ValueError("A complete rectangular tile mosaic is required")
    images, paths, cells = [], [metadata], set()
    for row in tiles:
        path = folder / row["Filename"].strip()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or (images and image.shape != images[0][1].shape):
            raise ValueError("Missing tile or inconsistent tile dimensions")
        x = west.index(float(row["Top_left_lon"]))
        y = north.index(float(row["Top_left_lat"]))
        if (x, y) in cells:
            raise ValueError("Duplicate tile location")
        cells.add((x, y))
        images.append(((x, y), image))
        paths.append(path)
    h, w = images[0][1].shape[:2]
    mosaic = np.zeros((len(north) * h, len(west) * w, 3), np.uint8)
    bounds = (
        min(float(r["Bottom_right_lat"]) for r in tiles),
        max(north),
        min(west),
        max(float(r["Bottom_right_long"]) for r in tiles),
    )
    south, top, left, right = bounds
    for (x, y), image in images:
        mosaic[y * h : (y + 1) * h, x * w : (x + 1) * w] = image
    # Check every metadata corner against its actual stitched pixel location.
    for row in tiles:
        x = west.index(float(row["Top_left_lon"]))
        y = north.index(float(row["Top_left_lat"]))
        corners = [
            ((float(row["Top_left_lon"]) - left) / (right - left), x / len(west)),
            ((top - float(row["Top_left_lat"])) / (top - south), y / len(north)),
            (
                (float(row["Bottom_right_long"]) - left) / (right - left),
                (x + 1) / len(west),
            ),
            (
                (top - float(row["Bottom_right_lat"])) / (top - south),
                (y + 1) / len(north),
            ),
        ]
        if any(abs(a - b) > 5e-4 for a, b in corners):
            raise ValueError("Tile metadata is not consistent with the stitched grid")
    return mosaic, bounds, paths


def gps_to_pixel(lat, lon, bounds, shape):
    south, north, west, east = bounds
    return np.array(
        [
            (lon - west) / (east - west) * shape[1],
            (north - lat) / (north - south) * shape[0],
        ]
    )


def load_rows(metadata, matches):
    with metadata.open(newline="") as stream:
        source = [{k.lower(): v for k, v in r.items()} for r in csv.DictReader(stream)]
    source.sort(key=lambda r: float(r["timestamp"]))
    records = json.loads(matches.read_text())
    prediction = {str(r["frame"]): r for r in records}
    frames = [Path(r["filename"]).stem for r in source]
    if not source or len(set(frames)) != len(frames) or len(prediction) != len(records):
        raise ValueError("Empty or ambiguous frame identities")
    if set(prediction) - set(frames):
        raise ValueError("Prediction frame is missing from source metadata")
    rows = []
    for item, frame in zip(source, frames):
        row = dict(
            frame=frame,
            metadata_timestamp=float(item["timestamp"]),
            reference_latitude=float(item["latitude"]),
            reference_longitude=float(item["longitude"]),
            image_header_timestamp=None,
            predicted_latitude=None,
            predicted_longitude=None,
            horizontal_reference_distance_m=None,
            position_method=None,
        )
        if frame in prediction:
            record = prediction[frame]
            if (
                abs(
                    record.get("metadata_timestamp", record["timestamp"])
                    - row["metadata_timestamp"]
                )
                > 1e-6
            ):
                raise ValueError("Prediction/source timestamp mismatch")
            if record.get("timestamp_frame") == "native_camera_image_header":
                row["image_header_timestamp"] = record["timestamp"]
            row.update(
                predicted_latitude=float(record["predicted_lat"]),
                predicted_longitude=float(record["predicted_lon"]),
                position_method=record.get("position_method", "planar_pnp"),
            )
            row["horizontal_reference_distance_m"] = 1000 * haversine_distance(
                GpsCoordinate(
                    lat=row["reference_latitude"], long=row["reference_longitude"]
                ),
                GpsCoordinate(
                    lat=row["predicted_latitude"], long=row["predicted_longitude"]
                ),
            )
        rows.append(row)
    if any(
        not np.isfinite(v)
        for row in rows
        for v in row.values()
        if isinstance(v, (int, float))
    ):
        raise ValueError("Non-finite plot coordinates/timestamps")
    return rows


def prediction_edges(rows, max_gap_s):
    return [
        (i - 1, i)
        for i in range(1, len(rows))
        if rows[i - 1]["predicted_latitude"] is not None
        and rows[i]["predicted_latitude"] is not None
        and 0
        < rows[i]["metadata_timestamp"] - rows[i - 1]["metadata_timestamp"]
        <= max_gap_s
    ]


def outlined_text(image, label, point, color, scale=0.6):
    for thickness, shade in ((4, (0, 0, 0)), (1, color)):
        cv2.putText(
            image,
            label,
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            shade,
            thickness,
            cv2.LINE_AA,
        )


def render(
    mosaic,
    references,
    predictions,
    rows,
    edges,
    viewport,
    zoom,
    errors,
    prediction_label="Planar PnP",
):
    x0, y0, x1, y1 = viewport
    image = cv2.resize(
        mosaic[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR
    )

    def point(pixel):
        return tuple(np.rint((pixel - [x0, y0]) * zoom).astype(int))

    green, magenta = (0, 255, 0), (255, 0, 255)
    for i in range(1, len(references)):
        cv2.line(
            image, point(references[i - 1]), point(references[i]), green, 2, cv2.LINE_AA
        )
    for pixel in references:
        cv2.circle(image, point(pixel), 3, green, -1, cv2.LINE_AA)
    cv2.circle(image, point(references[0]), 7, green, 2, cv2.LINE_AA)
    for a, b in edges:
        cv2.line(
            image, point(predictions[a]), point(predictions[b]), magenta, 2, cv2.LINE_AA
        )
    for n, (i, pixel) in enumerate(predictions.items()):
        target, reference = point(pixel), point(references[i])
        cv2.line(image, reference, target, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(image, target, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, target, 5, magenta, -1, cv2.LINE_AA)
        label = f"{rows[i]['frame']}  {rows[i]['horizontal_reference_distance_m']:.2f} m"
        if len(predictions) <= 8:
            outlined_text(
                image,
                label,
                (target[0] + 12, target[1] + (-16 if n % 2 == 0 else 28)),
                magenta,
                0.55,
            )
    legends = [
        (f"Reference (green): {len(rows)} frames", green),
        (
            f"{prediction_label} (magenta): {len(predictions)} / {len(rows)} predictions",
            magenta,
        ),
        (
            f"Horizontal MAE {np.mean(errors):.2f} m | RMSE {np.sqrt(np.mean(np.square(errors))):.2f} m ({len(errors)} samples)"
            if errors
            else "No accepted predictions",
            (255, 255, 255),
        ),
        (
            "Image ground footprint; tilt offset NOT corrected"
            if prediction_label == "Homography center"
            else "Metadata GPS reference; camera/antenna offset unmodelled",
            (255, 255, 255),
        ),
    ]
    for line, (label, color) in enumerate(legends):
        outlined_text(image, label, (15, 25 + line * 25), color)
    return image


def plot(metadata, matches, tiles, output, max_gap_s=5.0):
    if output.exists():
        raise FileExistsError(str(output))
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("A finite positive prediction gap is required")
    mosaic, bounds, paths = load_mosaic(tiles)
    rows = load_rows(metadata, matches)
    methods = sorted({r["position_method"] for r in rows if r["position_method"]})
    prediction_label = (
        "Homography center"
        if methods == ["homography_center"]
        else (
            "Planar PnP"
            if not methods or methods == ["planar_pnp"]
            else "Mixed predictions"
        )
    )
    references = [
        gps_to_pixel(
            r["reference_latitude"], r["reference_longitude"], bounds, mosaic.shape
        )
        for r in rows
    ]
    predictions = {
        i: gps_to_pixel(
            r["predicted_latitude"], r["predicted_longitude"], bounds, mosaic.shape
        )
        for i, r in enumerate(rows)
        if r["predicted_latitude"] is not None
    }
    points = np.array(references + list(predictions.values()))
    if np.any(points < 0) or np.any(
        points >= np.array([mosaic.shape[1], mosaic.shape[0]])
    ):
        raise ValueError(
            "Some coordinates lie outside the satellite mosaic; refusing to hide them"
        )
    edges = prediction_edges(rows, max_gap_s)
    errors = [
        r["horizontal_reference_distance_m"]
        for r in rows
        if r["predicted_latitude"] is not None
    ]
    viewports = {"trajectory.jpg": ((0, 0, mosaic.shape[1], mosaic.shape[0]), 1)}
    if predictions:
        local = np.array(
            list(predictions.values()) + [references[i] for i in predictions]
        )
        center = (local.min(axis=0) + local.max(axis=0)) / 2
        size = np.maximum(
            local.max(axis=0) - local.min(axis=0) + [100, 140], [480, 300]
        ).astype(int)
        size = np.minimum(size, [mosaic.shape[1], mosaic.shape[0]])
        start = np.minimum(
            np.maximum(np.floor(center - size / 2).astype(int), 0),
            [mosaic.shape[1], mosaic.shape[0]] - size,
        )
        viewports["trajectory_zoom.jpg"] = (tuple(start) + tuple(start + size), 2)
    inputs = paths + [metadata, matches, Path(__file__).resolve()]
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
    }
    output.mkdir(parents=True)
    for name, (viewport, zoom) in viewports.items():
        image = render(
            mosaic,
            references,
            predictions,
            rows,
            edges,
            viewport,
            zoom,
            errors,
            prediction_label,
        )
        if not cv2.imwrite(str(output / name), image, [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise RuntimeError("Could not save trajectory image")
    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        reference_frames=len(rows),
        predicted_frames=len(predictions),
        position_methods=methods,
        prediction_label=prediction_label,
        prediction_frames=[rows[i]["frame"] for i in predictions],
        prediction_segments=[[rows[a]["frame"], rows[b]["frame"]] for a, b in edges],
        missing_predictions_interpolated=False,
        prediction_max_gap_s=max_gap_s,
        reference_source="Original query metadata GPS, not SLAM; camera/antenna lever arm unmodelled",
        prediction_scope="Image-center ground projection, not 3D camera position; tilt offset uncorrected"
        if methods == ["homography_center"]
        else "Camera position; antenna lever arm unmodelled",
        errors_m=dict(
            mae=float(np.mean(errors)),
            rmse=float(np.sqrt(np.mean(np.square(errors)))),
            max=float(max(errors)),
        )
        if errors
        else None,
        map_source="Actual matching tiles stitched using tile map.csv, not hardcoded legacy mosaic bounds",
        mosaic_bounds_south_north_west_east=bounds,
        viewports={
            k: dict(crop_xyxy=[int(n) for n in v[0]], magnification=v[1])
            for k, v in viewports.items()
        },
        GT_used_to_filter_or_align_predictions=False,
        reran_matching_or_AIRSLAM=False,
        input_sha256=hashes,
    )
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()
    ):
        raise ValueError("Plot inputs changed")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "matches", "tiles", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-gap-s", type=float, default=5.0)
    args = parser.parse_args()
    plot(args.metadata, args.matches, args.tiles, args.output, args.max_gap_s)
