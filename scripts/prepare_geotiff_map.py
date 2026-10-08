#!/usr/bin/env python3
"""Split a GeoTIFF satellite map into tiles with CSV metadata for SatelliteMapReader.

Usage:
    python scripts/prepare_geotiff_map.py \
        --geotiff data/geovins/map/lingshan.tif \
        --output data/geovins/map/tiles \
        --tile-size 800
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import tifffile


def tile_geotiff(
    geotiff_path: str,
    output_dir: str,
    tile_size: int = 800,
    overlap: int = 0,
):
    """Split a GeoTIFF into tiles and write CSV metadata.

    Parameters
    ----------
    geotiff_path : str
        Path to the input GeoTIFF file.
    output_dir : str
        Directory to write tiles and metadata CSV.
    tile_size : int
        Size of each square tile in pixels (default 800).
    overlap : int
        Overlap between adjacent tiles in pixels (default 0).
    """
    geotiff_path = Path(geotiff_path)
    output_dir = Path(output_dir)

    if not geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

    print(f"Reading GeoTIFF: {geotiff_path}")
    img = tifffile.imread(str(geotiff_path))

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 2:
        pass  # grayscale, keep as-is
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h} pixels")

    # --- Parse geographic bounds from the companion .txt file ---
    txt_path = geotiff_path.with_suffix(".txt")
    geo_bounds = _parse_lingshan_txt(txt_path)
    top_left_lat = geo_bounds["top_left_lat"]
    top_left_lon = geo_bounds["top_left_lon"]
    bottom_right_lat = geo_bounds["bottom_right_lat"]
    bottom_right_lon = geo_bounds["bottom_right_lon"]

    print(f"Geographic bounds:")
    print(f"  Top-left:     ({top_left_lat:.6f}, {top_left_lon:.6f})")
    print(f"  Bottom-right: ({bottom_right_lat:.6f}, {bottom_right_lon:.6f})")

    # --- Create output directory ---
    output_dir.mkdir(parents=True, exist_ok=True)

    lat_range = top_left_lat - bottom_right_lat  # positive since top > bottom
    lon_range = bottom_right_lon - top_left_lon  # positive since right > left

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than tile_size ({tile_size})")

    cols = max(1, (w - overlap) // stride)
    rows = max(1, (h - overlap) // stride)

    print(f"Tiling into {cols}x{rows} = {cols * rows} tiles (stride={stride}px, overlap={overlap}px)")

    metadata = []
    tile_idx = 0

    for row in range(rows):
        for col in range(cols):
            x0 = col * stride
            y0 = row * stride
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)

            # Pad if the edge tile is smaller than tile_size
            tile_img = np.zeros((tile_size, tile_size, 3) if img.ndim == 3 else (tile_size, tile_size),
                                dtype=img.dtype)
            crop = img[y0:y1, x0:x1]
            tile_img[:crop.shape[0], :crop.shape[1]] = crop

            # Compute geographic bounds for this tile via linear interpolation
            # x: lon, y: lat
            tile_top_left_lat = top_left_lat - (y0 / h) * lat_range
            tile_top_left_lon = top_left_lon + (x0 / w) * lon_range
            tile_bottom_right_lat = top_left_lat - (y1 / h) * lat_range
            tile_bottom_right_lon = top_left_lon + (x1 / w) * lon_range

            filename = f"tile_{row:03d}_{col:03d}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), tile_img)

            metadata.append({
                "Filename": filename,
                "Top_left_lat": round(tile_top_left_lat, 8),
                "Top_left_lon": round(tile_top_left_lon, 8),
                "Bottom_right_lat": round(tile_bottom_right_lat, 8),
                "Bottom_right_long": round(tile_bottom_right_lon, 8),
            })

            tile_idx += 1
            if tile_idx % 50 == 0:
                print(f"  Created {tile_idx} tiles...")

    # --- Write metadata CSV ---
    csv_path = output_dir / "map.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Filename", "Top_left_lat", "Top_left_lon",
            "Bottom_right_lat", "Bottom_right_long",
        ])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nDone! Created {tile_idx} tiles in {output_dir}")
    print(f"Metadata CSV: {csv_path}")
    print(f"\nTo use with main.py:")
    print(f"  python scripts/main.py --image-folder <query> --map-db {output_dir}")


def _parse_lingshan_txt(txt_path: Path) -> dict:
    """Parse the lingshan.txt metadata file.

    Returns dict with keys: top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {txt_path}")

    # Try several encodings (the file contains Chinese characters, likely GBK)
    content = None
    for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            with open(txt_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if content is None:
        raise RuntimeError(f"Could not decode {txt_path} with any supported encoding")

    bounds = {}

    # Parse WGS84 geographic coordinates section
    in_wgs84 = False
    for line in content.split("\n"):
        line = line.strip()
        if "WGS 84 geographic coordinates" in line:
            in_wgs84 = True
            continue
        if in_wgs84:
            if "Upper-left coordinate:" in line:
                parts = line.split(":")[1].strip().split(",")
                bounds["top_left_lon"] = float(parts[0].strip())
                bounds["top_left_lat"] = float(parts[1].strip())
            elif "Upper-right coordinate:" in line:
                parts = line.split(":")[1].strip().split(",")
                bounds["top_right_lon"] = float(parts[0].strip())
                bounds["top_right_lat"] = float(parts[1].strip())
            elif "Lower-right coordinate:" in line:
                parts = line.split(":")[1].strip().split(",")
                bounds["bottom_right_lon"] = float(parts[0].strip())
                bounds["bottom_right_lat"] = float(parts[1].strip())
            elif "Lower-left coordinate:" in line:
                parts = line.split(":")[1].strip().split(",")
                bounds["bottom_left_lon"] = float(parts[0].strip())
                bounds["bottom_left_lat"] = float(parts[1].strip())

    if len(bounds) < 8:
        # Try UTM coordinate section as fallback
        raise RuntimeError(
            f"Could not parse WGS84 bounds from {txt_path}. "
            f"Found: {bounds}"
        )

    return {
        "top_left_lat": bounds["top_left_lat"],
        "top_left_lon": bounds["top_left_lon"],
        "bottom_right_lat": bounds["bottom_right_lat"],
        "bottom_right_lon": bounds["bottom_right_lon"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split a GeoTIFF satellite map into tiles for SatelliteMapReader"
    )
    parser.add_argument("--geotiff", type=str, required=True,
                        help="Path to the input GeoTIFF file")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to write tiles and metadata CSV")
    parser.add_argument("--tile-size", type=int, default=800,
                        help="Size of each square tile in pixels (default: 800)")
    parser.add_argument("--overlap", type=int, default=0,
                        help="Overlap between adjacent tiles in pixels (default: 0)")
    args = parser.parse_args()

    tile_geotiff(
        geotiff_path=args.geotiff,
        output_dir=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
