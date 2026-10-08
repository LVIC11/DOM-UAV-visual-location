#!/usr/bin/env python3
"""Generate a tile index JSON file from a directory of satellite tiles.

The tile index maps each tile image to its GPS bounding box.
Output format (one JSON array):
[
  {
    "name": "tile_0_0.jpg",
    "top_left_lat": 60.408615,
    "top_left_lon": 22.460445,
    "bottom_right_lat": 60.407500,
    "bottom_right_lon": 22.461560,
    "pixel_width": 800,
    "pixel_height": 800
  },
  ...
]
"""

import argparse
import json
import os
from pathlib import Path

import cv2
from svl.tms.data_structures import FlightZone, Tile
from svl.tms.geo import get_lat_long_from_tile_xy


def generate_tile_index_from_flight_zone(
    flight_zone: FlightZone, zoom_level: int, tile_dir: str, output_path: str
) -> list:
    """Generate tile index from a flight zone by checking which tiles exist on disk."""
    tiles = []
    top_left_tile = Tile.from_lat_long(
        flight_zone.top_left_lat, flight_zone.top_left_long, zoom_level
    )
    bottom_right_tile = Tile.from_lat_long(
        flight_zone.bottom_right_lat, flight_zone.bottom_right_long, zoom_level
    )

    min_x = min(top_left_tile.x, bottom_right_tile.x)
    max_x = max(top_left_tile.x, bottom_right_tile.x)
    min_y = min(top_left_tile.y, bottom_right_tile.y)
    max_y = max(top_left_tile.y, bottom_right_tile.y)

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            filename = f"{x}_{y}_{zoom_level}.png"
            filepath = os.path.join(tile_dir, filename)
            if os.path.exists(filepath):
                # Get GPS bounds
                tl_lat, tl_lon = get_lat_long_from_tile_xy(x, y, zoom_level)
                br_lat, br_lon = get_lat_long_from_tile_xy(x + 1, y + 1, zoom_level)

                # Get pixel dimensions
                img = cv2.imread(filepath)
                h, w = img.shape[:2]

                tiles.append({
                    "name": filename,
                    "top_left_lat": tl_lat,
                    "top_left_lon": tl_lon,
                    "bottom_right_lat": br_lat,
                    "bottom_right_lon": br_lon,
                    "pixel_width": w,
                    "pixel_height": h,
                })

    return tiles


def generate_tile_index_from_csv(
    csv_path: str, tile_dir: str, output_path: str
) -> list:
    """Generate tile index from a CSV metadata file.

    CSV format: Filename, Top_left_lat, Top_left_lon, Bottom_right_lat, Bottom_right_long
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    tiles = []

    for _, row in df.iterrows():
        filename = row["Filename"]
        filepath = os.path.join(tile_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found in {tile_dir}, skipping")
            continue

        img = cv2.imread(filepath)
        if img is None:
            print(f"Warning: Cannot read {filename}, skipping")
            continue
        h, w = img.shape[:2]

        tiles.append({
            "name": filename,
            "top_left_lat": float(row["Top_left_lat"]),
            "top_left_lon": float(row["Top_left_lon"]),
            "bottom_right_lat": float(row["Bottom_right_lat"]),
            "bottom_right_lon": float(row["Bottom_right_long"]),
            "pixel_width": w,
            "pixel_height": h,
        })

    return tiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tile index JSON")
    parser.add_argument("--tile-dir", type=str, required=True, help="Directory containing satellite tile images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--csv", type=str, default=None, help="CSV metadata file (optional)")
    parser.add_argument("--zoom-level", type=int, default=19, help="TMS zoom level")
    parser.add_argument("--top-left-lat", type=float, default=None, help="Flight zone top-left latitude")
    parser.add_argument("--top-left-lon", type=float, default=None, help="Flight zone top-left longitude")
    parser.add_argument("--bottom-right-lat", type=float, default=None, help="Flight zone bottom-right latitude")
    parser.add_argument("--bottom-right-lon", type=float, default=None, help="Flight zone bottom-right longitude")

    args = parser.parse_args()

    if args.csv:
        tiles = generate_tile_index_from_csv(args.csv, args.tile_dir, args.output)
    elif all([
        args.top_left_lat, args.top_left_lon,
        args.bottom_right_lat, args.bottom_right_lon,
    ]):
        flight_zone = FlightZone(
            top_left_lat=args.top_left_lat,
            top_left_long=args.top_left_lon,
            bottom_right_lat=args.bottom_right_lat,
            bottom_right_long=args.bottom_right_lon,
        )
        tiles = generate_tile_index_from_flight_zone(
            flight_zone, args.zoom_level, args.tile_dir, args.output
        )
    else:
        print("Error: Provide either --csv or flight zone coordinates")
        exit(1)

    with open(args.output, "w") as f:
        json.dump(tiles, f, indent=2)

    print(f"Generated tile index with {len(tiles)} tiles -> {args.output}")
