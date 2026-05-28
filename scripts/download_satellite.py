"""Download satellite imagery from TMS servers.

Usage examples:
    # Download with default Google Satellite source
    python scripts/download_satellite.py \
        --top-left 60.408615,22.460445 \
        --bottom-right 60.400855,22.471289 \
        --zoom 18 \
        --output data/maps/my_map

    # Download with ArcGIS source
    python scripts/download_satellite.py \
        --top-left 60.408615,22.460445 \
        --bottom-right 60.400855,22.471289 \
        --zoom 18 \
        --source arcgis \
        --output data/maps/my_map

    # Download with custom TMS URL
    python scripts/download_satellite.py \
        --top-left 60.408615,22.460445 \
        --bottom-right 60.400855,22.471289 \
        --zoom 18 \
        --url "https://custom-tms-server.com/{z}/{x}/{y}.png" \
        --output data/maps/my_map

    # Download only individual tiles (no mosaic)
    python scripts/download_satellite.py \
        --top-left 60.408615,22.460445 \
        --bottom-right 60.400855,22.471289 \
        --zoom 18 \
        --output data/maps/my_map \
        --no-mosaic
"""

import argparse
import sys
from pathlib import Path

from svl.tms import FlightZone, FlightZoneDownloader, TileDownloader
from svl.tms.geo import resolution_at_zoom_level

TMS_SOURCES = {
    "google": {
        "url": "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "channels": 3,
        "img_format": "png",
        "description": "Google Satellite (no labels)",
    },
    "google_hybrid": {
        "url": "https://mt.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "channels": 3,
        "img_format": "png",
        "description": "Google Satellite + Labels",
    },
    "arcgis": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "channels": 3,
        "img_format": "png",
        "description": "ArcGIS World Imagery",
    },
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "channels": 3,
        "img_format": "png",
        "description": "OpenStreetMap (map, not satellite)",
    },
}


def parse_gps(s: str) -> tuple:
    """Parse 'lat,long' string into (lat, long) floats."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'lat,long', got: {s}")
    return float(parts[0]), float(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Download satellite imagery from TMS servers."
    )
    parser.add_argument(
        "--top-left",
        type=parse_gps,
        required=True,
        help="Top-left corner as 'lat,long' (e.g. 60.408615,22.460445)",
    )
    parser.add_argument(
        "--bottom-right",
        type=parse_gps,
        required=True,
        help="Bottom-right corner as 'lat,long' (e.g. 60.400855,22.471289)",
    )
    parser.add_argument(
        "--zoom", "-z",
        type=int,
        required=True,
        help="Zoom level (1-20). Higher = more detail. Recommended: 16-18.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory path.",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=list(TMS_SOURCES.keys()),
        default="google",
        help="Predefined TMS source (default: google).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom TMS URL template with {x},{y},{z} placeholders. Overrides --source.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the TMS server (replaces {api_key} in URL if present).",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["png", "tiff", "tif", "jpg", "jpeg"],
        default="tiff",
        help="Mosaic output format (default: tiff).",
    )
    parser.add_argument(
        "--no-mosaic",
        action="store_true",
        help="Only download individual tiles, skip mosaic stitching.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Image channels: 3 for RGB, 1 for grayscale (default: 3).",
    )

    args = parser.parse_args()

    # Validate zoom level
    if not (1 <= args.zoom <= 20):
        print("Error: zoom level must be between 1 and 20.")
        sys.exit(1)

    # Build flight zone
    tl_lat, tl_long = args.top_left
    br_lat, br_long = args.bottom_right

    try:
        flight_zone = FlightZone(
            top_left_lat=tl_lat,
            top_left_long=tl_long,
            bottom_right_lat=br_lat,
            bottom_right_long=br_long,
        )
    except Exception as e:
        print(f"Error creating flight zone: {e}")
        sys.exit(1)

    # Determine TMS URL
    if args.url:
        tms_url = args.url
        source_desc = "Custom"
    else:
        src = TMS_SOURCES[args.source]
        tms_url = src["url"]
        if args.channels == 3:
            args.channels = src["channels"]
        source_desc = src["description"]

    # Print download info
    w_tiles, h_tiles = flight_zone.size_in_tiles(args.zoom)
    w_px, h_px = flight_zone.size_in_pixels(args.zoom)
    resolution = resolution_at_zoom_level(tl_lat, args.zoom)

    print("=" * 60)
    print("TMS Satellite Download")
    print("=" * 60)
    print(f"Source:          {source_desc}")
    print(f"URL:             {tms_url}")
    print(f"Zoom level:      {args.zoom}")
    print(f"Resolution:      {resolution:.2f} m/px")
    print(f"Top-left:        ({tl_lat}, {tl_long})")
    print(f"Bottom-right:    ({br_lat}, {br_long})")
    print(f"Area size:       {flight_zone.width_in_meters:.0f} x {flight_zone.height_in_meters:.0f} m")
    print(f"Tiles:           {w_tiles} x {h_tiles} = {w_tiles * h_tiles} tiles")
    print(f"Pixel size:      {w_px} x {h_px}")
    print(f"Output:          {args.output}")
    print(f"Mosaic format:   {'none' if args.no_mosaic else args.format}")
    print("=" * 60)

    # Create downloader
    tile_downloader = TileDownloader(
        url=tms_url,
        channels=args.channels,
        api_key=args.api_key,
        headers=None,
        img_format="png",
    )

    flight_zone_downloader = FlightZoneDownloader(
        tile_downloader=tile_downloader,
        flight_zone=flight_zone,
    )

    # Download
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    tiles = list(flight_zone.yield_tiles(args.zoom))

    if args.no_mosaic:
        print("\nDownloading tiles...")
        tiles_dir = output_path / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        tile_downloader.download_tiles(tiles, str(tiles_dir))
        print(f"\nDone! {len(tiles)} tiles saved to {output_path / 'tiles'}")
    else:
        print("\nDownloading tiles and stitching mosaic...")
        flight_zone_downloader.download_tiles_and_save_as_mosaic(
            zoom_level=args.zoom,
            output_path=str(output_path),
            mosaic_format=args.format,
        )
        print(f"\nDone! Tiles saved to {output_path / 'tiles'}")
        print(f"Mosaic saved to {output_path / f'mosaic.{args.format}'}")

    # Save per-tile GPS coordinates CSV (compatible with SatelliteMapReader)
    map_csv_path = output_path / "map.csv"
    with open(map_csv_path, "w") as f:
        f.write("Filename,Top_left_lat,Top_left_lon,Bottom_right_lat,Bottom_right_long\n")
        for tile in tiles:
            br = tile.bottom_right_corner
            fname = f"{tile.file_name}.png"
            f.write(
                f'"{fname}", {tile.lat:.6f}, {tile.long:.6f}, '
                f'{br.lat:.6f}, {br.long:.6f}\n'
            )
    print(f"Map CSV saved to {map_csv_path} ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
