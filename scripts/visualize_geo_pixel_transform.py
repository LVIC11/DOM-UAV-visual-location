#!/usr/bin/env python3
"""Visualize the geographic plane to pixel coordinate transformation."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize geo-pixel transform")
    parser.add_argument("--poses-csv", type=str, required=True)
    parser.add_argument("--transforms-csv", type=str, required=True)
    parser.add_argument("--query-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0, help="Frame to visualize")
    parser.add_argument("--grid-spacing-m", type=float, default=50.0, help="Grid spacing in meters")

    args = parser.parse_args()

    # Load pose
    with open(args.poses_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['frame']) == args.frame:
                pose = row
                break

    # Load transformation
    with open(args.transforms_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['frame']) == args.frame:
                H = np.array([float(x) for x in row['H'].split(',')]).reshape(3, 3)
                break

    print(f"Frame {args.frame}:")
    print(f"  GPS: ({pose['latitude']}, {pose['longitude']})")
    print(f"  Altitude: {pose['altitude']}m")
    print(f"  Heading: {pose['heading']}°")

    # Load image
    image_path = Path(args.query_dir) / f"{args.frame:06d}.jpg"
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    h, w = image.shape[:2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Draw grid on image showing ground plane coordinates
    viz = image.copy()

    # Draw grid lines at regular intervals in ground meters
    # Image center corresponds to some ground point
    # Use inverse homography to find ground coordinates of image corners
    H_inv = np.linalg.inv(H)

    # Image corners to ground
    corners_pixel = np.array([
        [0, 0, 1],
        [w-1, 0, 1],
        [w-1, h-1, 1],
        [0, h-1, 1],
    ], dtype=np.float64).T

    corners_ground = H_inv @ corners_pixel
    corners_ground /= corners_ground[2]

    print(f"\nImage corners in ground coordinates (meters):")
    print(f"  Top-left: ({corners_ground[0,0]:.1f}, {corners_ground[1,0]:.1f})")
    print(f"  Top-right: ({corners_ground[0,1]:.1f}, {corners_ground[1,1]:.1f})")
    print(f"  Bottom-right: ({corners_ground[0,2]:.1f}, {corners_ground[1,2]:.1f})")
    print(f"  Bottom-left: ({corners_ground[0,3]:.1f}, {corners_ground[1,3]:.1f})")

    # Draw grid
    spacing = args.grid_spacing_m
    x_min = int(corners_ground[0].min() / spacing) * spacing
    x_max = int(corners_ground[0].max() / spacing + 1) * spacing
    y_min = int(corners_ground[1].min() / spacing) * spacing
    y_max = int(corners_ground[1].max() / spacing + 1) * spacing

    # Draw vertical grid lines (constant x)
    for x in range(int(x_min), int(x_max) + 1, int(spacing)):
        points = []
        for y in range(int(y_min), int(y_max) + 1, int(spacing / 2)):
            ground_pt = np.array([x, y, 1.0])
            pixel_pt = H @ ground_pt
            pixel_pt /= pixel_pt[2]
            if 0 <= pixel_pt[0] < w and 0 <= pixel_pt[1] < h:
                points.append((int(pixel_pt[0]), int(pixel_pt[1])))
        if len(points) > 1:
            cv2.polylines(viz, [np.array(points)], False, (0, 255, 0), 1)
            # Label
            cv2.putText(viz, f"{x}m", (points[0][0], points[0][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw horizontal grid lines (constant y)
    for y in range(int(y_min), int(y_max) + 1, int(spacing)):
        points = []
        for x in range(int(x_min), int(x_max) + 1, int(spacing / 2)):
            ground_pt = np.array([x, y, 1.0])
            pixel_pt = H @ ground_pt
            pixel_pt /= pixel_pt[2]
            if 0 <= pixel_pt[0] < w and 0 <= pixel_pt[1] < h:
                points.append((int(pixel_pt[0]), int(pixel_pt[1])))
        if len(points) > 1:
            cv2.polylines(viz, [np.array(points)], False, (0, 255, 0), 1)
            # Label
            cv2.putText(viz, f"{y}m", (points[0][0] + 5, points[0][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Mark image center
    cv2.circle(viz, (w//2, h//2), 10, (0, 0, 255), 2)
    cv2.putText(viz, "Image Center", (w//2 + 15, h//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Add info text
    lat = float(pose['latitude'])
    lon = float(pose['longitude'])
    alt = float(pose['altitude'])
    hdg = float(pose['heading'])
    info = f"Frame {args.frame} | GPS: ({lat:.4f}, {lon:.4f})"
    info += f" | Alt: {alt:.1f}m | Heading: {hdg:.1f} deg"
    cv2.putText(viz, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save
    output_path = output_dir / f"frame_{args.frame:06d}_geo_grid.jpg"
    cv2.imwrite(str(output_path), viz)
    print(f"\nSaved visualization to {output_path}")


if __name__ == "__main__":
    main()
