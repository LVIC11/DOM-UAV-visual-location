#!/usr/bin/env python3
import csv
import os
import math
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

meta_csv = 'data/Uav-data/map/satellite_ coordinates_range.csv'
with open(meta_csv) as f:
    import csv as _csv
    row = next(_csv.DictReader(f))
lt_lat = float(row['LT_lat_map']); lt_lon = float(row['LT_lon_map']); rb_lat = float(row['RB_lat_map']); rb_lon = float(row['RB_lon_map'])

img_path = 'data/Uav-data/map/satellite03.tif'
if not os.path.exists(img_path):
    img_path = 'data/Uav-data/map/satellite03.png'
pil = Image.open(img_path)
W, H = pil.size

tiles_meta = 'data/Uav-data/map/tiles/subset_15_30/map.csv'
tiles = []
with open(tiles_meta) as f:
    r = csv.DictReader(f)
    for row in r:
        tiles.append(row)

queries_csv = 'data/Uav-data/query/subset_15_30/photo_metadata.csv'
queries = []
with open(queries_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        queries.append(row)

print('Full satellite size W,H =', W, H)
print('Checking consistency for each query:')
print('name, global_px,global_py, tile, tile_local_px, tile_local_py, global_from_tile_x,global_from_tile_y, delta_px,delta_py,delta_euc')
for q in queries:
    name = q['Filename']
    lat = float(q['Latitude']); lon = float(q['Longitude'])
    # global pixel from full image mapping
    gx = (lon - lt_lon) / (rb_lon - lt_lon) * W
    gy = (lt_lat - lat) / (lt_lat - rb_lat) * H
    # find tile containing point
    found = None
    for t in tiles:
        tl_lat = float(t['Top_left_lat']); tl_lon = float(t['Top_left_lon']); br_lat = float(t['Bottom_right_lat']); br_lon = float(t['Bottom_right_long'])
        if (br_lat <= lat <= tl_lat) and (tl_lon <= lon <= br_lon):
            found = t; break
    if not found:
        print(f"{name}, NO_TILE, lat={lat}, lon={lon}")
        continue
    tile_img = os.path.join('data/Uav-data/map/tiles/subset_15_30', found['Filename'])
    if not os.path.exists(tile_img):
        print(f"{name}, MISSING_TILE_IMAGE, {tile_img}")
        continue
    ti = Image.open(tile_img)
    tw, th = ti.size
    # tile local pixel
    tile_lon_span = float(found['Bottom_right_long']) - float(found['Top_left_lon'])
    tile_lat_span = float(found['Top_left_lat']) - float(found['Bottom_right_lat'])
    local_x = (lon - float(found['Top_left_lon'])) / tile_lon_span * tw
    local_y = (float(found['Top_left_lat']) - lat) / tile_lat_span * th
    # tile origin in global pixels (top-left)
    left_x = (float(found['Top_left_lon']) - lt_lon) / (rb_lon - lt_lon) * W
    top_y = (lt_lat - float(found['Top_left_lat'])) / (lt_lat - rb_lat) * H
    # global from tile local
    gx2 = left_x + local_x
    gy2 = top_y + local_y
    dx = gx2 - gx
    dy = gy2 - gy
    deuc = math.hypot(dx, dy)
    print(f"{name}, {gx:.2f}, {gy:.2f}, {found['Filename']}, {local_x:.2f}, {local_y:.2f}, {gx2:.2f}, {gy2:.2f}, {dx:.2f}, {dy:.2f}, {deuc:.2f}")
