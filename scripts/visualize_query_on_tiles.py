"""将 subset_15_30 的 query 图像经纬度映射到子集瓦片的像素坐标，并在瓦片上绘制标记保存可视化。

输出：data/Uav-data/output/subset_15_30/tile_viz/{query}_on_{tile}.png
"""
import csv
import os
import cv2

tiles_csv = 'data/Uav-data/map/tiles/subset_15_30/map.csv'
queries_csv = 'data/Uav-data/query/subset_15_30/photo_metadata.csv'
tiles_dir = 'data/Uav-data/map/tiles/subset_15_30'
out_dir = 'data/Uav-data/output/subset_15_30/tile_viz'

os.makedirs(out_dir, exist_ok=True)

def load_tiles(csvp):
    tiles = []
    with open(csvp) as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row['Filename']
            tl_lat = float(row['Top_left_lat'])
            tl_lon = float(row['Top_left_lon'])
            br_lat = float(row['Bottom_right_lat'])
            br_lon = float(row['Bottom_right_long'])
            tiles.append({'filename': fn, 'tl_lat': tl_lat, 'tl_lon': tl_lon, 'br_lat': br_lat, 'br_lon': br_lon})
    return tiles

def in_bounds(lat, lon, tile):
    return (tile['br_lat'] <= lat <= tile['tl_lat']) and (tile['tl_lon'] <= lon <= tile['br_lon'])

tiles = load_tiles(tiles_csv)

queries = []
with open(queries_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        queries.append({'filename': row['Filename'], 'lat': float(row['Latitude']), 'lon': float(row['Longitude'])})

created = []
for q in queries:
    found = None
    for t in tiles:
        if in_bounds(q['lat'], q['lon'], t):
            found = t
            break
    if not found:
        print('No tile found for', q['filename'])
        continue

    tile_path = os.path.join(tiles_dir, found['filename'])
    if not os.path.exists(tile_path):
        print('Missing tile image', tile_path)
        continue

    img = cv2.imread(tile_path, cv2.IMREAD_COLOR)
    if img is None:
        print('Failed to read', tile_path)
        continue

    h, w = img.shape[:2]
    # map lon->x, lat->y
    lon_span = found['br_lon'] - found['tl_lon']
    lat_span = found['tl_lat'] - found['br_lat']
    if lon_span == 0 or lat_span == 0:
        print('Zero span for tile', found['filename'])
        continue

    x = (q['lon'] - found['tl_lon']) / lon_span * w
    y = (found['tl_lat'] - q['lat']) / lat_span * h
    ix, iy = int(round(x)), int(round(y))

    # draw marker and label
    vis = img.copy()
    cv2.circle(vis, (ix, iy), max(4, int(min(w,h)*0.01)), (0,0,255), -1)
    label = q['filename']
    cv2.putText(vis, label, (ix+8, iy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    out_name = f"{q['filename'].replace('.JPG','')}_on_{found['filename'].rsplit('.',1)[0]}.png"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, vis)
    created.append(out_path)
    print('Wrote', out_path, 'pixel=(%d,%d)'%(ix,iy))

print('Done, created', len(created), 'visualizations in', out_dir)
