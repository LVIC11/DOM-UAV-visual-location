"""运行 pipeline（短模式）以获取预测位置，并在原始未切分卫星图上绘制真值与预测轨迹。

输出文件：data/Uav-data/output/subset_15_30/trajectory_on_original_map.png
"""
import csv
import os
from pathlib import Path
import logging
import cv2
import numpy as np

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import SuperGlueConfig, SuperPointConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel

# config
image_folder = 'data/Uav-data/query/subset_15_30/rotated'
map_db = 'data/Uav-data/map/tiles/subset_block'
output_dir = Path('data/Uav-data/output/subset_15_30')
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# initialize models
spc = SuperPointConfig(device='cpu', max_keypoints=1024)
sgc = SuperGlueConfig(device='cpu', weights='outdoor')
sp = SuperPointAlgorithm(spc)
sg = SuperGlueMatcher(sgc)

# map reader points at full map folder (not tiles)
map_reader = SatelliteMapReader(db_path=map_db, resize_size=(800,), logger=logging.getLogger('MapReader'))
map_reader.initialize_db()
map_reader.setup_db()
map_reader.resize_db_images()
map_reader.describe_db_images(sp)

streamer = DroneImageStreamer(image_folder=image_folder, has_gt=True, logger=logging.getLogger('Streamer'))

camera_model = CameraModel(focal_length=4.5/1000, resolution_height=3040, resolution_width=4056, hfov_deg=82.9)
qp = QueryProcessor(processings=['resize'], camera_model=camera_model, satellite_resolution=None, size=(800,))

pipeline = Pipeline(map_reader=map_reader, drone_streamer=streamer, detector=sp, matcher=sg, query_processor=qp, config=PipelineConfig(), logger=logging.getLogger('Pipeline'))

preds = pipeline.run(output_path=output_dir)

# write preds to CSV if returned
pred_csv = output_dir / 'preds_from_pipeline.csv'
# threshold in meters: predictions with distance > THRESH will be treated as no-match
THRESH = 50.0
with open(pred_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Filename','Pred_lat','Pred_lon','GT_lat','GT_lon','Distance_m'])
    for idx, p in enumerate(preds):
        # pipeline.run() returns preds in the same order as streamer.image_names
        try:
            fname = streamer.image_names[idx]
        except Exception:
            fname = f'img_{idx}'
        pred = p.get('predicted_coordinate')
        gt = p.get('gt_coordinate')
        dist = p.get('distance')
        if pred:
            plat = getattr(pred, 'lat', '')
            plon = getattr(pred, 'long', '')
        else:
            plat = ''
            plon = ''
        if gt:
            glat = getattr(gt, 'lat', '')
            glon = getattr(gt, 'long', '')
        else:
            glat = ''
            glon = ''
        w.writerow([fname, plat, plon, glat, glon, dist])

print('Wrote preds CSV to', pred_csv)

# now plot on original satellite03.png (use sat_map if TIF unavailable)
# First try: if map_db points to a tiles folder and contains a map.csv with a single tile image,
# plot on that tile image using its Top_left/Bottom_right lat/lon extents.
tile_map_csv = Path(map_db) / 'map.csv'
tile_mode = False
if tile_map_csv.exists():
    import csv as _csv
    with open(tile_map_csv) as _f:
        rr = _csv.DictReader(_f)
        rows = list(rr)
    if len(rows) >= 1:
        # take the first tile entry (suitable for subset_block which contains one image)
        row = rows[0]
        tile_fname = Path(map_db) / row['Filename']
        if tile_fname.exists():
            tile_mode = True
            tile_img_path = tile_fname
            lt_lat = float(row['Top_left_lat'])
            lt_lon = float(row['Top_left_lon'])
            rb_lat = float(row['Bottom_right_lat'])
            # accept either Bottom_right_long or Bottom_right_lon
            rb_lon = float(row.get('Bottom_right_long') or row.get('Bottom_right_lon'))

if tile_mode:
    # open tile image
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    pil_img = Image.open(str(tile_img_path))
    W, H = pil_img.size
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    def geo_to_pix(lat, lon):
        x = (lon - lt_lon) / (rb_lon - lt_lon) * W
        y = (lt_lat - lat) / (lt_lat - rb_lat) * H
        return int(round(x)), int(round(y))
    out_path = output_dir / 'trajectory_on_subset_block.png'
else:
    # fallback to full satellite image
    sat_tif = Path('data/Uav-data/map/satellite03.tif')
    if not sat_tif.exists():
        sat_tif = Path('data/Uav-data/map/satellite03.png') if Path('data/Uav-data/map/satellite03.png').exists() else None
    if sat_tif is None or not sat_tif.exists():
        raise SystemExit('No satellite03.tif or satellite03.png found in data/Uav-data/map')
    # open large tif safely
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    pil_img = Image.open(str(sat_tif))
    W, H = pil_img.size
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # load map metadata for full map to convert lat/lon -> pixels
    meta_csv = Path('data/Uav-data/map') / 'satellite_ coordinates_range.csv'
    import csv as _csv
    with open(meta_csv) as _f:
        rr = _csv.DictReader(_f)
        row = next(rr)
        lt_lat = float(row['LT_lat_map'])
        lt_lon = float(row['LT_lon_map'])
        rb_lat = float(row['RB_lat_map'])
        rb_lon = float(row['RB_lon_map'])
    def geo_to_pix(lat, lon):
        x = (lon - lt_lon) / (rb_lon - lt_lon) * W
        y = (lt_lat - lat) / (lt_lat - rb_lat) * H
        return int(round(x)), int(round(y))

pred_pts = []
gt_pts = []
pred_labels = []
gt_labels = []
with open(pred_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        fname = row['Filename']
        # treat as unmatched if distance missing or > THRESH
        try:
            dist = float(row.get('Distance_m') or 0)
        except Exception:
            dist = None
        if row['GT_lat']:
            gx, gy = geo_to_pix(float(row['GT_lat']), float(row['GT_lon']))
            gt_pts.append((gx, gy))
            gt_labels.append(fname)
        # only include prediction if distance available and <= THRESH
        if row['Pred_lat'] and dist is not None and dist <= THRESH:
            px, py = geo_to_pix(float(row['Pred_lat']), float(row['Pred_lon']))
            pred_pts.append((px, py))
            pred_labels.append(fname)

vis = img.copy()
# draw connected trajectories
if len(gt_pts) > 0:
    for i in range(len(gt_pts)-1):
        cv2.line(vis, gt_pts[i], gt_pts[i+1], (0,255,0), 2)
    for i,p in enumerate(gt_pts):
        cv2.circle(vis, p, 6, (0,255,0), -1)
        cv2.putText(vis, gt_labels[i], (p[0]+8,p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
if len(pred_pts) > 0:
    for i in range(len(pred_pts)-1):
        cv2.line(vis, pred_pts[i], pred_pts[i+1], (0,0,255), 2)
    for i,p in enumerate(pred_pts):
        cv2.circle(vis, p, 6, (0,0,255), -1)
        cv2.putText(vis, pred_labels[i], (p[0]+8,p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# legend
cv2.rectangle(vis, (10,10), (260,70), (0,0,0), -1)
cv2.putText(vis, 'Green: GT (connected)', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
cv2.putText(vis, f'Red: Pred (<= {THRESH} m)', (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# write using the previously selected out_path (tile-mode sets subset filename)
cv2.imwrite(str(out_path), vis)
print('Wrote trajectory visualization to', out_path)
