"""根据 photo_metadata.csv 中的角度字段对 query 图像做旋转矫正以与卫星正射影像（北向上）对齐。

假设与约定：
- 使用列 `Flight_Yaw`（以度为单位，顺时针为正）作为航向（heading），图像的上边缘是相机前方。
- 为使图像北向上，按 -Flight_Yaw 度旋转（即逆时针旋转 Flight_Yaw 度）。
- 若 `Flight_Yaw` 为空，脚本会尝试回退到 `Flight_Roll` 或 `Flight_Pitch`，否则不旋转。

输出：
- data/Uav-data/query/subset_15_30/rotated/{Filename} （与原图同名）
- data/Uav-data/output/subset_15_30/rotated_viz/{Filename_without_ext}_compare.png （原图|旋转后 并排对比，可选）

注意：这是一种近似校正（假设正射图为北上、影像为透视相机且相机前方向在图像上方）。如需更精确的配准，请使用相机内参、云台角度或基于关键点的几何变换（可进一步实现）。
"""
import os
import csv
import math
import cv2

q_csv = 'data/Uav-data/query/subset_15_30/photo_metadata.csv'
q_dir = 'data/Uav-data/query/subset_15_30'
rot_dir = os.path.join(q_dir, 'rotated')
viz_dir = 'data/Uav-data/output/subset_15_30/rotated_viz'
os.makedirs(rot_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

def safe_float(s):
    try:
        return float(s)
    except Exception:
        return None

rows = []
with open(q_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

processed = 0
for row in rows:
    fname = row['Filename']
    src_path = os.path.join(q_dir, fname)
    if not os.path.exists(src_path):
        print('Missing query image', src_path)
        continue

    yaw = safe_float(row.get('Flight_Yaw') or '')
    roll = safe_float(row.get('Flight_Roll') or '')
    pitch = safe_float(row.get('Flight_Pitch') or '')

    use_field = None
    angle = 0.0
    if yaw is not None:
        angle = -yaw
        use_field = 'Flight_Yaw'
    elif roll is not None:
        angle = -roll
        use_field = 'Flight_Roll'
    elif pitch is not None:
        angle = -pitch
        use_field = 'Flight_Pitch'
    else:
        angle = 0.0
        use_field = None

    img = cv2.imread(src_path)
    if img is None:
        print('Failed to read', src_path)
        continue

    h, w = img.shape[:2]
    # rotate around center, keep same image size
    center = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    out_path = os.path.join(rot_dir, fname)
    cv2.imwrite(out_path, rotated)

    # make side-by-side viz
    try:
        left = cv2.resize(img, (w, h))
        right = cv2.resize(rotated, (w, h))
        combo = cv2.hconcat([left, right])
        viz_name = os.path.splitext(fname)[0] + '_compare.png'
        viz_path = os.path.join(viz_dir, viz_name)
        cv2.putText(combo, f'orig', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(combo, f'rotated ({use_field}:{angle:.2f} deg)', (w+20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite(viz_path, combo)
    except Exception as e:
        print('Viz failed for', fname, e)

    print('Rotated', fname, 'by', angle, 'degrees (used', use_field, ') ->', out_path)
    processed += 1

print('Done. Rotated', processed, 'images. Rotated files in', rot_dir, 'visualizations in', viz_dir)
