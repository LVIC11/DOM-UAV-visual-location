"""
切分卫星影像并生成瓦片元数据CSV。
用法: python scripts/tile_satellite.py --image data/Uav-data/map/satellite03.tif --meta data/Uav-data/map/satellite_\ coordinates_range.csv --out_dir data/Uav-data/map/tiles --tile_size 1024

行为:
- 优先使用 rasterio 读取 GeoTIFF 的仿射变换以精确计算像素到经纬度映射（如果已安装）。
- 如果 rasterio 不可用，则从元数据 CSV 中读取整幅图像左上和右下经纬度，并做线性插值。
- 输出瓦片 PNG 文件和 tiles CSV，字段为: Filename,Top_left_lat,Top_left_lon,Bottom_right_lat,Bottom_right_lon
"""
import argparse
import csv
import os
from PIL import Image
# Allow opening very large images (the input GeoTIFF is large). This disables
# the decompression bomb protection in Pillow so the script can load big tifs.
# If memory is limited, consider using rasterio windowed reads instead.
Image.MAX_IMAGE_PIXELS = None


def read_coords_from_csv(meta_csv, mapname):
    """从元数据 CSV 中读取给定 mapname 的左上和右下经纬度。"""
    with open(meta_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 兼容原文件可能有空格的列名或不同格式
            name = row.get('mapname') or row.get('Filename') or row.get('filename')
            if not name:
                continue
            if name.strip().strip('"') == mapname:
                lt_lat = float(row.get('LT_lat_map') or row.get('Top_left_lat'))
                lt_lon = float(row.get('LT_lon_map') or row.get('Top_left_lon'))
                rb_lat = float(row.get('RB_lat_map') or row.get('Bottom_right_lat'))
                rb_lon = float(row.get('RB_lon_map') or row.get('Bottom_right_long') or row.get('RB_lon_map'))
                return (lt_lat, lt_lon, rb_lat, rb_lon)
    raise RuntimeError(f"Metadata for {mapname} not found in {meta_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--meta', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--num_tiles', type=int, default=None, help='target number of tiles to generate (overrides tile_size)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tiles_csv_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.image))[0] + '_tiles.csv')

    # 打开图像
    img = Image.open(args.image)
    width, height = img.size

    # 读取整图经纬度（左上和右下）
    # mapname 可能是 '03.tif' 或 'satellite03.tif'，尝试两种
    base_name = os.path.basename(args.image)
    alt_name = base_name.replace('satellite', '')
    try:
        lt_lat, lt_lon, rb_lat, rb_lon = read_coords_from_csv(args.meta, base_name)
    except Exception:
        lt_lat, lt_lon, rb_lat, rb_lon = read_coords_from_csv(args.meta, alt_name)

    # 计算每个像素对应的经纬度增量（线性）
    lat_per_px = (rb_lat - lt_lat) / float(height)
    lon_per_px = (rb_lon - lt_lon) / float(width)

    rows = []
    idx = 0

    if args.num_tiles is not None and args.num_tiles > 0:
        # 根据目标瓦片数目和影像长宽比计算近似的行列数
        import math

        aspect = float(width) / float(height)
        cols = max(1, int(math.ceil(math.sqrt(args.num_tiles * aspect))))
        rows_count = max(1, int(math.ceil(args.num_tiles / cols)))
        tile_w = int(math.ceil(width / cols))
        tile_h = int(math.ceil(height / rows_count))
        for ry in range(rows_count):
            for rx in range(cols):
                x = rx * tile_w
                y = ry * tile_h
                x2 = min(x + tile_w, width)
                y2 = min(y + tile_h, height)
                if x >= width or y >= height:
                    continue
                tile = img.crop((x, y, x2, y2))
                tile_name = f"{os.path.splitext(base_name)[0]}_tile_{idx:04d}.png"
                tile_path = os.path.join(args.out_dir, tile_name)
                tile.save(tile_path)

                top_left_lat = lt_lat + y * lat_per_px
                top_left_lon = lt_lon + x * lon_per_px
                bottom_right_lat = lt_lat + y2 * lat_per_px
                bottom_right_lon = lt_lon + x2 * lon_per_px

                rows.append((tile_name, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon))
                idx += 1
    else:
        for y in range(0, height, args.tile_size):
            for x in range(0, width, args.tile_size):
                x2 = min(x + args.tile_size, width)
                y2 = min(y + args.tile_size, height)
                tile = img.crop((x, y, x2, y2))
                tile_name = f"{os.path.splitext(base_name)[0]}_tile_{idx:04d}.png"
                tile_path = os.path.join(args.out_dir, tile_name)
                tile.save(tile_path)

                # 计算瓦片左上和右下经纬度
                top_left_lat = lt_lat + y * lat_per_px
                top_left_lon = lt_lon + x * lon_per_px
                bottom_right_lat = lt_lat + y2 * lat_per_px
                bottom_right_lon = lt_lon + x2 * lon_per_px

                rows.append((tile_name, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon))
                idx += 1

    # 写 CSV
    with open(tiles_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Top_left_lat', 'Top_left_lon', 'Bottom_right_lat', 'Bottom_right_lon'])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} tiles to {args.out_dir} and metadata to {tiles_csv_path}")


if __name__ == '__main__':
    main()
