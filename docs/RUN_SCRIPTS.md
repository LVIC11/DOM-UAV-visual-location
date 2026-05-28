# 脚本运行说明

本文档记录了仓库中新增或修改的脚本的用途、运行示例、输出位置与常见故障排查，便于复现。

## 运行前准备

1. 在项目根目录运行（含 `pyproject.toml`）。
2. 使用 Poetry 或项目虚拟环境；示例激活：
```bash
source /home/sy/.cache/pypoetry/virtualenvs/svl-FMcZIkqU-py3.10/bin/activate
```
3. 安装依赖（若未安装）：
```bash
poetry install
```

常见依赖：Python 3.10、numpy、pandas、opencv-python、Pillow、torch、tqdm 等。

## 脚本清单与运行示例

说明：所有命令均在项目根目录执行，且处于已激活的虚拟环境中。

1) `scripts/tile_satellite.py`

- 作用：将大卫星影像（例如 `data/Uav-data/map/satellite03.tif`）切分为瓦片并生成 `map.csv`。
- 示例：
```bash
python scripts/tile_satellite.py \
  --input data/Uav-data/map/satellite03.tif \
  --out-dir data/Uav-data/map/tiles/my_tiles \
  --num-tiles 50
```
- 输出：`data/Uav-data/map/tiles/my_tiles/*.png` 与 `map.csv`。

2) `scripts/generate_photo_metadata.py`

- 作用：从原始 UAV CSV（例如 `data/Uav-data/query/03.csv`）生成 `photo_metadata.csv`，并按项目约定映射角度字段（当前实现：Flight_Roll=Kappa、Flight_Yaw=Phi1、Flight_Pitch=Omega，Gimbal 留空）。
- 示例：
```bash
python scripts/generate_photo_metadata.py \
  --input data/Uav-data/query/03.csv \
  --output data/Uav-data/query/photo_metadata.csv
```

3) `scripts/write_subset_photo_metadata.py`

- 作用：从 `photo_metadata.csv` 中筛选指定范围（例如 03_0015–03_0030）并写入子目录。
- 示例：
```bash
python scripts/write_subset_photo_metadata.py \
  --input data/Uav-data/query/photo_metadata.csv \
  --start 15 --end 30 \
  --out-dir data/Uav-data/query/subset_15_30
```

4) `scripts/visualize_query_on_tiles.py`

- 作用：将每张 query 的经纬度映射到瓦片像素并在瓦片上画点用于校验。
- 示例：
```bash
python scripts/visualize_query_on_tiles.py \
  --photo-metadata data/Uav-data/query/subset_15_30/photo_metadata.csv \
  --tiles-dir data/Uav-data/map/tiles/subset_block \
  --out-dir data/Uav-data/output/subset_15_30/tile_viz
```

5) `scripts/rotate_queries_align_satellite.py`

- 作用：按 `photo_metadata.csv` 中的角度旋转 query 图像以与卫星影像方向对齐，保存旋转图与对比图。
- 示例：
```bash
python scripts/rotate_queries_align_satellite.py \
  --photo-metadata data/Uav-data/query/subset_15_30/photo_metadata.csv \
  --image-dir data/Uav-data/query/subset_15_30 \
  --out-dir data/Uav-data/query/subset_15_30/rotated
```

6) `scripts/plot_trajectories_on_original_map.py`

- 作用：运行 pipeline 以获取预测 (preds)，并在图像上绘制 GT/Pred 轨迹。逻辑如下：
  - 若 `map_db` 指向一个 tiles 目录（例如 `data/Uav-data/map/tiles/subset_block`）且其 `map.csv` 包含可读取的 tile 图像，则在该 tile 图像上绘制轨迹并写出 `trajectory_on_subset_block.png`；
  - 否则回退到全图 `satellite03.tif/.png` 并生成 `trajectory_on_original_map.png`。

- 运行（默认脚本内已设置好路径）：
```bash
python scripts/plot_trajectories_on_original_map.py
```
- 重要参数：脚本内常量 `THRESH`（米），默认 `50.0`，表示只在图上绘制 distance <= THRESH 的预测。可编辑脚本或后续改为命令行参数。
- 输出：
  - `data/Uav-data/output/subset_15_30/preds_from_pipeline.csv`
  - `data/Uav-data/output/subset_15_30/trajectory_on_subset_block.png` 或 `trajectory_on_original_map.png`

7) `scripts/diagnose_mapping.py`

- 作用：验证经纬度→像素映射一致性（全图 vs. 瓦片→全图映射）。
- 示例：
```bash
python scripts/diagnose_mapping.py \
  --photo-metadata data/Uav-data/query/subset_15_30/photo_metadata.csv \
  --tiles-dir data/Uav-data/map/tiles/subset_block
```

8) 项目主流程（可选）

使用 `scripts/main.py` 运行完整 pipeline：
```bash
python scripts/main.py \
  --image-folder data/Uav-data/query/subset_15_30/rotated \
  --map-db data/Uav-data/map/tiles/subset_block \
  --output-path data/Uav-data/output/subset_15_30/matching_rotated \
  --device cpu \
  --no-gt
```

## 常见故障与排查

- Pillow 读取大 TIFF 报错（DecompressionBombError）：脚本中通过 `Image.MAX_IMAGE_PIXELS = None` 解除限制；若仍报错，请检查 Pillow 版本与可用内存。
- `map.csv` 列名不匹配：脚本期望 `Top_left_lat,Top_left_lon,Bottom_right_lat,Bottom_right_long`（也会接受 `Bottom_right_lon`）；如不一致请手动调整 CSV 头。
- 匹配数少或 `No match found`：检查旋转后的 query 是否生成、使用 `visualize_query_on_tiles.py` 确认经纬位置、尝试增加 SuperPoint `max_keypoints` 或降低 SuperGlue match threshold，或使用 GPU 运行以允许更大参数。
- 若想把 `THRESH` 参数化：建议改为 argparse 接收参数，避免手动编辑脚本。

## 输出文件汇总（相对路径）

- Tiles：`data/Uav-data/map/tiles/<set>/`（含 `map.csv`、tile PNG）
- Photo metadata：`data/Uav-data/query/photo_metadata.csv`、`.../subset_15_30/photo_metadata.csv`
- Rotated queries：`data/Uav-data/query/subset_15_30/rotated/`
- Match visualizations：`data/Uav-data/output/subset_15_30/*.jpg`、`matching_rotated/`
- Trajectory 图：`data/Uav-data/output/subset_15_30/trajectory_on_subset_block.png` 或 `trajectory_on_original_map.png`
- Preds CSV：`data/Uav-data/output/subset_15_30/preds_from_pipeline.csv`

## 建议的后续改进（可选）

- 将 `plot_trajectories_on_original_map.py` 参数化（map_db、image_folder、THRESH、out-dir）。
- 批量化尝试不同 SuperPoint/SuperGlue 参数并记录每次匹配统计。
- 使用 `rasterio`/`GDAL` 用 GeoTIFF 仿射信息替代当前的线性经纬映射（若影像包含真实投影信息更准确）。

---

如果需要，我可以：

- 把本文件提交到仓库（已写入 `docs/RUN_SCRIPTS.md`）。
- 或继续把 `plot_trajectories_on_original_map.py` 参数化并做一次验证运行。
