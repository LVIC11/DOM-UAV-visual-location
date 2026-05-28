通过 CLI 运行可视化定位流水线

目的

- 提供一个最小说明，解释如何通过命令行运行 `scripts/main.py`，并覆盖输入影像文件夹、地图数据库路径、输出路径和运行设备。

先决条件

- 已初始化子模块（superglue_lib）：

```bash
git submodule update --init --recursive
```

- 已安装依赖（推荐使用 poetry）：

```bash
pip install poetry
poetry install
poetry shell
```

可用参数（`scripts/main.py`）

- `--image-folder`：无人机查询图像文件夹，默认 `data/query/`
- `--map-db`：卫星地图数据库目录，默认 `data/map/`
- `--output-path`：可视化输出目录，默认 `data/output`
- `--device`：模型运行设备，示例 `cuda` 或 `cpu`，默认 `cuda`

示例

- 使用默认参数直接运行（与以前相同的行为）：

```bash
python scripts/main.py
```

- 指定查询图像文件夹与输出路径：

```bash
python scripts/main.py --image-folder ~/datasets/drone_images/ --output-path /tmp/viz_output
```

- 在 CPU 上运行（没有 GPU 时）： 

```bash
python scripts/main.py --device cpu
```

注意事项

- 如果在没有 GPU 的环境使用 `--device cuda`，会报错或回退失败，请改用 `--device cpu`。
- 确保 `--map-db` 指向包含 `map.csv` 与瓦片/拼接图像的目录，或者使用项目自带的 `data/map/`。
- 可视化与模型权重（例如 SuperGlue/SuperPoint）需要相应文件与子模块存在，确保 `superglue_lib` 已正确初始化。

如果你想把更多参数暴露为 CLI（例如 resize 大小、match-threshold 或 homography 参数），告诉我我会为你添加。
