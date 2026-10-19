# 150_15 固定旋转与平面 PnP 卫星预匹配

日期：2026-10-19。状态：本批次处理与核验完成。

用户已删除部分低空图像。本次仅处理原目录中现存的 `000008.png`–`000064.png`，共 57 张。原图来自按相机 header 时间每 4 秒目标网格抽取的左目 mono8 图像，原分辨率为 1280×720。

## 输入与固定旋转

原目录：`/home/sy/sy/visual_localization/data/my_experiment/query_test/winter_150_15`。

原 metadata 仍含 63 条记录，其中 65–70 的六张图已经不存在。本次保留原 metadata 和原图，另存 57 张完整输入快照及筛选后的 metadata；没有重新提取或恢复已删除图像。

| 图号（包含端点） | 用户指定旋转 | CLI 顺时针角度 | 张数 | 通过 PnP |
|---|---|---:|---:|---:|
| 8–17 | 逆时针 90° | 270° | 10 | 7 |
| 18–23 | 顺时针 90° | 90° | 6 | 6 |
| 24–27 | 顺时针 180° | 180° | 4 | 2 |
| 28–38 | 顺时针 270° | 270° | 11 | 6 |
| 39–42 | 不变 | 0° | 4 | 1 |
| 43–54 | 顺时针 90° | 90° | 12 | 9 |
| 55–64 | 顺时针 180° | 180° | 10 | 0 |
| 合计 | | | 57 | 31 |

旋转角度全部固定，未运行自动角度搜索或角度微调。即使 PnP 未通过，也保存用户指定的完整分辨率旋转图。所有 57 张旋转图均通过与原图 `np.rot90` 的逐像素一致性检查；原图保持不变。

## 匹配政策

- 入口：`scripts/match_rotated_queries.py`；方法为 `planar_pnp`。
- SuperPoint 阈值 0.005，SuperGlue outdoor，匹配阈值 0.5、20 次 Sinkhorn。
- 每张图在指定角度下检查长边 480、640、800、1280 四个尺度及全部 18 个卫星瓦片，共 4104 个角度/尺度/瓦片尝试。
- 启用显式 `--pnp-homography-prefilter`：将单应性共识送入平面 PnP。内点率分母为筛选后的共识集，另存相对于原始 SuperGlue 匹配的内点率。
- PnP 仍要求至少 20 内点、内点率至少 0.35、6 px RANSAC、覆盖比例至少 0.03、正深度、相机高度 5–1000 m 和光轴倾斜不超过 30°。未放宽几何门槛。
- 原始标定复用 `winter/nav_left.yaml`；没有重新标定。所有导出像素均逆映射至原始未旋转、带畸变左目坐标。
- 未使用 RTK XY/GT 选择瓦片、匹配、角度或通过判据。用户外部手动筛选低空图像的过程不由本次实验审计。

这是固定旋转、多尺度、检测阈值和显式单应性预筛选的组合流程，不是仅改变旋转角度的对照。

## 结果与核验

57 张全部处理完成，31 张通过当前平面 PnP 检验，26 张未通过；导出 3333 个 PnP 内点。26 张未通过图像在本次全部尝试中的最大单应性内点数均不足 20。逐次尝试和逐图诊断完整保留，不替换用户角度。

通过的 31 张相机位置相对 metadata RTK 的事后水平距离 RMSE 为 **14.364 m**，中位数 11.254 m，最大值 28.386 m。统计仅覆盖这 31 张，不经最终对齐、不筛除差点；相机/RTK 天线杆臂未建模。这不是全航次 SLAM 精度或图融合收益。

83 个匹配输入哈希通过检查，原图/metadata 未变，57 张物理旋转精确，导出点逆映射误差为 0。另有 18 项既有固定旋转、逆像素/PnP 和报告检查通过。

本批次是 57 张均匀抽样原图，**不是历史离线图的 59 张精确关键帧查询**。`query_idx` 是局部特征索引，文件编号也不是地图关键帧 ID。尚未完成本批次与 AIRSLAM 地图的关联，没有插入 BA/位姿图因子，没有运行 AIRSLAM 或反馈前端。

## 结果位置

SVL 入口链接：`data/my_experiment/output_winter150_15_fixed_rotation_20261019`。

实际目录：`/media/sy/data2/AIRSLAM_results/experiments/winter15015_fixed_rotation_20261019/fixed_user_angles_r1`。

- `images_rotated/`：57 张按用户角度保存的原分辨率旋转图及 metadata。
- `visualizations_rotated/`、`visualizations_original/`：旋转匹配输入与原始像素坐标的对应可视化，包含未通过的候选标签。
- `matched_keypoints.json`：31 张 PnP 成功记录、3333 个内点、位姿及原始像素。
- `matched_keypoints_header_time.json`：保持上述几何结果，额外标注相机 header 时间语义、精确整数纳秒和源图身份；未吸附到地图关键帧。
- `rotation_manifest.json`、`frame_summary.csv`、`per_frame_audit.json`：完整政策与逐帧/逐尝试结果。
- `summary.json`、`verification.json`：统计和核验记录。
- `before_after.png`、`rotation_summary.png`：固定旋转预览与统计。
- `trajectory_visualization_r1/`：相机预测与 RTK 参考的卫星底图、CSV 和事后统计；拒绝帧不插值。

实验根目录还保留 `selection.json`、原始 metadata 副本、输入快照、运行命令、代码快照和日志。此前被用户中止的自动搜索记录仍独立保留，不计入本结果。
