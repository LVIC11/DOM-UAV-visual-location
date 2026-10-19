# 150_15 删除图像后的完整重匹配（r3）

日期：2026-10-19。状态：本次匹配与核验完成。

本次按用户当前目录的 **54 张现存原图**重新计算全部匹配。相较 r2，实际删除的是 000008、000027、000031，保持其余文件的原编号，不重新编号，不恢复已删除图像。用户明确确认包含文件仍在但 metadata 行缺失的 000014、000015；这两张的时间戳由经过图像 SHA 核对的上一版原始输入记录补入本次快照，原 metadata 不变。

## 政策与结果

沿用已确认的固定角度，55–64 为逆时针 90°（CW270）。四个尺度 480/640/800/1280、全部 18 瓦片、SuperPoint 0.005、SuperGlue outdoor、显式单应性共识预筛选和平面 PnP 保持不变。未搜索或微调旋转角度，未放宽 PnP 门槛，未使用 RTK XY/GT 选择匹配。

PnP 要求至少 20 内点、0.35 内点率、6 px RANSAC、0.03 覆盖、正深度、高度 5–1000 m、光轴倾斜不超过 30°。0.35 的分母为单应性预筛后的共识集，原始 SuperGlue 分母的比例也保留。

**54 张全部重新匹配，38 张通过、16 张未通过，导出 3863 个 PnP 内点。** 本轮没有复用 r1/r2 的匹配记录。

| 原编号范围 | 固定旋转 | 现存张数 | 通过 PnP |
|---|---|---:|---:|
| 8–17 | 逆时针 90° | 9 | 7 |
| 18–23 | 顺时针 90° | 6 | 6 |
| 24–27 | 顺时针 180° | 3 | 2 |
| 28–38 | 顺时针 270° | 10 | 6 |
| 39–42 | 不变 | 4 | 1 |
| 43–54 | 顺时针 90° | 12 | 9 |
| 55–64 | 逆时针 90° | 10 | 7 |

每张检查四尺度×18 瓦片，共 3888 次尝试。所有 54 张完整分辨率固定旋转图均已保存，即使几何未通过也不丢弃其旋转结果。逐次失败原因见 rotation_manifest.json，逐图统计见 frame_summary.csv。

## 核验与事后评价

原图/metadata 和 80 项匹配输入哈希检查通过，54 张旋转图均与指定角度的原图逐像素一致，原始像素逆映射误差为 0.0 px，精确相机 header 纳秒与 bag RECORD 时间分开保留。r1/r2 的 450 项历史产物哈希检查通过。

相对 r2，相同当前输入范围内通过判定变化 0 张；共同成功记录逐项相同 38/38，完整差异在 comparison_with_r2.json。样本删减造成的通过比例变化不代表定位精度改善。

成功的 38 张相机位置相对 metadata RTK 的事后水平距离 RMSE 为 13.470 m，中位数 10.607 m，最大值 28.386 m。未做最终拟合、GT 筛除或插值，相机/天线杆臂未建模；这不是全航次 SLAM 精度。

本批次仍是均匀抽样原图，未关联 AIRSLAM 关键帧或地图特征 ID，未插入图因子、未运行 AIRSLAM 前端或反馈。

## 文件

入口：data/my_experiment/output_winter150_15_fixed_rotation_20261019_r3。

实际目录：/media/sy/data2/AIRSLAM_results/experiments/winter15015_fixed_rotation_20261019_r3/fixed_user_angles_r3。

- images_rotated/：54 张原分辨率旋转图与 metadata。
- visualizations_rotated/、visualizations_original/：各自像素坐标下的匹配可视化。
- matched_keypoints.json：本轮成功记录、位姿、原始像素与 PnP 内点。
- matched_keypoints_header_time.json：附完整整数纳秒时间和原图身份；没有吸附到地图关键帧。
- summary.json、verification.json、frame_summary.csv、per_frame_audit.json：统计与核验。
- comparison_with_r2.json：相同图像范围上的结果对比。
- trajectory_visualization_r1/：本轮位置的卫星底图、CSV 与参考统计，未通过图像不插值。

实验根目录保留 selection.json、原 metadata 副本、54 张输入快照、来源代码快照、run.json 与日志。原目录和所有历史结果保持不变。
