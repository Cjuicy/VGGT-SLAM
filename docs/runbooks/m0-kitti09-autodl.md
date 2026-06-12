# M0 KITTI 09 AutoDL 长序列验证

KITTI odometry 09 共 1591 帧，轨迹包含回到已访问区域的运动，适合在 M0
阶段同时检查长序列前端缓存导出、Sim(3) 基线和 SALAD 回环候选。所有命令
从云端仓库根目录 `~/autodl-tmp/VGGT-SLAM` 执行。

先完成 [`m0-autodl-baseline.md`](m0-autodl-baseline.md) 的环境、权重和 TUM
desk 验收。KITTI 长序列运行耗时和显存压力明显更高，不作为本地 Mac 验收项。

## 1. 上传位置与预检

通过 AutoDL 文件管理器把 KITTI 09 放到以下固定结构：

```text
~/autodl-tmp/VGGT-SLAM/data/09/
├── image_2/
├── image_3/
├── calib.txt
├── times.txt
└── poses.txt
```

`image_2` 是本手册实际使用的左彩色相机。`poses.txt` 必须来自 KITTI
odometry 官方真值，不能用算法输出替代。运行前检查帧数和首尾文件：

```bash
test "$(find data/09/image_2 -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 1591
test "$(wc -l < data/09/times.txt)" -eq 1591
test "$(wc -l < data/09/poses.txt)" -eq 1591
test -f data/09/image_2/000000.png
test -f data/09/image_2/001590.png
```

任一命令失败都应先修复上传内容，不要开始推理。创建引用目录并把 KITTI
每行 `3 x 4` 位姿转换为 TUM 八列格式：

```bash
mkdir -p artifacts/reference

python -m vggt_slam_pp.cli.convert_kitti_trajectory \
  --input data/09/poses.txt \
  --output artifacts/reference/kitti09-groundtruth-tum.txt
```

转换后的 `timestamp` 是从 `0` 开始的帧号。原因是 VGGT-SLAM 从
`000000.png` 这类图像文件名提取帧号写入估计轨迹；不能把 `times.txt` 中的
秒数混入 ATE。`times.txt` 在这里仅用于数据完整性检查。

## 2. 无回环桥接与子图缓存

该运行验证 VGGT-SLAM++ 的前端桥接入口，不调用 SALAD/DINOv2，也不把原版
回环结果混入待实现的后端。每次运行都生成新的编号：

```bash
RUN_ID=kitti09-pp-bridge-$(date +%Y%m%d-%H%M)
mkdir -p "artifacts/m0/$RUN_ID" "artifacts/m0/submaps"

python VGGT-SLAM-version1.0/main.py \
  --image_folder data/09/image_2 \
  --vggt_weight weights/model.pt \
  --device cuda --use_sim3 --submap_size 32 --max_loops 0 \
  --run_id "$RUN_ID" --run_purpose pp_frontend_bridge \
  --export_submaps_dir artifacts/m0/submaps \
  --log_results --skip_dense_log \
  --log_path "artifacts/m0/$RUN_ID/poses.txt"
```

运行完成后立即检查缓存并评测：

```bash
test -s "artifacts/m0/$RUN_ID/poses.txt"

python -m vggt_slam_pp.cli.inspect_submap_cache \
  "artifacts/m0/submaps/$RUN_ID" \
  > "artifacts/m0/$RUN_ID/cache-inspection.json"

python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth artifacts/reference/kitti09-groundtruth-tum.txt \
  --estimate "artifacts/m0/$RUN_ID/poses.txt" \
  --output "artifacts/m0/$RUN_ID/ate.json"
```

保存当前 `RUN_ID`。后续打包命令依赖它，不要复用编号覆盖缓存。

## 3. Sim(3) + SALAD 回环基线

该运行保留原版 VGGT-SLAM 的 DINOv2/SALAD 回环流程，作为后续
VGGT-SLAM++ VPR 和全局优化的对照。它需要已审核的本地源码与三份权重：

```bash
RUN_ID=kitti09-sim3-salad-$(date +%Y%m%d-%H%M)
mkdir -p "artifacts/m0/$RUN_ID"

python VGGT-SLAM-version1.0/main.py \
  --image_folder data/09/image_2 \
  --vggt_weight weights/model.pt \
  --salad_checkpoint weights/dino_salad.ckpt \
  --dinov2_source external_sources/dinov2 \
  --dinov2_weight weights/dinov2_vitb14_pretrain.pth \
  --device cuda --use_sim3 --submap_size 32 --max_loops 1 \
  --run_id "$RUN_ID" --run_purpose baseline_reference \
  --log_results --skip_dense_log \
  --log_path "artifacts/m0/$RUN_ID/poses.txt"

test -s "artifacts/m0/$RUN_ID/poses.txt"

python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth artifacts/reference/kitti09-groundtruth-tum.txt \
  --estimate "artifacts/m0/$RUN_ID/poses.txt" \
  --output "artifacts/m0/$RUN_ID/ate.json"
```

从终端末尾记录 `Total number of submaps in map` 和
`Total number of loop closures in map`。M0 只如实记录检测到的回环数量；
在实际运行和人工检查前，不预设 KITTI 09 的回环数必须大于零。

## 4. 人工验收

每个运行逐项确认：

- 预检找到 1591 张 `image_2` 输入图像；
- 推理过程没有 traceback，进程正常返回；
- `poses.txt` 存在且非空；
- `ate.json` 中两套 `translation_ape_rmse` 都是有限值；
- `associated_pose_count` 至少为 10；若明显偏少，先检查帧号时间戳；
- 桥接的 `cache-inspection.json` 报告 `"ok": true`；
- 人工抽查第一个、中间和最后一个 `submaps` 项，所有数组均为
  `"finite": true`，形状与 dtype 合理；
- 三个抽查子图的 `unit_state` 都是 `relative_map_unit`。此时尺度仍是
  VGGT 相对地图单位，不能解释为米；
- 桥接最终图状态的 `loop_count` 为 0；
- SALAD 基线的子图数、回环数和 ATE 已抄录到本次审核记录。

ATE 字段和两套报告的解释见 [`../algorithms/ate.md`](../algorithms/ate.md)。
缓存单位是相对尺度；后续 DEM 的米/像素分辨率必须经过场景尺度策略或外部
度量信息确定，不能从 M0 缓存字段直接假定。

## 5. 打包回传

桥接缓存体积较大，不提交到 Git。保持桥接运行的 `RUN_ID`，严格按
[`cloud-artifact-transfer.md`](cloud-artifact-transfer.md) 生成逐文件校验
清单、压缩包和压缩包 SHA-256，再通过 AutoDL 文件管理器下载到本地。

SALAD 基线当前没有导出子图缓存。其 `poses.txt`、`ate.json` 和小型审核记录
可单独归档；不要为了共用桥接打包命令而伪造空的 `submaps` 目录。
