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
set -euo pipefail

RUN_ID=kitti09-pp-bridge-$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/m0 artifacts/m0/submaps
test ! -e "artifacts/m0/$RUN_ID"
test ! -e "artifacts/m0/submaps/$RUN_ID"
mkdir "artifacts/m0/$RUN_ID"

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
set -euo pipefail

RUN_ID=kitti09-sim3-salad-$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/m0
test ! -e "artifacts/m0/$RUN_ID"
mkdir "artifacts/m0/$RUN_ID"

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

## 6. 自主运行不同数据集与方法

本节用于反复执行方法对比，不导出子图缓存。每次实验自动创建独立的
`artifacts/m0/<run-id>/`，其中保存：

```text
poses.txt    VGGT-SLAM 输出的 TUM 八列估计轨迹
ate.json     Sim(3) 对齐后的 ATE 和重复边界帧诊断
```

### 6.1 可选数据集

| `DATASET` | 图像 | 真值 | 说明 |
| --- | --- | --- | --- |
| `kitti09` | `data/09/image_2` | `data/09/poses.txt` | 1591 帧长回环序列 |
| `tum-desk` | `data/rgbd_dataset_freiburg1_desk/rgb` | 同目录 `groundtruth.txt` | 室内短序列 |

KITTI 真值会由脚本转换为使用图像帧号的 TUM 八列格式。Office-loop 没有
配套真值，可检查流程和回环数量，但不能使用本项目命令计算 ATE。

### 6.2 可选方法

| `METHOD` | 几何路径 | 点过滤 | SL(4) 求解器 |
| --- | --- | --- | --- |
| `sl4-original` | SL(4) | `legacy` | 原始 RANSAC |
| `sl4-ransac-joint` | SL(4) | `joint` | RANSAC，固定随机种子 |
| `sl4-irls-joint` | SL(4) | `joint` | RANSAC 初始化后 IRLS 精修 |
| `sim3` | Sim(3) | 原版 Sim(3) 对应点逻辑 | 不调用 SL(4) 求解器 |

`sl4-original` 用于复现修改前的默认行为。公平判断 IRLS 本身是否有效时，
必须比较 `sl4-ransac-joint` 和 `sl4-irls-joint`，并保持数据集、随机种子、
子图大小、回环数量和阈值完全一致。

### 6.3 单次实验模板

从云端仓库根目录执行。通常只修改开头的六个实验变量：

```bash
cd ~/autodl-tmp/VGGT-SLAM
set -euo pipefail

# ===== 实验变量：每次主要修改这里 =====
DATASET=kitti09
METHOD=sl4-ransac-joint
LOOPS=0
SUBMAP_SIZE=32
SEED=7
PROJECTIVE_THRESHOLD=0.01
# =====================================

case "$DATASET" in
  kitti09)
    IMAGE_DIR=data/09/image_2
    GT=artifacts/reference/kitti09-groundtruth-tum.txt
    mkdir -p artifacts/reference
    python -m vggt_slam_pp.cli.convert_kitti_trajectory \
      --input data/09/poses.txt \
      --output "$GT"
    ;;
  tum-desk)
    IMAGE_DIR=data/rgbd_dataset_freiburg1_desk/rgb
    GT=data/rgbd_dataset_freiburg1_desk/groundtruth.txt
    ;;
  *)
    echo "未知数据集: $DATASET" >&2
    exit 1
    ;;
esac

METHOD_ARGS=()
case "$METHOD" in
  sl4-original)
    METHOD_ARGS=(
      --projective_solver ransac
      --projective_confidence_mode legacy
      --projective_threshold "$PROJECTIVE_THRESHOLD"
    )
    ;;
  sl4-ransac-joint)
    METHOD_ARGS=(
      --projective_solver ransac
      --projective_confidence_mode joint
      --projective_threshold "$PROJECTIVE_THRESHOLD"
      --projective_seed "$SEED"
    )
    ;;
  sl4-irls-joint)
    METHOD_ARGS=(
      --projective_solver ransac_irls
      --projective_confidence_mode joint
      --projective_threshold "$PROJECTIVE_THRESHOLD"
      --projective_seed "$SEED"
      --irls_max_iterations 10
    )
    ;;
  sim3)
    METHOD_ARGS=(--use_sim3)
    ;;
  *)
    echo "未知方法: $METHOD" >&2
    exit 1
    ;;
esac

LOOP_ARGS=()
if [ "$LOOPS" -gt 0 ]; then
  LOOP_ARGS=(
    --salad_checkpoint weights/dino_salad.ckpt
    --dinov2_source external_sources/dinov2
    --dinov2_weight weights/dinov2_vitb14_pretrain.pth
  )
fi

RUN_ID="${DATASET}-${METHOD}-loops${LOOPS}-$(date +%Y%m%d-%H%M%S)"
OUT="artifacts/m0/$RUN_ID"
mkdir -p "$OUT"

python VGGT-SLAM-version1.0/main.py \
  --image_folder "$IMAGE_DIR" \
  --vggt_weight weights/model.pt \
  --device cuda \
  --submap_size "$SUBMAP_SIZE" \
  --max_loops "$LOOPS" \
  --run_id "$RUN_ID" \
  --run_purpose baseline_reference \
  --log_results \
  --skip_dense_log \
  --log_path "$OUT/poses.txt" \
  "${METHOD_ARGS[@]}" \
  "${LOOP_ARGS[@]}"

test -s "$OUT/poses.txt"

python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth "$GT" \
  --estimate "$OUT/poses.txt" \
  --output "$OUT/ate.json"

python -c '
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
canonical = report["canonical_unique"]
paper = report["paper_compatible"]
duplicates = report["duplicate_diagnostics"]["estimate"]
print("ATE 文件:", path)
print("canonical_unique ATE:", canonical["translation_ape_rmse"], "m")
print("paper_compatible ATE:", paper["translation_ape_rmse"], "m")
print("关联位姿数:", canonical["associated_pose_count"])
print("重复边界诊断:", duplicates)
' "$OUT/ate.json"

echo "RUN_ID=$RUN_ID"
echo "结果目录: $OUT"
```

`PROJECTIVE_THRESHOLD=0.01` 使用 VGGT 的相对地图单位，不是米。比较方法时
先保持该值不变；只有建立明确的尺度策略后，才能按物理距离解释或跨场景
统一调整。

### 6.4 建议实验矩阵

先在同一个数据集上依次运行下列组合。每次修改 `METHOD` 和 `LOOPS` 后，
重新执行完整模板：

| 目的 | `METHOD` | `LOOPS` |
| --- | --- | ---: |
| 修改前 SL(4) 参考 | `sl4-original` | `0` |
| IRLS 公平控制组 | `sl4-ransac-joint` | `0` |
| IRLS 实验组 | `sl4-irls-joint` | `0` |
| 原版 Sim(3) 无回环 | `sim3` | `0` |
| 原版 Sim(3) + SALAD | `sim3` | `1` |
| SL(4) IRLS + SALAD | `sl4-irls-joint` | `1` |

不要只根据 ATE 判断运行质量。还应记录终端末尾的子图数、回环数，以及
`duplicate_diagnostics.estimate.max_rotation_deg`。接近 `180` 度的边界
旋转差说明子图连接可能发生翻转，即使 ATE 略有下降也需要继续排查。

### 6.5 汇总已有实验

下面的命令只读取小型 `ate.json`，不会重新运行 VGGT：

```bash
cd ~/autodl-tmp/VGGT-SLAM

for file in artifacts/m0/*/ate.json; do
  python -c '
import json
import os
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
result = report["canonical_unique"]
print(
    os.path.basename(os.path.dirname(path)),
    "ATE =", round(result["translation_ape_rmse"], 6), "m",
    "poses =", result["associated_pose_count"],
)
' "$file"
done
```

主要对比 `canonical_unique.translation_ape_rmse`，数值越低越好。同时确认
各组 `associated_pose_count` 一致或足够接近；关联帧数量明显不同时，ATE
不能直接作为公平对比。
