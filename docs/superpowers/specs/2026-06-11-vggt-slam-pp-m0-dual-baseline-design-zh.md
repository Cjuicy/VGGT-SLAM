# VGGT-SLAM++ M0 双模式基线与桥接修订规格

日期：2026-06-11

状态：用户已确认设计方向，等待书面规格审核。

本规格修订 M0，不改变 M1-M4 的阶段边界。若本规格与早期 M0
计划冲突，以本规格为准。

## 1. 修订目标

M0 必须同时证明以下四件事：

1. 新提供的原始 VGGT-SLAM 源码可以在 AutoDL CUDA 环境运行。
2. 默认 SL(4) 和源码中的 `--use_sim3` 路径可以被明确区分和评测。
3. VGGT-SLAM++ 所需的前端数据能够从最小桥接接口稳定导出。
4. 基线轨迹可用论文相同核心指标 ATE RMSE 进行重复评测。

M0 不实现 VGGT-SLAM++ 的 7DoF Sim(3) 后端。源码中的
`--use_sim3` 只作为前端兼容路径和 VGGT-SLAM 论文对照。

## 2. 基线源码迁移

### 2.1 唯一基线

新提供的压缩包是唯一原始基线：

```text
VGGT-SLAM-version1.0.zip
SHA-256:
f34897e5745c6380dfd819bf87c8a016aebb8e9ffe7a0025304015fa7b0f0411
```

经逐文件比较，压缩包内容与临时目录 `VGGT-SLAM-version1.0 2/`
一致。迁移后使用稳定路径：

```text
VGGT-SLAM-version1.0/
```

目录名中不保留空格和数字后缀，保证本地、GitHub 和 AutoDL 命令一致。

### 2.2 旧修改目录

当前 `VGGT-SLAM-version1.0/` 含有人工注释、`main_offline.py`、
内嵌 VGGT 源码、解压演示数据和临时文档，不再作为基线。

迁移前执行差异清单。以下内容不进入正式源码：

- `findings.md`
- `progress.md`
- `task_plan.md`
- `main_offline.py`
- 内嵌的 `vggt/`
- 解压后的 `office_loop/`

仍然有效的技术结论必须写入 `docs/provenance/` 或本规格，不通过保留
修改版源码来传递。确认新基线与压缩包一致后，删除旧修改目录，并把
新基线移动到规范路径。

### 2.3 Git 边界

Git 跟踪基线 Python 源码、配置、README 和 LICENSE。以下内容始终忽略：

- `weights/`
- `data/`
- `artifacts/`
- 基线压缩包
- `office_loop/` 及其压缩包
- 本地克隆的 VGGT、SALAD 和 GTSAM 源码
- 模型权重、演示 GIF、缓存和运行日志

根 `.gitignore` 是安全边界，任何运行前先用 `git check-ignore` 验证。

## 3. 模式命名

### 3.1 `baseline_sl4`

对应源码默认路径：

```bash
python main.py
```

特征：

- 子图间使用 5 点 RANSAC 估计 15DoF 3D projective homography。
- 图节点为 `gtsam.SL4`。
- 使用 `PriorFactorSL4` 和 `BetweenFactorSL4`。
- 需要包含 SL(4) Python 绑定的 GTSAM 构建。

该模式用于复现 VGGT-SLAM 的 SL(4) 对照，但不是 VGGT-SLAM++ 前端。

### 3.2 `baseline_sim3_compat`

对应源码路径：

```bash
python main.py --use_sim3
```

源码行为是：

1. 用重叠帧点云估计一个尺度因子。
2. 把尺度直接乘入当前子图点云和相机平移。
3. 使用 6DoF `gtsam.Pose3` 图优化旋转和平移。

因此该模式不能称为“7DoF Sim(3) 图优化”。正式文档、缓存元数据和
实验记录统一使用 `baseline_sim3_compat`。

该路径与 VGGT-SLAM 论文描述的简化 Sim(3) 对照一致：尺度在图外估计，
图中添加 SE(3) 因子。它也是 M0 中最接近 VGGT-SLAM++ Sim(3) 前端的
可运行起点，但后续 M4 不能直接复用其 Pose3 图。

### 3.3 `local_sim3_backend`

该名称仅保留给 M4 实现的真正 7DoF Sim(3) 状态和因子图。M0 不实现它，
也不能用 `graph_se3.py` 冒充它。

## 4. 求解模式与运行目的分离

求解模式和回环策略是两个独立维度。

### 4.1 `baseline_reference`

用于复现 VGGT-SLAM 原始行为：

- `max_loops=1`
- TUM 正式比较使用 `submap_size=32`
- `min_disparity=50`
- 分别运行 `baseline_sl4` 和 `baseline_sim3_compat`

该配置输出 VGGT-SLAM 基线 ATE。

### 4.2 `pp_frontend_bridge`

用于生成 VGGT-SLAM++ 后端输入：

- 求解模式固定为 `baseline_sim3_compat`
- `max_loops=0`
- `submap_size=32`
- 原始 SALAD 回环不得向前端图添加边
- DEM/VPR/空间回环将在 M1-M4 旁路实现

VGGT-SLAM++ 论文报告的 disparity 阈值文本存在单位歧义。M0 默认保留
源码的 `50`，并在后续论文对齐实验中把 `40` 作为单独命名配置评测，
不能静默替换。

## 5. M0 最小运行矩阵

### 5.1 Office-loop 冒烟测试

运行 `baseline_sl4 + baseline_reference`：

- 导出关闭运行一次。
- 导出开启运行一次。
- 子图数相同。
- 回环数相同。
- TUM 格式位姿逐项数值一致。

该测试只证明桥接不改变基线行为，不计算 ATE，因为 office-loop
没有正式真值。

### 5.2 TUM Freiburg1 desk

使用相同 RGB 输入和真值执行：

1. `baseline_sl4 + baseline_reference`
2. `baseline_sim3_compat + baseline_reference`
3. `baseline_sim3_compat + pp_frontend_bridge`

三个运行均输出：

- 轨迹文件。
- ATE RMSE。
- 帧数、关联成功数量、子图数和回环数。
- 环境、权重和命令记录。

前两个运行验证 VGGT-SLAM 基线；第三个运行验证 VGGT-SLAM++ 前端桥接。

### 5.3 长序列

后续获得 KITTI 数据后，第一条长回环序列使用 KITTI Odometry 09。
论文明确把 Sequence 09 描述为完整回环，适合检查全局一致性和空间回环。

KITTI Odometry 06 作为第二条压力测试：论文报告路径约 1230 m，适合观察
长距离漂移，但不是首选回环验收序列。

## 6. 基线兼容修改边界

原始源码存在会影响可复现运行的环境耦合：

- VGGT 权重通过 URL 在运行时下载。
- SALAD 通过 `torch.hub` 在运行时下载。
- SALAD checkpoint 路径固定在 Torch Hub 缓存。
- SALAD 设备固定为 `cuda`。
- `solver.py` 缺少 `Dict` 和 `List` 导入。
- CPU 环境仍调用 `torch.cuda.get_device_capability()`。
- 普通 `gtsam==4.2.1` 不含 SL(4) Python 符号。

允许的基线改动仅限：

1. 增加显式本地权重、checkpoint、设备和导出路径参数。
2. 修复缺失类型导入和无 CUDA 时的明确错误。
3. 将依赖可用性检查提前到推理前。
4. 在图更新后调用只读导出适配器。

禁止改变：

- 关键帧选择算法。
- VGGT 输入帧顺序。
- RANSAC 求解。
- 尺度估计公式。
- 图因子、噪声和优化顺序。
- 原始回环候选和阈值。

每项兼容修改都必须有禁用路径测试和云端 A/B 输出比较。

## 7. 桥接缓存

### 7.1 导出数据源

桥接不能只读取 `predictions`。

在 `baseline_sim3_compat` 中，尺度是在 `Solver.add_points()` 内乘入
`world_points` 和相机平移，原始 `predictions["depth"]` 不包含这个最终
尺度。只导出 predictions 会使 DEM 使用错误尺度。

正式导出数据源是完成 `add_points()` 后的 `Submap`：

- `Submap.pointclouds`
- `Submap.colors`
- `Submap.conf`
- `Submap.conf_threshold`
- `Submap.vggt_intrinscs`
- `Submap.poses`
- `Submap.frame_ids`
- `Submap.last_non_loop_frame_index`
- `Submap.H_world_map`

适配器只读这些字段，不修改数组或基线对象。

### 7.2 不可变子图载荷

每个子图只写一次：

```text
artifacts/submaps/tum-desk-pp-bridge/submaps/000000/
├── metadata.json
├── geometry.npz
└── checksums.json
```

`geometry.npz` 至少包含：

- `points_submap`: `(S,H,W,3)`
- `colors_rgb`: `(S,H,W,3)`
- `confidence`: `(S,H,W)`
- `confidence_mask`: `(S,H,W)`
- `intrinsics`: `(S,3,3)`
- `camera_to_submap`: `(S,4,4)`

`metadata.json` 至少包含：

- schema 版本。
- 求解模式和运行目的。
- 帧 ID 与最后一个非回环帧索引。
- 坐标系和变换方向。
- 单位状态 `relative_map_unit`。
- `scale_baked_into_geometry`。
- 基线、权重、运行配置的 SHA-256。
- 回环来源列表；`pp_frontend_bridge` 中必须为空。

### 7.3 可变图状态

全局图优化可能更新已存在子图的参考变换，因此不能把
`H_world_map` 仅固化在首次导出的巨型 NPZ 中。

每次图更新写一个小型状态快照：

```text
artifacts/submaps/tum-desk-pp-bridge/states/000003.json
artifacts/submaps/tum-desk-pp-bridge/final_state.json
```

状态文件包含：

- 图更新序号。
- 每个子图的 `world_from_submap` 变换。
- 变换类型 `SL4` 或 `SE3_with_baked_scale`。
- 图边计数和回环计数。
- 当前完整 TUM 轨迹校验和。

M1 默认读取 `final_state.json`。这使局部几何保持不可变，同时准确反映
后续图优化。

## 8. ATE 评测契约

核心指标为平移 Absolute Pose Error 的 RMSE，单位为米。

TUM 和单目估计使用 evo 的时间关联、Sim(3) 对齐和尺度校正：

```bash
evo_ape tum groundtruth.txt estimated.txt -as
```

其中：

- `-a` 对轨迹执行 Umeyama 对齐。
- `-s` 校正单目尺度。
- 报告字段以 evo 的 translation APE `rmse` 为准。

基线的 `GraphMap.write_poses_to_file()` 会同时写出前一子图末帧和下一
子图继承的同一个过渡帧，因此原始日志可能含重复时间戳。M0 同时报告：

1. `paper_compatible`：不改写原始日志，执行官方基线脚本相同的
   `evo_ape ... -as`，用于与论文及官方仓库结果比较。
2. `canonical_unique`：按文件顺序保留重复时间戳的第一次出现，生成
   时间戳唯一的审计轨迹，再执行相同 ATE；同时记录每个重复过渡帧的
   两个位置估计之间的平移和旋转差异。

原始日志始终保留，去重文件不能覆盖它。评测前必须验证：

- 原始估计时间戳有限且非递减。
- canonical 估计时间戳唯一且严格递增。
- 四元数顺序为 `qx qy qz qw`。
- 平移和四元数均为有限值。
- 关联成功位姿不少于 10 个。
- 记录 evo 版本和完整参数。

机器可读结果写入：

```text
artifacts/evaluation/tum-desk-sl4/ate.json
artifacts/evaluation/tum-desk-sim3-compat/ate.json
artifacts/evaluation/tum-desk-pp-bridge/ate.json
```

JSON 分别包含 `paper_compatible` 和 `canonical_unique` 的 `rmse_m`、
mean、median、std、min、max、关联数量、估计数量、真值数量、对齐模式
和输入文件 SHA-256，并保存重复时间戳数量及边界位姿差异统计。小型摘要
复制到 `docs/reviews/m0/` 供人工审核。

论文表格数值只作为参考目标，不作为首次云端运行的硬性通过阈值。
M0 的硬性要求是命令成功、格式正确、结果有限、可重复并有完整来源记录。

## 9. 本地与云端职责

### 9.1 本地 macOS

使用 `vggt-dem`：

- 运行数据契约、序列化、校验和与 ATE 合成测试。
- 静态检查基线桥接为 opt-in。
- 验证模型和数据资产的路径与 SHA-256。
- 不把完整 VGGT、DINOv2 或 SL(4) 推理作为本地验收条件。

已验证本地状态：

- Torch 无 CUDA。
- MPS 当前不可用。
- `gtsam==4.2.1` 有 Pose3，但无 SL4。
- evo 已安装。
- VGGT 和 SALAD 尚未安装。

### 9.2 AutoDL

云端保持与本地相同项目相对路径。需要手工上传：

```text
weights/model.pt
weights/dino_salad.ckpt
data/rgbd_dataset_freiburg1_desk/
VGGT-SLAM-version1.0/office_loop/
```

SL(4) 运行还需要安装具有以下 Python 符号的 GTSAM：

```text
SL4
PriorFactorSL4
BetweenFactorSL4
```

预检缺少任何资产或符号时立即失败，不自动下载，也不静默切换模式。

## 10. M0 审核门

进入 M1 前，用户需要审核：

1. 新基线来源、SHA-256 和旧目录删除清单。
2. 基线允许修改文件清单。
3. `baseline_sl4` 与 `baseline_sim3_compat` 的命名和实际行为。
4. Submap 导出数组、坐标系、尺度状态和最终图状态。
5. 禁用导出与启用导出的 office-loop A/B 比较。
6. 三个 TUM desk 运行的 ATE JSON 和命令记录。
7. 本地与 AutoDL 环境差异及已知偏差。
8. Git 对象库大小保持稳定，没有权重或数据进入历史。

M0 未经人工审核时，不开始 M1 DEM 实现。
