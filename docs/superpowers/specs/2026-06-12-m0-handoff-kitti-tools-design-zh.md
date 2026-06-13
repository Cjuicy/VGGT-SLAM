# M0 收尾、KITTI 09 与工具文档整理设计

## 1. 目标

本轮工作完成 M0 云端验证后的本地交接，并为下一阶段 KITTI 09 长序列
验证建立可重复、可人工审核的数据流。范围包括：

1. 明确 AutoDL 产物如何打包、下载、校验和本地归档。
2. 为 KITTI Odometry 09 编写上传、运行、缓存导出和 ATE 评测手册。
3. 补充 ATE 公式、实现调用链、JSON 字段解释和关键中文代码注释。
4. 保持 VGGT-SLAM 基线改动最小，不在本轮修改 SLAM 算法行为。

## 2. 文档结构

采用分层文档，避免把工具、算法和运行命令混入同一个超长 runbook。

```text
README.md
docs/
├── algorithms/
│   └── ate.md
├── runbooks/
│   ├── cloud-artifact-transfer.md
│   ├── m0-autodl-baseline.md
│   ├── m0-kitti09-autodl.md
│   └── m0-local-validation.md
```

- `README.md` 只提供项目入口、阶段状态和文档导航。
- `runbooks` 只描述可直接执行的操作步骤。
- `algorithms` 解释数学定义、实现选择和指标含义。

## 3. 云端产物回传

### 3.1 原则

- Git 只传输源码、配置和小型文档。
- 权重、数据集、子图缓存和运行产物不进入 Git。
- 每次云端运行使用唯一 `run_id`，不可覆盖已有不可变缓存。
- 云端将一次运行的完整目录打包为一个 `tar.gz`。
- 本地保留原始压缩包和解压后的只读归档，方便复查校验和。

### 3.2 目录约定

云端：

```text
artifacts/
├── m0/
│   └── <run-id>/
├── m0/submaps/
│   └── <run-id>/
└── packages/
    └── <run-id>.tar.gz
```

本地：

```text
artifacts/
├── packages/
│   └── <run-id>.tar.gz
└── imported/
    └── <run-id>/
```

压缩包同时包含：

- 位姿文件；
- ATE JSON；
- 子图缓存；
- graph state；
- 运行摘要；
- SHA-256 清单。

下载使用 AutoDL 文件管理器，不依赖 Git、Git LFS 或未配置的 SSH。

## 4. KITTI 09 数据与运行

### 4.1 本地源数据

本地数据目录固定为：

```text
data/09/
├── image_2/       # 1591 帧左彩色图像
├── image_3/       # 右彩色图像，本轮不使用
├── calib.txt
├── times.txt
└── poses.txt      # KITTI 官方 3x4 真值，每行 12 个数
```

VGGT-SLAM 单目输入使用 `image_2/`。`image_3/` 和 `calib.txt` 在 M0 前端
运行中不参与推理，但保留为数据集来源记录。

### 4.2 云端上传位置

云端保持与本地一致：

```text
~/autodl-tmp/VGGT-SLAM/data/09/
```

runbook 在运行前校验：

- `image_2/` 恰有 1591 个图像文件；
- `times.txt` 恰有 1591 行；
- `poses.txt` 恰有 1591 行且每行 12 个有限数值；
- 首尾图像文件名为 `000000.png` 和 `001590.png`。

### 4.3 运行层次

KITTI 09 分两层验证：

1. **M0 前端桥接缓存**
   - 使用 Sim(3) 兼容模式；
   - 关闭 SALAD 回环，验证长序列子图导出稳定性；
   - 生成全部子图缓存和前端轨迹。
2. **VGGT-SLAM 基线回环**
   - 使用现有 SALAD 回环；
   - 记录子图数、回环数和 ATE；
   - 作为后续 VGGT-SLAM++ 后端的对照。

本轮文档提供命令和验收方式，不把 KITTI 特定分支写入 SLAM 主循环。

## 5. KITTI 真值转换与 ATE

### 5.1 转换工具

新增独立 CLI，将 KITTI 每行 `3x4` 位姿转换为项目统一使用的 TUM 格式：

```text
timestamp tx ty tz qx qy qz qw
```

输入：

- `data/09/poses.txt`

输出：

```text
artifacts/reference/kitti09-groundtruth-tum.txt
```

转换器职责仅限：

- 校验每行 12 个有限数值；
- 补齐齐次矩阵最后一行；
- 将旋转矩阵转换为归一化四元数；
- 保持输入顺序；
- 将零基行号写为时间戳列，即 `0, 1, ..., 1590`；
- 创建输出父目录。

它不进行坐标系猜测、尺度变换或轨迹对齐。

这里不能直接使用 `times.txt` 的秒时间戳。当前 VGGT-SLAM
`Submap.set_frame_ids()` 从图像文件名提取数字，因此 KITTI 估计轨迹第一列
也是帧号 `0..1590`。真值必须使用相同键才能在默认 `0.01` 关联阈值下正确
匹配。`times.txt` 仍在 runbook 中用于验证数据集完整性，但不参与本轮 ATE
轨迹转换。

### 5.2 ATE 实现说明

ATE 继续使用 evo Python API：

1. 根据时间戳最近邻关联真值和估计位姿；
2. 最大时间差默认 `0.01 s`；
3. 使用 Umeyama Sim(3) 对齐并校正单目尺度；
4. 计算每个关联位姿的平移误差：

   \[
   e_i =
   \left\|
   \mathbf{t}^{gt}_i -
   \left(s\mathbf{R}\mathbf{t}^{est}_i+\mathbf{t}\right)
   \right\|_2
   \]

5. 报告：

   \[
   \operatorname{ATE}_{RMSE} =
   \sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}
   \]

JSON 中继续同时记录：

- `paper_compatible`：保留子图边界重复时间戳；
- `canonical_unique`：每个时间戳只保留第一条；
- `associated_pose_count`：实际参与评测的关联位姿数；
- `duplicate_diagnostics`：重复边界位姿的最大分歧；
- 输入文件绝对路径和 SHA-256。

人工记录主指标使用 `paper_compatible.translation_ape_rmse`，同时检查
`canonical_unique` 与关联数量，避免只抄一个 RMSE 而忽略数据质量。

### 5.3 代码注释范围

只在以下关键位置添加简洁中文注释：

- `vggt_slam_pp/evaluation/ate.py`
  - 时间关联；
  - Sim(3) 对齐；
  - translation APE RMSE。
- `vggt_slam_pp/evaluation/tum.py`
  - TUM 八列格式；
  - 重复时间戳来源；
  - 四元数符号等价性；
  - 规范轨迹保留首项的规则。
- `vggt_slam_pp/cli/evaluate_ate.py`
  - 原始与去重两套报告的用途；
  - 临时规范轨迹仅用于评测，不写入正式产物。

不在显而易见的赋值、文件打开或参数解析处堆叠注释。

## 6. 仓库清理

### 6.1 删除

只删除操作系统生成的 `.DS_Store` 文件。它们不包含项目信息，且已有
`.gitignore` 规则。

### 6.2 不删除

- `data/03`、`data/09`、权重和论文；
- 用户尚未审核的 M1/M2-M4 计划。

## 7. 测试与验收

### 7.1 自动测试

- KITTI 位姿转换：
  - 正确转换单位旋转和已知旋转；
  - 拒绝行数不一致；
  - 拒绝非 12 列或非有限数值；
  - 输出为合法 TUM 八列格式。
- ATE：
  - 现有 Sim(3) 对齐测试继续通过；
  - 中文注释不改变行为；
  - KITTI 转换后的合成轨迹可得到接近零的 ATE。
- 全量测试保持通过。

### 7.2 文档验收

- 新用户可仅按 KITTI 09 runbook 完成上传、运行、评测、打包和下载。
- 所有命令创建输出父目录，避免再次发生 `FileNotFoundError`。
- runbook 明确每次使用新 `run_id`，避免不可变缓存覆盖错误。
- 云端包内存在 SHA-256 清单，本地解压前后可验证。
- ATE 文档能回答“算了什么、为什么使用 Sim(3)、主看哪个字段”。

## 8. 非目标

本轮不实现：

- VGGT-SLAM++ 的 DEM、DINOv2 DEM tile 嵌入、FAISS-HNSW 或全局后端；
- KITTI 专用尺度或 tile 参数调优；
- 立体视觉；
- KITTI 官方序列误差指标；
- 将大型运行产物上传 Git；
- 为 CUDA 基线强行保证逐元素确定性。
