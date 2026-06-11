# VGGT-SLAM++ 复现设计规格（中文版）

日期：2026-06-11

英文对照版：
`docs/superpowers/specs/2026-06-11-vggt-slam-pp-reproduction-design.md`

M0 的基线替换、双模式验收、桥接缓存和 ATE 细节以以下修订规格为准：
`docs/superpowers/specs/2026-06-11-vggt-slam-pp-m0-dual-baseline-design-zh.md`

## 1. 项目目标

本项目将在已提供的 VGGT-SLAM 1.0 源码基础上，分阶段复现
VGGT-SLAM++。整个过程中需要保持基线代码清晰、易懂，并尽可能减少
对基线源码的修改。

实现过程必须满足：

- 每一个里程碑都经过人工代码审核。
- 每一个算法阶段都能够离线重放。
- 明确区分论文直接定义的行为、从外部源码迁移的行为和实验性假设。
- 本地开发使用 macOS 上已经存在的 `vggt-dem` Conda 环境。
- 完整推理和性能实验在 AutoDL CUDA 环境运行。
- 通过 Git 在本地和云端之间同步代码，但不提交数据集、权重、缓存和大型生成文件。

第一轮不会直接实现完整在线系统。首要目标是：

1. 冻结并跑通 VGGT-SLAM 基线。
2. 建立可人工审查的数据契约。
3. 独立实现并验证 DEM 流水线。

## 2. 复现证据标签

每一个非平凡算法和重要配置字段，都必须在文档和实验元数据中标记为以下类别之一：

- `paper_exact`：论文正文或附录提供了足够细节，可以直接按论文实现。
- `source_ported`：算法从放入 `external_sources/` 的引用论文官方源码中迁移。
- `experimental`：论文缺少实现细节，本项目采用可替换的实验性假设。

任何 `experimental` 实现都不能被描述为论文作者的原始实现。

## 3. 总体架构决定

采用“基线隔离 + 旁路扩展包”的结构：

```text
VGGT-SLAM-version1.0/    固定的 VGGT-SLAM 基线快照及最小导出桥接
vggt_slam_pp/            新增的 VGGT-SLAM++ 实现
configs/                 数据集和运行环境配置
environment/             本地 macOS 与 AutoDL CUDA 环境定义
external_sources/        被 Git 忽略的第三方源码及被跟踪的来源信息
artifacts/               离线缓存和生成结果，默认被 Git 忽略
tests/                   合成测试、单元测试和集成测试
docs/                    设计、算法说明、来源记录和运行手册
```

`VGGT-SLAM-version1.0/` 始终作为基线。DEM、检索、VPR 和 Sim(3)
后端代码不能直接写入基线包。

只有在无法从基线外部取得必要数据时，才允许增加很小的导出接口。
每一次基线修改必须记录：

- 为什么必须修改。
- 输出被论文流程中的哪个阶段使用。
- 输入和输出的形状、单位及坐标系。
- 为什么该改动不会改变基线原有估计行为。

## 4. 建议目录结构

```text
.
├── VGGT-SLAM-version1.0/
├── vggt_slam_pp/
│   ├── contracts/
│   │   ├── submap.py
│   │   ├── scale.py
│   │   ├── dem.py
│   │   ├── retrieval.py
│   │   └── graph.py
│   ├── adapters/
│   │   └── vggt_slam_v1.py
│   ├── geometry/
│   │   ├── depth_filter.py
│   │   ├── plane.py
│   │   ├── canonical_frame.py
│   │   └── sim3_measurement.py
│   ├── dem/
│   │   ├── lattice.py
│   │   ├── rasterizer.py
│   │   ├── reducers.py
│   │   ├── normalization.py
│   │   └── visualization.py
│   ├── embedding/
│   │   ├── dinov2.py
│   │   ├── weighting.py
│   │   └── signatures.py
│   ├── retrieval/
│   │   ├── hnsw.py
│   │   ├── voting.py
│   │   └── candidate_graph.py
│   ├── vpr/
│   │   ├── anyloc.py
│   │   └── verifier.py
│   ├── backend/
│   │   ├── interface.py
│   │   ├── sim3_graph.py
│   │   └── scheduler.py
│   └── cli/
├── configs/
│   ├── profiles/
│   │   ├── generic_relative.yaml
│   │   ├── tum_indoor.yaml
│   │   └── kitti_outdoor.yaml
│   └── runtime/
│       ├── local_cpu.yaml
│       └── autodl_cuda.yaml
├── environment/
│   ├── local-macos.yml
│   ├── autodl-cuda.yml
│   └── compatibility.md
├── external_sources/
│   ├── manifest.yaml
│   ├── README.md
│   └── migration_notes/
├── artifacts/
│   ├── submaps/
│   ├── dem/
│   ├── descriptors/
│   ├── indices/
│   ├── graph/
│   └── runs/
├── tests/
│   ├── synthetic/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── algorithms/
    ├── provenance/
    └── runbooks/
```

实现时文件数量可以调整，但模块职责和接口边界必须保持清晰。

## 5. 运行环境

### 5.1 本地 macOS

使用已有环境：

```text
Conda 环境：vggt-dem
Python：3.11.15
PyTorch：2.11.0
CPU：可用
MPS：当前不可用
```

以下功能必须能够在本地 CPU 上运行：

- 合成几何测试。
- DEM 构建和可视化。
- 小规模 DINOv2 描述子测试。
- FAISS-HNSW 正确性和召回率测试。
- 不依赖最终优化器绑定的 Sim(3) 数学测试。

设备选择顺序为：

```text
CUDA -> MPS -> CPU
```

但 CPU 路径必须始终有效。MPS 只是可选加速，不是完成项目的必要条件。

### 5.2 AutoDL CUDA

AutoDL 负责：

- 完整 VGGT 推理。
- 基线序列运行。
- 全分辨率 DEM 构建。
- 大批量描述子计算。
- 端到端精度评估和性能评估。

本地和 AutoDL 使用独立配置及运行记录。代码中不能写入机器相关的绝对路径。

## 6. 里程碑与人工审核门

### M0：冻结基线并在云端跑通

1. 记录基线源码来源和压缩包 SHA-256。
2. 将基线核心源码纳入 Git。
3. 在 AutoDL 运行 `office_loop` 冒烟测试。
4. 将 TUM `freiburg1_desk` 作为第一个正式基线数据集。
5. 导出可人工审查的子图缓存。
6. 记录镜像、CUDA、Python、依赖、权重、命令和输出信息。

当前 `main_offline.py` 不能直接作为正式缓存契约，原因包括：

- `--save_only` 分支目前会跳过实际保存调用。
- 缓存没有保存能够完整重放的回环信息。
- 注释声称可以忽略图像，但 `Solver.add_points()` 实际读取
  `pred_dict["images"]`。

因此 M0 会新增最小导出器，而不是继续扩展该临时脚本。

### M1：带尺度管理的 DEM

1. 验证子图缓存格式。
2. 过滤低置信度点和深度异常点。
3. 将子图间相对尺度传播到统一的 map-unit 世界坐标。
4. 拟合并验证主平面。
5. 构建并冻结 DEM 规范坐标系。
6. 构建稳定的 metric 或 relative tile lattice。
7. 实现 mean、max 和 softmax 三种规约器。
8. 生成灰度 DEM 和供人查看的可视化结果。
9. 使用合成数据和真实子图进行验证。

只有在代码、公式映射、统计结果和 DEM 图像全部人工审核后，M1 才算完成。

### M2：DINOv2 与共视候选

1. 为 global tiles 和 query chips 生成加权 DINOv2 描述子。
2. 实现精确搜索和 FAISS-HNSW 搜索。
3. 使用精确搜索结果验证 HNSW 的召回率。
4. 将 tile 匹配聚合成 submap 候选。
5. 保存原始 chip-to-tile 匹配和聚合排名。

### M3：VPR 与几何边测量

1. 从提供的外部源码迁移并验证 AnyLoc。
2. 只在 M2 给出的共视候选窗口内执行精排。
3. 将检索证据转换为独立的几何证据。
4. 估计相对 Sim(3) 测量，或明确拒绝该候选。
5. 记录描述子分数、几何残差、退化检查和边的来源。

### M4：空间校正后端

1. 验证 Sim(3) 群运算和合成图恢复。
2. 定义与具体求解器无关的 `LocalSim3Backend` 接口。
3. 实现或绑定可靠的 7DoF Sim(3) 优化器。
4. 优化有空间边界的共视子图。
5. 修正位姿并使受影响的地图产物失效。
6. 只有离线确定性重放正确后，才增加在线异步调度。

## 7. 子图缓存契约

每个子图使用独立目录：

```text
artifacts/submaps/<run_id>/<submap_id>/
├── metadata.json
├── images.npz
├── cameras.npz
├── geometry.npz
├── confidence.npz
└── checksums.json
```

`metadata.json` 至少包含：

- 缓存格式版本。
- 子图和帧标识。
- 数据集名称和相对于配置根目录的数据路径。
- 数组形状和数据类型。
- 变换方向约定。
- 相机坐标约定。
- 坐标系名称。
- 单位和 `scale_status`。
- 已知情况下的 `meters_per_map_unit`。
- 尺度锚点来源和证据。
- 基线提交或压缩包哈希。
- 权重哈希。
- 运行设备和环境标识。
- 算法 profile 和证据标签。

几何缓存至少包含：

- 相机位姿和内参。
- 稠密点云，或能够重新生成点云的深度。
- 置信度图。
- 被选中的帧标识。
- 子图间时序关系。
- 相对尺度估计及其来源。

回环信息必须结构化保存，不能只保存回环数量。

## 8. 尺度策略

系统明确区分：

```text
scale_status = relative | metric
```

对于子图中的点 `p_i`：

```text
p_world = s_i R_i p_i + t_i
p_metric = gamma * p_world
```

其中 `gamma` 的单位为：

```text
meter / map-unit
```

Sim(3) 里程计可以将不同子图统一到同一个相对尺度，但纯单目几何不能唯一确定
`gamma`。因此，只有存在明确记录的尺度锚点时，才能将单位标为 metric。

可能的尺度锚点包括：

- 传感器深度。
- 已知相机高度或基线。
- 外部里程计。
- 对应源码中有文档支持的尺度策略。
- 真值只能用于诊断，不能在声称“无标定复现”时被静默使用。

本设计暂不提前选择各数据集的尺度锚点。

如果没有合法 metric anchor：

- 使用 `generic_relative` profile。
- 分辨率只能写成 map-unit/pixel。
- 使用显式配置的 `tile_size_map_units`。
- 结果标记为 `experimental`。
- 不能声称使用了 2m x 2m tile。

## 9. 数据集 Profile

场景相关行为通过配置实现，不能在算法代码中直接编写
`if KITTI` 或 `if TUM` 分支。

Profile 字段包括：

- 尺度策略和尺度锚点。
- 置信度策略。
- 最小和最大深度。
- 平面 RANSAC 阈值和迭代次数。
- 最小平面内点比例。
- 规范坐标系策略。
- DEM 分辨率策略。
- Tile 尺寸。
- 规约器和 softmax 温度。
- 归一化百分位。
- 边缘增强策略。
- DINO 输入尺寸和上下文范围。
- HNSW 参数。
- 检索和 VPR 阈值。

初始 profile：

- `generic_relative`：用于不声明米制的算法正确性测试。
- `tum_indoor`：参数在 TUM 室内数据验证时确定。
- `kitti_outdoor`：参数在 KITTI 室外数据验证时确定。

未知参数必须作为明确的验证任务存在，不能隐藏在没有来源的默认值中。

## 10. DEM 数学公式与坐标规则

### 10.1 平面与规范坐标系

输入点必须属于同一个世界坐标系和同一个尺度版本。

```text
Pi = { p in R^3 : n^T p + d = 0 }                         论文公式 (9)
R = [x y z] in SO(3), z = n                               论文公式 (10)
p_tilde_i = R^T (p_i - o) = (u_i, v_i, h_i)               论文公式 (11)
```

RANSAC 用于寻找平面内点，SVD/PCA 用于细化平面和面内坐标轴。

复现时增加以下规则：

- 平面法向量的正负方向必须确定且可重复。
- PCA 面内坐标轴的符号必须确定且可重复。
- 检测两个面内特征值接近时的轴方向不稳定问题。
- 图库建立后冻结规范坐标系；如果必须替换，需要进行显式对齐和版本升级。
- 保存 `T_world_dem`。

### 10.2 网格分辨率

```text
S = max(u_1 - u_0, v_1 - v_0)                             论文公式 (15)
mpp = S / target_px_long                                  论文公式 (16)
W_px = ceil((u_1 - u_0) / mpp)
H_px = ceil((v_1 - v_0) / mpp)                            论文公式 (17)
```

在 relative 模式下，`mpp` 表示 map-unit/pixel，而不是 meter/pixel。

论文同时提到了：

- 动态计算的 `mpp`。
- 固定的 `tile_px`。
- 固定 2m x 2m tile。

这三个约束通常不能同时成立。

初始设计将物理 tile 尺寸作为 metric 模式下的主约束：

```text
tile_px = round(tile_size_m / mpp)
tile_size_actual = tile_px * mpp
```

实际 tile 尺寸和量化误差必须写入元数据。如果后续官方源码显示了其他解释，
该策略可以替换。

论文没有充分解释“90k pixels”和“4096 spatial tiles”的准确含义，
因此它们属于后续 profile 验证任务。

### 10.3 稳定 Tile Lattice

论文根据当前点云包围框原点计算 tile 索引。但在在线地图中，如果地图向负方向扩张，
包围框原点会改变，所有旧 tile ID 可能整体移动。

因此需要：

- 图库初始化时冻结 `lattice_origin`、`mpp` 和 `tile_px`。
- 使用允许负数的整数 tile 坐标。
- 当前包围框只表示占用范围，不参与 tile 身份定义。

```text
x_hat_i = (u_i - lattice_u0) / mpp                         论文公式 (19)
y_hat_i = (v_i - lattice_v0) / mpp
I_u = floor(x_hat_i / tile_px)                             论文公式 (20)
I_v = floor(y_hat_i / tile_px)                             论文公式 (21)
x = round(x_hat_i - I_u * tile_px)                         论文公式 (22)
y = round(y_hat_i - I_v * tile_px)                         论文公式 (23)
```

### 10.4 高度规约器

同一像素内的高度集合使用：

```text
H(x, y) = reducer({h_k})                                   论文公式 (25)
mean = sum(h_k) / K                                        论文公式 (26)
max = max(h_k)                                             论文公式 (27)
softmax = sum(exp(h_k/tau) h_k) / sum(exp(h_k/tau))        论文公式 (28)
```

论文默认 softmax 温度：

```text
tau = 0.02
```

实际代码必须使用数值稳定实现，在指数运算前减去最大 logit。

### 10.5 灰度归一化

```text
h_min = percentile_0.5(H)
h_max = percentile_99.5(H)                                论文公式 (29)
I_0 = (clip(H, h_min, h_max) - h_min) / (h_max - h_min)   论文公式 (30)
```

空像素需要保留独立 mask，并在人工审查图中显示为白色。

图库的归一化统计量必须被冻结或版本化。Query 必须使用对应图库的统计量。
如果统计量改变，受影响的描述子和索引必须重建。

论文没有清楚说明边缘增强结果如何与 `I_0` 组合，因此该部分需要使用带证据标签的
可替换策略。

## 11. DINOv2 描述子

当前最可能符合论文的基础权重是：

```text
weights/dinov2_vitb14_pretrain.pth
SHA-256:
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

ViT-B/14 的输出维度与论文给出的 768 维描述子一致。模型保持冻结，不重新训练。

当前已有的 `dino_salad.ckpt` 属于 VGGT-SLAM 基线的 SALAD 检索权重，
不能作为 VGGT-SLAM++ 使用 AnyLoc 的证据。

论文公式：

```text
v_k = sum_j(w_j m_j f_theta(p_j)) / sum_j(w_j m_j)        论文公式 (1)
```

论文明确的信息：

- Global tile 描述子使用以当前 tile 为中心的 9 x 9 tile 邻域。
- `w_j` 是 Gaussian 位置权重。
- `m_j` 由 DEM 局部梯度幅值产生。
- 平坦或低信息区域的权重较低。
- 输出描述子经过归一化。

论文没有明确的信息：

- 使用 DINO 的哪一层。
- 使用哪种 facet，例如 token、key、query 或 value。
- DINO 输入图像尺寸。
- Gaussian 的 sigma。
- 梯度如何映射成 visibility mask。
- 邻域边界缺失 tile 时如何 padding。
- Query chip 描述子的准确聚合范围。

DINO patch 对应的物理范围必须显式计算：

```text
context_size = context_tiles * tile_size
dino_mpp = context_size / dino_input_px
patch_footprint = 14 * dino_mpp
```

改变 DINO 输入尺寸会改变每个 patch 的物理感受野，因此也会改变描述子签名。

## 12. M2 检索

描述子先进行归一化，再计算：

```text
s(chi_q, tau_k) =
    v_q^T v_k / (||v_q|| ||v_k||)                          论文公式 (2)
```

精确搜索作为正确性基准，FAISS-HNSW 作为可扩展实现。

每个 query chip 必须保存：

- 检索到的 tile ID。
- Tile 所属的 parent submap ID。
- 原始相似度分数。
- 排名。
- 索引签名和索引参数。

论文中的 submap 投票：

```text
Score(S) += sum_{tau_k in S} s(chi_q, tau_k)               论文公式 (3)
```

保留分数最高的 `K = 10` 个 submap 作为共视候选。

论文的原始分数求和实现标记为 `paper_exact`。为了避免 tile 数量多的 submap
天然占优，可以额外实验归一化投票，但只能标记为 `experimental`。

以下内容论文没有说明，需要留在 profile 中：

- 是否排除最近的时序邻居。
- 如何处理重复匹配。
- 每个 chip 检索多少近邻。
- 相似度阈值。

## 13. M3 VPR 与相对 Sim(3)

AnyLoc 只在 M2 提供的共视窗口内运行。

必须分成两个独立接口：

```text
VprRefiner.rank(query, candidates) -> ScoredTileMatches

RelativeSim3Estimator.verify(matches, geometry)
    -> Sim3Measurement | Rejection
```

检索分数不是几何变换，不能直接作为 Sim(3) 约束。

论文没有说明：

- AnyLoc 使用的 DINO 层和 facet。
- VLAD 词典来源和聚类数量。
- Chip-to-tile 匹配如何映射成 3D 对应。
- 相对 `T_hat_ij` 如何估计。
- 接受阈值和退化检查。

最终几何验证器至少检查：

- 是否有足够数量的独立对应。
- 3D 支撑是否非共线、非退化。
- 尺度是否处于合法范围。
- 正向和反向估计是否一致。
- 3D 对齐残差。
- 可重复的拒绝原因。

每一条边具有生命周期：

```text
proposed -> verified -> active
         -> rejected
active   -> stale
```

## 14. M4 Sim(3) 后端

论文优化目标为：

```text
min_{T_i in Sim(3)}
sum_(i,j in E) || Log_Sim3(T_j^-1 T_i T_hat_ij) ||^2_{Sigma_ij}
                                                               论文公式 (4)
```

状态变量：

```text
T_i = (s_i, R_i, t_i) in Sim(3)
```

论文图中明确写有 GTSAM Sim(3) manifold optimizer。但当前本地 `vggt-dem`
中的 GTSAM 虽然暴露了 `Similarity3`，Python 绑定中却没有：

- `Values` 对 `Similarity3` 的完整支持。
- `PriorFactorSimilarity3`。
- `BetweenFactorSimilarity3`。

因此后端必须隐藏在独立接口之后：

```text
LocalSim3Backend.optimize(active_subgraph)
    -> CorrectedSubmapPoses
```

后续需要调查的实现方向：

- 论文作者使用的 GTSAM 分支或 Python 绑定。
- 小型 GTSAM C++ 扩展。
- 经过验证的其他 Sim(3) 优化器。

禁止使用 Pose3 代替后仍将结果称为 Sim(3)。

论文将该步骤称为 local bundle adjustment，但公式 (4) 只展示了
motion-only Sim(3) pose graph，没有出现显式地图点或相机观测变量。

在获得更多源码证据前，本项目使用更准确的名称：

```text
局部 Sim(3) 位姿图优化
```

`Sigma_ij` 保留为独立接口，由描述子一致性和 3D 对齐残差共同构造。

## 15. 版本管理与失效机制

每个 DEM、描述子、索引、匹配和图边都必须携带签名：

- 尺度状态和尺度版本。
- 规范坐标系版本。
- Lattice 原点。
- `mpp`。
- Tile 物理尺寸和 `tile_px`。
- 规约器和归一化版本。
- DINO 权重哈希、层、facet、输入尺寸和 context。
- Profile 和代码版本。

Gallery 和 query 的签名必须匹配。否则检索直接失败，并明确提示需要重建。

后端位姿修正可能使以下内容失效：

- 点云位置。
- Tile 占用关系。
- DEM 数值。
- 归一化统计量。
- DINO 描述子。
- FAISS 索引条目。
- 几何回环测量。

M0-M4 的离线版本先使用确定性的完整重建。只有正确性建立后，才实现增量失效更新。

## 16. 错误处理

每个算法返回结构化结果或结构化拒绝原因。

拒绝原因示例：

- 有效深度不足。
- 平面内点不足。
- 规范坐标轴存在歧义。
- metric anchor 缺失或非法。
- DEM tile 为空。
- 描述子签名不匹配。
- HNSW 索引版本不匹配。
- VPR 匹配数量不足。
- Sim(3) 几何退化。
- 优化器不收敛。

任何阶段都不能静默替换单位、坐标系、profile、权重或设备。

## 17. 测试设计

### 17.1 合成几何测试

- 带噪声和离群点的平面恢复。
- 平面法向和 PCA 轴符号稳定性。
- Metric 与 relative 尺度传播。
- 地图扩张时 tile ID 保持稳定。
- 像素和 tile 边界行为。
- Mean、max 和数值稳定的 softmax 规约器。
- 空像素和 mask。

### 17.2 真实子图人工审核

- 导出的 TUM 子图元数据。
- 原始和过滤后的点云。
- 平面内点可视化。
- 高度直方图。
- 灰度 DEM tiles。
- 供人查看的彩色 DEM。
- Profile 和尺度标注。

### 17.3 描述子与检索测试

- 输出维度和 L2 归一化。
- CPU 推理的确定性误差范围。
- Query/gallery 签名不一致时拒绝。
- 精确搜索真值。
- HNSW recall@k 和延迟。
- 原始 tile 匹配和 parent-submap 投票。

### 17.4 Sim(3) 测试

- 组合与逆变换。
- Exp/Log 往返一致性。
- 支持情况下的 Jacobian 检查。
- 合成相对位姿图恢复。
- Gauge 锚定。
- 离群边和退化情况拒绝。
- 优化前后误差对比。

### 17.5 人工审核门

每一个里程碑必须交付：

- 带关键算法注释的源码。
- 论文公式到代码的映射。
- 测试输出。
- 可人工查看的生成产物。
- 已知偏差和证据标签。

当前里程碑没有经过用户审核时，不进入下一个里程碑。

## 18. 注释规范

关键算法注释需要说明：

- 坐标系和变换方向。
- 单位和尺度状态。
- 数组或张量形状。
- 对应论文公式和章节。
- 参数来源。
- 数值稳定措施。
- 拒绝和失效行为。

注释不应逐行重复显而易见的赋值和循环。

## 19. 外部源码管理

`external_sources/` 中的第三方源码默认被 Git 忽略，只跟踪：

- `manifest.yaml`
- `README.md`
- 许可证说明
- Commit hash
- 迁移记录和源文件到目标文件的对应关系

预期需要调查：

- AnyLoc。
- DINOv2 官方特征提取。
- 可靠的 Sim(3) 优化实现。
- 能够解释 DEM 加权或几何边生成的对应源码。

没有完成来源和许可证审核时，不能直接复制算法。

## 20. Git 与大型文件

Git 跟踪：

- 基线核心源码。
- `vggt_slam_pp/`。
- 配置。
- 环境定义。
- 测试。
- 文档。
- 外部源码 manifest。
- 小型确定性测试样例。

Git 忽略：

- `.DS_Store`
- `.superpowers/`
- `VGGT-SLAM-version1.0.zip`
- 与压缩包重复的解压样例数据
- `weights/`
- `data/`
- `artifacts/`
- 下载的第三方源码仓库
- 构建文件、虚拟环境、缓存、日志和可视化输出

大型文件使用固定预期路径和 SHA-256 信息。AutoDL 准备脚本只负责检查文件存在性
和哈希，不允许在算法运行时静默下载权重。

当前已知哈希：

```text
VGGT-SLAM-version1.0.zip
3283a68428ce86f95313e94d6a7a22b0b5f150c632c90aa57675ba35d5132aee

weights/dinov2_vitb14_pretrain.pth
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73

weights/dino_salad.ckpt
6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada
```

## 21. 延后决定的内容

以下内容有意延后到对应里程碑和数据集验证时决定：

- 每个数据集合法的 metric scale 来源。
- TUM 和 KITTI 的深度阈值。
- 平面 RANSAC 阈值。
- 每个 profile 的 `mpp`、tile 尺寸和 DINO 输入尺寸。
- 90k pixels 和 4096 tiles 的准确解释。
- DINO 层、facet、Gaussian sigma 和 visibility mask。
- HNSW 构建和搜索参数。
- AnyLoc VLAD 配置。
- 相对 Sim(3) 测量算法。
- 边的信息矩阵。
- 局部图窗口和调度频率。

每一个延后决定都已经有对应接口和证据标签，因此后续确定细节时不需要重构系统。

## 22. 第一轮实施范围

设计通过审核后，第一份实施计划只覆盖：

1. M0 基线来源记录和 AutoDL 运行手册。
2. 顶层 Git 忽略规则。
3. 外部源码 manifest 结构。
4. 可审查子图缓存 schema 和 exporter。
5. M1 合成测试和独立 DEM 实现。
6. TUM `freiburg1_desk` DEM 人工审查产物。

M2-M4 暂时只建立接口骨架和研究任务。M1 未审核前，不开始完整实现 M2-M4。
