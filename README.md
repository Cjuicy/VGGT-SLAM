# VGGT-SLAM++ 可审核复现

本仓库以 VGGT-SLAM 1.0 为前端基线，分阶段复现 VGGT-SLAM++ 的 DEM、
DINOv2/FAISS-HNSW 检索、VPR、相对 Sim(3) 测量和后端优化。实现目标不是
一次性堆出完整系统，而是让每个算法阶段、尺度状态、缓存和指标都能离线重放
并接受人工代码审核。

## 代码边界

- `VGGT-SLAM-version1.0/` 是冻结基线，只保留运行兼容和子图导出所需的最小
  修改。
- `vggt_slam_pp/` 是旁路扩展包，DEM、检索、VPR 和后端实现不得写入基线
  包。
- `external_sources/` 只放需人工提供的官方参考源码，并记录来源；不会凭空
  重写引用论文算法。
- `data/`、`weights/`、`artifacts/` 和本地工具数据库不通过 Git 传输。

总体架构、证据标签和 M1-M4 边界见
[`VGGT-SLAM++ 复现设计规格`](docs/superpowers/specs/2026-06-11-vggt-slam-pp-reproduction-design-zh.md)。

## 当前状态

M0 的 AutoDL 双基线、TUM desk ATE 和 VGGT-SLAM++ 前端桥接已跑通。当前
代码已具备不可变子图缓存、图状态、KITTI 真值转换、双口径 ATE 和云端产物
校验回传流程。KITTI 09 长序列仍需按手册在 CUDA 环境完成并人工记录结果。

本地 macOS 使用 Conda 环境 `vggt-dem`，负责契约、缓存、算法单测、文档和
导入产物审核；AutoDL 负责 VGGT、DINOv2/SALAD、SL(4) 和长序列 CUDA 推理。

## 操作入口

| 任务 | 文档 |
| --- | --- |
| 本地安装、测试与云端产物接收 | [`M0 本地 macOS 验证`](docs/runbooks/m0-local-validation.md) |
| AutoDL 基线与 TUM desk 桥接 | [`M0 AutoDL 基线与桥接运行`](docs/runbooks/m0-autodl-baseline.md) |
| KITTI 09 长序列 | [`M0 KITTI 09 AutoDL 长序列验证`](docs/runbooks/m0-kitti09-autodl.md) |
| 云端打包、SHA-256 与本地导入 | [`云端产物传输`](docs/runbooks/cloud-artifact-transfer.md) |
| ATE 公式、字段和已验证结果 | [`M0 ATE 评测`](docs/algorithms/ate.md) |
| 子图缓存形状、坐标系与单位 | [`M0 子图缓存契约`](docs/algorithms/submap-cache-contract.md) |
| 代码图谱查看与审查流程 | [`code-review-graph 中文使用说明`](docs/tools/code-review-graph-zh.md) |
| 本轮 M0 交接实现步骤 | [`M0 handoff/KITTI 计划`](docs/superpowers/plans/2026-06-12-m0-handoff-kitti-tools.md) |

M1 DEM 与 M2-M4 框架的阶段目标已经写入总体设计。主工作区中的
`2026-06-11-vggt-slam-pp-m1-dem.md` 和
`2026-06-11-vggt-slam-pp-m2-m4-framework.md` 目前仍是待人工审核草稿，
审核后再作为独立计划提交，避免把尚未确认的算法假设写成既定实现。
