# M0 本地验证摘要

验证日期：2026-06-11

## 已通过

- `conda run -n vggt-dem pytest -v`
  - 60 项测试通过。
  - 用时 0.98 秒。
- `conda run -n vggt-dem python -m compileall -q vggt_slam_pp VGGT-SLAM-version1.0`
  - 通过，无语法错误。
- `conda run -n vggt-dem python scripts/verify_assets.py --config configs/runtime/local_macos.yaml`
  - VGGT、SALAD、DINOv2 三份权重存在且 SHA-256 匹配。
  - `baseline_sim3_compat` 所需 `gtsam.Pose3` 可用。
- `git diff --check`
  - 通过。
- Git 对象库稳定：
  - 松散对象 327 个，共 10.75 MiB。
  - `.git` 目录约 11 MiB。
  - 垃圾对象 0 字节。

## 覆盖范围

- 干净 VGGT-SLAM v1 快照和来源哈希。
- SL(4)/Sim(3) 兼容运行身份约束。
- 子图与图状态形状、尺度语义和只读复制。
- 原子缓存写入、覆盖保护与损坏检测。
- 不导入 GPU 依赖的基线 `Submap` 适配。
- 原始/规范 TUM 轨迹与 Sim(3) 对齐 ATE。
- 离线权重预检、缓存检查和 Office A/B 比较器。
- 基线 CLI 的本地权重、设备、回环禁用和可选导出静态约束。

## 尚未通过

本地没有可用 CUDA/MPS，也没有自定义 GTSAM SL(4) 符号，因此没有声称
完成以下运行：

- VGGT 1B 完整前向推理。
- SALAD/DINOv2 完整检索。
- Office-loop 默认 SL(4) 基线。
- TUM desk 三运行矩阵和真实 ATE。
- 导出开启/关闭的真实 Office 位姿一致性。

这些项目必须按 `docs/runbooks/m0-autodl-baseline.md` 在 AutoDL 完成，并
把环境、命令、ATE 和已知偏差放入新的云端审查目录后，M0 才能通过人工
Review Gate。

## 2026-06-12 离线 DINOv2 加载增量验证

本次验证针对 SALAD 回环模块的本地 DINOv2 源码和权重加载入口：

- `conda run -n vggt-dem pytest tests/unit/adapters/test_salad_local.py tests/unit/test_verify_assets.py tests/static/test_baseline_cli.py tests/integration/test_baseline_export_hook.py -v`
  - 23 项聚焦测试通过。
  - 覆盖本地 Hub 参数、严格 checkpoint 加载、缺失资产错误、并发构造恢复、
    基线参数传递和 `max_loops=0` 隔离。
- `conda run -n vggt-dem pytest tests -v`
  - 74 项完整测试通过。
  - 用时 1.38 秒。
- `conda run -n vggt-dem python -m compileall -q vggt_slam_pp VGGT-SLAM-version1.0`
  - 通过，无语法错误。

本地没有提供 `external_sources/dinov2/`，因此本次没有加载真实官方 DINOv2
源码，也没有执行真实 SALAD/DINOv2 前向。以下内容仍是 AutoDL 人工审核
检查点：

- `external_sources/dinov2/hubconf.py` 与实际 commit 记录。
- 断网状态下加载 `weights/dinov2_vitb14_pretrain.pth` 和
  `weights/dino_salad.ckpt`。
- 终端不出现 `Downloading:`。
- Office-loop `max_loops=1` 完整运行及轨迹、子图和回环数量。
