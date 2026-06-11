# M0 本地 macOS 验证

本地使用已有 Conda 环境 `vggt-dem`。M0 的本地通过条件是契约、缓存、
适配器、ATE、静态基线修改和离线资产检查，不要求完成 VGGT、SALAD、
DINOv2 或 SL(4) 的完整推理。

## 安装与预检

```bash
conda run -n vggt-dem python -m pip install -e ".[dev]"
conda run -n vggt-dem python scripts/verify_assets.py \
  --config configs/runtime/local_macos.yaml
```

当前本地配置应显示：

- 三份权重 SHA-256 匹配。
- `cuda_available` 为 `false`。
- `baseline_sim3_compat` 所需的 `gtsam.Pose3` 存在。
- 不要求普通 GTSAM 提供 SL(4) 符号。

## 测试

```bash
conda run -n vggt-dem pytest -v
conda run -n vggt-dem python -m compileall -q \
  vggt_slam_pp VGGT-SLAM-version1.0
```

本地可以完整运行：

- 运行身份与模式校验。
- 子图/图状态数据契约。
- 原子缓存往返和损坏检测。
- 假 `Submap` 适配。
- TUM 去重与 evo ATE。
- 缓存检查和基线摘要比较。

本地暂不作为验收项：

- VGGT 1B 完整推理。
- SALAD/DINOv2 完整嵌入。
- CUDA 性能与显存。
- 自定义 GTSAM SL(4) 优化。

这些项目必须在 AutoDL 按云端手册执行后再进入 M0 人工审核。
