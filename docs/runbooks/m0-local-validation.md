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

## 从 Git 更新本地代码

云端验证提交合入 `codex/m0-reproduction` 后，在本地仓库根执行：

```bash
git pull origin codex/m0-reproduction
conda run -n vggt-dem python -m pip install -e ".[dev]"
conda run -n vggt-dem pytest -q
```

拉取前先用 `git status --short` 检查本地未提交修改。不要用 reset 或 checkout
覆盖本地数据说明、论文笔记或尚未审核的计划草稿。

## 接收 AutoDL 运行产物

1. 通过 AutoDL 文件管理器把 `<run-id>.tar.gz` 和
   `<run-id>.tar.gz.sha256` 下载到本地 `artifacts/packages/`。
2. 按 [`cloud-artifact-transfer.md`](cloud-artifact-transfer.md) 先验证压缩包
   SHA-256，再解压到 `artifacts/imported/<run-id>/` 并验证逐文件清单。
3. 查看运行目录中的 `ate.json` 和 `cache-inspection.json`。
4. 查看子图缓存根目录中的 `final_state.json`，确认变换类型、边数、回环数和
   轨迹 SHA-256 与云端审核记录一致。
5. 使用缓存检查 CLI 再次读取本地副本，抽查首、中、末子图数组的形状、有限
   值、只读状态、坐标系和 `relative_map_unit`。

本地接收阶段审核的是“云端输出是否完整、可追溯、可离线读取”，不是重新进行
完整模型推理。M0 不要求在 Mac 上运行完整 DINOv2/SALAD、VGGT CUDA 推理或
自定义 SL(4) 优化；这些结果以 AutoDL 运行记录和校验后的产物为准。
