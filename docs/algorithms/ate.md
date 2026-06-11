# M0 ATE 评测

核心指标是平移 Absolute Pose Error 的 RMSE，单位跟随真值轨迹，TUM/KITTI
真值通常为米。实现直接调用 evo Python API：

1. 按时间戳关联真值与估计轨迹。
2. 用 Umeyama Sim(3) 对齐估计轨迹到真值，并校正单目尺度。
3. 对每对关联位置计算
   `e_i = ||t_est_aligned_i - t_ref_i||_2`。
4. 报告 `sqrt(mean(e_i^2))`。

这与 `evo_ape tum groundtruth.txt estimated.txt -as` 的核心计算一致，但不
解析人类可读的终端输出。

## 两套报告

- `paper_compatible`：直接读取原始日志，便于对照论文和基线脚本。
- `canonical_unique`：每个时间戳只保留第一次出现，避免相邻子图共享过渡
  帧时重复计权。

去重不会静默丢弃信息。JSON 同时记录重复数量、重复位姿相对首项的最大
平移差和最大四元数夹角，以及两份输入文件的 SHA-256。

```bash
conda run -n vggt-dem python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth data/sequence/groundtruth.txt \
  --estimate artifacts/run/trajectory_raw.txt \
  --output artifacts/run/ate.json
```
