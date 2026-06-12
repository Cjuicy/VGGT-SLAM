# M0 ATE 评测

M0 以平移 Absolute Pose Error（APE）的 RMSE 作为核心复现指标。误差单位
跟随真值轨迹；TUM 和 KITTI 的真值平移通常以米表示。

## 输入格式与时间关联

评测输入统一为 TUM 八列轨迹：

```text
timestamp tx ty tz qx qy qz qw
```

其中四元数顺序是 `qx qy qz qw`。真值与估计轨迹先按 `timestamp` 关联，
默认只接受时间差不超过 `0.01` 的位姿对。未关联帧不会进入 ATE，也不会
影响分母。

VGGT-SLAM 从图像文件名提取帧号作为轨迹时间戳。因此 KITTI 转换后的真值
同样使用 `0, 1, 2, ...` 的帧号，而不是 `times.txt` 中以秒表示的采样时间。
`times.txt` 只用于检查数据集帧数与完整性。混用这两种时间轴会导致无法
关联，或关联到错误帧。

## Sim(3) 对齐与公式

单目系统只能恢复相对尺度，估计轨迹与真值之间通常同时存在尺度、旋转和
平移差。评测调用 evo 的 Umeyama 对齐，为估计位置求解一个 Sim(3)：

```text
t_i^aligned = s R t_i^est + u
```

其中 `s` 是统一尺度，`R` 是旋转，`u` 是平移。对第 `i` 个已关联位姿，
平移误差为：

```text
e_i = ||t_i^aligned - t_i^ref||_2
```

最终报告：

```text
ATE_RMSE = sqrt((1 / N) * sum(e_i^2))
```

实现位于 `vggt_slam_pp/evaluation/ate.py`，等价于
`evo_ape tum groundtruth.txt estimated.txt -as` 的核心计算，但通过 evo
Python API 直接取得数值，不解析面向人的终端文本。

## 两套报告

- `paper_compatible`：直接评测原始轨迹，保留相邻子图共享边界帧产生的
  重复时间戳，方便对照既有论文和基线脚本口径。
- `canonical_unique`：每个时间戳只保留第一次出现，再进行第二次评测，
  防止共享边界帧被重复计权。临时去重轨迹不会写入正式产物。

去重不会静默丢弃信息。`duplicate_diagnostics` 分别记录真值和估计轨迹的：

- `duplicate_count`：被丢弃的重复行数量；
- `max_translation`：重复位姿相对首个同时间戳位姿的最大平移差；
- `max_rotation_deg`：两者四元数的最小夹角，单位为度。

由于 `q` 与 `-q` 表示同一旋转，角度计算会先取四元数点积的绝对值。报告
还记录输入文件 SHA-256，使指标能追溯到确切轨迹内容。

## TUM desk 已验证结果

当前短序列桌面验证记录为：

| 模式 | `paper_compatible` 平移 APE RMSE |
| --- | ---: |
| VGGT-SLAM SL(4) | `0.029152 m` |
| VGGT-SLAM Sim(3) | `0.023354 m` |
| VGGT-SLAM++ 前端桥接 | `0.023354 m` |

桥接结果与 Sim(3) 基线 ATE 相同，只能说明最终误差数值一致；两份估计轨迹
的 SHA-256 完全相同，进一步证明每个输出字节一致，因此是更强的桥接等价
证据。后续后端功能真正改变轨迹后，SHA-256 应当变化，此时再比较 ATE 和
重复诊断。

## 运行方式

```bash
conda run -n vggt-dem python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth data/sequence/groundtruth.txt \
  --estimate artifacts/run/trajectory_raw.txt \
  --output artifacts/run/ate.json
```

阅读 JSON 时优先记录 `paper_compatible.translation_ape_rmse`、
`canonical_unique.translation_ape_rmse`、两者的 `associated_pose_count`
以及 `duplicate_diagnostics`。若关联位姿数异常偏少，先检查时间戳口径，
不要先调整对齐算法。
