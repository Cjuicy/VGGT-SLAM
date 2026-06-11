# M0 子图缓存契约

## 两类状态

`SubmapArrays` 保存一次写入后不再改变的局部几何。数据必须来自
VGGT-SLAM 完成 `Solver.add_points()` 后的 `Submap`，不能从原始
`predictions["depth"]` 重新构造，因为 Sim(3) 兼容路径已经把估计尺度乘入
点云和相机平移。

`GraphState` 单独保存每次图更新后的全局变换。这样全局优化可以改变子图
位姿，而不会重写体积较大的局部点云。

## 坐标与变换

点 `p_s` 位于子图局部坐标系，最终全局点为

```text
p_w ~ H_world_from_submap @ [p_s.x, p_s.y, p_s.z, 1]^T
```

- 默认基线模式记录 `transform_kind = SL4`。
- Sim(3) 兼容前端记录 `transform_kind = SE3_with_baked_scale`。
- 后一种情况下，尺度 `s` 已进入局部几何与相机平移：

```text
p_s = s * p_vggt
t_camera_to_submap = s * t_vggt
```

因此下游不能再对同一子图重复乘 `s`。两种变换类型与
`RunIdentity.solver_mode` 必须严格对应。

## 数组约定

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `points_submap` | `(S,H,W,3)` | 最终子图局部点 |
| `colors_rgb` | `(S,H,W,3)` | `uint8` RGB |
| `confidence` | `(S,H,W)` | VGGT 置信度 |
| `confidence_mask` | `(S,H,W)` | 基线阈值后的布尔掩码 |
| `intrinsics` | `(S,3,3)` | 每帧内参 |
| `camera_to_submap` | `(S,4,4)` | 相机到子图的齐次变换 |

构造契约对象时会检查形状、数据类型和有限值，然后执行防御性复制并关闭
NumPy 写权限。`pp_frontend_bridge` 的 `loop_sources` 必须为空，保证 M0
缓存只表达前端结果，不混入 VGGT-SLAM 自己的回环后端。
