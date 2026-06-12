# M0 SALAD 离线 DINOv2 加载设计

## 1. 目标

修复 VGGT-SLAM version1.0 的 SALAD 回环初始化在运行时调用
`torch.hub.load("facebookresearch/dinov2", ...)` 下载源码和权重的问题。

完成后，M0 基线必须满足：

- DINOv2 架构源码只来自 `external_sources/dinov2/`。
- DINOv2 预训练权重只来自
  `weights/dinov2_vitb14_pretrain.pth`。
- SALAD 聚合权重只来自 `weights/dino_salad.ckpt`。
- 运行时不访问 GitHub、Torch Hub 或 Hugging Face。
- 缺少源码、权重或版本记录时，在模型推理前明确失败。
- 不改变 SALAD 描述子、回环候选、阈值或 VGGT-SLAM 图优化算法。

## 2. 外部源码边界

新增外部源码目录：

```text
external_sources/dinov2/
```

该目录由用户在本地和 AutoDL 按相同相对路径提供，不纳入本项目 Git
仓库。来源固定为 Meta 官方仓库：

```text
https://github.com/facebookresearch/dinov2.git
```

首次云端跑通时，将实际使用的 commit 写入
`external_sources/manifest.yaml`。在 commit 未确认前，状态保持
`user_source_required`，不得标记为已经复现验证。

本项目不会迁移、复制或重新实现 DINOv2 的 Transformer、patch embedding
或权重转换算法。算法实现继续由官方源码负责。

## 3. 权重职责

两个权重文件不能混用：

```text
weights/dinov2_vitb14_pretrain.pth
weights/dino_salad.ckpt
```

- `dinov2_vitb14_pretrain.pth` 初始化 ViT-B/14 backbone。
- `dino_salad.ckpt` 恢复 SALAD 训练后的完整 VPR 状态。

加载顺序固定为：

1. 从本地 DINOv2 源码构造 `dinov2_vitb14`。
2. 从本地 DINOv2 权重初始化 backbone。
3. 构造 SALAD 聚合器。
4. 加载 SALAD checkpoint，覆盖训练后的完整模型状态。
5. 切换到 `eval()`，移动到用户指定设备。

即使 SALAD checkpoint 最终覆盖 backbone 参数，也仍显式加载并核验
DINOv2 基础权重。这样可以保证架构和资产来源完整、可追踪，并避免
Torch Hub 的隐式行为。

## 4. 加载适配器

在项目包中新增一个小型适配器，职责仅限：

- 校验 DINOv2 源码目录。
- 从源码目录加载官方 `hubconf.py`。
- 使用官方本地 Hub 调用只构造模型结构：

  ```python
  model = torch.hub.load(
      str(dinov2_source),
      "dinov2_vitb14",
      source="local",
      pretrained=False,
  )
  ```

- 使用 `torch.load(..., map_location="cpu", weights_only=True)` 直接读取
  本地权重，再调用 `model.load_state_dict(..., strict=True)`。
- 不把本地路径传给 Hub 的 `weights` 参数。该入口会把本地路径转换为
  `file://` URL，再复制到 Torch Hub 缓存并显示误导性的 `Downloading:`。
- 在加载期间禁止网络回退。
- 返回构造完成的 backbone。

适配器不得：

- 下载源码或权重。
- 修改官方 DINOv2 模型结构。
- 猜测 checkpoint 键名并静默忽略不匹配。
- 把本地路径写死为某台机器的绝对路径。

`source="local"` 仍使用 PyTorch 的标准 Hub 入口，但只读取指定本地目录，
不会访问远端仓库。

## 5. 基线参数流

VGGT-SLAM 基线新增两个显式参数：

```text
--dinov2_source external_sources/dinov2
--dinov2_weight weights/dinov2_vitb14_pretrain.pth
```

数据流为：

```text
main.py
  -> Solver
    -> ImageRetrieval
      -> reviewed local SALAD loader
        -> local DINOv2 adapter
```

`max_loops=0` 时继续完全跳过 SALAD 和 DINOv2 初始化。前端桥接运行不应
依赖任何 VPR 模型。

`max_loops>0` 时，SALAD checkpoint、DINOv2 源码目录和 DINOv2 权重都必须
存在，否则在启动阶段失败。

## 6. 对第三方 SALAD 的修改

不直接修改 `external_sources/salad/`，因为它是用户提供、被忽略的第三方
源码目录，无法通过本项目 Git 稳定传输修改。

项目内实现一个 reviewed local SALAD loader：

- 复用 `salad.vpr_model.VPRModel` 和原始模型配置。
- 在构造 VPRModel 时注入本地创建的 DINOv2 backbone。
- 复用 SALAD 原有聚合器和 checkpoint。
- 保持输出描述子维度和推理接口不变。

SALAD 当前 API 不支持直接注入 backbone。因此项目适配器先构造 VPRModel，
再把 `backbone.model` 替换为本地 Hub 构造的官方 DINOv2 模型。替换必须发生在
加载 SALAD checkpoint 之前，并由测试验证目标模块类型和 SALAD checkpoint
严格匹配。

## 7. 预检与失败策略

`verify_assets.py` 扩展以下检查：

- `external_sources/dinov2/hubconf.py` 存在。
- manifest 包含 DINOv2 来源声明。
- DINOv2 权重 SHA-256 与运行配置一致。
- 本地构造入口能够被发现，但预检不加载大模型、不占用大量显存。

运行时错误必须指明具体缺失项，例如：

```text
DINOv2 source is missing: external_sources/dinov2/hubconf.py
DINOv2 weight is missing: weights/dinov2_vitb14_pretrain.pth
DINOv2 local load failed; network fallback is disabled
```

不得在失败后自动改用在线下载、随机初始化或其他 DINOv2 变体。

## 8. 测试边界

静态测试：

- 基线 CLI 暴露两个本地 DINOv2 参数。
- 基线和项目适配器中不存在远端 `torch.hub.load()`。
- 本地 Hub 调用必须显式使用 `source="local"`。

单元测试：

- 缺少源码目录时失败。
- 缺少权重时失败。
- 本地构造器收到正确模型名和权重路径。
- 不发生网络回退。

集成测试：

- 使用轻量假模型验证 `main -> Solver -> ImageRetrieval` 参数传递。
- `max_loops=0` 不导入 SALAD，不加载 DINOv2。
- AutoDL 人工验证加载真实 SALAD checkpoint 后完成一次描述子前向传播。

完整 Office-loop 运行仍作为 M0 云端验收，不纳入本地 macOS 强制测试。

## 9. 云端工作流

用户需要在 AutoDL 项目根目录执行：

```bash
git clone https://github.com/facebookresearch/dinov2.git \
  external_sources/dinov2
git -C external_sources/dinov2 rev-parse HEAD
```

代码实现完成并推送后，用户拉取项目更新，重新安装项目自身。DINOv2
官方源码无需作为 Python 包安装；适配器直接读取其本地 Hub 入口。

运行前清理之前未完成的 Torch Hub 下载缓存：

```bash
rm -f ~/.cache/torch/hub/main.zip
```

不要求删除其他已存在缓存，因为新实现不会读取远端 DINOv2 缓存。

## 10. 验收标准

以下条件全部满足才算完成：

1. 断网状态下可以构造 SALAD 模型。
2. 终端不出现 `Downloading:`。
3. DINOv2 与 SALAD 权重均通过预检。
4. Office-loop 的 `max_loops=1` 运行进入图像处理阶段。
5. `max_loops=0` 的桥接运行不初始化 SALAD/DINOv2。
6. 所有新增测试和既有 M0 测试通过。
7. AutoDL 实际使用的 DINOv2、VGGT、SALAD 和 GTSAM commit 被记录。
