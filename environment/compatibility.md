# M0 运行环境边界

## 本地 macOS：`vggt-dem`

- Python 固定为 3.11。
- 本地用于契约测试、缓存读写、ATE 评估和静态审查。
- 当前 Torch 不提供可用 CUDA，且本机 GTSAM 4.2.1 不含 `SL4`、
  `PriorFactorSL4`、`BetweenFactorSL4`。
- 因此本地预检选择 `baseline_sim3_compat`。这不等于论文后端使用
  Sim(3)，它仅表示 VGGT-SLAM 前端桥接采用带尺度烘焙的兼容路径。
- DINOv2/Salad 的接口与权重哈希可在本地检查；完整推理是否可接受取决于
  Mac 型号、内存和 PyTorch MPS 支持，M0 不把本地完整推理作为通过条件。

```bash
conda run -n vggt-dem python scripts/verify_assets.py \
  --config configs/runtime/local_macos.yaml
```

## AutoDL CUDA

- 目录结构必须与仓库一致，权重放在 `weights/`，数据放在 `data/`。
- `autodl_cuda.yaml` 要求 `torch.cuda.is_available()` 为真。
- 当前 M0 桥接同样使用 `baseline_sim3_compat` 并设置 `max_loops=0`。
- 原论文基线对照若要运行默认 SL(4)，需要另外安装包含三个 SL(4) 符号的
  GTSAM 构建；普通 PyPI/Conda GTSAM 不能视为满足条件。

```bash
conda run -n vggt-dem python scripts/verify_assets.py \
  --config configs/runtime/autodl_cuda.yaml
```

预检只读取本地文件和 Python 模块，不会联网下载或修改权重。
