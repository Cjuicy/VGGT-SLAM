# 外部源码暂存区

这里用于放置论文复现依赖的原始仓库，导入的仓库本身被根 `.gitignore`
排除，只提交本说明和 `manifest.yaml`。

```text
external_sources/
├── vggt/
├── salad/
└── gtsam_with_sl4/
```

每次迁移算法前必须在 `manifest.yaml` 填写实际来源 URL、commit 和许可证。
不得只依据论文描述重写第三方核心算法，也不得在运行时通过 Torch Hub
静默下载代码或权重。

安装时从仓库根执行：

```bash
python -m pip install -e external_sources/vggt
python -m pip install -e external_sources/salad
```

`gtsam_with_sl4` 必须按其上游说明编译安装。安装后先确认：

```bash
python -c "import gtsam; print(all(hasattr(gtsam, x) for x in ('SL4', 'PriorFactorSL4', 'BetweenFactorSL4')))"
```

输出必须为 `True` 才能运行默认 SL(4) 基线。
