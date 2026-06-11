# 外部源码暂存区

这里用于放置论文复现依赖的原始仓库，导入的仓库本身被根 `.gitignore`
排除，只提交本说明和 `manifest.yaml`。

## 已核验来源

- [Facebook Research VGGT](https://github.com/facebookresearch/vggt)
- [VGGT-SLAM v1 使用的 SALAD fork](https://github.com/Dominic101/salad)
- [GTSAM develop](https://github.com/borglab/gtsam/tree/develop)
- [VGGT-SLAM 官方 version1.0](https://github.com/MIT-SPARK/VGGT-SLAM/tree/version1.0)

不使用 Git 时可直接下载：

- [VGGT main ZIP](https://github.com/facebookresearch/vggt/archive/refs/heads/main.zip)
- [SALAD main ZIP](https://github.com/Dominic101/salad/archive/refs/heads/main.zip)
- [GTSAM develop ZIP](https://github.com/borglab/gtsam/archive/refs/heads/develop.zip)
- [VGGT-SLAM version1.0 ZIP](https://github.com/MIT-SPARK/VGGT-SLAM/archive/refs/heads/version1.0.zip)

ZIP 不保留 Git commit 元数据，论文复现优先使用下方 `git clone` 命令。

2026-06-11 已核验这些仓库和分支存在。GTSAM `develop` 的源码包装模板包含
`SL4`、`PriorFactor<SL4>` 和 `BetweenFactor<SL4>`，构建 Python 绑定后应
导出基线需要的三个符号。

```text
external_sources/
├── vggt/
├── salad/
└── gtsam_with_sl4/
```

## AutoDL 下载

从项目仓库根目录执行：

```bash
git clone --branch main https://github.com/facebookresearch/vggt.git \
  external_sources/vggt

git clone --branch main https://github.com/Dominic101/salad.git \
  external_sources/salad

git clone --branch develop https://github.com/borglab/gtsam.git \
  external_sources/gtsam_with_sl4
```

下载后记录实际 commit：

```bash
git -C external_sources/vggt rev-parse HEAD
git -C external_sources/salad rev-parse HEAD
git -C external_sources/gtsam_with_sl4 rev-parse HEAD
```

首次 AutoDL 跑通后，把成功使用的 commit 写回 `manifest.yaml` 的 `commit`
字段再提交。`verified_remote_head` 只是 2026-06-11 的联网核验结果，不代表
未经云端测试就已经完成兼容性锁定。

每次迁移算法前必须在 `manifest.yaml` 填写实际来源 URL、commit 和许可证。
不得只依据论文描述重写第三方核心算法，也不得在运行时通过 Torch Hub
静默下载代码或权重。

安装时从仓库根执行：

```bash
python -m pip install -e external_sources/vggt
python -m pip install -e external_sources/salad
```

`gtsam_with_sl4` 必须按
[GTSAM 官方源码构建说明](https://github.com/borglab/gtsam/blob/develop/INSTALL.md)
编译安装 Python 绑定。安装后先确认：

```bash
python -c "import gtsam; print(all(hasattr(gtsam, x) for x in ('SL4', 'PriorFactorSL4', 'BetweenFactorSL4')))"
```

输出必须为 `True` 才能运行默认 SL(4) 基线。
