# M0 AutoDL 基线与桥接运行

所有命令从仓库根执行。先通过 Git 拉取小文件，再把大文件放到完全相同的
相对路径。

## 0. 通过 HTTPS 拉取并保留提交能力

AutoDL 不需要预先配置 SSH。项目仓库使用 HTTPS 拉取，推送时使用 GitHub
Fine-grained Personal Access Token。

### 0.1 创建最小权限 Token

打开 [GitHub Fine-grained personal access tokens](https://github.com/settings/personal-access-tokens/new)，
创建短期 Token：

- Repository access 只选择 `Cjuicy/VGGT-SLAM`。
- Repository permissions 中 `Contents` 设为 `Read and write`。
- 设置较短的有效期，M0 云端验证结束后撤销。

Token 不得写入命令、URL、脚本、YAML、Git remote 或审核文档。

### 0.2 克隆 M0 分支

```bash
git clone --branch codex/m0-reproduction \
  https://github.com/Cjuicy/VGGT-SLAM.git
cd VGGT-SLAM

git switch -c autodl/m0-validation
git config user.name "Cjuicy"
git config user.email "替换为你的 GitHub 邮箱"
```

公开仓库的克隆不要求 Token。若仓库以后设为私有，Git 会提示输入：

```text
Username: Cjuicy
Password: 粘贴 Fine-grained Token
```

这里的 `Password` 是 Token，不是 GitHub 登录密码。终端输入时不会显示字符。

### 0.3 提交云端审核证据

权重、数据、外部源码、NPZ 缓存和完整日志均已被 `.gitignore` 排除。只提交
`docs/reviews/m0/<run-id>/` 中的小型环境、命令、ATE 摘要和偏差说明：

```bash
git status --short
git add docs/reviews/m0
git commit -m "docs: record AutoDL M0 validation"
git push -u origin autodl/m0-validation
```

第一次推送时按提示输入 GitHub 用户名和 Token。需要减少重复输入时，可以
临时启用一小时的内存凭据缓存：

```bash
git config credential.helper "cache --timeout=3600"
```

验证完成后清理缓存并在 GitHub 撤销 Token：

```bash
git credential-cache exit
```

不要执行 `git add -f` 绕过大文件忽略规则。

## 文件放置

```text
weights/model.pt
weights/dino_salad.ckpt
weights/dinov2_vitb14_pretrain.pth
data/rgbd_dataset_freiburg1_desk/
VGGT-SLAM-version1.0/office_loop/
external_sources/vggt/
external_sources/salad/
external_sources/gtsam_with_sl4/
```

外部源码的官方链接、核验分支和下载命令见
[`external_sources/README.md`](../../external_sources/README.md)。这些目录不通过
项目 Git 仓库传输。

本地已有压缩包时可在云端解压：

```bash
tar -xzf data/rgbd_dataset_freiburg1_desk.tgz -C data
unzip VGGT-SLAM-version1.0/office_loop.zip -d VGGT-SLAM-version1.0
```

## 环境

AutoDL 镜像应先提供与 CUDA 匹配的 PyTorch。随后安装项目和已审核源码：

```bash
python -m pip install -e ".[dev]"
python -m pip install -e external_sources/vggt
python -m pip install -e external_sources/salad
```

按 `external_sources/gtsam_with_sl4` 的上游说明编译安装后，确认三个 SL(4)
符号存在。不要执行原版 `setup.sh`，它会克隆未锁定版本的依赖。

```bash
python scripts/verify_assets.py --config configs/runtime/autodl_cuda.yaml
python -c "import gtsam; print(gtsam.SL4, gtsam.PriorFactorSL4, gtsam.BetweenFactorSL4)"
```

## 1. Office-loop SL(4)，关闭导出

```bash
python VGGT-SLAM-version1.0/main.py \
  --image_folder VGGT-SLAM-version1.0/office_loop \
  --vggt_weight weights/model.pt \
  --salad_checkpoint weights/dino_salad.ckpt \
  --device cuda --submap_size 16 --max_loops 1 \
  --run_id office-sl4-no-export --run_purpose baseline_reference \
  --log_results --skip_dense_log \
  --log_path artifacts/m0/office-sl4-no-export/poses.txt
```

## 2. Office-loop SL(4)，开启导出

```bash
python VGGT-SLAM-version1.0/main.py \
  --image_folder VGGT-SLAM-version1.0/office_loop \
  --vggt_weight weights/model.pt \
  --salad_checkpoint weights/dino_salad.ckpt \
  --device cuda --submap_size 16 --max_loops 1 \
  --run_id office-sl4-export --run_purpose baseline_reference \
  --export_submaps_dir artifacts/m0/submaps \
  --log_results --skip_dense_log \
  --log_path artifacts/m0/office-sl4-export/poses.txt

python -m vggt_slam_pp.cli.inspect_submap_cache \
  artifacts/m0/submaps/office-sl4-export
```

根据两次终端输出记录 `submap_count` 和 `loop_count`，分别建立
`artifacts/m0/office-*/summary.json`，字段为：

```json
{"submap_count": 0, "loop_count": 0, "pose_log": "poses.txt"}
```

把占位的零替换为实际值，然后比较：

```bash
python -m vggt_slam_pp.cli.compare_baseline_runs \
  --left artifacts/m0/office-sl4-no-export/summary.json \
  --right artifacts/m0/office-sl4-export/summary.json \
  --output artifacts/m0/office-export-comparison.json
```

## 3. TUM desk 三种运行

### 默认 SL(4) 对照

```bash
python VGGT-SLAM-version1.0/main.py \
  --image_folder data/rgbd_dataset_freiburg1_desk/rgb \
  --vggt_weight weights/model.pt \
  --salad_checkpoint weights/dino_salad.ckpt \
  --device cuda --submap_size 32 --max_loops 1 \
  --run_id tum-desk-sl4 --run_purpose baseline_reference \
  --log_results --skip_dense_log \
  --log_path artifacts/m0/tum-desk-sl4/poses.txt
```

### Sim(3) 兼容基线对照

```bash
python VGGT-SLAM-version1.0/main.py \
  --image_folder data/rgbd_dataset_freiburg1_desk/rgb \
  --vggt_weight weights/model.pt \
  --salad_checkpoint weights/dino_salad.ckpt \
  --device cuda --use_sim3 --submap_size 32 --max_loops 1 \
  --run_id tum-desk-sim3 --run_purpose baseline_reference \
  --log_results --skip_dense_log \
  --log_path artifacts/m0/tum-desk-sim3/poses.txt
```

### VGGT-SLAM++ 前端桥接

```bash
python VGGT-SLAM-version1.0/main.py \
  --image_folder data/rgbd_dataset_freiburg1_desk/rgb \
  --vggt_weight weights/model.pt \
  --device cuda --use_sim3 --submap_size 32 --max_loops 0 \
  --run_id tum-desk-pp-bridge --run_purpose pp_frontend_bridge \
  --export_submaps_dir artifacts/m0/submaps \
  --log_results --skip_dense_log \
  --log_path artifacts/m0/tum-desk-pp-bridge/poses.txt

python -m vggt_slam_pp.cli.inspect_submap_cache \
  artifacts/m0/submaps/tum-desk-pp-bridge
```

桥接命令不需要 SALAD checkpoint，因为 `max_loops=0` 会跳过检索模型。

## 4. ATE

对三个 TUM 运行分别执行：

```bash
python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth data/rgbd_dataset_freiburg1_desk/groundtruth.txt \
  --estimate artifacts/m0/tum-desk-sl4/poses.txt \
  --output artifacts/m0/tum-desk-sl4/ate.json

python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth data/rgbd_dataset_freiburg1_desk/groundtruth.txt \
  --estimate artifacts/m0/tum-desk-sim3/poses.txt \
  --output artifacts/m0/tum-desk-sim3/ate.json

python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth data/rgbd_dataset_freiburg1_desk/groundtruth.txt \
  --estimate artifacts/m0/tum-desk-pp-bridge/poses.txt \
  --output artifacts/m0/tum-desk-pp-bridge/ate.json
```

人工审核要求：ATE 有限、规范轨迹关联帧数不少于 10、桥接图状态的回环数
为 0，并抽查第一个、中间和最后一个子图的形状、有限值与校验和。
