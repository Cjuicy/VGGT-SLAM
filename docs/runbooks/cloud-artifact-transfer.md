# 云端产物传输与本地校验

本手册用于把 AutoDL 上一次不可变运行的轨迹、ATE 报告和子图缓存传回
Mac。本流程不使用 Git 传输运行产物；Git 只保存代码、配置和小型审核文档。

所有命令都从仓库根目录执行。先把示例 `RUN_ID` 替换为本次实际运行编号：

```bash
RUN_ID=kitti09-pp-bridge-YYYYMMDD-HHMMSS
```

## 1. 云端打包

先确认主运行目录和子图目录同时存在，并估算下载体积：

```bash
set -Eeuo pipefail

test -d "artifacts/m0/$RUN_ID"
test -d "artifacts/m0/submaps/$RUN_ID"
test "$(find "artifacts/m0/$RUN_ID" "artifacts/m0/submaps/$RUN_ID" \
  -type f | wc -l)" -gt 0
du -sh "artifacts/m0/$RUN_ID" "artifacts/m0/submaps/$RUN_ID"

mkdir -p artifacts/packages
test ! -e "artifacts/packages/$RUN_ID.sha256"
test ! -e "artifacts/packages/$RUN_ID.tar.gz"
test ! -e "artifacts/packages/$RUN_ID.tar.gz.sha256"

trap 'rm -f \
  "artifacts/packages/$RUN_ID.sha256" \
  "artifacts/packages/$RUN_ID.tar.gz" \
  "artifacts/packages/$RUN_ID.tar.gz.sha256"' ERR

find "artifacts/m0/$RUN_ID" "artifacts/m0/submaps/$RUN_ID" \
  -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "artifacts/packages/$RUN_ID.sha256"

tar -czf "artifacts/packages/$RUN_ID.tar.gz" \
  "artifacts/m0/$RUN_ID" \
  "artifacts/m0/submaps/$RUN_ID" \
  "artifacts/packages/$RUN_ID.sha256"

sha256sum "artifacts/packages/$RUN_ID.tar.gz" \
  > "artifacts/packages/$RUN_ID.tar.gz.sha256"

trap - ERR
du -sh "artifacts/packages/$RUN_ID.tar.gz"
```

缺少任一目录、没有产物文件、目标包已存在或任一命令失败时，脚本都会停止。
`ERR` trap 会删除本次未完成的清单和压缩包，不会留下可被误认为有效交接件的
半成品。

压缩包只允许包含：

- `artifacts/m0/$RUN_ID/`；
- `artifacts/m0/submaps/$RUN_ID/`；
- 对应的逐文件校验清单。

不要打包 `weights/`、`data/`、`external_sources/`、完整 Conda 环境或其他运行
编号。打包前后都用 `du -sh` 检查体积；异常增大时先检查归档成员，不要直接
下载。

## 2. 从 AutoDL 下载

在 AutoDL 文件管理器中下载这两个文件：

```text
artifacts/packages/<RUN_ID>.tar.gz
artifacts/packages/<RUN_ID>.tar.gz.sha256
```

在本地仓库中保持相同相对路径，放入 `artifacts/packages/`。原始 `.tar.gz`
是本次云端运行的只读交接件，校验通过后也不要修改或重新压缩；后续分析使用
解压副本。

## 3. Mac 校验与导入

Linux 使用 `sha256sum`，macOS 使用 `shasum -a 256`。先验证压缩包，再解压，
最后按云端生成的清单逐文件验证：

```bash
set -Eeuo pipefail

RUN_ID=kitti09-pp-bridge-YYYYMMDD-HHMMSS
IMPORT_ROOT="artifacts/imported/$RUN_ID"
IMPORT_TMP="$IMPORT_ROOT.partial"

mkdir -p "artifacts/packages" "artifacts/imported"
test ! -e "$IMPORT_ROOT"
test ! -e "$IMPORT_TMP"

shasum -a 256 -c "artifacts/packages/$RUN_ID.tar.gz.sha256"
mkdir "$IMPORT_TMP"
trap 'rm -rf "$IMPORT_TMP"' ERR

tar -xzf "artifacts/packages/$RUN_ID.tar.gz" \
  -C "$IMPORT_TMP"

(
  cd "$IMPORT_TMP"
  shasum -a 256 -c \
    "artifacts/packages/$RUN_ID.sha256"
)

mv "$IMPORT_TMP" "$IMPORT_ROOT"
trap - ERR
```

压缩包内保留了从仓库根开始的 `artifacts/...` 相对布局，因此逐文件清单中的
路径能在 `artifacts/imported/$RUN_ID` 下直接解析。校验通过后的主要路径为：

```text
artifacts/imported/<RUN_ID>/artifacts/m0/<RUN_ID>/
artifacts/imported/<RUN_ID>/artifacts/m0/submaps/<RUN_ID>/
```

可再运行缓存结构检查：

```bash
python -m vggt_slam_pp.cli.inspect_submap_cache \
  "artifacts/imported/$RUN_ID/artifacts/m0/submaps/$RUN_ID"
```

## 4. 失败恢复

- 压缩包 SHA-256 不匹配：删除本次下载副本，通过 AutoDL 文件管理器重新
  下载；不要尝试修补压缩包。
- 解压后的逐文件校验失败：保留云端原包用于定位，删除本地解压副本，重新
  下载并从压缩包校验开始。失败的 `.partial` 目录会自动清理。
- 缺少 `artifacts/m0/submaps/$RUN_ID`：不接受该交接包；先回到云端检查桥接
  命令、`--export_submaps_dir` 和运行日志。
- 云端运行中断：使用新的 `run_id` 重新运行。不要覆盖旧运行目录、子图缓存
  或已经生成的包。
- `artifacts/` 必须保持在 `.gitignore` 中。不要用 `git add -f` 上传运行
  产物。

每个 `RUN_ID` 对应一次不可变运行。代码提交、命令、配置、输入校验和以及
输出包校验和共同构成该次结果的审核记录。
