# M0 Handoff, KITTI 09, and Tooling Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable KITTI 09 evaluation path, document AutoDL artifact transfer and code-review-graph usage, and make the existing ATE implementation easier to review without changing SLAM behavior.

**Architecture:** Keep dataset conversion in a small `evaluation/kitti.py` module with a thin CLI wrapper. Reuse the existing TUM writer and evo ATE pipeline after converting KITTI 3x4 poses to frame-indexed TUM rows. Keep cloud transfer, KITTI execution, and graph-tool instructions in separate runbooks, while repository-local generated data remains ignored.

**Tech Stack:** Python 3.11, NumPy, SciPy Rotation, evo, pytest, Markdown, Git

---

## File Structure

New files:

```text
vggt_slam_pp/evaluation/kitti.py
vggt_slam_pp/cli/convert_kitti_trajectory.py
tests/unit/evaluation/test_kitti.py
tests/integration/test_convert_kitti_trajectory.py
docs/runbooks/cloud-artifact-transfer.md
docs/runbooks/m0-kitti09-autodl.md
docs/tools/code-review-graph-zh.md
```

Modified files:

```text
.gitignore
.code-review-graphignore
README.md
pyproject.toml
docs/algorithms/ate.md
docs/runbooks/m0-autodl-baseline.md
vggt_slam_pp/evaluation/ate.py
vggt_slam_pp/evaluation/tum.py
vggt_slam_pp/cli/evaluate_ate.py
```

Generated local files under `.code-review-graph/`, `.claude/`, `.mcp.json`, and
`CLAUDE.md` remain outside Git.

### Task 1: KITTI Pose Parser and Converter

**Files:**
- Create: `vggt_slam_pp/evaluation/kitti.py`
- Create: `tests/unit/evaluation/test_kitti.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add SciPy as a direct dependency**

Add to `project.dependencies`:

```toml
"scipy>=1.11,<2",
```

The converter directly uses `scipy.spatial.transform.Rotation`; it must not rely
on evo installing SciPy transitively.

- [ ] **Step 2: Write failing tests for valid KITTI poses**

Create `tests/unit/evaluation/test_kitti.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from vggt_slam_pp.evaluation.kitti import read_kitti_pose_rows


def test_kitti_rows_become_frame_indexed_tum_rows(tmp_path: Path) -> None:
    path = tmp_path / "poses.txt"
    path.write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n"
        "0 -1 0 2 1 0 0 3 0 0 1 4\n",
        encoding="utf-8",
    )

    rows = read_kitti_pose_rows(path)

    np.testing.assert_allclose(rows[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(rows[:, 1:4], [[0, 0, 0], [2, 3, 4]])
    np.testing.assert_allclose(rows[0, 4:8], [0, 0, 0, 1])
    np.testing.assert_allclose(
        np.abs(rows[1, 4:8]),
        [0, 0, np.sqrt(0.5), np.sqrt(0.5)],
    )
```

- [ ] **Step 3: Write failing tests for malformed KITTI data**

Append:

```python
@pytest.mark.parametrize(
    "text, message",
    [
        ("1 0 0\n", "expected 12 columns"),
        ("1 0 0 0 0 1 0 0 0 0 1 nan\n", "finite"),
        ("1 0 0 0 0 2 0 0 0 0 1 0\n", "rotation"),
    ],
)
def test_invalid_kitti_rows_are_rejected(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    path = tmp_path / "poses.txt"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_kitti_pose_rows(path)
```

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
conda run -n vggt-dem pytest -q tests/unit/evaluation/test_kitti.py
```

Expected: collection fails because `vggt_slam_pp.evaluation.kitti` does not
exist.

- [ ] **Step 5: Implement strict KITTI-to-TUM conversion**

Create `vggt_slam_pp/evaluation/kitti.py`:

```python
"""KITTI odometry pose parsing for the project's TUM evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_kitti_pose_rows(path: Path) -> np.ndarray:
    """Convert KITTI 3x4 poses to frame-indexed TUM trajectory rows."""
    tum_rows: list[list[float]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 12:
            raise ValueError(
                f"{path}:{line_number}: expected 12 columns, got {len(fields)}"
            )
        try:
            values = np.asarray([float(field) for field in fields], dtype=np.float64)
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: non-numeric KITTI pose"
            ) from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}:{line_number}: pose values must be finite")

        matrix = values.reshape(3, 4)
        rotation = matrix[:, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"{path}:{line_number}: invalid rotation matrix")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"{path}:{line_number}: invalid rotation determinant")

        quaternion = Rotation.from_matrix(rotation).as_quat()
        translation = matrix[:, 3]
        frame_id = float(len(tum_rows))
        tum_rows.append([frame_id, *translation, *quaternion])

    if len(tum_rows) < 2:
        raise ValueError("KITTI trajectory requires at least two poses")
    return np.asarray(tum_rows, dtype=np.float64)
```

- [ ] **Step 6: Run unit tests and verify GREEN**

Run:

```bash
conda run -n vggt-dem pytest -q tests/unit/evaluation/test_kitti.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml vggt_slam_pp/evaluation/kitti.py \
  tests/unit/evaluation/test_kitti.py
git commit -m "feat: convert KITTI poses for TUM evaluation"
```

### Task 2: KITTI Conversion CLI

**Files:**
- Create: `vggt_slam_pp/cli/convert_kitti_trajectory.py`
- Create: `tests/integration/test_convert_kitti_trajectory.py`

- [ ] **Step 1: Write a failing CLI integration test**

Create `tests/integration/test_convert_kitti_trajectory.py`:

```python
import subprocess
import sys
from pathlib import Path

import numpy as np

from vggt_slam_pp.evaluation.tum import read_tum_rows


def test_cli_writes_frame_indexed_tum_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "poses.txt"
    output = tmp_path / "nested" / "groundtruth.txt"
    source.write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n"
        "1 0 0 1 0 1 0 2 0 0 1 3\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vggt_slam_pp.cli.convert_kitti_trajectory",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = read_tum_rows(output)
    np.testing.assert_allclose(rows[:, 0], [0, 1])
    np.testing.assert_allclose(rows[:, 1:4], [[0, 0, 0], [1, 2, 3]])
    assert '"pose_count": 2' in result.stdout
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n vggt-dem pytest -q \
  tests/integration/test_convert_kitti_trajectory.py
```

Expected: fails because the CLI module does not exist.

- [ ] **Step 3: Implement the thin CLI**

Create `vggt_slam_pp/cli/convert_kitti_trajectory.py`:

```python
"""Convert KITTI odometry 3x4 poses to frame-indexed TUM format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_slam_pp.evaluation.kitti import read_kitti_pose_rows
from vggt_slam_pp.evaluation.tum import write_tum_rows
from vggt_slam_pp.io.checksums import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    rows = read_kitti_pose_rows(source)
    write_tum_rows(output, rows)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "input": str(source),
                "output": str(output),
                "pose_count": int(rows.shape[0]),
                "input_sha256": sha256_file(source),
                "output_sha256": sha256_file(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI and evaluation tests**

Run:

```bash
conda run -n vggt-dem pytest -q \
  tests/integration/test_convert_kitti_trajectory.py \
  tests/integration/test_evaluate_ate.py \
  tests/unit/evaluation/test_tum.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add vggt_slam_pp/cli/convert_kitti_trajectory.py \
  tests/integration/test_convert_kitti_trajectory.py
git commit -m "feat: add KITTI trajectory conversion CLI"
```

### Task 3: ATE Review Comments and Algorithm Notes

**Files:**
- Modify: `vggt_slam_pp/evaluation/ate.py`
- Modify: `vggt_slam_pp/evaluation/tum.py`
- Modify: `vggt_slam_pp/cli/evaluate_ate.py`
- Modify: `docs/algorithms/ate.md`

- [ ] **Step 1: Add focused Chinese comments to the ATE core**

In `evaluate_translation_ape`, add comments immediately before the three
mathematical stages:

```python
# 只比较时间上足够接近的位姿；未关联帧不进入 ATE。
reference, estimate = sync.associate_trajectories(...)

# 单目轨迹没有绝对尺度，先用 Sim(3) 求统一的尺度、旋转和平移。
estimate.align(reference, correct_scale=True)

# APE 使用对齐后位置差的 L2 范数，最终报告所有关联帧的 RMSE。
metric = metrics.APE(metrics.PoseRelation.translation_part)
```

- [ ] **Step 2: Explain duplicate timestamps in `tum.py`**

Add comments that state:

```python
# TUM 每行固定为 timestamp tx ty tz qx qy qz qw。
```

Before duplicate handling:

```python
# 相邻子图共享边界帧，因此原始 VGGT-SLAM 日志可能重复同一时间戳。
# 规范轨迹保留第一次出现，并记录被丢弃项与首项的最大分歧。
```

Before the quaternion angle:

```python
# q 与 -q 表示同一旋转，取点积绝对值后再计算最小夹角。
```

- [ ] **Step 3: Explain dual reports in the CLI**

Before raw evaluation:

```python
# paper_compatible 保留原始边界重复帧，便于复现既有基线口径。
```

Before the temporary directory:

```python
# canonical_unique 只用于第二次评测；临时去重轨迹不会写入正式产物。
```

- [ ] **Step 4: Expand `docs/algorithms/ate.md`**

Document:

- TUM eight-column format;
- timestamp association and `0.01 s` threshold;
- Umeyama Sim(3) alignment;
- equations for `e_i` and RMSE;
- why monocular trajectories require scale correction;
- `paper_compatible` versus `canonical_unique`;
- duplicate diagnostics;
- how to read the TUM desk results:
  - SL(4): `0.029152 m`;
  - Sim(3): `0.023354 m`;
  - bridge: `0.023354 m`;
- why matching SHA-256 between Sim(3) and bridge is stronger than similar ATE;
- KITTI frame IDs versus `times.txt`.

- [ ] **Step 5: Run ATE tests and compile checks**

Run:

```bash
conda run -n vggt-dem pytest -q \
  tests/integration/test_evaluate_ate.py \
  tests/unit/evaluation/test_tum.py
conda run -n vggt-dem python -m compileall -q vggt_slam_pp
```

Expected: all tests pass and compileall exits zero.

- [ ] **Step 6: Commit**

```bash
git add vggt_slam_pp/evaluation/ate.py \
  vggt_slam_pp/evaluation/tum.py \
  vggt_slam_pp/cli/evaluate_ate.py \
  docs/algorithms/ate.md
git commit -m "docs: explain ATE evaluation workflow"
```

### Task 4: Cloud Artifact Transfer Runbook

**Files:**
- Create: `docs/runbooks/cloud-artifact-transfer.md`

- [ ] **Step 1: Write the cloud packaging procedure**

The runbook must define:

```bash
RUN_ID=kitti09-pp-bridge-YYYYMMDD-HHMMSS
set -Eeuo pipefail

test -d "artifacts/m0/$RUN_ID"
test -d "artifacts/m0/submaps/$RUN_ID"
mkdir -p artifacts/packages
test ! -e "artifacts/packages/$RUN_ID.sha256"
test ! -e "artifacts/packages/$RUN_ID.tar.gz"
test ! -e "artifacts/packages/$RUN_ID.tar.gz.sha256"

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
```

Also document:

- download the `.tar.gz` and `.tar.gz.sha256` through AutoDL file manager;
- never package weights, datasets, or external source trees;
- estimate package size with `du -sh` before download;
- preserve the original package locally.

- [ ] **Step 2: Write the local verification and import procedure**

Include:

```bash
RUN_ID=kitti09-pp-bridge-YYYYMMDD-HHMMSS
set -Eeuo pipefail

IMPORT_ROOT="artifacts/imported/$RUN_ID"
IMPORT_TMP="$IMPORT_ROOT.partial"
mkdir -p "artifacts/packages" "artifacts/imported"
test ! -e "$IMPORT_ROOT"
test ! -e "$IMPORT_TMP"

shasum -a 256 -c "artifacts/packages/$RUN_ID.tar.gz.sha256"
mkdir "$IMPORT_TMP"
tar -xzf "artifacts/packages/$RUN_ID.tar.gz" \
  -C "$IMPORT_TMP"

(
  cd "$IMPORT_TMP"
  shasum -a 256 -c \
    "artifacts/packages/$RUN_ID.sha256"
)

mv "$IMPORT_TMP" "$IMPORT_ROOT"
```

Add a note that Linux uses `sha256sum` while macOS uses `shasum -a 256`.

- [ ] **Step 3: Add failure recovery**

Document:

- checksum mismatch: discard the downloaded copy and download again;
- missing submap directory: do not accept the package;
- interrupted cloud run: use a new `run_id`, never overwrite immutable caches;
- keep `artifacts/` ignored by Git.

- [ ] **Step 4: Review shell paths**

Verify every archive member and verification path uses the same relative layout.
Run:

```bash
rg -n "RUN_ID|sha256|tar -|artifacts/imported" \
  docs/runbooks/cloud-artifact-transfer.md
```

Expected: packaging, download, verification, and recovery sections are present.

- [ ] **Step 5: Commit**

```bash
git add docs/runbooks/cloud-artifact-transfer.md
git commit -m "docs: add cloud artifact transfer runbook"
```

### Task 5: KITTI 09 AutoDL Runbook

**Files:**
- Create: `docs/runbooks/m0-kitti09-autodl.md`
- Modify: `docs/runbooks/m0-autodl-baseline.md`

- [ ] **Step 1: Document upload and preflight**

Specify cloud placement:

```text
~/autodl-tmp/VGGT-SLAM/data/09/
├── image_2/
├── image_3/
├── calib.txt
├── times.txt
└── poses.txt
```

Include:

```bash
test "$(find data/09/image_2 -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 1591
test "$(wc -l < data/09/times.txt)" -eq 1591
test "$(wc -l < data/09/poses.txt)" -eq 1591
test -f data/09/image_2/000000.png
test -f data/09/image_2/001590.png

python -m vggt_slam_pp.cli.convert_kitti_trajectory \
  --input data/09/poses.txt \
  --output artifacts/reference/kitti09-groundtruth-tum.txt
```

Explain that conversion timestamps are frame IDs because the baseline extracts
IDs from image filenames.

- [ ] **Step 2: Document the bridge cache run**

Use a unique `run_id`:

```bash
RUN_ID=kitti09-pp-bridge-$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/m0 artifacts/m0/submaps
test ! -e "artifacts/m0/$RUN_ID"
test ! -e "artifacts/m0/submaps/$RUN_ID"
mkdir "artifacts/m0/$RUN_ID"

python VGGT-SLAM-version1.0/main.py \
  --image_folder data/09/image_2 \
  --vggt_weight weights/model.pt \
  --device cuda --use_sim3 --submap_size 32 --max_loops 0 \
  --run_id "$RUN_ID" --run_purpose pp_frontend_bridge \
  --export_submaps_dir artifacts/m0/submaps \
  --log_results --skip_dense_log \
  --log_path "artifacts/m0/$RUN_ID/poses.txt"

python -m vggt_slam_pp.cli.inspect_submap_cache \
  "artifacts/m0/submaps/$RUN_ID" \
  > "artifacts/m0/$RUN_ID/cache-inspection.json"

python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth artifacts/reference/kitti09-groundtruth-tum.txt \
  --estimate "artifacts/m0/$RUN_ID/poses.txt" \
  --output "artifacts/m0/$RUN_ID/ate.json"
```

- [ ] **Step 3: Document the SALAD loop baseline**

Use a separate unique ID:

```bash
RUN_ID=kitti09-sim3-salad-$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/m0
test ! -e "artifacts/m0/$RUN_ID"
mkdir "artifacts/m0/$RUN_ID"

python VGGT-SLAM-version1.0/main.py \
  --image_folder data/09/image_2 \
  --vggt_weight weights/model.pt \
  --salad_checkpoint weights/dino_salad.ckpt \
  --dinov2_source external_sources/dinov2 \
  --dinov2_weight weights/dinov2_vitb14_pretrain.pth \
  --device cuda --use_sim3 --submap_size 32 --max_loops 1 \
  --run_id "$RUN_ID" --run_purpose baseline_reference \
  --log_results --skip_dense_log \
  --log_path "artifacts/m0/$RUN_ID/poses.txt"

python -m vggt_slam_pp.cli.evaluate_ate \
  --groundtruth artifacts/reference/kitti09-groundtruth-tum.txt \
  --estimate "artifacts/m0/$RUN_ID/poses.txt" \
  --output "artifacts/m0/$RUN_ID/ate.json"
```

State that M0 records the detected loop count but does not assume it must be
non-zero until the actual run is reviewed.

- [ ] **Step 4: Add acceptance checklist**

Require:

- 1591 input frames found;
- no traceback;
- pose log exists and is non-empty;
- ATE is finite;
- associated pose count is at least 10;
- bridge cache inspector reports `ok: true`;
- first, middle, and last submap arrays are finite;
- scale units remain `relative_map_unit`;
- package and download follow `cloud-artifact-transfer.md`.

- [ ] **Step 5: Link from the existing M0 runbook**

At the end of `m0-autodl-baseline.md`, add:

```markdown
## 5. 长序列 KITTI 09

TUM desk 验收通过后，按
[`m0-kitti09-autodl.md`](../../runbooks/m0-kitti09-autodl.md)
执行长序列桥接、回环基线、ATE 和产物回传。
```

- [ ] **Step 6: Commit**

```bash
git add docs/runbooks/m0-kitti09-autodl.md \
  docs/runbooks/m0-autodl-baseline.md
git commit -m "docs: add KITTI 09 AutoDL workflow"
```

### Task 6: code-review-graph Guide and Repository Hygiene

**Files:**
- Modify: `.gitignore`
- Add: `.code-review-graphignore`
- Create: `docs/tools/code-review-graph-zh.md`
- Delete: repository `.DS_Store` files only

- [ ] **Step 1: Normalize `.gitignore`**

Keep the existing generated-graph rule and add:

```gitignore
/.code-review-graph/
/.claude/
/.mcp.json
/CLAUDE.md
```

Do not add broad patterns that hide ordinary Markdown or JSON files.

- [ ] **Step 2: Review and stage `.code-review-graphignore`**

Keep exclusions for:

- `external_sources/**`;
- `VGGT-SLAM-version1.0/**`;
- data, weights, artifacts, outputs, logs, papers, archives;
- Python caches and package metadata;
- local assistant state.

Add these local graph/client files if missing:

```gitignore
.code-review-graph/**
.claude/**
.mcp.json
CLAUDE.md
```

- [ ] **Step 3: Write the Chinese graph guide**

Create `docs/tools/code-review-graph-zh.md` with:

- generated file table;
- why `graph.db`, `graph.html`, and `wiki/` are not committed;
- current graph scale: 43 files, 207 nodes, 1764 edges;
- opening `graph.html` from the tool or a local HTTP server;
- control reference for Search, Flows, Communities, Fit, Labels, node filters,
  and edge filters;
- recommended workflow:

```text
Search target
-> select execution flow
-> inspect callers/callees
-> inspect tests_for
-> inspect impact radius
-> edit
-> detect changes
-> run real tests
```

- explanation that a compressed global graph is normal;
- graph limitations around runtime imports and data-dependent behavior;
- local-only status of `.mcp.json`, `.claude/`, and `CLAUDE.md`.

- [ ] **Step 4: Remove only `.DS_Store` files**

Before deletion, list them:

```bash
find . -name .DS_Store -type f -print
```

Delete exactly those files after confirming the list contains no project files:

```bash
find . -name .DS_Store -type f -delete
```

- [ ] **Step 5: Verify ignored and tracked boundaries**

Run:

```bash
git check-ignore -v \
  .code-review-graph/graph.db \
  .claude/settings.json \
  .mcp.json \
  CLAUDE.md \
  data/09/image_2/000000.png \
  weights/model.pt

git status --short
```

Expected:

- local graph/client/data/weight files are ignored;
- `.code-review-graphignore` and the tool guide are visible for commit;
- user M1/M2-M4 plans and papers are not staged.

- [ ] **Step 6: Commit**

```bash
git add .gitignore .code-review-graphignore docs/tools/code-review-graph-zh.md
git commit -m "docs: document code review graph workflow"
```

### Task 7: README Navigation and M0 Handoff Summary

**Files:**
- Modify: `README.md`
- Modify: `docs/runbooks/m0-local-validation.md`

- [ ] **Step 1: Replace the one-line README with a concise project map**

Include:

- reproduction goal;
- baseline source isolation policy;
- current status: M0 cloud baseline and bridge validated;
- local versus AutoDL responsibilities;
- links to:
  - M0 local validation;
  - M0 AutoDL baseline;
  - KITTI 09 AutoDL;
  - cloud artifact transfer;
  - ATE algorithm;
  - submap cache contract;
  - code-review-graph guide;
  - M1/M2-M4 design and plans.

- [ ] **Step 2: Add local handoff steps**

Append to `m0-local-validation.md`:

```bash
git pull origin codex/m0-reproduction
conda run -n vggt-dem python -m pip install -e ".[dev]"
conda run -n vggt-dem pytest -q
```

Then document:

- download cloud packages into `artifacts/packages/`;
- verify and extract into `artifacts/imported/<run-id>/`;
- inspect `ate.json`, `cache-inspection.json`, and `final_state.json`;
- do not attempt full DINOv2/VGGT CUDA inference on the Mac as an M0
  acceptance requirement.

- [ ] **Step 3: Check all Markdown links**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

for path in [Path("README.md"), *Path("docs").rglob("*.md")]:
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        if not resolved.exists():
            raise SystemExit(f"{path}: missing link target {target}")
print("Markdown links: OK")
PY
```

Expected: `Markdown links: OK`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/runbooks/m0-local-validation.md
git commit -m "docs: add reproduction workflow navigation"
```

### Task 8: Full Verification and Review Checkpoint

**Files:**
- Review all files changed in Tasks 1-7

- [ ] **Step 1: Install the updated editable package**

Run:

```bash
conda run -n vggt-dem python -m pip install -e ".[dev]"
```

Expected: installation succeeds with SciPy satisfied.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
conda run -n vggt-dem pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run compile and whitespace checks**

Run:

```bash
conda run -n vggt-dem python -m compileall -q \
  vggt_slam_pp VGGT-SLAM-version1.0
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 4: Exercise the converter against local KITTI 09**

Run:

```bash
conda run -n vggt-dem python \
  -m vggt_slam_pp.cli.convert_kitti_trajectory \
  --input data/09/poses.txt \
  --output /tmp/kitti09-groundtruth-tum.txt

wc -l data/09/poses.txt /tmp/kitti09-groundtruth-tum.txt
head -n 2 /tmp/kitti09-groundtruth-tum.txt
tail -n 2 /tmp/kitti09-groundtruth-tum.txt
```

Expected:

- both files have 1591 rows;
- first TUM timestamp is `0`;
- last TUM timestamp is `1590`;
- every output row has 8 numeric columns.

- [ ] **Step 5: Review Git boundaries**

Run:

```bash
git status --short
git diff --stat HEAD~7..HEAD
```

Confirm:

- no data, weights, `.code-review-graph/`, `.claude/`, `.mcp.json`, or
  `CLAUDE.md` are committed;
- user-authored M1/M2-M4 plans and papers remain untouched;
- no baseline SLAM algorithm file changed.

- [ ] **Step 6: Request code review**

Use `superpowers:requesting-code-review` with emphasis on:

- KITTI frame-index association;
- rotation validation and quaternion ordering;
- cloud package checksum paths;
- runbook command safety;
- Git ignore boundaries;
- ATE comments matching actual evo behavior.

- [ ] **Step 7: Push the completed branch**

```bash
git push origin codex/m0-reproduction
```

Expected: remote branch contains all reviewed commits.
