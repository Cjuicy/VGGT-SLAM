# VGGT-SLAM++ M0 Dual-Mode Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 冻结新提供的 VGGT-SLAM 原始基线，分别验证 SL(4) 与 Sim3-compatible 路径，并建立不改变基线估计行为的子图桥接缓存和 ATE 评测闭环。

**Architecture:** `VGGT-SLAM-version1.0/` 只保存原始基线和最小兼容桥接；所有数据契约、缓存、评测、预检和命令行工具放在 `vggt_slam_pp/`。不可变子图几何与可变图状态分开保存，避免后续图优化覆盖或混淆局部尺度。

**Tech Stack:** Python 3.11、NumPy、Pydantic 2、PyYAML、pytest、evo、Conda、VGGT、SALAD、GTSAM Pose3/SL(4)。

---

关联规格：

- `docs/superpowers/specs/2026-06-11-vggt-slam-pp-m0-dual-baseline-design-zh.md`
- 本计划只完成 M0；用户审核 M0 产物后才进入 M1。

## File Map

```text
VGGT-SLAM-version1.0/               新原始基线；仅 main.py、solver.py、
                                    loop_closure.py 允许兼容性修改
pyproject.toml                      旁路包定义和 pytest 配置
configs/runtime/                    本地和 AutoDL 资产路径
environment/                        环境说明与云端依赖清单
external_sources/                   第三方源码来源清单
vggt_slam_pp/contracts/             子图、图状态、运行身份契约
vggt_slam_pp/io/                    原子缓存和校验和
vggt_slam_pp/adapters/              只读 VGGT-SLAM Submap 适配
vggt_slam_pp/evaluation/            TUM 轨迹规范化与 ATE
vggt_slam_pp/cli/                   缓存检查和 ATE 命令
scripts/verify_assets.py            运行前资产/能力预检
tests/                              单元、集成和静态基线测试
docs/provenance/                    基线与外部源码来源
docs/algorithms/                    缓存和 ATE 字段说明
docs/runbooks/                      本地和 AutoDL 操作步骤
```

## Task 1: Replace The Modified Baseline With The Verified Archive

**Files:**
- Replace directory: `VGGT-SLAM-version1.0/`
- Remove directory: `VGGT-SLAM-version1.0 2/`
- Create: `docs/provenance/vggt-slam-v1.md`
- Test: `tests/static/test_baseline_snapshot.py`

- [ ] **Step 1: Write the failing snapshot test**

```python
from hashlib import sha256
from pathlib import Path


ARCHIVE_SHA256 = (
    "f34897e5745c6380dfd819bf87c8a016"
    "aebb8e9ffe7a0025304015fa7b0f0411"
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_verified_archive_is_present() -> None:
    assert digest(Path("VGGT-SLAM-version1.0.zip")) == ARCHIVE_SHA256


def test_modified_baseline_artifacts_are_absent() -> None:
    root = Path("VGGT-SLAM-version1.0")
    for name in ("main_offline.py", "findings.md", "progress.md", "task_plan.md"):
        assert not (root / name).exists()
    assert not (root / "vggt").exists()


def test_temporary_baseline_directory_is_absent() -> None:
    assert not Path("VGGT-SLAM-version1.0 2").exists()
```

- [ ] **Step 2: Run the test and verify the current modified baseline fails**

Run:

```bash
conda run -n vggt-dem pytest tests/static/test_baseline_snapshot.py -v
```

Expected: FAIL because the old modified files and temporary directory still exist.

- [ ] **Step 3: Replace the directory from the verified archive**

Move the old modified directory to `/tmp/vggt-slam-modified-backup-20260611`,
extract `VGGT-SLAM-version1.0.zip` into the project root, compare the extracted
tree with `VGGT-SLAM-version1.0 2/`, then remove the temporary duplicate.

- [ ] **Step 4: Record provenance**

`docs/provenance/vggt-slam-v1.md` must record:

```markdown
# VGGT-SLAM 1.0 Baseline Provenance

- Upstream: https://github.com/MIT-SPARK/VGGT-SLAM
- Local archive: `VGGT-SLAM-version1.0.zip`
- Archive SHA-256: `f34897e5745c6380dfd819bf87c8a016aebb8e9ffe7a0025304015fa7b0f0411`
- Canonical source directory: `VGGT-SLAM-version1.0/`
- Replaced modified copy: `/tmp/vggt-slam-modified-backup-20260611`

The default path is 15DoF SL(4). `--use_sim3` estimates scale outside the
graph and optimizes 6DoF Pose3 factors; this project names it
`baseline_sim3_compat`.
```

- [ ] **Step 5: Verify and commit**

```bash
conda run -n vggt-dem pytest tests/static/test_baseline_snapshot.py -v
git diff --check
git add VGGT-SLAM-version1.0 tests/static docs/provenance
git commit -m "chore: replace baseline with verified VGGT-SLAM snapshot"
```

## Task 2: Scaffold The Auditable Sidecar Package

**Files:**
- Create: `pyproject.toml`
- Create: `vggt_slam_pp/__init__.py`
- Create: `vggt_slam_pp/contracts/__init__.py`
- Create: `vggt_slam_pp/io/__init__.py`
- Create: `vggt_slam_pp/adapters/__init__.py`
- Create: `vggt_slam_pp/evaluation/__init__.py`
- Create: `vggt_slam_pp/cli/__init__.py`
- Test: `tests/unit/test_package.py`

- [ ] **Step 1: Write the failing import test**

```python
def test_package_version() -> None:
    import vggt_slam_pp

    assert vggt_slam_pp.__version__ == "0.1.0"
```

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/unit/test_package.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Add package metadata**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vggt-slam-pp-reproduction"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "numpy>=1.26,<3",
  "pydantic>=2.7,<3",
  "PyYAML>=6,<7",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9"]

[tool.setuptools.packages.find]
include = ["vggt_slam_pp*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

`vggt_slam_pp/__init__.py`:

```python
"""Auditable VGGT-SLAM++ reproduction sidecar."""

__version__ = "0.1.0"
```

Each subpackage initializer contains a one-line responsibility docstring.

- [ ] **Step 4: Install, verify, and commit**

```bash
conda run -n vggt-dem python -m pip install -e ".[dev]"
conda run -n vggt-dem pytest tests/unit/test_package.py -v
git add pyproject.toml vggt_slam_pp tests/unit/test_package.py
git commit -m "build: scaffold VGGT-SLAM++ sidecar package"
```

## Task 3: Add Runtime Identities And Asset Preflight

**Files:**
- Create: `vggt_slam_pp/contracts/runtime.py`
- Create: `configs/runtime/local_macos.yaml`
- Create: `configs/runtime/autodl_cuda.yaml`
- Create: `scripts/verify_assets.py`
- Create: `tests/unit/test_runtime_contract.py`
- Create: `tests/unit/test_verify_assets.py`
- Create: `environment/compatibility.md`

- [ ] **Step 1: Write failing contract tests**

```python
import pytest
from pydantic import ValidationError

from vggt_slam_pp.contracts.runtime import RunIdentity


def test_bridge_requires_sim3_compat_and_no_baseline_loops() -> None:
    with pytest.raises(ValidationError):
        RunIdentity(
            run_id="bad",
            solver_mode="baseline_sl4",
            run_purpose="pp_frontend_bridge",
            max_loops=1,
            submap_size=32,
            min_disparity=50.0,
        )


def test_reference_accepts_sl4() -> None:
    identity = RunIdentity(
        run_id="tum-desk-sl4",
        solver_mode="baseline_sl4",
        run_purpose="baseline_reference",
        max_loops=1,
        submap_size=32,
        min_disparity=50.0,
    )
    assert identity.solver_mode == "baseline_sl4"
```

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/unit/test_runtime_contract.py -v
```

- [ ] **Step 3: Implement the frozen runtime contract**

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class RunIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    solver_mode: Literal["baseline_sl4", "baseline_sim3_compat"]
    run_purpose: Literal["baseline_reference", "pp_frontend_bridge"]
    max_loops: int
    submap_size: int
    min_disparity: float

    @model_validator(mode="after")
    def validate_bridge_mode(self) -> "RunIdentity":
        if self.run_purpose == "pp_frontend_bridge":
            if self.solver_mode != "baseline_sim3_compat":
                raise ValueError("pp_frontend_bridge requires baseline_sim3_compat")
            if self.max_loops != 0:
                raise ValueError("pp_frontend_bridge requires max_loops=0")
        return self
```

- [ ] **Step 4: Write failing asset tests**

Tests create a temporary YAML config and assert that:

- matching files and SHA-256 pass;
- a missing file fails with its exact relative path;
- `baseline_sl4` fails when `gtsam.SL4`, `PriorFactorSL4`, or
  `BetweenFactorSL4` is absent;
- `baseline_sim3_compat` requires Pose3 but not SL4;
- no network download is attempted.

- [ ] **Step 5: Implement `verify_assets.py`**

The script parses YAML with:

```yaml
runtime_id: local_macos
assets:
  vggt_weight:
    path: weights/model.pt
    sha256: d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0
  salad_weight:
    path: weights/dino_salad.ckpt
    sha256: 6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada
capabilities:
  require_cuda: false
  require_sl4: false
```

It prints a JSON report and exits non-zero on any missing asset, hash mismatch,
CUDA mismatch, or required GTSAM symbol mismatch.

- [ ] **Step 6: Verify and commit**

```bash
conda run -n vggt-dem pytest tests/unit/test_runtime_contract.py tests/unit/test_verify_assets.py -v
conda run -n vggt-dem python scripts/verify_assets.py --config configs/runtime/local_macos.yaml
git add configs environment scripts vggt_slam_pp/contracts tests/unit
git commit -m "feat: add M0 runtime and asset preflight"
```

## Task 4: Define Immutable Submap And Graph-State Contracts

**Files:**
- Create: `vggt_slam_pp/contracts/submap.py`
- Create: `vggt_slam_pp/contracts/graph_state.py`
- Create: `tests/unit/contracts/test_submap.py`
- Create: `tests/unit/contracts/test_graph_state.py`
- Create: `docs/algorithms/submap-cache-contract.md`

- [ ] **Step 1: Write failing shape and mode tests**

Tests assert:

- points are `(S,H,W,3)`;
- colors match points and are `uint8`;
- confidence and mask are `(S,H,W)`;
- intrinsics are `(S,3,3)`;
- camera transforms are `(S,4,4)`;
- `pp_frontend_bridge` has no loop sources;
- `SE3_with_baked_scale` is only valid for `baseline_sim3_compat`;
- all arrays are finite and copied read-only.

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/unit/contracts -v
```

- [ ] **Step 3: Implement contracts**

Define frozen Pydantic metadata models and NumPy payload dataclasses:

```python
@dataclass(frozen=True)
class SubmapArrays:
    points_submap: np.ndarray
    colors_rgb: np.ndarray
    confidence: np.ndarray
    confidence_mask: np.ndarray
    intrinsics: np.ndarray
    camera_to_submap: np.ndarray


class SubmapMetadata(BaseModel):
    schema_version: Literal[1] = 1
    run: RunIdentity
    submap_id: int
    frame_ids: tuple[str, ...]
    last_non_loop_frame_index: int
    coordinate_frame: Literal["vggt_submap"]
    unit_state: Literal["relative_map_unit"]
    scale_baked_into_geometry: bool
    loop_sources: tuple[int, ...]
    baseline_sha256: str
    weight_sha256: str
```

`GraphState` contains update index, all `world_from_submap` matrices, transform
kind, edge/loop counts, and trajectory checksum.

- [ ] **Step 4: Document transforms, verify, and commit**

```bash
conda run -n vggt-dem pytest tests/unit/contracts -v
git add vggt_slam_pp/contracts tests/unit/contracts docs/algorithms
git commit -m "feat: define M0 submap and graph-state contracts"
```

## Task 5: Implement Atomic Cache I/O

**Files:**
- Create: `vggt_slam_pp/io/checksums.py`
- Create: `vggt_slam_pp/io/submap_cache.py`
- Create: `vggt_slam_pp/io/graph_state.py`
- Create: `tests/unit/io/test_submap_cache.py`
- Create: `tests/unit/io/test_graph_state.py`

- [ ] **Step 1: Write failing round-trip and corruption tests**

Tests assert:

- write/read preserves every array and metadata field;
- existing immutable submap directories cannot be overwritten;
- temporary directories are removed after a failed write;
- a modified NPZ or JSON fails checksum verification;
- graph state snapshots may append monotonically and `final_state.json`
  atomically points to the latest complete state.

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/unit/io -v
```

- [ ] **Step 3: Implement atomic writers**

Write into a sibling temporary directory, fsync files, compute SHA-256, then
rename to the final directory. Use JSON with sorted keys and `allow_pickle=False`
for NPZ reads.

- [ ] **Step 4: Verify and commit**

```bash
conda run -n vggt-dem pytest tests/unit/io -v
git add vggt_slam_pp/io tests/unit/io
git commit -m "feat: add atomic M0 cache storage"
```

## Task 6: Adapt VGGT-SLAM Submaps Without Mutation

**Files:**
- Create: `vggt_slam_pp/adapters/vggt_slam_v1.py`
- Create: `tests/unit/adapters/test_vggt_slam_v1.py`
- Create: `tests/fixtures/submap_factory.py`

- [ ] **Step 1: Write failing adapter tests**

Use a fake object with the same public fields as the baseline `Submap`. Assert:

- adapter copies post-`add_points()` point clouds and poses;
- scale-baked Sim3-compatible data is not reconstructed from raw predictions;
- camera poses are converted to homogeneous `(S,4,4)`;
- confidence mask uses the baseline threshold;
- changing exported arrays does not change the source object;
- bridge mode rejects non-empty loop sources.

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/unit/adapters -v
```

- [ ] **Step 3: Implement the adapter**

Expose:

```python
def adapt_submap(
    submap: object,
    *,
    run: RunIdentity,
    baseline_sha256: str,
    weight_sha256: str,
    loop_sources: tuple[int, ...],
) -> tuple[SubmapMetadata, SubmapArrays]:
    ...
```

The function accesses only documented Submap fields and returns defensive
copies. It never imports VGGT, SALAD, CUDA, or GTSAM.

- [ ] **Step 4: Verify and commit**

```bash
conda run -n vggt-dem pytest tests/unit/adapters -v
git add vggt_slam_pp/adapters tests/unit/adapters tests/fixtures
git commit -m "feat: adapt VGGT-SLAM submaps for replay"
```

## Task 7: Implement Canonical TUM Trajectories And ATE

**Files:**
- Create: `vggt_slam_pp/evaluation/tum.py`
- Create: `vggt_slam_pp/evaluation/ate.py`
- Create: `vggt_slam_pp/cli/evaluate_ate.py`
- Create: `tests/unit/evaluation/test_tum.py`
- Create: `tests/integration/test_evaluate_ate.py`
- Create: `docs/algorithms/ate.md`

- [ ] **Step 1: Write failing duplicate-timestamp tests**

Synthetic TUM rows include one duplicated transition timestamp. Tests assert:

- raw rows remain unchanged;
- canonical rows keep the first occurrence;
- duplicate translation and quaternion-angle disagreement is reported;
- non-finite, decreasing, malformed, or too-short trajectories are rejected.

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/unit/evaluation/test_tum.py -v
```

- [ ] **Step 3: Implement TUM parsing and canonicalization**

Expose:

```python
def read_tum_rows(path: Path) -> np.ndarray: ...
def canonicalize_tum(rows: np.ndarray) -> CanonicalTrajectory: ...
def write_tum_rows(path: Path, rows: np.ndarray) -> None: ...
```

Quaternion input order is `qx qy qz qw`. Canonicalization preserves the first
duplicate and computes boundary disagreement statistics.

- [ ] **Step 4: Write failing ATE integration test**

Generate two synthetic trajectories related by a known Sim(3). Invoke the CLI
and assert both `paper_compatible` and `canonical_unique` RMSE are below
`1e-8`, input hashes are present, and associated pose count is recorded.

- [ ] **Step 5: Implement ATE using evo Python APIs**

Read with `evo.tools.file_interface`, associate with `evo.core.sync`, align
with scale, and compute translation APE via `evo.core.metrics.APE`. Do not parse
human CLI text. Emit deterministic JSON.

- [ ] **Step 6: Verify and commit**

```bash
conda run -n vggt-dem pytest tests/unit/evaluation tests/integration/test_evaluate_ate.py -v
git add vggt_slam_pp/evaluation vggt_slam_pp/cli tests docs/algorithms/ate.md
git commit -m "feat: add auditable TUM ATE evaluation"
```

## Task 8: Add Baseline Dependency Injection And Preflight

**Files:**
- Modify: `VGGT-SLAM-version1.0/main.py`
- Modify: `VGGT-SLAM-version1.0/vggt_slam/solver.py`
- Modify: `VGGT-SLAM-version1.0/vggt_slam/loop_closure.py`
- Test: `tests/static/test_baseline_cli.py`

- [ ] **Step 1: Write failing AST tests**

Tests assert that the baseline exposes:

```text
--vggt_weight
--salad_checkpoint
--device
--export_submaps_dir
--run_id
--run_purpose
```

They also assert no active call remains to `load_state_dict_from_url()` or
`torch.hub.load()` and that export defaults to `None`.

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/static/test_baseline_cli.py -v
```

- [ ] **Step 3: Add explicit dependency parameters**

`main.py` resolves device once, loads VGGT from `--vggt_weight`, validates
run identity, and passes device/checkpoint to `Solver`.

`loop_closure.py` receives a model or a reviewed local SALAD loader; it cannot
download at runtime. With `max_loops=0`, retrieval initialization is skipped
entirely so `pp_frontend_bridge` does not require SALAD.

`solver.py` imports `Dict` and `List`, and chooses autocast dtype without
calling CUDA APIs on CPU.

- [ ] **Step 4: Verify syntax, tests, and commit**

```bash
conda run -n vggt-dem python -m compileall -q VGGT-SLAM-version1.0
conda run -n vggt-dem pytest tests/static/test_baseline_cli.py -v
git diff -- VGGT-SLAM-version1.0
git add VGGT-SLAM-version1.0 tests/static/test_baseline_cli.py
git commit -m "fix: make baseline dependencies explicit and offline"
```

## Task 9: Add The Opt-In Submap And Graph-State Bridge

**Files:**
- Modify: `VGGT-SLAM-version1.0/main.py`
- Create: `vggt_slam_pp/adapters/export_session.py`
- Create: `tests/integration/test_baseline_export_hook.py`

- [ ] **Step 1: Write failing static and fake-session tests**

Tests assert:

- no exporter is constructed when `--export_submaps_dir` is absent;
- exporter runs only after `add_points`, `graph.optimize`, and
  `update_submap_homographies`;
- latest submap geometry is written once;
- every graph update writes a state snapshot for all submaps;
- finalization writes `final_state.json`;
- source submaps are unchanged.

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/integration/test_baseline_export_hook.py -v
```

- [ ] **Step 3: Implement `ExportSession`**

```python
class ExportSession:
    def export_latest_submap(self, submap: object, loop_sources: tuple[int, ...]) -> Path:
        ...

    def export_graph_state(self, graph_map: object, graph: object) -> Path:
        ...

    def finalize(self, graph_map: object, graph: object) -> Path:
        ...
```

The baseline calls these methods behind a single `if export_session is not None`
guard. No existing solver call is reordered.

- [ ] **Step 4: Verify and commit**

```bash
conda run -n vggt-dem pytest tests/integration/test_baseline_export_hook.py -v
git diff -- VGGT-SLAM-version1.0/main.py
git add VGGT-SLAM-version1.0/main.py vggt_slam_pp/adapters/export_session.py tests/integration
git commit -m "feat: add opt-in M0 baseline export bridge"
```

## Task 10: Add Cache Inspection And Run Comparison Tools

**Files:**
- Create: `vggt_slam_pp/cli/inspect_submap_cache.py`
- Create: `vggt_slam_pp/evaluation/compare_runs.py`
- Create: `vggt_slam_pp/cli/compare_baseline_runs.py`
- Create: `tests/integration/test_inspect_submap_cache.py`
- Create: `tests/unit/evaluation/test_compare_runs.py`

- [ ] **Step 1: Write failing CLI tests**

Tests create synthetic caches and run summaries. Assert:

- inspector verifies checksums and prints shapes, dtypes, frames and transforms;
- comparison accepts numerically identical pose logs;
- comparison rejects submap count, loop count, timestamp, or pose differences;
- JSON output is deterministic.

- [ ] **Step 2: Verify RED**

```bash
conda run -n vggt-dem pytest tests/integration/test_inspect_submap_cache.py tests/unit/evaluation/test_compare_runs.py -v
```

- [ ] **Step 3: Implement tools, verify, and commit**

```bash
conda run -n vggt-dem pytest tests/integration/test_inspect_submap_cache.py tests/unit/evaluation/test_compare_runs.py -v
git add vggt_slam_pp/cli vggt_slam_pp/evaluation tests
git commit -m "feat: add M0 cache and baseline comparison tools"
```

## Task 11: Write Local And AutoDL Runbooks

**Files:**
- Create: `environment/local-macos.yml`
- Create: `environment/autodl-cuda.yml`
- Create: `external_sources/README.md`
- Create: `external_sources/manifest.yaml`
- Create: `docs/runbooks/m0-local-validation.md`
- Create: `docs/runbooks/m0-autodl-baseline.md`

- [ ] **Step 1: Document local validation**

Use:

```bash
conda run -n vggt-dem python -m pip install -e ".[dev]"
conda run -n vggt-dem python scripts/verify_assets.py --config configs/runtime/local_macos.yaml
conda run -n vggt-dem pytest -v
```

State explicitly that local full VGGT, SALAD, DINOv2 and SL(4) inference are
not M0 acceptance requirements.

- [ ] **Step 2: Document AutoDL uploads**

Upload into the same relative paths:

```text
weights/model.pt
weights/dino_salad.ckpt
data/rgbd_dataset_freiburg1_desk/
VGGT-SLAM-version1.0/office_loop/
```

- [ ] **Step 3: Document the run matrix**

Commands must cover:

1. Office-loop SL(4), export disabled.
2. Office-loop SL(4), export enabled.
3. TUM desk SL(4) reference.
4. TUM desk Sim3-compatible reference.
5. TUM desk Sim3-compatible bridge with `max_loops=0`.
6. ATE for all TUM runs.
7. Export-enabled/disabled office comparison.

All commands run from the repository root and use only relative paths.

- [ ] **Step 4: Verify docs and commit**

```bash
rg -n '/home/|/Users/' docs/runbooks environment configs external_sources
git diff --check
git add environment external_sources docs/runbooks configs
git commit -m "docs: add M0 local and AutoDL runbooks"
```

Expected: no machine-specific absolute runtime path.

## Task 12: M0 Local Verification And Cloud Review Bundle

**Files:**
- Create locally: `docs/reviews/m0/local/environment.txt`
- Create locally: `docs/reviews/m0/local/test-summary.md`
- Create after AutoDL: `docs/reviews/m0/<run-id>/environment.txt`
- Create after AutoDL: `docs/reviews/m0/<run-id>/commands.md`
- Create after AutoDL: `docs/reviews/m0/<run-id>/ate-summary.json`
- Create after AutoDL: `docs/reviews/m0/<run-id>/known-deviations.md`

- [ ] **Step 1: Run local verification**

```bash
conda run -n vggt-dem pytest -v
conda run -n vggt-dem python -m compileall -q vggt_slam_pp VGGT-SLAM-version1.0
git diff --check
git status --short
git count-objects -vH
du -sh .git
```

- [ ] **Step 2: Record local evidence**

Record Python, NumPy, Pydantic, evo, torch, GTSAM versions; CUDA/MPS status;
SL4 symbol status; exact test counts; Git commit and `.git` size.

- [ ] **Step 3: Execute AutoDL matrix**

Follow `docs/runbooks/m0-autodl-baseline.md`. Inspect first, middle and final
submaps. A command returning zero is insufficient without valid checksums,
finite trajectory values and finite ATE.

- [ ] **Step 4: Compare export paths**

Acceptance:

```text
office submap counts equal
office loop counts equal
office pose logs numerically identical
all TUM ATE values finite
canonical associated poses >= 10
pp_frontend_bridge loop count == 0
all cache checksums valid
```

- [ ] **Step 5: Commit small review evidence**

```bash
git add docs/reviews/m0
git commit -m "docs: record M0 verification evidence"
```

Do not commit NPZ caches, weight files, datasets, full logs or generated plots.

## M0 Review Gate

Do not begin M1 until the user reviews:

1. Verified baseline replacement and provenance.
2. The complete allowlisted baseline diff.
3. Runtime mode and purpose validation.
4. Submap arrays, coordinate frames and baked-scale semantics.
5. Graph-state snapshots and final state.
6. Raw and canonical ATE outputs.
7. Office export-disabled/export-enabled equality.
8. TUM desk three-run matrix.
9. AutoDL environment and known deviations.
10. Stable Git object-store size.
