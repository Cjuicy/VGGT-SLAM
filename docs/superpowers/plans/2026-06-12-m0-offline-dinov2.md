# M0 Offline DINOv2 Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VGGT-SLAM v1 SALAD loop-closure path load DINOv2 architecture and weights only from reviewed local assets.

**Architecture:** Add one project-owned adapter that assembles the upstream SALAD model with an official DINOv2 model created through `torch.hub.load(..., source="local")`. Pass source and weight paths explicitly through `main.py`, `Solver`, and `ImageRetrieval`; keep `max_loops=0` lazy and independent of SALAD. Extend the existing offline preflight and runbook without modifying ignored third-party source trees.

**Tech Stack:** Python 3.11, PyTorch, official DINOv2 Hub entry point, SALAD, pytest, AST static tests, YAML.

---

## File Structure

```text
vggt_slam_pp/adapters/salad_local.py
    Validate local paths and assemble the reviewed SALAD model.

VGGT-SLAM-version1.0/main.py
    Expose local DINOv2 CLI paths and pass them to Solver.

VGGT-SLAM-version1.0/vggt_slam/solver.py
    Pass local model dependencies only when loop closure is enabled.

VGGT-SLAM-version1.0/vggt_slam/loop_closure.py
    Replace the upstream online loader with the project adapter.

vggt_slam_pp/cli/verify_assets.py
    Verify the local DINOv2 Hub entry without constructing the model.

external_sources/README.md
external_sources/manifest.yaml
docs/runbooks/m0-autodl-baseline.md
    Document provenance, cloud placement, installation and commands.
```

### Task 1: Add The Local DINOv2 And SALAD Adapter

**Files:**
- Create: `vggt_slam_pp/adapters/salad_local.py`
- Create: `tests/unit/adapters/test_salad_local.py`

- [ ] **Step 1: Write failing path-validation tests**

```python
from pathlib import Path

import pytest

from vggt_slam_pp.adapters.salad_local import LocalModelAssetError, load_local_salad


def test_missing_dinov2_source_fails_before_importing_salad(tmp_path: Path) -> None:
    checkpoint = tmp_path / "salad.ckpt"
    weight = tmp_path / "dinov2.pth"
    checkpoint.write_bytes(b"x")
    weight.write_bytes(b"x")

    with pytest.raises(LocalModelAssetError, match="hubconf.py"):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=tmp_path / "missing-source",
            dinov2_weight=weight,
            device="cpu",
        )


def test_missing_dinov2_weight_reports_exact_path(tmp_path: Path) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    checkpoint = tmp_path / "salad.ckpt"
    checkpoint.write_bytes(b"x")

    with pytest.raises(LocalModelAssetError, match="dinov2.pth"):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=source,
            dinov2_weight=tmp_path / "dinov2.pth",
            device="cpu",
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n vggt-dem pytest tests/unit/adapters/test_salad_local.py -v
```

Expected: collection fails because `vggt_slam_pp.adapters.salad_local` does not exist.

- [ ] **Step 3: Implement strict asset validation**

Create `vggt_slam_pp/adapters/salad_local.py` with:

```python
from pathlib import Path
from typing import Any


class LocalModelAssetError(RuntimeError):
    """Raised when a reviewed local model dependency is unavailable."""


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LocalModelAssetError(f"{label} is missing: {resolved}")
    return resolved


def load_local_salad(
    *,
    salad_checkpoint: Path,
    dinov2_source: Path,
    dinov2_weight: Path,
    device: Any,
) -> Any:
    source = Path(dinov2_source).expanduser().resolve()
    _require_file(source / "hubconf.py", "DINOv2 source")
    weight = _require_file(Path(dinov2_weight), "DINOv2 weight")
    checkpoint = _require_file(Path(salad_checkpoint), "SALAD checkpoint")
    raise NotImplementedError((source, weight, checkpoint, device))
```

- [ ] **Step 4: Run the path tests and verify GREEN**

Run:

```bash
conda run -n vggt-dem pytest tests/unit/adapters/test_salad_local.py -v
```

Expected: both path-validation tests pass.

- [ ] **Step 5: Add a failing local-Hub assembly test**

Append a test using injected fake modules so no model or network is used:

```python
from types import SimpleNamespace


def test_loader_uses_local_hub_and_strict_salad_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")
    calls: dict[str, object] = {}

    class FakeBackbone:
        model = None

    class FakeModel:
        backbone = FakeBackbone()

        def load_state_dict(self, state: object, *, strict: bool):
            calls["state"] = state
            calls["strict"] = strict

        def eval(self):
            return self

        def to(self, device: object):
            calls["device"] = str(device)
            return self

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(
            load=lambda repo, model, **kwargs: calls.update(
                repo=repo, model=model, kwargs=kwargs
            )
            or "local-dino"
        ),
        load=lambda path, **kwargs: {"loaded": str(path), "kwargs": kwargs},
        device=lambda value: value,
    )

    model = load_local_salad(
        salad_checkpoint=checkpoint,
        dinov2_source=source,
        dinov2_weight=weight,
        device="cpu",
        torch_module=fake_torch,
        model_factory=lambda: FakeModel(),
    )

    assert model.backbone.model == "local-dino"
    assert calls["model"] == "dinov2_vitb14"
    assert calls["kwargs"] == {
        "source": "local",
        "pretrained": True,
        "weights": str(weight.resolve()),
    }
    assert calls["strict"] is True
    assert calls["device"] == "cpu"
```

- [ ] **Step 6: Run the assembly test and verify RED**

Run:

```bash
conda run -n vggt-dem pytest \
  tests/unit/adapters/test_salad_local.py::test_loader_uses_local_hub_and_strict_salad_checkpoint -v
```

Expected: FAIL because `load_local_salad` does not accept injected dependencies and raises `NotImplementedError`.

- [ ] **Step 7: Implement the reviewed model assembly**

Update the loader:

```python
def _default_model_factory() -> Any:
    from salad.vpr_model import VPRModel

    return VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch="SALAD",
        agg_config={
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256,
        },
    )


def load_local_salad(
    *,
    salad_checkpoint: Path,
    dinov2_source: Path,
    dinov2_weight: Path,
    device: Any,
    torch_module: Any | None = None,
    model_factory: Any | None = None,
) -> Any:
    source = Path(dinov2_source).expanduser().resolve()
    _require_file(source / "hubconf.py", "DINOv2 source")
    weight = _require_file(Path(dinov2_weight), "DINOv2 weight")
    checkpoint = _require_file(Path(salad_checkpoint), "SALAD checkpoint")

    if torch_module is None:
        import torch as torch_module
    factory = model_factory or _default_model_factory

    local_dino = torch_module.hub.load(
        str(source),
        "dinov2_vitb14",
        source="local",
        pretrained=True,
        weights=str(weight),
    )
    model = factory()
    model.backbone.model = local_dino
    state = torch_module.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(torch_module.device(device))
```

- [ ] **Step 8: Run adapter tests**

Run:

```bash
conda run -n vggt-dem pytest tests/unit/adapters/test_salad_local.py -v
```

Expected: all tests pass and no network access occurs.

- [ ] **Step 9: Commit**

```bash
git add vggt_slam_pp/adapters/salad_local.py tests/unit/adapters/test_salad_local.py
git commit -m "feat: add offline SALAD model loader"
```

### Task 2: Pass Local DINOv2 Assets Through The Baseline

**Files:**
- Modify: `tests/static/test_baseline_cli.py`
- Modify: `VGGT-SLAM-version1.0/main.py`
- Modify: `VGGT-SLAM-version1.0/vggt_slam/solver.py`
- Modify: `VGGT-SLAM-version1.0/vggt_slam/loop_closure.py`

- [ ] **Step 1: Extend the failing static CLI test**

Add `--dinov2_source` and `--dinov2_weight` to
`test_baseline_exposes_explicit_runtime_and_bridge_arguments`.

Add:

```python
def test_loop_closure_uses_reviewed_local_salad_loader() -> None:
    source = LOOP_CLOSURE.read_text(encoding="utf-8")
    assert "from vggt_slam_pp.adapters.salad_local import load_local_salad" in source
    assert "from salad.eval import load_model" not in source
    assert "dinov2_source" in source
    assert "dinov2_weight" in source


def test_project_adapter_allows_only_explicit_local_hub() -> None:
    adapter = (
        Path(__file__).resolve().parents[2]
        / "vggt_slam_pp"
        / "adapters"
        / "salad_local.py"
    ).read_text(encoding="utf-8")
    assert 'source="local"' in adapter
    assert "facebookresearch/dinov2" not in adapter
```

- [ ] **Step 2: Run the static tests and verify RED**

Run:

```bash
conda run -n vggt-dem pytest tests/static/test_baseline_cli.py -v
```

Expected: FAIL because the two CLI arguments and local loader call are absent.

- [ ] **Step 3: Add CLI arguments and pass resolved paths**

In `main.py`, add:

```python
parser.add_argument(
    "--dinov2_source",
    type=str,
    default="external_sources/dinov2",
    help="Local official DINOv2 source directory; runtime downloads are disabled",
)
parser.add_argument(
    "--dinov2_weight",
    type=str,
    default="weights/dinov2_vitb14_pretrain.pth",
    help="Local DINOv2 ViT-B/14 pretraining weight",
)
```

Pass to `Solver`:

```python
dinov2_source=Path(args.dinov2_source),
dinov2_weight=Path(args.dinov2_weight),
```

- [ ] **Step 4: Extend Solver only at the dependency boundary**

Add constructor parameters:

```python
dinov2_source: Path | None = None,
dinov2_weight: Path | None = None,
```

Inside the existing `if self.enable_loop_closure:` block, require all three paths
and pass them to `ImageRetrieval`. Do not inspect them when loop closure is disabled.

- [ ] **Step 5: Replace the upstream SALAD loader**

Change `ImageRetrieval.__init__` to:

```python
def __init__(
    self,
    checkpoint_path: Path,
    dinov2_source: Path,
    dinov2_weight: Path,
    device: torch.device,
    input_size=224,
):
    from vggt_slam_pp.adapters.salad_local import load_local_salad

    self.device = torch.device(device)
    self.model = load_local_salad(
        salad_checkpoint=checkpoint_path,
        dinov2_source=dinov2_source,
        dinov2_weight=dinov2_weight,
        device=self.device,
    )
    self.transform = input_transform((input_size, input_size))
```

- [ ] **Step 6: Run static and bridge tests**

Run:

```bash
conda run -n vggt-dem pytest \
  tests/static/test_baseline_cli.py \
  tests/integration/test_baseline_export_hook.py -v
```

Expected: all tests pass; `max_loops=0` remains independent of SALAD.

- [ ] **Step 7: Commit**

```bash
git add tests/static/test_baseline_cli.py \
  VGGT-SLAM-version1.0/main.py \
  VGGT-SLAM-version1.0/vggt_slam/solver.py \
  VGGT-SLAM-version1.0/vggt_slam/loop_closure.py
git commit -m "fix: route baseline through local DINOv2"
```

### Task 3: Extend Offline Asset Preflight

**Files:**
- Modify: `tests/unit/test_verify_assets.py`
- Modify: `vggt_slam_pp/cli/verify_assets.py`
- Modify: `configs/runtime/autodl_cuda.yaml`
- Modify: `configs/runtime/local_macos.yaml`

- [ ] **Step 1: Add a failing source-entry test**

Extend `_write_config` to accept:

```python
"sources": {
    "dinov2": {
        "path": "external_sources/dinov2",
        "entrypoint": "hubconf.py",
    }
}
```

Add:

```python
def test_missing_dinov2_hub_entry_reports_exact_path(tmp_path: Path) -> None:
    asset = tmp_path / "weights" / "model.pt"
    asset.parent.mkdir()
    asset.write_bytes(b"offline-weight")
    config = tmp_path / "runtime.yaml"
    _write_config(config, "weights/model.pt", hashlib.sha256(asset.read_bytes()).hexdigest())

    with pytest.raises(
        AssetVerificationError,
        match=r"external_sources/dinov2/hubconf\.py",
    ):
        verify_assets(
            config,
            project_root=tmp_path,
            torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            gtsam_module=SimpleNamespace(Pose3=object()),
        )
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n vggt-dem pytest \
  tests/unit/test_verify_assets.py::test_missing_dinov2_hub_entry_reports_exact_path -v
```

Expected: FAIL because `sources` is not checked.

- [ ] **Step 3: Implement source entry verification**

Before capability checks, validate:

```python
sources = config.get("sources", {})
if not isinstance(sources, Mapping):
    raise AssetVerificationError("sources must be a mapping")
for source_name, declaration in sources.items():
    source_path = str(declaration["path"])
    entrypoint = str(declaration["entrypoint"])
    relative_entry = Path(source_path) / entrypoint
    if not (project_root / relative_entry).is_file():
        raise AssetVerificationError(f"missing source entry: {relative_entry}")
    report.setdefault("sources", {})[source_name] = {
        "path": source_path,
        "entrypoint": entrypoint,
        "status": "ok",
    }
```

- [ ] **Step 4: Declare DINOv2 source in both runtime configs**

Add:

```yaml
sources:
  dinov2:
    path: external_sources/dinov2
    entrypoint: hubconf.py
```

- [ ] **Step 5: Run preflight unit tests**

Run:

```bash
conda run -n vggt-dem pytest tests/unit/test_verify_assets.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_verify_assets.py vggt_slam_pp/cli/verify_assets.py \
  configs/runtime/autodl_cuda.yaml configs/runtime/local_macos.yaml
git commit -m "feat: verify local DINOv2 source"
```

### Task 4: Record DINOv2 Provenance And Cloud Commands

**Files:**
- Modify: `external_sources/README.md`
- Modify: `external_sources/manifest.yaml`
- Modify: `docs/runbooks/m0-autodl-baseline.md`

- [ ] **Step 1: Add the official DINOv2 source declaration**

Add to `manifest.yaml`:

```yaml
  dinov2:
    path: external_sources/dinov2
    upstream: https://github.com/facebookresearch/dinov2.git
    browser_url: https://github.com/facebookresearch/dinov2
    ref: main
    commit: null
    license_reviewed: false
    status: user_source_required
```

Do not invent `verified_remote_head`; record the actual cloud commit after cloning.

- [ ] **Step 2: Add download and commit-recording commands**

Add to `external_sources/README.md`:

```bash
git clone --branch main https://github.com/facebookresearch/dinov2.git \
  external_sources/dinov2
git -C external_sources/dinov2 rev-parse HEAD
```

State explicitly that DINOv2 is not installed with pip and is loaded through its
local `hubconf.py`.

- [ ] **Step 3: Update the AutoDL runbook**

Before preflight, add:

```bash
test -f external_sources/dinov2/hubconf.py
rm -f ~/.cache/torch/hub/main.zip
export OMP_NUM_THREADS=4
```

Add both arguments to every run with `--max_loops 1`:

```bash
--dinov2_source external_sources/dinov2 \
--dinov2_weight weights/dinov2_vitb14_pretrain.pth \
```

Keep `max_loops=0` bridge commands explicit but allow defaults because they must not
load SALAD/DINOv2.

- [ ] **Step 4: Check documentation and secrets**

Run:

```bash
git diff --check
rg -n "github_pat_|torch\\.hub\\.load\\(['\"]facebookresearch" \
  docs external_sources VGGT-SLAM-version1.0 vggt_slam_pp
```

Expected: no credential and no remote DINOv2 Hub call.

- [ ] **Step 5: Commit**

```bash
git add external_sources/README.md external_sources/manifest.yaml \
  docs/runbooks/m0-autodl-baseline.md
git commit -m "docs: add offline DINOv2 cloud setup"
```

### Task 5: Verify Locally And Prepare The AutoDL Checkpoint

**Files:**
- Modify: `docs/reviews/m0/local/test-summary.md`

- [ ] **Step 1: Run focused tests**

```bash
conda run -n vggt-dem pytest \
  tests/unit/adapters/test_salad_local.py \
  tests/unit/test_verify_assets.py \
  tests/static/test_baseline_cli.py \
  tests/integration/test_baseline_export_hook.py -v
```

Expected: all pass.

- [ ] **Step 2: Run the complete M0 test suite**

```bash
conda run -n vggt-dem pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Compile the touched Python packages**

```bash
conda run -n vggt-dem python -m compileall -q \
  vggt_slam_pp VGGT-SLAM-version1.0
```

Expected: exit code 0.

- [ ] **Step 4: Record local evidence**

Append the exact commands and results to `docs/reviews/m0/local/test-summary.md`.
Record that real SALAD/DINOv2 loading was not executed on macOS and remains an AutoDL
checkpoint.

- [ ] **Step 5: Commit local verification**

```bash
git add docs/reviews/m0/local/test-summary.md
git commit -m "test: record offline DINOv2 verification"
```

- [ ] **Step 6: Push the implementation branch**

```bash
git push origin codex/m0-reproduction
```

- [ ] **Step 7: Clone DINOv2 and record the cloud commit**

On AutoDL:

```bash
cd ~/autodl-tmp/VGGT-SLAM
git pull
git clone --branch main https://github.com/facebookresearch/dinov2.git \
  external_sources/dinov2
git -C external_sources/dinov2 rev-parse HEAD
python -m pip install -e ".[dev]"
```

Expected: `external_sources/dinov2/hubconf.py` exists and the commit hash is printed.

- [ ] **Step 8: Run AutoDL preflight**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
python scripts/verify_assets.py --config configs/runtime/autodl_cuda.yaml
```

Expected: JSON contains `"ok": true`, DINOv2 source status `ok`, CUDA `true`, and
GTSAM `ok`.

- [ ] **Step 9: Run one real SALAD load before the full baseline**

```bash
python - <<'PY'
from pathlib import Path
from vggt_slam_pp.adapters.salad_local import load_local_salad

model = load_local_salad(
    salad_checkpoint=Path("weights/dino_salad.ckpt"),
    dinov2_source=Path("external_sources/dinov2"),
    dinov2_weight=Path("weights/dinov2_vitb14_pretrain.pth"),
    device="cuda",
)
print(type(model).__name__)
print("Offline SALAD load: OK")
PY
```

Expected: `Offline SALAD load: OK` and no `Downloading:` line.

- [ ] **Step 10: Run Office-loop without export**

Use the command in `docs/runbooks/m0-autodl-baseline.md` section 1.

Expected: initialization completes, image discovery starts, and no network download
is attempted. Stop after the run completes and review `submap_count`, `loop_count`,
and `artifacts/m0/office-sl4-no-export/poses.txt`.

- [ ] **Step 11: Record confirmed external commits**

Write the printed DINOv2, VGGT, SALAD and GTSAM commit hashes into
`external_sources/manifest.yaml`, change only sources actually run successfully to
the reviewed status selected by the project, and commit the cloud evidence on
`autodl/m0-validation`.
