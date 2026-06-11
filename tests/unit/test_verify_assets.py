import hashlib
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vggt_slam_pp.cli.verify_assets import (
    AssetVerificationError,
    verify_assets,
    verify_gtsam_capabilities,
)


def _write_config(path: Path, asset_path: str, sha256: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "runtime_id": "test",
                "assets": {
                    "test_weight": {
                        "path": asset_path,
                        "sha256": sha256,
                    }
                },
                "capabilities": {
                    "require_cuda": False,
                    "solver_mode": "baseline_sim3_compat",
                },
            }
        ),
        encoding="utf-8",
    )


def test_matching_asset_and_hash_pass_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "weights" / "model.pt"
    asset.parent.mkdir()
    asset.write_bytes(b"offline-weight")
    config = tmp_path / "runtime.yaml"
    _write_config(config, "weights/model.pt", hashlib.sha256(asset.read_bytes()).hexdigest())

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("asset preflight must not access the network")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    report = verify_assets(
        config,
        project_root=tmp_path,
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        gtsam_module=SimpleNamespace(Pose3=object()),
    )

    assert report["ok"] is True
    assert report["assets"]["test_weight"]["status"] == "ok"


def test_missing_asset_reports_exact_relative_path(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    _write_config(config, "weights/missing.pt", "0" * 64)

    with pytest.raises(AssetVerificationError, match=r"weights/missing\.pt"):
        verify_assets(
            config,
            project_root=tmp_path,
            torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            gtsam_module=SimpleNamespace(Pose3=object()),
        )


def test_sl4_mode_requires_all_sl4_symbols() -> None:
    incomplete_gtsam = SimpleNamespace(Pose3=object(), SL4=object())

    with pytest.raises(AssetVerificationError, match="PriorFactorSL4"):
        verify_gtsam_capabilities("baseline_sl4", incomplete_gtsam)


def test_sim3_compat_requires_pose3_but_not_sl4() -> None:
    verify_gtsam_capabilities(
        "baseline_sim3_compat",
        SimpleNamespace(Pose3=object()),
    )

    with pytest.raises(AssetVerificationError, match="Pose3"):
        verify_gtsam_capabilities("baseline_sim3_compat", SimpleNamespace())
