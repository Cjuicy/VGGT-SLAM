#!/usr/bin/env python3
"""Verify local assets and runtime capabilities without downloading anything."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import yaml


class AssetVerificationError(RuntimeError):
    """Raised when an offline prerequisite does not match its declaration."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_gtsam_capabilities(solver_mode: str, gtsam_module: Any) -> None:
    """Check only the symbols used by the selected baseline solver path."""
    required_symbols = (
        ("SL4", "PriorFactorSL4", "BetweenFactorSL4")
        if solver_mode == "baseline_sl4"
        else ("Pose3",)
    )
    missing = [name for name in required_symbols if not hasattr(gtsam_module, name)]
    if missing:
        raise AssetVerificationError(
            f"GTSAM lacks required symbols for {solver_mode}: {', '.join(missing)}"
        )


def _load_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise AssetVerificationError(f"required Python module is missing: {name}") from exc


def verify_assets(
    config_path: Path,
    *,
    project_root: Path,
    torch_module: Any | None = None,
    gtsam_module: Any | None = None,
) -> dict[str, Any]:
    """Return a machine-readable report or raise on the first mismatch."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise AssetVerificationError(f"invalid YAML mapping: {config_path}")

    report: dict[str, Any] = {
        "ok": False,
        "runtime_id": config.get("runtime_id"),
        "config": str(config_path),
        "assets": {},
        "capabilities": {},
    }

    assets = config.get("assets", {})
    if not isinstance(assets, Mapping):
        raise AssetVerificationError("assets must be a mapping")

    for asset_name, declaration in assets.items():
        if not isinstance(declaration, Mapping):
            raise AssetVerificationError(f"invalid asset declaration: {asset_name}")
        relative_path = str(declaration["path"])
        expected_hash = str(declaration["sha256"]).lower()
        absolute_path = project_root / relative_path
        if not absolute_path.is_file():
            raise AssetVerificationError(f"missing asset: {relative_path}")
        actual_hash = _sha256(absolute_path)
        if actual_hash != expected_hash:
            raise AssetVerificationError(
                f"SHA-256 mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        report["assets"][asset_name] = {
            "path": relative_path,
            "sha256": actual_hash,
            "status": "ok",
        }

    capabilities = config.get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        raise AssetVerificationError("capabilities must be a mapping")

    torch_module = torch_module or _load_module("torch")
    cuda_available = bool(torch_module.cuda.is_available())
    require_cuda = bool(capabilities.get("require_cuda", False))
    if require_cuda and not cuda_available:
        raise AssetVerificationError("CUDA is required but torch.cuda.is_available() is false")

    solver_mode = str(capabilities["solver_mode"])
    gtsam_module = gtsam_module or _load_module("gtsam")
    verify_gtsam_capabilities(solver_mode, gtsam_module)
    report["capabilities"] = {
        "cuda_available": cuda_available,
        "require_cuda": require_cuda,
        "solver_mode": solver_mode,
        "gtsam": "ok",
    }
    report["ok"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    try:
        report = verify_assets(
            args.config.resolve(),
            project_root=args.project_root.resolve(),
        )
    except (AssetVerificationError, KeyError, OSError, yaml.YAMLError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
