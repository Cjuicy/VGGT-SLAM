"""Assemble SALAD from reviewed local DINOv2 and checkpoint assets."""

from __future__ import annotations

from pathlib import Path
import pickle
import threading
from typing import Any

_SALAD_CONSTRUCTION_LOCK = threading.Lock()


class LocalModelAssetError(RuntimeError):
    """Raised when a reviewed local model dependency is unavailable."""


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LocalModelAssetError(f"{label} is missing: {resolved}")
    return resolved


def _default_model_factory(local_dino: Any, torch_module: Any) -> Any:
    from salad.models_salad import backbones, helper
    from salad.vpr_model import VPRModel

    # The upstream initializer performs a remote DINOv2 Hub load.
    backbone = backbones.DINOv2.__new__(backbones.DINOv2)
    torch_module.nn.Module.__init__(backbone)
    backbone.model = local_dino
    backbone.num_channels = 768
    backbone.num_trainable_blocks = 4
    backbone.norm_layer = True
    backbone.return_token = True

    # Model initialization happens only at startup. Serialize this adapter's
    # SALAD construction so temporary helper replacement cannot interleave.
    with _SALAD_CONSTRUCTION_LOCK:
        original_get_backbone = helper.get_backbone
        helper.get_backbone = lambda *_args, **_kwargs: backbone
        try:
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
        finally:
            helper.get_backbone = original_get_backbone


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

    try:
        local_dino = torch_module.hub.load(
            str(source),
            "dinov2_vitb14",
            source="local",
            pretrained=True,
            weights=str(weight),
        )
    except (OSError, RuntimeError) as error:
        raise LocalModelAssetError(
            "DINOv2 local load failed for "
            f"{weight} from {source}; network fallback is disabled"
        ) from error

    if model_factory is None:
        model = _default_model_factory(local_dino, torch_module)
    else:
        model = model_factory(local_dino)

    try:
        state = torch_module.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise LocalModelAssetError(
            "SALAD checkpoint load failed for "
            f"{checkpoint}; network fallback is disabled"
        ) from error

    return model.eval().to(torch_module.device(device))
