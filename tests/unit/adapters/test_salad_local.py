from pathlib import Path
import pickle
import re
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from vggt_slam_pp.adapters.salad_local import LocalModelAssetError, load_local_salad


class StubDino:
    def load_state_dict(self, state: object, *, strict: bool) -> None:
        pass


def test_missing_dinov2_source_reports_absolute_hubconf_path(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "salad.ckpt"
    weight = tmp_path / "dinov2.pth"
    checkpoint.write_bytes(b"x")
    weight.write_bytes(b"x")
    missing_hubconf = (tmp_path / "missing-source" / "hubconf.py").resolve()

    with pytest.raises(LocalModelAssetError, match=re.escape(str(missing_hubconf))):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=tmp_path / "missing-source",
            dinov2_weight=weight,
            device="cpu",
        )


def test_missing_dinov2_weight_reports_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    checkpoint = tmp_path / "salad.ckpt"
    checkpoint.write_bytes(b"x")
    missing_weight = (tmp_path / "dinov2.pth").resolve()

    with pytest.raises(LocalModelAssetError, match=re.escape(str(missing_weight))):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=source,
            dinov2_weight=missing_weight,
            device="cpu",
        )


def test_missing_salad_checkpoint_reports_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    weight.write_bytes(b"x")
    missing_checkpoint = (tmp_path / "salad.ckpt").resolve()

    with pytest.raises(
        LocalModelAssetError,
        match=re.escape(str(missing_checkpoint)),
    ):
        load_local_salad(
            salad_checkpoint=missing_checkpoint,
            dinov2_source=source,
            dinov2_weight=weight,
            device="cpu",
        )


def test_loader_uses_local_hub_and_strict_salad_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")
    calls: dict[str, object] = {}

    class FakeDino:
        def load_state_dict(self, state: object, *, strict: bool) -> None:
            calls["dinov2_state"] = state
            calls["dinov2_strict"] = strict

    class FakeBackbone:
        model = None

    class FakeModel:
        def __init__(self, local_dino: object) -> None:
            self.backbone = FakeBackbone()
            self.backbone.model = local_dino

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            calls["state"] = state
            calls["strict"] = strict

        def eval(self) -> "FakeModel":
            calls["eval"] = True
            return self

        def to(self, device: object) -> "FakeModel":
            calls["device"] = str(device)
            return self

    def fake_hub_load(repo: str, model: str, **kwargs: object) -> FakeDino:
        calls["repo"] = repo
        calls["model"] = model
        calls["hub_kwargs"] = kwargs
        return FakeDino()

    def fake_torch_load(path: Path, **kwargs: object) -> dict[str, object]:
        calls.setdefault("load_calls", []).append((path, kwargs))
        return {"loaded": str(path)}

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=fake_hub_load),
        load=fake_torch_load,
        device=lambda value: value,
    )

    model = load_local_salad(
        salad_checkpoint=checkpoint,
        dinov2_source=source,
        dinov2_weight=weight,
        device="cpu",
        torch_module=fake_torch,
        model_factory=FakeModel,
    )

    assert isinstance(model.backbone.model, FakeDino)
    assert calls["repo"] == str(source.resolve())
    assert calls["model"] == "dinov2_vitb14"
    assert calls["hub_kwargs"] == {
        "source": "local",
        "pretrained": False,
    }
    assert calls["load_calls"] == [
        (
            weight.resolve(),
            {"map_location": "cpu", "weights_only": True},
        ),
        (
            checkpoint.resolve(),
            {"map_location": "cpu", "weights_only": True},
        ),
    ]
    assert calls["dinov2_state"] == {"loaded": str(weight.resolve())}
    assert calls["dinov2_strict"] is True
    assert calls["state"] == {"loaded": str(checkpoint.resolve())}
    assert calls["strict"] is True
    assert calls["eval"] is True
    assert calls["device"] == "cpu"


def test_default_factory_never_calls_salad_remote_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")
    calls: dict[str, object] = {"remote_hub_calls": 0}

    class FakeModule:
        def __init__(self) -> None:
            calls["module_initialized"] = True

    class FakeDINOv2(FakeModule):
        def __init__(self, **kwargs: object) -> None:
            calls["remote_hub_calls"] = int(calls["remote_hub_calls"]) + 1
            raise AssertionError(f"remote DINOv2 construction: {kwargs}")

    helper_module = ModuleType("salad.models_salad.helper")

    def remote_get_backbone(*args: object, **kwargs: object) -> object:
        return FakeDINOv2(args=args, kwargs=kwargs)

    helper_module.get_backbone = remote_get_backbone  # type: ignore[attr-defined]
    original_get_backbone = helper_module.get_backbone  # type: ignore[attr-defined]

    class FakeVPRModel:
        def __init__(self, **kwargs: object) -> None:
            calls["vpr_kwargs"] = kwargs
            self.backbone = helper_module.get_backbone(  # type: ignore[attr-defined]
                kwargs["backbone_arch"],
                kwargs["backbone_config"],
            )

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            calls["strict"] = strict

        def eval(self) -> "FakeVPRModel":
            return self

        def to(self, device: object) -> "FakeVPRModel":
            return self

    salad_module = ModuleType("salad")
    models_module = ModuleType("salad.models_salad")
    backbones_module = ModuleType("salad.models_salad.backbones")
    vpr_module = ModuleType("salad.vpr_model")
    backbones_module.DINOv2 = FakeDINOv2  # type: ignore[attr-defined]
    models_module.helper = helper_module  # type: ignore[attr-defined]
    models_module.backbones = backbones_module  # type: ignore[attr-defined]
    vpr_module.VPRModel = FakeVPRModel  # type: ignore[attr-defined]
    salad_module.models_salad = models_module  # type: ignore[attr-defined]
    salad_module.vpr_model = vpr_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "salad", salad_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad", models_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad.helper", helper_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad.backbones", backbones_module)
    monkeypatch.setitem(sys.modules, "salad.vpr_model", vpr_module)

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=lambda *args, **kwargs: StubDino()),
        load=lambda *args, **kwargs: {},
        device=lambda value: value,
        nn=SimpleNamespace(Module=FakeModule),
    )

    model = load_local_salad(
        salad_checkpoint=checkpoint,
        dinov2_source=source,
        dinov2_weight=weight,
        device="cpu",
        torch_module=fake_torch,
    )

    assert isinstance(model.backbone.model, StubDino)
    assert model.backbone.num_channels == 768
    assert model.backbone.num_trainable_blocks == 4
    assert model.backbone.norm_layer is True
    assert model.backbone.return_token is True
    assert calls["remote_hub_calls"] == 0
    assert helper_module.get_backbone is original_get_backbone  # type: ignore[attr-defined]


def test_default_factory_restores_helper_when_vpr_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")

    class FakeModule:
        def __init__(self) -> None:
            pass

    class FakeDINOv2(FakeModule):
        pass

    helper_module = ModuleType("salad.models_salad.helper")
    original_get_backbone = lambda *args, **kwargs: "remote-backbone"
    helper_module.get_backbone = original_get_backbone  # type: ignore[attr-defined]

    class FailingVPRModel:
        def __init__(self, **kwargs: object) -> None:
            raise ValueError("VPR construction failed")

    salad_module = ModuleType("salad")
    models_module = ModuleType("salad.models_salad")
    backbones_module = ModuleType("salad.models_salad.backbones")
    vpr_module = ModuleType("salad.vpr_model")
    backbones_module.DINOv2 = FakeDINOv2  # type: ignore[attr-defined]
    models_module.helper = helper_module  # type: ignore[attr-defined]
    models_module.backbones = backbones_module  # type: ignore[attr-defined]
    vpr_module.VPRModel = FailingVPRModel  # type: ignore[attr-defined]
    salad_module.models_salad = models_module  # type: ignore[attr-defined]
    salad_module.vpr_model = vpr_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "salad", salad_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad", models_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad.helper", helper_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad.backbones", backbones_module)
    monkeypatch.setitem(sys.modules, "salad.vpr_model", vpr_module)

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=lambda *args, **kwargs: StubDino()),
        load=lambda *args, **kwargs: {},
        nn=SimpleNamespace(Module=FakeModule),
    )

    with pytest.raises(ValueError, match="VPR construction failed"):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=source,
            dinov2_weight=weight,
            device="cpu",
            torch_module=fake_torch,
        )

    assert helper_module.get_backbone is original_get_backbone  # type: ignore[attr-defined]


def test_default_factory_serializes_concurrent_salad_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")
    first_entered = threading.Event()
    release_first = threading.Event()
    overlap_detected = threading.Event()
    active_constructors = 0
    active_guard = threading.Lock()

    class FakeModule:
        def __init__(self) -> None:
            pass

    class FakeDINOv2(FakeModule):
        pass

    helper_module = ModuleType("salad.models_salad.helper")
    original_get_backbone = lambda *args, **kwargs: "remote-backbone"
    helper_module.get_backbone = original_get_backbone  # type: ignore[attr-defined]

    class FakeVPRModel:
        def __init__(self, **kwargs: object) -> None:
            nonlocal active_constructors
            with active_guard:
                active_constructors += 1
                if active_constructors > 1:
                    overlap_detected.set()
            first_entered.set()
            release_first.wait(timeout=2)
            self.backbone = helper_module.get_backbone()  # type: ignore[attr-defined]
            with active_guard:
                active_constructors -= 1

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            pass

        def eval(self) -> "FakeVPRModel":
            return self

        def to(self, device: object) -> "FakeVPRModel":
            return self

    salad_module = ModuleType("salad")
    models_module = ModuleType("salad.models_salad")
    backbones_module = ModuleType("salad.models_salad.backbones")
    vpr_module = ModuleType("salad.vpr_model")
    backbones_module.DINOv2 = FakeDINOv2  # type: ignore[attr-defined]
    models_module.helper = helper_module  # type: ignore[attr-defined]
    models_module.backbones = backbones_module  # type: ignore[attr-defined]
    vpr_module.VPRModel = FakeVPRModel  # type: ignore[attr-defined]
    salad_module.models_salad = models_module  # type: ignore[attr-defined]
    salad_module.vpr_model = vpr_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "salad", salad_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad", models_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad.helper", helper_module)
    monkeypatch.setitem(sys.modules, "salad.models_salad.backbones", backbones_module)
    monkeypatch.setitem(sys.modules, "salad.vpr_model", vpr_module)

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=lambda *args, **kwargs: StubDino()),
        load=lambda *args, **kwargs: {},
        device=lambda value: value,
        nn=SimpleNamespace(Module=FakeModule),
    )
    errors: list[BaseException] = []

    def construct() -> None:
        try:
            load_local_salad(
                salad_checkpoint=checkpoint,
                dinov2_source=source,
                dinov2_weight=weight,
                device="cpu",
                torch_module=fake_torch,
            )
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=construct)
    second = threading.Thread(target=construct)
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()
    overlap_detected.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert not overlap_detected.is_set()
    assert helper_module.get_backbone is original_get_backbone  # type: ignore[attr-defined]


def test_local_hub_failure_reports_weight_and_disables_network_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")

    def fail_local_hub(*args: object, **kwargs: object) -> None:
        raise RuntimeError("invalid local DINOv2 weight")

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=fail_local_hub),
    )

    with pytest.raises(
        LocalModelAssetError,
        match=(
            rf"{re.escape(str(weight.resolve()))}.*"
            r"network fallback is disabled"
        ),
    ):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=source,
            dinov2_weight=weight,
            device="cpu",
            torch_module=fake_torch,
        )


def test_checkpoint_failure_reports_checkpoint_and_disables_network_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")

    class FakeModel:
        def load_state_dict(self, state: object, *, strict: bool) -> None:
            raise AssertionError("checkpoint load should fail first")

    def load_weights(path: Path, **kwargs: object) -> dict[str, object]:
        if Path(path) == weight.resolve():
            return {}
        raise RuntimeError("invalid SALAD checkpoint")

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=lambda *args, **kwargs: StubDino()),
        load=load_weights,
    )

    with pytest.raises(
        LocalModelAssetError,
        match=(
            rf"{re.escape(str(checkpoint.resolve()))}.*"
            r"network fallback is disabled"
        ),
    ):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=source,
            dinov2_weight=weight,
            device="cpu",
            torch_module=fake_torch,
            model_factory=lambda local_dino: FakeModel(),
        )


def test_unpickling_failure_reports_checkpoint_and_disables_network_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")

    class FakeModel:
        pass

    def load_weights(path: Path, **kwargs: object) -> dict[str, object]:
        if Path(path) == weight.resolve():
            return {}
        raise pickle.UnpicklingError("invalid pickle stream")

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=lambda *args, **kwargs: StubDino()),
        load=load_weights,
    )

    with pytest.raises(
        LocalModelAssetError,
        match=(
            rf"{re.escape(str(checkpoint.resolve()))}.*"
            r"network fallback is disabled"
        ),
    ):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=source,
            dinov2_weight=weight,
            device="cpu",
            torch_module=fake_torch,
            model_factory=lambda local_dino: FakeModel(),
        )


def test_local_hub_programming_error_is_not_wrapped(tmp_path: Path) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    weight = tmp_path / "dinov2.pth"
    checkpoint = tmp_path / "salad.ckpt"
    weight.write_bytes(b"weight")
    checkpoint.write_bytes(b"checkpoint")

    def fail_local_hub(*args: object, **kwargs: object) -> None:
        raise TypeError("bad adapter call")

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(load=fail_local_hub),
    )

    with pytest.raises(TypeError, match="bad adapter call"):
        load_local_salad(
            salad_checkpoint=checkpoint,
            dinov2_source=source,
            dinov2_weight=weight,
            device="cpu",
            torch_module=fake_torch,
        )
