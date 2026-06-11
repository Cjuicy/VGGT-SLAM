import ast
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[2] / "VGGT-SLAM-version1.0"
MAIN = BASELINE / "main.py"
SOLVER = BASELINE / "vggt_slam" / "solver.py"
LOOP_CLOSURE = BASELINE / "vggt_slam" / "loop_closure.py"


def _argument_defaults(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                defaults[name] = keyword.value.value
    return defaults


def test_baseline_exposes_explicit_runtime_and_bridge_arguments() -> None:
    defaults = _argument_defaults(MAIN)
    for argument in (
        "--vggt_weight",
        "--salad_checkpoint",
        "--device",
        "--export_submaps_dir",
        "--run_id",
        "--run_purpose",
    ):
        assert argument in defaults
    assert defaults["--export_submaps_dir"] is None


def test_baseline_has_no_active_network_model_loaders() -> None:
    forbidden_calls = {"load_state_dict_from_url"}
    for path in (MAIN, SOLVER, LOOP_CLOSURE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls
                if (
                    node.func.attr == "load"
                    and isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "torch"
                    and node.func.value.attr == "hub"
                ):
                    raise AssertionError(f"torch.hub.load remains active in {path}")


def test_loop_retrieval_is_optional_when_disabled() -> None:
    source = SOLVER.read_text(encoding="utf-8")
    assert "enable_loop_closure" in source
    assert "if self.enable_loop_closure" in source
