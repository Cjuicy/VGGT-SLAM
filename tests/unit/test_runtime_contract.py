import pytest
from pydantic import ValidationError

from vggt_slam_pp.contracts.runtime import RunIdentity


def test_bridge_requires_sim3_compat_and_no_baseline_loops() -> None:
    with pytest.raises(ValidationError):
        RunIdentity(
            run_id="bad-solver",
            solver_mode="baseline_sl4",
            run_purpose="pp_frontend_bridge",
            max_loops=0,
            submap_size=32,
            min_disparity=50.0,
        )

    with pytest.raises(ValidationError):
        RunIdentity(
            run_id="bad-loops",
            solver_mode="baseline_sim3_compat",
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


@pytest.mark.parametrize("field", ["run_id", "submap_size", "min_disparity"])
def test_identity_rejects_invalid_basic_fields(field: str) -> None:
    values = {
        "run_id": "valid",
        "solver_mode": "baseline_sim3_compat",
        "run_purpose": "baseline_reference",
        "max_loops": 0,
        "submap_size": 32,
        "min_disparity": 50.0,
    }
    values[field] = {"run_id": "", "submap_size": 0, "min_disparity": 0.0}[field]

    with pytest.raises(ValidationError):
        RunIdentity(**values)
